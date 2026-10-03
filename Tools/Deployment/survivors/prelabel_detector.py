"""Survivors 自動下書き用の torchvision 物体検出器を学習・保存・推論する。

確認済みラベル（画面の一部範囲だけのラベルも可）から Faster R-CNN MobileNetV3 を
学習し、未確認フレームの下書き矩形を出す。設定は annotation_prelabel_v2.yaml に置き、
学習 CLI と下書き CLI の両方が同じ load_config を通して検証する。
"""

from __future__ import annotations

import math
import os
import pickle
import random
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
import yaml
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from survivors.annotation_labels import (
    REGION_LABEL,
    WORLD_CLASSES,
    LabelBox,
    clip_box,
    iter_frame_files,
    read_label_file,
    validate_label,
)


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "annotation_prelabel_v2.yaml"
DEFAULT_WEIGHTS_NAME = "prelabel_detector.pt"
_WEIGHTS_FORMAT = "annotation_prelabel_detector.v1"
_IMAGE_WIDTH = 1920
_IMAGE_HEIGHT = 1080
_DETECTOR_KEYS = {
    "labels", "label_aliases", "score_threshold", "input_scale", "crop_size", "min_box_size",
    "iterations", "batch_size", "learning_rate", "momentum", "weight_decay", "seed",
}
_LOG_EVERY = 100
# 1画像あたりの検出数上限。torchvision 既定の 100 では敵が多い画面で下書きが打ち切られる。
# RPN の推論時提案数 1000 と同じ値。これより大きくしても検出数は増えない。
_MAX_DETECTIONS = 1000


@dataclass(frozen=True)
class DetectorSettings:
    """検出器の対象ラベルと学習・推論パラメータ。

    labels の並びがそのまま検出器のクラス番号（1 始まり、0 は背景）になる。
    label_aliases は確認済みラベルを学習前に別名へまとめる対応表。
    """

    labels: tuple[str, ...]
    label_aliases: dict[str, str]
    score_threshold: float
    input_scale: float
    crop_size: int
    min_box_size: float
    iterations: int
    batch_size: int
    learning_rate: float
    momentum: float
    weight_decay: float
    seed: int


@dataclass(frozen=True)
class PrelabelConfig:
    """annotation_prelabel.v2 設定ファイル全体。

    毎フレームに置く固定矩形と、検出器の設定を持つ。
    """

    fixed_boxes: tuple[LabelBox, ...]
    detector: DetectorSettings


@dataclass(frozen=True)
class TrainingFrame:
    """学習に使う確認済みフレーム1枚分の情報。

    boxes は別名置換済みの検出対象矩形、regions は labeled_region の矩形。
    regions が空なら画像全体がラベル済みという意味になる。
    """

    png_path: Path
    boxes: tuple[LabelBox, ...]
    regions: tuple[LabelBox, ...]


def _is_number(value: object) -> bool:
    """bool を除く有限の int/float かを返す。

    YAML の true や .nan を数値として受け入れないために使う。
    """
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def _is_positive_int(value: object) -> bool:
    """bool を除く正の int かを返す。

    crop_size や iterations のような個数・画素数の検証に使う。
    """
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


def _load_fixed_boxes(path: Path, items: object) -> tuple[LabelBox, ...]:
    """fixed_boxes セクションを検証して LabelBox の並びへ変換する。

    各要素は label と bbox だけを持ち、bbox は 1920x1080 画面の内側に収まる必要がある。
    labeled_region は物体ではないので固定矩形に使えない。
    """
    if not isinstance(items, list):
        raise ValueError(f"{path}: fixed_boxes must be a list")
    boxes = []
    for index, item in enumerate(items):
        if not isinstance(item, dict) or set(item) != {"label", "bbox"}:
            raise ValueError(f"{path}: fixed_boxes[{index}] must contain only label and bbox")
        label, bbox = item["label"], item["bbox"]
        if not isinstance(label, str):
            raise ValueError(f"{path}: fixed_boxes[{index}].label must be a string")
        if label == REGION_LABEL:
            raise ValueError(f"{path}: fixed_boxes[{index}] cannot use {REGION_LABEL}")
        try:
            validate_label(label)
        except ValueError as exc:
            raise ValueError(f"{path}: fixed_boxes[{index}]: {exc}") from exc
        if not isinstance(bbox, list) or len(bbox) != 4 or not all(_is_number(value) for value in bbox):
            raise ValueError(f"{path}: fixed_boxes[{index}].bbox must contain four finite numbers")
        left, top, right, bottom = (float(value) for value in bbox)
        if left < 0 or top < 0 or right <= left or bottom <= top or right > _IMAGE_WIDTH or bottom > _IMAGE_HEIGHT:
            raise ValueError(f"{path}: fixed_boxes[{index}].bbox is outside 1920x1080 image bounds")
        boxes.append(LabelBox(label, left, top, right, bottom))
    return tuple(boxes)


def _check_detector_label(path: Path, where: str, label: object) -> str:
    """下書き検出器で扱えるラベル（WORLD_CLASSES）かを検証する。

    画面内の物体と武器エフェクト（class map v2 で WORLD_CLASSES に入った）は下書きできるが、
    UI クラスや labeled_region、未知の名前は設定エラーにする。
    """
    if not isinstance(label, str):
        raise ValueError(f"{path}: {where} must be a string")
    if label == REGION_LABEL:
        raise ValueError(f"{path}: {where} cannot use {REGION_LABEL}")
    try:
        validate_label(label)
    except ValueError as exc:
        raise ValueError(f"{path}: {where}: {exc}") from exc
    if label not in WORLD_CLASSES:
        raise ValueError(f"{path}: {where} must be a world class: {label!r}")
    return label


def _load_detector_settings(path: Path, section: object) -> DetectorSettings:
    """detector セクションのキー・型・範囲を検証して DetectorSettings にする。

    キーの過不足、bool や文字列の数値、非有限値、範囲外の値を全て ValueError で止める。
    """
    if not isinstance(section, dict) or set(section) != _DETECTOR_KEYS:
        raise ValueError(f"{path}: detector has missing or unknown keys")

    raw_labels = section["labels"]
    if not isinstance(raw_labels, list) or not raw_labels:
        raise ValueError(f"{path}: detector.labels must be a non-empty list")
    labels = tuple(_check_detector_label(path, "detector.labels", label) for label in raw_labels)
    if len(set(labels)) != len(labels):
        raise ValueError(f"{path}: detector.labels must not contain duplicates")

    raw_aliases = section["label_aliases"]
    if not isinstance(raw_aliases, dict):
        raise ValueError(f"{path}: detector.label_aliases must be a mapping")
    aliases = {}
    for key, value in raw_aliases.items():
        key = _check_detector_label(path, "detector.label_aliases key", key)
        value = _check_detector_label(path, f"detector.label_aliases[{key}]", value)
        if key in labels:
            raise ValueError(f"{path}: detector.label_aliases key {key!r} is already a detector label")
        if value not in labels:
            raise ValueError(f"{path}: detector.label_aliases[{key}] target {value!r} is not in detector.labels")
        aliases[key] = value

    numbers = {}
    for key in ("score_threshold", "input_scale", "min_box_size", "learning_rate", "momentum", "weight_decay"):
        if not _is_number(section[key]):
            raise ValueError(f"{path}: detector.{key} must be a finite number")
        numbers[key] = float(section[key])
    if not 0 < numbers["score_threshold"] <= 1:
        raise ValueError(f"{path}: detector.score_threshold must be in (0, 1]")
    if numbers["input_scale"] <= 0:
        raise ValueError(f"{path}: detector.input_scale must be positive")
    if numbers["min_box_size"] < 0:
        raise ValueError(f"{path}: detector.min_box_size must be non-negative")
    if numbers["learning_rate"] <= 0:
        raise ValueError(f"{path}: detector.learning_rate must be positive")
    if not 0 <= numbers["momentum"] < 1:
        raise ValueError(f"{path}: detector.momentum must be in [0, 1)")
    if numbers["weight_decay"] < 0:
        raise ValueError(f"{path}: detector.weight_decay must be non-negative")
    for key in ("crop_size", "iterations", "batch_size"):
        if not _is_positive_int(section[key]):
            raise ValueError(f"{path}: detector.{key} must be a positive integer")
    seed = section["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"{path}: detector.seed must be an integer")

    return DetectorSettings(
        labels=labels,
        label_aliases=aliases,
        crop_size=section["crop_size"],
        iterations=section["iterations"],
        batch_size=section["batch_size"],
        seed=seed,
        **numbers,
    )


def load_config(path: Path | str) -> PrelabelConfig:
    """annotation_prelabel.v2 設定を読み込み、全項目を検証する。

    学習 CLI と下書き CLI はこの関数だけで設定を読むので、検証ルールは1か所にまとまる。
    不正な設定はパス付き ValueError にして、処理を始める前に止める。
    """
    path = Path(path)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"{path}: cannot read config: {exc}") from exc
    if not isinstance(data, dict) or set(data) != {"schema_version", "fixed_boxes", "detector"}:
        raise ValueError(f"{path}: expected schema_version, fixed_boxes, and detector")
    if data["schema_version"] != "annotation_prelabel.v2":
        raise ValueError(f"{path}: unsupported schema_version: {data['schema_version']!r}")
    return PrelabelConfig(
        fixed_boxes=_load_fixed_boxes(path, data["fixed_boxes"]),
        detector=_load_detector_settings(path, data["detector"]),
    )


def collect_training_frames(work_root: Path | str, settings: DetectorSettings) -> list[TrainingFrame]:
    """work-root 内の確認済みフレームを学習用の一覧にする。

    セッション名順・フレーム番号順に走査し、checked: true の JSON だけを使う。
    別名を置換したうえで検出対象ラベルの矩形だけを残し、labeled_region は範囲として分ける。
    対象矩形が無いフレームも背景の手本として残す。PNG が無い確認済み JSON はエラー。
    """
    frames = []
    for session_dir in sorted(path for path in Path(work_root).iterdir() if path.is_dir()):
        for _, png_path, json_path in iter_frame_files(session_dir):
            if json_path is None:
                continue
            boxes, checked = read_label_file(json_path)
            if not checked:
                continue
            if png_path is None:
                raise ValueError(f"{json_path}: checked label has no matching PNG")
            targets = []
            for box in boxes:
                box = replace(box, label=settings.label_aliases.get(box.label, box.label))
                if box.label in settings.labels:
                    targets.append(box)
            regions = tuple(box for box in boxes if box.label == REGION_LABEL)
            frames.append(TrainingFrame(png_path, tuple(targets), regions))
    return frames


def sample_crop(
    image: np.ndarray,
    frame: TrainingFrame,
    *,
    crop_size: int,
    min_box_size: float,
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """ラベル済み範囲の内側から crop_size 四方の学習用画像と矩形を切り出す。

    範囲（無ければ画像全体）の中で左上位置を乱数で選び、範囲外の画素は一切使わない。
    範囲が crop_size より狭い辺は範囲の左上に揃えて、足りない分を黒（0）で埋める。
    矩形は切り出し範囲へクリップし、幅か高さが min_box_size 未満になったものは捨てる。
    半分の確率で左右反転し、矩形の x 座標も合わせて反転する。乱数は rng だけを使う。
    戻り値は (BGR uint8 画像, (N, 4) float32 の xyxy, N 個のラベル名)。
    """
    height, width = image.shape[:2]
    region = rng.choice(frame.regions) if frame.regions else LabelBox(REGION_LABEL, 0, 0, width, height)
    clipped = clip_box(region, image_width=width, image_height=height)
    # 画素単位で範囲の内側だけを使うよう、端は内側へ丸める。
    if clipped is not None:
        left, top = math.ceil(clipped.left), math.ceil(clipped.top)
        right, bottom = math.floor(clipped.right), math.floor(clipped.bottom)
    if clipped is None or right <= left or bottom <= top:
        raise ValueError(f"{frame.png_path}: {REGION_LABEL} has no pixels inside the image")

    crop_width, crop_height = min(crop_size, right - left), min(crop_size, bottom - top)
    x0 = rng.randint(left, right - crop_size) if crop_width == crop_size else left
    y0 = rng.randint(top, bottom - crop_size) if crop_height == crop_size else top
    crop = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
    crop[:crop_height, :crop_width] = image[y0:y0 + crop_height, x0:x0 + crop_width, :3]

    boxes = np.array([[box.left, box.top, box.right, box.bottom] for box in frame.boxes], np.float32).reshape(-1, 4)
    names = [box.label for box in frame.boxes]
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(x0, x0 + crop_width) - x0
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(y0, y0 + crop_height) - y0
    keep = ((boxes[:, 2] - boxes[:, 0]) >= min_box_size) & ((boxes[:, 3] - boxes[:, 1]) >= min_box_size)
    # 面積0の矩形は min_box_size=0 でも学習に渡せないので必ず捨てる。
    keep &= (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes = boxes[keep]
    names = [name for name, kept in zip(names, keep) if kept]

    if rng.random() < 0.5:
        crop = crop[:, ::-1].copy()
        boxes[:, [0, 2]] = crop_size - boxes[:, [2, 0]]
    return crop, boxes, names


def build_model(num_labels: int, settings: DetectorSettings, *, pretrained: bool) -> torch.nn.Module:
    """Faster R-CNN MobileNetV3 を作り、分類 head を num_labels + 背景に差し替える。

    pretrained=True なら COCO 事前学習重みを使う（初回はダウンロードが走る）。
    False なら重みを一切ダウンロードせずに構築する（推論時の重み読み込みやテスト用）。
    入力は学習用の切り出し画像を input_scale 倍に拡大して扱う。
    """
    size = settings.crop_size * settings.input_scale
    if pretrained:
        model = fasterrcnn_mobilenet_v3_large_fpn(
            weights="DEFAULT", min_size=size, max_size=size, box_detections_per_img=_MAX_DETECTIONS
        )
    else:
        model = fasterrcnn_mobilenet_v3_large_fpn(
            weights=None, weights_backbone=None, min_size=size, max_size=size, box_detections_per_img=_MAX_DETECTIONS
        )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_labels + 1)
    return model


def default_device() -> str:
    """学習・推論に使う torch デバイス名を返す（GPU があれば cuda、無ければ cpu）。

    学習 CLI と下書き CLI が同じ決め方をするよう、ここ1か所で判定する。
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def _to_tensor(image_bgr: np.ndarray, device: str) -> torch.Tensor:
    """BGR uint8 画像を検出器入力の RGB float テンソル（0〜1）へ変換する。

    torchvision の検出器は CHW 形式の RGB を受け取るので並びを入れ替える。
    """
    rgb = cv2.cvtColor(np.ascontiguousarray(image_bgr), cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).to(device)


def train_detector(
    frames: list[TrainingFrame],
    settings: DetectorSettings,
    *,
    device: str,
    pretrained: bool = True,
    log: Callable[[str], object] = print,
) -> torch.nn.Module:
    """確認済みフレームから下書き用検出器を学習して返す。

    毎回ランダムにフレームを選び、ラベル済み範囲の内側から切り出した画像で学習する。
    最適化は SGD + OneCycleLR。seed を固定するので同じ入力なら同じ乱数列になる。
    途中経過（iter と loss）を log へ定期的に出す。戻り値は推論モードのモデル。
    """
    if not frames:
        raise ValueError("no checked frames to train the prelabel detector")
    rng = random.Random(settings.seed)
    torch.manual_seed(settings.seed)
    model = build_model(len(settings.labels), settings, pretrained=pretrained).to(device).train()
    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=settings.learning_rate, momentum=settings.momentum, weight_decay=settings.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=settings.learning_rate, total_steps=settings.iterations, pct_start=0.1
    )
    # ponytail: 全フレームをメモリに保持する。枚数が数千を超えるなら LRU へ切り替える。
    images: dict[Path, np.ndarray] = {}

    for step in range(1, settings.iterations + 1):
        inputs, targets = [], []
        for frame in rng.choices(frames, k=settings.batch_size):
            if frame.png_path not in images:
                image = cv2.imread(str(frame.png_path))
                if image is None:
                    raise ValueError(f"{frame.png_path}: cannot read image")
                images[frame.png_path] = image
            crop, boxes, names = sample_crop(
                images[frame.png_path], frame,
                crop_size=settings.crop_size, min_box_size=settings.min_box_size, rng=rng,
            )
            inputs.append(_to_tensor(crop, device))
            targets.append({
                "boxes": torch.from_numpy(boxes).to(device),
                "labels": torch.tensor([settings.labels.index(name) + 1 for name in names], dtype=torch.int64, device=device),
            })
        loss = sum(model(inputs, targets).values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if step % _LOG_EVERY == 0 or step == settings.iterations:
            log(f"iter {step}/{settings.iterations} loss={loss.item():.4f}")
    return model.eval()


def save_detector(model: torch.nn.Module, path: Path | str, settings: DetectorSettings) -> None:
    """学習済み重みを labels / input_scale のメタデータ付きで保存する。

    同じフォルダの一時ファイルへ書いてから置き換えるので、途中で失敗しても
    既存の重みファイルが壊れた状態で残らない。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": _WEIGHTS_FORMAT,
        "labels": list(settings.labels),
        "input_scale": float(settings.input_scale),
        "state_dict": model.state_dict(),
    }
    handle, temp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(handle)
    try:
        torch.save(payload, temp_name)
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


def load_detector(path: Path | str, settings: DetectorSettings, *, device: str) -> torch.nn.Module:
    """保存済み重みを安全に読み込み、設定と一致するか確かめて推論モデルを返す。

    torch.load は weights_only=True で読み、任意コードを実行しない。
    保存時の labels / input_scale が設定と違う場合は、学習と推論の条件がずれるのでエラー。
    モデルはダウンロード無しで構築してから重みを流し込む。
    """
    path = Path(path)
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except (OSError, RuntimeError, pickle.UnpicklingError, EOFError) as exc:
        raise ValueError(f"{path}: cannot read detector weights: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("format") != _WEIGHTS_FORMAT:
        raise ValueError(f"{path}: unsupported detector weights format")
    if payload.get("labels") != list(settings.labels):
        raise ValueError(f"{path}: detector labels {payload.get('labels')!r} do not match config {list(settings.labels)!r}")
    if payload.get("input_scale") != float(settings.input_scale):
        raise ValueError(
            f"{path}: detector input_scale {payload.get('input_scale')!r} does not match config {settings.input_scale!r}"
        )
    model = build_model(len(settings.labels), settings, pretrained=False)
    try:
        model.load_state_dict(payload.get("state_dict"))
    except (RuntimeError, TypeError, AttributeError) as exc:
        raise ValueError(f"{path}: detector weights do not fit the model: {exc}") from exc
    return model.to(device).eval()


def predict_boxes(
    model: torch.nn.Module, image_bgr: np.ndarray, settings: DetectorSettings, *, device: str
) -> list[LabelBox]:
    """フレーム全体を検出器にかけ、しきい値以上の矩形をスコア付きで返す。

    画像は学習時と同じ input_scale 倍に拡大して推論する。
    ラベル番号 i（1 始まり）は settings.labels[i-1] に対応し、矩形は画像内へクリップする。
    """
    height, width = image_bgr.shape[:2]
    model.eval()
    model.transform.min_size = (min(height, width) * settings.input_scale,)
    model.transform.max_size = max(height, width) * settings.input_scale
    with torch.no_grad():
        output = model([_to_tensor(image_bgr[..., :3], device)])[0]
    results = []
    for box, score, label in zip(
        output["boxes"].cpu().tolist(), output["scores"].cpu().tolist(), output["labels"].cpu().tolist()
    ):
        if score < settings.score_threshold:
            continue
        clipped = clip_box(
            LabelBox(settings.labels[label - 1], *box, score=float(score)), image_width=width, image_height=height
        )
        if clipped is not None:
            results.append(clipped)
    return results
