"""Survivors アノテーションの共通ラベル一覧とファイル処理を提供する。

各 CLI はこのモジュールのクラス検証、矩形、X-AnyLabeling 入出力を共有する。
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from reinbalance_survivors_contracts.deploy_obs_v2_features import load_deploy_obs_v2_feature_params
from survivors.vision.world_dataset import load_class_map


DEFAULT_CLASS_MAP_PATH = Path(__file__).resolve().parents[1] / "configs" / "world_class_map_v2.yaml"
_CLASS_MAP = load_class_map(DEFAULT_CLASS_MAP_PATH)
WORLD_CLASSES = tuple(
    item["name"] for item in sorted(_CLASS_MAP.foreground_classes, key=lambda item: item["id"])
)
UI_CLASSES = ("hud_hp", "hud_xp", "card", "button", "death_result")
# この矩形を持つ確認済みフレームは「矩形の内側だけ漏れなくラベル済み」を意味する（物体ではない）。
REGION_LABEL = "labeled_region"
# プレイヤー自身の武器エフェクト（敵側の hazard_* とは別）。class map v2 の大分類 weapon のクラスで、
# WORLD_CLASSES にも含まれる。種類は sim の EProjectileObsKind（Projectile / GroundZone / Orbit / Aura）に対応する。
WEAPON_EFFECT_CLASSES = tuple(name for name in WORLD_CLASSES if _CLASS_MAP.coarse_for(name) == "weapon")
# 武器（C++ EWeaponType 名）→ 画面に付けるラベル集合。Common の deploy_obs_v2_features.yaml
# weapon_effect_kinds（種類名 projectile 等）に weapon_ を前置して作り、表を二重に持たない。
# Common に無い武器（Pentagram など）は空集合で、ラベルを付けない。
_FEATURE_PARAMS = load_deploy_obs_v2_feature_params()
WEAPON_EFFECT_KINDS: dict[str, tuple[str, ...]] = {
    weapon: tuple(f"weapon_{kind}" for kind in _FEATURE_PARAMS["weapon_effect_kinds"].get(weapon, ()))
    for weapon in _FEATURE_PARAMS["weapon_vocabulary"][1:-1]
}
ALL_CLASSES = WORLD_CLASSES + UI_CLASSES + (REGION_LABEL,)
_FRAME_STEM = re.compile(r"^\d{8}$")


@dataclass(frozen=True)
class LabelBox:
    """画像内の名前付き矩形と任意の検出スコアを表す。

    left, top, right, bottom はピクセル座標で、right と bottom は排他的な端点。
    """

    label: str
    left: float
    top: float
    right: float
    bottom: float
    score: float | None = None


def validate_label(name: str) -> str:
    """validate_label(name) -> str として共有クラス一覧を検証する。

    有効な名前をそのまま返し、未知の名前は ValueError にする。
    """
    if name not in ALL_CLASSES:
        raise ValueError(f"unknown annotation label: {name!r}")
    return name


def clip_box(
    box: LabelBox, *, image_width: int, image_height: int
) -> LabelBox | None:
    """clip_box(box, *, image_width, image_height) で矩形を画像内へ収める。

    面積が 0 になる矩形は None、画像範囲に残る矩形は新しい LabelBox で返す。
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    validate_label(box.label)
    coordinates = (box.left, box.top, box.right, box.bottom)
    if not all(math.isfinite(float(value)) for value in coordinates):
        raise ValueError("box coordinates must be finite")

    left = min(max(float(box.left), 0.0), float(image_width))
    top = min(max(float(box.top), 0.0), float(image_height))
    right = min(max(float(box.right), 0.0), float(image_width))
    bottom = min(max(float(box.bottom), 0.0), float(image_height))
    if right <= left or bottom <= top:
        return None
    return LabelBox(box.label, left, top, right, bottom, box.score)


def write_label_file(
    path: Path | str,
    boxes: Iterable[LabelBox],
    *,
    image_width: int = 1920,
    image_height: int = 1080,
    checked: bool = False,
    overwrite: bool = True,
) -> bool:
    """write_label_file(path, boxes, *, overwrite=True) でラベルを保存する。

    上書き時は一時ファイルから置換し、no-clobber 時は既存ファイルをそのまま残す。
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if not isinstance(checked, bool):
        raise ValueError("checked must be a boolean")
    if not isinstance(overwrite, bool):
        raise ValueError("overwrite must be a boolean")

    path = Path(path)
    shapes: list[dict] = []
    for box in boxes:
        validate_label(box.label)
        clipped = clip_box(box, image_width=image_width, image_height=image_height)
        if clipped is None:
            continue
        score = None if clipped.score is None else float(clipped.score)
        if score is not None and not math.isfinite(score):
            raise ValueError("box score must be finite")
        shapes.append(
            {
                "label": clipped.label,
                "score": score,
                "points": [
                    [float(clipped.left), float(clipped.top)],
                    [float(clipped.right), float(clipped.top)],
                    [float(clipped.right), float(clipped.bottom)],
                    [float(clipped.left), float(clipped.bottom)],
                ],
                "group_id": None,
                "description": "",
                "difficult": False,
                "shape_type": "rectangle",
                "flags": {},
                "attributes": {},
            }
        )

    payload = {
        "version": "4.0.6",
        "flags": {},
        "shapes": shapes,
        "imagePath": path.with_suffix(".png").name,
        "imageData": None,
        "imageHeight": image_height,
        "imageWidth": image_width,
        "checked": checked,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        if overwrite:
            os.replace(temporary_name, path)
        else:
            try:
                os.link(temporary_name, path)
            except FileExistsError:
                return False
            except OSError:
                try:
                    descriptor = os.open(
                        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666
                    )
                except FileExistsError:
                    return False
                try:
                    with os.fdopen(descriptor, "wb") as destination, open(
                        temporary_name, "rb"
                    ) as source:
                        shutil.copyfileobj(source, destination)
                except BaseException:
                    path.unlink(missing_ok=True)
                    raise
        return True
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
                pass
        raise
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


_SHAPE_POINT_COUNTS = {"rectangle": (2, 4), "circle": (2,)}


def read_label_file(path: Path | str) -> tuple[list[LabelBox], bool]:
    """read_label_file(path) -> (boxes, checked) としてラベルファイルを読む。

    rectangle（対角二点または四隅四点）と circle（中心点＋円周上の一点）を受け入れる。
    未知ラベルや未対応 shape_type の shape は警告を表示して読み飛ばし、
    欠損キーや不正な点座標など壊れた形式だけをパス付き ValueError にする。
    """
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: cannot read label file: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: label file must contain a JSON object")

    missing = [key for key in ("shapes", "imagePath", "imageData") if key not in payload]
    if missing:
        raise ValueError(f"{path}: missing required keys: {', '.join(missing)}")
    if not isinstance(payload["shapes"], list):
        raise ValueError(f"{path}: shapes must be a list")

    boxes: list[LabelBox] = []
    for index, shape in enumerate(payload["shapes"]):
        prefix = f"{path}: shape {index}"
        if not isinstance(shape, dict):
            raise ValueError(f"{prefix} must be an object")
        if "shape_type" not in shape or "points" not in shape or "label" not in shape:
            raise ValueError(f"{prefix} is missing label, shape_type, or points")
        shape_type = shape["shape_type"]
        if shape_type not in _SHAPE_POINT_COUNTS:
            print(f"warning: {prefix} has unsupported shape_type {shape_type!r}, skipped", file=sys.stderr)
            continue
        label = shape["label"]
        if label not in ALL_CLASSES:
            print(f"warning: {prefix} has unknown annotation label {label!r}, skipped", file=sys.stderr)
            continue

        points = shape["points"]
        if not isinstance(points, list) or len(points) not in _SHAPE_POINT_COUNTS[shape_type]:
            raise ValueError(f"{prefix} {shape_type} points must contain {_SHAPE_POINT_COUNTS[shape_type]} points")
        parsed_points: list[tuple[float, float]] = []
        for point in points:
            if not isinstance(point, list) or len(point) != 2:
                raise ValueError(f"{prefix} has an invalid point: {point!r}")
            if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in point):
                raise ValueError(f"{prefix} point coordinates must be numbers")
            x, y = float(point[0]), float(point[1])
            if not math.isfinite(x) or not math.isfinite(y):
                raise ValueError(f"{prefix} point coordinates must be finite")
            parsed_points.append((x, y))

        score = shape.get("score")
        if score is not None:
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(float(score)):
                raise ValueError(f"{prefix} score must be a finite number or null")
            score = float(score)

        if shape_type == "circle":
            (cx, cy), (ex, ey) = parsed_points
            radius = math.hypot(ex - cx, ey - cy)
            boxes.append(LabelBox(label, cx - radius, cy - radius, cx + radius, cy + radius, score))
        else:
            xs = [x for x, _ in parsed_points]
            ys = [y for _, y in parsed_points]
            boxes.append(LabelBox(label, min(xs), min(ys), max(xs), max(ys), score))

    return boxes, payload.get("checked", False) is True


def iter_frame_files(
    session_dir: Path | str,
) -> Iterator[tuple[int, Path | None, Path | None]]:
    """iter_frame_files(session_dir) は (frame_id, png_path, json_path) を昇順に返す。

    八桁数字 stem の PNG/JSON だけを列挙し、片方の欠落は None で示す。
    """
    files: dict[str, dict[str, Path]] = {}
    for path in Path(session_dir).iterdir():
        if not path.is_file() or path.suffix not in {".png", ".json"}:
            continue
        if not _FRAME_STEM.fullmatch(path.stem):
            continue
        files.setdefault(path.stem, {})[path.suffix] = path

    for stem in sorted(files, key=int):
        frame_files = files[stem]
        yield (
            int(stem),
            frame_files.get(".png"),
            frame_files.get(".json"),
        )
