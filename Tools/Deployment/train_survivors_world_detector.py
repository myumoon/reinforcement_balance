"""world detector 学習ハーネス CLI。

COCO アノテーション済み dataset から SSDLite320 を学習し、
development checkpoint（formal_detector_eligible=false）を保存する。

学習前に split manifest（--split）を読み込み、error_calibration / final_e2e_test
session を dataset loader に渡して自動除外する。

session-balanced sampler で DatasetSample の実画像・target を構築して optimizer step する。
torch が不在の場合だけ stub ループにフォールバックする（例外の握り潰しなし）。

チェックポイント選択規則は学習開始前に manifest へ固定する。
best checkpoint は選択専用、model / optimizer / epoch を含む last checkpoint は resume 専用とする。

使用例:
    python train_survivors_world_detector.py \\
        --annotations data/world_annotations.json \\
        --split data/split.json \\
        --config configs/world_detector_v1.yaml \\
        --class-map configs/world_class_map_v1.yaml \\
        --output runs/world_detector_dev
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from typing import Any

import yaml


def _sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="world detector 学習ハーネス（開発用）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--annotations", required=True, help="COCO JSON アノテーションファイル")
    p.add_argument(
        "--split",
        required=True,
        help=(
            "SessionSplit JSON ファイル。train/validation/error_calibration/final_e2e_test"
            " の session ID を列挙する。error_calibration / final_e2e_test は自動除外され、"
            " train / validation は独立した Dataset view に分割される。"
        ),
    )
    p.add_argument("--config", default="configs/world_detector_v1.yaml", help="detector config YAML")
    p.add_argument("--class-map", default="configs/world_class_map_v1.yaml", help="class map YAML")
    p.add_argument("--output", required=True, help="checkpoint 出力ディレクトリ")
    p.add_argument("--epochs", type=int, default=None, help="学習エポック数（config 優先）")
    p.add_argument("--dry-run", action="store_true", help="preflight チェックのみ実行して終了")
    p.add_argument("--resume", help="resume する checkpoint ディレクトリ")
    p.add_argument(
        "--feasibility",
        help="feasibility JSON ファイル（04-06 の verdict=pass を要求する formal preflight 用）",
    )
    return p


# ---- checkpoint selection ----

@dataclass
class CheckpointRecord:
    """1 エポックのチェックポイント記録。

    val_map50_95 が高いほど良い checkpoint とする。
    """

    epoch: int
    val_map50_95: float
    checkpoint_path: str  # str で保存して JSON に対応する


@dataclass
class CheckpointSelector:
    """学習前にルールを manifest へ固定し、validation で best を一度だけ選ぶ。

    metric / keep_top_k は学習開始前に凍結され、後から変更できない。
    """

    metric: str
    keep_top_k: int
    _records: list[CheckpointRecord] = field(default_factory=list, repr=False)

    def record(self, rec: CheckpointRecord) -> None:
        """チェックポイントを記録し keep_top_k を超えた分を削除する。"""
        self._records.append(rec)
        self._records.sort(key=lambda r: r.val_map50_95, reverse=True)
        if len(self._records) > self.keep_top_k:
            self._records = self._records[: self.keep_top_k]

    @property
    def best(self) -> CheckpointRecord | None:
        """val_map50_95 が最大の checkpoint を返す（未選択なら None）。"""
        return self._records[0] if self._records else None

    def to_dict(self) -> dict:
        """manifest に保存できる dict 形式にシリアライズする。"""
        return {
            "metric": self.metric,
            "keep_top_k": self.keep_top_k,
            "best_epoch": self.best.epoch if self.best else None,
            "best_val_map50_95": self.best.val_map50_95 if self.best else None,
            "records": [asdict(r) for r in self._records],
        }

    @staticmethod
    def from_dict(d: dict) -> "CheckpointSelector":
        """dict から CheckpointSelector を復元する（resume 用）。"""
        sel = CheckpointSelector(
            metric=d.get("metric", "val_map50_95"),
            keep_top_k=d.get("keep_top_k", 3),
        )
        for r in d.get("records", []):
            sel._records.append(CheckpointRecord(**r))
        return sel


# ---- resume state ----

_RESUME_STATE_NAME = "training_state.json"
_LAST_CHECKPOINT_NAME = "last_checkpoint.pt"


def _save_resume_state(
    out_dir: pathlib.Path,
    epoch: int,
    selector: CheckpointSelector,
    model: Any,
    optimizer: Any,
) -> None:
    """model / optimizer / epoch / selector を単一 last checkpoint に保存する。

    best checkpoint は選択・publish 専用であり、resume には使用しない。
    last checkpoint の SHA-256 は metadata に保存し、復元前に検証する。
    """
    import torch

    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "selector": selector.to_dict(),
    }

    checkpoint_path = out_dir / _LAST_CHECKPOINT_NAME
    tmp_checkpoint: pathlib.Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=out_dir, prefix=f".{_LAST_CHECKPOINT_NAME}.", delete=False
        ) as tmp_file:
            tmp_checkpoint = pathlib.Path(tmp_file.name)
        torch.save(checkpoint, tmp_checkpoint)
        tmp_checkpoint.replace(checkpoint_path)
    finally:
        if tmp_checkpoint is not None and tmp_checkpoint.exists():
            tmp_checkpoint.unlink()

    metadata = {
        "schema_version": 1,
        "last_epoch": epoch,
        "last_checkpoint": _LAST_CHECKPOINT_NAME,
        "last_checkpoint_hash": _sha256_file(checkpoint_path),
    }
    metadata_path = out_dir / _RESUME_STATE_NAME
    tmp_metadata = metadata_path.with_suffix(".json.tmp")
    tmp_metadata.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    tmp_metadata.replace(metadata_path)


def _load_resume_state(
    resume_dir: pathlib.Path,
    selector: CheckpointSelector,
    model: Any,
    optimizer: Any,
) -> int:
    """resume_dir から training state を復元し、次の epoch 番号を返す。

    metadata / last checkpoint の欠落・hash 不一致・state 欠落は ValueError。
    best checkpoint は参照せず、単一 last checkpoint だけを復元する。
    """
    state_path = resume_dir / _RESUME_STATE_NAME
    if not state_path.exists():
        raise ValueError(
            f"[RESUME] training_state.json が見つかりません: {state_path}\n"
            "新規学習の場合は --resume を省略してください。"
        )

    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"[RESUME] training_state.json の解析に失敗しました: {e}") from e

    last_epoch: int = state.get("last_epoch")
    if not isinstance(last_epoch, int) or last_epoch < 1:
        raise ValueError(
            f"[RESUME] training_state.json の last_epoch が不正です: {last_epoch!r}"
        )

    if state.get("schema_version") != 1:
        raise ValueError(
            f"[RESUME] training_state.json の schema_version が不正です: "
            f"{state.get('schema_version')!r}"
        )
    if state.get("last_checkpoint") != _LAST_CHECKPOINT_NAME:
        raise ValueError("[RESUME] last checkpoint path が不正です。")
    expected_hash = state.get("last_checkpoint_hash")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError("[RESUME] last_checkpoint_hash が不正です。")

    checkpoint_path = resume_dir / _LAST_CHECKPOINT_NAME
    if not checkpoint_path.is_file():
        raise ValueError(f"[RESUME] last checkpoint が見つかりません: {checkpoint_path}")
    actual_hash = _sha256_file(checkpoint_path)
    if actual_hash != expected_hash:
        raise ValueError(
            f"[RESUME] last checkpoint hash が一致しません: "
            f"expected={expected_hash}, actual={actual_hash}"
        )

    try:
        import torch
    except ImportError as e:
        raise RuntimeError("[RESUME] last checkpoint の復元には torch が必要です。") from e
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if not isinstance(checkpoint, dict):
        raise ValueError("[RESUME] last checkpoint state が dict ではありません。")
    required_keys = {"epoch", "model_state_dict", "optimizer_state_dict", "selector"}
    missing = sorted(required_keys - checkpoint.keys())
    if missing:
        raise ValueError(f"[RESUME] last checkpoint state に {', '.join(missing)} がありません。")
    if checkpoint["epoch"] != last_epoch:
        raise ValueError(
            f"[RESUME] metadata/checkpoint epoch が一致しません: "
            f"{last_epoch} != {checkpoint['epoch']!r}"
        )

    saved_selector = CheckpointSelector.from_dict(checkpoint["selector"])
    if saved_selector.metric != selector.metric or saved_selector.keep_top_k != selector.keep_top_k:
        raise ValueError("[RESUME] checkpoint selection rule が現在の config と一致しません。")

    try:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    except Exception as e:
        raise ValueError(f"[RESUME] model/optimizer state の復元に失敗しました: {e}") from e
    selector._records = list(saved_selector._records)
    print(f"[RESUME] model/optimizer state を {checkpoint_path} から復元しました。")

    return last_epoch + 1


# ---- dataset split views ----

class _DatasetView:
    """WorldDataset の sample index を固定した読み取り専用 view。"""

    def __init__(self, dataset: Any, indices: list[int]) -> None:
        self._dataset = dataset
        self._indices = tuple(indices)

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int) -> Any:
        return self._dataset[self._indices[index]]

    def __iter__(self):
        for index in self._indices:
            yield self._dataset[index]


def _create_split_views(ds: Any, split: Any) -> tuple[_DatasetView, _DatasetView]:
    """session assignment から互いに素な train / validation view を作る。

    アノテーション無しフレーム（負例）は画像レベルの session_id で split に振り分ける。
    未知 session や train/validation が同一フレームに混在するデータは fail-closed で拒否する。
    """
    from survivors.vision.world_dataset import DatasetPreflightError

    train_sessions = set(split.train)
    validation_sessions = set(split.validation)
    allowed_sessions = train_sessions | validation_sessions
    train_indices: list[int] = []
    validation_indices: list[int] = []

    for index in range(len(ds)):
        sample = ds[index]
        sample_sessions = {ann.session_id for ann in sample.annotations if ann.session_id}
        if not sample_sessions:
            # 負例フレーム: 画像レベルの session_id で振り分ける
            img_session = getattr(sample, "session_id", "")
            if not img_session:
                continue
            sample_sessions = {img_session}
        unknown_sessions = sample_sessions - allowed_sessions
        if unknown_sessions:
            raise DatasetPreflightError(
                f"split に未割当の session {sorted(unknown_sessions)} が image_id="
                f"{sample.image_id} に含まれています。"
            )
        if sample_sessions <= train_sessions:
            train_indices.append(index)
        elif sample_sessions <= validation_sessions:
            validation_indices.append(index)
        else:
            raise DatasetPreflightError(
                f"train/validation session が image_id={sample.image_id} に混在しています。"
            )

    if not train_indices or not validation_indices:
        raise DatasetPreflightError("train/validation Dataset view の一方が空です。")
    return _DatasetView(ds, train_indices), _DatasetView(ds, validation_indices)


# ---- config guard ----

def _run_formal_preflight(feasibility_path: pathlib.Path, cfg: dict) -> None:
    """feasibility JSON を読み込み、formal 学習に必要な条件を検証する。

    04-06 フェーズの verdict が pass でなければ ValueError を送出する。
    architecture が config と一致しなければ ValueError を送出する。
    """
    raw = json.loads(feasibility_path.read_text(encoding="utf-8"))
    verdict = raw.get("verdict")
    if verdict != "pass":
        raise ValueError(
            f"feasibility verdict が 'pass' ではありません: {verdict!r}。"
            " 04-06 の feasibility チェックを通過してから再実行してください。"
        )
    # architecture 一致確認
    feas_arch = raw.get("architecture")
    cfg_arch = cfg.get("model", {}).get("architecture")
    if feas_arch and cfg_arch and feas_arch != cfg_arch:
        raise ValueError(
            f"feasibility の architecture {feas_arch!r} が config {cfg_arch!r} と一致しません。"
        )


def _reject_unimplemented_config(cfg: dict) -> None:
    """未実装の config キーが設定されていれば ValueError を送出する。

    config hash と実際の学習条件が乖離しないよう、未実装設定を早期に拒否する。
    """
    checks = [
        ("input.augmentation", cfg.get("input", {}).get("augmentation")),
        ("training.lr_scheduler", cfg.get("training", {}).get("lr_scheduler")),
        ("training.class_weights", cfg.get("training", {}).get("class_weights")),
        ("training.near_duplicate_suppression",
         cfg.get("training", {}).get("near_duplicate_suppression")),
    ]
    for key, val in checks:
        if val is not None:
            raise ValueError(
                f"config '{key}' は未実装です。このバージョンでは設定を除去してください。"
            )


# ---- training loop ----

def _run_training(
    detector: Any,
    train_ds: Any,
    validation_ds: Any,
    cfg: dict,
    max_epochs: int,
    selector: CheckpointSelector,
    out_dir: pathlib.Path,
    resume_dir: pathlib.Path | None = None,
) -> None:
    """学習ループ。

    torch が利用できる場合は実 SGD ステップを実行し、
    そうでない場合は stub エポックループを回す。
    resume_dir が指定された場合は model / optimizer / selector を復元する。
    """
    try:
        import torch as _torch
        import torch.optim as optim
    except ImportError:
        # torch 不在: stub ループ。resume は torch 必須なので即 RuntimeError。
        if resume_dir is not None:
            raise RuntimeError("[RESUME] model/optimizer state の復元には torch が必要です。")
        _run_training_stub(max_epochs, selector, out_dir, start_epoch=1)
        return

    # torch 利用可能。以降の ImportError（cv2・torchvision 等）は伝播させる。
    opt_cfg = cfg.get("training", {}).get("optimizer", {}).get("sgd", {})
    optimizer = optim.SGD(
        detector._model.parameters(),
        lr=opt_cfg.get("lr", 0.01),
        momentum=opt_cfg.get("momentum", 0.9),
        weight_decay=opt_cfg.get("weight_decay", 0.0005),
    )

    start_epoch = 1
    if resume_dir is not None:
        start_epoch = _load_resume_state(resume_dir, selector, detector._model, optimizer)
        print(f"[RESUME] epoch {start_epoch} から再開します。")

    _run_training_torch(
        detector,
        train_ds,
        validation_ds,
        cfg,
        max_epochs,
        selector,
        out_dir,
        optimizer,
        start_epoch,
    )


def _run_training_torch(
    detector: Any,
    train_ds: Any,
    validation_ds: Any,
    cfg: dict,
    max_epochs: int,
    selector: CheckpointSelector,
    out_dir: pathlib.Path,
    optimizer: Any,
    start_epoch: int,
) -> None:
    """torch DataLoader + SGD optimizer の実学習ループ。

    DatasetSample の実 annotation から target を構築し optimizer step を実行する。
    画像ファイルが存在しない場合は合成画像にフォールバックする（smoke 対応）。
    """
    import torch

    save_every = cfg.get("checkpoint_selection", {}).get("save_every_n_epochs", 5)

    for epoch in range(start_epoch, max_epochs + 1):
        # session-balanced sampler: train view の実サンプルだけで optimizer step
        _train_epoch(detector, train_ds, optimizer, cfg)

        # selector の score は独立した validation view だけから得る
        val_score = _evaluate_validation(detector, validation_ds, cfg)

        if epoch % save_every == 0 or epoch == max_epochs:
            ckpt_path = out_dir / f"epoch_{epoch:04d}.pt"
            torch.save(detector._model.state_dict(), ckpt_path)
            selector.record(CheckpointRecord(epoch, val_score, str(ckpt_path)))
            print(f"  epoch {epoch}/{max_epochs}: val_map50_95={val_score:.4f} → saved {ckpt_path.name}")
        # resume 用 last は best 候補の保存間隔に関係なく毎 epoch 更新する
        _save_resume_state(out_dir, epoch, selector, detector._model, optimizer)

    print(f"[INFO] best checkpoint: epoch={selector.best.epoch if selector.best else None}")


def _load_training_sample(sample: Any) -> tuple[Any, dict[str, Any]]:
    """DatasetSample を torchvision detection model の image/target へ変換する。"""
    import torch

    img_path = pathlib.Path(sample.file_name)
    frame: Any = None
    if img_path.exists():
        # ファイルが存在する場合は ImportError・デコード失敗を伝播させる（smoke モード用ゼロ画像不可）
        import cv2
        import torchvision.transforms.functional as TF
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            raise OSError(f"cv2.imread デコード失敗: {img_path}")
        frame = TF.to_tensor(bgr[..., ::-1].copy())

    if frame is None:
        frame = torch.zeros(3, sample.image_height, sample.image_width)

    boxes_list = []
    labels_list = []
    for ann in sample.annotations:
        x, y, w, h = ann.bbox_xywh
        boxes_list.append([x, y, x + w, y + h])
        labels_list.append(ann.category_id)

    if boxes_list:
        target = {
            "boxes": torch.tensor(boxes_list, dtype=torch.float32),
            "labels": torch.tensor(labels_list, dtype=torch.long),
        }
    else:
        target = {
            "boxes": torch.zeros(0, 4),
            "labels": torch.zeros(0, dtype=torch.long),
        }
    return frame, target


def _session_balanced_indices(ds: Any) -> list[int]:
    """各 session から等数ずつサンプルを取り session 間不均衡を抑える。

    最短 session の長さを quota として全 session を上限制限し round-robin で並べる。
    長期 session が 1 epoch を独占しなくなる。
    # ponytail: min-quota。サンプル量が問題になるなら per-session weight sampling へ移行。
    """
    session_groups: dict[str, list[int]] = {}
    for index in range(len(ds)):
        sample = ds[index]
        session_ids = sorted({ann.session_id for ann in sample.annotations if ann.session_id})
        session_id = session_ids[0] if session_ids else ""
        session_groups.setdefault(session_id, []).append(index)

    quota = min(len(indices) for indices in session_groups.values())
    ordered: list[int] = []
    for offset in range(quota):
        for indices in session_groups.values():
            ordered.append(indices[offset])
    return ordered


def _train_epoch(
    detector: Any, ds: Any, optimizer: Any, cfg: dict
) -> None:
    """session-balanced sampler で config batch_size 単位の 1 epoch を学習する。

    複数 session を round-robin して batch を作り、SSDLite BatchNorm が
    batch=1 にならないよう singleton remainder は学習へ渡さない。
    画像ファイルが不在の場合は合成画像（ゼロ）にフォールバックする。
    例外を握り潰さない（FileNotFoundError のみ合成画像でリカバー）。
    """
    if len(ds) == 0:
        return

    batch_size = cfg.get("training", {}).get("batch_size")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 2:
        raise ValueError("training.batch_size は 2 以上の整数である必要があります。")

    ordered_indices = _session_balanced_indices(ds)
    for start in range(0, len(ordered_indices), batch_size):
        batch_indices = ordered_indices[start:start + batch_size]
        if len(batch_indices) == 1:
            print("[WARN] BatchNorm 保護のため singleton training batch をスキップします。")
            continue
        images = []
        targets = []
        for sample_index in batch_indices:
            image, target = _load_training_sample(ds[sample_index])
            images.append(image)
            targets.append(target)

        optimizer.zero_grad()
        detector._model.train()
        loss_dict = detector._model(images, targets)
        if isinstance(loss_dict, dict):
            total_loss = sum(loss_dict.values())  # type: ignore[arg-type]
            total_loss.backward()
            optimizer.step()
        # stub model が list を返す場合はスキップ（smoke: 推論のみ）


def _evaluate_validation(detector: Any, validation_ds: Any, cfg: dict) -> float:
    """validation 専用 view で detector を推論評価し proxy_ap50_95 を返す。

    torch eval モードでサンプルを1件ずつ推論し、evaluate_from_predictions で AP を計算する。
    stub model（テスト環境）は空予測を返すため AP=0.0 になる。
    """
    import torch
    from eval_survivors_world_detector import evaluate_from_predictions

    if len(validation_ds) == 0:
        raise ValueError("validation Dataset view が空です。")

    gt_anns: list[dict] = []
    pred_anns: list[dict] = []

    detector._model.eval()
    with torch.no_grad():
        for idx in range(len(validation_ds)):
            sample = validation_ds[idx]
            image_id = int(getattr(sample, "image_id", idx))
            img, target = _load_training_sample(sample)
            for box, label in zip(target["boxes"].tolist(), target["labels"].tolist()):
                x1, y1, x2, y2 = box
                gt_anns.append({
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                })
            preds = detector._model([img])
            if isinstance(preds, list) and preds and isinstance(preds[0], dict):
                for box, label, score in zip(
                    preds[0].get("boxes", torch.zeros(0, 4)).tolist(),
                    preds[0].get("labels", torch.zeros(0, dtype=torch.long)).tolist(),
                    preds[0].get("scores", torch.zeros(0)).tolist(),
                ):
                    x1, y1, x2, y2 = box
                    pred_anns.append({
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(score),
                    })

    num_classes = cfg.get("model", {}).get("num_classes", detector.num_classes) + 1
    metrics = evaluate_from_predictions(gt_anns, pred_anns, num_classes=num_classes)
    return metrics.proxy_ap50_95


def _run_training_stub(
    max_epochs: int,
    selector: CheckpointSelector,
    out_dir: pathlib.Path,
    start_epoch: int,
) -> None:
    """torch 不在環境用のスタブ学習ループ。"""
    for epoch in range(start_epoch, max_epochs + 1):
        val_score = 0.0
        ckpt_path = out_dir / f"epoch_{epoch:04d}.ckpt"
        selector.record(CheckpointRecord(epoch, val_score, str(ckpt_path)))
        print(f"  epoch {epoch}/{max_epochs} ... (stub: weight 更新なし)")


# ---- entry point ----

def main(argv: list[str] | None = None) -> int:
    """学習ハーネスのエントリポイント。

    preflight 失敗時は exit code 1、成功時は 0。
    """
    args = _build_arg_parser().parse_args(argv)

    from survivors.vision.world_dataset import WorldDataset, DatasetPreflightError, SessionSplit
    from survivors.vision.world_detector import (
        WorldDetector,
        CheckpointManifest,
        load_detector_config,
        UnknownArchitectureError,
    )

    ann_path = pathlib.Path(args.annotations)
    cfg_path = pathlib.Path(args.config)
    cm_path = pathlib.Path(args.class_map)
    out_dir = pathlib.Path(args.output)

    # -- config
    try:
        cfg = load_detector_config(cfg_path)
    except ValueError as e:
        print(f"[ERROR] config 読み込み失敗: {e}", file=sys.stderr)
        return 1

    # -- 必須 split manifest 読み込み（error_calibration / final_e2e_test を自動除外）
    split_data = json.loads(pathlib.Path(args.split).read_text(encoding="utf-8"))
    split = SessionSplit(
        train=split_data.get("train", []),
        validation=split_data.get("validation", []),
        error_calibration=split_data.get("error_calibration", []),
        final_e2e_test=split_data.get("final_e2e_test", []),
    )
    rejected = set(split.error_calibration) | set(split.final_e2e_test)
    print(f"[INFO] split: train={len(split.train)}, val={len(split.validation)}"
          f", rejected={len(rejected)} session")

    # -- preflight
    preflight_cfg = cfg.get("training", {}).get("preflight", {})
    try:
        ds = WorldDataset(
            ann_path, cm_path,
            validate_bounds=True,
            rejected_sessions=rejected,
        )
        ds.run_preflight(
            min_frames=preflight_cfg.get("min_frames", 300),
            min_entities=preflight_cfg.get("min_entities", 500),
            min_classes=preflight_cfg.get("min_classes", 6),
            min_time_bands=preflight_cfg.get("min_time_bands", 4),
        )
        ds.run_split_preflight(
            split,
            min_frames_per_split=preflight_cfg.get("min_frames_per_split", 50),
            min_entities_per_split=preflight_cfg.get("min_entities_per_split", 100),
            min_classes_per_split=preflight_cfg.get("min_classes_per_split", 4),
        )
        train_ds, validation_ds = _create_split_views(ds, split)
    except DatasetPreflightError as e:
        print(f"[PREFLIGHT FAIL] {e}", file=sys.stderr)
        print("[INFO] training step 0 のまま停止します。", file=sys.stderr)
        return 1

    print(
        f"[PREFLIGHT OK] train={len(train_ds)}, validation={len(validation_ds)} "
        "フレームへ分離しました。"
    )

    # formal preflight（--feasibility が指定された場合のみ）
    if args.feasibility:
        try:
            _run_formal_preflight(pathlib.Path(args.feasibility), cfg)
        except (ValueError, OSError) as e:
            print(f"[FORMAL PREFLIGHT ERROR] {e}", file=sys.stderr)
            return 1

    # 未実装設定の早期拒否（config hash と学習条件の乖離を防ぐ）
    try:
        _reject_unimplemented_config(cfg)
    except ValueError as e:
        print(f"[CONFIG ERROR] {e}", file=sys.stderr)
        return 1

    if args.dry_run:
        print("[DRY-RUN] --dry-run が指定されたため学習をスキップします。")
        return 0

    # -- build model
    try:
        detector = WorldDetector.from_config(cfg, cm_path)
    except UnknownArchitectureError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1

    # -- checkpoint selection rules を学習開始前に固定する
    sel_cfg = cfg.get("checkpoint_selection", {})
    selector = CheckpointSelector(
        metric=sel_cfg.get("metric", "val_map50_95"),
        keep_top_k=sel_cfg.get("keep_top_k", 3),
    )
    print(f"[INFO] checkpoint_selection: metric={selector.metric}, keep_top_k={selector.keep_top_k}")

    # -- train
    print(f"[INFO] アーキテクチャ: {cfg['model']['architecture']}, num_classes={detector.num_classes}")
    max_epochs = args.epochs or cfg.get("training", {}).get("max_epochs", 1)
    print(f"[INFO] max_epochs={max_epochs} で学習を開始します...")

    out_dir.mkdir(parents=True, exist_ok=True)
    resume_dir: pathlib.Path | None = pathlib.Path(args.resume) if args.resume else None
    _run_training(
        detector, train_ds, validation_ds, cfg,
        max_epochs=max_epochs,
        selector=selector,
        out_dir=out_dir,
        resume_dir=resume_dir,
    )

    # -- build hash
    _build_hash = _sha256_str(f"dev-python:{sys.version}")

    # split hash: split JSON ファイルの内容 SHA-256
    _split_hash = _sha256_file(pathlib.Path(args.split))

    # resolved config hash: CLI override 適用後の config dict を正規化してハッシュ
    _resolved_cfg = dict(cfg)
    if args.epochs:
        _resolved_cfg.setdefault("training", {})["max_epochs"] = args.epochs
    _resolved_config_hash = _sha256_str(json.dumps(_resolved_cfg, sort_keys=True, ensure_ascii=False))

    # best checkpoint hash: 実ファイルがあれば hash、なければ config hash で代用
    best = selector.best
    weight_path: pathlib.Path | None = None
    if best and pathlib.Path(best.checkpoint_path).exists():
        weight_path = pathlib.Path(best.checkpoint_path)
        model_hash = _sha256_file(weight_path)
    else:
        model_hash = _sha256_file(cfg_path)  # stub: config hash で代用

    manifest = CheckpointManifest(
        model_hash=model_hash,
        data_hash=_sha256_file(ann_path),
        config_hash=_sha256_file(cfg_path),
        build_hash=_build_hash,
        class_map_hash=_sha256_file(cm_path),
        formal_detector_eligible=False,
        split_hash=_split_hash,
        resolved_config_hash=_resolved_config_hash,
    )
    manifest.save(out_dir / "manifest.json")

    # checkpoint selection 規則を manifest JSON に追記する
    m_path = out_dir / "manifest.json"
    m_data = json.loads(m_path.read_text(encoding="utf-8"))
    m_data["checkpoint_selection"] = selector.to_dict()
    m_path.write_text(json.dumps(m_data, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[INFO] manifest を {m_path} へ保存しました。")
    print("[INFO] formal_detector_eligible=false — このままでは正式パッケージへ昇格できません。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
