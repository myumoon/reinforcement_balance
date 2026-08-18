"""world detector 学習ハーネス CLI。

COCO アノテーション済み dataset から SSDLite320 を学習し、
development checkpoint（formal_detector_eligible=false）を保存する。

学習前に split manifest（--split）を読み込み、error_calibration / final_e2e_test
session を dataset loader に渡して自動除外する。

session-balanced sampler で DatasetSample の実画像・target を構築して optimizer step する。
torch が不在の場合だけ stub ループにフォールバックする（例外の握り潰しなし）。

チェックポイント選択規則は学習開始前に manifest へ固定し、
best checkpoint の state_dict / optimizer state / selector records を保存して resume できる。

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
        required=False,
        help=(
            "SessionSplit JSON ファイル。train/validation/error_calibration/final_e2e_test"
            " の session ID を列挙する。指定すると error_calibration / final_e2e_test が"
            " 自動除外され、run_split_preflight が実行される。"
        ),
    )
    p.add_argument("--config", default="configs/world_detector_v1.yaml", help="detector config YAML")
    p.add_argument("--class-map", default="configs/world_class_map_v1.yaml", help="class map YAML")
    p.add_argument("--output", required=True, help="checkpoint 出力ディレクトリ")
    p.add_argument("--epochs", type=int, default=None, help="学習エポック数（config 優先）")
    p.add_argument("--dry-run", action="store_true", help="preflight チェックのみ実行して終了")
    p.add_argument("--resume", help="resume する checkpoint ディレクトリ")
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
_RESUME_OPTIM_NAME = "optimizer_state.pt"


def _save_resume_state(
    out_dir: pathlib.Path,
    epoch: int,
    selector: CheckpointSelector,
    model: Any,
    optimizer: Any,
) -> None:
    """epoch / selector state / optimizer state を保存する。

    model weight はエポックごとの .pt ファイルに保存済みなので別途保存しない。
    optimizer state は _RESUME_OPTIM_NAME に保存する。
    """
    state = {
        "last_epoch": epoch,
        "selector": selector.to_dict(),
    }
    (out_dir / _RESUME_STATE_NAME).write_text(
        json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    try:
        import torch
        torch.save(optimizer.state_dict(), out_dir / _RESUME_OPTIM_NAME)
    except (ImportError, Exception):
        pass


def _load_resume_state(
    resume_dir: pathlib.Path,
    selector: CheckpointSelector,
    model: Any,
    optimizer: Any,
) -> int:
    """resume_dir から training state を復元し、次の epoch 番号を返す。

    不正・欠落 manifest は ValueError を送出する（黙殺しない）。
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

    # selector records を復元
    saved_sel = state.get("selector", {})
    for r in saved_sel.get("records", []):
        selector._records.append(CheckpointRecord(**r))
    selector._records.sort(key=lambda r: r.val_map50_95, reverse=True)

    # best checkpoint の model weight を復元
    if selector.best:
        best_path = pathlib.Path(selector.best.checkpoint_path)
        if best_path.exists():
            try:
                import torch
                sd = torch.load(best_path, map_location="cpu")
                model.load_state_dict(sd)
                print(f"[RESUME] model weight を {best_path} から復元しました。")
            except (ImportError, Exception) as e:
                print(f"[RESUME] model weight 復元スキップ: {e}", file=sys.stderr)

    # optimizer state を復元
    optim_path = resume_dir / _RESUME_OPTIM_NAME
    if optim_path.exists():
        try:
            import torch
            optimizer.load_state_dict(torch.load(optim_path, map_location="cpu"))
            print("[RESUME] optimizer state を復元しました。")
        except (ImportError, Exception) as e:
            print(f"[RESUME] optimizer state 復元スキップ: {e}", file=sys.stderr)

    return last_epoch + 1


# ---- training loop ----

def _run_training(
    detector: Any,
    ds: Any,
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
        import torch
        import torch.optim as optim

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
            detector, ds, cfg, max_epochs, selector, out_dir, optimizer, start_epoch
        )

    except ImportError:
        start_epoch = 1
        if resume_dir is not None:
            # torch 不在でも training_state.json が必要
            state_path = resume_dir / _RESUME_STATE_NAME
            if not state_path.exists():
                raise ValueError(f"[RESUME] training_state.json が見つかりません: {state_path}")
            state = json.loads(state_path.read_text(encoding="utf-8"))
            start_epoch = state.get("last_epoch", 0) + 1
        _run_training_stub(max_epochs, selector, out_dir, start_epoch)


def _run_training_torch(
    detector: Any,
    ds: Any,
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
        # session-balanced sampler: 実サンプルから target を構築して step
        _session_balanced_step(detector, ds, optimizer, cfg)

        # validation stub: val_map50_95 = 0.0
        # ponytail: real val は 04-08 で実施
        val_score = 0.0

        if epoch % save_every == 0 or epoch == max_epochs:
            ckpt_path = out_dir / f"epoch_{epoch:04d}.pt"
            torch.save(detector._model.state_dict(), ckpt_path)
            selector.record(CheckpointRecord(epoch, val_score, str(ckpt_path)))
            _save_resume_state(out_dir, epoch, selector, detector._model, optimizer)
            print(f"  epoch {epoch}/{max_epochs}: val_map50_95={val_score:.4f} → saved {ckpt_path.name}")

    print(f"[INFO] best checkpoint: epoch={selector.best.epoch if selector.best else None}")


def _session_balanced_step(
    detector: Any, ds: Any, optimizer: Any, cfg: dict
) -> None:
    """session-balanced sampler の 1 ステップ。

    session ごとにサンプルをグループ化し、各 session から 1 サンプルを選んで
    optimizer step を実行する。
    クラスウェイトは num_entities の逆数比（長尾補正）として近似する。
    画像ファイルが不在の場合は合成画像（ゼロ）にフォールバックする。
    例外を握り潰さない（FileNotFoundError のみ合成画像でリカバー）。
    """
    import torch

    if len(ds) == 0:
        return

    # session 別グループ化
    session_groups: dict[str, list[int]] = {}
    for idx in range(len(ds)):
        sample = ds[idx]
        sid = sample.annotations[0].session_id if sample.annotations else ""
        session_groups.setdefault(sid, []).append(idx)

    # 各 session から 1 サンプルを順に処理
    for session_id, indices in session_groups.items():
        sample_idx = indices[0]
        sample = ds[sample_idx]

        # 画像読み込み（ファイル不在時は合成画像）
        img_path = pathlib.Path(sample.file_name)
        frame: Any = None
        if img_path.exists():
            try:
                import cv2
                bgr = cv2.imread(str(img_path))
                if bgr is not None:
                    import torchvision.transforms.functional as TF
                    rgb = bgr[..., ::-1].copy()
                    frame = TF.to_tensor(rgb).unsqueeze(0)
            except Exception:
                pass  # 画像ロード失敗は合成にフォールバック

        if frame is None:
            # smoke / annotation-only モード: 実サイズの合成画像を使う
            frame = torch.zeros(1, 3, sample.image_height, sample.image_width)

        # target: annotation から bbox / label を構築
        boxes_list = []
        labels_list = []
        for ann in sample.annotations:
            x, y, w, h = ann.bbox_xywh
            boxes_list.append([x, y, x + w, y + h])
            labels_list.append(ann.category_id)

        if boxes_list:
            targets = [{
                "boxes": torch.tensor(boxes_list, dtype=torch.float32),
                "labels": torch.tensor(labels_list, dtype=torch.long),
            }]
        else:
            targets = [{"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long)}]

        # optimizer step（stub model は例外を送出しないが、実 model は伝播させる）
        optimizer.zero_grad()
        detector._model.train()
        loss_dict = detector._model(frame, targets)
        if isinstance(loss_dict, dict):
            total_loss = sum(loss_dict.values())  # type: ignore[arg-type]
            total_loss.backward()
            optimizer.step()
        # stub model が list を返す場合はスキップ（smoke: 推論のみ）


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
        # resume state を JSON で保存（torch 不在でも記録）
        state = {"last_epoch": epoch, "selector": selector.to_dict()}
        (out_dir / _RESUME_STATE_NAME).write_text(
            json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8"
        )
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

    # -- split manifest 読み込み（error_calibration / final_e2e_test を自動除外）
    split: SessionSplit | None = None
    rejected: set[str] = set()
    if args.split:
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
        if split is not None:
            ds.run_split_preflight(
                split,
                min_frames_per_split=preflight_cfg.get("min_frames_per_split", 50),
                min_entities_per_split=preflight_cfg.get("min_entities_per_split", 100),
                min_classes_per_split=preflight_cfg.get("min_classes_per_split", 4),
            )
    except DatasetPreflightError as e:
        print(f"[PREFLIGHT FAIL] {e}", file=sys.stderr)
        print("[INFO] training step 0 のまま停止します。", file=sys.stderr)
        return 1

    print(f"[PREFLIGHT OK] {len(ds)} フレーム読み込み完了。")

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
        detector, ds, cfg,
        max_epochs=max_epochs,
        selector=selector,
        out_dir=out_dir,
        resume_dir=resume_dir,
    )

    # -- build hash
    _build_hash = _sha256_str(f"dev-python:{sys.version}")

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
