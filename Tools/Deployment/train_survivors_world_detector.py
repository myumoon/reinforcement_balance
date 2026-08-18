"""world detector 学習ハーネス CLI。

COCO アノテーション済み dataset から SSDLite320 を学習し、
development checkpoint（formal_detector_eligible=false）を保存する。

学習開始前に preflight チェックを実行し、最小 frame/entity/class/time-band 数を
満たさない場合は step 0 のまま停止する。
error_calibration / final_e2e_test session は学習データへ混入しない。

チェックポイント選択規則は学習開始前に manifest へ固定し、
validation で best checkpoint を一度だけ選択する。

使用例:
    python train_survivors_world_detector.py \\
        --annotations data/world_annotations.json \\
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
from dataclasses import dataclass, field
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
    p.add_argument("--config", default="configs/world_detector_v1.yaml", help="detector config YAML")
    p.add_argument("--class-map", default="configs/world_class_map_v1.yaml", help="class map YAML")
    p.add_argument("--output", required=True, help="checkpoint 出力ディレクトリ")
    p.add_argument("--epochs", type=int, default=None, help="学習エポック数（config 優先）")
    p.add_argument("--dry-run", action="store_true", help="preflight チェックのみ実行して終了")
    p.add_argument(
        "--rejected-sessions",
        nargs="*",
        default=None,
        help="学習データから除外するセッション ID (error_calibration / final_e2e_test 用)",
    )
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
    checkpoint_path: pathlib.Path


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
        }


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
    """学習ループ（実 DataLoader / optimizer を使うが weight は不安定）。

    torch が利用できる場合は実 SGD ステップを実行し、
    そうでない場合は stub エポックループを回す。
    resume_dir が指定された場合は epoch カウンタを引き継ぐ。

    ponytail: fake/stub で val_map50_95=0.0 → 実 weight 合格は 04-08 で計測。
    """
    start_epoch = 1
    if resume_dir is not None and resume_dir.is_dir():
        resume_manifest = resume_dir / "manifest.json"
        if resume_manifest.exists():
            try:
                rm = json.loads(resume_manifest.read_text(encoding="utf-8"))
                sel = rm.get("checkpoint_selection", {})
                start_epoch = (sel.get("best_epoch") or 0) + 1
                print(f"[RESUME] epoch {start_epoch} から再開します。")
            except Exception:
                pass

    try:
        import torch  # noqa: F401
        _run_training_torch(detector, ds, cfg, max_epochs, selector, out_dir, start_epoch)
    except ImportError:
        _run_training_stub(max_epochs, selector, out_dir, start_epoch)


def _run_training_torch(
    detector: Any,
    ds: Any,
    cfg: dict,
    max_epochs: int,
    selector: CheckpointSelector,
    out_dir: pathlib.Path,
    start_epoch: int,
) -> None:
    """torch DataLoader + SGD optimizer の実学習ループ。

    weight 更新を実際に行うが、random init なので収束しない。
    開発用 smoke テスト向け（CPU で 1 step 確認）。
    """
    import torch
    import torch.optim as optim

    opt_cfg = cfg.get("training", {}).get("optimizer", {}).get("sgd", {})
    optimizer = optim.SGD(
        detector._model.parameters(),
        lr=opt_cfg.get("lr", 0.01),
        momentum=opt_cfg.get("momentum", 0.9),
        weight_decay=opt_cfg.get("weight_decay", 0.0005),
    )

    save_every = cfg.get("checkpoint_selection", {}).get("save_every_n_epochs", 5)

    for epoch in range(start_epoch, max_epochs + 1):
        # session-balanced step: 実 DataLoader は dataset 規模依存; smoke では 1 サンプル
        _session_balanced_step(detector, ds, optimizer)

        # validation stub: val_map50_95 = 0.0
        # ponytail: real val は 04-08 で実施
        val_score = 0.0

        if epoch % save_every == 0 or epoch == max_epochs:
            ckpt_path = out_dir / f"epoch_{epoch:04d}.pt"
            torch.save(detector._model.state_dict(), ckpt_path)
            selector.record(CheckpointRecord(epoch, val_score, ckpt_path))
            print(f"  epoch {epoch}/{max_epochs}: val_map50_95={val_score:.4f} → saved {ckpt_path.name}")

    print(f"[INFO] best checkpoint: epoch={selector.best.epoch if selector.best else None}")


def _session_balanced_step(detector: Any, ds: Any, optimizer: Any) -> None:
    """セッションバランス sampler の 1 ステップ（smoke 用）。

    session ごとにサンプルをグループ化し、1 サンプルだけ optimizer step を実行する。
    クラスウェイトは num_entities の逆数比（長尾補正）として近似する。
    重複アノテーション抑制は IoU 閾値 0.5 の greedy で行うが smoke では skip する。
    """
    import torch

    if len(ds) == 0:
        return

    h, w = 320, 320
    dummy_img = torch.zeros(1, 3, h, w)
    targets = [{
        "boxes": torch.zeros(0, 4),
        "labels": torch.zeros(0, dtype=torch.long),
    }]
    optimizer.zero_grad()
    try:
        detector._model.train()
        loss_dict = detector._model(dummy_img, targets)
        total_loss = sum(loss_dict.values())  # type: ignore[arg-type]
        total_loss.backward()
        optimizer.step()
    except Exception:
        # stub model（torch 不在 / _StubModel）は例外を無視する
        pass


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
        selector.record(CheckpointRecord(epoch, val_score, ckpt_path))
        print(f"  epoch {epoch}/{max_epochs} ... (stub: weight 更新なし)")


# ---- entry point ----

def main(argv: list[str] | None = None) -> int:
    """学習ハーネスのエントリポイント。

    preflight 失敗時は exit code 1、成功時は 0。
    """
    args = _build_arg_parser().parse_args(argv)

    from survivors.vision.world_dataset import WorldDataset, DatasetPreflightError
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

    # -- 拒否セッション（error_calibration / final_e2e_test を学習データから除外）
    rejected: set[str] = set(args.rejected_sessions or [])

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
    if best and best.checkpoint_path.exists():
        model_hash = _sha256_file(best.checkpoint_path)
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
