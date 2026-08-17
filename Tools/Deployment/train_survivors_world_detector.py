"""world detector 学習ハーネス CLI。

COCO アノテーション済み dataset から SSDLite320 を学習し、
development checkpoint（formal_detector_eligible=false）を保存する。

学習開始前に preflight チェックを実行し、最小 frame/entity/class/time-band 数を
満たさない場合は step 0 のまま停止する。

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
from typing import Any

import yaml


def _sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


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
    return p


def main(argv: list[str] | None = None) -> int:
    """学習ハーネスのエントリポイント。

    preflight 失敗時は exit code 1、成功時は 0。
    """
    args = _build_arg_parser().parse_args(argv)

    # -- import here to keep CI fast even without torch
    from survivors.vision.world_dataset import WorldDataset, load_class_map, DatasetPreflightError
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

    # -- preflight
    preflight_cfg = cfg.get("training", {}).get("preflight", {})
    try:
        ds = WorldDataset(ann_path, cm_path)
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

    # -- train loop (minimal harness; 04-07 が本格実装)
    print(f"[INFO] アーキテクチャ: {cfg['model']['architecture']}, num_classes={detector.num_classes}")
    max_epochs = args.epochs or cfg.get("training", {}).get("max_epochs", 1)
    print(f"[INFO] max_epochs={max_epochs} で学習を開始します...")

    _run_training(detector, ds, cfg, max_epochs=max_epochs)

    # -- checkpoint manifest
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = CheckpointManifest(
        model_hash=_sha256_file(cfg_path),  # weight なしなので config hash で代用
        data_hash=_sha256_file(ann_path),
        config_hash=_sha256_file(cfg_path),
        build_hash=_sha256_file(cm_path),
        class_map_hash=_sha256_file(cm_path),
        formal_detector_eligible=False,  # development のみ
    )
    manifest.save(out_dir / "manifest.json")
    print(f"[INFO] manifest を {out_dir / 'manifest.json'} へ保存しました。")
    print("[INFO] formal_detector_eligible=false — このままでは正式パッケージへ昇格できません。")
    return 0


def _run_training(detector: Any, ds: Any, cfg: dict, max_epochs: int) -> None:
    """学習ループのスタブ。04-07 が本格 DataLoader / optimizer を実装する。

    実際の weight 更新は行わず、エポック数だけログを出す。
    """
    for epoch in range(1, max_epochs + 1):
        print(f"  epoch {epoch}/{max_epochs} ... (stub: weight 更新なし)")


if __name__ == "__main__":
    sys.exit(main())
