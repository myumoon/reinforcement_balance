#!/usr/bin/env python3
"""frozen capture split を ArtifactStore descriptor と restore verdict にする。

formal publisher を一度実行し、共有 restore helper の結果を表示する。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from reinbalance_survivors_contracts.artifact_store import ArtifactStore
from survivors.artifact_restore_test import run_restore_test
from survivors.capture_formal import ModelExportPaths, publish_capture_dataset


def _parser() -> argparse.ArgumentParser:
    """dataset publisher が必要とする store と任意の世代入力を宣言する。

    model export pair は両方同時に指定し、sealed inputs は session path 規約から読む。
    """
    parser = argparse.ArgumentParser(description="Publish a formal Survivors capture dataset")
    parser.add_argument("--store-root", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--artifact-store", required=True, type=Path)
    parser.add_argument("--backup-root", required=True, type=Path)
    parser.add_argument("--restore-root", required=True, type=Path)
    parser.add_argument("--logical-prefix", required=True)
    parser.add_argument("--evaluation-windows", type=Path)
    parser.add_argument("--allow-missing-formal-annotations", action="store_true")
    parser.add_argument("--world-coco-model", type=Path)
    parser.add_argument("--ui-coco-model", type=Path)
    parser.add_argument("--agreement", type=Path)
    parser.add_argument("--formal-gates", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    """descriptor を発行して全 object の restore test を記録する。

    restore verdict が fail なら結果 JSON を残して failure exit code を返す。
    """
    args = _parser().parse_args(argv)
    try:
        if (args.world_coco_model is None) != (args.ui_coco_model is None):
            raise ValueError("--world-coco-model and --ui-coco-model must be specified together")
        model_exports = (
            ModelExportPaths(args.world_coco_model, args.ui_coco_model)
            if args.world_coco_model is not None else None
        )
        store = ArtifactStore(args.artifact_store)
        descriptor = publish_capture_dataset(
            store,
            store_root=args.store_root,
            split_manifest_path=args.split_manifest,
            logical_prefix=args.logical_prefix,
            evaluation_windows_path=args.evaluation_windows,
            model_exports=model_exports,
            agreement_path=args.agreement,
            formal_gates_path=args.formal_gates,
            allow_missing_formal_annotations=args.allow_missing_formal_annotations,
        )
        generation_root = descriptor.logical_id.rsplit("/", 1)[0]
        verdict_id = f"{generation_root}/restore_test_verdict.json"
        verdict, verdict_descriptor = run_restore_test(
            store,
            descriptor,
            backup_root=args.backup_root,
            restore_root=args.restore_root,
            verdict_logical_id=verdict_id,
        )
        print(json.dumps({
            "descriptor_logical_id": descriptor.logical_id,
            "restore_verdict_logical_id": verdict_descriptor.logical_id,
            "dataset_identity_hash": descriptor.identity_metadata["dataset_identity_hash"],
            "passed": verdict.passed,
        }))
        return 0 if verdict.passed else 1
    except (OSError, ValueError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
