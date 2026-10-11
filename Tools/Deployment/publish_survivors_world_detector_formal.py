"""world detector formal package と artifact release を実行する CLI。

preflight、verdict、checkpoint の identity が一致した release だけを保存する。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from reinbalance_survivors_contracts.artifact_identity import ArtifactDescriptor
from reinbalance_survivors_contracts.artifact_store import ArtifactStore
from survivors.artifact_restore_test import run_restore_test
from survivors.vision.world_detector_package import publish_formal_package
from survivors.vision.world_formal_release import (
    ValidationVerdict,
    emit_resolved_detector_yaml,
    publish_formal_detector_artifact,
    read_preflight_report,
)


def _build_arg_parser() -> argparse.ArgumentParser:
    """formal publisher の必須 file と store root を定義する。

    build attestation は game_build_id が SHA-256 のときだけ必要になる。
    """
    configs = Path(__file__).parent / "configs"
    parser = argparse.ArgumentParser(description="formal world detector package publisher")
    parser.add_argument("--verdict", required=True)
    parser.add_argument("--preflight-report", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--artifact-store", required=True)
    parser.add_argument("--backup-root", required=True)
    parser.add_argument("--restore-root", required=True)
    parser.add_argument("--dataset-descriptor", required=True)
    parser.add_argument("--logical-prefix", required=True)
    parser.add_argument("--resolved-yaml-out", required=True)
    parser.add_argument("--config", default=str(configs / "world_detector_v2.yaml"))
    parser.add_argument("--class-map", default=str(configs / "world_class_map_v2.yaml"))
    parser.add_argument("--build-attestation")
    return parser


def main(argv: list[str] | None = None) -> int:
    """formal package、envelope、restore verdict と resolved YAML を順に生成する。

    restore が full verify で通らない場合は resolved config を発行しない。
    """
    args = _build_arg_parser().parse_args(argv)
    try:
        checkpoint = Path(args.checkpoint_dir)
        run_manifest = json.loads((checkpoint / "run_manifest.json").read_text(encoding="utf-8"))
        tracker = run_manifest.get("resolved_config", {}).get("tracker")
        if not isinstance(tracker, dict):
            raise ValueError("checkpoint run manifest に resolved tracker config がありません")
        verdict = ValidationVerdict.from_dict(json.loads(Path(args.verdict).read_text(encoding="utf-8")))
        preflight = read_preflight_report(args.preflight_report)
        dataset_descriptor = ArtifactDescriptor.from_wire(
            json.loads(Path(args.dataset_descriptor).read_text(encoding="utf-8"))
        )
        package_manifest = publish_formal_package(
            verdict_path=Path(args.verdict),
            checkpoint_dir=checkpoint,
            out_root=Path(args.out_root),
            class_map_path=Path(args.class_map),
            config_path=Path(args.config),
            tracker_config=tracker,
        )
        store = ArtifactStore(args.artifact_store)
        descriptor = publish_formal_detector_artifact(
            store,
            package_dir=package_manifest.parent,
            preflight=preflight,
            verdict=verdict,
            dataset_descriptor=dataset_descriptor,
            logical_prefix=args.logical_prefix,
            build_attestation_path=args.build_attestation,
        )
        _, restore_descriptor = run_restore_test(
            store,
            descriptor,
            backup_root=args.backup_root,
            restore_root=args.restore_root,
            verdict_logical_id=f"{args.logical_prefix.rstrip('/')}/{descriptor.identity_hash[:16]}/restore-verdict",
        )
        resolved_path = emit_resolved_detector_yaml(
            verdict, descriptor, restore_descriptor, preflight, args.resolved_yaml_out,
        )
        print(json.dumps({
            "descriptor_logical_id": descriptor.logical_id,
            "restore_verdict_logical_id": restore_descriptor.logical_id,
            "resolved_yaml": str(resolved_path),
        }, ensure_ascii=False))
        return 0
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        print(f"[FORMAL PUBLISH FAIL] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
