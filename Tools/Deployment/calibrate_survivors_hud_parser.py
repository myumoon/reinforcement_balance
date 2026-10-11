"""Survivors HUD parser の development calibration package 発行 CLI。

JSON の train/validation 注釈を読み、split 分離された fit と11 gate report を作って
内容 hash 付き package を atomic publish します。formal 発行は04-05まで停止します。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml
from reinbalance_survivors_contracts.artifact_identity import ArtifactDescriptor
from reinbalance_survivors_contracts.artifact_store import ArtifactStore
from reinbalance_survivors_contracts.canonical_json import canonical_json_bytes
from survivors.artifact_restore_test import run_restore_test
from survivors.capture_dataset import DatasetWriter
from survivors.vision.hud_calibration import CalibrationConfig, PackageWriter, fit, validate
from survivors.vision.hud_formal_release import (
    FormalGateError,
    FormalPreflightError,
    build_formal_atlas,
    emit_resolved_layout_yaml,
    predict_validation_records,
    publish_formal_parser_artifact,
    run_formal_preflight,
    validate_formal,
)
from survivors.vision.hud_truth import read_hud_truth
from survivors.vision.icon_matcher import IconMatcher, serialize_manifest

def _build_arg_parser() -> argparse.ArgumentParser:
    """calibration CLI の引数 parser を構築する。

train と validation の入力を別 option にして、operator が同じファイルを誤用しにくくします。
    """
    parser = argparse.ArgumentParser(description="Survivors HUD parser calibration package を発行します")
    parser.add_argument("--train-annotations", type=Path, help="model_train 注釈 JSON")
    parser.add_argument("--validation-annotations", type=Path, help="model_validation 注釈 JSON")
    parser.add_argument("--output-package", type=Path, help="development package 出力先")
    parser.add_argument("--formal", action="store_true", help="formal release を実行")
    parser.add_argument("--formal-dataset-eligible", action="store_true", help="正式 dataset の operator 証明")
    parser.add_argument("--capture-store", type=Path)
    parser.add_argument("--dataset-descriptor", type=Path)
    parser.add_argument("--restore-verdict", type=Path)
    parser.add_argument("--target-profile-hash")
    parser.add_argument("--vocabulary", type=Path)
    parser.add_argument("--minimum-counts", type=Path)
    parser.add_argument("--artifact-store", type=Path)
    parser.add_argument("--backup-root", type=Path)
    parser.add_argument("--restore-root", type=Path)
    parser.add_argument("--logical-prefix")
    parser.add_argument("--out-dir", type=Path)
    return parser

def _load_annotations(path: Path) -> list[dict[str, object]]:
    """JSON 配列から annotation mapping を読み込む。

top-level と各要素の型を検査し、不完全な入力を model 層へ渡す前に明示的に拒否します。
    """
    try:
        wire = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"注釈 JSON を読めません: {path}") from exc
    if not isinstance(wire, list) or not all(isinstance(item, dict) for item in wire):
        raise ValueError("注釈 JSON は object の配列である必要があります")
    return wire

def main(argv: list[str] | None = None) -> int:
    """注釈を calibration して parser package を発行する。

gate 合格時のみ atomic publish し、gate 失敗は exit code 2、入力エラーは exit code 1 を返します。
formal 指定時は descriptor-bound truth から atlas と formal package を発行します。
    """
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    if args.formal:
        formal_names = (
            "capture_store", "dataset_descriptor", "restore_verdict", "target_profile_hash",
            "vocabulary", "minimum_counts", "artifact_store", "backup_root", "restore_root",
            "logical_prefix", "out_dir",
        )
        missing = [name.replace("_", "-") for name in formal_names if getattr(args, name) is None]
        if missing:
            parser.error("--formal requires " + ", ".join(f"--{name}" for name in missing))
        if args.train_annotations is not None or args.validation_annotations is not None or args.output_package is not None:
            parser.error("--formal cannot be combined with development annotation/package arguments")
        try:
            return _run_formal(args)
        except FormalGateError as exc:
            print(f"[GATE] {exc}", file=sys.stderr)
            return 2
        except (OSError, TypeError, ValueError, KeyError, yaml.YAMLError) as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 1
    if args.train_annotations is None or args.validation_annotations is None or args.output_package is None:
        parser.error("development mode requires --train-annotations, --validation-annotations, and --output-package")
    try:
        train_result = fit(
            _load_annotations(args.train_annotations), formal_mode=args.formal,
            formal_dataset_eligible=args.formal_dataset_eligible,
        )
        report = validate(
            _load_annotations(args.validation_annotations), formal_mode=args.formal,
            formal_dataset_eligible=args.formal_dataset_eligible,
        )
        if not report.all_passed:
            print(json.dumps({"all_gates_passed": False, "gates": report.to_wire()["gates"]}), file=sys.stderr)
            return 2
        meta = PackageWriter.publish_development(args.output_package, train_result, report)
    except (OSError, TypeError, ValueError, NotImplementedError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"package_hash": meta.package_hash, "all_gates_passed": True}))
    return 0


def _run_formal(args: argparse.Namespace) -> int:
    """descriptor bound HUD truth から formal package と restore verdict を発行する。

    入力検証、fit、validation、publish を同一 CLI 実行で連結する。
    """
    capture_root = args.capture_store.expanduser().resolve()
    dataset = ArtifactDescriptor.from_wire(json.loads(args.dataset_descriptor.read_text(encoding="utf-8")))
    restore = ArtifactDescriptor.from_wire(json.loads(args.restore_verdict.read_text(encoding="utf-8")))
    minimum_counts = yaml.safe_load(args.minimum_counts.read_text(encoding="utf-8"))
    preflight = run_formal_preflight(
        capture_store_root=capture_root,
        dataset_descriptor=dataset,
        restore_verdict_descriptor=restore,
        target_profile_hash=args.target_profile_hash,
        minimum_counts=minimum_counts,
    )
    train_frames = {
        session_id: DatasetWriter.restore(capture_root, session_id, metadata_only=True)
        for session_id in preflight.train_sessions
    }
    validation_frames = {
        session_id: DatasetWriter.restore(capture_root, session_id, metadata_only=True)
        for session_id in preflight.validation_sessions
    }
    train_truth = [
        record
        for session_id in preflight.train_sessions
        for record in read_hud_truth(capture_root / "capture_sessions" / session_id)
        if record.confirmed is True
    ]
    validation_truth = [
        record
        for session_id in preflight.validation_sessions
        for record in read_hud_truth(capture_root / "capture_sessions" / session_id)
        if record.confirmed is True
    ]
    train_annotations = [
        {
            "sample_id": f"{record.session_id}:{record.frame_id}",
            "split": "model_train",
            "roi_name": record.roi_name,
            "expected_roi": record.expected_roi,
        }
        for record in train_truth
    ]
    fit_result = fit(train_annotations, formal_mode=True, formal_dataset_eligible=True)
    output_dir = args.out_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    atlas_path = output_dir / "formal_icon_atlas.json"
    atlas, provenance = build_formal_atlas(
        train_frames_by_session=train_frames,
        train_truth=train_truth,
        vocabulary_path=args.vocabulary,
        profile_hash=preflight.target_profile_hash,
        build_hash=preflight.game_build_id,
        output_path=atlas_path,
    )
    provenance_path = output_dir / "formal_icon_atlas_provenance.json"
    provenance_path.write_bytes(canonical_json_bytes(provenance))
    matcher = IconMatcher(atlas)
    predicted = predict_validation_records(
        validation_frames_by_session=validation_frames,
        validation_truth=validation_truth,
        atlas=matcher,
    )
    formal_report = validate_formal(
        predicted,
        config=CalibrationConfig(),
        preflight=preflight,
        atlas_content_hash=atlas.atlas_content_hash,
        fit_canonical_hash=fit_result.fit_canonical_hash,
    )
    if not formal_report.all_passed:
        print(json.dumps({
            "all_gates_passed": False,
            "gates": [gate.to_wire() for gate in formal_report.pooled_report.gates],
            "ci_passed": dict(formal_report.ci_passed),
        }), file=sys.stderr)
        return 2
    package_path = output_dir / "hud_parser_package.formal.json"
    PackageWriter.publish_formal(
        package_path,
        fit_result,
        formal_report,
        preflight=preflight,
        atlas_manifest=atlas,
    )
    descriptor = publish_formal_parser_artifact(
        ArtifactStore(args.artifact_store),
        package_path=package_path,
        atlas_manifest_path=atlas_path,
        atlas_provenance_path=provenance_path,
        preflight=preflight,
        formal_report=formal_report,
        dataset_descriptor=dataset,
        logical_prefix=args.logical_prefix,
    )
    verdict, verdict_descriptor = run_restore_test(
        ArtifactStore(args.artifact_store),
        descriptor,
        backup_root=args.backup_root,
        restore_root=args.restore_root,
        verdict_logical_id=f"{args.logical_prefix}/{descriptor.identity_metadata['package_hash'][:16]}/restore-verdict",
    )
    if verdict.passed is not True:
        raise FormalGateError("formal artifact restore test failed: " + "; ".join(verdict.blocking_reasons))
    resolved_path = emit_resolved_layout_yaml(
        fit_result,
        formal_report,
        preflight,
        descriptor,
        verdict_descriptor,
        output_dir / "hud_layout_1080p_ja_v1.yaml",
    )
    print(json.dumps({
        "descriptor_logical_id": descriptor.logical_id,
        "restore_verdict_logical_id": verdict_descriptor.logical_id,
        "package_hash": descriptor.identity_metadata["package_hash"],
        "fit_canonical_hash": fit_result.fit_canonical_hash,
        "formal_report_canonical_hash": formal_report.formal_report_canonical_hash,
        "preflight_canonical_hash": preflight.preflight_canonical_hash,
        "atlas_content_hash": atlas.atlas_content_hash,
        "resolved_yaml_path": str(resolved_path),
    }))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
