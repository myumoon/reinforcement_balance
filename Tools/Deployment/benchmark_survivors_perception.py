"""ArtifactStore-backed Survivors perception formal benchmark runner。"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from Tools.Artifacts.artifact_store import ArtifactStore
from reinbalance_survivors_contracts.artifact_dag import validate_artifact_dag
from reinbalance_survivors_contracts.artifact_identity import (
    ArtifactDescriptor,
    ArtifactRef,
    artifact_uri,
)
from reinbalance_survivors_contracts.canonical_json import canonical_json_bytes

from survivors.capture_dataset import (
    DatasetWriter,
    SessionManifest,
    SplitManifest,
    _split_manifest_from_bytes,
)
from survivors.perception_benchmark import (
    ExpectedTick,
    SnapshotReplayTick,
    _formalize_benchmark_report,
    run_benchmark,
)
from survivors.perception_error_fit import (
    CalibrationResidual,
    FittedPerceptionErrorProfile,
    PerceptionFinalVerdict,
    _create_formal_final_verdict,
    _create_formal_lineage_seal,
    _write_formal_calibration_profile,
    _write_formal_final_verdict,
    fit_error_profile,
)
from survivors.perception_session_split import SessionSplit, validate_split


CalibrationProvider = Callable[
    [tuple[SessionManifest, ...], Mapping[str, bytes]], Sequence[CalibrationResidual]
]
FinalReplayProvider = Callable[
    [tuple[SessionManifest, ...], Mapping[str, bytes]], Sequence[SnapshotReplayTick]
]


@dataclass(frozen=True, slots=True)
class FormalBenchmarkRequest:
    store: ArtifactStore
    dependency_refs: Mapping[str, ArtifactRef]
    capture_store_root: Path
    calibration_provider: CalibrationProvider
    final_replay_provider: FinalReplayProvider
    calibration_logical_id: str = "perception/calibration/profile.json"
    verdict_logical_id: str = "perception/final/verdict.json"


@dataclass(frozen=True, slots=True)
class FormalBenchmarkResult:
    split: SessionSplit
    profile: FittedPerceptionErrorProfile
    verdict: PerceptionFinalVerdict
    calibration_ref: ArtifactRef
    verdict_ref: ArtifactRef
    descriptors: tuple[ArtifactDescriptor, ...]


_DEPENDENCY_NAMES = frozenset(
    {
        "capture_dataset", "parser_package", "detector_package",
        "assembler_config", "target_config",
    }
)
# パッケージ種別ごとの required field 集合。JSON 内容を exact schema で検証する。
_PACKAGE_REQUIRED_FIELDS: dict[str, frozenset[str]] = {
    "parser_package": frozenset({"schema_version", "development_only", "formal_eligible", "artifact_hash"}),
    "detector_package": frozenset({"schema_version", "development_only", "formal_eligible", "artifact_hash"}),
    "assembler_config": frozenset({
        "schema_version", "development_only", "formal_eligible",
        "assembler_schema_hash", "ui_presentation_schema_hash",
        "atlas_vocabulary_hash", "assembler_impl_hash", "roi_resolver_input_hash",
    }),
    "target_config": frozenset({"schema_version", "development_only", "formal_eligible", "threshold_hash"}),
}
_CLI_PROVIDER_FACTORY: Callable[[Mapping[str, bytes]], tuple[CalibrationProvider, FinalReplayProvider]] | None = None


def _restore_dependencies(request: FormalBenchmarkRequest) -> dict[str, bytes]:
    """全 dependency を content store から restore → hash/size → JSON 内容検証する。"""
    if not isinstance(request.store, ArtifactStore):
        raise TypeError("formal runner requires ArtifactStore")
    refs = dict(request.dependency_refs)
    if set(refs) != _DEPENDENCY_NAMES:
        raise ValueError("formal dependency ref set is incomplete or contains unknown entries")
    restored: dict[str, bytes] = {}
    for name in sorted(_DEPENDENCY_NAMES):
        ref = refs[name]
        if not isinstance(ref, ArtifactRef):
            raise TypeError(f"{name} must be an ArtifactRef")
        verification = request.store.verify(ref)
        if not verification.ok:
            raise ValueError(f"{name} ArtifactStore verification failed: {verification.reason}")
        data = request.store.object_path(ref.store_uri).read_bytes()
        if len(data) != ref.size_bytes:
            raise ValueError(f"{name} restored size changed after verification")
        # capture split は専用 typed parser で内容検証する。他 package/config は
        # exact JSON mapping と formal marker を検証し、任意ファイルを Verified としない。
        if name != "capture_dataset":
            try:
                payload = json.loads(data)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"{name} is not a valid JSON package") from exc
            if (
                not isinstance(payload, dict)
                or type(payload.get("schema_version")) is not str
                or not payload["schema_version"]
                or payload.get("development_only") is not False
                or payload.get("formal_eligible") is not True
            ):
                raise ValueError(f"{name} package content is not formal eligible")
            # パッケージ種別別の exact required field 検証。
            required_fields = _PACKAGE_REQUIRED_FIELDS[name]
            missing = required_fields - set(payload)
            if missing:
                raise ValueError(f"{name} is missing required fields: {sorted(missing)}")
            # parser/detector パッケージ: artifact_hash が 64 char hex であることを確認する。
            # replay との照合は run_formal_pipeline で行う。
            if name in {"parser_package", "detector_package"}:
                decl_hash = payload.get("artifact_hash")
                if not isinstance(decl_hash, str) or len(decl_hash) != 64 or not all(
                    c in "0123456789abcdef" for c in decl_hash
                ):
                    raise ValueError(f"{name} artifact_hash must be a 64-char lowercase SHA-256 hex string")
        restored[name] = data
        # TOCTOU を閉じるため内容を読んだ後にも object を再検証する。
        if not request.store.verify(ref).ok:
            raise ValueError(f"{name} changed during restore")
    return restored


def _restore_split_and_sessions(
    request: FormalBenchmarkRequest, capture_bytes: bytes
) -> tuple[SplitManifest, dict[str, SessionManifest], SessionSplit]:
    split_ref = request.dependency_refs["capture_dataset"]
    split = _split_manifest_from_bytes(
        request.store.object_path(split_ref.store_uri), capture_bytes
    )
    if split.manifest_sha256 != split_ref.sha256:
        raise ValueError("capture split manifest is not exact-hash bound to ArtifactRef")
    manifests: dict[str, SessionManifest] = {}
    for units in split.splits.values():
        for unit in units:
            if unit.session_id in manifests:
                # typed validate_split also rejects this; stop before a second restore/open.
                raise ValueError(f"duplicate session in split: {unit.session_id}")
            manifests[unit.session_id] = DatasetWriter.restore(
                request.capture_store_root, unit.session_id
            )
    validated = validate_split(split, manifests)
    for session in validated.sessions:
        if session.kind in {"error_calibration", "final_e2e_test"}:
            manifest = manifests[session.session_id]
            if manifest.formal_dataset_eligible is not True:
                raise ValueError(
                    f"session {session.session_id!r} is not formal_dataset_eligible"
                )
    return split, manifests, validated


def _package_payloads(restored: Mapping[str, bytes]) -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    for name in _DEPENDENCY_NAMES - {"capture_dataset"}:
        payload = json.loads(restored[name])
        assert isinstance(payload, dict)
        payloads[name] = payload
    return payloads


def _subject_hashes(
    request: FormalBenchmarkRequest,
    restored_payloads: Mapping[str, Mapping[str, Any]],
    profile: FittedPerceptionErrorProfile,
    *, lineage_seal_hash: str,
) -> dict[str, str]:
    assembler = restored_payloads["assembler_config"]
    target = restored_payloads["target_config"]
    values = {
        "parser_artifact_hash": request.dependency_refs["parser_package"].sha256,
        "detector_artifact_hash": request.dependency_refs["detector_package"].sha256,
        "assembler_schema_hash": assembler.get("assembler_schema_hash"),
        "ui_presentation_schema_hash": assembler.get("ui_presentation_schema_hash"),
        "config_hash": request.dependency_refs["target_config"].sha256,
        "capture_dataset_hash": request.dependency_refs["capture_dataset"].sha256,
        "calibration_profile_hash": profile.artifact_hash,
        "threshold_hash": target.get("threshold_hash"),
        "atlas_vocabulary_hash": assembler.get("atlas_vocabulary_hash"),
        "assembler_impl_hash": assembler.get("assembler_impl_hash"),
        "roi_resolver_input_hash": assembler.get("roi_resolver_input_hash"),
        "benchmark_fit_code_hash": profile.fit_code_hash,
        "lineage_seal_hash": lineage_seal_hash,
    }
    for name, value in values.items():
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"formal subject {name} is not a SHA-256")
    return values  # type: ignore[return-value]


def _predicted_ref(logical_id: str, data: bytes) -> ArtifactRef:
    import hashlib

    digest = hashlib.sha256(data).hexdigest()
    return ArtifactRef(
        logical_id=logical_id, sha256=digest, size_bytes=len(data),
        media_type="application/json", store_uri=artifact_uri(digest),
    )


def _descriptor_chain(
    request: FormalBenchmarkRequest,
    profile: FittedPerceptionErrorProfile,
    verdict: PerceptionFinalVerdict,
) -> tuple[ArtifactDescriptor, ...]:
    source = ArtifactDescriptor(
        logical_id="perception/capture/source",
        node_kind="source_descriptor",
        producer_id="benchmark_survivors_perception",
        producer_version="v2",
        identity_metadata={"split_manifest_hash": request.dependency_refs["capture_dataset"].sha256},
        files=(request.dependency_refs["capture_dataset"],),
    )
    profile_bytes = canonical_json_bytes(profile.to_artifact_wire())
    profile_node = ArtifactDescriptor(
        logical_id=request.calibration_logical_id,
        node_kind="perception_calibration_profile",
        producer_id="perception_error_fit",
        producer_version="v2",
        identity_metadata={
            "profile_hash": profile.profile_hash,
            "fit_code_hash": profile.fit_code_hash,
        },
        parents=(source.node_ref(),),
        files=(_predicted_ref(request.calibration_logical_id, profile_bytes),),
    )
    verdict_bytes = canonical_json_bytes(verdict.to_wire())
    final_node = ArtifactDescriptor(
        logical_id=request.verdict_logical_id,
        node_kind="perception_final_verdict",
        producer_id="benchmark_survivors_perception",
        producer_version="v2",
        identity_metadata={"verdict_id": verdict.verdict_id, "seal_id": verdict.seal_id},
        parents=(profile_node.node_ref(),),
        files=(_predicted_ref(request.verdict_logical_id, verdict_bytes),),
    )
    descriptors = (source, profile_node, final_node)
    validate_artifact_dag(descriptors)
    return descriptors


def run_formal_pipeline(request: FormalBenchmarkRequest) -> FormalBenchmarkResult:
    """formal runner の実経路。依存検証失敗時は ArtifactStore へ何も書かない。"""
    # Phase A は read-only。ここで失敗しても seal/profile/verdict は一切 publish されない。
    restored = _restore_dependencies(request)
    _, manifests, validated_split = _restore_split_and_sessions(
        request, restored["capture_dataset"]
    )
    payloads = _package_payloads(restored)
    calibration_manifests = tuple(
        manifests[session.session_id]
        for session in validated_split.sessions
        if session.kind == "error_calibration"
    )
    final_manifests = tuple(
        manifests[session.session_id]
        for session in validated_split.sessions
        if session.kind == "final_e2e_test"
    )
    residuals = list(request.calibration_provider(calibration_manifests, restored))
    calibration_hashes = {
        manifest.session_id: manifest.manifest_sha256
        for manifest in calibration_manifests
    }
    final_hashes = {
        manifest.session_id: manifest.manifest_sha256 for manifest in final_manifests
    }
    profile = fit_error_profile(
        residuals,
        [manifest.session_id for manifest in calibration_manifests],
        [manifest.session_id for manifest in final_manifests],
        calibration_session_hashes=calibration_hashes,
    )

    # seal identity の lineage_seal_hash 自身は含めず、seal wire hash を verdict subject にする。
    preseal_subjects = _subject_hashes(
        request, payloads, profile, lineage_seal_hash="0" * 64
    )
    seal_subjects = {
        name: value for name, value in preseal_subjects.items()
        if name != "lineage_seal_hash"
    }
    # まず pure staged seal を構築する。replay/gate/DAG が失敗しても store は未変更。
    seal = _create_formal_lineage_seal(
        final_session_hashes=final_hashes,
        store=request.store,
        _publish=False,
        **seal_subjects,
    )
    lineage_seal_hash = __import__("hashlib").sha256(
        canonical_json_bytes(seal.to_wire())
    ).hexdigest()
    subjects = _subject_hashes(
        request, payloads, profile, lineage_seal_hash=lineage_seal_hash
    )

    replay_ticks = list(request.final_replay_provider(final_manifests, restored))
    expected_ticks = [
        # 04-09 assembler の canonical frame identity は session/frame/world index。
        ExpectedTick(session.session_id, f"{session.session_id}:{frame_id}:{frame_id}")
        for session in validated_split.sessions
        if session.kind == "final_e2e_test"
        for frame_id in session.expected_frame_ids
    ]
    # formal runner: typed SnapshotReplayTick のみ受理する。
    for tick in replay_ticks:
        if not isinstance(tick, SnapshotReplayTick):
            raise TypeError(
                f"formal replay provider must yield SnapshotReplayTick, got {type(tick).__name__}"
            )
    # session_kind / source_policy / raw fields を validated split の session record と照合する。
    final_session_records = {
        s.session_id: s for s in validated_split.sessions if s.kind == "final_e2e_test"
    }
    for tick in replay_ticks:
        record = final_session_records.get(tick.session_id)
        if record is None:
            raise ValueError(
                f"tick session {tick.session_id!r} is not in the validated final sessions"
            )
        if tick.session_kind != record.kind:
            raise ValueError(
                f"tick {tick.session_id}/{tick.frame_id}: session_kind {tick.session_kind!r} "
                f"does not match split record kind {record.kind!r}"
            )
        if tick.source_policy != record.source_policy:
            raise ValueError(
                f"tick {tick.session_id}/{tick.frame_id}: source_policy {tick.source_policy!r} "
                f"does not match split record source_policy {record.source_policy!r}"
            )
        # formal replay では raw_card_ids / raw_inventory が必須（hash 独立検証のため）。
        for label, snapshot in (("ground_truth", tick.ground_truth), ("predicted", tick.predicted)):
            if snapshot is None:
                continue
            ui = snapshot.ui_presentation
            if ui.raw_card_ids is None or ui.raw_inventory is None:
                raise ValueError(
                    f"tick {tick.session_id}/{tick.frame_id} {label}: "
                    "raw_card_ids and raw_inventory are required in formal replay"
                )
    # replay の parser_artifact_hash / detector_artifact_hash が登録済みパッケージ宣言と一致することを確認する。
    parser_decl_hash = json.loads(restored["parser_package"])["artifact_hash"]
    detector_decl_hash = json.loads(restored["detector_package"])["artifact_hash"]
    for tick in replay_ticks:
        for label, snapshot in (("ground_truth", tick.ground_truth), ("predicted", tick.predicted)):
            if snapshot is None:
                continue
            actual_parser = snapshot.ui_presentation.parser_artifact_hash
            if actual_parser != parser_decl_hash:
                raise ValueError(
                    f"tick {tick.session_id}/{tick.frame_id} {label}: "
                    f"parser_artifact_hash ({actual_parser[:8]}…) does not match "
                    f"parser_package artifact_hash ({parser_decl_hash[:8]}…)"
                )
            # detector_artifact_hash はスナップショット構造内（SnapshotReplayTick の自己申告ではない）で確認する。
            # ponytail: detector 観測内容の真正性はこのチェックでは保証できない。provider が宣言 hash を
            # コピーして任意の deploy_obs を注入することは防げない。完全な provenance 検証には runner 側で
            # restored detector package を実行し観測を生成する実装（runner-side execution）が必要。
            # 今 PR は code-only（正式実行なし）のため、hash の構造的検証のみとし実行は follow-up とする。
            actual_detector = snapshot.detector_artifact_hash
            if not actual_detector:
                raise ValueError(
                    f"tick {tick.session_id}/{tick.frame_id} {label}: "
                    "detector_artifact_hash is empty; formal replay snapshots must carry detector identity"
                )
            if actual_detector != detector_decl_hash:
                raise ValueError(
                    f"tick {tick.session_id}/{tick.frame_id} {label}: "
                    f"snapshot.detector_artifact_hash ({actual_detector[:8]}…) does not match "
                    f"detector_package artifact_hash ({detector_decl_hash[:8]}…)"
                )
    report = _formalize_benchmark_report(
        run_benchmark(replay_ticks, expected_ticks=expected_ticks)
    )
    verdict = _create_formal_final_verdict(
        report, seal_id=seal.seal_id,
        final_session_ids=[manifest.session_id for manifest in final_manifests],
        **subjects,
    )
    descriptors = _descriptor_chain(request, profile, verdict)

    # Phase B: stage → batch commit（single atomic）→ seal → canonical logical_id の順序で publish する。
    # ・batch commit が唯一の commit 点（単一 put_bytes）：これが存在すれば cal + verdict が確定。
    # ・staging 後に失敗 → batch commit 未発行 → canonical logical_id 不可視（clean state）。
    # ・batch commit 後に失敗 → 再実行は全操作が idempotent なので安全に回復できる。

    # 1) content を staging logical_id（content hash ベース・canonical 名前空間外）へ書く。
    staged_cal_ref = _write_formal_calibration_profile(
        profile, store=request.store,
        logical_id=f"perception/staging/calibration/{descriptors[1].files[0].sha256}",
    )
    staged_ver_ref = _write_formal_final_verdict(
        verdict, store=request.store,
        logical_id=f"perception/staging/verdict/{descriptors[2].files[0].sha256}",
    )
    if (
        staged_cal_ref.sha256 != descriptors[1].files[0].sha256
        or staged_ver_ref.sha256 != descriptors[2].files[0].sha256
    ):
        raise ValueError("staged refs sha256 differ from prevalidated descriptor refs")
    validate_artifact_dag(descriptors)

    # 2) batch commit manifest を単一 put_bytes で発行する（= commit 点）。
    #    calibration と verdict の sha256 を同時に可視化する publication manifest index。
    #    同一 lineage の再実行は同一 sha256 → idempotent。
    batch_commit_payload = canonical_json_bytes({
        "schema_version": "perception_formal_batch_commit.v1",
        "seal_id": seal.seal_id,
        "calibration_artifact_sha256": staged_cal_ref.sha256,
        "verdict_artifact_sha256": staged_ver_ref.sha256,
    })
    request.store.put_bytes(
        logical_id=f"perception/batch_commit/{seal.seal_id}",
        data=batch_commit_payload,
        media_type="application/json",
    )

    # 3) 以降はすべて idempotent（batch commit 済みのため re-run 安全）。
    persisted_seal = _create_formal_lineage_seal(
        final_session_hashes=final_hashes,
        store=request.store,
        **seal_subjects,
    )
    if persisted_seal.seal_id != seal.seal_id:
        raise ValueError("persisted lineage seal identity changed after validation")
    calibration_ref = _write_formal_calibration_profile(
        profile, store=request.store, logical_id=request.calibration_logical_id
    )
    verdict_ref = _write_formal_final_verdict(
        verdict, store=request.store, logical_id=request.verdict_logical_id
    )
    for manifest in final_manifests:
        persisted_seal.open_session(
            manifest.session_id, manifest.session_path / "session_manifest.json"
        )
    return FormalBenchmarkResult(
        validated_split, profile, verdict, calibration_ref, verdict_ref, descriptors
    )


def _run_dry() -> int:
    """公開 synthetic 入口は常に development-only。"""
    from survivors.perception_benchmark import BenchmarkRecord

    records = []
    for session_id in ("synthetic-a", "synthetic-b"):
        for frame_id in ("0", "1"):
            records.append(
                BenchmarkRecord(
                    frame_id, session_id, "error_calibration", "raw",
                    "screen_state", "gameplay", "gameplay", 1.0, 1.0,
                )
            )
    report = run_benchmark(records)
    print(json.dumps({
        "development_only": report.development_only,
        "formal_perception_verdict_eligible": report.formal_perception_verdict_eligible,
        "passed": report.passed,
        "blocking_reasons": report.blocking_reasons,
    }, indent=2))
    return 0


def _load_ref(path_str: str) -> ArtifactRef:
    path = Path(path_str).expanduser().resolve()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid ArtifactRef file: {path}") from exc
    return ArtifactRef.from_wire(data)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Survivors perception formal benchmark")
    for option in (
        "capture-dataset", "parser-package", "detector-package",
        "assembler-config", "target-config",
    ):
        parser.add_argument(f"--{option}", help=f"{option} ArtifactRef JSON")
    parser.add_argument("--artifact-store", help="ArtifactStore root")
    parser.add_argument("--capture-store", help="restorable capture session store root")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.dry_run:
        return _run_dry()
    required = {
        "capture_dataset": args.capture_dataset,
        "parser_package": args.parser_package,
        "detector_package": args.detector_package,
        "assembler_config": args.assembler_config,
        "target_config": args.target_config,
        "artifact_store": args.artifact_store,
        "capture_store": args.capture_store,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        print(f"BLOCKED: missing formal inputs: {', '.join(sorted(missing))}", file=sys.stderr)
        return 2
    try:
        refs = {name: _load_ref(required[name]) for name in _DEPENDENCY_NAMES}
        store = ArtifactStore(required["artifact_store"])
        # CLI replay engines are supplied by the integrated parser/detector package adapter.
        # Unit tests inject only this adapter; the pipeline below remains identical.
        restored_preview = _restore_dependencies(
            FormalBenchmarkRequest(
                store=store, dependency_refs=refs,
                capture_store_root=Path(required["capture_store"]),
                calibration_provider=lambda *_: (), final_replay_provider=lambda *_: (),
            )
        )
        if _CLI_PROVIDER_FACTORY is None:
            raise ValueError("formal parser/detector replay adapter is not installed")
        calibration_provider, final_provider = _CLI_PROVIDER_FACTORY(restored_preview)
        result = run_formal_pipeline(
            FormalBenchmarkRequest(
                store=store, dependency_refs=refs,
                capture_store_root=Path(required["capture_store"]),
                calibration_provider=calibration_provider,
                final_replay_provider=final_provider,
            )
        )
    except Exception as exc:  # fail closed: dependency failure never prints Verified
        print(f"BLOCKED: formal benchmark failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "verdict_id": result.verdict.verdict_id,
        "passed": result.verdict.passed,
        "verdict_artifact": result.verdict_ref.store_uri,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
