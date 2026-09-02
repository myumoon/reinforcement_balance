"""ArtifactStore-backed Survivors perception formal benchmark runner。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol

from Tools.Artifacts.artifact_store import ArtifactStore
from reinbalance_survivors_contracts.artifact_dag import validate_artifact_dag
from reinbalance_survivors_contracts.artifact_identity import (
    ArtifactDescriptor,
    ArtifactRef,
    artifact_uri,
)
from reinbalance_survivors_contracts.canonical_json import canonical_json_bytes

from survivors.capture.captured_frame import CapturedFrame
from survivors.perception_snapshot import FormalReplayEvidence
from survivors.capture_dataset import (
    DatasetWriter,
    SessionManifest,
    SplitManifest,
    _split_manifest_from_bytes,
)
from survivors.perception_snapshot import PerceptionSnapshot
from survivors.perception_benchmark import (
    ExpectedTick,
    SnapshotReplayTick,
    _formalize_benchmark_report,
    run_benchmark,
)
from survivors.perception_error_fit import (
    _HASH_FIELDS,
    CalibrationResidual,
    FittedPerceptionErrorProfile,
    PerceptionFinalVerdict,
    _create_formal_final_verdict,
    _create_formal_lineage_seal,
    _reserve_final_session,
    fit_error_profile,
    load_final_verdict,
)
from survivors.perception_session_split import SessionSplit, validate_split


@dataclass(frozen=True, slots=True)
class RestoredFormalArtifact:
    """descriptor/verdict と全 content file を再検証済みの formal dependency。"""

    descriptor: ArtifactDescriptor
    restore_verdict: ArtifactDescriptor
    files: Mapping[str, bytes]
    manifest: Mapping[str, Any] | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "files", MappingProxyType(dict(self.files)))
        if self.manifest is not None:
            object.__setattr__(self, "manifest", MappingProxyType(dict(self.manifest)))


class FormalPredictor(Protocol):
    """verified package だけを入力にし、prediction のみを返す formal 境界。"""

    def predict_frame(
        self, frame: CapturedFrame, session_id: str, frame_index: int
    ) -> tuple[PerceptionSnapshot | None, float]:
        """自身の package に束縛した prediction と latency だけを返す。"""
        ...


class FormalGroundTruthLoader(Protocol):
    """runner-owned annotation loader。provider から分離した真正性境界。"""

    def load_frame(
        self, frame: CapturedFrame, session_id: str, frame_index: int
    ) -> tuple[PerceptionSnapshot | None, FormalReplayEvidence | None]:
        ...


FormalPredictorFactory = Callable[
    [Mapping[str, RestoredFormalArtifact]], FormalPredictor
]
FormalGroundTruthFactory = Callable[
    [Path, Mapping[str, RestoredFormalArtifact]], FormalGroundTruthLoader
]


@dataclass(frozen=True, slots=True)
class FormalBenchmarkRequest:
    """formal benchmark 実行リクエスト。

    provider/assembler はここに持たない。run_formal_pipeline が _CLI_PROVIDER_FACTORY
    から hash-bound entrypoint をロードする。callback 注入は development-only API に限定する。
    """

    store: ArtifactStore
    capture_store_root: Path
    dependency_descriptors: Mapping[str, ArtifactDescriptor] = field(default_factory=dict)
    restore_verdicts: Mapping[str, ArtifactDescriptor] = field(default_factory=dict)
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
    "parser_package": frozenset({
        "schema_version", "development_only", "formal_eligible", "files",
        "parser_artifact_hash",
    }),
    "detector_package": frozenset({
        "schema_version", "development_only", "formal_eligible", "files",
        "detector_artifact_hash", "model_hash", "build_hash",
    }),
    "assembler_config": frozenset({
        "schema_version", "development_only", "formal_eligible", "files",
        "assembler_schema_hash", "ui_presentation_schema_hash",
        "ui_presentation_golden_fixture_hash", "atlas_vocabulary_hash",
        "assembler_impl_hash", "roi_resolver_input_hash",
    }),
    "target_config": frozenset({
        "schema_version", "development_only", "formal_eligible", "files",
        "threshold_hash",
    }),
}
def _load_cli_provider_factory() -> (
    FormalPredictorFactory | None
):
    """インストール済み package から prediction-only factory をロードする。

    survivors.formal_perception_provider が未インストールの場合は None を返す。
    本番環境では parser/detector package とともにインストールし、
    None 以外の factory が返るようにすること。
    """
    try:
        from survivors.formal_perception_provider import build_formal_predictor  # type: ignore[import]
        return build_formal_predictor
    except ImportError:
        return None


def _load_ground_truth_factory() -> FormalGroundTruthFactory | None:
    """reviewed annotation package loader。prediction provider とは別 module に固定する。"""
    try:
        from survivors.formal_annotation_loader import build_ground_truth_loader  # type: ignore[import]
        return build_ground_truth_loader
    except ImportError:
        return None


_CLI_PROVIDER_FACTORY: FormalPredictorFactory | None = _load_cli_provider_factory()
_GROUND_TRUTH_FACTORY: FormalGroundTruthFactory | None = _load_ground_truth_factory()


def _validate_restore_verdict(
    dependency: ArtifactDescriptor, verdict: ArtifactDescriptor
) -> None:
    if verdict.node_kind != "restore_test_verdict":
        raise ValueError("formal dependency requires restore_test_verdict")
    if tuple(verdict.parents) != (dependency.node_ref(),):
        raise ValueError("restore verdict is not bound to exact dependency descriptor")
    metadata = verdict.identity_metadata
    if (
        metadata.get("passed") is not True
        or metadata.get("blocking_reasons") != []
        or metadata.get("subject_identity_hash") != dependency.identity_hash
        or type(metadata.get("checked_object_count")) is not int
        or metadata["checked_object_count"] < len(dependency.files)
    ):
        raise ValueError("restore verdict did not pass exact dependency contents")


def _manifest_ref(descriptor: ArtifactDescriptor) -> ArtifactRef:
    manifest_id = descriptor.identity_metadata.get("manifest_logical_id")
    if manifest_id is not None:
        matches = [ref for ref in descriptor.files if ref.logical_id == manifest_id]
        if len(matches) != 1:
            raise ValueError("descriptor manifest_logical_id does not identify one file")
        return matches[0]
    json_files = [ref for ref in descriptor.files if ref.media_type == "application/json"]
    if len(json_files) != 1:
        raise ValueError("formal descriptor must identify exactly one JSON manifest")
    return json_files[0]


def _restore_dependencies(
    request: FormalBenchmarkRequest,
) -> dict[str, RestoredFormalArtifact]:
    """descriptor → restore verdict → 全 package files の順で exact restore する。"""
    if not isinstance(request.store, ArtifactStore):
        raise TypeError("formal runner requires ArtifactStore")
    descriptors = dict(request.dependency_descriptors)
    verdicts = dict(request.restore_verdicts)
    if set(descriptors) != _DEPENDENCY_NAMES or set(verdicts) != _DEPENDENCY_NAMES:
        raise ValueError(
            "formal dependency descriptors/restore verdicts are incomplete or unknown"
        )
    restored: dict[str, RestoredFormalArtifact] = {}
    for name in sorted(_DEPENDENCY_NAMES):
        descriptor = descriptors[name]
        verdict = verdicts[name]
        if not isinstance(descriptor, ArtifactDescriptor) or not isinstance(
            verdict, ArtifactDescriptor
        ):
            raise TypeError(f"{name} descriptor/verdict must be ArtifactDescriptor")
        if not descriptor.files:
            raise ValueError(f"{name} descriptor has no package files")
        _validate_restore_verdict(descriptor, verdict)
        files: dict[str, bytes] = {}
        for ref in descriptor.files:
            verification = request.store.verify(ref)
            if not verification.ok:
                raise ValueError(
                    f"{name} ArtifactStore verification failed: {verification.reason}"
                )
            data = request.store.object_path(ref.store_uri).read_bytes()
            if len(data) != ref.size_bytes or hashlib.sha256(data).hexdigest() != ref.sha256:
                raise ValueError(f"{name} restored file changed after verification")
            files[ref.logical_id] = data
            if not request.store.verify(ref).ok:
                raise ValueError(f"{name} changed during restore")
        manifest_ref = _manifest_ref(descriptor)
        manifest_bytes = files[manifest_ref.logical_id]
        manifest: dict[str, Any] | None = None
        if name != "capture_dataset":
            try:
                payload = json.loads(manifest_bytes)
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
            required_fields = _PACKAGE_REQUIRED_FIELDS[name]
            if set(payload) != required_fields:
                raise ValueError(
                    f"{name} package fields mismatch; "
                    f"missing={sorted(required_fields - set(payload))}, "
                    f"unknown={sorted(set(payload) - required_fields)}"
                )
            role_files = payload["files"]
            hash_fields = required_fields - {
                "schema_version", "development_only", "formal_eligible", "files"
            }
            if not isinstance(role_files, dict) or set(role_files) != hash_fields:
                raise ValueError(f"{name} files must exactly cover declared hash fields")
            refs_by_id = {ref.logical_id: ref for ref in descriptor.files}
            for hash_field in sorted(hash_fields):
                logical_id = role_files[hash_field]
                if type(logical_id) is not str or logical_id not in refs_by_id:
                    raise ValueError(f"{name} {hash_field} has no descriptor file")
                actual_hash = refs_by_id[logical_id].sha256
                if payload[hash_field] != actual_hash:
                    raise ValueError(
                        f"{name} {hash_field} does not match actual file content"
                    )
            manifest = payload
        restored[name] = RestoredFormalArtifact(
            descriptor, verdict, files, manifest
        )
    return restored


def _capture_manifest_bytes(restored: Mapping[str, RestoredFormalArtifact]) -> bytes:
    capture = restored["capture_dataset"]
    return capture.files[_manifest_ref(capture.descriptor).logical_id]


def _restore_split_and_sessions(
    request: FormalBenchmarkRequest, capture_bytes: bytes
) -> tuple[SplitManifest, dict[str, SessionManifest], SessionSplit]:
    split_ref = _manifest_ref(request.dependency_descriptors["capture_dataset"])
    split = _split_manifest_from_bytes(
        request.store.object_path(split_ref.store_uri), capture_bytes
    )
    if split.manifest_sha256 != split_ref.sha256:
        raise ValueError("capture split manifest is not exact-hash bound to ArtifactRef")
    # Phase 1: final session は metadata_only で restore してから reservation し、
    # その後に PNG デコード（full restore）する。競合 runner が reservation 前に
    # 同じ PNG を読まないよう、デコードと予約の順序を逆転させる。
    # calibration session は reservation 不要なので最初から full restore する。
    final_session_ids: set[str] = {
        unit.session_id
        for unit in split.splits.get("final_e2e_test", ())
    }

    manifests: dict[str, SessionManifest] = {}
    for split_name, units in split.splits.items():
        for unit in units:
            if unit.session_id in manifests:
                raise ValueError(f"duplicate session in split: {unit.session_id}")
            is_final = unit.session_id in final_session_ids
            manifests[unit.session_id] = DatasetWriter.restore(
                request.capture_store_root, unit.session_id,
                metadata_only=is_final,  # final は予約前に PNG をデコードしない
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


def _package_payloads(
    restored: Mapping[str, RestoredFormalArtifact],
) -> dict[str, Mapping[str, Any]]:
    return {
        name: artifact.manifest
        for name, artifact in restored.items()
        if name != "capture_dataset" and artifact.manifest is not None
    }


def _subject_hashes(
    request: FormalBenchmarkRequest,
    restored_payloads: Mapping[str, Mapping[str, Any]],
    profile: FittedPerceptionErrorProfile,
    *, calibration_profile_hash: str, lineage_seal_hash: str,
) -> dict[str, str]:
    parser = restored_payloads["parser_package"]
    detector = restored_payloads["detector_package"]
    assembler = restored_payloads["assembler_config"]
    target = restored_payloads["target_config"]
    values = {
        "parser_artifact_hash": parser.get("parser_artifact_hash"),
        "detector_artifact_hash": detector.get("detector_artifact_hash"),
        "model_hash": detector.get("model_hash"),
        "build_hash": detector.get("build_hash"),
        "assembler_schema_hash": assembler.get("assembler_schema_hash"),
        "ui_presentation_schema_hash": assembler.get("ui_presentation_schema_hash"),
        "ui_presentation_golden_fixture_hash": assembler.get(
            "ui_presentation_golden_fixture_hash"
        ),
        "config_hash": _manifest_ref(
            request.dependency_descriptors["target_config"]
        ).sha256,
        "capture_dataset_hash": _manifest_ref(
            request.dependency_descriptors["capture_dataset"]
        ).sha256,
        "calibration_profile_hash": calibration_profile_hash,
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


def _calibration_provenance_bytes(
    profile: FittedPerceptionErrorProfile,
    subject_hashes: Mapping[str, str],
) -> bytes:
    for name, value in subject_hashes.items():
        if type(name) is not str or not name or type(value) is not str or len(value) != 64:
            raise ValueError("calibration provenance subject hashes are invalid")
    return canonical_json_bytes({
        "schema_version": "perception_calibration_package.v1",
        "profile_artifact": profile.to_artifact_wire(),
        "subject_hashes": dict(sorted(subject_hashes.items())),
    })


def _calibration_descriptor_chain(
    request: FormalBenchmarkRequest,
    profile: FittedPerceptionErrorProfile,
    provenance_subject_hashes: Mapping[str, str],
) -> tuple[ArtifactDescriptor, ArtifactDescriptor]:
    capture_ref = _manifest_ref(request.dependency_descriptors["capture_dataset"])
    source = ArtifactDescriptor(
        logical_id="perception/capture/source",
        node_kind="source_descriptor",
        producer_id="benchmark_survivors_perception",
        producer_version="v2",
        identity_metadata={"split_manifest_hash": capture_ref.sha256},
        files=(capture_ref,),
    )
    profile_bytes = canonical_json_bytes(profile.to_wire())
    provenance_bytes = _calibration_provenance_bytes(
        profile, provenance_subject_hashes
    )
    profile_node = ArtifactDescriptor(
        logical_id=request.calibration_logical_id,
        node_kind="perception_calibration_profile",
        producer_id="perception_error_fit",
        producer_version="v2",
        identity_metadata={
            "profile_hash": profile.profile_hash,
            "fit_code_hash": profile.fit_code_hash,
            "subject_hashes": dict(sorted(provenance_subject_hashes.items())),
        },
        parents=(source.node_ref(),),
        files=(
            _predicted_ref(
                f"perception/package/calibration/{hashlib.sha256(profile_bytes).hexdigest()}/profile.json",
                profile_bytes,
            ),
            _predicted_ref(
                f"perception/package/calibration/{hashlib.sha256(provenance_bytes).hexdigest()}/provenance.json",
                provenance_bytes,
            ),
        ),
    )
    validate_artifact_dag((source, profile_node))
    return source, profile_node


def _descriptor_chain(
    calibration_descriptors: tuple[ArtifactDescriptor, ArtifactDescriptor],
    request: FormalBenchmarkRequest,
    verdict: PerceptionFinalVerdict,
) -> tuple[ArtifactDescriptor, ...]:
    source, profile_node = calibration_descriptors
    verdict_bytes = canonical_json_bytes(verdict.to_wire())
    final_node = ArtifactDescriptor(
        logical_id=request.verdict_logical_id,
        node_kind="perception_final_verdict",
        producer_id="benchmark_survivors_perception",
        producer_version="v2",
        identity_metadata={
            "verdict_id": verdict.verdict_id,
            "seal_id": verdict.seal_id,
            "passed": verdict.passed,
            "development_only": verdict.development_only,
            "subject_hashes": {
                name: getattr(verdict, name)
                for name in verdict.to_wire()
                if name.endswith("_hash") and name not in {"verdict_id", "seal_id"}
            },
        },
        parents=(profile_node.node_ref(),),
        files=(_predicted_ref(
            f"perception/package/verdict/{hashlib.sha256(verdict_bytes).hexdigest()}/verdict.json",
            verdict_bytes,
        ),),
    )
    descriptors = (source, profile_node, final_node)
    validate_artifact_dag(descriptors)
    return descriptors


def _prediction_confidence(snapshot: PerceptionSnapshot | None) -> float:
    if snapshot is None:
        return 0.0
    return min(
        (
            target.confidence
            for target in (
                *snapshot.ui_presentation.candidates,
                *snapshot.ui_presentation.buttons,
            )
        ),
        default=1.0,
    )


def _category_index(value: str) -> int:
    from reinbalance_survivors_contracts.perception_error import ITEM_CATEGORY_SIZE

    return int(hashlib.sha256(value.encode("utf-8")).hexdigest(), 16) % ITEM_CATEGORY_SIZE


def _unpack_prediction_result(
    value: object,
) -> tuple[PerceptionSnapshot | None, float]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError("formal predictor must return (prediction, latency_ms)")
    predicted, latency_ms = value
    if predicted is not None and not isinstance(predicted, PerceptionSnapshot):
        raise TypeError("formal predictor returned a non-snapshot prediction")
    if isinstance(latency_ms, bool) or not isinstance(latency_ms, (int, float)):
        raise TypeError("formal predictor latency_ms must be numeric")
    return predicted, float(latency_ms)


def _derive_calibration_residuals(
    ticks: list[SnapshotReplayTick],
) -> list[CalibrationResidual]:
    """runner-owned GT/prediction pair から全 calibration residual を導出する。"""
    residuals: list[CalibrationResidual] = []
    for tick in ticks:
        ground = tick.ground_truth
        predicted = tick.predicted
        ground_context = ground.item_context
        predicted_context = predicted.item_context if predicted is not None else None
        confidence = _prediction_confidence(predicted)
        missing = float(predicted is None)

        def append(field: str, value: float, **categories: int) -> None:
            residuals.append(CalibrationResidual(
                tick.session_id, tick.frame_id, field, value, confidence,
                0, tick.latency_ms, **categories,
            ))

        ground_nearest = (
            ground_context.nearest_enemy_screen_dist
            if ground_context is not None
            and ground_context.nearest_enemy_screen_dist is not None
            else 0.0
        )
        predicted_nearest = (
            predicted_context.nearest_enemy_screen_dist
            if predicted_context is not None
            and predicted_context.nearest_enemy_screen_dist is not None
            else ground_nearest
        )
        append("coord_noise", abs(ground_nearest - predicted_nearest))
        for field, attribute in (
            ("hp_ratio", "hp_ratio"),
            ("xp_ratio", "xp_ratio"),
            ("timer_seconds", "elapsed_time"),
        ):
            ground_value = getattr(ground_context, attribute) if ground_context else 0.0
            predicted_value = (
                getattr(predicted_context, attribute)
                if predicted_context is not None else ground_value
            )
            append(field, float(predicted_value - ground_value))
        append(
            "inventory_hash",
            float(
                predicted is None
                or predicted.ui_presentation.inventory_hash
                != ground.ui_presentation.inventory_hash
            ),
        )
        ground_targets = ground.ui_presentation.candidates
        predicted_targets = predicted.ui_presentation.candidates if predicted else ()
        if ground_targets and predicted_targets:
            ground_center = (
                (ground_targets[0].roi.left + ground_targets[0].roi.right) / 2,
                (ground_targets[0].roi.top + ground_targets[0].roi.bottom) / 2,
            )
            predicted_center = (
                (predicted_targets[0].roi.left + predicted_targets[0].roi.right) / 2,
                (predicted_targets[0].roi.top + predicted_targets[0].roi.bottom) / 2,
            )
            quantization = (
                (predicted_center[0] - ground_center[0]) ** 2
                + (predicted_center[1] - ground_center[1]) ** 2
            ) ** 0.5 * 1920.0
        else:
            quantization = missing
        append("coord_quantization_px", quantization)
        append("burst_enter", missing)
        append("burst_exit", 1.0 - missing)
        append("burst_dropout", missing)
        collapsed = float(
            predicted is None
            or (
                ground.screen_state != "unknown"
                and predicted.screen_state == "unknown"
            )
        )
        append("unknown_screen_collapse", collapsed)
        append("unknown_screen_collapse_duration", collapsed)
        ground_item = ground_targets[0].choice_id if ground_targets else "unknown"
        predicted_item = (
            predicted_targets[0].choice_id if predicted_targets else "unknown"
        )
        append(
            "item_category", 0.0,
            ground_truth_category=_category_index(ground_item),
            predicted_category=_category_index(predicted_item),
        )
        ground_enemy = (
            "boss" if ground_context and ground_context.boss_flag
            else "hazard" if ground_context and ground_context.hazard_flag
            else "normal"
        )
        predicted_enemy = (
            "boss" if predicted_context and predicted_context.boss_flag
            else "hazard" if predicted_context and predicted_context.hazard_flag
            else "normal"
        )
        append(
            "enemy_category", 0.0,
            ground_truth_category=_category_index(ground_enemy),
            predicted_category=_category_index(predicted_enemy),
        )
    return residuals


def _replay_sessions(
    manifests: tuple[SessionManifest, ...],
    session_records: Mapping[str, Any],
    predictor: FormalPredictor,
    ground_truth_loader: FormalGroundTruthLoader,
    *,
    parser_hash: str,
    detector_hash: str,
) -> tuple[list[SnapshotReplayTick], list[ExpectedTick], dict[str, FormalReplayEvidence]]:
    ticks: list[SnapshotReplayTick] = []
    expected: list[ExpectedTick] = []
    evidence_by_frame: dict[str, FormalReplayEvidence] = {}
    for manifest in manifests:
        record = session_records[manifest.session_id]
        for frame in manifest.frames:
            frame_id = f"{manifest.session_id}:{frame.session_frame_index}"
            expected.append(ExpectedTick(manifest.session_id, frame_id))
            ground, evidence = ground_truth_loader.load_frame(
                frame, manifest.session_id, frame.session_frame_index
            )
            if ground is None:
                continue
            predicted, latency_ms = _unpack_prediction_result(
                predictor.predict_frame(
                    frame, manifest.session_id, frame.session_frame_index
                )
            )
            if evidence is None:
                raise ValueError(f"annotation evidence missing for {frame_id}")
            if ground.frame_id != frame_id:
                raise ValueError("annotation ground truth frame_id is not capture-bound")
            if predicted is not None and predicted.frame_id != frame_id:
                raise ValueError("prediction frame_id is not capture-bound")
            snapshots = (ground,) if predicted is None else (ground, predicted)
            if any(
                snapshot.ui_presentation.parser_artifact_hash != parser_hash
                for snapshot in snapshots
            ):
                raise ValueError("snapshot parser hash is not bound to restored package")
            if predicted is not None and predicted.detector_artifact_hash != detector_hash:
                raise ValueError("prediction detector hash is not bound to restored package")
            # annotation snapshot の detector field は評価対象ではない。record へ渡す
            # identity は verified detector package へ統一する。
            ground = replace(ground, detector_artifact_hash=detector_hash)
            evidence_by_frame[frame_id] = evidence
            ticks.append(SnapshotReplayTick(
                manifest.session_id, record.kind, record.source_policy,
                frame_id, ground, predicted, latency_ms,
            ))
    return ticks, expected, evidence_by_frame


def _put_descriptor_file(
    store: ArtifactStore, ref: ArtifactRef, data: bytes
) -> ArtifactRef:
    written = store.put_bytes(
        logical_id=ref.logical_id, data=data, media_type=ref.media_type
    )
    if written != ref or not store.verify(written).ok:
        raise ValueError("staged descriptor file does not match prevalidated ArtifactRef")
    return written


def _put_descriptor(store: ArtifactStore, descriptor: ArtifactDescriptor) -> ArtifactRef:
    data = canonical_json_bytes(descriptor.to_wire())
    return store.put_bytes(
        logical_id=f"perception/package/descriptors/{descriptor.identity_hash}.json",
        data=data,
        media_type="application/json",
    )


def _load_json_ref(store: ArtifactStore, ref: ArtifactRef) -> dict[str, Any]:
    if not store.verify(ref).ok:
        raise ValueError(f"committed ArtifactRef {ref.logical_id!r} is not restorable")
    try:
        value = json.loads(store.object_path(ref.store_uri).read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("committed artifact is not JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("committed JSON artifact must be an object")
    return value


def _batch_run_key(
    request: FormalBenchmarkRequest, validated_split: SessionSplit
) -> str:
    return hashlib.sha256(canonical_json_bytes({
        "dependencies": {
            name: request.dependency_descriptors[name].identity_hash
            for name in sorted(_DEPENDENCY_NAMES)
        },
        "sessions": [
            {
                "session_id": session.session_id,
                "session_hash": session.session_hash,
                "kind": session.kind,
            }
            for session in sorted(
                validated_split.sessions, key=lambda value: value.session_id
            )
            if session.kind in {"error_calibration", "final_e2e_test"}
        ],
    })).hexdigest()


def _verify_batch_refs(
    store: ArtifactStore, refs_wire: object
) -> dict[str, ArtifactRef]:
    if not isinstance(refs_wire, list) or not refs_wire:
        raise ValueError("formal batch refs must be a non-empty list")
    refs: dict[str, ArtifactRef] = {}
    for wire in refs_wire:
        ref = ArtifactRef.from_wire(wire)
        if ref.logical_id in refs:
            raise ValueError("formal batch contains duplicate logical ids")
        if not store.verify(ref).ok:
            raise ValueError("formal batch references missing/corrupt content")
        refs[ref.logical_id] = ref
    return refs


def _recover_committed_result(
    request: FormalBenchmarkRequest,
    validated_split: SessionSplit,
    run_key: str,
) -> FormalBenchmarkResult | None:
    """batch commit 済み state から canonical alias だけを idempotent 回復する。"""
    batch_ref = request.store.resolve(f"perception/batch_commit/{run_key}")
    if batch_ref is None:
        return None
    payload = _load_json_ref(request.store, batch_ref)
    expected_fields = {
        "schema_version", "run_key", "descriptors", "refs",
        "descriptor_hashes", "profile_artifact", "verdict",
    }
    if set(payload) != expected_fields or payload.get("schema_version") != "perception_formal_batch_commit.v2":
        raise ValueError("existing formal batch commit has an invalid schema")
    if payload["run_key"] != run_key:
        raise ValueError("existing formal batch commit has a different run key")
    refs = _verify_batch_refs(request.store, payload["refs"])
    descriptors_wire = payload["descriptors"]
    if not isinstance(descriptors_wire, list):
        raise ValueError("formal batch descriptors must be a list")
    descriptors = tuple(
        ArtifactDescriptor.from_wire(wire) for wire in descriptors_wire
    )
    validate_artifact_dag(descriptors)
    if len(descriptors) != 3:
        raise ValueError("formal batch must contain source/profile/verdict descriptors")
    if payload["descriptor_hashes"] != [
        descriptor.identity_hash for descriptor in descriptors
    ]:
        raise ValueError("formal batch descriptor hashes do not match descriptors")
    descriptor_ref_ids = {
        file_ref.logical_id
        for descriptor in descriptors
        for file_ref in descriptor.files
    }
    if not descriptor_ref_ids <= set(refs):
        raise ValueError("formal batch did not stage every descriptor ArtifactRef")
    profile = FittedPerceptionErrorProfile.from_artifact_wire(
        payload["profile_artifact"]
    )
    verdict_wire = payload["verdict"]
    if not isinstance(verdict_wire, dict):
        raise ValueError("formal batch verdict must be an object")
    verdict = load_final_verdict(
        verdict_wire,
        current_subject_hashes={name: verdict_wire[name] for name in _HASH_FIELDS},
    )
    if (
        descriptors[1].identity_hash != verdict.calibration_profile_hash
        or descriptors[2].identity_metadata.get("verdict_id") != verdict.verdict_id
    ):
        raise ValueError("formal batch descriptor/content identities do not match")

    profile_file = descriptors[1].files[0]
    provenance_file = descriptors[1].files[1]
    verdict_file = descriptors[2].files[0]
    if _load_json_ref(request.store, profile_file) != profile.to_wire():
        raise ValueError("committed raw profile does not match profile artifact")
    provenance = _load_json_ref(request.store, provenance_file)
    if (
        set(provenance) != {"schema_version", "profile_artifact", "subject_hashes"}
        or provenance["schema_version"] != "perception_calibration_package.v1"
        or provenance["profile_artifact"] != profile.to_artifact_wire()
        or provenance["subject_hashes"] != descriptors[1].identity_metadata.get("subject_hashes")
    ):
        raise ValueError("committed calibration provenance/package binding is invalid")
    if _load_json_ref(request.store, verdict_file) != verdict.to_wire():
        raise ValueError("committed verdict file does not match verdict payload")
    calibration_ref = request.store.put_bytes(
        logical_id=request.calibration_logical_id,
        data=request.store.object_path(profile_file.store_uri).read_bytes(),
        media_type="application/json",
    )
    request.store.put_bytes(
        logical_id=f"{request.calibration_logical_id}.provenance.json",
        data=request.store.object_path(provenance_file.store_uri).read_bytes(),
        media_type="application/json",
    )
    verdict_ref = request.store.put_bytes(
        logical_id=request.verdict_logical_id,
        data=request.store.object_path(verdict_file.store_uri).read_bytes(),
        media_type="application/json",
    )
    return FormalBenchmarkResult(
        validated_split, profile, verdict, calibration_ref, verdict_ref, descriptors
    )


def run_formal_pipeline(request: FormalBenchmarkRequest) -> FormalBenchmarkResult:
    """descriptor/restore/package/annotation-bound formal benchmark state machine。"""
    restored = _restore_dependencies(request)
    capture_bytes = _capture_manifest_bytes(restored)
    _, manifests, validated_split = _restore_split_and_sessions(
        request, capture_bytes
    )
    run_key = _batch_run_key(request, validated_split)
    recovered = _recover_committed_result(request, validated_split, run_key)
    if recovered is not None:
        return recovered
    if _CLI_PROVIDER_FACTORY is None or _GROUND_TRUTH_FACTORY is None:
        raise ValueError(
            "formal pipeline requires prediction-only provider and reviewed annotation loader"
        )

    payloads = _package_payloads(restored)
    detector_payload = payloads["detector_package"]
    parser_payload = payloads["parser_package"]
    parser_hash = parser_payload["parser_artifact_hash"]
    detector_hash = detector_payload["detector_artifact_hash"]
    build_hash = detector_payload["build_hash"]
    if any(
        session.build_hash != build_hash
        for session in validated_split.sessions
        if session.kind in {"error_calibration", "final_e2e_test"}
    ):
        raise ValueError("capture sessions are not bound to verified game build content")

    predictor = _CLI_PROVIDER_FACTORY(restored)
    ground_truth_loader = _GROUND_TRUTH_FACTORY(
        request.capture_store_root, restored
    )
    if not hasattr(predictor, "predict_frame"):
        raise TypeError("formal provider must implement prediction-only predict_frame")
    if not hasattr(ground_truth_loader, "load_frame"):
        raise TypeError("formal annotation loader must implement load_frame")

    records_by_id = {session.session_id: session for session in validated_split.sessions}
    calibration_manifests = tuple(
        manifests[session.session_id]
        for session in validated_split.sessions
        if session.kind == "error_calibration"
    )
    final_metadata_manifests = tuple(
        manifests[session.session_id]
        for session in validated_split.sessions
        if session.kind == "final_e2e_test"
    )

    # Phase A: calibration だけを replay/fit し、profile package を commit/freeze する。
    calibration_ticks, _, _ = _replay_sessions(
        calibration_manifests, records_by_id, predictor, ground_truth_loader,
        parser_hash=parser_hash, detector_hash=detector_hash,
    )
    residuals = _derive_calibration_residuals(calibration_ticks)
    calibration_hashes = {
        manifest.session_id: manifest.manifest_sha256
        for manifest in calibration_manifests
    }
    profile = fit_error_profile(
        residuals,
        [manifest.session_id for manifest in calibration_manifests],
        [manifest.session_id for manifest in final_metadata_manifests],
        calibration_session_hashes=calibration_hashes,
        formal=True,
    )
    provisional_subjects = _subject_hashes(
        request, payloads, profile,
        calibration_profile_hash="0" * 64,
        lineage_seal_hash="0" * 64,
    )
    provenance_subjects = {
        name: value for name, value in provisional_subjects.items()
        if name not in {"calibration_profile_hash", "lineage_seal_hash"}
    }
    calibration_descriptors = _calibration_descriptor_chain(
        request, profile, provenance_subjects
    )
    profile_node = calibration_descriptors[1]
    profile_bytes = canonical_json_bytes(profile.to_wire())
    provenance_bytes = _calibration_provenance_bytes(profile, provenance_subjects)
    staged_refs = [
        _put_descriptor_file(request.store, profile_node.files[0], profile_bytes),
        _put_descriptor_file(request.store, profile_node.files[1], provenance_bytes),
        *(
            _put_descriptor(request.store, descriptor)
            for descriptor in calibration_descriptors
        ),
    ]
    calibration_commit = canonical_json_bytes({
        "schema_version": "perception_calibration_commit.v1",
        "run_key": run_key,
        "profile_descriptor_hash": profile_node.identity_hash,
        "refs": [ref.to_wire() for ref in staged_refs],
    })
    request.store.put_bytes(
        logical_id=f"perception/calibration_commit/{run_key}",
        data=calibration_commit,
        media_type="application/json",
    )
    calibration_ref = request.store.put_bytes(
        logical_id=request.calibration_logical_id,
        data=profile_bytes,
        media_type="application/json",
    )
    request.store.put_bytes(
        logical_id=f"{request.calibration_logical_id}.provenance.json",
        data=provenance_bytes,
        media_type="application/json",
    )

    # Phase B: committed calibration Artifact を subject に lineage seal を構築する。
    final_hashes = {
        manifest.session_id: manifest.manifest_sha256
        for manifest in final_metadata_manifests
    }
    preseal_subjects = _subject_hashes(
        request, payloads, profile,
        calibration_profile_hash=profile_node.identity_hash,
        lineage_seal_hash="0" * 64,
    )
    seal_subjects = {
        name: value for name, value in preseal_subjects.items()
        if name != "lineage_seal_hash"
    }
    seal = _create_formal_lineage_seal(
        final_session_hashes=final_hashes,
        store=request.store,
        _publish=False,
        **seal_subjects,
    )
    seal_bytes = canonical_json_bytes(seal.to_wire())
    lineage_seal_hash = hashlib.sha256(seal_bytes).hexdigest()
    subjects = _subject_hashes(
        request, payloads, profile,
        calibration_profile_hash=profile_node.identity_hash,
        lineage_seal_hash=lineage_seal_hash,
    )

    # Phase C: calibration freeze と seal 完了後にだけ final を予約・full restore する。
    for manifest in final_metadata_manifests:
        _reserve_final_session(
            request.store, manifest.session_id, manifest.manifest_sha256,
            manifest.session_path / "session_manifest.json",
        )
    final_manifests = tuple(
        DatasetWriter.restore(request.capture_store_root, manifest.session_id)
        for manifest in final_metadata_manifests
    )
    final_ticks, expected_ticks, evidence = _replay_sessions(
        final_manifests, records_by_id, predictor, ground_truth_loader,
        parser_hash=parser_hash, detector_hash=detector_hash,
    )
    report = _formalize_benchmark_report(run_benchmark(
        final_ticks, expected_ticks=expected_ticks, formal_evidence=evidence
    ))
    if not report.passed:
        raise ValueError(
            "formal perception gate failed; calibration remains frozen: "
            + "; ".join(report.blocking_reasons)
        )
    verdict = _create_formal_final_verdict(
        report, seal_id=seal.seal_id,
        final_session_ids=[manifest.session_id for manifest in final_manifests],
        **subjects,
    )
    descriptors = _descriptor_chain(calibration_descriptors, request, verdict)
    verdict_bytes = canonical_json_bytes(verdict.to_wire())
    final_node = descriptors[2]
    staged_refs.extend([
        _put_descriptor_file(request.store, final_node.files[0], verdict_bytes),
        request.store.put_bytes(
            logical_id=f"perception/package/lineage/{seal.seal_id}/seal.json",
            data=seal_bytes,
            media_type="application/json",
        ),
        *(
            _put_descriptor(request.store, descriptor)
            for descriptor in descriptors
        ),
    ])
    staged_refs.extend(
        ref
        for artifact in restored.values()
        for ref in artifact.descriptor.files
    )
    # 全 descriptor file ref が batch commit 前に実在することを再検証する。
    staged_by_logical = {ref.logical_id: ref for ref in staged_refs}
    for descriptor in descriptors:
        for file_ref in descriptor.files:
            staged = staged_by_logical.get(file_ref.logical_id)
            if staged != file_ref or not request.store.verify(file_ref).ok:
                raise ValueError("descriptor ArtifactRef was not staged before batch commit")

    staged_refs = list({ref.logical_id: ref for ref in staged_refs}.values())
    batch_payload = canonical_json_bytes({
        "schema_version": "perception_formal_batch_commit.v2",
        "run_key": run_key,
        "descriptors": [descriptor.to_wire() for descriptor in descriptors],
        "descriptor_hashes": [descriptor.identity_hash for descriptor in descriptors],
        "refs": [ref.to_wire() for ref in staged_refs],
        "profile_artifact": profile.to_artifact_wire(),
        "verdict": verdict.to_wire(),
    })
    request.store.put_bytes(
        logical_id=f"perception/batch_commit/{run_key}",
        data=batch_payload,
        media_type="application/json",
    )
    verdict_ref = request.store.put_bytes(
        logical_id=request.verdict_logical_id,
        data=verdict_bytes,
        media_type="application/json",
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


def _load_descriptor(path_str: str) -> ArtifactDescriptor:
    path = Path(path_str).expanduser().resolve()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid ArtifactDescriptor file: {path}") from exc
    return ArtifactDescriptor.from_wire(data)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Survivors perception formal benchmark")
    for option in (
        "capture-dataset", "parser-package", "detector-package",
        "assembler-config", "target-config",
    ):
        parser.add_argument(f"--{option}-descriptor", help=f"{option} ArtifactDescriptor JSON")
        parser.add_argument(
            f"--{option}-restore-verdict",
            help=f"{option} restore-test ArtifactDescriptor JSON",
        )
    parser.add_argument("--artifact-store", help="ArtifactStore root")
    parser.add_argument("--capture-store", help="restorable capture session store root")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.dry_run:
        return _run_dry()
    required = {
        **{
            f"{name}_descriptor": getattr(args, f"{name}_descriptor")
            for name in _DEPENDENCY_NAMES
        },
        **{
            f"{name}_restore_verdict": getattr(args, f"{name}_restore_verdict")
            for name in _DEPENDENCY_NAMES
        },
        "artifact_store": args.artifact_store,
        "capture_store": args.capture_store,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        print(f"BLOCKED: missing formal inputs: {', '.join(sorted(missing))}", file=sys.stderr)
        return 2
    try:
        descriptors = {
            name: _load_descriptor(required[f"{name}_descriptor"])
            for name in _DEPENDENCY_NAMES
        }
        restore_verdicts = {
            name: _load_descriptor(required[f"{name}_restore_verdict"])
            for name in _DEPENDENCY_NAMES
        }
        store = ArtifactStore(required["artifact_store"])
        result = run_formal_pipeline(
            FormalBenchmarkRequest(
                store=store, dependency_descriptors=descriptors,
                restore_verdicts=restore_verdicts,
                capture_store_root=Path(required["capture_store"]),
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
