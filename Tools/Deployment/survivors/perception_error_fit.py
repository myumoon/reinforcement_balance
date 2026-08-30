"""Perception error fit、formal lineage seal、stale-proof verdict 契約。"""

from __future__ import annotations

import hashlib
import math
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Mapping, Sequence

import numpy as np

from reinbalance_survivors_contracts.canonical_json import canonical_hash, canonical_json_bytes
from reinbalance_survivors_contracts.perception_error import (
    ITEM_CATEGORY_SIZE,
    PerceptionErrorProfile,
)

from .perception_benchmark import BenchmarkReport, recompute_gate_from_metrics

LINEAGE_SEAL_SCHEMA_VERSION: Final[str] = "perception_lineage_seal.v2"
CALIBRATION_VERDICT_SCHEMA_VERSION: Final[str] = "perception_calibration_verdict.v1"
CALIBRATION_ARTIFACT_SCHEMA_VERSION: Final[str] = "perception_calibration_profile.v1"
FINAL_VERDICT_SCHEMA_VERSION: Final[str] = "perception_final_verdict.v2"

_HASH_FIELDS: Final[tuple[str, ...]] = (
    "parser_artifact_hash",
    "detector_artifact_hash",
    "assembler_schema_hash",
    "ui_presentation_schema_hash",
    "config_hash",
    "capture_dataset_hash",
    "calibration_profile_hash",
    "threshold_hash",
    "atlas_vocabulary_hash",
    "assembler_impl_hash",
    "roi_resolver_input_hash",
    "benchmark_fit_code_hash",
    "lineage_seal_hash",
)
_SEAL_SUBJECT_HASH_FIELDS: Final[tuple[str, ...]] = tuple(
    name for name in _HASH_FIELDS if name != "lineage_seal_hash"
)
_FINAL_VERDICT_REQUIRED_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version", "verdict_id", "seal_id", "final_session_ids",
        "metrics", "passed", "blocking_reasons", "development_only",
        "formal_perception_verdict_eligible", *_HASH_FIELDS,
    }
)
_SEAL_REQUIRED_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version", "seal_id", "final_session_hashes", "development_only",
        *_SEAL_SUBJECT_HASH_FIELDS,
    }
)
_RESIDUAL_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "hp_ratio", "xp_ratio", "timer_seconds", "inventory_hash", "coord_noise",
        "coord_quantization_px", "burst_enter", "burst_exit", "burst_dropout",
        "unknown_screen_collapse", "unknown_screen_collapse_duration",
        "item_category", "enemy_category",
    }
)
_FORMAL_FACTORY_TOKEN = object()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: object, label: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{label} must be a 64-character lowercase SHA-256")
    return value  # type: ignore[return-value]


def _strict_number(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not math.isfinite(float(value))
    ):
        raise InvalidResidualError(f"{label} must be a finite number (bool is forbidden)")
    return float(value)


class FinalSessionAlreadyOpenedError(ValueError):
    """final session の create-once marker が既に存在する。"""


class FinalSessionNotInSealError(ValueError):
    """final session が seal identity の固定集合に含まれない。"""


class FinalFitMixingError(ValueError):
    """calibration residual 集合へ final/unknown session が混入した。"""


class StaleVerdictError(ValueError):
    """producer/profile/threshold hash または gate 結果が stale である。"""


class HashMismatchError(ValueError):
    """実 artifact content hash が seal/verdict の exact hash と一致しない。"""


class SessionOverlapError(ValueError):
    """calibration/final session identity が重複している。"""


class EmptyResidualError(ValueError):
    """residual が 0 件、または calibration session の一部が無標本である。"""


class InvalidResidualError(ValueError):
    """residual field/type/range が fit 契約外である。"""


class FormalVerdictPromotionError(ValueError):
    """synthetic public constructor から formal flag を構築しようとした。"""


@dataclass(frozen=True, slots=True)
class CalibrationResidual:
    session_id: str
    frame_id: str
    field: str
    residual: float
    confidence: float
    age_frames: int
    latency_frames: float = 0.0
    ground_truth_category: int | None = None
    predicted_category: int | None = None

    def __post_init__(self) -> None:
        if type(self.session_id) is not str or not self.session_id:
            raise InvalidResidualError("session_id must be a non-empty string")
        if type(self.frame_id) is not str or not self.frame_id:
            raise InvalidResidualError("frame_id must be a non-empty string")
        if self.field not in _RESIDUAL_FIELDS:
            raise InvalidResidualError(f"unsupported residual field {self.field!r}")
        object.__setattr__(self, "residual", _strict_number(self.residual, "residual"))
        confidence = _strict_number(self.confidence, "confidence")
        if not 0.0 <= confidence <= 1.0:
            raise InvalidResidualError("confidence must be in [0, 1]")
        object.__setattr__(self, "confidence", confidence)
        if type(self.age_frames) is not int or self.age_frames < 0:
            raise InvalidResidualError("age_frames must be a non-negative integer")
        latency = _strict_number(self.latency_frames, "latency_frames")
        if latency < 0.0:
            raise InvalidResidualError("latency_frames must be non-negative")
        object.__setattr__(self, "latency_frames", latency)
        category_pair = (self.ground_truth_category, self.predicted_category)
        if (category_pair[0] is None) != (category_pair[1] is None):
            raise InvalidResidualError("category ground truth/prediction must be provided together")
        if category_pair[0] is not None:
            if self.field not in {"item_category", "enemy_category"}:
                raise InvalidResidualError("category labels require a category residual field")
            if any(type(value) is not int or not 0 <= value < ITEM_CATEGORY_SIZE for value in category_pair):
                raise InvalidResidualError("category labels are outside the fixed vocabulary")
        elif self.field in {"item_category", "enemy_category"}:
            raise InvalidResidualError("category residual fields require category labels")


@dataclass(frozen=True)
class FittedPerceptionErrorProfile(PerceptionErrorProfile):
    """既存 profile wire を維持しつつ fit artifact metadata を保持する subtype。"""

    calibration_session_hashes: Mapping[str, str] = field(default_factory=dict)
    field_sample_counts: Mapping[str, int] = field(default_factory=dict)
    fit_code_hash: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        hashes = dict(self.calibration_session_hashes)
        if set(hashes) != set(self.calibration_session_ids):
            raise ValueError("calibration_session_hashes must exactly match calibration ids")
        for session_id, content_hash in hashes.items():
            if type(session_id) is not str or not session_id:
                raise ValueError("calibration session hash key must be non-empty")
            _require_sha256(content_hash, f"calibration_session_hashes[{session_id!r}]")
        counts = dict(self.field_sample_counts)
        if not counts or not all(type(name) is str and type(count) is int and count > 0 for name, count in counts.items()):
            raise ValueError("field_sample_counts must contain positive integer counts")
        _require_sha256(self.fit_code_hash, "fit_code_hash")
        object.__setattr__(self, "calibration_session_hashes", MappingProxyType(hashes))
        object.__setattr__(self, "field_sample_counts", MappingProxyType(counts))

    def to_artifact_wire(self) -> dict[str, Any]:
        return {
            "schema_version": CALIBRATION_ARTIFACT_SCHEMA_VERSION,
            "profile": self.to_wire(),
            "profile_hash": self.profile_hash,
            "calibration_session_hashes": dict(self.calibration_session_hashes),
            "field_sample_counts": dict(self.field_sample_counts),
            "fit_code_hash": self.fit_code_hash,
        }

    @property
    def artifact_hash(self) -> str:
        return canonical_hash(self.to_artifact_wire())


def _weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    return float(np.average(np.asarray(values, dtype=float), weights=np.asarray(weights, dtype=float)))


def _weighted_std(values: Sequence[float], weights: Sequence[float]) -> float:
    mean = _weighted_mean(values, weights)
    return float(
        math.sqrt(
            np.average(
                (np.asarray(values, dtype=float) - mean) ** 2,
                weights=np.asarray(weights, dtype=float),
            )
        )
    )


def _confusion_matrix(rows: Sequence[CalibrationResidual]) -> list[list[float]]:
    if not rows:
        return []
    counts = np.zeros((ITEM_CATEGORY_SIZE, ITEM_CATEGORY_SIZE), dtype=float)
    for residual in rows:
        assert residual.ground_truth_category is not None
        assert residual.predicted_category is not None
        counts[residual.ground_truth_category, residual.predicted_category] += (
            residual.confidence / (1.0 + residual.age_frames)
        )
    matrix: list[list[float]] = []
    for index, row in enumerate(counts):
        if row.sum() == 0.0:
            values = [0.0] * ITEM_CATEGORY_SIZE
            values[index] = 1.0
        else:
            values = [float(value / row.sum()) for value in row]
        matrix.append(values)
    return matrix


def fit_error_profile(
    residuals: Sequence[CalibrationResidual],
    calibration_session_ids: Sequence[str],
    final_e2e_session_ids: Sequence[str],
    *,
    calibration_session_hashes: Mapping[str, str] | None = None,
) -> FittedPerceptionErrorProfile:
    """exact calibration residual 集合から既存 consumer-compatible profile を fit する。"""
    if not residuals:
        raise EmptyResidualError("at least one calibration residual is required")
    if not all(isinstance(residual, CalibrationResidual) for residual in residuals):
        raise InvalidResidualError("residuals must contain CalibrationResidual values")
    cal_ids = list(calibration_session_ids)
    final_ids = list(final_e2e_session_ids)
    if not cal_ids or any(type(value) is not str or not value for value in cal_ids):
        raise EmptyResidualError("calibration_session_ids must be non-empty")
    if len(cal_ids) != len(set(cal_ids)) or len(final_ids) != len(set(final_ids)):
        raise SessionOverlapError("session id lists must be unique")
    overlap = set(cal_ids) & set(final_ids)
    if overlap:
        raise SessionOverlapError(f"calibration/final session overlap: {sorted(overlap)}")
    residual_ids = {residual.session_id for residual in residuals}
    unknown = residual_ids - set(cal_ids)
    if unknown:
        if unknown & set(final_ids):
            raise FinalFitMixingError(f"final residuals present: {sorted(unknown & set(final_ids))}")
        raise FinalFitMixingError(f"residuals outside exact calibration set: {sorted(unknown)}")
    missing = set(cal_ids) - residual_ids
    if missing:
        raise EmptyResidualError(f"calibration sessions have 0 residuals: {sorted(missing)}")

    if calibration_session_hashes is None:
        # synthetic compatibility: identity is still exact and explicit in the artifact.
        hashes = {session_id: canonical_hash({"synthetic_session_id": session_id}) for session_id in cal_ids}
    else:
        hashes = dict(calibration_session_hashes)
        if set(hashes) != set(cal_ids):
            raise ValueError("calibration_session_hashes must exactly match calibration ids")
        for session_id, content_hash in hashes.items():
            _require_sha256(content_hash, f"calibration_session_hashes[{session_id!r}]")

    by_field: dict[str, list[CalibrationResidual]] = {}
    for residual in residuals:
        by_field.setdefault(residual.field, []).append(residual)
    sample_counts = {name: len(rows) for name, rows in by_field.items()}
    underpowered = {name: count for name, count in sample_counts.items() if count < 2}
    if underpowered:
        raise EmptyResidualError(
            f"residual fields are underpowered (minimum 2 samples): {underpowered}"
        )

    def weighted_rows(name: str) -> tuple[list[float], list[float]]:
        rows = by_field.get(name, [])
        return (
            [row.residual for row in rows],
            [max(row.confidence / (1.0 + row.age_frames), 1e-12) for row in rows],
        )

    def mean(name: str, default: float = 0.0) -> float:
        values, weights = weighted_rows(name)
        return _weighted_mean(values, weights) if values else default

    def std(name: str) -> float:
        values, weights = weighted_rows(name)
        return _weighted_std(values, weights) if values else 0.0

    def probability(name: str, threshold: float) -> float:
        values, weights = weighted_rows(name)
        if not values:
            return 0.0
        indicators = [float(abs(value) > threshold) for value in values]
        return _weighted_mean(indicators, weights)

    all_weights = [max(row.confidence / (1.0 + row.age_frames), 1e-12) for row in residuals]
    latency_values = [row.latency_frames for row in residuals]
    fit_code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    clamp = lambda value: min(1.0, max(0.0, value))
    return FittedPerceptionErrorProfile(
        latency_mean_frames=max(0.0, _weighted_mean(latency_values, all_weights)),
        latency_std_frames=max(0.0, _weighted_std(latency_values, all_weights)),
        burst_enter_prob=clamp(mean("burst_enter")),
        burst_exit_prob=clamp(mean("burst_exit", 1.0)),
        burst_dropout_prob=clamp(mean("burst_dropout")),
        coord_noise_std=max(0.0, std("coord_noise")),
        coord_quantization_px=max(0.0, mean("coord_quantization_px")),
        hud_hp_misread_std=max(0.0, std("hp_ratio")),
        hud_xp_stale_prob=clamp(probability("xp_ratio", 0.1)),
        hud_timer_stale_prob=clamp(probability("timer_seconds", 1.0)),
        hud_inventory_stale_prob=clamp(probability("inventory_hash", 0.5)),
        unknown_screen_collapse_prob=clamp(mean("unknown_screen_collapse")),
        unknown_screen_collapse_duration_frames=max(0.0, mean("unknown_screen_collapse_duration")),
        item_confusion_matrix=_confusion_matrix(by_field.get("item_category", [])),
        enemy_confusion_matrix=_confusion_matrix(by_field.get("enemy_category", [])),
        calibration_session_ids=cal_ids,
        final_e2e_session_ids=final_ids,
        calibration_session_hashes=hashes,
        field_sample_counts=sample_counts,
        fit_code_hash=fit_code_hash,
    )


def simulator_distance_report(
    calibrated: PerceptionErrorProfile, simulator: PerceptionErrorProfile
) -> dict[str, float]:
    return {
        "latency_mean_diff": abs(calibrated.latency_mean_frames - simulator.latency_mean_frames),
        "latency_std_diff": abs(calibrated.latency_std_frames - simulator.latency_std_frames),
        "coord_noise_std_diff": abs(calibrated.coord_noise_std - simulator.coord_noise_std),
        "hp_misread_std_diff": abs(calibrated.hud_hp_misread_std - simulator.hud_hp_misread_std),
        "xp_stale_prob_diff": abs(calibrated.hud_xp_stale_prob - simulator.hud_xp_stale_prob),
        "timer_stale_prob_diff": abs(calibrated.hud_timer_stale_prob - simulator.hud_timer_stale_prob),
        "burst_enter_diff": abs(calibrated.burst_enter_prob - simulator.burst_enter_prob),
        "burst_exit_diff": abs(calibrated.burst_exit_prob - simulator.burst_exit_prob),
        "burst_dropout_diff": abs(calibrated.burst_dropout_prob - simulator.burst_dropout_prob),
        "unknown_collapse_prob_diff": abs(calibrated.unknown_screen_collapse_prob - simulator.unknown_screen_collapse_prob),
    }


@dataclass
class FinalLineageSeal:
    seal_id: str
    final_session_hashes: Mapping[str, str]
    parser_artifact_hash: str
    detector_artifact_hash: str
    assembler_schema_hash: str
    ui_presentation_schema_hash: str
    config_hash: str
    capture_dataset_hash: str
    calibration_profile_hash: str
    threshold_hash: str
    atlas_vocabulary_hash: str
    assembler_impl_hash: str
    roi_resolver_input_hash: str
    benchmark_fit_code_hash: str
    development_only: bool = True
    schema_version: str = LINEAGE_SEAL_SCHEMA_VERSION
    opened_session_ids: list[str] = field(default_factory=list, repr=False)
    _store: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.schema_version != LINEAGE_SEAL_SCHEMA_VERSION:
            raise ValueError("unsupported lineage seal schema")
        hashes = dict(self.final_session_hashes)
        if not hashes:
            raise ValueError("final session set must not be empty")
        for session_id, content_hash in hashes.items():
            if type(session_id) is not str or not session_id:
                raise ValueError("final session ids must be non-empty strings")
            _require_sha256(content_hash, f"final_session_hashes[{session_id!r}]")
        self.final_session_hashes = MappingProxyType(hashes)
        for name in _SEAL_SUBJECT_HASH_FIELDS:
            _require_sha256(getattr(self, name), name)
        expected_id = canonical_hash(self.identity_payload())
        if self.seal_id != expected_id:
            raise HashMismatchError("seal_id does not bind the complete lineage identity")
        if type(self.development_only) is not bool:
            raise ValueError("development_only must be bool")
        if not self.development_only and self._store is None:
            raise FormalVerdictPromotionError("formal lineage seal requires ArtifactStore")

    @property
    def final_session_set(self) -> frozenset[str]:
        return frozenset(self.final_session_hashes)

    def identity_payload(self) -> dict[str, Any]:
        return {
            **{name: getattr(self, name) for name in _SEAL_SUBJECT_HASH_FIELDS},
            "final_sessions": [
                {"session_id": session_id, "session_hash": self.final_session_hashes[session_id]}
                for session_id in sorted(self.final_session_hashes)
            ],
        }

    def verify_hashes(self, **current_hashes: str) -> None:
        if set(current_hashes) != set(_SEAL_SUBJECT_HASH_FIELDS):
            raise ValueError("verify_hashes requires the complete sealed hash set")
        changed = [name for name in _SEAL_SUBJECT_HASH_FIELDS if current_hashes[name] != getattr(self, name)]
        if changed:
            raise StaleVerdictError(f"lineage seal is stale: {changed}")

    def open_session(self, session_id: str, session_manifest_path: Path) -> None:
        expected_hash = self.final_session_hashes.get(session_id)
        if expected_hash is None:
            raise FinalSessionNotInSealError(f"session {session_id!r} is not sealed")
        path = Path(session_manifest_path).resolve()
        if not path.is_file():
            raise HashMismatchError(f"missing final session manifest: {path}")
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise HashMismatchError(
                f"final session {session_id!r} hash mismatch: {actual_hash} != {expected_hash}"
            )
        if session_id in self.opened_session_ids:
            raise FinalSessionAlreadyOpenedError(f"final session {session_id!r} already opened")
        if self._store is not None:
            marker = canonical_json_bytes(
                {"schema_version": "perception_final_open.v1", "seal_id": self.seal_id,
                 "session_id": session_id, "session_hash": expected_hash}
            )
            try:
                self._store.put_bytes_create_once(
                    logical_id=f"perception/lineage/{self.seal_id}/opened/{session_id}.json",
                    data=marker, media_type="application/json",
                )
            except Exception as exc:
                raise FinalSessionAlreadyOpenedError(
                    f"final session {session_id!r} already opened or marker publish failed"
                ) from exc
        self.opened_session_ids.append(session_id)

    def to_wire(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "seal_id": self.seal_id,
            "final_session_hashes": dict(self.final_session_hashes),
            "development_only": self.development_only,
            **{name: getattr(self, name) for name in _SEAL_SUBJECT_HASH_FIELDS},
        }


def _create_lineage_seal(
    *, final_session_hashes: Mapping[str, str], development_only: bool, store: Any,
    logical_id: str | None = None, publish: bool = True, **subject_hashes: str,
) -> FinalLineageSeal:
    if set(subject_hashes) != set(_SEAL_SUBJECT_HASH_FIELDS):
        missing = sorted(set(_SEAL_SUBJECT_HASH_FIELDS) - set(subject_hashes))
        extra = sorted(set(subject_hashes) - set(_SEAL_SUBJECT_HASH_FIELDS))
        raise ValueError(f"seal subject hash set mismatch; missing={missing}, extra={extra}")
    identity = {
        **subject_hashes,
        "final_sessions": [
            {"session_id": session_id, "session_hash": final_session_hashes[session_id]}
            for session_id in sorted(final_session_hashes)
        ],
    }
    seal = FinalLineageSeal(
        seal_id=canonical_hash(identity), final_session_hashes=final_session_hashes,
        development_only=development_only, _store=store, **subject_hashes,
    )
    if store is not None and publish:
        ref = store.put_bytes(
            logical_id=logical_id or f"perception/lineage/{seal.seal_id}/seal.json",
            data=canonical_json_bytes(seal.to_wire()), media_type="application/json",
        )
        verification = store.verify(ref)
        if not verification.ok:
            raise HashMismatchError("lineage seal failed ArtifactStore revalidation")
    return seal


def create_lineage_seal(
    *, final_session_hashes: Mapping[str, str], store: Any = None,
    logical_id: str | None = None, **subject_hashes: str,
) -> FinalLineageSeal:
    """synthetic 入口。development_only は公開引数にせず True 固定。"""
    if "development_only" in subject_hashes:
        raise ValueError("development_only cannot be overridden on the synthetic API")
    return _create_lineage_seal(
        final_session_hashes=final_session_hashes, development_only=True,
        store=store, logical_id=logical_id, **subject_hashes,
    )


def _create_formal_lineage_seal(
    *, final_session_hashes: Mapping[str, str], store: Any,
    logical_id: str | None = None, _publish: bool = True, **subject_hashes: str,
) -> FinalLineageSeal:
    if store is None:
        raise FormalVerdictPromotionError("formal lineage seal requires ArtifactStore")
    return _create_lineage_seal(
        final_session_hashes=final_session_hashes, development_only=False,
        store=store, logical_id=logical_id, publish=_publish, **subject_hashes,
    )


@dataclass
class PerceptionCalibrationVerdict:
    profile: PerceptionErrorProfile
    calibration_session_ids: list[str]
    development_only: bool = True
    formal_perception_verdict_eligible: bool = False
    schema_version: str = CALIBRATION_VERDICT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.development_only is not True or self.formal_perception_verdict_eligible is not False:
            raise FormalVerdictPromotionError("synthetic calibration verdict flags are fixed")


@dataclass
class PerceptionFinalVerdict:
    verdict_id: str
    seal_id: str
    final_session_ids: list[str]
    parser_artifact_hash: str
    detector_artifact_hash: str
    assembler_schema_hash: str
    ui_presentation_schema_hash: str
    config_hash: str
    capture_dataset_hash: str
    calibration_profile_hash: str
    threshold_hash: str
    atlas_vocabulary_hash: str
    assembler_impl_hash: str
    roi_resolver_input_hash: str
    benchmark_fit_code_hash: str
    lineage_seal_hash: str
    metrics: dict[str, Any]
    passed: bool
    blocking_reasons: list[str]
    development_only: bool = True
    formal_perception_verdict_eligible: bool = False
    schema_version: str = FINAL_VERDICT_SCHEMA_VERSION
    _factory_token: InitVar[object | None] = None

    def __post_init__(self, _factory_token: object | None) -> None:
        _require_sha256(self.verdict_id, "verdict_id")
        _require_sha256(self.seal_id, "seal_id")
        for name in _HASH_FIELDS:
            _require_sha256(getattr(self, name), name)
        if (
            not isinstance(self.final_session_ids, list)
            or not self.final_session_ids
            or not all(type(value) is str and value for value in self.final_session_ids)
            or len(self.final_session_ids) != len(set(self.final_session_ids))
        ):
            raise ValueError("final_session_ids must be a non-empty unique list")
        if type(self.passed) is not bool or type(self.development_only) is not bool or type(self.formal_perception_verdict_eligible) is not bool:
            raise ValueError("verdict flags must be exact bool values")
        if self.development_only and self.formal_perception_verdict_eligible:
            raise FormalVerdictPromotionError("development verdict cannot be formal eligible")
        if not self.development_only and not self.formal_perception_verdict_eligible:
            raise FormalVerdictPromotionError("formal verdict must be formal eligible")
        if not self.development_only and _factory_token is not _FORMAL_FACTORY_TOKEN:
            raise FormalVerdictPromotionError("formal verdict requires the verified formal factory")
        if not isinstance(self.metrics, dict):
            raise ValueError("metrics must be a dict")
        if not isinstance(self.blocking_reasons, list) or not all(
            type(value) is str for value in self.blocking_reasons
        ):
            raise ValueError("blocking_reasons must be a string list")

    def to_wire(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version, "verdict_id": self.verdict_id,
            "seal_id": self.seal_id, "final_session_ids": list(self.final_session_ids),
            **{name: getattr(self, name) for name in _HASH_FIELDS},
            "metrics": self.metrics, "passed": self.passed,
            "blocking_reasons": list(self.blocking_reasons),
            "development_only": self.development_only,
            "formal_perception_verdict_eligible": self.formal_perception_verdict_eligible,
        }


def _verdict_identity_payload(
    report: BenchmarkReport, seal_id: str, final_session_ids: Sequence[str],
    subject_hashes: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "seal_id": seal_id, "final_session_ids": sorted(final_session_ids),
        "subject_hashes": dict(subject_hashes), "metrics": report.metrics_wire(),
    }


def _create_final_verdict(
    report: BenchmarkReport, *, seal_id: str, final_session_ids: Sequence[str],
    development_only: bool, formal_eligible: bool, **subject_hashes: str,
) -> PerceptionFinalVerdict:
    if set(subject_hashes) != set(_HASH_FIELDS):
        raise ValueError("final verdict requires the complete hash subject set")
    passed, blocking = recompute_gate_from_metrics(report.metrics_wire())
    if report.passed != passed or report.blocking_reasons != blocking:
        raise StaleVerdictError("BenchmarkReport gate fields do not match metric recomputation")
    identity = _verdict_identity_payload(report, seal_id, final_session_ids, subject_hashes)
    return PerceptionFinalVerdict(
        verdict_id=canonical_hash(identity), seal_id=seal_id,
        final_session_ids=list(sorted(final_session_ids)), metrics=report.metrics_wire(),
        passed=passed, blocking_reasons=blocking, development_only=development_only,
        formal_perception_verdict_eligible=formal_eligible,
        _factory_token=_FORMAL_FACTORY_TOKEN if not development_only else None,
        **subject_hashes,
    )


def create_synthetic_final_verdict(
    report: BenchmarkReport, *, seal_id: str, final_session_ids: Sequence[str],
    **subject_hashes: str,
) -> PerceptionFinalVerdict:
    if report.development_only is not True or report.formal_perception_verdict_eligible is not False:
        raise FormalVerdictPromotionError("synthetic verdict requires a synthetic benchmark report")
    return _create_final_verdict(
        report, seal_id=seal_id, final_session_ids=final_session_ids,
        development_only=True, formal_eligible=False, **subject_hashes,
    )


def _create_formal_final_verdict(
    report: BenchmarkReport, *, seal_id: str, final_session_ids: Sequence[str],
    **subject_hashes: str,
) -> PerceptionFinalVerdict:
    if report.development_only is not False or report.formal_perception_verdict_eligible is not True:
        raise FormalVerdictPromotionError("formal verdict requires a formal benchmark report")
    return _create_final_verdict(
        report, seal_id=seal_id, final_session_ids=final_session_ids,
        development_only=False, formal_eligible=True, **subject_hashes,
    )


def _write_formal_calibration_profile(
    profile: FittedPerceptionErrorProfile, *, store: Any, logical_id: str,
) -> Any:
    if not isinstance(profile, FittedPerceptionErrorProfile) or store is None:
        raise FormalVerdictPromotionError("formal calibration writer requires fitted profile and ArtifactStore")
    ref = store.put_bytes(
        logical_id=logical_id, data=canonical_json_bytes(profile.to_artifact_wire()),
        media_type="application/json",
    )
    if not store.verify(ref).ok:
        raise HashMismatchError("calibration profile publish revalidation failed")
    return ref


def _write_formal_final_verdict(
    verdict: PerceptionFinalVerdict, *, store: Any, logical_id: str,
) -> Any:
    if verdict.development_only or not verdict.formal_perception_verdict_eligible or store is None:
        raise FormalVerdictPromotionError("formal writer accepts only factory-built formal verdicts")
    ref = store.put_bytes(
        logical_id=logical_id, data=canonical_json_bytes(verdict.to_wire()),
        media_type="application/json",
    )
    if not store.verify(ref).ok:
        raise HashMismatchError("final verdict publish revalidation failed")
    return ref


def load_final_verdict(
    data: dict[str, Any], *, current_parser_hash: str, current_detector_hash: str,
    current_assembler_hash: str, current_config_hash: str,
    current_ui_schema_hash: str,
) -> PerceptionFinalVerdict:
    """exact schema/hash を検証し、保存済み metric から gate を必ず再計算する。"""
    if not isinstance(data, dict) or set(data) != _FINAL_VERDICT_REQUIRED_FIELDS:
        raise ValueError("final verdict fields must exactly match the v2 schema")
    if data["schema_version"] != FINAL_VERDICT_SCHEMA_VERSION:
        raise ValueError("unsupported final verdict schema_version")
    for name in ("verdict_id", "seal_id", *_HASH_FIELDS):
        _require_sha256(data[name], name)
    current = {
        "parser_artifact_hash": current_parser_hash,
        "detector_artifact_hash": current_detector_hash,
        "assembler_schema_hash": current_assembler_hash,
        "config_hash": current_config_hash,
        "ui_presentation_schema_hash": current_ui_schema_hash,
    }
    for name, value in current.items():
        _require_sha256(value, f"current {name}")
    stale = [name for name, value in current.items() if data[name] != value]
    if stale:
        raise StaleVerdictError(f"final perception verdict is stale: {stale}")
    if any(type(data[name]) is not bool for name in ("passed", "development_only", "formal_perception_verdict_eligible")):
        raise ValueError("verdict flags must be exact bool values")
    if not isinstance(data["blocking_reasons"], list) or not all(type(value) is str for value in data["blocking_reasons"]):
        raise ValueError("blocking_reasons must be a string list")
    passed, blocking = recompute_gate_from_metrics(data["metrics"])
    if data["passed"] != passed or data["blocking_reasons"] != blocking:
        raise StaleVerdictError("stored pass/blocking result does not match metric gate")
    identity_payload = {
        "seal_id": data["seal_id"],
        "final_session_ids": sorted(data["final_session_ids"]),
        "subject_hashes": {name: data[name] for name in _HASH_FIELDS},
        "metrics": data["metrics"],
    }
    if canonical_hash(identity_payload) != data["verdict_id"]:
        raise HashMismatchError("verdict_id does not bind metrics and complete subject hashes")
    return PerceptionFinalVerdict(
        **{name: data[name] for name in _FINAL_VERDICT_REQUIRED_FIELDS if name != "schema_version"},
        schema_version=data["schema_version"],
        _factory_token=_FORMAL_FACTORY_TOKEN if not data["development_only"] else None,
    )
