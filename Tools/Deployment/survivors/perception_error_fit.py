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
from reinbalance_survivors_contracts.perception_profile import (
    CALIBRATION_ARTIFACT_SCHEMA_VERSION,
    CalibrationResidual,
    FittedPerceptionErrorProfile,
    FormalVerdictPromotionError,
    HashMismatchError,
    InvalidResidualError,
    _FORMAL_FACTORY_TOKEN,
)
# _RESIDUAL_FIELDS は CalibrationResidual (Common) が使うためここでは不要
from reinbalance_survivors_contracts.artifact_store import ArtifactStoreError

from .perception_benchmark import BenchmarkReport, recompute_gate_from_metrics

LINEAGE_SEAL_SCHEMA_VERSION: Final[str] = "perception_lineage_seal.v2"
CALIBRATION_VERDICT_SCHEMA_VERSION: Final[str] = "perception_calibration_verdict.v1"
FINAL_VERDICT_SCHEMA_VERSION: Final[str] = "perception_final_verdict.v2"

_HASH_FIELDS: Final[tuple[str, ...]] = (
    "parser_artifact_hash",
    "detector_artifact_hash",
    "model_hash",
    "build_hash",
    "assembler_schema_hash",
    "ui_presentation_schema_hash",
    "ui_presentation_golden_fixture_hash",
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
_FORMAL_REQUIRED_RESIDUAL_FIELDS: Final[frozenset[str]] = frozenset({
    "coord_noise", "hp_ratio", "xp_ratio", "timer_seconds",
    "inventory_hash", "coord_quantization_px",
    "burst_enter", "burst_exit", "burst_dropout",
    "unknown_screen_collapse", "unknown_screen_collapse_duration",
    "item_category", "enemy_category",
})
_FORMAL_RESIDUAL_MIN_SAMPLES: Final[int] = 3
_FORMAL_RESIDUAL_MIN_SESSIONS: Final[int] = 3


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


# residual 導出は runner (benchmark_survivors_perception)、metric gate は
# perception_benchmark、fit は本ファイルに分散する。producer closure hash はこの三者を
# 束ねて計算し、いずれか一つでも変更すると calibration profile / final verdict の subject が
# 必ず変わるようにする（fit ファイル単独 hash では runner/metric 変更を捕捉できない）。
_PRODUCER_CLOSURE_FILES: Final[tuple[str, ...]] = (
    "survivors/perception_error_fit.py",
    "survivors/perception_benchmark.py",
    "benchmark_survivors_perception.py",
    # Common の共有型定義（validation/wire/schema）も fit 契約の一部。
    # deployment_root 基準で ../Common/ へアクセスする。
    "../Common/src/reinbalance_survivors_contracts/perception_profile.py",
)


def _current_fit_code_hash() -> str:
    """producer closure（fit + benchmark runner + metric gate）の content hash。"""
    deployment_root = Path(__file__).resolve().parent.parent  # Tools/Deployment
    digests: list[str] = []
    for relative in _PRODUCER_CLOSURE_FILES:
        path = deployment_root / relative
        digests.append(hashlib.sha256(path.read_bytes()).hexdigest())
    return hashlib.sha256("\0".join(digests).encode("ascii")).hexdigest()


class FinalSessionAlreadyOpenedError(ValueError):
    """final session の create-once marker が既に存在する。"""


class FinalSessionNotInSealError(ValueError):
    """final session が seal identity の固定集合に含まれない。"""


class FinalFitMixingError(ValueError):
    """calibration residual 集合へ final/unknown session が混入した。"""


class StaleVerdictError(ValueError):
    """producer/profile/threshold hash または gate 結果が stale である。"""


class SessionOverlapError(ValueError):
    """calibration/final session identity が重複している。"""


class EmptyResidualError(ValueError):
    """residual が 0 件、または calibration session の一部が無標本である。"""




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
    for row in counts:
        row_sum = row.sum()
        if row_sum == 0.0:
            # 未観測カテゴリは all-zeros（identity fallback は confidence=0 の未観測を
            # 完全正解として扱い formal underpowered 要件を偽って通過させるため除去）。
            matrix.append([0.0] * ITEM_CATEGORY_SIZE)
        else:
            matrix.append([float(value / row_sum) for value in row])
    return matrix


def fit_error_profile(
    residuals: Sequence[CalibrationResidual],
    calibration_session_ids: Sequence[str],
    final_e2e_session_ids: Sequence[str],
    *,
    calibration_session_hashes: Mapping[str, str] | None = None,
    formal: bool = False,
    _factory_token: object | None = None,
) -> FittedPerceptionErrorProfile:
    """exact calibration residual 集合から既存 consumer-compatible profile を fit する。

    formal=True の場合は必須 field 集合と最低サンプル数／セッション数を厳格に検証します。
    疎な synthetic fit が必要なら formal=False（既定）で呼び出してください。
    公開 API は常に development_only=True を返します。formal profile は
    _fit_formal_error_profile() 経由でのみ生成できます。
    """
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

    # confidence=0 の行は有効な観測として扱わない。formal power チェックも
    # 実効サンプル（confidence>0 かつ valid な観測）のみを数える。
    by_field: dict[str, list[CalibrationResidual]] = {}
    for residual in residuals:
        if residual.confidence > 0.0:
            by_field.setdefault(residual.field, []).append(residual)
    sample_counts = {name: len(rows) for name, rows in by_field.items()}
    underpowered = {name: count for name, count in sample_counts.items() if count < 2}
    if underpowered:
        raise EmptyResidualError(
            f"residual fields are underpowered (minimum 2 samples): {underpowered}"
        )

    if formal:
        missing_required = _FORMAL_REQUIRED_RESIDUAL_FIELDS - set(by_field)
        if missing_required:
            raise EmptyResidualError(
                f"formal fit missing required residual fields: {sorted(missing_required)}"
            )
        for fname in sorted(_FORMAL_REQUIRED_RESIDUAL_FIELDS):
            rows = by_field[fname]
            if len(rows) < _FORMAL_RESIDUAL_MIN_SAMPLES:
                raise EmptyResidualError(
                    f"formal fit: field {fname!r} underpowered "
                    f"({len(rows)} samples < {_FORMAL_RESIDUAL_MIN_SAMPLES} required)"
                )
            sessions = {r.session_id for r in rows}
            if len(sessions) < _FORMAL_RESIDUAL_MIN_SESSIONS:
                raise EmptyResidualError(
                    f"formal fit: field {fname!r} from too few sessions "
                    f"({len(sessions)} < {_FORMAL_RESIDUAL_MIN_SESSIONS} required)"
                )

    def weighted_rows(name: str) -> tuple[list[float], list[float]]:
        # by_field は confidence>0 の行だけを含む。weight は正値が保証される。
        rows = by_field.get(name, [])
        return (
            [row.residual for row in rows],
            [row.confidence / (1.0 + row.age_frames) for row in rows],
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

    # latency は (session_id, frame_id) ごとに 1 観測。burst/collapse 遷移行は
    # session レベルで生成される別 field（burst_enter/exit/dropout 等）のため除外する。
    # 0 frame latency は per-tick 観測として有効なので値によるフィルタは使わない。
    _LATENCY_TRANSITION_FIELDS: Final[frozenset[str]] = frozenset({
        "burst_enter", "burst_exit", "burst_dropout",
        "unknown_screen_collapse", "unknown_screen_collapse_duration",
    })
    tick_latency: dict[tuple[str, str], float] = {}
    for row in residuals:
        if row.field in _LATENCY_TRANSITION_FIELDS:
            continue
        key = (row.session_id, row.frame_id)
        if key in tick_latency and tick_latency[key] != row.latency_frames:
            raise InvalidResidualError(
                f"conflicting latency for tick {key!r}: "
                f"{tick_latency[key]} vs {row.latency_frames}"
            )
        tick_latency[key] = row.latency_frames
    latency_values = list(tick_latency.values())
    latency_weights = [1.0] * len(latency_values)
    fit_code_hash = _current_fit_code_hash()
    return FittedPerceptionErrorProfile(
        latency_mean_frames=_weighted_mean(latency_values, latency_weights) if latency_values else 0.0,
        latency_std_frames=_weighted_std(latency_values, latency_weights) if latency_values else 0.0,
        burst_enter_prob=mean("burst_enter"),
        burst_exit_prob=mean("burst_exit", 1.0),
        burst_dropout_prob=mean("burst_dropout"),
        coord_noise_std=std("coord_noise"),
        coord_quantization_px=mean("coord_quantization_px"),
        hud_hp_misread_std=std("hp_ratio"),
        hud_xp_stale_prob=probability("xp_ratio", 0.1),
        hud_timer_stale_prob=probability("timer_seconds", 1.0),
        hud_inventory_stale_prob=probability("inventory_hash", 0.5),
        unknown_screen_collapse_prob=mean("unknown_screen_collapse"),
        unknown_screen_collapse_duration_frames=mean("unknown_screen_collapse_duration"),
        item_confusion_matrix=_confusion_matrix(by_field.get("item_category", [])),
        enemy_confusion_matrix=_confusion_matrix(by_field.get("enemy_category", [])),
        calibration_session_ids=cal_ids,
        final_e2e_session_ids=final_ids,
        calibration_session_hashes=hashes,
        field_sample_counts=sample_counts,
        fit_code_hash=fit_code_hash,
        development_only=_factory_token is not _FORMAL_FACTORY_TOKEN,
        _factory_token=_factory_token,
    )


def _fit_formal_error_profile(
    residuals: Sequence[CalibrationResidual],
    calibration_session_ids: Sequence[str],
    final_e2e_session_ids: Sequence[str],
    *,
    calibration_session_hashes: Mapping[str, str] | None = None,
) -> FittedPerceptionErrorProfile:
    """Formal runner 専用 factory。development_only=False を生成できる唯一の経路。

    split・restore・provenance 検証済みの formal runner pipeline だけが呼ぶ。
    公開 fit_error_profile() は常に development_only=True を返す。
    """
    return fit_error_profile(
        residuals, calibration_session_ids, final_e2e_session_ids,
        calibration_session_hashes=calibration_session_hashes,
        formal=True,
        _factory_token=_FORMAL_FACTORY_TOKEN,
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
    model_hash: str
    build_hash: str
    assembler_schema_hash: str
    ui_presentation_schema_hash: str
    ui_presentation_golden_fixture_hash: str
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
        if session_id in self.opened_session_ids:
            raise FinalSessionAlreadyOpenedError(f"final session {session_id!r} already opened")
        _reserve_final_session(
            self._store, session_id, expected_hash, session_manifest_path
        )
        self.opened_session_ids.append(session_id)

    def to_wire(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "seal_id": self.seal_id,
            "final_session_hashes": dict(self.final_session_hashes),
            "development_only": self.development_only,
            **{name: getattr(self, name) for name in _SEAL_SUBJECT_HASH_FIELDS},
        }


def _reserve_final_session(
    store: Any, session_id: str, expected_hash: str, session_manifest_path: Path
) -> None:
    """manifest を再検証し、formal final session を global create-once 予約する。"""
    path = Path(session_manifest_path).resolve()
    if not path.is_file():
        raise HashMismatchError(f"missing final session manifest: {path}")
    actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_hash != expected_hash:
        raise HashMismatchError(
            f"final session {session_id!r} hash mismatch: {actual_hash} != {expected_hash}"
        )
    if store is None:
        return
    marker = canonical_json_bytes(
        {"schema_version": "perception_final_open.v1",
         "session_id": session_id, "content_hash": expected_hash}
    )
    # session marker は seal identity に依存せず、失敗時にも消費済みとして残す。
    # provider/publish より前の atomic create-once が競合 session の再利用を防ぐ。
    try:
        store.put_bytes_create_once(
            logical_id=f"perception/lineage/opened/{session_id}.json",
            data=marker, media_type="application/json",
        )
    except ArtifactStoreError as exc:
        raise FinalSessionAlreadyOpenedError(
            f"final session {session_id!r} already opened"
        ) from exc

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
        seal_bytes = canonical_json_bytes(seal.to_wire())
        seal_logical_id = logical_id or f"perception/lineage/{seal.seal_id}/seal.json"
        # seal は idempotent put_bytes：同一 lineage の再実行でも同一 sha256 を再 publish できる。
        # commit 責務は呼び出し元の batch commit manifest が担う（そちらが single atomic 操作）。
        ref = store.put_bytes(
            logical_id=seal_logical_id, data=seal_bytes, media_type="application/json",
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
    model_hash: str
    build_hash: str
    assembler_schema_hash: str
    ui_presentation_schema_hash: str
    ui_presentation_golden_fixture_hash: str
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
    passed, blocking = recompute_gate_from_metrics(
        report.metrics_wire(), formal=formal_eligible
    )
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


def _write_formal_final_verdict(
    verdict: PerceptionFinalVerdict, *, store: Any, logical_id: str,
) -> Any:
    if verdict.development_only or not verdict.formal_perception_verdict_eligible or store is None:
        raise FormalVerdictPromotionError("formal writer accepts only factory-built formal verdicts")
    if not verdict.passed:
        raise FormalVerdictPromotionError(
            "formal final verdict must have passed=True before publish"
        )
    ref = store.put_bytes(
        logical_id=logical_id, data=canonical_json_bytes(verdict.to_wire()),
        media_type="application/json",
    )
    if not store.verify(ref).ok:
        raise HashMismatchError("final verdict publish revalidation failed")
    return ref


def load_final_verdict(
    data: dict[str, Any], *,
    current_subject_hashes: Mapping[str, str],
) -> PerceptionFinalVerdict:
    """exact schema/hash を検証し、保存済み metric から gate を必ず再計算する。

    current_subject_hashes は _HASH_FIELDS の全 13 フィールドを含む必要があります。
    5 フィールドのみを検証する旧インターフェースを廃止し、完全な subject mapping の一致を要求します。
    """
    if not isinstance(data, dict) or set(data) != _FINAL_VERDICT_REQUIRED_FIELDS:
        raise ValueError("final verdict fields must exactly match the v2 schema")
    if data["schema_version"] != FINAL_VERDICT_SCHEMA_VERSION:
        raise ValueError("unsupported final verdict schema_version")
    for name in ("verdict_id", "seal_id", *_HASH_FIELDS):
        _require_sha256(data[name], name)
    if set(current_subject_hashes) != set(_HASH_FIELDS):
        missing = sorted(set(_HASH_FIELDS) - set(current_subject_hashes))
        extra = sorted(set(current_subject_hashes) - set(_HASH_FIELDS))
        raise ValueError(
            f"current_subject_hashes must exactly cover all _HASH_FIELDS "
            f"(missing: {missing}, extra: {extra})"
        )
    for name, value in current_subject_hashes.items():
        _require_sha256(value, f"current {name}")
    stale = [name for name in _HASH_FIELDS if data[name] != current_subject_hashes[name]]
    if stale:
        raise StaleVerdictError(f"final perception verdict is stale: {stale}")
    if any(type(data[name]) is not bool for name in ("passed", "development_only", "formal_perception_verdict_eligible")):
        raise ValueError("verdict flags must be exact bool values")
    if not isinstance(data["blocking_reasons"], list) or not all(type(value) is str for value in data["blocking_reasons"]):
        raise ValueError("blocking_reasons must be a string list")
    passed, blocking = recompute_gate_from_metrics(
        data["metrics"], formal=not data["development_only"]
    )
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
    # wire ローダーは formal token を付与しない。formal verdict はストア検証経路のみで取得する。
    if not data["development_only"]:
        raise FormalVerdictPromotionError(
            "formal final verdict cannot be loaded from raw wire; "
            "use _load_formal_verdict_from_verified_store() instead"
        )
    return PerceptionFinalVerdict(
        **{name: data[name] for name in _FINAL_VERDICT_REQUIRED_FIELDS if name != "schema_version"},
        schema_version=data["schema_version"],
        _factory_token=None,
    )


def _load_formal_verdict_from_verified_store(
    data: dict[str, Any], *,
    store: Any,
    ref: Any,
    current_subject_hashes: Mapping[str, str],
) -> PerceptionFinalVerdict:
    """ArtifactStore の store.verify(ref) を内部で実行してから formal verdict をロードする。

    store と ref を必須にすることで、呼び出し元の事前検証に依存しない
    fail-closed 境界を維持する。raw dict だけを受け取る旧呼び出しは拒否される。
    """
    verification = store.verify(ref)
    if not verification.ok:
        raise HashMismatchError(
            f"formal verdict store verification failed: {verification.reason}"
        )
    # load_final_verdict と同じ検証を実施する（development_only チェックのみ緩和）。
    if not isinstance(data, dict) or set(data) != _FINAL_VERDICT_REQUIRED_FIELDS:
        raise ValueError("final verdict fields must exactly match the v2 schema")
    if data["schema_version"] != FINAL_VERDICT_SCHEMA_VERSION:
        raise ValueError("unsupported final verdict schema_version")
    for name in ("verdict_id", "seal_id", *_HASH_FIELDS):
        _require_sha256(data[name], name)
    if set(current_subject_hashes) != set(_HASH_FIELDS):
        missing = sorted(set(_HASH_FIELDS) - set(current_subject_hashes))
        extra = sorted(set(current_subject_hashes) - set(_HASH_FIELDS))
        raise ValueError(
            f"current_subject_hashes must exactly cover all _HASH_FIELDS "
            f"(missing: {missing}, extra: {extra})"
        )
    for name, value in current_subject_hashes.items():
        _require_sha256(value, f"current {name}")
    stale = [name for name in _HASH_FIELDS if data[name] != current_subject_hashes[name]]
    if stale:
        raise StaleVerdictError(f"final perception verdict is stale: {stale}")
    if any(type(data[name]) is not bool for name in ("passed", "development_only", "formal_perception_verdict_eligible")):
        raise ValueError("verdict flags must be exact bool values")
    if not isinstance(data["blocking_reasons"], list) or not all(type(value) is str for value in data["blocking_reasons"]):
        raise ValueError("blocking_reasons must be a string list")
    passed, blocking = recompute_gate_from_metrics(
        data["metrics"], formal=not data["development_only"]
    )
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
    # ストア検証経路から呼ばれるため formal token を付与する。
    factory_token = _FORMAL_FACTORY_TOKEN if not data["development_only"] else None
    return PerceptionFinalVerdict(
        **{name: data[name] for name in _FINAL_VERDICT_REQUIRED_FIELDS if name != "schema_version"},
        schema_version=data["schema_version"],
        _factory_token=factory_token,
    )
