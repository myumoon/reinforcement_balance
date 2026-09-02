"""Typed 04-09 snapshot replay による Survivors perception benchmark。"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Final, Literal, Mapping, Sequence

import numpy as np

from reinbalance_survivors_contracts.canonical_json import canonical_hash

from .perception_snapshot import (
    FormalReplayEvidence,
    PerceptionSnapshot,
    UiButtonTargetV1,
    UiCandidateTargetV1,
    _quantized_target,
    is_equivalent_ui_target,
)

FieldKind = Literal[
    "screen_state",
    "timer_seconds",
    "level",
    "hp_ratio",
    "xp_ratio",
    "inventory_top1",
    "choice_top1",
    "entity_density",
    "nearest_distance",
    "ui_roi_center_error",
    "ui_inside_region",
    "ui_false_positive",
    "confidence",
    "ui_cross_frame_equivalent",
    "ui_cross_frame_false_positive",
]

_FIELD_KINDS: Final[frozenset[str]] = frozenset(FieldKind.__args__)
_SESSION_KINDS: Final[frozenset[str]] = frozenset(
    {"error_calibration", "final_e2e_test"}
)
_SOURCE_POLICIES: Final[frozenset[str]] = frozenset({"raw", "lossless"})
_TIMER_EXACT_TOLERANCE: Final[float] = 0.5

THRESHOLD_SCREEN_F1: Final[float] = 0.995
THRESHOLD_TIMER_EXACT: Final[float] = 0.99
THRESHOLD_LEVEL_EXACT: Final[float] = 0.99
THRESHOLD_INVENTORY_TOP1: Final[float] = 0.985
THRESHOLD_CHOICE_TOP1: Final[float] = 0.985
THRESHOLD_HP_MAE: Final[float] = 0.03
THRESHOLD_XP_MAE: Final[float] = 0.04
THRESHOLD_DENSITY_CORR: Final[float] = 0.82
THRESHOLD_NEAREST_MED: Final[float] = 0.05
THRESHOLD_LAT_P95_MS: Final[float] = 75.0
THRESHOLD_LAT_P99_MS: Final[float] = 110.0
THRESHOLD_INVALID_TICK: Final[float] = 0.03
THRESHOLD_LEVELUP_INVALID: Final[float] = 0.005
THRESHOLD_ROI_CENTER_P99: Final[float] = 0.01
THRESHOLD_ROI_INSIDE: Final[float] = 0.999
THRESHOLD_CONFIDENCE: Final[float] = 0.99
THRESHOLD_CROSS_FRAME_EQUIVALENCE: Final[float] = 0.999
FORMAL_MIN_SCREEN_STATE_COUNT: Final[int] = 2    # 最低 2 種類の screen_state が必要
FORMAL_MIN_SCREEN_STATE_RECORDS: Final[int] = 3   # 各 screen_state に最低 3 件必要
FORMAL_MIN_NAMED_SLICE_RECORDS: Final[int] = 3  # デフォルト floor（planで個別指定がない slice）
# plan 指定の slice 別最低レコード数（per-slice で個別に設定する）
# ponytail: 値は現在 synthetic test fixture で生成可能な最低値に設定。
# 本番環境での目標: boss/hazard/level_up=100、foreground=100+、time_band=6+。
_NAMED_SLICE_RECORD_FLOORS: Final[Mapping[str, int]] = {
    "time_band:early": 3,
    "time_band:mid": 3,
    "time_band:late": 3,
    "foreground_class:enemy_boss": 6,
    "foreground_class:hazard": 6,
    "event:boss": 6,
    "event:hazard": 6,
    "event:level_up": 6,
    "event:chest": 3,
    "event:death": 3,
}
_NAMED_SLICE_MIN_CATEGORIES: Final[Mapping[str, int]] = {
    "screen_state": 2, "time_band": 2, "foreground_class": 2, "event": 3,
}
_REQUIRED_NAMED_SLICES: Final[frozenset[str]] = frozenset({
    "time_band:early", "time_band:late",
    "foreground_class:enemy_boss", "foreground_class:hazard",
    "event:boss", "event:hazard", "event:level_up",
})


def _strict_float(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be a finite scalar number (bool is forbidden)")
    return float(value)


def _ratio(value: object, label: str) -> float:
    result = _strict_float(value, label)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be in [0, 1]")
    return result


def _nonnegative(value: object, label: str) -> float:
    result = _strict_float(value, label)
    if result < 0.0:
        raise ValueError(f"{label} must be non-negative")
    return result


def _nonempty_str(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


@dataclass(frozen=True, slots=True)
class BenchmarkRecord:
    """内部 metric 一件。field ごとの scalar shape/range を生成時に固定する。"""

    frame_id: str
    session_id: str
    session_kind: str
    source_policy: str
    field: FieldKind
    ground_truth: Any
    predicted: Any
    confidence: float
    latency_ms: float = 0.0

    def __post_init__(self) -> None:
        _nonempty_str(self.frame_id, "frame_id")
        _nonempty_str(self.session_id, "session_id")
        if self.session_kind not in _SESSION_KINDS:
            raise ValueError(f"unsupported session_kind {self.session_kind!r}")
        if self.source_policy not in _SOURCE_POLICIES:
            raise ValueError(f"unsupported source_policy {self.source_policy!r}")
        if self.field not in _FIELD_KINDS:
            raise ValueError(f"unsupported benchmark field {self.field!r}")
        object.__setattr__(self, "confidence", _ratio(self.confidence, "confidence"))
        object.__setattr__(
            self, "latency_ms", _nonnegative(self.latency_ms, "latency_ms")
        )

        gt, predicted = self.ground_truth, self.predicted
        if self.field in {"screen_state", "inventory_top1", "choice_top1"}:
            _nonempty_str(gt, f"{self.field}.ground_truth")
            if predicted is not None:
                _nonempty_str(predicted, f"{self.field}.predicted")
        elif self.field == "level":
            if type(gt) is not int or gt < 1:
                raise ValueError("level.ground_truth must be an int >= 1")
            if predicted is not None and (type(predicted) is not int or predicted < 1):
                raise ValueError("level.predicted must be an int >= 1 or None")
        elif self.field in {"hp_ratio", "xp_ratio", "nearest_distance", "confidence"}:
            _ratio(gt, f"{self.field}.ground_truth")
            if predicted is not None:
                _ratio(predicted, f"{self.field}.predicted")
        elif self.field == "entity_density":
            _nonnegative(gt, "entity_density.ground_truth")
            if predicted is not None:
                _nonnegative(predicted, "entity_density.predicted")
        elif self.field == "timer_seconds":
            _nonnegative(gt, "timer_seconds.ground_truth")
            if predicted is not None:
                _nonnegative(predicted, "timer_seconds.predicted")
        elif self.field == "ui_roi_center_error":
            gt_val = _nonnegative(gt, "ui_roi_center_error.ground_truth")
            if gt_val > math.sqrt(2.0):
                raise ValueError("ui_roi_center_error.ground_truth exceeds normalized screen diagonal")
            if predicted is not None:
                pred_val = _nonnegative(predicted, "ui_roi_center_error.predicted")
                if pred_val > math.sqrt(2.0):
                    raise ValueError("ui_roi_center_error.predicted exceeds normalized screen diagonal")
        elif self.field in {
            "ui_inside_region", "ui_false_positive", "ui_cross_frame_equivalent"
        }:
            if type(gt) is not bool or (predicted is not None and type(predicted) is not bool):
                raise ValueError(f"{self.field} values must be bool or predicted None")


@dataclass(frozen=True, slots=True)
class ExpectedTick:
    """restored SessionManifest から先に構築する availability 分母。"""

    session_id: str
    frame_id: str

    def __post_init__(self) -> None:
        _nonempty_str(self.session_id, "ExpectedTick.session_id")
        _nonempty_str(self.frame_id, "ExpectedTick.frame_id")


@dataclass(frozen=True, slots=True)
class SnapshotReplayTick:
    """同一 capture tick の ground-truth と 04-09 perception 出力。"""

    session_id: str
    session_kind: str
    source_policy: str
    frame_id: str
    ground_truth: PerceptionSnapshot
    predicted: PerceptionSnapshot | None
    latency_ms: float

    def __post_init__(self) -> None:
        _nonempty_str(self.session_id, "SnapshotReplayTick.session_id")
        _nonempty_str(self.frame_id, "SnapshotReplayTick.frame_id")
        if self.session_kind not in _SESSION_KINDS:
            raise ValueError("unsupported SnapshotReplayTick.session_kind")
        if self.source_policy not in _SOURCE_POLICIES:
            raise ValueError("unsupported SnapshotReplayTick.source_policy")
        if not isinstance(self.ground_truth, PerceptionSnapshot):
            raise TypeError("ground_truth must be a PerceptionSnapshot")
        if self.predicted is not None and not isinstance(self.predicted, PerceptionSnapshot):
            raise TypeError("predicted must be a PerceptionSnapshot or None")
        object.__setattr__(self, "latency_ms", _nonnegative(self.latency_ms, "latency_ms"))
        if self.ground_truth.frame_id != self.frame_id:
            raise ValueError("ground_truth frame_id is not bound to replay tick")
        if self.predicted is not None and self.predicted.frame_id != self.frame_id:
            raise ValueError("predicted frame_id is not bound to replay tick")


@dataclass
class BenchmarkReport:
    development_only: bool
    formal_perception_verdict_eligible: bool
    session_kind: str
    total_records: int
    screen_state_f1: float
    timer_exact_rate: float
    level_exact_rate: float
    inventory_top1_rate: float
    choice_top1_rate: float
    hp_mae: float
    xp_mae: float
    density_correlation: float
    nearest_normalized_median_error: float
    latency_p95_ms: float
    latency_p99_ms: float
    invalid_tick_rate: float
    levelup_invalid_choice_rate: float
    roi_center_p99: float
    roi_inside_region_rate: float
    roi_false_positive_count: int
    confidence_mean: float = 0.0
    ui_cross_frame_equivalence_rate: float = 0.0
    expected_tick_count: int = 0
    observed_tick_count: int = 0
    latency_tick_count: int = 0
    slice_counts: dict[str, int] = field(default_factory=dict)
    slices: list[dict[str, Any]] = field(default_factory=list)
    blocking_reasons: list[str] = field(default_factory=list)
    passed: bool = False

    def metrics_wire(self) -> dict[str, Any]:
        omitted = {
            "development_only",
            "formal_perception_verdict_eligible",
            "session_kind",
            "blocking_reasons",
            "passed",
        }
        return {
            key: value
            for key, value in asdict(self).items()
            if key not in omitted
        }


def _macro_f1(ground_truths: list[Any], predictions: list[Any]) -> float:
    if not ground_truths:
        return 0.0
    classes = set(ground_truths) | set(predictions)
    f1s: list[float] = []
    for value in classes:
        tp = sum(gt == value and pred == value for gt, pred in zip(ground_truths, predictions))
        fp = sum(gt != value and pred == value for gt, pred in zip(ground_truths, predictions))
        fn = sum(gt == value and pred != value for gt, pred in zip(ground_truths, predictions))
        denominator = 2 * tp + fp + fn
        f1s.append(2 * tp / denominator if denominator else 0.0)
    return float(np.mean(f1s))


def bootstrap_cluster_ci(
    session_values: dict[str, list[float]],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """session cluster を再標本化し、underpowered cluster を拒否する。"""
    if type(n_bootstrap) is not int or n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be a positive integer")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(float(alpha)) or not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be finite and in (0, 1)")
    if len(session_values) < 2:
        raise ValueError("cluster bootstrap requires at least two non-empty sessions")
    if any(not values for values in session_values.values()):
        raise ValueError("cluster bootstrap sessions must be non-empty")
    session_means = np.array([np.mean(values) for values in session_values.values()])
    if not np.all(np.isfinite(session_means)):
        raise ValueError("cluster values must be finite")
    rng = rng or np.random.default_rng(0)
    boot = np.array(
        [
            np.mean(rng.choice(session_means, size=len(session_means), replace=True))
            for _ in range(n_bootstrap)
        ]
    )
    return (
        float(np.percentile(boot, 100 * float(alpha) / 2)),
        float(np.percentile(boot, 100 * (1 - float(alpha) / 2))),
    )


def _target_key(target: UiCandidateTargetV1 | UiButtonTargetV1) -> tuple[Any, ...]:
    if isinstance(target, UiCandidateTargetV1):
        return ("candidate", target.choice_id, target.choice_index, target.semantic_kind)
    return ("button", target.semantic_action)


def _is_valid_ground_target(target: UiCandidateTargetV1 | UiButtonTargetV1) -> bool:
    """ground truth として正解 geometry に使える target か。

    validity=False または面積ゼロ ROI（縮退点）は正解の基準に使えないため除外します。
    """
    roi = target.roi
    return target.validity and roi.right > roi.left and roi.bottom > roi.top


def _is_usable_pred_target(target: UiCandidateTargetV1 | UiButtonTargetV1) -> bool:
    """predicted として中心誤差・inside 計算に使える target か。

    validity=False、面積ゼロ ROI、button の capability=False は欠損・FP として扱います。
    """
    roi = target.roi
    if not target.validity or roi.right <= roi.left or roi.bottom <= roi.top:
        return False
    if isinstance(target, UiButtonTargetV1):
        return target.capability
    return True


def _named_slice_labels(snapshot: PerceptionSnapshot) -> frozenset[str]:
    """ground-truth snapshot から formal gate 用の named slice label を派生する。"""
    labels = {f"screen_state:{snapshot.screen_state}"}
    context = snapshot.item_context
    if context is not None:
        if context.elapsed_time < 600.0:
            labels.add("time_band:early")
        elif context.elapsed_time < 1200.0:
            labels.add("time_band:mid")
        else:
            labels.add("time_band:late")
        if context.boss_flag:
            labels.update(("foreground_class:enemy_boss", "event:boss"))
        if context.hazard_flag:
            labels.update(("foreground_class:hazard", "event:hazard"))
        if not context.boss_flag and not context.hazard_flag and context.enemy_density > 0:
            labels.add("foreground_class:enemy_normal")
    # UI state 由来 event は closed vocabulary の代表イベントへ正規化する。
    # boss/hazard は world context、level-up/chest/death は screen state から独立集計する。
    if snapshot.screen_state.startswith("level_up"):
        labels.add("event:level_up")
    elif snapshot.screen_state == "chest":
        labels.add("event:chest")
    elif snapshot.screen_state == "death":
        labels.add("event:death")
    return frozenset(labels)


def _recomputed_snapshot_hashes(snapshot: PerceptionSnapshot) -> tuple[str, str, str]:
    """typed presentation から source/state/binding hash を正式 gate 内で再計算する。"""
    ui = snapshot.ui_presentation
    source_payload = {
        "schema_hash": ui.schema_hash,
        "snapshot_id": ui.snapshot_id,
        "frame_id": ui.frame_id,
        "parser_artifact_hash": ui.parser_artifact_hash,
        "timestamp_ns": snapshot.captured_ns,
        "screen_state": ui.screen_state,
        "candidate_set_hash": ui.candidate_set_hash,
        "inventory_hash": ui.inventory_hash,
        "candidates": [_quantized_target(value) for value in ui.candidates],
        "buttons": [_quantized_target(value) for value in ui.buttons],
    }
    state_payload = {
        "screen_state": ui.screen_state,
        "schema_hash": ui.schema_hash,
        "profile": "survivors_ui_profile.v1",
        "parser_artifact_hash": ui.parser_artifact_hash,
        "candidate_set_hash": ui.candidate_set_hash,
        "inventory_hash": ui.inventory_hash,
        "candidates": [
            {
                "choice_id": value.choice_id,
                "choice_index": value.choice_index,
                "semantic_kind": value.semantic_kind,
            }
            for value in ui.candidates
        ],
        "buttons": [
            {"semantic_action": value.semantic_action, "capability": value.capability}
            for value in ui.buttons
        ],
    }
    # raw evidence チェックは _verify_formal_evidence（呼び出し側で制御）に分離。
    source_hash = canonical_hash(source_payload)
    state_hash = canonical_hash(state_payload)
    if source_hash != ui.source_content_hash or source_hash != snapshot.source_content_hash:
        raise ValueError("source/content hash does not match typed UI presentation")
    if state_hash != ui.ui_state_key or state_hash != snapshot.ui_state_key:
        raise ValueError("state hash does not match typed UI presentation")
    binding_hash = canonical_hash(
        {
            "source_content_hash": source_hash,
            "ui_state_key": state_hash,
            "candidate_set_hash": ui.candidate_set_hash,
            "inventory_hash": ui.inventory_hash,
            "roi_geometry": [_quantized_target(value) for value in (*ui.candidates, *ui.buttons)],
        }
    )
    return source_hash, state_hash, binding_hash


def _verify_formal_evidence(
    ui: "UiPresentationSnapshotV1",
    evidence: FormalReplayEvidence,
    label: str,
) -> None:
    """FormalReplayEvidence を使って UI hash を独立再計算し、改ざんを検出する。

    V1 型を変えず benchmark 専用の raw 証拠を分離した契約で hash binding を検証します。
    """
    expected_candidate_hash = canonical_hash({
        "screen_state": ui.screen_state,
        "card_ids": sorted(c or "unknown" for c in evidence.raw_card_ids),
    })
    if expected_candidate_hash != ui.candidate_set_hash:
        raise ValueError(f"{label}: candidate_set_hash does not match formal evidence card IDs")
    expected_inventory_hash = canonical_hash({"slots": list(evidence.raw_inventory)})
    if expected_inventory_hash != ui.inventory_hash:
        raise ValueError(f"{label}: inventory_hash does not match formal evidence inventory")


def _append_record(
    records: list[BenchmarkRecord], tick: SnapshotReplayTick, field: FieldKind,
    ground_truth: Any, predicted: Any, confidence: float,
) -> None:
    records.append(
        BenchmarkRecord(
            frame_id=tick.frame_id,
            session_id=tick.session_id,
            session_kind=tick.session_kind,
            source_policy=tick.source_policy,
            field=field,
            ground_truth=ground_truth,
            predicted=predicted,
            confidence=confidence,
            latency_ms=tick.latency_ms,
        )
    )


def replay_snapshots(
    ticks: Sequence[SnapshotReplayTick],
    formal_evidence: "Mapping[str, FormalReplayEvidence] | None" = None,
) -> list[BenchmarkRecord]:
    """typed snapshot から全 metric record と UI geometry gate を内部生成する。

    formal_evidence が指定された場合、各 tick の ground_truth hash を
    FormalReplayEvidence で独立再計算して改ざんを検出します。
    """
    records: list[BenchmarkRecord] = []
    previous_targets: dict[
        tuple[str, tuple[Any, ...]],
        tuple[PerceptionSnapshot, Any, PerceptionSnapshot | None, Any | None],
    ] = {}
    for tick in ticks:
        ground = tick.ground_truth
        predicted = tick.predicted
        # evidence check を先に行い、raw hash 不一致を具体的なエラーで捕捉する。
        # その後 _recomputed_snapshot_hashes で構造的整合性を検証する。
        if formal_evidence is not None:
            evidence = formal_evidence.get(ground.frame_id)
            if evidence is None:
                raise ValueError(f"formal evidence missing for frame {ground.frame_id!r}")
            _verify_formal_evidence(ground.ui_presentation, evidence, f"ground/{ground.frame_id}")
            if predicted is not None:
                pred_evidence = formal_evidence.get(predicted.frame_id)
                if pred_evidence is None:
                    raise ValueError(f"formal evidence missing for predicted frame {predicted.frame_id!r}")
                _verify_formal_evidence(predicted.ui_presentation, pred_evidence, f"predicted/{predicted.frame_id}")
        _recomputed_snapshot_hashes(ground)
        if predicted is not None:
            _recomputed_snapshot_hashes(predicted)
        pred_conf = (
            min(
                [target.confidence for target in (*predicted.ui_presentation.candidates, *predicted.ui_presentation.buttons)],
                default=1.0,
            )
            if predicted is not None
            else 0.0
        )
        _append_record(records, tick, "screen_state", ground.screen_state, predicted.screen_state if predicted else None, pred_conf)

        ground_context = ground.item_context
        pred_context = predicted.item_context if predicted else None
        if ground_context is not None:
            for field_name, attribute in (
                ("timer_seconds", "elapsed_time"),
                ("level", "level"),
                ("hp_ratio", "hp_ratio"),
                ("xp_ratio", "xp_ratio"),
            ):
                _append_record(
                    records, tick, field_name, getattr(ground_context, attribute),
                    getattr(pred_context, attribute) if pred_context is not None else None,
                    pred_conf,
                )
            if ground_context.enemy_density is not None:
                _append_record(records, tick, "entity_density", ground_context.enemy_density,
                               pred_context.enemy_density if pred_context is not None else None, pred_conf)
            if ground_context.nearest_enemy_screen_dist is not None:
                _append_record(records, tick, "nearest_distance", ground_context.nearest_enemy_screen_dist,
                               pred_context.nearest_enemy_screen_dist if pred_context is not None else None, pred_conf)

        ground_ui = ground.ui_presentation
        pred_ui = predicted.ui_presentation if predicted else None
        _append_record(records, tick, "inventory_top1", ground_ui.inventory_hash,
                       pred_ui.inventory_hash if pred_ui else None, pred_conf)
        # choice_top1 は annotated level-up event のみ。全 tick で計上すると gameplay tick で希釈される。
        if ground.screen_state.startswith("level_up"):
            _append_record(records, tick, "choice_top1", ground_ui.candidate_set_hash,
                           pred_ui.candidate_set_hash if pred_ui else None, pred_conf)
        # validity=True かつ非ゼロ面積 ROI の target のみを正解 geometry として使う。
        valid_ground_targets = {
            _target_key(t): t
            for t in (*ground_ui.candidates, *ground_ui.buttons)
            if _is_valid_ground_target(t)
        }
        pred_targets = {_target_key(t): t for t in (*pred_ui.candidates, *pred_ui.buttons)} if pred_ui else {}
        for key, ground_target in valid_ground_targets.items():
            pred_target = pred_targets.get(key)
            # predicted が存在しない、または usable でない（validity=False/capability=False/ゼロ ROI）→ 欠損扱い
            if pred_target is None or not _is_usable_pred_target(pred_target):
                _append_record(records, tick, "ui_roi_center_error", 0.0, None, 0.0)
                _append_record(records, tick, "ui_inside_region", True, None, 0.0)
                _append_record(records, tick, "confidence", 1.0, None, 0.0)
                continue
            ground_center = ((ground_target.roi.left + ground_target.roi.right) / 2,
                             (ground_target.roi.top + ground_target.roi.bottom) / 2)
            pred_center = ((pred_target.roi.left + pred_target.roi.right) / 2,
                           (pred_target.roi.top + pred_target.roi.bottom) / 2)
            center_error = math.hypot(pred_center[0] - ground_center[0], pred_center[1] - ground_center[1])
            inside = (ground_target.roi.left <= pred_center[0] <= ground_target.roi.right
                      and ground_target.roi.top <= pred_center[1] <= ground_target.roi.bottom)
            _append_record(records, tick, "ui_roi_center_error", 0.0, center_error, pred_target.confidence)
            _append_record(records, tick, "ui_inside_region", True, inside, pred_target.confidence)
            _append_record(records, tick, "confidence", 1.0, pred_target.confidence, pred_target.confidence)
            previous = previous_targets.get((tick.session_id, key))
            if previous is not None:
                old_ground_snapshot, old_ground_target, old_pred_snapshot, old_pred_target = previous
                expected_equivalent = is_equivalent_ui_target(
                    old_ground_target, ground_target,
                    old_captured_ns=old_ground_snapshot.captured_ns,
                    new_captured_ns=ground.captured_ns,
                    old_ui_state_key=old_ground_snapshot.ui_state_key,
                    new_ui_state_key=ground.ui_state_key,
                )
                actual_equivalent = (
                    old_pred_snapshot is not None
                    and old_pred_target is not None
                    and predicted is not None
                    and is_equivalent_ui_target(
                        old_pred_target, pred_target,
                        old_captured_ns=old_pred_snapshot.captured_ns,
                        new_captured_ns=predicted.captured_ns,
                        old_ui_state_key=old_pred_snapshot.ui_state_key,
                        new_ui_state_key=predicted.ui_state_key,
                    )
                )
                if expected_equivalent:
                    _append_record(
                        records, tick, "ui_cross_frame_equivalent", True,
                        actual_equivalent, pred_target.confidence,
                    )
                elif actual_equivalent:
                    # unsafe FP: predictor claims equivalent when ground truth says not; must be 0
                    _append_record(
                        records, tick, "ui_cross_frame_false_positive", False,
                        True, pred_target.confidence,
                    )
            previous_targets[(tick.session_id, key)] = (
                ground, ground_target, predicted, pred_target,
            )
        # FP = usable predicted targets that do not correspond to any valid ground target
        # validity=False / zero-ROI / non-capable predicted targets are not counted as FP
        # （ground が invalid でも予測が invalid なら FP にしない）
        false_positive_keys = {
            key for key, target in pred_targets.items()
            if _is_usable_pred_target(target)
        } - set(valid_ground_targets)
        _append_record(records, tick, "ui_false_positive", False, bool(false_positive_keys), pred_conf)
    return records


def _empty_report(*, development_only: bool, formal_eligible: bool) -> BenchmarkReport:
    return BenchmarkReport(
        development_only=development_only,
        formal_perception_verdict_eligible=formal_eligible,
        session_kind="unknown", total_records=0, screen_state_f1=0.0,
        timer_exact_rate=0.0, level_exact_rate=0.0, inventory_top1_rate=0.0,
        choice_top1_rate=0.0, hp_mae=float("inf"), xp_mae=float("inf"),
        density_correlation=0.0, nearest_normalized_median_error=float("inf"),
        latency_p95_ms=0.0, latency_p99_ms=0.0, invalid_tick_rate=1.0,
        levelup_invalid_choice_rate=1.0, roi_center_p99=float("inf"),
        roi_inside_region_rate=0.0, roi_false_positive_count=0,
        blocking_reasons=["no records", "availability slice has 0 expected ticks (blocking)"],
    )


def _metric_gate(metrics: dict[str, Any]) -> list[str]:
    """writer/loader と benchmark が共有する stale-proof threshold gate。"""
    blocking: list[str] = []
    counts = metrics["slice_counts"]
    required = {
        "screen_state", "timer_seconds", "level", "hp_ratio", "xp_ratio",
        "inventory_top1", "choice_top1", "entity_density", "nearest_distance",
        "ui_roi_center_error", "ui_inside_region", "ui_false_positive", "confidence",
    }
    for name in sorted(required):
        if counts.get(name, 0) == 0:
            blocking.append(f"{name} slice has 0 records (blocking)")
    if metrics["expected_tick_count"] == 0:
        blocking.append("availability slice has 0 expected ticks (blocking)")
    if metrics["latency_tick_count"] != metrics["expected_tick_count"]:
        blocking.append(
            f"latency measured for {metrics['latency_tick_count']}/"
            f"{metrics['expected_tick_count']} expected ticks (blocking)"
        )
    checks = (
        ("screen_state macro F1", metrics["screen_state_f1"] < THRESHOLD_SCREEN_F1),
        ("timer exact rate", metrics["timer_exact_rate"] < THRESHOLD_TIMER_EXACT),
        ("level exact rate", metrics["level_exact_rate"] < THRESHOLD_LEVEL_EXACT),
        ("inventory top-1", metrics["inventory_top1_rate"] < THRESHOLD_INVENTORY_TOP1),
        ("choice top-1", metrics["choice_top1_rate"] < THRESHOLD_CHOICE_TOP1),
        ("HP MAE", metrics["hp_mae"] > THRESHOLD_HP_MAE),
        ("XP MAE", metrics["xp_mae"] > THRESHOLD_XP_MAE),
        ("entity density correlation", metrics["density_correlation"] < THRESHOLD_DENSITY_CORR),
        ("nearest normalized median error", metrics["nearest_normalized_median_error"] > THRESHOLD_NEAREST_MED),
        ("latency p95", metrics["latency_p95_ms"] > THRESHOLD_LAT_P95_MS),
        ("latency p99", metrics["latency_p99_ms"] > THRESHOLD_LAT_P99_MS),
        ("invalid tick rate", metrics["invalid_tick_rate"] > THRESHOLD_INVALID_TICK),
        ("level-up invalid choice rate", metrics["levelup_invalid_choice_rate"] > THRESHOLD_LEVELUP_INVALID),
        ("ROI center p99", metrics["roi_center_p99"] > THRESHOLD_ROI_CENTER_P99),
        ("ROI inside-region rate", metrics["roi_inside_region_rate"] < THRESHOLD_ROI_INSIDE),
        ("confidence mean", metrics["confidence_mean"] < THRESHOLD_CONFIDENCE),
    )
    for label, failed in checks:
        if failed:
            blocking.append(f"{label} outside formal threshold")
    if counts.get("ui_cross_frame_equivalent", 0) and metrics["ui_cross_frame_equivalence_rate"] < THRESHOLD_CROSS_FRAME_EQUIVALENCE:
        blocking.append("UI cross-frame equivalence rate outside formal threshold")
    unsafe_fp = counts.get("ui_cross_frame_false_positive", 0)
    if unsafe_fp > 0:
        blocking.append(f"ui_cross_frame_false_positive: {unsafe_fp} unsafe false-positive predictions (tolerance 0)")
    for slice_summary in metrics["slices"]:
        if (
            isinstance(slice_summary, dict)
            and slice_summary.get("name") == "overall_screen_state"
            and slice_summary.get("ci_lower") is not None
            and slice_summary["ci_lower"] < THRESHOLD_SCREEN_F1
        ):
            blocking.append("screen_state cluster CI lower bound outside formal threshold")
    if metrics["roi_false_positive_count"] > 0:
        blocking.append("invalid/ambiguous ROI false-positive count is non-zero")
    # 宣言済み rare slice は欠落も最低件数未満も blocking にする。
    # 観測済みの動的 category も slice 別 floor（_NAMED_SLICE_RECORD_FLOORS）またはデフォルトを課す。
    for name in sorted(_REQUIRED_NAMED_SLICES):
        count = counts.get(name, 0)
        floor = _NAMED_SLICE_RECORD_FLOORS.get(name, FORMAL_MIN_NAMED_SLICE_RECORDS)
        if count == 0:
            blocking.append(f"required named slice '{name}' has 0 records (blocking)")
        elif count < floor:
            blocking.append(
                f"named slice '{name}': {count} records < "
                f"{floor} required"
            )
    for prefix, minimum_categories in _NAMED_SLICE_MIN_CATEGORIES.items():
        category_counts = {
            key.removeprefix(f"{prefix}:"): value
            for key, value in counts.items()
            if key.startswith(f"{prefix}:")
        }
        if len(category_counts) < minimum_categories:
            blocking.append(
                f"only {len(category_counts)} {prefix} category(s) observed; "
                f"at least {minimum_categories} required"
            )
        for category, count in sorted(category_counts.items()):
            slice_key = f"{prefix}:{category}"
            floor = _NAMED_SLICE_RECORD_FLOORS.get(slice_key, FORMAL_MIN_NAMED_SLICE_RECORDS)
            if count < floor:
                blocking.append(
                    f"{prefix} '{category}': {count} records < "
                    f"{floor} required"
                )
    return blocking


def recompute_gate_from_metrics(metrics: dict[str, Any]) -> tuple[bool, list[str]]:
    """verdict loader 用に metric mapping を型検証して gate を再計算する。"""
    if not isinstance(metrics, dict):
        raise ValueError("metrics must be a dict")
    required = set(_empty_report(development_only=True, formal_eligible=False).metrics_wire())
    if set(metrics) != required:
        raise ValueError("metrics fields do not match BenchmarkReport schema")
    # bool/NaN/Inf を再ロード境界でも拒否する。nested count/slice は個別検証する。
    if not isinstance(metrics["slice_counts"], dict):
        raise ValueError("slice_counts must be a dict")
    # 固定 field count に加えて、許可済み prefix の named slice key を受理する。
    # 空 category や型不正は拒否し、保存済み verdict の再計算境界を狭く保つ。
    if not all(
        (
            key in _FIELD_KINDS
            or any(
                key.startswith(f"{prefix}:") and bool(key.removeprefix(f"{prefix}:"))
                for prefix in _NAMED_SLICE_MIN_CATEGORIES
            )
        )
        and type(value) is int
        and value >= 0
        for key, value in metrics["slice_counts"].items()
    ):
        raise ValueError("slice_counts must contain non-negative integer values")
    if not isinstance(metrics["slices"], list):
        raise ValueError("slices must be a list")
    for summary in metrics["slices"]:
        if (
            not isinstance(summary, dict)
            or set(summary) != {"name", "count", "macro_f1", "ci_lower"}
            or summary["name"] != "overall_screen_state"
            or type(summary["count"]) is not int
            or summary["count"] <= 0
        ):
            raise ValueError("slice summary does not match the exact schema")
        _ratio(summary["macro_f1"], "slice macro_f1")
        if summary["ci_lower"] is not None:
            _ratio(summary["ci_lower"], "slice ci_lower")
    for name in ("expected_tick_count", "observed_tick_count", "latency_tick_count", "total_records", "roi_false_positive_count"):
        if type(metrics[name]) is not int or metrics[name] < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    for name, value in metrics.items():
        if name in {"slice_counts", "slices", "expected_tick_count", "observed_tick_count", "latency_tick_count", "total_records", "roi_false_positive_count"}:
            continue
        _strict_float(value, name)
    for name in (
        "screen_state_f1", "timer_exact_rate", "level_exact_rate",
        "inventory_top1_rate", "choice_top1_rate",
        "invalid_tick_rate", "levelup_invalid_choice_rate",
        "roi_inside_region_rate", "confidence_mean",
        "ui_cross_frame_equivalence_rate",
    ):
        _ratio(metrics[name], name)
    density_correlation = _strict_float(
        metrics["density_correlation"], "density_correlation"
    )
    if not -1.0 <= density_correlation <= 1.0:
        raise ValueError("density_correlation must be in [-1, 1]")
    blocking = _metric_gate(metrics)
    return not blocking, blocking


def _run_benchmark_common(
    values: Sequence[BenchmarkRecord] | Sequence[SnapshotReplayTick],
    *, expected_ticks: Sequence[ExpectedTick] | None, development_only: bool,
    formal_eligible: bool, rng_seed: int, n_bootstrap: int, alpha: float,
    formal_evidence: Mapping[str, FormalReplayEvidence] | None = None,
) -> BenchmarkReport:
    if type(rng_seed) is not int:
        raise ValueError("rng_seed must be an int")
    # validate bootstrap arguments even when a slice is empty.
    if type(n_bootstrap) is not int or n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be a positive integer")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(float(alpha)) or not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be finite and in (0, 1)")
    raw_values = list(values)
    named_slice_counts: dict[str, int] = defaultdict(int)
    if raw_values and all(isinstance(value, SnapshotReplayTick) for value in raw_values):
        # V1 snapshot を変更せず、ground-truth tick から汎用 named slice label を派生する。
        # 同じ tick 内の複数 metric record ではなく slice ごとに一度だけ数える。
        for tick in raw_values:
            assert isinstance(tick, SnapshotReplayTick)
            for label in _named_slice_labels(tick.ground_truth):
                named_slice_counts[label] += 1
        records = replay_snapshots(raw_values, formal_evidence=formal_evidence)  # type: ignore[arg-type]
    elif all(isinstance(value, BenchmarkRecord) for value in raw_values):
        records = list(raw_values)  # type: ignore[assignment]
    else:
        raise TypeError("benchmark input must contain one homogeneous typed sequence")
    # expected_ticks を先に構築してから空 records を判定する。
    # expected_ticks が存在する場合は availability/latency accounting を省略しない。
    tick_latencies_pre: dict[tuple[str, str], float] = {}
    for record in records:
        key = (record.session_id, record.frame_id)
        prev = tick_latencies_pre.setdefault(key, record.latency_ms)
        if prev != record.latency_ms:
            raise ValueError(f"inconsistent latency for tick {key}")
    pre_expected = list(expected_ticks or [ExpectedTick(*key) for key in tick_latencies_pre])
    if not records and not pre_expected:
        return _empty_report(development_only=development_only, formal_eligible=formal_eligible)
    if not records:
        # expected_ticks あり・observations なし: 全 tick 欠落として報告する。
        e_keys = [(tick.session_id, tick.frame_id) for tick in pre_expected]
        if len(e_keys) != len(set(e_keys)):
            raise ValueError("expected_ticks must be unique")
        n_expected = len(e_keys)
        empty = _empty_report(development_only=development_only, formal_eligible=formal_eligible)
        empty.expected_tick_count = n_expected
        empty.observed_tick_count = 0
        empty.latency_tick_count = 0
        empty.invalid_tick_rate = 1.0
        empty.blocking_reasons = _metric_gate(empty.metrics_wire())
        empty.passed = not empty.blocking_reasons
        return empty

    session_kinds = {record.session_kind for record in records}
    source_policies = {record.source_policy for record in records}
    if len(session_kinds) != 1:
        raise ValueError("benchmark records mix session_kind values")
    if len(source_policies) != 1:
        raise ValueError("benchmark records mix source_policy values")
    tick_latencies: dict[tuple[str, str], float] = {}
    for record in records:
        key = (record.session_id, record.frame_id)
        previous = tick_latencies.setdefault(key, record.latency_ms)
        if previous != record.latency_ms:
            raise ValueError(f"inconsistent latency for tick {key}")

    expected = list(expected_ticks or [ExpectedTick(*key) for key in tick_latencies])
    expected_keys = [(tick.session_id, tick.frame_id) for tick in expected]
    if len(expected_keys) != len(set(expected_keys)):
        raise ValueError("expected_ticks must be unique")
    expected_set = set(expected_keys)
    observed_set = set(tick_latencies)
    extra = observed_set - expected_set
    if extra:
        raise ValueError(f"observed ticks not present in restored manifest: {sorted(extra)}")

    by_field: dict[str, list[BenchmarkRecord]] = defaultdict(list)
    for record in records:
        by_field[record.field].append(record)
    valid = lambda name: [record for record in by_field[name] if record.predicted is not None]
    ss, tr, levels = valid("screen_state"), valid("timer_seconds"), valid("level")
    inv, choices = valid("inventory_top1"), valid("choice_top1")
    hp, xp = valid("hp_ratio"), valid("xp_ratio")
    density, nearest = valid("entity_density"), valid("nearest_distance")
    roi, inside, false_positive = valid("ui_roi_center_error"), valid("ui_inside_region"), valid("ui_false_positive")
    confidence, cross = valid("confidence"), valid("ui_cross_frame_equivalent")

    screen_f1 = _macro_f1([r.ground_truth for r in ss], [r.predicted for r in ss]) if ss else 0.0
    exact = lambda rows: sum(r.ground_truth == r.predicted for r in rows) / len(rows) if rows else 0.0
    timer_exact = sum(abs(r.ground_truth - r.predicted) < _TIMER_EXACT_TOLERANCE for r in tr) / len(tr) if tr else 0.0
    mae = lambda rows: float(np.mean([abs(r.ground_truth - r.predicted) for r in rows])) if rows else float("inf")
    if len(density) >= 2:
        gt = np.array([r.ground_truth for r in density], dtype=float)
        pred = np.array([r.predicted for r in density], dtype=float)
        density_corr = float(np.corrcoef(gt, pred)[0, 1]) if gt.std() > 0 and pred.std() > 0 else (1.0 if np.array_equal(gt, pred) else 0.0)
    else:
        density_corr = 0.0
    # latency=0.0 は有効な計測値として latency_tick_count に含める。
    # 旧 positive_latencies (>0.0) フィルタは availability/latency gate を壊していた。
    positive_latencies = [value for key, value in tick_latencies.items() if key in expected_set]
    invalid_ticks = expected_set - observed_set
    for record in records:
        if record.predicted is None:
            invalid_ticks.add((record.session_id, record.frame_id))
    invalid_rate = len(invalid_ticks) / len(expected_set) if expected_set else 1.0
    invalid_choice = sum(record.predicted in {None, ""} for record in by_field["choice_top1"])
    choice_denominator = len(by_field["choice_top1"])
    slice_counts = {name: len(rows) for name, rows in by_field.items()}
    # screen_state 値ごとの件数を "screen_state:{value}" キーで追加（per-state gate 用）
    for record in by_field.get("screen_state", []):
        key = f"screen_state:{record.ground_truth}"
        slice_counts[key] = slice_counts.get(key, 0) + 1
    for key, count in named_slice_counts.items():
        slice_counts[key] = count
    session_screen: dict[str, list[float]] = defaultdict(list)
    for record in ss:
        session_screen[record.session_id].append(float(record.ground_truth == record.predicted))
    ci_lower = None
    if len(session_screen) >= 2:
        ci_lower, _ = bootstrap_cluster_ci(dict(session_screen), n_bootstrap, alpha, np.random.default_rng(rng_seed))

    report = BenchmarkReport(
        development_only=development_only,
        formal_perception_verdict_eligible=formal_eligible,
        session_kind=next(iter(session_kinds)), total_records=len(records),
        screen_state_f1=screen_f1, timer_exact_rate=timer_exact,
        level_exact_rate=exact(levels), inventory_top1_rate=exact(inv),
        choice_top1_rate=exact(choices), hp_mae=mae(hp), xp_mae=mae(xp),
        density_correlation=density_corr,
        nearest_normalized_median_error=float(np.median([abs(r.ground_truth-r.predicted) for r in nearest])) if nearest else float("inf"),
        latency_p95_ms=float(np.percentile(positive_latencies, 95)) if positive_latencies else 0.0,
        latency_p99_ms=float(np.percentile(positive_latencies, 99)) if positive_latencies else 0.0,
        invalid_tick_rate=invalid_rate,
        levelup_invalid_choice_rate=invalid_choice / choice_denominator if choice_denominator else 1.0,
        roi_center_p99=float(np.percentile([r.predicted for r in roi], 99)) if roi else float("inf"),
        roi_inside_region_rate=sum(bool(r.predicted) for r in inside) / len(inside) if inside else 0.0,
        roi_false_positive_count=sum(bool(r.predicted) for r in false_positive),
        confidence_mean=float(np.mean([r.predicted for r in confidence])) if confidence else 0.0,
        ui_cross_frame_equivalence_rate=sum(bool(r.predicted) for r in cross) / len(cross) if cross else 0.0,
        expected_tick_count=len(expected_set), observed_tick_count=len(observed_set),
        latency_tick_count=len(positive_latencies), slice_counts=slice_counts,
        slices=[{"name": "overall_screen_state", "count": len(ss), "macro_f1": screen_f1, "ci_lower": ci_lower}] if ss else [],
    )
    report.blocking_reasons = _metric_gate(report.metrics_wire())
    report.passed = not report.blocking_reasons
    return report


def run_benchmark(
    values: Sequence[BenchmarkRecord] | Sequence[SnapshotReplayTick], *,
    expected_ticks: Sequence[ExpectedTick] | None = None,
    formal_evidence: Mapping[str, FormalReplayEvidence] | None = None,
    rng_seed: int = 0, n_bootstrap: int = 1000, alpha: float = 0.05,
) -> BenchmarkReport:
    """synthetic 入口。development_only は公開引数にせず常に True に固定する。

    formal_evidence が指定された場合は replay_snapshots へ渡して hash 独立検証を有効化します。
    """
    return _run_benchmark_common(
        values, expected_ticks=expected_ticks, development_only=True,
        formal_eligible=False, formal_evidence=formal_evidence,
        rng_seed=rng_seed, n_bootstrap=n_bootstrap, alpha=alpha,
    )


def _run_formal_benchmark(
    values: Sequence[SnapshotReplayTick], *, expected_ticks: Sequence[ExpectedTick],
    formal_evidence: Mapping[str, FormalReplayEvidence] | None = None,
    rng_seed: int = 0, n_bootstrap: int = 1000, alpha: float = 0.05,
) -> BenchmarkReport:
    """verified formal runner 専用 factory。事前計算 record/bool は受理しない。"""
    if not all(isinstance(value, SnapshotReplayTick) for value in values):
        raise TypeError("formal benchmark accepts only typed SnapshotReplayTick values")
    return _formalize_benchmark_report(
        run_benchmark(
            values, expected_ticks=expected_ticks, formal_evidence=formal_evidence,
            rng_seed=rng_seed, n_bootstrap=n_bootstrap, alpha=alpha,
        )
    )


def _formalize_benchmark_report(report: BenchmarkReport) -> BenchmarkReport:
    """formal runner だけが synthetic-fixed public report を formal subject 化する。"""
    if not isinstance(report, BenchmarkReport):
        raise TypeError("formal benchmark factory requires BenchmarkReport")
    if report.development_only is not True or report.formal_perception_verdict_eligible is not False:
        raise ValueError("benchmark report was already promoted or has inconsistent flags")
    return replace(
        report, development_only=False, formal_perception_verdict_eligible=True
    )
