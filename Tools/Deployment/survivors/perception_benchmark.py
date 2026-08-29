"""Survivors perception benchmark のメトリクス計算。

synthetic または formal calibration/final の予測レコードから
分類・回帰・レイテンシ・UI ROI・session-cluster CI を集計します。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final, Literal

import numpy as np

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
]

_TIMER_EXACT_TOLERANCE: Final[float] = 0.5  # 秒


@dataclass(frozen=True, slots=True)
class BenchmarkRecord:
    """1 フレーム分の予測・正解ペア。

    field に応じて ground_truth/predicted の型が変わります。
    latency_ms=0 はレイテンシ未計測を示します。
    """

    frame_id: str
    session_id: str
    session_kind: str
    source_policy: str
    field: FieldKind
    ground_truth: Any
    predicted: Any
    confidence: float
    latency_ms: float = 0.0


@dataclass
class BenchmarkReport:
    """run_benchmark() が返す集計結果。

    development_only=True のとき formal verdict には使えません。
    """

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
    slices: list[dict[str, Any]] = field(default_factory=list)
    blocking_reasons: list[str] = field(default_factory=list)
    passed: bool = False


def bootstrap_cluster_ci(
    session_values: dict[str, list[float]],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """セッション単位でクラスター bootstrap CI を計算する。

    セッション平均をブートストラップの単位として再サンプリングし、
    overall 平均で代用せず rare-slice の分散を正確に推定します。
    """
    sessions = list(session_values.keys())
    if not sessions:
        return (0.0, 0.0)
    rng = rng or np.random.default_rng(0)
    session_means = np.array([np.mean(session_values[s]) for s in sessions])
    boot = np.array(
        [
            np.mean(rng.choice(session_means, size=len(session_means), replace=True))
            for _ in range(n_bootstrap)
        ]
    )
    lo = float(np.percentile(boot, 100 * alpha / 2))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return (lo, hi)


def run_benchmark(
    records: list[BenchmarkRecord],
    *,
    development_only: bool = True,
) -> BenchmarkReport:
    """予測レコードから benchmark メトリクスを集計する。

    formal 入力なしで呼ばれるとき development_only=True を設定し、
    formal_perception_verdict_eligible を常に False にします。
    """
    if not records:
        return BenchmarkReport(
            development_only=development_only,
            formal_perception_verdict_eligible=False,
            session_kind="unknown",
            total_records=0,
            screen_state_f1=0.0,
            timer_exact_rate=0.0,
            level_exact_rate=0.0,
            inventory_top1_rate=0.0,
            choice_top1_rate=0.0,
            hp_mae=float("inf"),
            xp_mae=float("inf"),
            density_correlation=0.0,
            nearest_normalized_median_error=float("inf"),
            latency_p95_ms=0.0,
            latency_p99_ms=0.0,
            invalid_tick_rate=1.0,
            levelup_invalid_choice_rate=0.0,
            roi_center_p99=float("inf"),
            roi_inside_region_rate=0.0,
            roi_false_positive_count=0,
            blocking_reasons=["no records"],
            passed=False,
        )

    by_field: dict[str, list[BenchmarkRecord]] = {}
    for r in records:
        by_field.setdefault(r.field, []).append(r)

    session_kind = records[0].session_kind

    # --- 分類メトリクス ---
    ss = by_field.get("screen_state", [])
    screen_f1 = (
        sum(1 for r in ss if r.ground_truth == r.predicted) / len(ss) if ss else 0.0
    )

    tr = by_field.get("timer_seconds", [])
    timer_exact = (
        sum(
            1 for r in tr if abs(r.ground_truth - r.predicted) < _TIMER_EXACT_TOLERANCE
        )
        / len(tr)
        if tr
        else 0.0
    )

    lr = by_field.get("level", [])
    level_exact = (
        sum(1 for r in lr if r.ground_truth == r.predicted) / len(lr) if lr else 0.0
    )

    inv = by_field.get("inventory_top1", [])
    inv_rate = (
        sum(1 for r in inv if r.ground_truth == r.predicted) / len(inv) if inv else 0.0
    )

    ch = by_field.get("choice_top1", [])
    choice_rate = (
        sum(1 for r in ch if r.ground_truth == r.predicted) / len(ch) if ch else 0.0
    )

    # --- 回帰メトリクス ---
    hpr = by_field.get("hp_ratio", [])
    hp_mae = (
        float(np.mean([abs(r.ground_truth - r.predicted) for r in hpr]))
        if hpr
        else float("inf")
    )

    xpr = by_field.get("xp_ratio", [])
    xp_mae = (
        float(np.mean([abs(r.ground_truth - r.predicted) for r in xpr]))
        if xpr
        else float("inf")
    )

    # --- world entity メトリクス ---
    dr = by_field.get("entity_density", [])
    if len(dr) >= 2:
        gt_v = np.array([r.ground_truth for r in dr], dtype=float)
        pred_v = np.array([r.predicted for r in dr], dtype=float)
        density_corr = (
            float(np.corrcoef(gt_v, pred_v)[0, 1])
            if gt_v.std() > 0 and pred_v.std() > 0
            else 0.0
        )
    else:
        density_corr = 0.0

    nr = by_field.get("nearest_distance", [])
    nearest_med = (
        float(np.median([abs(r.ground_truth - r.predicted) for r in nr]))
        if nr
        else float("inf")
    )

    # --- レイテンシ ---
    all_lat = [r.latency_ms for r in records if r.latency_ms > 0]
    lat_p95 = float(np.percentile(all_lat, 95)) if all_lat else 0.0
    lat_p99 = float(np.percentile(all_lat, 99)) if all_lat else 0.0

    # --- 可用性 ---
    invalid_ticks = sum(1 for r in records if r.predicted is None)
    invalid_tick_rate = invalid_ticks / len(records)

    levelup_ch = [r for r in ch if r.ground_truth is not None]
    levelup_invalid = sum(
        1 for r in levelup_ch if r.predicted is None or r.predicted == ""
    )
    levelup_invalid_rate = levelup_invalid / len(levelup_ch) if levelup_ch else 0.0

    # --- UI ROI ---
    roi_cr = by_field.get("ui_roi_center_error", [])
    roi_p99 = (
        float(np.percentile([r.predicted for r in roi_cr], 99))
        if roi_cr
        else float("inf")
    )

    inside_r = by_field.get("ui_inside_region", [])
    inside_rate = (
        sum(1 for r in inside_r if r.predicted) / len(inside_r) if inside_r else 0.0
    )

    fp_r = by_field.get("ui_false_positive", [])
    fp_count = sum(1 for r in fp_r if r.predicted)

    return BenchmarkReport(
        development_only=development_only,
        formal_perception_verdict_eligible=False,
        session_kind=session_kind,
        total_records=len(records),
        screen_state_f1=screen_f1,
        timer_exact_rate=timer_exact,
        level_exact_rate=level_exact,
        inventory_top1_rate=inv_rate,
        choice_top1_rate=choice_rate,
        hp_mae=hp_mae,
        xp_mae=xp_mae,
        density_correlation=density_corr,
        nearest_normalized_median_error=nearest_med,
        latency_p95_ms=lat_p95,
        latency_p99_ms=lat_p99,
        invalid_tick_rate=invalid_tick_rate,
        levelup_invalid_choice_rate=levelup_invalid_rate,
        roi_center_p99=roi_p99,
        roi_inside_region_rate=inside_rate,
        roi_false_positive_count=fp_count,
    )
