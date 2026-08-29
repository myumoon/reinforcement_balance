"""Survivors perception benchmark のメトリクス計算。

synthetic または formal calibration/final の予測レコードから
分類・回帰・レイテンシ・UI ROI・session-cluster CI を集計します。

## 設計方針

- screen_state_f1 は macro F1（accuracy ではない）
- availability は unique (session_id, frame_id) を分母にする
- latency はパイプライン tick ごとに 1 回だけ集計する
- 全入力で NaN/Inf を入口検証し、metric/report へ残さない
- rare slice は session-cluster bootstrap 95% CI lower bound で判定する
- overall 平均で rare slice を代用しない
"""

from __future__ import annotations

import math
from collections import defaultdict
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

# デフォルト合格閾値（plan 04-10 End-to-end判定から）
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


def _assert_finite(value: float, name: str) -> float:
    """NaN/Inf を拒否して値を返す。"""
    if not math.isfinite(value):
        raise ValueError(f"BenchmarkRecord.{name} must be finite, got {value!r}")
    return value


@dataclass(frozen=True, slots=True)
class BenchmarkRecord:
    """1 フレーム分の予測・正解ペア。

    field に応じて ground_truth/predicted の型が変わります。
    latency_ms は必ずパイプライン tick ごとに 1 回だけ設定してください。
    latency_ms=0 はレイテンシ未計測を示します。負の値は拒否します。
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

    def __post_init__(self) -> None:
        """float 型フィールドの NaN/Inf と負の latency を拒否する。"""
        if not math.isfinite(self.confidence):
            raise ValueError(f"confidence must be finite, got {self.confidence!r}")
        if not math.isfinite(self.latency_ms):
            raise ValueError(f"latency_ms must be finite, got {self.latency_ms!r}")
        if self.latency_ms < 0:
            raise ValueError(f"latency_ms must be non-negative, got {self.latency_ms!r}")


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


def _macro_f1(ground_truths: list[Any], predictions: list[Any]) -> float:
    """macro F1 を計算する（accuracy ではない）。

    全クラスの per-class F1 の単純平均です。クラス数が少ない
    rare class も等しく扱われます。
    """
    if not ground_truths:
        return 0.0
    classes = set(ground_truths) | set(predictions)
    f1s: list[float] = []
    for c in classes:
        tp = sum(1 for gt, pr in zip(ground_truths, predictions) if gt == c and pr == c)
        fp = sum(1 for gt, pr in zip(ground_truths, predictions) if gt != c and pr == c)
        fn = sum(1 for gt, pr in zip(ground_truths, predictions) if gt == c and pr != c)
        denom = 2 * tp + fp + fn
        f1s.append((2 * tp / denom) if denom > 0 else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def bootstrap_cluster_ci(
    session_values: dict[str, list[float]],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """セッション単位でクラスター bootstrap CI を計算する。

    セッション平均をブートストラップの単位として再サンプリングし、
    overall 平均で代用せず rare-slice の分散を正確に推定します。
    フレームを独立標本として扱わず、session cluster 全体を再標本化します。
    """
    sessions = list(session_values.keys())
    if not sessions:
        return (0.0, 0.0)
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
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


def _percentile_exact(values: list[float], p: float) -> float:
    """np.percentile と同じ線形補間で p パーセンタイルを返す。

    boundary: p=0 → min, p=100 → max。
    """
    if not values:
        return 0.0
    return float(np.percentile(values, p))


def run_benchmark(
    records: list[BenchmarkRecord],
    *,
    development_only: bool = True,
    rng_seed: int = 0,
    n_bootstrap: int = 1000,
) -> BenchmarkReport:
    """予測レコードから benchmark メトリクスを集計する。

    formal 入力なしで呼ばれるとき development_only=True を設定し、
    formal_perception_verdict_eligible を常に False にします。

    random seed を固定（rng_seed=0）すると byte-identical な report になります。
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
    rng = np.random.default_rng(rng_seed)

    # --- 分類メトリクス (macro F1) ---
    ss = by_field.get("screen_state", [])
    screen_f1 = _macro_f1(
        [r.ground_truth for r in ss],
        [r.predicted for r in ss],
    ) if ss else 0.0

    tr = by_field.get("timer_seconds", [])
    timer_exact = (
        sum(1 for r in tr if abs(r.ground_truth - r.predicted) < _TIMER_EXACT_TOLERANCE)
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
    hpr = [r for r in by_field.get("hp_ratio", []) if r.predicted is not None]
    hp_mae = (
        float(np.mean([abs(r.ground_truth - r.predicted) for r in hpr]))
        if hpr
        else float("inf")
    )

    xpr = [r for r in by_field.get("xp_ratio", []) if r.predicted is not None]
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
        if gt_v.ndim != 1 or pred_v.ndim != 1:
            raise ValueError(
                "entity_density ground_truth/predicted must be scalar (1D per record)"
            )
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

    # --- レイテンシ（unique tick ごとに 1 回のみ集計）---
    # (session_id, frame_id) の一意 tick ごとに latency_ms を取る
    tick_latencies: dict[tuple[str, str], float] = {}
    for r in records:
        if r.latency_ms > 0:
            key = (r.session_id, r.frame_id)
            if key not in tick_latencies:
                tick_latencies[key] = r.latency_ms
    all_lat = list(tick_latencies.values())
    lat_p95 = _percentile_exact(all_lat, 95) if all_lat else 0.0
    lat_p99 = _percentile_exact(all_lat, 99) if all_lat else 0.0

    # --- 可用性（unique tick を分母にする）---
    # unique (session_id, frame_id) が分母
    unique_ticks: set[tuple[str, str]] = {(r.session_id, r.frame_id) for r in records}
    invalid_ticks: set[tuple[str, str]] = set()
    # required フィールドが 1 つでも invalid (predicted=None) の tick をカウント
    for r in records:
        if r.predicted is None:
            invalid_ticks.add((r.session_id, r.frame_id))
    invalid_tick_rate = len(invalid_ticks) / len(unique_ticks) if unique_ticks else 0.0

    levelup_ch = [r for r in ch if r.ground_truth is not None]
    levelup_invalid = sum(
        1 for r in levelup_ch if r.predicted is None or r.predicted == ""
    )
    levelup_invalid_rate = levelup_invalid / len(levelup_ch) if levelup_ch else 0.0

    # --- UI ROI ---
    roi_cr = by_field.get("ui_roi_center_error", [])
    roi_p99 = (
        _percentile_exact([r.predicted for r in roi_cr], 99)
        if roi_cr
        else float("inf")
    )

    inside_r = by_field.get("ui_inside_region", [])
    inside_rate = (
        sum(1 for r in inside_r if r.predicted) / len(inside_r) if inside_r else 0.0
    )

    fp_r = by_field.get("ui_false_positive", [])
    fp_count = sum(1 for r in fp_r if r.predicted)

    # --- rare slice CI（session-cluster bootstrap）---
    # session ごとの screen_state 一致率を集計して CI を計算
    session_screen_correct: dict[str, list[float]] = defaultdict(list)
    for r in ss:
        session_screen_correct[r.session_id].append(
            1.0 if r.ground_truth == r.predicted else 0.0
        )
    rare_ci_lo: float | None = None
    if session_screen_correct:
        rare_ci_lo, _ = bootstrap_cluster_ci(
            dict(session_screen_correct),
            n_bootstrap=n_bootstrap,
            rng=rng,
        )

    # --- threshold gate → passed / blocking_reasons ---
    blocking: list[str] = []

    if ss and screen_f1 < THRESHOLD_SCREEN_F1:
        blocking.append(f"screen_state macro F1 {screen_f1:.4f} < {THRESHOLD_SCREEN_F1}")
    if tr and timer_exact < THRESHOLD_TIMER_EXACT:
        blocking.append(f"timer exact rate {timer_exact:.4f} < {THRESHOLD_TIMER_EXACT}")
    if lr and level_exact < THRESHOLD_LEVEL_EXACT:
        blocking.append(f"level exact rate {level_exact:.4f} < {THRESHOLD_LEVEL_EXACT}")
    if inv and inv_rate < THRESHOLD_INVENTORY_TOP1:
        blocking.append(f"inventory top-1 {inv_rate:.4f} < {THRESHOLD_INVENTORY_TOP1}")
    if ch and choice_rate < THRESHOLD_CHOICE_TOP1:
        blocking.append(f"choice top-1 {choice_rate:.4f} < {THRESHOLD_CHOICE_TOP1}")
    if hpr and hp_mae > THRESHOLD_HP_MAE:
        blocking.append(f"HP MAE {hp_mae:.4f} > {THRESHOLD_HP_MAE}")
    if xpr and xp_mae > THRESHOLD_XP_MAE:
        blocking.append(f"XP MAE {xp_mae:.4f} > {THRESHOLD_XP_MAE}")
    if dr and density_corr < THRESHOLD_DENSITY_CORR:
        blocking.append(
            f"entity density correlation {density_corr:.4f} < {THRESHOLD_DENSITY_CORR}"
        )
    if nr and nearest_med > THRESHOLD_NEAREST_MED:
        blocking.append(
            f"nearest normalized median error {nearest_med:.4f} > {THRESHOLD_NEAREST_MED}"
        )
    if all_lat:
        if lat_p95 > THRESHOLD_LAT_P95_MS:
            blocking.append(f"latency p95 {lat_p95:.1f}ms > {THRESHOLD_LAT_P95_MS}ms")
        if lat_p99 > THRESHOLD_LAT_P99_MS:
            blocking.append(f"latency p99 {lat_p99:.1f}ms > {THRESHOLD_LAT_P99_MS}ms")
    if invalid_tick_rate > THRESHOLD_INVALID_TICK:
        blocking.append(
            f"invalid tick rate {invalid_tick_rate:.4f} > {THRESHOLD_INVALID_TICK}"
        )
    if levelup_ch and levelup_invalid_rate > THRESHOLD_LEVELUP_INVALID:
        blocking.append(
            f"level-up invalid choice rate {levelup_invalid_rate:.4f} "
            f"> {THRESHOLD_LEVELUP_INVALID}"
        )
    if roi_cr and roi_p99 > THRESHOLD_ROI_CENTER_P99:
        blocking.append(f"ROI center p99 {roi_p99:.4f} > {THRESHOLD_ROI_CENTER_P99}")
    if inside_r and inside_rate < THRESHOLD_ROI_INSIDE:
        blocking.append(
            f"ROI inside-region rate {inside_rate:.4f} < {THRESHOLD_ROI_INSIDE}"
        )
    if fp_count > 0:
        blocking.append(f"invalid/ambiguous ROI false-positive count: {fp_count}")

    # rare slice: CI lower bound チェック
    if rare_ci_lo is not None:
        if rare_ci_lo < THRESHOLD_SCREEN_F1:
            blocking.append(
                f"screen_state cluster CI lower bound {rare_ci_lo:.4f} < {THRESHOLD_SCREEN_F1}"
            )

    # rare slice 0 件チェック
    if not ss:
        blocking.append("screen_state slice has 0 records (blocking)")

    # slices サマリー
    slices: list[dict[str, Any]] = []
    if ss:
        slices.append(
            {
                "name": "overall_screen_state",
                "count": len(ss),
                "macro_f1": screen_f1,
                "ci_lower": rare_ci_lo,
            }
        )

    passed = len(blocking) == 0

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
        slices=slices,
        blocking_reasons=blocking,
        passed=passed,
    )
