"""perception_benchmark のメトリクス計算テスト。

synthetic predictions から分類・回帰・レイテンシ・UI ROI・cluster CI を検証します。
全結果は development_only=True、formal verdict 不可です。
"""

from __future__ import annotations

import numpy as np
import pytest
from survivors.perception_benchmark import (
    BenchmarkRecord,
    BenchmarkReport,
    bootstrap_cluster_ci,
    run_benchmark,
)


def _rec(
    field: str,
    gt: object,
    pred: object,
    *,
    session: str = "s0",
    kind: str = "calibration",
    confidence: float = 0.9,
    latency_ms: float = 10.0,
) -> BenchmarkRecord:
    return BenchmarkRecord(
        frame_id="f0",
        session_id=session,
        session_kind=kind,
        source_policy="raw",
        field=field,  # type: ignore[arg-type]
        ground_truth=gt,
        predicted=pred,
        confidence=confidence,
        latency_ms=latency_ms,
    )


class TestScreenStateMetric:
    """screen_state F1（ここでは accuracy proxy）の計算を確認する。"""

    def test_perfect_f1(self) -> None:
        records = [_rec("screen_state", "gameplay", "gameplay") for _ in range(10)]
        report = run_benchmark(records)
        assert report.screen_state_f1 == pytest.approx(1.0)

    def test_half_correct(self) -> None:
        records = (
            [_rec("screen_state", "gameplay", "gameplay")] * 5
            + [_rec("screen_state", "gameplay", "unknown")] * 5
        )
        report = run_benchmark(records)
        assert report.screen_state_f1 == pytest.approx(0.5)

    def test_zero_correct(self) -> None:
        records = [_rec("screen_state", "gameplay", "unknown") for _ in range(4)]
        report = run_benchmark(records)
        assert report.screen_state_f1 == pytest.approx(0.0)


class TestTimerAndLevelMetrics:
    """timer/level の exact accuracy を確認する。"""

    def test_timer_exact_perfect(self) -> None:
        records = [_rec("timer_seconds", 120.0, 120.0) for _ in range(5)]
        report = run_benchmark(records)
        assert report.timer_exact_rate == pytest.approx(1.0)

    def test_timer_exact_within_half_second(self) -> None:
        """0.49 秒差は exact とみなす。"""
        records = [_rec("timer_seconds", 100.0, 100.49) for _ in range(4)]
        report = run_benchmark(records)
        assert report.timer_exact_rate == pytest.approx(1.0)

    def test_timer_exact_outside_half_second(self) -> None:
        """0.51 秒差は exact 失敗。"""
        records = [_rec("timer_seconds", 100.0, 100.51) for _ in range(4)]
        report = run_benchmark(records)
        assert report.timer_exact_rate == pytest.approx(0.0)

    def test_level_exact_perfect(self) -> None:
        records = [_rec("level", 5, 5) for _ in range(6)]
        report = run_benchmark(records)
        assert report.level_exact_rate == pytest.approx(1.0)

    def test_level_exact_half(self) -> None:
        records = [_rec("level", 5, 5)] * 3 + [_rec("level", 5, 4)] * 3
        report = run_benchmark(records)
        assert report.level_exact_rate == pytest.approx(0.5)


class TestRegressionMetrics:
    """HP/XP MAE の計算を確認する。"""

    def test_hp_mae_zero_on_perfect(self) -> None:
        records = [_rec("hp_ratio", 0.5, 0.5) for _ in range(5)]
        report = run_benchmark(records)
        assert report.hp_mae == pytest.approx(0.0)

    def test_hp_mae_correct(self) -> None:
        records = [_rec("hp_ratio", 1.0, 0.9) for _ in range(4)]
        report = run_benchmark(records)
        assert report.hp_mae == pytest.approx(0.1, abs=1e-6)

    def test_xp_mae_correct(self) -> None:
        records = [_rec("xp_ratio", 0.6, 0.5) for _ in range(4)]
        report = run_benchmark(records)
        assert report.xp_mae == pytest.approx(0.1, abs=1e-6)


class TestLatencyPercentiles:
    """latency p95/p99 の計算を確認する。"""

    def test_latency_p95_p99(self) -> None:
        records = [
            _rec("screen_state", "gameplay", "gameplay", latency_ms=float(i))
            for i in range(1, 101)
        ]
        report = run_benchmark(records)
        assert report.latency_p95_ms == pytest.approx(95.0, abs=2.0)
        assert report.latency_p99_ms == pytest.approx(99.0, abs=2.0)

    def test_zero_latency_records_yield_zero(self) -> None:
        records = [_rec("screen_state", "g", "g", latency_ms=0.0)]
        report = run_benchmark(records)
        assert report.latency_p95_ms == pytest.approx(0.0)


class TestDevelopmentOnlyFlag:
    """development_only と formal_perception_verdict_eligible を確認する。"""

    def test_development_only_true_by_default(self) -> None:
        records = [_rec("screen_state", "gameplay", "gameplay")]
        report = run_benchmark(records, development_only=True)
        assert report.development_only is True
        assert report.formal_perception_verdict_eligible is False

    def test_formal_eligible_always_false(self) -> None:
        """formal 入力なしでは development_only=False でも eligible は False。"""
        records = [_rec("screen_state", "gameplay", "gameplay")]
        report = run_benchmark(records, development_only=False)
        assert report.formal_perception_verdict_eligible is False


class TestBootstrapClusterCI:
    """session-cluster bootstrap CI を確認する。"""

    def test_perfect_accuracy_ci_near_one(self) -> None:
        session_values = {f"s{i}": [1.0, 1.0, 1.0] for i in range(5)}
        lo, hi = bootstrap_cluster_ci(session_values, n_bootstrap=200, rng=np.random.default_rng(42))
        assert lo == pytest.approx(1.0, abs=1e-6)
        assert hi == pytest.approx(1.0, abs=1e-6)

    def test_ci_lower_le_upper(self) -> None:
        session_values = {f"s{i}": [float(i % 2)] for i in range(10)}
        lo, hi = bootstrap_cluster_ci(session_values, n_bootstrap=200, rng=np.random.default_rng(0))
        assert lo <= hi

    def test_empty_sessions_returns_zero(self) -> None:
        lo, hi = bootstrap_cluster_ci({}, n_bootstrap=100)
        assert lo == 0.0 and hi == 0.0

    def test_single_session_ci_equals_mean(self) -> None:
        """セッションが 1 件のとき lo と hi は同じ値（bootstrap で変化しない）。"""
        session_values = {"only": [0.7, 0.8, 0.9]}
        lo, hi = bootstrap_cluster_ci(session_values, n_bootstrap=100, rng=np.random.default_rng(0))
        assert lo == pytest.approx(hi, abs=1e-6)


class TestInvalidTickRate:
    """predicted=None をインバリッドティックとしてカウントする。"""

    def test_none_prediction_counts_as_invalid(self) -> None:
        records = (
            [_rec("screen_state", "gameplay", None)] * 2
            + [_rec("screen_state", "gameplay", "gameplay")] * 8
        )
        report = run_benchmark(records)
        assert report.invalid_tick_rate == pytest.approx(0.2)

    def test_all_valid(self) -> None:
        records = [_rec("screen_state", "gameplay", "gameplay") for _ in range(5)]
        report = run_benchmark(records)
        assert report.invalid_tick_rate == pytest.approx(0.0)


class TestEmptyRecords:
    """レコードなしのとき失敗扱いになる。"""

    def test_empty_input_fails(self) -> None:
        report = run_benchmark([])
        assert report.passed is False
        assert any("no records" in r for r in report.blocking_reasons)


class TestRoiMetrics:
    """UI ROI center error / inside-region / false-positive を確認する。"""

    def test_roi_center_p99(self) -> None:
        records = [
            _rec("ui_roi_center_error", None, float(i) * 0.001)
            for i in range(1, 101)
        ]
        report = run_benchmark(records)
        assert report.roi_center_p99 == pytest.approx(0.099, abs=0.005)

    def test_roi_inside_region_rate(self) -> None:
        records = (
            [_rec("ui_inside_region", None, True)] * 9
            + [_rec("ui_inside_region", None, False)]
        )
        report = run_benchmark(records)
        assert report.roi_inside_region_rate == pytest.approx(0.9)

    def test_roi_false_positive_zero(self) -> None:
        records = [_rec("ui_false_positive", None, False) for _ in range(5)]
        report = run_benchmark(records)
        assert report.roi_false_positive_count == 0

    def test_roi_false_positive_count(self) -> None:
        records = (
            [_rec("ui_false_positive", None, True)] * 3
            + [_rec("ui_false_positive", None, False)] * 7
        )
        report = run_benchmark(records)
        assert report.roi_false_positive_count == 3


class TestDensityMetrics:
    """entity density correlation / nearest distance median を確認する。"""

    def test_perfect_density_correlation(self) -> None:
        records = [_rec("entity_density", float(i), float(i)) for i in range(1, 10)]
        report = run_benchmark(records)
        assert report.density_correlation == pytest.approx(1.0, abs=1e-6)

    def test_nearest_distance_zero_on_perfect(self) -> None:
        records = [_rec("nearest_distance", 0.1, 0.1) for _ in range(5)]
        report = run_benchmark(records)
        assert report.nearest_normalized_median_error == pytest.approx(0.0)
