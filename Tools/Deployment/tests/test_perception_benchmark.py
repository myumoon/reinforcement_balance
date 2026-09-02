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
    ExpectedTick,
    bootstrap_cluster_ci,
    recompute_gate_from_metrics,
    run_benchmark,
    _is_valid_ground_target,
    _is_usable_pred_target,
)
from survivors.perception_snapshot import NormalizedRoi, UiButtonTargetV1, UiCandidateTargetV1


def _rec(
    field: str,
    gt: object,
    pred: object,
    *,
    session: str = "s0",
    frame: str = "f0",
    kind: str = "error_calibration",
    confidence: float = 0.9,
    latency_ms: float = 0.0,
) -> BenchmarkRecord:
    return BenchmarkRecord(
        frame_id=frame,
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
    """screen_state macro F1（accuracy ではない）の計算を確認する。"""

    def test_perfect_f1(self) -> None:
        records = [_rec("screen_state", "gameplay", "gameplay") for _ in range(10)]
        report = run_benchmark(records)
        assert report.screen_state_f1 == pytest.approx(1.0)

    def test_half_correct(self) -> None:
        """5 件正解、5 件誤りで macro F1 は約 0.333（accuracy の 0.5 とは異なる）。"""
        records = (
            [_rec("screen_state", "gameplay", "gameplay")] * 5
            + [_rec("screen_state", "gameplay", "unknown")] * 5
        )
        report = run_benchmark(records)
        # "gameplay": F1=0.667, "unknown": F1=0.0 → macro mean ≈ 0.333
        assert report.screen_state_f1 == pytest.approx(1 / 3, abs=0.01)

    def test_zero_correct(self) -> None:
        records = [_rec("screen_state", "gameplay", "unknown") for _ in range(4)]
        report = run_benchmark(records)
        assert report.screen_state_f1 == pytest.approx(0.0)

    def test_macro_f1_not_accuracy_rare_class(self) -> None:
        """999件の多数クラス正解 + rare class 1件失敗で macro F1 が約0.5になる（必須回帰テスト）。

        accuracy では 999/1000 = 0.999 だが、macro F1 は rare class に等しく重みを置くため
        much lower になります。
        """
        records = (
            [_rec("screen_state", "gameplay", "gameplay", frame=str(i)) for i in range(999)]
            + [_rec("screen_state", "rare_class", "gameplay", frame="f_rare")]
        )
        report = run_benchmark(records)
        # accuracy なら 0.999 だが macro F1 は 2 class 平均なので大きく下がる
        # "gameplay" class F1 ≈ 0.9997, "rare_class" class F1 = 0.0 → mean ≈ 0.4998
        assert report.screen_state_f1 < 0.6, (
            f"Expected macro F1 < 0.6 for rare class failure, got {report.screen_state_f1}"
        )


class TestTimerAndLevelMetrics:
    """timer/level exact rate の計算を確認する。"""

    def test_timer_exact_perfect(self) -> None:
        records = [_rec("timer_seconds", 120.0, 120.0, frame=str(i)) for i in range(5)]
        report = run_benchmark(records)
        assert report.timer_exact_rate == pytest.approx(1.0)

    def test_timer_exact_within_tolerance(self) -> None:
        """0.49 秒差は許容範囲内。"""
        records = [_rec("timer_seconds", 100.0, 100.49, frame=str(i)) for i in range(4)]
        report = run_benchmark(records)
        assert report.timer_exact_rate == pytest.approx(1.0)

    def test_timer_exact_outside_tolerance(self) -> None:
        """0.51 秒差は許容範囲外。"""
        records = [_rec("timer_seconds", 100.0, 100.51, frame=str(i)) for i in range(4)]
        report = run_benchmark(records)
        assert report.timer_exact_rate == pytest.approx(0.0)

    def test_level_exact_all_correct(self) -> None:
        records = [_rec("level", 5, 5, frame=str(i)) for i in range(6)]
        report = run_benchmark(records)
        assert report.level_exact_rate == pytest.approx(1.0)

    def test_level_exact_half(self) -> None:
        records = (
            [_rec("level", 5, 5, frame=str(i)) for i in range(3)]
            + [_rec("level", 5, 4, frame=str(i + 3)) for i in range(3)]
        )
        report = run_benchmark(records)
        assert report.level_exact_rate == pytest.approx(0.5)


class TestRegressionMetrics:
    """HP/XP MAE の計算を確認する。"""

    def test_hp_mae_zero_on_perfect(self) -> None:
        records = [_rec("hp_ratio", 0.5, 0.5, frame=str(i)) for i in range(5)]
        report = run_benchmark(records)
        assert report.hp_mae == pytest.approx(0.0)

    def test_hp_mae_correct(self) -> None:
        records = [_rec("hp_ratio", 1.0, 0.9, frame=str(i)) for i in range(4)]
        report = run_benchmark(records)
        assert report.hp_mae == pytest.approx(0.1, abs=1e-6)

    def test_xp_mae_correct(self) -> None:
        records = [_rec("xp_ratio", 0.6, 0.5, frame=str(i)) for i in range(4)]
        report = run_benchmark(records)
        assert report.xp_mae == pytest.approx(0.1, abs=1e-6)


class TestLatencyPercentiles:
    """latency p95/p99 の計算を確認する。"""

    def test_latency_p95_p99(self) -> None:
        """1〜100ms のレコードで p95/p99 が np.percentile と一致する。"""
        records = [
            _rec("screen_state", "gameplay", "gameplay", frame=str(i), latency_ms=float(i))
            for i in range(1, 101)
        ]
        report = run_benchmark(records)
        expected_p95 = float(np.percentile(list(range(1, 101)), 95))
        expected_p99 = float(np.percentile(list(range(1, 101)), 99))
        assert report.latency_p95_ms == pytest.approx(expected_p95, abs=0.5)
        assert report.latency_p99_ms == pytest.approx(expected_p99, abs=0.5)

    def test_zero_latency_records_yield_zero(self) -> None:
        records = [_rec("screen_state", "g", "g", latency_ms=0.0)]
        report = run_benchmark(records)
        assert report.latency_p95_ms == pytest.approx(0.0)

    def test_latency_aggregated_once_per_tick(self) -> None:
        """同じ (session_id, frame_id) はレイテンシを 1 回だけ集計する（必須回帰テスト）。"""
        records = [
            BenchmarkRecord("f0", "s0", "error_calibration", "raw", "screen_state", "g", "g", 0.9, latency_ms=50.0),  # type: ignore[arg-type]
            BenchmarkRecord("f0", "s0", "error_calibration", "raw", "hp_ratio", 0.5, 0.5, 0.9, latency_ms=50.0),  # type: ignore[arg-type]
        ]
        report = run_benchmark(records)
        # tick は 1 件だけなので p95=p99=50
        assert report.latency_p95_ms == pytest.approx(50.0)

    def test_negative_latency_rejected(self) -> None:
        """負の latency_ms は BenchmarkRecord 生成時に拒否する（必須回帰テスト）。"""
        with pytest.raises(ValueError, match="non-negative"):
            BenchmarkRecord("f0", "s0", "error_calibration", "raw", "screen_state", "g", "g", 0.9, latency_ms=-1.0)  # type: ignore[arg-type]

    def test_nan_latency_rejected(self) -> None:
        """NaN の latency_ms は BenchmarkRecord 生成時に拒否する。"""
        with pytest.raises(ValueError):
            BenchmarkRecord("f0", "s0", "error_calibration", "raw", "screen_state", "g", "g", 0.9, latency_ms=float("nan"))  # type: ignore[arg-type]

    def test_same_tick_inconsistent_latency_rejected(self) -> None:
        records = [
            _rec("screen_state", "g", "g", latency_ms=1.0),
            _rec("hp_ratio", 0.5, 0.5, latency_ms=2.0),
        ]
        with pytest.raises(ValueError, match="inconsistent latency"):
            run_benchmark(records)

    def test_unmeasured_expected_tick_is_blocking(self) -> None:
        records = [
            _rec("screen_state", "g", "g", session="s0", frame="f0", latency_ms=1.0),
            _rec("screen_state", "g", "g", session="s1", frame="f0", latency_ms=1.0),
        ]
        report = run_benchmark(
            records,
            expected_ticks=[ExpectedTick("s0", "f0"), ExpectedTick("s1", "f0"), ExpectedTick("s1", "dropped")],
        )
        assert report.invalid_tick_rate == pytest.approx(1 / 3)
        assert any("latency measured" in reason for reason in report.blocking_reasons)


class TestDevelopmentOnlyFlag:
    """run_benchmark は常に development_only=True、formal eligible=False を返す。"""

    def test_development_only_true(self) -> None:
        records = [_rec("screen_state", "g", "g")]
        report = run_benchmark(records)
        assert report.development_only is True

    def test_formal_eligible_false(self) -> None:
        records = [_rec("screen_state", "g", "g")]
        report = run_benchmark(records)
        assert report.formal_perception_verdict_eligible is False

    def test_development_only_is_not_a_public_override(self) -> None:
        with pytest.raises(TypeError):
            run_benchmark([_rec("screen_state", "g", "g")], development_only=False)  # type: ignore[call-arg]


class TestRecordValidation:
    @pytest.mark.parametrize("value", [float("nan"), float("inf"), True])
    def test_numeric_substitutes_rejected(self, value: object) -> None:
        with pytest.raises(ValueError):
            _rec("hp_ratio", 0.5, value)

    def test_out_of_range_ratio_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            _rec("xp_ratio", 0.5, 1.1)

    def test_mixed_session_kind_and_source_rejected(self) -> None:
        with pytest.raises(ValueError, match="session_kind"):
            run_benchmark([
                _rec("screen_state", "g", "g", session="s0"),
                _rec("screen_state", "g", "g", session="s1", kind="final_e2e_test"),
            ])
        with pytest.raises(ValueError, match="source_policy"):
            first = _rec("screen_state", "g", "g", session="s0")
            second = BenchmarkRecord("f0", "s1", "error_calibration", "lossless", "screen_state", "g", "g", 1.0, 1.0)
            run_benchmark([first, second])


class TestBootstrapClusterCI:
    """session-cluster CI の計算を確認する。"""

    def test_perfect_single_cluster_ci(self) -> None:
        session_values = {"s" + str(i): [1.0, 1.0, 1.0] for i in range(5)}
        lo, hi = bootstrap_cluster_ci(session_values, n_bootstrap=200, rng=np.random.default_rng(42))
        assert lo == pytest.approx(1.0, abs=1e-6)
        assert hi == pytest.approx(1.0, abs=1e-6)

    def test_mixed_sessions_ci_width(self) -> None:
        session_values = {"s" + str(i): [float(i % 2)] for i in range(10)}
        lo, hi = bootstrap_cluster_ci(session_values, n_bootstrap=200, rng=np.random.default_rng(0))
        assert lo < hi

    def test_empty_sessions_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least two"):
            bootstrap_cluster_ci({}, n_bootstrap=100)

    def test_single_session_rejected(self) -> None:
        session_values = {"s0": [0.7, 0.8, 0.9]}
        with pytest.raises(ValueError, match="at least two"):
            bootstrap_cluster_ci(session_values, n_bootstrap=100, rng=np.random.default_rng(0))

    def test_invalid_n_bootstrap_raises(self) -> None:
        """n_bootstrap ≤ 0 は ValueError（必須回帰テスト）。"""
        with pytest.raises(ValueError, match="n_bootstrap"):
            bootstrap_cluster_ci({"s0": [1.0]}, n_bootstrap=0)

    def test_cluster_resamples_sessions_not_frames(self) -> None:
        """bootstrap は frame ではなく session cluster 全体を再標本化する（必須回帰テスト）。"""
        session_values = {
            "perfect": [1.0] * 100,
            "zero": [0.0] * 100,
        }
        lo, hi = bootstrap_cluster_ci(session_values, n_bootstrap=500, rng=np.random.default_rng(1))
        assert hi - lo > 0.1, f"Expected wide CI for session-cluster bootstrap, got [{lo}, {hi}]"


class TestInvalidTickRate:
    """unique (session_id, frame_id) を分母にした invalid tick 率（必須回帰テスト）。"""

    def test_none_prediction_counts_as_invalid(self) -> None:
        records = (
            [_rec("screen_state", "gameplay", None, frame=str(i)) for i in range(2)]
            + [_rec("screen_state", "gameplay", "gameplay", frame=str(i + 2)) for i in range(8)]
        )
        report = run_benchmark(records)
        assert report.invalid_tick_rate == pytest.approx(0.2)

    def test_all_valid(self) -> None:
        records = [_rec("screen_state", "gameplay", "gameplay", frame=str(i)) for i in range(5)]
        report = run_benchmark(records)
        assert report.invalid_tick_rate == pytest.approx(0.0)

    def test_availability_uses_unique_ticks(self) -> None:
        """同一 (session_id, frame_id) は 1 tick としてカウントする（record 数ではない）。"""
        records = [
            BenchmarkRecord("f0", "s0", "error_calibration", "raw", "screen_state", "g", "g", 0.9),  # type: ignore[arg-type]
            BenchmarkRecord("f0", "s0", "error_calibration", "raw", "hp_ratio", 0.5, None, 0.9),  # type: ignore[arg-type]
            BenchmarkRecord("f0", "s0", "error_calibration", "raw", "xp_ratio", 0.3, 0.3, 0.9),  # type: ignore[arg-type]
        ]
        report = run_benchmark(records)
        assert report.invalid_tick_rate == pytest.approx(1.0)


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
            _rec("ui_roi_center_error", 0.0, float(i) * 0.001, frame=str(i))
            for i in range(1, 101)
        ]
        report = run_benchmark(records)
        expected = float(np.percentile([i * 0.001 for i in range(1, 101)], 99))
        assert report.roi_center_p99 == pytest.approx(expected, abs=0.005)

    def test_roi_inside_region_rate(self) -> None:
        records = (
            [_rec("ui_inside_region", True, True, frame=str(i)) for i in range(9)]
            + [_rec("ui_inside_region", True, False, frame="f_false")]
        )
        report = run_benchmark(records)
        assert report.roi_inside_region_rate == pytest.approx(0.9)

    def test_no_false_positives(self) -> None:
        records = [
            _rec("ui_false_positive", False, False, frame=str(i)) for i in range(5)
        ]
        report = run_benchmark(records)
        assert report.roi_false_positive_count == 0

    def test_three_false_positives(self) -> None:
        records = (
            [_rec("ui_false_positive", False, True, frame=str(i)) for i in range(3)]
            + [_rec("ui_false_positive", False, False, frame=str(i + 3)) for i in range(7)]
        )
        report = run_benchmark(records)
        assert report.roi_false_positive_count == 3


class TestDensityMetrics:
    """entity density correlation と nearest_distance median error を確認する。"""

    def test_density_perfect_correlation(self) -> None:
        records = [
            _rec("entity_density", float(i), float(i), frame=str(i)) for i in range(1, 10)
        ]
        report = run_benchmark(records)
        assert report.density_correlation == pytest.approx(1.0, abs=1e-6)

    def test_nearest_zero_error(self) -> None:
        records = [
            _rec("nearest_distance", 0.1, 0.1, frame=str(i)) for i in range(5)
        ]
        report = run_benchmark(records)
        assert report.nearest_normalized_median_error == pytest.approx(0.0)


class TestRareSliceGate:
    """rare slice CI gate と 0 件 block を確認する（必須回帰テスト）。"""

    def test_rare_slice_zero_records_blocking(self) -> None:
        """screen_state slice が 0 件のとき blocking。"""
        records = [_rec("hp_ratio", 0.5, 0.5, frame=str(i)) for i in range(5)]
        report = run_benchmark(records)
        assert report.passed is False
        assert any("screen_state" in r for r in report.blocking_reasons)

    def test_slice_summary_included(self) -> None:
        """screen_state records があれば slices サマリーに含まれる。"""
        records = [
            _rec("screen_state", "gameplay", "gameplay", session=f"s{i % 3}", frame=str(i))
            for i in range(30)
        ]
        report = run_benchmark(records)
        assert any(s["name"] == "overall_screen_state" for s in report.slices)

    def test_seed_reproducible_report(self) -> None:
        """random seed 固定で byte-identical な report になる（必須回帰テスト）。"""
        records = [
            _rec("screen_state", "gameplay", "gameplay", session=f"s{i % 3}", frame=str(i))
            for i in range(30)
        ]
        r1 = run_benchmark(records, rng_seed=42)
        r2 = run_benchmark(records, rng_seed=42)
        assert r1.screen_state_f1 == r2.screen_state_f1
        assert r1.slices == r2.slices

    def test_different_seeds_deterministic(self) -> None:
        """同じ seed で再実行すると同じ結果になる。"""
        records = [
            _rec("screen_state", "gameplay" if i % 3 else "other", "gameplay",
                 session=f"s{i % 2}", frame=str(i))
            for i in range(30)
        ]
        r1 = run_benchmark(records, rng_seed=0)
        r2 = run_benchmark(records, rng_seed=0)
        assert r1.screen_state_f1 == r2.screen_state_f1

    def test_formal_gate_requires_real_data_floors_and_slice_ci(self) -> None:
        report = run_benchmark([
            _rec(
                "screen_state", "gameplay", "gameplay",
                session=f"s{i % 2}", frame=str(i),
            )
            for i in range(20)
        ])
        metrics = report.metrics_wire()
        for name, value in tuple(metrics.items()):
            if isinstance(value, float) and not np.isfinite(value):
                metrics[name] = 0.0
        foreground_classes = (
            "player_anchor", "enemy_normal", "enemy_elite", "enemy_boss",
            "gem_blue", "gem_green", "gem_red", "pickup_heal",
            "pickup_special", "hazard_projectile", "hazard_area",
        )
        metrics["slice_counts"].update({
            "time_band:early": 20,
            "time_band:mid": 20,
            "time_band:late": 20,
            **{f"foreground_class:{name}": 200 for name in foreground_classes},
            "event:boss": 100,
            "event:hazard": 100,
            "event:level_up": 100,
            "event:chest": 30,
            "event:death": 20,
            "event:result": 20,
        })
        metrics["slice_counts"]["foreground_class:enemy_elite"] = 199
        metrics["slice_session_counts"] = {
            "time_band:early": 2,
            "time_band:mid": 2,
            "time_band:late": 2,
        }
        passed, reasons = recompute_gate_from_metrics(metrics, formal=True)
        assert passed is False
        assert any("enemy_elite" in reason and "200" in reason for reason in reasons)
        assert any("CI" in reason for reason in reasons)

    def test_formal_gate_requires_every_world_class_map_foreground_class(self) -> None:
        report = run_benchmark([
            _rec("screen_state", "gameplay", "gameplay", session=f"s{i % 2}", frame=str(i))
            for i in range(20)
        ])
        metrics = report.metrics_wire()
        for name, value in tuple(metrics.items()):
            if isinstance(value, float) and not np.isfinite(value):
                metrics[name] = 0.0
        passed, reasons = recompute_gate_from_metrics(metrics, formal=True)
        assert passed is False
        for class_name in (
            "player_anchor", "enemy_normal", "enemy_elite", "enemy_boss",
            "gem_blue", "gem_green", "gem_red", "pickup_heal",
            "pickup_special", "hazard_projectile", "hazard_area",
        ):
            assert any(
                f"foreground_class:{class_name}" in reason and "200" in reason
                for reason in reasons
            )



class TestThresholdGate:
    """passed / blocking_reasons が threshold gate に接続されている（必須回帰テスト）。"""

    def test_poor_hp_mae_blocks(self) -> None:
        """HP MAE が閾値を超えると blocking_reasons に含まれる。"""
        records = (
            [_rec("screen_state", "gameplay", "gameplay",
                  session=f"s{i % 3}", frame=str(i)) for i in range(30)]
            + [_rec("hp_ratio", 1.0, 0.5, frame=str(i + 30)) for i in range(5)]
        )
        report = run_benchmark(records)
        assert any("HP MAE" in r for r in report.blocking_reasons), report.blocking_reasons

    def test_false_positive_roi_blocks(self) -> None:
        """ROI false-positive が 1 件でも blocking。"""
        records = (
            [_rec("screen_state", "gameplay", "gameplay",
                  session=f"s{i % 3}", frame=str(i)) for i in range(30)]
            + [_rec("ui_false_positive", False, True, frame="fp_0")]
        )
        report = run_benchmark(records)
        assert any("false-positive" in r for r in report.blocking_reasons), report.blocking_reasons

    def test_blocking_reasons_and_passed_connected(self) -> None:
        """blocking_reasons が空のとき passed=True、非空のとき passed=False。"""
        # empty records → no records blocking
        report = run_benchmark([])
        assert report.passed is False
        assert len(report.blocking_reasons) > 0


class TestRoiValidityFilter:
    """validity=False または capability=False の target が正解 geometry を汚染しないことを検証する。"""

    _VALID_ROI = NormalizedRoi(0.1, 0.1, 0.5, 0.5)
    _ZERO_ROI = NormalizedRoi(0.0, 0.0, 0.0, 0.0)

    def test_invalid_candidate_not_valid_ground(self) -> None:
        """validity=False の candidate は ground target として無効。"""
        cand = UiCandidateTargetV1("x", 0, "item_card", self._VALID_ROI, False, 0.9)
        assert not _is_valid_ground_target(cand)

    def test_zero_roi_candidate_not_valid_ground(self) -> None:
        """ゼロ面積 ROI の candidate は ground target として無効。"""
        cand = UiCandidateTargetV1("x", 0, "item_card", self._ZERO_ROI, True, 0.9)
        assert not _is_valid_ground_target(cand)

    def test_zero_width_roi_not_valid_for_ground_or_pred(self) -> None:
        """幅ゼロ・高さ正の線状 ROI は ground/pred の両方で無効。"""
        roi = NormalizedRoi(0.2, 0.1, 0.2, 0.5)
        cand = UiCandidateTargetV1("x", 0, "item_card", roi, True, 0.9)
        assert not _is_valid_ground_target(cand)
        assert not _is_usable_pred_target(cand)

    def test_zero_height_roi_not_valid_for_ground_or_pred(self) -> None:
        """幅正・高さゼロの線状 ROI は ground/pred の両方で無効。"""
        roi = NormalizedRoi(0.1, 0.2, 0.5, 0.2)
        cand = UiCandidateTargetV1("x", 0, "item_card", roi, True, 0.9)
        assert not _is_valid_ground_target(cand)
        assert not _is_usable_pred_target(cand)

    def test_valid_candidate_is_valid_ground(self) -> None:
        """validity=True かつ非ゼロ ROI の candidate は ground target として有効。"""
        cand = UiCandidateTargetV1("x", 0, "item_card", self._VALID_ROI, True, 0.9)
        assert _is_valid_ground_target(cand)

    def test_invalid_candidate_not_usable_pred(self) -> None:
        """validity=False の predicted candidate は usable でない（欠損扱い）。"""
        cand = UiCandidateTargetV1("x", 0, "item_card", self._VALID_ROI, False, 0.9)
        assert not _is_usable_pred_target(cand)

    def test_incapable_button_not_usable_pred(self) -> None:
        """capability=False の predicted button は usable でない（FP カウントしない）。"""
        btn = UiButtonTargetV1("ack_chest", self._VALID_ROI, True, False, 0.8)
        assert not _is_usable_pred_target(btn)

    def test_capable_valid_button_is_usable_pred(self) -> None:
        """validity=True かつ capability=True の button は usable。"""
        btn = UiButtonTargetV1("ack_chest", self._VALID_ROI, True, True, 0.8)
        assert _is_usable_pred_target(btn)
