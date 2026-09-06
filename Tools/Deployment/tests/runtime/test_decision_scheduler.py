"""DecisionScheduler: 15 Hz 固定 cadence・バックログ・時間ゲートを検証する。

cadence と backlog skip に加えて、snapshot age gate と inference timeout gate が
monotonic 時計で正しく判定されることを確認する。
"""
import pytest
from survivors.runtime.decision_scheduler import (
    DECISION_HZ,
    DEFAULT_INFERENCE_TIMEOUT_NS,
    DEFAULT_MAX_SNAPSHOT_AGE_NS,
    DecisionScheduler,
    DecisionTiming,
)


class TestDecisionScheduler:
    def test_default_hz(self):
        s = DecisionScheduler()
        assert s.hz == DECISION_HZ
        assert s.interval_ns == 1_000_000_000 // DECISION_HZ

    def test_custom_hz(self):
        s = DecisionScheduler(hz=30)
        assert s.hz == 30

    def test_invalid_hz_raises(self):
        with pytest.raises(ValueError, match="hz"):
            DecisionScheduler(hz=0)

    def test_float_hz_raises(self):
        with pytest.raises(ValueError, match="hz"):
            DecisionScheduler(hz=15.0)  # type: ignore[arg-type]

    def test_first_call_always_decides(self):
        s = DecisionScheduler()
        assert s.should_decide(0)
        assert s.should_decide(1_000_000_000)

    def test_within_interval_no_decide(self):
        s = DecisionScheduler()
        now = 0
        s.advance(now)
        assert not s.should_decide(now + 1_000_000)  # 1 ms < 66 ms interval

    def test_after_interval_decides(self):
        s = DecisionScheduler()
        now = 0
        s.advance(now)
        assert s.should_decide(now + s.interval_ns)

    def test_backlog_consumed_on_advance(self):
        """バックログ時 (複数 tick 遅延) は次回 tick を now から再設定する。"""
        s = DecisionScheduler()
        s.advance(0)
        # 5 tick 分バックログ
        late_now = 5 * s.interval_ns
        assert s.should_decide(late_now)
        s.advance(late_now)
        # 次は late_now + interval 後
        assert not s.should_decide(late_now + 1_000_000)
        assert s.should_decide(late_now + s.interval_ns)

    def test_advance_returns_scheduled_ns(self):
        s = DecisionScheduler()
        scheduled = s.advance(1_000_000)
        assert scheduled == 1_000_000  # first call: scheduled = now

    def test_reset_clears_schedule(self):
        s = DecisionScheduler()
        s.advance(0)
        s.reset()
        assert s.should_decide(0)  # first-tick 動作に戻る

    def test_invalid_now_ns_raises(self):
        s = DecisionScheduler()
        with pytest.raises(ValueError, match="now_ns"):
            s.should_decide(-1)

    def test_float_now_ns_raises(self):
        s = DecisionScheduler()
        with pytest.raises(ValueError, match="now_ns"):
            s.should_decide(1.0)  # type: ignore[arg-type]


class TestSnapshotAgeGate:
    """[指摘5] captured_ns の鮮度を monotonic deadline で判定する。"""

    def test_zero_captured_ns_is_not_fresh(self):
        """assembler が timestamp を埋めなかった snapshot を使わせない。"""
        scheduler = DecisionScheduler()
        assert scheduler.is_snapshot_fresh(0, 1_000_000_000) is False

    def test_negative_captured_ns_is_not_fresh(self):
        scheduler = DecisionScheduler()
        assert scheduler.is_snapshot_fresh(-1, 1_000_000_000) is False

    def test_recent_snapshot_is_fresh(self):
        scheduler = DecisionScheduler()
        captured = 1_000_000_000
        assert scheduler.is_snapshot_fresh(captured, captured + 10_000_000) is True

    def test_snapshot_older_than_limit_is_not_fresh(self):
        scheduler = DecisionScheduler()
        captured = 1_000_000_000
        stale_now = captured + DEFAULT_MAX_SNAPSHOT_AGE_NS + 1
        assert scheduler.is_snapshot_fresh(captured, stale_now) is False

    def test_snapshot_at_exact_limit_is_fresh(self):
        scheduler = DecisionScheduler()
        captured = 1_000_000_000
        edge = captured + DEFAULT_MAX_SNAPSHOT_AGE_NS
        assert scheduler.is_snapshot_fresh(captured, edge) is True

    def test_future_snapshot_is_not_fresh(self):
        """時計の逆行で未来 timestamp が来た場合も拒否する。"""
        scheduler = DecisionScheduler()
        captured = 2_000_000_000
        assert scheduler.is_snapshot_fresh(captured, captured - 1) is False

    def test_snapshot_age_ns_is_difference(self):
        scheduler = DecisionScheduler()
        assert scheduler.snapshot_age_ns(100, 350) == 250

    def test_custom_age_limit_is_honoured(self):
        scheduler = DecisionScheduler(max_snapshot_age_ns=5_000_000)
        captured = 1_000_000_000
        assert scheduler.is_snapshot_fresh(captured, captured + 4_000_000) is True
        assert scheduler.is_snapshot_fresh(captured, captured + 6_000_000) is False

    def test_non_positive_age_limit_rejected(self):
        with pytest.raises(ValueError, match="max_snapshot_age_ns"):
            DecisionScheduler(max_snapshot_age_ns=0)


class TestInferenceTimeoutGate:
    """[指摘5] 推論の締切超過を検出する。"""

    def test_fast_inference_is_within_timeout(self):
        scheduler = DecisionScheduler()
        assert scheduler.exceeded_inference_timeout(0, 1_000_000) is False

    def test_slow_inference_exceeds_timeout(self):
        scheduler = DecisionScheduler()
        assert (
            scheduler.exceeded_inference_timeout(0, DEFAULT_INFERENCE_TIMEOUT_NS + 1)
            is True
        )

    def test_exact_timeout_is_not_exceeded(self):
        scheduler = DecisionScheduler()
        assert (
            scheduler.exceeded_inference_timeout(0, DEFAULT_INFERENCE_TIMEOUT_NS) is False
        )

    def test_custom_timeout_is_honoured(self):
        scheduler = DecisionScheduler(inference_timeout_ns=1_000)
        assert scheduler.exceeded_inference_timeout(0, 1_001) is True

    def test_non_positive_timeout_rejected(self):
        with pytest.raises(ValueError, match="inference_timeout_ns"):
            DecisionScheduler(inference_timeout_ns=-1)


class TestBacklogAccounting:
    """[指摘7] backlog で古い tick を skip し、件数を telemetry へ残す。"""

    def test_backlog_skips_are_counted(self):
        scheduler = DecisionScheduler()
        scheduler.advance(0)
        # 5 interval ぶん遅れて呼ばれた場合、間の tick は捨てられる。
        scheduler.advance(scheduler.interval_ns * 5)
        assert scheduler.skipped_tick_count >= 4

    def test_on_cadence_calls_skip_nothing(self):
        scheduler = DecisionScheduler()
        now = 0
        scheduler.advance(now)
        for _ in range(5):
            now += scheduler.interval_ns
            assert scheduler.should_decide(now) is True
            scheduler.advance(now)
        assert scheduler.skipped_tick_count == 0

    def test_decision_count_tracks_advances(self):
        scheduler = DecisionScheduler()
        for index in range(4):
            scheduler.advance(index * scheduler.interval_ns)
        assert scheduler.decision_count == 4

    def test_thirty_minute_schedule_holds_cadence(self):
        """30 分 (27000 tick) 相当を回しても cadence が崩れない。"""
        scheduler = DecisionScheduler()
        ticks = DECISION_HZ * 60 * 30
        now = 0
        scheduled: list[int] = []
        for _ in range(ticks):
            assert scheduler.should_decide(now) is True
            scheduled.append(scheduler.advance(now))
            now += scheduler.interval_ns
        assert scheduler.decision_count == ticks
        assert scheduler.skipped_tick_count == 0
        gaps = {b - a for a, b in zip(scheduled, scheduled[1:])}
        assert gaps == {scheduler.interval_ns}
        elapsed_ns = scheduled[-1] - scheduled[0]
        # 27000 tick でおよそ 30 分 (許容 1 tick)。
        assert abs(elapsed_ns - (ticks - 1) * scheduler.interval_ns) <= scheduler.interval_ns


class TestDecisionTiming:
    """DecisionTiming の latency 計算を確認する。"""

    def test_inference_latency_ms(self):
        timing = DecisionTiming("d", 0, 1_000_000, 4_000_000)
        assert timing.inference_latency_ms == pytest.approx(3.0)

    def test_wall_latency_ms(self):
        timing = DecisionTiming("d", 0, 1_000_000, 4_000_000)
        assert timing.wall_latency_ms == pytest.approx(4.0)
