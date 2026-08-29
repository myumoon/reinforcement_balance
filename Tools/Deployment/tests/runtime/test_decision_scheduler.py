"""DecisionScheduler: 15 Hz 固定 cadence とバックログ処理を検証する。"""
import pytest
from survivors.runtime.decision_scheduler import DecisionScheduler, DECISION_HZ


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
