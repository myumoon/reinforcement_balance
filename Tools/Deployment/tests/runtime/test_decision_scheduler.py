"""DecisionScheduler: 15 Hz 固定 cadence・バックログ・時間ゲートを検証する。

cadence と backlog skip に加えて、snapshot age gate と inference timeout gate が
monotonic 時計で正しく判定されることを確認する。
"""
import pytest
from reinbalance_survivors_contracts.ui_policy import (
    NonModelUiPolicyConfigV1,
    ScreenState,
    UiPolicyInputV1,
)
from survivors.runtime.decision_scheduler import (
    DECISION_HZ,
    DEFAULT_INFERENCE_TIMEOUT_NS,
    DEFAULT_MAX_SNAPSHOT_AGE_NS,
    DecisionScheduler,
    DecisionTiming,
    ScreenDecisionScheduler,
)


class TestDecisionScheduler:
    """TestDecisionScheduler の契約を検証する。

    やさしい説明: このまとまりが担当する境界を分かりやすく示します。
    """
    def test_default_hz(self):
        """test_default_hz の契約を検証する。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        s = DecisionScheduler()
        assert s.hz == DECISION_HZ
        assert s.interval_ns == 1_000_000_000 // DECISION_HZ

    def test_custom_hz(self):
        """test_custom_hz の契約を検証する。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        s = DecisionScheduler(hz=30)
        assert s.hz == 30

    def test_invalid_hz_raises(self):
        """test_invalid_hz_raises の契約を検証する。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        with pytest.raises(ValueError, match="hz"):
            DecisionScheduler(hz=0)

    def test_float_hz_raises(self):
        """test_float_hz_raises の契約を検証する。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        with pytest.raises(ValueError, match="hz"):
            DecisionScheduler(hz=15.0)  # type: ignore[arg-type]

    def test_first_call_always_decides(self):
        """test_first_call_always_decides の契約を検証する。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        s = DecisionScheduler()
        assert s.should_decide(0)
        assert s.should_decide(1_000_000_000)

    def test_within_interval_no_decide(self):
        """test_within_interval_no_decide の契約を検証する。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        s = DecisionScheduler()
        now = 0
        s.advance(now)
        assert not s.should_decide(now + 1_000_000)  # 1 ms < 66 ms interval

    def test_after_interval_decides(self):
        """test_after_interval_decides の契約を検証する。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        s = DecisionScheduler()
        now = 0
        s.advance(now)
        assert s.should_decide(now + s.interval_ns)

    def test_backlog_consumed_on_advance(self):
        """バックログ時 (複数 tick 遅延) は次回 tick を now から再設定する。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
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
        """test_advance_returns_scheduled_ns の契約を検証する。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        s = DecisionScheduler()
        scheduled = s.advance(1_000_000)
        assert scheduled == 1_000_000  # first call: scheduled = now

    def test_reset_clears_schedule(self):
        """test_reset_clears_schedule の契約を検証する。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        s = DecisionScheduler()
        s.advance(0)
        s.reset()
        assert s.should_decide(0)  # first-tick 動作に戻る

    def test_invalid_now_ns_raises(self):
        """test_invalid_now_ns_raises の契約を検証する。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        s = DecisionScheduler()
        with pytest.raises(ValueError, match="now_ns"):
            s.should_decide(-1)

    def test_float_now_ns_raises(self):
        """test_float_now_ns_raises の契約を検証する。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        s = DecisionScheduler()
        with pytest.raises(ValueError, match="now_ns"):
            s.should_decide(1.0)  # type: ignore[arg-type]


class TestSnapshotAgeGate:
    """[指摘5] captured_ns の鮮度を monotonic deadline で判定する。

    やさしい説明: このまとまりが担当する境界を分かりやすく示します。
    """

    def test_zero_captured_ns_is_not_fresh(self):
        """assembler が timestamp を埋めなかった snapshot を使わせない。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        scheduler = DecisionScheduler()
        assert scheduler.is_snapshot_fresh(0, 1_000_000_000) is False

    def test_negative_captured_ns_is_not_fresh(self):
        """test_negative_captured_ns_is_not_fresh の契約を検証する。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        scheduler = DecisionScheduler()
        assert scheduler.is_snapshot_fresh(-1, 1_000_000_000) is False

    def test_recent_snapshot_is_fresh(self):
        """test_recent_snapshot_is_fresh の契約を検証する。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        scheduler = DecisionScheduler()
        captured = 1_000_000_000
        assert scheduler.is_snapshot_fresh(captured, captured + 10_000_000) is True

    def test_snapshot_older_than_limit_is_not_fresh(self):
        """test_snapshot_older_than_limit_is_not_fresh の契約を検証する。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        scheduler = DecisionScheduler()
        captured = 1_000_000_000
        stale_now = captured + DEFAULT_MAX_SNAPSHOT_AGE_NS + 1
        assert scheduler.is_snapshot_fresh(captured, stale_now) is False

    def test_snapshot_at_exact_limit_is_fresh(self):
        """test_snapshot_at_exact_limit_is_fresh の契約を検証する。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        scheduler = DecisionScheduler()
        captured = 1_000_000_000
        edge = captured + DEFAULT_MAX_SNAPSHOT_AGE_NS
        assert scheduler.is_snapshot_fresh(captured, edge) is True

    def test_future_snapshot_is_not_fresh(self):
        """時計の逆行で未来 timestamp が来た場合も拒否する。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        scheduler = DecisionScheduler()
        captured = 2_000_000_000
        assert scheduler.is_snapshot_fresh(captured, captured - 1) is False

    def test_snapshot_age_ns_is_difference(self):
        """test_snapshot_age_ns_is_difference の契約を検証する。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        scheduler = DecisionScheduler()
        assert scheduler.snapshot_age_ns(100, 350) == 250

    def test_custom_age_limit_is_honoured(self):
        """test_custom_age_limit_is_honoured の契約を検証する。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        scheduler = DecisionScheduler(max_snapshot_age_ns=5_000_000)
        captured = 1_000_000_000
        assert scheduler.is_snapshot_fresh(captured, captured + 4_000_000) is True
        assert scheduler.is_snapshot_fresh(captured, captured + 6_000_000) is False

    def test_non_positive_age_limit_rejected(self):
        """test_non_positive_age_limit_rejected の契約を検証する。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        with pytest.raises(ValueError, match="max_snapshot_age_ns"):
            DecisionScheduler(max_snapshot_age_ns=0)


class TestInferenceTimeoutGate:
    """[指摘5] 推論の締切超過を検出する。

    やさしい説明: このまとまりが担当する境界を分かりやすく示します。
    """

    def test_fast_inference_is_within_timeout(self):
        """test_fast_inference_is_within_timeout の契約を検証する。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        scheduler = DecisionScheduler()
        assert scheduler.exceeded_inference_timeout(0, 1_000_000) is False

    def test_slow_inference_exceeds_timeout(self):
        """test_slow_inference_exceeds_timeout の契約を検証する。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        scheduler = DecisionScheduler()
        assert (
            scheduler.exceeded_inference_timeout(0, DEFAULT_INFERENCE_TIMEOUT_NS + 1)
            is True
        )

    def test_exact_timeout_is_not_exceeded(self):
        """test_exact_timeout_is_not_exceeded の契約を検証する。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        scheduler = DecisionScheduler()
        assert (
            scheduler.exceeded_inference_timeout(0, DEFAULT_INFERENCE_TIMEOUT_NS) is False
        )

    def test_custom_timeout_is_honoured(self):
        """test_custom_timeout_is_honoured の契約を検証する。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        scheduler = DecisionScheduler(inference_timeout_ns=1_000)
        assert scheduler.exceeded_inference_timeout(0, 1_001) is True

    def test_non_positive_timeout_rejected(self):
        """test_non_positive_timeout_rejected の契約を検証する。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        with pytest.raises(ValueError, match="inference_timeout_ns"):
            DecisionScheduler(inference_timeout_ns=-1)


class TestBacklogAccounting:
    """[指摘7] backlog で古い tick を skip し、件数を telemetry へ残す。

    やさしい説明: このまとまりが担当する境界を分かりやすく示します。
    """

    def test_backlog_skips_are_counted(self):
        """test_backlog_skips_are_counted の契約を検証する。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        scheduler = DecisionScheduler()
        scheduler.advance(0)
        # 5 interval ぶん遅れて呼ばれた場合、間の tick は捨てられる。
        scheduler.advance(scheduler.interval_ns * 5)
        assert scheduler.skipped_tick_count >= 4

    def test_on_cadence_calls_skip_nothing(self):
        """test_on_cadence_calls_skip_nothing の契約を検証する。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        scheduler = DecisionScheduler()
        now = 0
        scheduler.advance(now)
        for _ in range(5):
            now += scheduler.interval_ns
            assert scheduler.should_decide(now) is True
            scheduler.advance(now)
        assert scheduler.skipped_tick_count == 0

    def test_decision_count_tracks_advances(self):
        """test_decision_count_tracks_advances の契約を検証する。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        scheduler = DecisionScheduler()
        for index in range(4):
            scheduler.advance(index * scheduler.interval_ns)
        assert scheduler.decision_count == 4

    def test_thirty_minute_schedule_holds_cadence(self):
        """30 分 (27000 tick) 相当を回しても cadence が崩れない。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
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
    """DecisionTiming の latency 計算を確認する。

    やさしい説明: このまとまりが担当する境界を分かりやすく示します。
    """

    def test_inference_latency_ms(self):
        """test_inference_latency_ms の契約を検証する。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        timing = DecisionTiming("d", 0, 1_000_000, 4_000_000)
        assert timing.inference_latency_ms == pytest.approx(3.0)

    def test_wall_latency_ms(self):
        """test_wall_latency_ms の契約を検証する。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        timing = DecisionTiming("d", 0, 1_000_000, 4_000_000)
        assert timing.wall_latency_ms == pytest.approx(4.0)


class TestScreenDecisionScheduler:
    """capture frame ごとの UI transition scheduler を検証する。

    やさしい説明: combat の 15 Hz deadline と独立し、30 Hz frame を一つも間引かないことを見ます。
    """

    @staticmethod
    def _input(state: ScreenState, frame: str) -> UiPolicyInputV1:
        """指定 UI state の共有 policy input を返す。

        やさしい説明: ROI や model tensor を持たない最小 fixture で scheduler 境界を試します。
        """
        return UiPolicyInputV1(
            source_snapshot_hash=f"snapshot-{frame}",
            source_frame_hash=frame,
            source_content_hash=f"content-{frame}",
            ui_state_key=f"state-{frame}",
            screen_state=state,
            hp_fraction=1.0,
        )

    def test_every_capture_frame_is_evaluated_independently_of_combat_deadline(self) -> None:
        """33 ms 間隔の二 frame を両方評価する。

        やさしい説明: 15 Hz の 66 ms gate が UI 側へ波及する回帰を検出します。
        """
        scheduler = ScreenDecisionScheduler()
        config = NonModelUiPolicyConfigV1.load_default()
        assert scheduler.evaluate(self._input(ScreenState.GAMEPLAY, "1"), config) is None
        intent = scheduler.evaluate(self._input(ScreenState.CHEST, "2"), config)
        assert scheduler.evaluation_count == 2
        assert scheduler.transition_count == 1
        assert intent is not None and intent.semantic_action == "ack_chest"
