"""controller health/watchdog(M5)の閾値判定を検証する。

やさしい説明: focus 喪失・capture gap・perception p99・観測無効の継続・
UNKNOWN の継続・推論エラーのそれぞれで STOP が latch され、
counters と first-failure context が報告されることを確認します。
"""

from __future__ import annotations

import numpy as np
import pytest

from reinbalance_survivors_contracts.deploy_obs import DeployObsSchema, DeployObservation
from survivors.controller.health_monitor import (
    HealthMonitor,
    HealthReason,
    HealthThresholds,
    HealthVerdict,
    observation_is_valid,
)
from survivors.controller.state_machine import ControllerState

MS = 1_000_000


def _frame(monitor: HealthMonitor, t_ms: int, **overrides) -> HealthVerdict:
    """正常値を既定にして1フレーム ingest する。

    やさしい説明: テストごとに変えたい引数だけを上書きします。
    """
    kwargs = dict(
        now_ns=t_ms * MS,
        capture_timestamp_ns=t_ms * MS,
        window_focused=True,
        perception_latency_ns=10 * MS,
        observation_valid=True,
        controller_state=ControllerState.GAMEPLAY,
    )
    kwargs.update(overrides)
    return monitor.ingest(**kwargs)


def test_healthy_frames_stay_ok_and_boundary_values_do_not_stop() -> None:
    """閾値ちょうどの値は STOP にしない。

    やさしい説明: 「超えたら」停止なので、200ms ちょうどの gap は許容されます。
    """
    monitor = HealthMonitor()
    assert _frame(monitor, 0) is HealthVerdict.OK
    assert _frame(monitor, 200) is HealthVerdict.OK
    assert monitor.poll(400 * MS) is HealthVerdict.OK
    report = monitor.report()
    assert report.first_failure is None
    assert set(report.stop_counts.values()) == {0}


def test_focus_lost_stops_immediately_and_latches() -> None:
    """focus 喪失は1フレームで STOP、以後の正常フレームでも STOP のまま。

    やさしい説明: 一度止めると決めたら自動復帰しません。
    """
    monitor = HealthMonitor()
    _frame(monitor, 0)
    assert _frame(monitor, 50, window_focused=False) is HealthVerdict.STOP
    assert _frame(monitor, 100) is HealthVerdict.STOP
    report = monitor.report()
    assert report.verdict is HealthVerdict.STOP
    assert report.first_failure.reason is HealthReason.FOCUS_LOST
    assert report.first_failure.now_ns == 50 * MS
    assert report.stop_counts["focus_lost"] == 1


def test_capture_gap_detected_by_ingest_and_by_poll() -> None:
    """capture gap は次フレーム到着時にも、フレームが来ない間の poll でも検知する。

    やさしい説明: capture が完全に止まった場合も watchdog として停止できます。
    """
    monitor = HealthMonitor()
    _frame(monitor, 0)
    assert _frame(monitor, 201) is HealthVerdict.STOP
    assert monitor.report().first_failure.observed == 201 * MS

    stalled = HealthMonitor()
    _frame(stalled, 0)
    assert stalled.poll(200 * MS) is HealthVerdict.OK
    assert stalled.poll(201 * MS) is HealthVerdict.STOP
    failure = stalled.report().first_failure
    assert (failure.reason, failure.threshold) == (HealthReason.CAPTURE_GAP, 200 * MS)


def test_perception_p99_needs_min_samples_then_stops() -> None:
    """p99 は min_samples 到達後に評価し、110ms 超で STOP。

    やさしい説明: 起動直後の1件のスパイクでは止めず、十分な件数で遅いときに止めます。
    """
    monitor = HealthMonitor(HealthThresholds(perception_window=10, perception_min_samples=5))
    for i in range(4):
        assert _frame(monitor, i * 10, perception_latency_ns=500 * MS) is HealthVerdict.OK
    assert _frame(monitor, 40, perception_latency_ns=500 * MS) is HealthVerdict.STOP
    failure = monitor.report().first_failure
    assert failure.reason is HealthReason.PERCEPTION_P99
    assert failure.observed == 500 * MS

    ok = HealthMonitor(HealthThresholds(perception_window=100, perception_min_samples=100))
    for i in range(100):
        # 100件中1件だけ遅い: nearest-rank p99 は99番目 = 110ms 以下なので停止しない
        latency = 900 * MS if i == 0 else 110 * MS
        assert _frame(ok, i * 10, perception_latency_ns=latency) is HealthVerdict.OK


@pytest.mark.parametrize(
    ("field", "bad", "reason", "limit_ms"),
    [
        ("observation_valid", False, HealthReason.OBS_INVALID_STREAK, 500),
        ("controller_state", ControllerState.UNKNOWN, HealthReason.UNKNOWN_STREAK, 1000),
    ],
)
def test_streak_warns_then_stops_and_resets_on_recovery(field, bad, reason, limit_ms) -> None:
    """継続系は閾値内で WARNING、回復でリセット、閾値超えで STOP。

    やさしい説明: 一瞬の無効/UNKNOWN では止めず、続いたときだけ止めます。
    """
    monitor = HealthMonitor()
    # 0ms 開始の streak が limit ちょうどまで続いても WARNING のまま
    for t in range(0, limit_ms + 1, 100):
        assert _frame(monitor, t, **{field: bad}) is HealthVerdict.WARNING
    # 1フレーム回復すると streak はリセットされる
    t = limit_ms + 100
    assert _frame(monitor, t) is HealthVerdict.OK
    # 再開始した streak が limit を 1ms 超えた時点で STOP
    start = t + 100
    for t in range(start, start + limit_ms + 1, 100):
        assert _frame(monitor, t, **{field: bad}) is HealthVerdict.WARNING
    assert _frame(monitor, start + limit_ms + 1, **{field: bad}) is HealthVerdict.STOP
    report = monitor.report()
    assert report.first_failure.reason is reason
    assert report.first_failure.observed > limit_ms * MS
    assert report.warning_counts[reason.value] > 0
    assert report.to_wire()["first_failure"]["reason"] == reason.value


def test_inference_error_stops_once_and_first_failure_is_not_overwritten() -> None:
    """推論エラーは1回で STOP。後続の別原因は counter のみ増える。

    やさしい説明: 最初に壊れた原因が事後調査で上書きされないことを確認します。
    """
    monitor = HealthMonitor()
    _frame(monitor, 0)
    assert monitor.record_inference_error(now_ns=10 * MS, detail="onnx runtime error") is HealthVerdict.STOP
    _frame(monitor, 20, window_focused=False)
    report = monitor.report().to_wire()
    assert report["verdict"] == "stop"
    assert report["first_failure"] == {
        "reason": "inference_error",
        "now_ns": 10 * MS,
        "observed": None,
        "threshold": None,
        "detail": "onnx runtime error",
    }
    assert report["stop_counts"]["inference_error"] == 1
    assert report["stop_counts"]["focus_lost"] == 1


def test_invalid_inputs_fail_loudly() -> None:
    """型不正・時計の巻き戻り・不正な閾値は ValueError。

    やさしい説明: 異常な入力を黙って無視すると監視が効かなくなるため拒否します。
    """
    monitor = HealthMonitor()
    _frame(monitor, 100)
    with pytest.raises(ValueError):
        _frame(monitor, 50)
    with pytest.raises(ValueError):
        _frame(monitor, 150, capture_timestamp_ns=10 * MS)
    with pytest.raises(ValueError):
        _frame(monitor, 200, window_focused=1)
    with pytest.raises(ValueError):
        _frame(monitor, 200, controller_state="unknown")
    with pytest.raises(ValueError):
        monitor.record_inference_error(now_ns=300 * MS, detail="")
    with pytest.raises(ValueError):
        HealthThresholds(capture_gap_ns=0)
    with pytest.raises(ValueError):
        HealthThresholds(perception_window=5, perception_min_samples=6)


def test_observation_is_valid_uses_existing_contract() -> None:
    """None と schema 不一致の観測は無効、契約どおりの観測は有効。

    やさしい説明: 判定は ``DeployObservation.validate_for`` に委ねています。
    """
    schema = DeployObsSchema.default_v2()
    values = np.zeros(schema.dim, dtype=np.float32)
    for field in schema.fields:
        offset, size = schema.layout[field.name]
        values[offset:offset + size] = field.neutral
    planes = dict(values=values, validity=np.zeros(schema.dim), age=np.ones(schema.dim), timestamp_ns=0)
    assert observation_is_valid(DeployObservation(schema_hash=schema.schema_hash, **planes), schema)
    assert not observation_is_valid(DeployObservation(schema_hash="f" * 64, **planes), schema)
    assert not observation_is_valid(None, schema)
