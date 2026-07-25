"""Fidelity audit の抽出・整列・CI・unmeasurable 処理を検証する。

実 video を必要としない合成 session fixture でコア計測を固定します。
"""

import pytest

from audit_survivors_fidelity import (
    AuditRunProfile,
    ExtractedVideoFrame,
    TelemetrySample,
    align_and_compare,
    extract_action_telemetry,
    extract_target_session,
    run_fidelity_audit,
)


def _wire(session: str, speed: float | None, width: float = 100.0) -> dict:
    """テスト用 telemetry wire を返す。

    speed が None の場合だけ accepted uncertainty を付けます。
    """
    metrics = {
        "direction_x": 1.0, "direction_y": 0.0, "speed_px": speed, "viewport_width": width,
        "enemy_density": 1.0, "chest_visible": 0.0, "cadence_hz": 10.0,
        "timer_seconds": 1.0, "level": 2.0, "offer_count": 3.0, "terminal_event": 0.0,
    }
    return {"session_id": session, "time_seconds": 1.0, "metrics": metrics, "uncertainties": {"speed_px": "occluded"} if speed is None else {}}


def _sample(session: str, time_seconds: float, speed: float) -> TelemetrySample:
    """必須 metric を全て持つ比較 sample を返す。

    個別テストは speed と時刻だけを変え、監査入力契約は常に満たします。
    """
    wire = _wire(session, speed)
    wire["time_seconds"] = time_seconds
    return TelemetrySample.from_wire(wire)


def test_extract_alignment_cluster_ci_and_unmeasurable() -> None:
    """viewport alignment と session 単位 CI、None 除外を検証する。

    推測値で穴埋めせず、計測できた session だけが差分へ寄与します。
    """
    target = extract_action_telemetry([_wire("a", 20), _wire("b", None)])
    simulator = extract_action_telemetry([_wire("a", 10, 50), _wire("b", 50)])
    result = align_and_compare(target, simulator)
    speed = next(row for row in result if row.metric == "speed_px")
    assert speed.mean_difference == pytest.approx(0.0)
    assert speed.session_count == 1


def test_alignment_does_not_reuse_simulator_session_or_sample() -> None:
    """simulator session/sample を複数 target cluster へ再利用しない。

    一つの simulator run を二つの独立観測として数えず、標本不足を N に反映します。
    """
    target = (
        _sample("target-a", 1.00, 10.0),
        _sample("target-b", 1.02, 20.0),
    )
    simulator = (_sample("sim-one", 1.01, 5.0),)
    result = align_and_compare(target, simulator, time_tolerance=0.1)
    assert all(row.session_count == 1 for row in result)


def test_alignment_minimizes_total_time_distance_without_crossing() -> None:
    """最大対応数が同じ場合に総時間差が最小の sample 対応を選ぶ。

    完全な一対一対応が可能でも交差させず、近い時刻同士の値で session 集計します。
    """
    target = (
        _sample("target", 0.00, 0.0),
        _sample("target", 0.09, 100.0),
    )
    simulator = (
        _sample("simulator", 0.08, 100.0),
        _sample("simulator", 0.10, 0.0),
    )
    result = align_and_compare(target, simulator, time_tolerance=0.1)
    speed = next(row for row in result if row.metric == "speed_px")
    assert speed.mean_difference == pytest.approx(0.0)
    assert speed.session_count == 1


def test_nonfinite_and_unknown_fields_fail_closed() -> None:
    """非 finite 値と未知 field を抽出境界で拒否する。

    集計前に不正 telemetry を隔離します。
    """
    with pytest.raises(ValueError):
        TelemetrySample("a", float("nan"), {}, {})
    wire = _wire("a", 1)
    wire["unknown"] = 1
    with pytest.raises(ValueError):
        extract_action_telemetry([wire])


def test_required_video_telemetry_and_matching_simulator_run() -> None:
    """必須 video/telemetry 抽出と同 profile/time band の sim 実行を検証する。

    runner へ監査条件がそのまま渡り、合成入力だけで end-to-end 比較できます。
    """
    video = [{"session_id": "a", "time_seconds": 1.0,
              "frame": {"player_xy": [10.0, 10.0], "previous_player_xy": [8.0, 10.0],
                        "elapsed_seconds": 0.1, "viewport_width": 100.0,
                        "enemy_centers": [[20.0, 20.0], [30.0, 30.0], [40.0, 40.0]],
                        "chest_centers": []}}]
    telemetry = [{"session_id": "a", "time_seconds": 1.0, "cadence_hz": 10.0,
                  "timer_seconds": 1.0, "level": 2.0, "offer_count": 3.0,
                  "terminal_event": 0.0}]
    target = extract_target_session(video, telemetry)
    profile = AuditRunProfile("profile-hash", 0.0, 5.0)
    seen = []

    def runner(request):
        """要求条件を記録して同形式の simulator sample を返す。

        production simulator の代わりに deterministic fixture を注入します。
        """
        seen.append(request)
        return tuple(
            TelemetrySample("sim-" + row.session_id, row.time_seconds, row.metrics, row.uncertainties)
            for row in target
        )

    result = run_fidelity_audit(target, profile, runner)
    assert seen == [profile]
    assert {row.metric for row in result} >= {"speed_px", "timer_seconds", "terminal_event"}


def test_video_frames_use_extraction_interface() -> None:
    """frame payload を metric 抽出 interface に必ず通す。

    数値 metric の素通しではなく、位置差と elapsed time から方向・速度を導出します。
    """
    frame = ExtractedVideoFrame.from_wire({
        "player_xy": [4.0, 2.0], "previous_player_xy": [2.0, 2.0],
        "elapsed_seconds": 0.5, "viewport_width": 20.0,
        "enemy_centers": [[1.0, 1.0]], "chest_centers": [],
    })
    assert frame.metrics["direction_x"] == 1.0
    assert frame.metrics["speed_px"] == 4.0


def test_missing_required_source_field_is_not_guessed() -> None:
    """必須抽出 field の欠落を推測で補わない。

    video と telemetry のどちらの欠落も accepted uncertainty の明示なしでは拒否します。
    """
    with pytest.raises(ValueError):
        extract_target_session([], [])


def test_required_metric_coverage_and_nonempty_alignment_fail_closed() -> None:
    """target/simulator の必須 metric 欠落と空比較を拒否する。

    積集合が空でも成功扱いせず、accepted uncertainty は key を明示した場合だけ許可します。
    """
    complete = _sample("target", 1.0, 10.0)
    missing_metrics = dict(complete.metrics)
    missing_metrics.pop("terminal_event")
    missing = TelemetrySample("simulator", 1.0, missing_metrics, {})
    with pytest.raises(ValueError, match="required metric keys mismatch"):
        align_and_compare((complete,), (missing,))
    with pytest.raises(ValueError, match="no target/simulator sessions align"):
        align_and_compare((complete,), (_sample("simulator", 5.0, 10.0),))

    all_unmeasurable_metrics = {name: None for name in complete.metrics}
    all_uncertainties = {name: "accepted unavailable" for name in complete.metrics}
    unmeasurable = TelemetrySample("simulator", 1.0, all_unmeasurable_metrics, all_uncertainties)
    with pytest.raises(ValueError, match="no measurable metrics"):
        align_and_compare((TelemetrySample("target", 1.0, all_unmeasurable_metrics, all_uncertainties),), (unmeasurable,))
