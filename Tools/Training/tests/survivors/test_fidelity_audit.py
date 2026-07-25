"""Fidelity audit の抽出・整列・CI・unmeasurable 処理を検証する。

実 video を必要としない合成 session fixture でコア計測を固定します。
"""

import pytest

from audit_survivors_fidelity import (
    AuditRunProfile,
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
    return {"session_id": session, "time_seconds": 1.0, "metrics": {"speed_px": speed, "viewport_width": width}, "uncertainties": {"speed_px": "occluded"} if speed is None else {}}


def test_extract_alignment_cluster_ci_and_unmeasurable() -> None:
    """viewport alignment と session 単位 CI、None 除外を検証する。

    推測値で穴埋めせず、計測できた session だけが差分へ寄与します。
    """
    target = extract_action_telemetry([_wire("a", 20), _wire("b", None)])
    simulator = extract_action_telemetry([_wire("a", 10, 50), _wire("b", 50)])
    result = align_and_compare(target, simulator)
    assert result[0].metric == "speed_px"
    assert result[0].mean_difference == pytest.approx(0.0)
    assert result[0].session_count == 1


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
    video = [{"session_id": "a", "time_seconds": 1.0, "direction_x": 1.0,
              "direction_y": 0.0, "speed_px": 20.0, "viewport_width": 100.0,
              "enemy_density": 3.0, "chest_visible": 0.0}]
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
        return target

    result = run_fidelity_audit(target, profile, runner)
    assert seen == [profile]
    assert {row.metric for row in result} >= {"speed_px", "timer_seconds", "terminal_event"}


def test_missing_required_source_field_is_not_guessed() -> None:
    """必須抽出 field の欠落を推測で補わない。

    video と telemetry のどちらの欠落も accepted uncertainty の明示なしでは拒否します。
    """
    with pytest.raises(ValueError):
        extract_target_session([], [])
