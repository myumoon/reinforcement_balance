"""Fidelity audit の抽出・整列・CI・unmeasurable 処理を検証する。

実 video を必要としない合成 session fixture でコア計測を固定します。
"""

import pytest

from audit_survivors_fidelity import TelemetrySample, align_and_compare, extract_action_telemetry


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
