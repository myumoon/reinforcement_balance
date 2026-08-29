"""perception_session_split の split 検証テスト。

calibration/final の overlap、mixed build、underpowered slice、
source policy 制約を synthetic セッションで確認します。
"""

from __future__ import annotations

import pytest
from survivors.perception_session_split import (
    MixedBuildError,
    SessionRecord,
    SessionSplit,
    SplitOverlapError,
    UnderpoweredSliceError,
    validate_source_policy_consistency,
    validate_split,
)

_BH = "a" * 64
_PH = "b" * 64


def _session(
    session_id: str,
    kind: str,
    *,
    build_hash: str = _BH,
    profile_hash: str = _PH,
    source_policy: str = "raw",
) -> SessionRecord:
    return SessionRecord(
        session_id=session_id,
        session_hash="c" * 64,
        build_hash=build_hash,
        target_profile_hash=profile_hash,
        resolution_wh=(1920, 1080),
        duration_seconds=1800.0,
        kind=kind,  # type: ignore[arg-type]
        source_policy=source_policy,  # type: ignore[arg-type]
    )


class TestSplitOverlap:
    """calibration/final/train/validation の session_id 重複を拒否する。"""

    def test_calibration_final_overlap_fails(self) -> None:
        """同じ session_id が calibration と final_e2e_test の両方に現れるとき SplitOverlapError。"""
        sessions = [_session("s1", "calibration")] * 3 + [_session("s1", "final_e2e_test")] * 3
        with pytest.raises(SplitOverlapError, match="session overlap"):
            validate_split(sessions)

    def test_train_calibration_overlap_fails(self) -> None:
        sessions = [
            _session("shared", "train"),
            _session("shared", "calibration"),
        ]
        with pytest.raises(SplitOverlapError):
            validate_split(sessions, min_benchmark_sessions=1)

    def test_validation_final_overlap_fails(self) -> None:
        sessions = [
            _session("shared", "validation"),
            _session("shared", "final_e2e_test"),
        ]
        with pytest.raises(SplitOverlapError):
            validate_split(sessions, min_benchmark_sessions=1)

    def test_no_overlap_ok(self) -> None:
        """異なる session_id で calibration/final が揃っているとき成功する。"""
        sessions = (
            [_session(f"cal_{i}", "calibration") for i in range(3)]
            + [_session(f"final_{i}", "final_e2e_test") for i in range(3)]
        )
        result = validate_split(sessions)
        assert isinstance(result, SessionSplit)
        assert len(result.sessions) == 6


class TestMixedBuild:
    """calibration セッション間で build/profile/resolution が異なるとき拒否する。"""

    def test_mixed_build_hash_fails(self) -> None:
        sessions = [
            _session("cal_0", "calibration", build_hash="a" * 64),
            _session("cal_1", "calibration", build_hash="b" * 64),
            _session("cal_2", "calibration", build_hash="a" * 64),
        ]
        with pytest.raises(MixedBuildError):
            validate_split(sessions)

    def test_mixed_profile_hash_fails(self) -> None:
        sessions = [
            _session("c0", "calibration", profile_hash="a" * 64),
            _session("c1", "calibration", profile_hash="b" * 64),
            _session("c2", "calibration", profile_hash="a" * 64),
        ]
        with pytest.raises(MixedBuildError):
            validate_split(sessions)

    def test_consistent_build_ok(self) -> None:
        sessions = [_session(f"c{i}", "calibration") for i in range(3)]
        result = validate_split(sessions)
        assert len(result.sessions) == 3


class TestUnderpowered:
    """calibration/final スライスが min_benchmark_sessions 未満のとき拒否する。"""

    def test_one_calibration_session_fails(self) -> None:
        sessions = [_session("c0", "calibration")]
        with pytest.raises(UnderpoweredSliceError):
            validate_split(sessions)

    def test_two_calibration_sessions_fails(self) -> None:
        sessions = [_session(f"c{i}", "calibration") for i in range(2)]
        with pytest.raises(UnderpoweredSliceError):
            validate_split(sessions)

    def test_three_calibration_sessions_ok(self) -> None:
        sessions = [_session(f"c{i}", "calibration") for i in range(3)]
        result = validate_split(sessions)
        assert len(result.sessions) == 3

    def test_no_benchmark_sessions_ok(self) -> None:
        """calibration/final がゼロ件のとき UnderpoweredSliceError は起きない。"""
        sessions = [_session("t0", "train")]
        result = validate_split(sessions)
        assert len(result.sessions) == 1

    def test_empty_sessions_ok(self) -> None:
        result = validate_split([])
        assert len(result.sessions) == 0


class TestSourcePolicy:
    """MP4 decode は domain-shift 比較専用で benchmark 入力に使えない。"""

    def test_mp4_in_calibration_fails_source_check(self) -> None:
        sessions = [_session(f"c{i}", "calibration", source_policy="mp4") for i in range(3)]
        validate_split(sessions)  # split 自体は通る
        with pytest.raises(ValueError, match="mp4"):
            validate_source_policy_consistency(sessions, benchmark_kind="calibration")

    def test_raw_lossless_in_calibration_ok(self) -> None:
        sessions = [
            _session("c0", "calibration", source_policy="raw"),
            _session("c1", "calibration", source_policy="lossless"),
            _session("c2", "calibration", source_policy="raw"),
        ]
        validate_source_policy_consistency(sessions, benchmark_kind="calibration")  # エラーなし

    def test_mp4_in_train_ok(self) -> None:
        """train セッションでは MP4 を許可する。"""
        sessions = [_session("t0", "train", source_policy="mp4")]
        validate_source_policy_consistency(sessions, benchmark_kind="calibration")  # エラーなし

    def test_raw_and_lossless_distinguished(self) -> None:
        """raw と lossless は source_policy で明示的に区別される。"""
        r = _session("s0", "calibration", source_policy="raw")
        l_ = _session("s1", "calibration", source_policy="lossless")
        assert r.source_policy == "raw"
        assert l_.source_policy == "lossless"
        assert r.source_policy != l_.source_policy
