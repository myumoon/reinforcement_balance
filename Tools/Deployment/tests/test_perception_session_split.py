"""perception_session_split の split 検証テスト。

calibration/final の overlap、mixed build、underpowered slice、
source policy 制約、content hash 一意性、session 最小時間、
clock regression を synthetic セッションで確認します。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from survivors.capture_dataset import FrameRecord, SessionManifest, SplitManifest, SplitUnit
from survivors.perception_session_split import (
    ClockRegressionError,
    DuplicateFrameError,
    MissingBenchmarkSplitError,
    MixedBuildError,
    SessionRecord,
    SessionSplit,
    ShortSessionError,
    SplitOverlapError,
    UnderpoweredSliceError,
    validate_frame_timestamps,
    validate_source_policy_consistency,
    validate_split,
)

_BH = "a" * 64
_PH = "b" * 64


def _manifest(tmp_path: Path, session_id: str) -> SessionManifest:
    session_path = tmp_path / session_id
    session_path.mkdir(parents=True, exist_ok=True)
    manifest_bytes = session_id.encode()
    (session_path / "session_manifest.json").write_bytes(manifest_bytes)
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    records = tuple(
        FrameRecord(
            frame_id=index,
            captured_monotonic_ns=index * 1_800_000_000_000,
            object_path=f"frames/{index}.png",
            object_sha256=hashlib.sha256(f"{session_id}:{index}".encode()).hexdigest(),
            client_rect_screen_px=(0, 0, 1920, 1080), foreground=True,
            target_profile_hash=_PH, game_build_id="build-1",
        )
        for index in (0, 1)
    )
    return SessionManifest(
        1, session_id, _PH, "build-1", False, "synthetic", "lossless_png",
        "frames.jsonl", hashlib.sha256(f"metadata:{session_id}".encode()).hexdigest(), len(records), session_path,
        manifest_hash, records, (),
    )


def _session(
    session_id: str,
    kind: str,
    *,
    build_hash: str = _BH,
    profile_hash: str = _PH,
    source_policy: str = "raw",
    duration_seconds: float = 1800.0,
    session_hash: str | None = None,
    frame_content_hashes: tuple[str, ...] = (),
    capture_lineage: tuple[str, ...] = (),
) -> SessionRecord:
    if session_hash is None:
        session_hash = hashlib.sha256(session_id.encode()).hexdigest()
    return SessionRecord(
        session_id=session_id,
        session_hash=session_hash,
        build_hash=build_hash,
        target_profile_hash=profile_hash,
        resolution_wh=(1920, 1080),
        duration_seconds=duration_seconds,
        kind=kind,  # type: ignore[arg-type]
        source_policy=source_policy,  # type: ignore[arg-type]
        frame_content_hashes=frame_content_hashes,
        capture_lineage=capture_lineage,
    )


class TestSplitOverlap:
    """calibration/final/train/validation の session_id 重複を拒否する。"""

    def test_calibration_final_overlap_fails(self) -> None:
        """同じ session_id が calibration と final の両方に現れると SplitOverlapError。"""
        sessions = (
            [_session("s1", "error_calibration")] * 3
            + [_session("s1", "final_e2e_test")] * 3
        )
        with pytest.raises(SplitOverlapError):
            validate_split(sessions)

    def test_train_calibration_overlap_fails(self) -> None:
        sessions = [
            _session("shared", "model_train"),
            _session("shared", "error_calibration"),
        ]
        with pytest.raises(SplitOverlapError):
            validate_split(sessions, min_benchmark_sessions=1)

    def test_validation_final_overlap_fails(self) -> None:
        sessions = [
            _session("shared", "model_validation"),
            _session("shared", "final_e2e_test"),
        ]
        with pytest.raises(SplitOverlapError):
            validate_split(sessions, min_benchmark_sessions=1)

    def test_no_overlap_ok(self) -> None:
        """異なる session_id で calibration/final が揃っているとき成功する。"""
        sessions = (
            [_session(f"cal_{i}", "error_calibration") for i in range(3)]
            + [_session(f"final_{i}", "final_e2e_test") for i in range(3)]
        )
        result = validate_split(sessions)
        assert isinstance(result, SessionSplit)
        assert len(result.sessions) == 6

    def test_different_session_id_same_content_hash_fails(self) -> None:
        """別 session_id でも同一 content hash は split 横断で拒否する（必須回帰テスト）。"""
        shared_hash = "f" * 64
        sessions = (
            [_session(f"cal_{i}", "error_calibration", session_hash=shared_hash) for i in range(3)]
            + [_session(f"final_{i}", "final_e2e_test", session_hash=shared_hash) for i in range(3)]
        )
        with pytest.raises(SplitOverlapError, match="session_hash"):
            validate_split(sessions)

    def test_same_kind_duplicate_id_and_hash_fails(self) -> None:
        duplicate = _session("same", "model_train")
        with pytest.raises(SplitOverlapError, match="session_id"):
            validate_split([duplicate, duplicate])

    def test_shared_raw_pixels_with_distinct_capture_lineage_allowed(self) -> None:
        """生ピクセルが split 間で偶然一致しても、source-capture identity が異なれば許可する。

        pause/menu 等の正当な静止 frame は別 session で byte-identical になりうるため、
        重複判定は raw pixel hash ではなく source 由来の capture_lineage の intersection に限定する。
        """
        shared = "9" * 64
        sessions = [
            _session(
                "cal", "error_calibration",
                frame_content_hashes=("1" * 64, shared),
                capture_lineage=("3" * 64, "4" * 64),
            ),
            _session(
                "final", "final_e2e_test",
                frame_content_hashes=(shared, "2" * 64),
                capture_lineage=("5" * 64, "6" * 64),
            ),
        ]
        result = validate_split(sessions, min_benchmark_sessions=1)
        assert isinstance(result, SessionSplit)
        assert len(result.sessions) == 2

    def test_session_internal_identical_pixels_allowed(self) -> None:
        """同一 session 内の byte-identical frame（静止画面）を拒否しない。"""
        shared = "7" * 64
        sessions = [
            _session(
                "cal", "error_calibration",
                frame_content_hashes=(shared, shared),
                capture_lineage=("3" * 64, "4" * 64),
            ),
            _session(
                "final", "final_e2e_test",
                frame_content_hashes=(shared, shared),
                capture_lineage=("5" * 64, "6" * 64),
            ),
        ]
        result = validate_split(sessions, min_benchmark_sessions=1)
        assert isinstance(result, SessionSplit)
        assert len(result.sessions) == 2

    def test_one_shared_capture_lineage_across_transcodes_fails(self) -> None:
        lineage = "8" * 64
        sessions = [
            _session(
                "cal", "error_calibration",
                frame_content_hashes=("1" * 64,), capture_lineage=(lineage,),
            ),
            _session(
                "final", "final_e2e_test",
                frame_content_hashes=("2" * 64,), capture_lineage=(lineage,),
            ),
        ]
        with pytest.raises(DuplicateFrameError, match="capture lineage"):
            validate_split(sessions, min_benchmark_sessions=1)


class TestMixedBuild:
    """benchmark セッション間で build/profile/resolution が異なるとき拒否する。"""

    def test_mixed_build_hash_fails(self) -> None:
        sessions = [
            _session("cal_0", "error_calibration", build_hash="a" * 64),
            _session("cal_1", "error_calibration", build_hash="b" * 64),
            _session("cal_2", "error_calibration", build_hash="a" * 64),
        ] + [_session(f"fin_{i}", "final_e2e_test") for i in range(3)]
        with pytest.raises(MixedBuildError):
            validate_split(sessions)

    def test_mixed_profile_hash_fails(self) -> None:
        sessions = [
            _session("c0", "error_calibration", profile_hash="a" * 64),
            _session("c1", "error_calibration", profile_hash="b" * 64),
            _session("c2", "error_calibration", profile_hash="a" * 64),
        ] + [_session(f"fin_{i}", "final_e2e_test") for i in range(3)]
        with pytest.raises(MixedBuildError):
            validate_split(sessions)

    def test_consistent_build_ok(self) -> None:
        sessions = (
            [_session(f"c{i}", "error_calibration") for i in range(3)]
            + [_session(f"f{i}", "final_e2e_test") for i in range(3)]
        )
        result = validate_split(sessions)
        assert len(result.sessions) == 6


class TestUnderpowered:
    """benchmark スライスが min_benchmark_sessions 未満のとき拒否する。"""

    def test_one_calibration_one_final_fails(self) -> None:
        """両方存在するが各1件 → underpowered。"""
        sessions = [
            _session("c0", "error_calibration"),
            _session("f0", "final_e2e_test"),
        ]
        with pytest.raises(UnderpoweredSliceError):
            validate_split(sessions)

    def test_two_calibration_fails(self) -> None:
        sessions = (
            [_session(f"c{i}", "error_calibration") for i in range(2)]
            + [_session(f"f{i}", "final_e2e_test") for i in range(3)]
        )
        with pytest.raises(UnderpoweredSliceError):
            validate_split(sessions)

    def test_three_calibration_three_final_ok(self) -> None:
        sessions = (
            [_session(f"c{i}", "error_calibration") for i in range(3)]
            + [_session(f"f{i}", "final_e2e_test") for i in range(3)]
        )
        result = validate_split(sessions)
        assert len(result.sessions) == 6

    def test_train_only_ok(self) -> None:
        """calibration/final がゼロ件（train のみ）は通る。"""
        sessions = [_session("t0", "model_train")]
        result = validate_split(sessions)
        assert len(result.sessions) == 1

    def test_empty_sessions_ok(self) -> None:
        result = validate_split([])
        assert len(result.sessions) == 0


class TestMissingBenchmarkSplit:
    """calibration か final の片側だけ存在するとき拒否する（必須回帰テスト）。"""

    def test_calibration_only_fails(self) -> None:
        """calibration 3件、final 0件 → MissingBenchmarkSplitError。"""
        sessions = [_session(f"c{i}", "error_calibration") for i in range(3)]
        with pytest.raises(MissingBenchmarkSplitError, match="final_e2e_test"):
            validate_split(sessions)

    def test_final_only_fails(self) -> None:
        """final 3件、calibration 0件 → MissingBenchmarkSplitError。"""
        sessions = [_session(f"f{i}", "final_e2e_test") for i in range(3)]
        with pytest.raises(MissingBenchmarkSplitError, match="error_calibration"):
            validate_split(sessions)

    def test_empty_calibration_with_final_fails(self) -> None:
        """empty calibration split は fail closed（必須回帰テスト）。"""
        sessions = [_session("f0", "final_e2e_test")]
        with pytest.raises(MissingBenchmarkSplitError):
            validate_split(sessions, min_benchmark_sessions=1)


class TestShortSession:
    """30分未満のセッションを拒否する（必須回帰テスト）。"""

    def test_one_second_sessions_fail(self) -> None:
        """1秒セッションを3件並べても ShortSessionError で拒否。"""
        sessions = (
            [_session(f"c{i}", "error_calibration", duration_seconds=1.0) for i in range(3)]
            + [_session(f"f{i}", "final_e2e_test", duration_seconds=1.0) for i in range(3)]
        )
        with pytest.raises(ShortSessionError):
            validate_split(sessions)

    def test_exactly_30_minutes_ok(self) -> None:
        sessions = (
            [_session(f"c{i}", "error_calibration", duration_seconds=1800.0) for i in range(3)]
            + [_session(f"f{i}", "final_e2e_test", duration_seconds=1800.0) for i in range(3)]
        )
        result = validate_split(sessions)
        assert len(result.sessions) == 6

    def test_29_minutes_fails(self) -> None:
        sessions = (
            [_session("c0", "error_calibration", duration_seconds=1799.0)]
            + [_session(f"c{i}", "error_calibration") for i in range(1, 3)]
            + [_session(f"f{i}", "final_e2e_test") for i in range(3)]
        )
        with pytest.raises(ShortSessionError):
            validate_split(sessions)

    def test_nan_duration_fails_at_entry(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            _session("nan", "model_train", duration_seconds=float("nan"))


class TestClockRegression:
    """フレームタイムスタンプの strict monotonicity を確認する（必須回帰テスト）。"""

    def test_monotonic_timestamps_ok(self) -> None:
        validate_frame_timestamps([100, 200, 300, 400])

    def test_decreasing_timestamp_fails(self) -> None:
        with pytest.raises(ClockRegressionError):
            validate_frame_timestamps([100, 200, 150, 300])

    def test_equal_timestamp_fails(self) -> None:
        """同値タイムスタンプも clock regression として拒否。"""
        with pytest.raises(ClockRegressionError):
            validate_frame_timestamps([100, 200, 200, 300])

    def test_empty_timestamps_ok(self) -> None:
        validate_frame_timestamps([])

    def test_single_timestamp_ok(self) -> None:
        validate_frame_timestamps([100])


class TestSourcePolicy:
    """MP4 decode は domain-shift 比較専用で benchmark 入力に使えない。"""

    def test_mp4_in_calibration_fails_source_check(self) -> None:
        sessions = [
            _session(f"c{i}", "error_calibration", source_policy="mp4") for i in range(3)
        ] + [_session(f"f{i}", "final_e2e_test") for i in range(3)]
        with pytest.raises(ValueError, match="mp4"):
            validate_split(sessions)

    def test_raw_lossless_in_calibration_ok(self) -> None:
        sessions = [
            _session("c0", "error_calibration", source_policy="raw"),
            _session("c1", "error_calibration", source_policy="lossless"),
            _session("c2", "error_calibration", source_policy="raw"),
        ] + [_session(f"f{i}", "final_e2e_test") for i in range(3)]
        validate_source_policy_consistency(sessions, benchmark_kind="error_calibration")  # エラーなし

    def test_mp4_in_train_ok(self) -> None:
        """train セッションでは MP4 を許可する。"""
        sessions = [_session("t0", "model_train", source_policy="mp4")]
        validate_source_policy_consistency(sessions, benchmark_kind="error_calibration")  # エラーなし

    def test_raw_and_lossless_distinguished(self) -> None:
        """raw と lossless は source_policy で明示的に区別される。"""
        r = _session("s0", "error_calibration", source_policy="raw")
        l_ = _session("s1", "error_calibration", source_policy="lossless")
        assert r.source_policy == "raw"
        assert l_.source_policy == "lossless"
        assert r.source_policy != l_.source_policy


class TestSplitNamesMatchCapture:
    """split 名が capture_dataset.py SPLIT_NAMES と一致する（必須回帰テスト）。"""

    def test_exact_split_names(self) -> None:
        """model_train, model_validation, error_calibration, final_e2e_test を受理する。"""
        sessions = [
            _session("t0", "model_train"),
            _session("v0", "model_validation"),
        ]
        result = validate_split(sessions)
        assert len(result.sessions) == 2

    def test_model_train_error_calibration_same_session_fails(self) -> None:
        """model_train と error_calibration の同一 session は失敗（必須回帰テスト）。"""
        sessions = [
            _session("shared", "model_train"),
            _session("shared", "error_calibration"),
        ]
        with pytest.raises(SplitOverlapError):
            validate_split(sessions, min_benchmark_sessions=1)


class TestTypedCaptureManifests:
    def test_split_and_session_manifest_are_the_formal_input(self, tmp_path: Path) -> None:
        manifests = {
            name: _manifest(tmp_path, name)
            for name in ("cal", "final")
        }
        split = SplitManifest(
            1, True,
            {
                "model_train": (), "model_validation": (),
                "error_calibration": (SplitUnit("cal", "build-1", manifests["cal"].manifest_sha256),),
                "final_e2e_test": (SplitUnit("final", "build-1", manifests["final"].manifest_sha256),),
            },
            tmp_path / "capture_split_manifest.json", "e" * 64,
        )
        validated = validate_split(split, manifests, min_benchmark_sessions=1)
        assert validated.split_manifest_hash == "e" * 64
        assert validated.expected_ticks == (("cal", "0"), ("cal", "1"), ("final", "0"), ("final", "1"))

    def test_typed_same_kind_duplicate_is_rejected(self, tmp_path: Path) -> None:
        manifest = _manifest(tmp_path, "train")
        unit = SplitUnit("train", "build-1", manifest.manifest_sha256)
        split = SplitManifest(
            1, True,
            {"model_train": (unit, unit), "model_validation": (), "error_calibration": (), "final_e2e_test": ()},
            tmp_path / "capture_split_manifest.json", "e" * 64,
        )
        with pytest.raises(SplitOverlapError):
            validate_split(split, {"train": manifest})
