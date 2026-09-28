"""capture_survivors.py CLI のオペレーター耐性機能を検証する。

Ctrl+Cによる部分公開・ABORTED_EMPTY、フォアグラウンド待機の
リトライ/中断、--session-id autoの採番が仕様通り動くことを確認する。
"""

from __future__ import annotations

import contextlib
import json
import signal
import threading

import numpy as np
import pytest

import capture_survivors
from survivors.capture.captured_frame import CapturedFrame
from survivors.capture.window_locator import TargetWindowForegroundLost, TargetWindowStateError


def _frame(frame_id: int, timestamp_ns: int) -> CapturedFrame:
    """synthetic分岐のidentityと一致する決定的CapturedFrameを生成する。

    テストのために画素は全て0で構成し、frame_idとtimestampだけを変える。
    """
    pixels = np.zeros((1080, 1920, 4), dtype=np.uint8)
    pixels[..., 3] = 255
    return CapturedFrame(
        pixels,
        timestamp_ns,
        frame_id,
        (0, 0, 1920, 1080),
        True,
        capture_survivors.SYNTHETIC_PROFILE_HASH,
        capture_survivors.SYNTHETIC_BUILD_ID,
    )


class _FakeForegroundLocator:
    """フォアグラウンド喪失を指定回数だけ模擬するテスト用locator。"""

    def __init__(self, failures_then_success: int):
        self.calls = 0
        self._failures = failures_then_success

    def validate_lightweight(self, target, *, require_foreground):
        self.calls += 1
        if self.calls <= self._failures:
            raise TargetWindowForegroundLost("still lost")


def test_wait_for_foreground_retries_until_success():
    """規定回数だけフォアグラウンド喪失が続いても復帰を待てることを確認する。"""
    locator = _FakeForegroundLocator(failures_then_success=3)
    interrupted = threading.Event()
    assert (
        capture_survivors._wait_for_foreground(
            locator, object(), interrupted, poll_interval_sec=0
        )
        is True
    )
    assert locator.calls == 4


def test_wait_for_foreground_returns_false_when_interrupted():
    """待機中にCtrl+Cされた場合は例外を投げずFalseで抜けることを確認する。"""
    locator = _FakeForegroundLocator(failures_then_success=10**6)
    interrupted = threading.Event()
    interrupted.set()
    assert (
        capture_survivors._wait_for_foreground(
            locator, object(), interrupted, poll_interval_sec=0
        )
        is False
    )


def test_wait_for_foreground_propagates_non_foreground_state_errors():
    """フォアグラウンド消失以外の状態異常はリトライせず即座に伝播することを確認する。"""

    class _FatalLocator:
        def validate_lightweight(self, target, *, require_foreground):
            raise TargetWindowStateError("resize")

    with pytest.raises(TargetWindowStateError):
        capture_survivors._wait_for_foreground(
            _FatalLocator(), object(), threading.Event(), poll_interval_sec=0
        )


class _FakeCaptureSession:
    """_capture_liveをCaptureSessionの実装なしで検証するための最小dummy。

    locator/target/start/capture_next/closeだけをduck-typingで
    提供し、start呼び出し有無・close呼び出し有無を観測する。
    """

    def __init__(self, frames_to_yield=(), *, foreground_ok=True):
        self.locator = self
        self.target = object()
        self._frames = list(frames_to_yield)
        self._foreground_ok = foreground_ok
        self.started = False
        self.closed = False

    def validate_lightweight(self, target, *, require_foreground):
        if not self._foreground_ok:
            raise TargetWindowForegroundLost("still lost")

    def start(self):
        self.started = True

    def capture_next(self):
        if self._frames:
            return self._frames.pop(0)
        return None

    def close(self):
        self.closed = True


def test_capture_live_closes_without_starting_when_interrupted_before_foreground():
    """起動前に中断された場合、start()を呼ばずcloseだけ行いended_reasonをinterruptedにすることを確認する。

    session.start()はDXcamの実キャプチャを起動する重い呼び出しのため、
    フォアグラウンドを一度も確認できないまま呼んでしまわないことが重要。
    """
    session = _FakeCaptureSession(foreground_ok=False)
    interrupted = threading.Event()
    interrupted.set()  # 起動前から中断済みだった状況を再現する
    stats: dict = {}

    frames = list(capture_survivors._capture_live(session, 1.0, interrupted, stats))

    assert frames == []
    assert session.started is False
    assert session.closed is True
    assert stats["ended_reason"] == "interrupted"


def test_capture_live_yields_until_deadline_then_closes():
    """フォアグラウンド確認後にframeをyieldし、deadline到達後は必ずcloseすることを確認する。"""
    session = _FakeCaptureSession([_frame(0, 100)])
    interrupted = threading.Event()
    stats: dict = {}

    frames = list(capture_survivors._capture_live(session, 0.05, interrupted, stats))

    assert [frame.session_frame_index for frame in frames] == [0]
    assert session.started is True
    assert session.closed is True
    assert stats["ended_reason"] == "duration_elapsed"
    assert "started_at" in stats


def test_graceful_interrupt_sets_flag_without_raising_on_first_sigint():
    """1回目のCtrl+Cは例外を投げずフラグを立てるだけであることを確認する。"""
    flag = threading.Event()
    with capture_survivors._graceful_interrupt(flag):
        signal.raise_signal(signal.SIGINT)
        assert flag.is_set()


def test_graceful_interrupt_escalates_to_keyboardinterrupt_on_second_sigint():
    """2回目のCtrl+Cは本来のKeyboardInterruptへエスケープすることを確認する。"""
    flag = threading.Event()
    with pytest.raises(KeyboardInterrupt):
        with capture_survivors._graceful_interrupt(flag):
            signal.raise_signal(signal.SIGINT)
            signal.raise_signal(signal.SIGINT)


def test_graceful_interrupt_restores_previous_handler_on_exit():
    """with文を抜けた後は元のSIGINTハンドラへ戻すことを確認する。"""
    previous = signal.getsignal(signal.SIGINT)
    with capture_survivors._graceful_interrupt(threading.Event()):
        pass
    assert signal.getsignal(signal.SIGINT) is previous


def test_next_auto_session_id_starts_at_0001_for_empty_store(tmp_path):
    """capture_sessionsが存在しない場合はsession-0001から始まることを確認する。"""
    assert capture_survivors._next_auto_session_id(tmp_path) == "session-0001"


def test_next_auto_session_id_increments_past_existing_max(tmp_path):
    """既存の最大連番より1大きい値を、無関係な名前を無視して採番することを確認する。"""
    sessions_dir = tmp_path / "capture_sessions"
    sessions_dir.mkdir()
    (sessions_dir / "session-0001").mkdir()
    (sessions_dir / "session-0007").mkdir()
    (sessions_dir / "not-a-session").mkdir()
    assert capture_survivors._next_auto_session_id(tmp_path) == "session-0008"


def test_main_resolves_session_id_auto_to_first_available_slot(tmp_path, capsys):
    """--session-id autoがmain()経由でsession-0001に解決されることを確認する。"""
    result = capture_survivors.main(
        [
            "--store-root", str(tmp_path),
            "--session-id", "auto",
            "--duration-sec", "0.5",
            "--synthetic",
        ]
    )
    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["session_id"] == "session-0001"


def test_main_aborts_empty_without_publishing_when_interrupted_before_any_frame(
    tmp_path, monkeypatch, capsys
):
    """1枚も撮れないまま中断された場合、公開せずABORTED_EMPTYで終わることを確認する。

    未公開tempのrmtreeによる全データ消失を避けつつ、実際に何も
    無かった場合は空セッションを残さないことを両立させる経路を検証する。
    """

    @contextlib.contextmanager
    def fake_graceful_interrupt(flag):
        flag.set()  # 収録開始前から中断済みだった状況を再現する
        yield flag

    def empty_frames(self, duration_sec):
        return iter(())

    monkeypatch.setattr(capture_survivors, "_graceful_interrupt", fake_graceful_interrupt)
    monkeypatch.setattr(capture_survivors.FakeCaptureBackend, "frames", empty_frames)

    result = capture_survivors.main(
        [
            "--store-root", str(tmp_path),
            "--session-id", "aborted-empty",
            "--duration-sec", "1.0",
            "--synthetic",
        ]
    )

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ABORTED_EMPTY"
    assert payload["session_id"] == "aborted-empty"
    assert payload["ended_reason"] == "interrupted"
    assert payload["requested_duration_sec"] == 1.0
    assert payload["elapsed_sec"] >= 0
    assert not (tmp_path / "capture_sessions" / "aborted-empty").exists()


def test_main_publishes_partial_result_on_interrupt_mid_capture(tmp_path, monkeypatch, capsys):
    """撮影途中の中断でも、そこまでのフレームがPUBLISHEDされることを確認する。

    2枚目をyieldする直前に中断フラグを立て、その最後の1枚まで
    含めて完全に公開されること・ended_reasonがinterruptedになることを見る。
    """
    captured = {}

    @contextlib.contextmanager
    def fake_graceful_interrupt(flag):
        captured["flag"] = flag
        yield flag

    def two_then_interrupt(self, duration_sec):
        yield _frame(0, 100)
        captured["flag"].set()
        yield _frame(1, 200)

    monkeypatch.setattr(capture_survivors, "_graceful_interrupt", fake_graceful_interrupt)
    monkeypatch.setattr(capture_survivors.FakeCaptureBackend, "frames", two_then_interrupt)

    result = capture_survivors.main(
        [
            "--store-root", str(tmp_path),
            "--session-id", "partial-session",
            "--duration-sec", "1.0",
            "--synthetic",
        ]
    )

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "PUBLISHED"
    assert payload["frame_count"] == 2
    assert payload["ended_reason"] == "interrupted"
    assert payload["requested_duration_sec"] == 1.0
    assert payload["elapsed_sec"] >= 0
