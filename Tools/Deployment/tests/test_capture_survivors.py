"""capture_survivors.py CLI のオペレーター耐性機能を検証する。

Ctrl+Cによる部分公開・ABORTED_EMPTY、フォアグラウンド待機の
リトライ/中断、--session-id autoの採番が仕様通り動くことを確認する。
"""

from __future__ import annotations

import contextlib
import json
import signal
import threading
import time
from types import SimpleNamespace

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
    """フォアグラウンド喪失を指定回数だけ模擬するテスト用locator。

    実際のWindowLocatorを使わずに、「あと何回失敗させるか」だけを
    指定して_wait_for_foregroundのリトライ挙動を検証するためのdouble。
    """

    def __init__(self, failures_then_success: int):
        """呼び出し回数カウンタと失敗させる残り回数を初期化する。

        failures_then_successに指定した回数だけvalidate_lightweightが
        例外を投げ、それ以降は成功するように状態を持つ。
        """
        self.calls = 0
        self._failures = failures_then_success

    def validate_lightweight(self, target, *, require_foreground):
        """呼び出し回数を数え、規定回数までは喪失例外を投げる。

        WindowLocator.validate_lightweightの代わりに呼ばれるdouble本体。
        規定回数を超えると何もせず正常終了(=フォアグラウンド確認成功)する。
        """
        self.calls += 1
        if self.calls <= self._failures:
            raise TargetWindowForegroundLost("still lost")


def test_wait_for_foreground_retries_until_success():
    """規定回数だけフォアグラウンド喪失が続いても復帰を待てることを確認する。

    起動直後にオペレーターがウィンドウを前面に出すまでの猶予として、
    _wait_for_foregroundが例外で落ちずにポーリングし続け、最終的に
    復帰した時点でTrueを返すことを見る。
    """
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
    """待機中にCtrl+Cされた場合は例外を投げずFalseで抜けることを確認する。

    フォアグラウンドが永遠に復帰しない状況(failures_then_successを
    十分大きくして再現)でも、interruptedフラグが立っていれば
    無限リトライにならずFalseを返して呼び出し元へ制御を戻すことを見る。
    """
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
    """フォアグラウンド消失以外の状態異常はリトライせず即座に伝播することを確認する。

    resizeなどのfail-closed対象はフォアグラウンド待機のリトライ対象では
    ないため、TargetWindowForegroundLost以外の状態異常はその場で
    呼び出し元へ伝播しなければならないことを見る。
    """

    class _FatalLocator:
        """validate_lightweightで常にfail-closedな状態異常を返すdouble。"""

        def validate_lightweight(self, target, *, require_foreground):
            """フォアグラウンド喪失ではない致命的な状態異常を常に投げる。"""
            raise TargetWindowStateError("resize")

    with pytest.raises(TargetWindowStateError):
        capture_survivors._wait_for_foreground(
            _FatalLocator(), object(), threading.Event(), poll_interval_sec=0
        )


_PAUSE = "PAUSE"
_RESUME = "RESUME"


class _FakeCaptureSession:
    """_capture_liveをCaptureSessionの実装なしで検証するための最小dummy。

    locator/target/start/capture_next/closeだけをduck-typingで
    提供し、start呼び出し有無・close呼び出し有無を観測する。
    """

    def __init__(self, frames_to_yield=(), *, foreground_ok=True):
        """収録済みにするframe列とforeground状態、観測用フラグを初期化する。

        locator/targetは自分自身とダミーオブジェクトで済ませ、
        started/closedでstart()・close()が呼ばれたかを外から確認できるようにする。
        frames_to_yieldに_PAUSE/_RESUMEマーカーを混ぜると、そのtickで
        pausedプロパティだけ切り替えてNoneを返す(実CaptureSessionの
        一時停止/復帰tickを模擬する)。
        """
        self.locator = self
        self.target = object()
        self._frames = list(frames_to_yield)
        self._foreground_ok = foreground_ok
        self.started = False
        self.closed = False
        self.paused = False

    def validate_lightweight(self, target, *, require_foreground):
        """foreground_okがFalseの間だけフォアグラウンド喪失を模擬する。

        WindowLocator.validate_lightweightの代わりに呼ばれ、
        _wait_for_foregroundのポーリング対象として振る舞う。
        """
        if not self._foreground_ok:
            raise TargetWindowForegroundLost("still lost")

    def start(self):
        """実キャプチャ開始の代わりにstartedフラグを立てるだけのdouble。"""
        self.started = True

    def capture_next(self):
        """用意したframeを1つずつ払い出し、尽きたらNoneを返す。

        _capture_liveのwhileループが「frameが来ないtick」を経験できるよう、
        frames_to_yieldを使い切った後はNoneを返し続ける。_PAUSE/_RESUME
        マーカーはpausedを切り替えるだけでframeは返さない。
        """
        if self._frames:
            item = self._frames.pop(0)
            if item == _PAUSE:
                self.paused = True
                return None
            if item == _RESUME:
                self.paused = False
                return None
            return item
        return None

    def close(self):
        """実バックエンド解放の代わりにclosedフラグを立てるだけのdouble。"""
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
    """フォアグラウンド確認後にframeをyieldし、deadline到達後は必ずcloseすることを確認する。

    中断が一度も起きない正常系で、用意したframeがそのままyieldされ、
    ended_reasonがduration_elapsedになり、started_atが記録されることを見る。
    """
    session = _FakeCaptureSession([_frame(0, 100)])
    interrupted = threading.Event()
    stats: dict = {}

    frames = list(capture_survivors._capture_live(session, 0.05, interrupted, stats))

    assert [frame.session_frame_index for frame in frames] == [0]
    assert session.started is True
    assert session.closed is True
    assert stats["ended_reason"] == "duration_elapsed"
    assert "started_at" in stats


def test_capture_live_logs_pause_and_resume_transitions_to_stderr(capsys):
    """foreground一時停止・復帰のtick遷移でstderrへログが出ることを確認する。

    以前はsession.pausedの遷移がコンソールへ一切出ず、operatorが
    alt-tabによる一時停止・復帰を実行結果から確認できなかった。
    frameを伴わないtickでも遷移を検知してログすることを見る。
    """
    session = _FakeCaptureSession([_PAUSE, _RESUME, _frame(0, 100)])
    interrupted = threading.Event()
    stats: dict = {}

    frames = list(capture_survivors._capture_live(session, 0.5, interrupted, stats))

    assert [frame.session_frame_index for frame in frames] == [0]
    err = capsys.readouterr().err
    assert "capture paused" in err
    assert "capture resumed" in err
    assert err.index("capture paused") < err.index("capture resumed")


def test_capture_live_stops_promptly_when_interrupted_while_foreground_paused():
    """frameが来ないtickが続く間でもinterruptedを即座に検知して抜けることを確認する(M8回帰)。

    実運用ではCtrl+Cを押すためにコンソールを前面化すると、対象ゲームは
    foregroundを失い一時停止(capture_nextがNoneを返し続ける)に入る。
    このtick中に立てたinterruptedが、duration_secの満了を待たずに
    即座に収録を終わらせることを確認する。以前の実装はwhileループの
    条件にinterruptedを含めておらず、1回目のCtrl+Cが無視されていた。
    """
    session = _FakeCaptureSession()  # frames_to_yield=()なのでcapture_nextは常にNoneを返す
    interrupted = threading.Event()
    stats: dict = {}
    timer = threading.Timer(0.05, interrupted.set)
    timer.start()
    try:
        started = time.monotonic()
        frames = list(capture_survivors._capture_live(session, 3.0, interrupted, stats))
        elapsed = time.monotonic() - started
    finally:
        timer.cancel()

    assert frames == []
    assert elapsed < 1.0  # duration_sec=3.0よりずっと早く終わる
    assert session.closed is True
    assert stats["ended_reason"] == "interrupted"


def test_graceful_interrupt_sets_flag_without_raising_on_first_sigint():
    """1回目のCtrl+Cは例外を投げずフラグを立てるだけであることを確認する。

    KeyboardInterruptがそのまま飛んで収録ループを中断させないように、
    withブロック内でSIGINTを発生させてもflagが立つだけで例外にならないことを見る。
    """
    flag = threading.Event()
    with capture_survivors._graceful_interrupt(flag):
        signal.raise_signal(signal.SIGINT)
        assert flag.is_set()


def test_graceful_interrupt_escalates_to_keyboardinterrupt_on_second_sigint():
    """2回目のCtrl+Cは本来のKeyboardInterruptへエスケープすることを確認する。

    1回目で収録済み分を安全に止められなかった場合の最終手段として、
    2回目のCtrl+Cだけは通常どおりKeyboardInterruptを送出させることを見る。
    """
    flag = threading.Event()
    with pytest.raises(KeyboardInterrupt):
        with capture_survivors._graceful_interrupt(flag):
            signal.raise_signal(signal.SIGINT)
            signal.raise_signal(signal.SIGINT)


def test_graceful_interrupt_restores_previous_handler_on_exit():
    """with文を抜けた後は元のSIGINTハンドラへ戻すことを確認する。

    _graceful_interruptが差し替えたSIGINTハンドラを、収録終了後も
    差し替えたまま放置してpytest本体のCtrl+C挙動を壊さないことを見る。
    """
    previous = signal.getsignal(signal.SIGINT)
    with capture_survivors._graceful_interrupt(threading.Event()):
        pass
    assert signal.getsignal(signal.SIGINT) is previous


def test_next_auto_session_id_starts_at_0001_for_empty_store(tmp_path):
    """capture_sessionsが存在しない場合はsession-0001から始まることを確認する。

    初回実行でstore_root配下にcapture_sessionsディレクトリすら
    無い状態でも、採番処理が例外にならず初期値を返すことを見る。
    """
    assert capture_survivors._next_auto_session_id(tmp_path) == "session-0001"


def test_next_auto_session_id_increments_past_existing_max(tmp_path):
    """既存の最大連番より1大きい値を、無関係な名前を無視して採番することを確認する。

    session-0001とsession-0007が混在し、かつパターンに合わない
    ディレクトリも存在する状況で、最大値+1だけを正しく拾えることを見る。
    """
    sessions_dir = tmp_path / "capture_sessions"
    sessions_dir.mkdir()
    (sessions_dir / "session-0001").mkdir()
    (sessions_dir / "session-0007").mkdir()
    (sessions_dir / "not-a-session").mkdir()
    assert capture_survivors._next_auto_session_id(tmp_path) == "session-0008"


def test_main_resolves_session_id_auto_to_first_available_slot(tmp_path, capsys):
    """--session-id autoがmain()経由でsession-0001に解決されることを確認する。

    _next_auto_session_id単体ではなくCLIエントリポイント経由でも
    自動採番が実際に配線されており、結果JSONにその値が出ることを見る。
    """
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
        """_graceful_interruptの代わりに、開始直後から中断済みにするdouble。

        実際にSIGINTを送る代わりにflagを即座にsetすることで、
        「起動直後にCtrl+Cが来た」状況を再現する。
        """
        flag.set()  # 収録開始前から中断済みだった状況を再現する
        yield flag

    def empty_frames(self, duration_sec):
        """FakeCaptureBackend.framesの代わりに1枚も出さないdouble。

        中断が撮影開始前に来た場合を模擬するため、常に空の
        イテレータを返す。
        """
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
        """_graceful_interruptの代わりに、実flagをテスト側から握るためのdouble。

        two_then_interruptが同じflagを外側から立てられるよう、
        受け取ったflagをcaptured辞書へ保存するだけの入れ物にする。
        """
        captured["flag"] = flag
        yield flag

    def two_then_interrupt(self, duration_sec):
        """1枚目を出した直後に中断フラグを立て、2枚目も出す2枚構成のdouble。

        「撮影中に中断されたが、最後の1枚まで書き込みは続く」という
        main()側の部分公開ロジックを検証するための入力源。
        """
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


def test_main_closes_threaded_writer_and_cleans_tmp_when_live_capture_raises(
    tmp_path, monkeypatch
):
    """live収録中の例外でもThreadedFrameWriterが閉じられtmpが空になることを確認する(回帰)。

    fail-closedなTargetWindowStateError(resizeなど)や2回目Ctrl+Cの
    KeyboardInterruptがフレーム投入中に発生しても、with文により
    ThreadedFrameWriter.close()(executor shutdown)が必ず呼ばれてから
    DatasetWriter.__exit__のrmtreeが走ることを見る。裸のthreaded.close()
    呼び出しに戻すと、例外発生時にclose()がスキップされexecutorが
    生きたままrmtreeと競合する(過去の回帰)。
    """
    close_calls = []
    original_close = capture_survivors.ThreadedFrameWriter.close

    def spy_close(self):
        close_calls.append(True)
        original_close(self)

    def fake_start_live_session():
        return SimpleNamespace(
            target=SimpleNamespace(
                target_profile_hash=capture_survivors.SYNTHETIC_PROFILE_HASH,
                game_build_id=capture_survivors.SYNTHETIC_BUILD_ID,
            )
        )

    def fake_capture_live(session, duration_sec, interrupted, stats):
        yield _frame(0, 100)
        raise TargetWindowStateError("resize mid-capture")

    monkeypatch.setattr(capture_survivors.ThreadedFrameWriter, "close", spy_close)
    monkeypatch.setattr(capture_survivors, "_start_live_session", fake_start_live_session)
    monkeypatch.setattr(capture_survivors, "_capture_live", fake_capture_live)

    result = capture_survivors.main(
        [
            "--store-root", str(tmp_path),
            "--session-id", "live-fail",
            "--duration-sec", "1.0",
        ]
    )

    assert result == 1  # main()のexcept Exceptionへ抜けてエラー終了する
    assert close_calls  # 例外発生後もThreadedFrameWriter.close()が呼ばれた
    tmp_root = tmp_path / ".capture-tmp"
    assert list(tmp_root.iterdir()) == []  # workerがPNG保存を終えてからrmtreeが完了している
