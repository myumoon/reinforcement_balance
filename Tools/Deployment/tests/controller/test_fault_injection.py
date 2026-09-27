"""``controller.py`` へ障害を注入し、fail-closed に止まることを確かめる fault injection test(M12)。

遅い detector・stage 例外・telemetry の disk full・入力 helper の切断を fake 部品へ注入し、
health STOP / 非0終了 / 入力解放 / structured error が必ず起きることを確認します。
さらに実 ``HelperRuntime`` が書く release audit(別 JSONL)を、helper と同じ時計の値で
telemetry の input_release 行へ関連付けられることを確かめます。
dropped frames・out-of-order・queue full は同一機構なので ``test_controller.py`` の
``test_latest_only_discards_stale_frames_and_counts_drops`` に任せ、ここでは重複させません。

controller hang(frame が来ない / stage が長時間ブロックする)は、``test_controller.py`` の
``test_capture_stall_is_detected_by_poll``(exit code と理由だけ)に加え、本ファイルの
``test_capture_stall_releases_input_and_completes_shutdown`` と
``test_blocked_stage_is_stopped_by_capture_gap_on_next_frame`` で health 行・入力解放・shutdown 完了まで検証します。
peer death(controller が死んで lease を更新しなくなる / pipe が閉じる)は controller 側から注入できない
別プロセスの安全網なので、``tests/input/test_helper_faults.py`` の
``test_helper_subprocess_releases_expired_lease_with_bounded_observed_latency``(lease 失効で独立 helper が解放)と
``test_controller_atexit_close_pipe_triggers_helper_exit_and_release``(pipe close で helper が解放)が扱います。
実 capture の focus 喪失(例外で伝わる)は ``test_real_capture_focus_loss_is_health_stop`` が扱います。
"""

from __future__ import annotations

import errno
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from survivors.capture.frame_capture import CaptureSession
from survivors.capture.window_locator import MonitorInfo, TargetWindowPolicy, WindowLocator
from survivors.controller.controller import EXIT_ERROR, EXIT_HEALTH_STOP, EXIT_OK
from survivors.controller.telemetry import TelemetryWriter
from survivors.input.audit_log import AuditLog
from survivors.input.controller import HelperUnavailable
from survivors.input.dry_run_backend import DryRunBackend
from survivors.input.helper import HelperRuntime
from survivors.input.lease_protocol import LeaseValidator
from survivors.target_profile import load_target_profile
from survivors.vision.entity_tracker import EntityTracker

from .test_controller import (
    _INPUT_EFFECTS,
    _PIXELS,
    MS,
    FakeAssembler,
    FakeDetector,
    FakeHud,
    FakeInput,
    FakeRuntime,
    _build,
    _rows,
    _stages,
)

_FULL_SHUTDOWN = ["capture_stop", "queue_drain", "input_release", "artifact_finalize"]


@pytest.mark.parametrize(("delay_ms", "expected"), [(0, EXIT_OK), (150, EXIT_HEALTH_STOP)])
def test_slow_detector_stops_on_perception_p99_after_min_samples(tmp_path, monkeypatch, delay_ms, expected) -> None:
    """遅い detector は perception p99 で STOP し、その判定は min_samples(100)件目で初めて起きる。

    detector が呼ばれるたびに共有 Clock を delay_ms 進め、実 sleep なしで遅延を模擬します。
    150ms は p99 閾値(110ms)を超えるが capture gap 閾値(200ms)は超えない値です。
    delay 0 の対照では同じ 150 frame が正常終了し、STOP が遅延だけで起きていることを示します。
    """
    parts = _build(tmp_path, "live", list(range(150)))
    original = FakeDetector.infer

    def slow_infer(self, frame_bgr, *, score_threshold):
        parts.clock.now += delay_ms * MS
        return original(self, frame_bgr, score_threshold=score_threshold)

    monkeypatch.setattr(FakeDetector, "infer", slow_infer)
    assert parts.controller.run(max_frames=150) == expected
    rows = _rows(parts)
    if expected == EXIT_OK:
        assert parts.detector.calls == 150
        assert _stages(rows, "health") == []
        return
    assert parts.controller.exit_reason == "health_stop:perception_p99"
    # 99 件目までは評価しないので policy まで進み、100 件目で初めて STOP する(off-by-one 検出)。
    assert parts.detector.calls == 100
    assert len(_stages(rows, "policy")) == 99
    (health,) = _stages(rows, "health")
    failure = health["payload"]["first_failure"]
    assert failure["reason"] == "perception_p99"
    assert failure["observed"] > failure["threshold"] == 110 * MS
    assert parts.events[-1] == "input_release"
    assert parts.controller.shutdown_steps == _FULL_SHUTDOWN


@pytest.mark.parametrize(
    ("stage", "owner", "method"),
    [
        ("tracker", EntityTracker, "update"),
        ("hud_parser", FakeHud, "parse"),
        ("obs", FakeAssembler, "assemble"),
        ("policy", FakeRuntime, "decide"),
    ],
)
def test_non_detector_stage_exception_takes_inference_error_path(tmp_path, monkeypatch, stage, owner, method) -> None:
    """detector 以外の stage 例外も同じ ``_StageError`` 経路で health STOP・入力解放・非0終了になる。

    各 stage の実装を例外を出す関数へ差し替え、error record の stage 名と
    health の first_failure detail に失敗した stage が残ることを確認します。
    """

    def boom(*_args, **_kwargs):
        raise RuntimeError(f"{stage} exploded")

    monkeypatch.setattr(owner, method, boom)
    parts = _build(tmp_path, "live", [0], effects=_INPUT_EFFECTS)
    assert parts.controller.run(max_frames=1) == EXIT_HEALTH_STOP
    assert parts.controller.exit_reason == "health_stop:inference_error"
    rows = _rows(parts)
    (error,) = _stages(rows, "error")
    assert error["payload"] == {"stage": stage, "type": "RuntimeError", "message": f"{stage} exploded"}
    assert error["correlation_id"] == "session-1:0"
    (health,) = _stages(rows, "health")
    assert health["payload"]["first_failure"]["detail"] == f"{stage}: RuntimeError: {stage} exploded"
    assert _stages(rows, "effect") == [] and parts.input.sent == []
    assert parts.events[-1] == "input_release"
    assert parts.controller.shutdown_steps == _FULL_SHUTDOWN


def _install_disk_full(monkeypatch, *, persistent: bool) -> None:
    """frame 1 の detector record 書き込みで ENOSPC を起こす write_stage を差し込む。

    persistent=False なら1回だけ失敗して空きが戻った状況、True なら以後すべて失敗し続ける状況です。
    """
    original = TelemetryWriter.write_stage
    state = {"full": False}

    def write_stage(self, stage, **kwargs):
        if state["full"] or (stage == "detector" and kwargs["correlation_id"] == "session-1:1"):
            state["full"] = persistent
            raise OSError(errno.ENOSPC, "No space left on device")
        return original(self, stage, **kwargs)

    monkeypatch.setattr(TelemetryWriter, "write_stage", write_stage)


_ENOSPC = {"type": "OSError", "message": str(OSError(errno.ENOSPC, "No space left on device"))}


def test_transient_disk_full_is_recorded_and_shutdown_completes(tmp_path, monkeypatch) -> None:
    """run 中の telemetry 書き込み失敗は握り潰さず structured error になり、shutdown record にも残る。

    書き込み失敗は frame 処理を止めて非0終了させますが、shutdown の全段と入力解放は続行されます。
    """
    _install_disk_full(monkeypatch, persistent=False)
    parts = _build(tmp_path, "live", [0, 1, 2])
    assert parts.controller.run(max_frames=3) == EXIT_ERROR
    assert parts.controller.exit_reason == "exception:OSError"
    assert parts.controller.errors == [{"stage": "controller", **_ENOSPC}]
    assert parts.detector.calls == 2
    rows = _rows(parts)
    (error,) = _stages(rows, "error")
    assert error["payload"] == {"stage": "controller", **_ENOSPC}
    (shutdown,) = _stages(rows, "shutdown")
    assert shutdown["payload"]["exit_code"] == EXIT_ERROR
    assert shutdown["payload"]["errors"] == parts.controller.errors
    assert _stages(rows, "input_release")[0]["payload"]["released"] is True
    assert parts.controller.shutdown_steps == _FULL_SHUTDOWN


def test_persistent_disk_full_still_releases_input_and_exits_nonzero(tmp_path, monkeypatch) -> None:
    """disk full が続いても shutdown の各段は止まらず、入力解放して非0終了する。

    telemetry に書けない失敗はすべて ``errors`` に積まれ、失敗以降の record はファイルに残りません。
    """
    _install_disk_full(monkeypatch, persistent=True)
    parts = _build(tmp_path, "live", [0, 1, 2])
    assert parts.controller.run(max_frames=3) == EXIT_ERROR
    assert parts.controller.shutdown_steps == _FULL_SHUTDOWN
    assert parts.events[-1] == "input_release"
    errors = parts.controller.errors
    assert [error["stage"] for error in errors] == [
        "controller", "telemetry", "telemetry", "telemetry", "telemetry", "artifact_finalize", "telemetry",
    ]
    assert all(error["type"] == "OSError" for error in errors)
    assert [row["stage"] for row in _rows(parts)][-1] == "capture"


@pytest.mark.parametrize(("method", "script"), [("send_action", [0]), ("send_ui_key", [0, 1])])
def test_input_helper_disconnect_releases_and_exits_nonzero(tmp_path, monkeypatch, method, script) -> None:
    """入力 helper の死亡(HelperUnavailable)は入力送信でも解放でも structured error になり非0終了する。

    ``InputLeaseController`` は helper の死亡・IPC 失敗・timeout を ``HelperUnavailable`` へ正規化するため、
    それを送出する fake で peer death を模擬します。入力 effect は ``_timed`` の外で実行されるので、
    stage exception(health STOP=2)ではなく controller 例外(EXIT_ERROR=3)として止まります。
    死んだ helper への emergency release も失敗し、その失敗も shutdown record に残ります。
    """

    def dead_send(self, *_args):
        raise HelperUnavailable("input helper died")

    def dead_release(self):
        self._events.append("input_release")
        raise HelperUnavailable("input helper is unavailable")

    monkeypatch.setattr(FakeInput, method, dead_send)
    monkeypatch.setattr(FakeInput, "emergency_release", dead_release)
    parts = _build(tmp_path, "live", script, effects=_INPUT_EFFECTS)
    assert parts.controller.run(max_frames=len(script)) == EXIT_ERROR
    assert parts.controller.exit_reason == "exception:HelperUnavailable"
    assert parts.controller.errors == [
        {"stage": "controller", "type": "HelperUnavailable", "message": "input helper died"},
        {"stage": "input_release", "type": "HelperUnavailable", "message": "input helper is unavailable"},
    ]
    rows = _rows(parts)
    # 失敗した入力 effect は executed として記録されない。
    assert all(row["payload"]["kind"] == "move" for row in _stages(rows, "effect"))
    assert _stages(rows, "input_release")[0]["payload"] == {"failed": True}
    (shutdown,) = _stages(rows, "shutdown")
    assert shutdown["payload"]["exit_code"] == EXIT_ERROR
    assert shutdown["payload"]["errors"] == parts.controller.errors
    assert parts.events[-1] == "input_release"
    assert parts.controller.shutdown_steps == _FULL_SHUTDOWN


# helper audit と input_release 行の実時計差の許容上限。helper は release 開始時に time.monotonic_ns()、
# controller は emergency_release() が戻った直後に同じ関数を呼ぶので、差は「0 以上・release 1回分の所要時間
# + 時計分解能」に収まる。実測(2026-09-28、reinbalance env、Windows): time.get_clock_info("monotonic").resolution
# = 15.625ms、連続2回呼び出しの差は monotonic/time_ns とも 0ms(2000回の最大)。つまり相関差は 0〜約16ms の範囲で、
# CI の scheduling 揺れを見込んで 100ms とする(perf_counter との差 23.6〜29.8ms のドメインずれは起きない)。
_RELEASE_CORRELATION_TOLERANCE_NS = 100 * MS


def test_release_observer_audit_correlates_with_telemetry(tmp_path, monkeypatch) -> None:
    """実 helper の release audit を、命名規約と helper と同じ時計の時刻で telemetry へ関連付けられる。

    audit path は ``run_survivors_controller.py`` と同じく telemetry の stem に ``.input_audit.jsonl`` を付けます。
    emergency release では実 ``HelperRuntime`` + ``DryRunBackend``(test_only) + 実 ``AuditLog`` を動かし、
    helper 自身が ``time.monotonic_ns()``/``time.time_ns()`` で "release" event を書きます。
    telemetry 時計(テストでは fake、本番は perf_counter_ns)は helper と別ドメインなので比較に使わず、
    input_release 行が同じ時計で持つ ``release_monotonic_ns``/``release_timestamp_ns`` と突き合わせます。
    """
    parts = _build(tmp_path, "live", [0])
    audit_path = parts.path.with_name(parts.path.stem + ".input_audit.jsonl")
    runtime = HelperRuntime(
        LeaseValidator("1" * 32, "a" * 64, "b" * 64, 42, 84),
        DryRunBackend(foreground_pid=42, foreground_hwnd=84, focused=True),
        AuditLog(audit_path),
    )

    def helper_release(self):
        self._events.append("input_release")
        runtime.emergency_release()
        return True

    monkeypatch.setattr(FakeInput, "emergency_release", helper_release)
    assert parts.controller.run(max_frames=1) == EXIT_OK

    assert audit_path.parent == parts.path.parent and audit_path.name == "telemetry.input_audit.jsonl"
    (release,) = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
    assert release["event"] == "release" and release["reason"] == "emergency"
    (input_release,) = _stages(_rows(parts), "input_release")
    payload = input_release["payload"]
    assert payload["released"] is True
    for key in ("release_monotonic_ns", "release_timestamp_ns"):
        delta = payload[key] - release[key]
        assert 0 <= delta <= _RELEASE_CORRELATION_TOLERANCE_NS, (key, delta)


def test_capture_stall_releases_input_and_completes_shutdown(tmp_path) -> None:
    """controller hang(frame が来なくなる)は poll の capture_gap で STOP し、入力解放と shutdown を完遂する。

    ``test_capture_stall_is_detected_by_poll`` は exit code と理由だけを見るので、ここでは
    health 行の first_failure・入力解放が最後の event であること・shutdown 全段の実行を確認します。
    """
    parts = _build(tmp_path, "live", [0], effects=_INPUT_EFFECTS)
    assert parts.controller.run() == EXIT_HEALTH_STOP
    assert parts.controller.exit_reason == "health_stop:capture_gap"
    (health,) = _stages(_rows(parts), "health")
    failure = health["payload"]["first_failure"]
    assert failure["reason"] == "capture_gap" and failure["observed"] > failure["threshold"] == 200 * MS
    assert parts.events[-1] == "input_release"
    assert parts.controller.shutdown_steps == _FULL_SHUTDOWN


def test_blocked_stage_is_stopped_by_capture_gap_on_next_frame(tmp_path, monkeypatch) -> None:
    """stage が capture_gap 閾値を大きく超えてブロックすると、次の frame で STOP して入力を出さない。

    frame 1 の detector で共有 Clock を 5 秒進め、controller が固まった状況を模擬します。
    latest-only capture が次に渡す frame 2 は前 frame から 5 秒離れているので ingest が capture_gap で STOP し、
    frame 2 では policy/effect へ進まずに入力解放・非0終了になります。
    なお、ブロックしていた frame 1 自身の effect は既に出ています(p99 は min_samples 前は評価しないため)。
    その入力は helper 側の lease 失効で有界に止まります。
    """
    parts = _build(tmp_path, "live", [0, 1, 2], effects=[()] * 3)
    original = FakeDetector.infer

    def hanging_infer(self, frame_bgr, *, score_threshold):
        if self.calls == 1:
            parts.clock.now += 5_000 * MS
        return original(self, frame_bgr, score_threshold=score_threshold)

    monkeypatch.setattr(FakeDetector, "infer", hanging_infer)
    assert parts.controller.run(max_frames=3) == EXIT_HEALTH_STOP
    assert parts.controller.exit_reason == "health_stop:capture_gap"
    rows = _rows(parts)
    (health,) = _stages(rows, "health")
    assert health["payload"]["first_failure"]["reason"] == "capture_gap"
    assert health["payload"]["first_failure"]["observed"] > 5_000 * MS
    assert [row["correlation_id"] for row in _stages(rows, "policy")] == ["session-1:0", "session-1:1"]
    assert parts.events[-1] == "input_release"
    assert parts.controller.shutdown_steps == _FULL_SHUTDOWN


def _capture_fakes():
    """``tests/capture/conftest.py`` の Win32/backend fake を別名で読み込む。

    conftest はディレクトリごとに解決されるため、controller テストからは import できません。
    fake の重複定義を避けるため、ファイルを直接 module として読み込みます。
    """
    path = Path(__file__).resolve().parents[1] / "capture" / "conftest.py"
    spec = importlib.util.spec_from_file_location("_capture_fakes_for_controller", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclass の型解決が sys.modules から module を引くため
    spec.loader.exec_module(module)
    return module


def test_real_capture_focus_loss_is_health_stop(tmp_path) -> None:
    """M5: 実 CaptureSession が focus 喪失で送出する例外は health STOP(focus_lost)・入力解放・exit 2 になる。

    実 ``WindowLocator`` + ``CaptureSession`` に fake Win32 API と fake backend を渡し、
    2 枚目の読み取り直後に前面ウィンドウを別 hwnd へ切り替えます。実 capture は foreground=False の
    frame を返さず ``TargetWindowStateError`` を送出するので、それが controller 例外(exit 3)ではなく
    health STOP として扱われることを確認します。
    """
    fakes = _capture_fakes()
    profile = load_target_profile()
    object.__setattr__(profile, "provenance", "operator-attested")  # frozen bypass: テスト用
    policy = TargetWindowPolicy(
        process_executable="VampireSurvivors.exe", window_class="YYGameMakerYY", window_title="Vampire Survivors",
    )
    window = fakes.FakeWindow(
        hwnd=101, pid=2001, executable=r"C:\Games\Vampire Survivors\VampireSurvivors.exe",
        window_class="YYGameMakerYY", title="Vampire Survivors", client_rect=(0, 0, 1920, 1080),
        monitor=MonitorInfo(
            rect_screen_px=(0, 0, 1920, 1080), device_name=r"\\.\DISPLAY1", primary=True, dxgi_output_idx=0,
        ),
    )
    api = fakes.FakeWin32Api(
        [window], foreground_hwnd=window.hwnd, exe_hash=profile.sections["build"]["executable_hash"],
    )
    reads = [0]

    def lose_focus_on_second_read():
        reads[0] += 1
        if reads[0] >= 2:
            api.foreground_hwnd = 999

    def make_session(clock):
        locator = WindowLocator(api, profile, policy)
        base = clock.now
        backend = fakes.FakeCaptureBackend(
            [(_PIXELS, base + 1 * MS), (_PIXELS, base + 2 * MS)], after_read=lose_focus_on_second_read,
        )
        return CaptureSession(locator, locator.locate(), backend)

    parts = _build(tmp_path, "live", [], capture_factory=make_session)
    assert parts.controller.run(max_frames=5) == EXIT_HEALTH_STOP
    assert parts.controller.exit_reason == "health_stop:focus_lost"
    rows = _rows(parts)
    assert _stages(rows, "error") == []
    (health,) = _stages(rows, "health")
    failure = health["payload"]["first_failure"]
    assert failure["reason"] == "focus_lost" and failure["detail"] == "target window lost foreground"
    assert [row["correlation_id"] for row in _stages(rows, "capture")] == ["session-1:0"]
    assert parts.events[-1] == "input_release"
    assert parts.controller.shutdown_steps == _FULL_SHUTDOWN
