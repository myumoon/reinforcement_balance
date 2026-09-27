"""``controller.py`` へ障害を注入し、fail-closed に止まることを確かめる fault injection test(M12)。

遅い detector・stage 例外・telemetry の disk full・入力 helper の切断を fake 部品へ注入し、
health STOP / 非0終了 / 入力解放 / structured error が必ず起きることを確認します。
さらに入力 helper 側の release audit(別 JSONL)を telemetry と同じ run のものとして関連付けられることを確かめます。
dropped frames・out-of-order・queue full は同一機構なので ``test_controller.py`` の
``test_latest_only_discards_stale_frames_and_counts_drops`` に任せ、ここでは重複させません。
controller hang は ``test_capture_stall_is_detected_by_poll`` が既に扱っています。
"""

from __future__ import annotations

import errno
import json
import time

import pytest

from survivors.controller.controller import EXIT_ERROR, EXIT_HEALTH_STOP, EXIT_OK
from survivors.controller.telemetry import TelemetryWriter
from survivors.input.audit_log import AuditLog
from survivors.input.controller import HelperUnavailable
from survivors.vision.entity_tracker import EntityTracker

from .test_controller import (
    _INPUT_EFFECTS,
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
    assert _stages(rows, "input_release")[0]["payload"] == {"released": True}
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


def test_release_observer_audit_correlates_with_telemetry(tmp_path, monkeypatch) -> None:
    """helper 側 release audit を、命名規約と shutdown 付近の時刻近接で telemetry へ関連付けられる。

    audit path は ``run_survivors_controller.py`` と同じく telemetry の stem に
    ``.input_audit.jsonl`` を付けて作ります。emergency release の呼び出しで実 ``AuditLog`` へ
    helper ``_release()`` と同じ形の "release" event を書き、その monotonic 時刻が
    telemetry の queue_drain と input_release record の間に入ることを確認します。
    """
    parts = _build(tmp_path, "live", [0])
    audit_path = parts.path.with_name(parts.path.stem + ".input_audit.jsonl")
    audit = AuditLog(audit_path)
    original = FakeInput.emergency_release

    def audited_release(self):
        released = original(self)
        audit.write(
            "release", sequence=None, reason="emergency",
            release_timestamp_ns=time.time_ns(), release_monotonic_ns=parts.clock(),
        )
        return released

    monkeypatch.setattr(FakeInput, "emergency_release", audited_release)
    assert parts.controller.run(max_frames=1) == EXIT_OK

    assert audit_path.parent == parts.path.parent and audit_path.name == "telemetry.input_audit.jsonl"
    (release,) = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
    assert release["event"] == "release" and release["reason"] == "emergency"
    rows = _rows(parts)
    (drain,) = _stages(rows, "queue_drain")
    (input_release,) = _stages(rows, "input_release")
    (shutdown,) = _stages(rows, "shutdown")
    assert input_release["payload"] == {"released": True}
    released_ns = release["release_monotonic_ns"]
    assert drain["timestamp_ns"] < released_ns < input_release["timestamp_ns"] < shutdown["timestamp_ns"]
    assert input_release["timestamp_ns"] - released_ns <= 2 * MS
