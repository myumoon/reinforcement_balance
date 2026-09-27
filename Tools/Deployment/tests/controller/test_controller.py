"""``controller.py`` の bounded pipeline・shutdown 順序・shadow/live 分離を検証する。

capture から effect までを fake stage でつなぎ、実際の TelemetryWriter・HealthMonitor・
EntityTracker・execute_effect を通して、M1/M2/M3/M6/M7 と I1/I3/I7 を確認します。
shadow mode では入力 adapter も execute_effect も一切呼ばれないことを、呼ばれたら失敗する罠で確かめます。
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from reinbalance_survivors_contracts.deploy_obs import DeployObsSchema, DeployObservation
from survivors.capture.captured_frame import CapturedFrame
from survivors.capture.frame_capture import LatestFrameQueue
from survivors.controller import controller as controller_module
from survivors.controller.controller import (
    EXIT_ERROR,
    EXIT_HEALTH_STOP,
    EXIT_OK,
    EXIT_TERMINAL_FAILURE,
    SurvivorsController,
)
from survivors.controller.health_monitor import HealthMonitor
from survivors.controller.state_machine import CampaignRunMode, ControllerState
from survivors.controller.telemetry import TelemetrySessionHeader, TelemetryWriter
from survivors.controller.ui_navigation import Effect
from survivors.vision.entity_tracker import EntityTracker
from survivors.vision.world_detector import DetectionResult

MS = 1_000_000
_PIXELS = np.zeros((1080, 1920, 4), dtype=np.uint8)
_SCHEMA = DeployObsSchema.default_v1()
_MODEL_HASHES = {"combat_model": "b" * 64}


def _valid_obs(timestamp_ns: int) -> DeployObservation:
    """schema に適合する中立値の観測を作る。

    health の observation_valid が True になる最小の観測です。
    """
    values = np.zeros(_SCHEMA.dim, dtype=np.float32)
    for field in _SCHEMA.fields:
        offset, size = _SCHEMA.layout[field.name]
        values[offset:offset + size] = field.neutral
    return DeployObservation(
        values=values, validity=np.zeros(_SCHEMA.dim), age=np.ones(_SCHEMA.dim),
        schema_hash=_SCHEMA.schema_hash, timestamp_ns=timestamp_ns,
    )


class Clock:
    """呼ぶたびに 1ms 進む単調時計。

    テストの時刻を決定的にし、health の gap/latency を制御します。
    """

    def __init__(self) -> None:
        """開始時刻を 1 秒に置く。"""
        self.now = 1_000 * MS

    def __call__(self) -> int:
        """1ms 進めた時刻を返す。"""
        self.now += MS
        return self.now


class FakeCapture:
    """script どおりに frame を LatestFrameQueue へ積む capture。

    script の要素は frame 番号、(番号, foreground) の組、または None(新規 frame なし)です。
    """

    def __init__(self, clock: Clock, script: list, events: list[str]) -> None:
        """script と共有 event 列を保持する。"""
        self.frames = LatestFrameQueue()
        self._clock = clock
        self._script = list(script)
        self._events = events

    def start(self) -> None:
        """開始を記録する。"""
        self._events.append("capture_start")

    def capture_next(self):
        """script の次の要素に応じて frame を1枚積むか None を返す。"""
        item = self._script.pop(0) if self._script else None
        if item is None:
            return None
        index, foreground = item if isinstance(item, tuple) else (item, True)
        frame = CapturedFrame(
            frame_bgra=_PIXELS, captured_monotonic_ns=self._clock(), session_frame_index=index,
            client_rect_screen_px=(0, 0, 1920, 1080), foreground=foreground,
            target_profile_hash="a" * 64, game_build_id="build-1",
        )
        self.frames.put_latest(frame)
        return frame

    def close(self) -> None:
        """停止を記録する(CaptureSession.close と同じく queue は残す側に任せる)。"""
        self._events.append("capture_stop")


class FakeDetector:
    """空の DetectionResult を返し、指定 frame 番号で例外を出す detector。"""

    def __init__(self, fail_on_call: int | None = None) -> None:
        """何回目の呼び出しで失敗させるかを保持する。"""
        self.calls = 0
        self._fail_on_call = fail_on_call

    def infer(self, frame_bgr, *, score_threshold):
        """BGR 3ch 入力を確認して空の検出結果を返す。"""
        assert frame_bgr.shape == (1080, 1920, 3)
        self.calls += 1
        if self.calls == self._fail_on_call:
            raise RuntimeError("detector exploded")
        return DetectionResult(
            boxes_xyxy=np.zeros((0, 4), np.float32), scores=np.zeros(0, np.float32),
            class_ids=np.zeros(0, np.int32), image_width=1920, image_height=1080,
        )


class FakeHud:
    """固定の gameplay HUD を返す parser。"""

    def __init__(self) -> None:
        """呼び出し回数を初期化する。"""
        self.parses = 0
        self.resets = 0

    def reset_temporal_state(self) -> None:
        """arm 時のリセット回数を数える。"""
        self.resets += 1

    def parse(self, frame_bgra, *, session_id, frame_index, captured_monotonic_ns):
        """HUD の最小属性だけを返す。"""
        self.parses += 1
        return SimpleNamespace(screen_state="gameplay", screen_state_confidence=1.0, parser_artifact_hash="d" * 64)


class FakeAssembler:
    """``emit_every`` frame ごとに snapshot を返す assembler(それ以外は None)。"""

    def __init__(self, emit_every: int = 1) -> None:
        """発行間隔を保持する。"""
        self._emit_every = emit_every
        self._calls = 0

    def assemble(self, hud, world, schema, viewport, config):
        """15 Hz cadence を模して間引いた snapshot を返す。"""
        self._calls += 1
        if self._calls % self._emit_every:
            return None
        return SimpleNamespace(
            snapshot_id=f"snap-{self._calls}", frame_id=f"frame-{world.frame_index}",
            screen_state="gameplay", ui_state_key="ui-key", source_content_hash="f" * 64,
            deploy_obs=_valid_obs(world.timestamp_ns),
        )


class FakeDecision:
    """AgentDecision の controller が使う属性だけを持つ decision。"""

    def __init__(self, snapshot_id: str) -> None:
        """source snapshot から decision id を作る。"""
        self.decision_id = f"decision-{snapshot_id}"
        self.kind = "move"
        self._snapshot_id = snapshot_id

    def to_wire(self) -> dict:
        """wire 形式を返す。"""
        return {"decision_id": self.decision_id, "kind": self.kind, "source_snapshot_id": self._snapshot_id}

    def decision_hash(self) -> str:
        """decision id 由来の擬似 hash を返す。"""
        return f"hash-{self.decision_id}"


class FakeRuntime:
    """decide 呼び出しと episode reset を記録する policy runtime。"""

    def __init__(self) -> None:
        """記録用の列を初期化する。"""
        self.episode_flags: list[bool] = []
        self.resets = 0

    def reset_episode(self) -> None:
        """LSTM reset 回数を数える。"""
        self.resets += 1

    def decide(self, snapshot, *, now_ns, episode_start):
        """episode_start を記録して move decision を返す。"""
        self.episode_flags.append(episode_start)
        return FakeDecision(snapshot.snapshot_id)


class FakeStateMachine:
    """step ごとに script の effect 列を返す state machine。"""

    def __init__(self, effects: list[tuple[Effect, ...]], terminal_state=None, fail: bool = False) -> None:
        """返す effect 列・terminal state・例外を出すかを保持する。"""
        self.context = SimpleNamespace(state=ControllerState.GAMEPLAY, terminal_state=terminal_state)
        self._effects = list(effects)
        self._fail = fail
        self.armed: list[dict] = []

    def arm(self, **kwargs) -> None:
        """arm 引数を記録する。"""
        self.armed.append(kwargs)

    def step(self, snapshot, decision, *, now_ns, window_focused):
        """script の次の effect 列を返す(尽きたら空)。"""
        if self._fail:
            raise RuntimeError("state machine bug")
        return self._effects.pop(0) if self._effects else ()


class FakeInput:
    """InputLeaseController の公開 API を記録する fake。"""

    def __init__(self, events: list[str], ack: bool = True, release_ok: bool = True) -> None:
        """ack と release の結果を固定する。"""
        self._events = events
        self._ack = ack
        self._release_ok = release_ok
        self.sent: list = []

    def send_action(self, action_index: int) -> bool:
        """移動入力を記録する。"""
        self.sent.append(("move", action_index))
        return self._ack

    def send_ui_click(self, x: float, y: float) -> bool:
        """UI click を記録する。"""
        self.sent.append(("click", x, y))
        return self._ack

    def send_ui_key(self, key: str) -> bool:
        """UI key を記録する。"""
        self.sent.append(("key", key))
        return self._ack

    def emergency_release(self) -> bool:
        """解放を記録する。"""
        self._events.append("input_release")
        return self._release_ok


def _build(tmp_path, mode, script, *, effects=(), terminal_state=None, detector=None, sm_fail=False,
           emit_every=1, input_ack=True, release_ok=True, name="telemetry.jsonl"):
    """fake stage をつないだ controller と観察用の部品を返す。

    mode ごとに同じ部品構成で controller を作り、live だけ FakeInput を渡します。
    """
    clock = Clock()
    events: list[str] = []
    header = TelemetrySessionHeader(
        session_id="session-1", mode=mode, target_profile_hash="a" * 64, game_build_id="build-1",
        controller_build_id="controller-1", artifact_hashes=_MODEL_HASHES, host={"os": "test"},
        device={"inference": "cpu"}, dependency_versions={"python": "3.11"},
    )
    parts = SimpleNamespace(
        clock=clock, events=events, path=tmp_path / name,
        capture=FakeCapture(clock, script, events), detector=detector or FakeDetector(),
        hud=FakeHud(), runtime=FakeRuntime(),
        sm=FakeStateMachine(list(effects), terminal_state=terminal_state, fail=sm_fail),
        input=FakeInput(events, ack=input_ack, release_ok=release_ok) if mode == "live" else None,
    )
    parts.controller = SurvivorsController(
        mode=mode, session_id="session-1", capture=parts.capture, detector=parts.detector,
        tracker=EntityTracker({i: 5 for i in range(12)}, 0.7, 0.6, 0.9), hud_parser=parts.hud,
        assembler=FakeAssembler(emit_every), runtime=parts.runtime, state_machine=parts.sm,
        health=HealthMonitor(), telemetry=TelemetryWriter(parts.path, header), schema=_SCHEMA,
        model_hashes=_MODEL_HASHES, input_controller=parts.input, clock_ns=clock, sleep=lambda _: None,
    )
    return parts


def _rows(parts) -> list[dict]:
    """telemetry の stage record だけを読み出す。"""
    lines = parts.path.read_text(encoding="utf-8").splitlines()
    return [row for row in map(json.loads, lines) if row["event"] == "stage"]


def _stages(rows, stage: str) -> list[dict]:
    """指定 stage の record だけを返す。"""
    return [row for row in rows if row["stage"] == stage]


_INPUT_EFFECTS = (
    (Effect(kind="move", action_index=3),),
    (Effect(kind="ui_key", key="ENTER"), Effect(kind="release_all", reason="test")),
)


def test_shadow_runs_full_pipeline_without_touching_input(tmp_path, monkeypatch) -> None:
    """M3: shadow は全 stage を実行するが execute_effect/InputLeaseController を一切呼ばない。"""
    from survivors.input import controller as input_module

    def trap(*_args, **_kwargs):
        raise AssertionError("shadow mode must not touch the input path")

    monkeypatch.setattr(controller_module, "execute_effect", trap)
    monkeypatch.setattr(input_module.InputLeaseController, "__init__", trap)
    parts = _build(tmp_path, "shadow", [0, 1], effects=_INPUT_EFFECTS)

    assert parts.controller.run(max_frames=2) == EXIT_OK
    rows = _rows(parts)
    per_frame = [row["stage"] for row in rows if row["correlation_id"] == "session-1:0"]
    assert per_frame == ["capture", "detector", "tracker", "hud_parser", "obs", "policy", "state_machine", "effect"]
    effects = _stages(rows, "effect")
    assert [(row["payload"]["kind"], row["payload"]["disposition"], row["payload"]["ack"]) for row in effects] == [
        ("move", "proposed", None), ("ui_key", "proposed", None), ("release_all", "proposed", None),
    ]
    assert "input_release" not in parts.events


def test_mode_and_input_adapter_must_match(tmp_path) -> None:
    """M3: shadow に入力 adapter を渡すと拒否し、live は adapter 必須。"""
    parts = _build(tmp_path, "shadow", [])
    kwargs = dict(
        session_id="s", capture=None, detector=None, tracker=None, hud_parser=None, assembler=None,
        runtime=None, state_machine=None, health=HealthMonitor(), telemetry=None, schema=_SCHEMA,
        model_hashes=_MODEL_HASHES,
    )
    with pytest.raises(ValueError, match="shadow"):
        SurvivorsController(mode="shadow", input_controller=FakeInput([]), **kwargs)
    with pytest.raises(ValueError, match="live"):
        SurvivorsController(mode="live", **kwargs)
    parts.controller.run(max_frames=0)


def test_live_executes_input_effects_and_shares_shadow_path(tmp_path) -> None:
    """I1: live は入力を送るが、stage 列と correlation は shadow と完全に同じ。"""
    shadow = _build(tmp_path, "shadow", [0, 1], effects=_INPUT_EFFECTS, name="shadow.jsonl")
    live = _build(tmp_path, "live", [0, 1], effects=_INPUT_EFFECTS, name="live.jsonl")
    assert shadow.controller.run(max_frames=2) == EXIT_OK
    assert live.controller.run(max_frames=2) == EXIT_OK

    key = lambda rows: [(row["stage"], row["correlation_id"]) for row in rows]  # noqa: E731
    assert key(_rows(shadow)) == key(_rows(live))
    assert live.input.sent == [("move", 3), ("key", "ENTER")]
    assert [row["payload"]["disposition"] for row in _stages(_rows(live), "effect")] == ["executed"] * 3
    assert all(row["payload"]["ack"] is True for row in _stages(_rows(live), "effect"))


def test_latest_only_discards_stale_frames_and_counts_drops(tmp_path) -> None:
    """M1: 逆順の古い frame は捨て、番号の飛びは drop として記録する。"""
    parts = _build(tmp_path, "shadow", [0, 2, 1, 3])
    assert parts.controller.run(max_frames=3) == EXIT_OK
    rows = _rows(parts)
    captures = _stages(rows, "capture")
    assert [row["payload"]["frame_index"] for row in captures] == [0, 2, 3]
    assert [row["payload"]["dropped_frames"] for row in captures] == [0, 1, 0]
    discards = _stages(rows, "discard")
    assert [row["correlation_id"] for row in discards] == ["session-1:1"]
    assert parts.detector.calls == 3


def test_parser_every_frame_policy_only_on_snapshot(tmp_path) -> None:
    """M1: HUD parser は毎 frame、policy は assembler が snapshot を返した tick だけ。"""
    parts = _build(tmp_path, "shadow", [0, 1, 2, 3], emit_every=2)
    assert parts.controller.run(max_frames=4) == EXIT_OK
    rows = _rows(parts)
    assert parts.hud.parses == 4
    assert [row["correlation_id"] for row in _stages(rows, "policy")] == ["session-1:1", "session-1:3"]


def test_every_effect_traces_to_frame_obs_and_model(tmp_path) -> None:
    """M7: effect record から source frame・obs hash・decision・model hash を辿れる。"""
    parts = _build(tmp_path, "shadow", [0], effects=_INPUT_EFFECTS[:1])
    assert parts.controller.run(max_frames=1) == EXIT_OK
    rows = _rows(parts)
    (effect,) = _stages(rows, "effect")
    (obs,) = _stages(rows, "obs")
    (policy,) = _stages(rows, "policy")
    source = effect["payload"]["source"]
    assert effect["correlation_id"] == obs["correlation_id"] == policy["correlation_id"] == "session-1:0"
    assert source["frame_index"] == 0
    assert source["obs_hash"] == obs["payload"]["obs_hash"] and len(source["obs_hash"]) == 64
    assert source["snapshot_id"] == obs["payload"]["snapshot_id"]
    assert source["decision_hash"] == policy["payload"]["decision_hash"]
    assert source["model_hashes"] == _MODEL_HASHES == policy["payload"]["model_hashes"]


def test_graceful_shutdown_order(tmp_path) -> None:
    """M2: capture stop → queue drain → input release → artifact finalize の順。"""
    parts = _build(tmp_path, "live", [0])
    assert parts.controller.run(max_frames=1) == EXIT_OK
    assert parts.controller.shutdown_steps == ["capture_stop", "queue_drain", "input_release", "artifact_finalize"]
    assert parts.events == ["capture_start", "capture_stop", "input_release"]
    tail = [row["stage"] for row in _rows(parts)][-4:]
    assert tail == ["capture_stop", "queue_drain", "input_release", "shutdown"]
    assert _stages(_rows(parts), "shutdown")[0]["payload"]["exit_code"] == EXIT_OK
    with pytest.raises(ValueError):
        parts.controller._telemetry.write_stage("x", correlation_id="c", timestamp_ns=0, latency_ns=0, queue_depth=0)


def test_health_stop_on_focus_loss_releases_and_exits_nonzero(tmp_path) -> None:
    """M6: focus 喪失で health STOP → policy へ進まず、入力解放して非0終了。"""
    parts = _build(tmp_path, "live", [0, (1, False), 2], effects=_INPUT_EFFECTS)
    assert parts.controller.run(max_frames=3) == EXIT_HEALTH_STOP
    assert parts.controller.exit_reason == "health_stop:focus_lost"
    assert "input_release" in parts.events
    rows = _rows(parts)
    (health,) = _stages(rows, "health")
    assert health["payload"]["verdict"] == "stop"
    assert [row["correlation_id"] for row in _stages(rows, "policy")] == ["session-1:0"]
    assert _stages(rows, "capture")[-1]["payload"]["frame_index"] == 1


def test_inference_error_stops_with_structured_error(tmp_path) -> None:
    """M6/I3: detector 例外は1回で health STOP、structured error と入力解放を伴う。"""
    parts = _build(tmp_path, "live", [0, 1], detector=FakeDetector(fail_on_call=2))
    assert parts.controller.run(max_frames=2) == EXIT_HEALTH_STOP
    assert parts.controller.exit_reason == "health_stop:inference_error"
    (error,) = _stages(_rows(parts), "error")
    assert error["payload"] == {"stage": "detector", "type": "RuntimeError", "message": "detector exploded"}
    assert error["correlation_id"] == "session-1:1"
    assert parts.events[-1] == "input_release"


def test_capture_stall_is_detected_by_poll(tmp_path) -> None:
    """M6: frame が途絶えたら poll が capture gap で STOP させる。"""
    parts = _build(tmp_path, "shadow", [0])
    assert parts.controller.run() == EXIT_HEALTH_STOP
    assert parts.controller.exit_reason == "health_stop:capture_gap"


def test_unexpected_exception_releases_and_exits_error(tmp_path) -> None:
    """I3: stage 外の例外も structured error・入力解放・非0終了。"""
    parts = _build(tmp_path, "live", [0], sm_fail=True)
    assert parts.controller.run(max_frames=1) == EXIT_ERROR
    assert parts.controller.errors[0] == {"stage": "controller", "type": "RuntimeError", "message": "state machine bug"}
    assert parts.controller.shutdown_steps[-2:] == ["input_release", "artifact_finalize"]
    assert parts.events[-1] == "input_release"


def test_combat_reset_is_handled_by_controller(tmp_path) -> None:
    """I7: combat_reset は controller が runtime.reset_episode() で処理し、入力は送らない。"""
    parts = _build(tmp_path, "live", [0], effects=[(Effect(kind="combat_reset", reason="death"),)])
    assert parts.controller.run(max_frames=1) == EXIT_OK
    assert parts.runtime.resets == 1
    assert parts.input.sent == []
    (effect,) = _stages(_rows(parts), "effect")
    assert effect["payload"]["disposition"] == "handled"


@pytest.mark.parametrize(
    ("mode", "terminal_state", "expected"),
    [
        ("shadow", ControllerState.COMPLETE, EXIT_OK),
        ("live", ControllerState.COMPLETE, EXIT_OK),
        ("live", ControllerState.FORMAL_RUN_TERMINAL_FAILURE, EXIT_TERMINAL_FAILURE),
    ],
)
def test_terminal_effects_stop_controller(tmp_path, mode, terminal_state, expected) -> None:
    """I7: controller_stop/process_terminate で以降の frame を処理せず終了する。"""
    terminal = (
        Effect(kind="release_all", reason="done"),
        Effect(kind="controller_stop", reason="done"),
        Effect(kind="process_terminate", reason="done"),
    )
    parts = _build(tmp_path, mode, [0, 1, 2], effects=[terminal], terminal_state=terminal_state)
    assert parts.controller.run(max_frames=3) == expected
    assert parts.controller.exit_reason == "controller_stop:done"
    assert parts.detector.calls == 1
    kinds = [(row["payload"]["kind"], row["payload"]["disposition"]) for row in _stages(_rows(parts), "effect")]
    first = "executed" if mode == "live" else "proposed"
    assert kinds == [("release_all", first), ("controller_stop", "handled"), ("process_terminate", "handled")]


def test_input_ack_failure_is_fatal(tmp_path) -> None:
    """live の入力 ack が False なら fail-closed に異常終了する。"""
    parts = _build(tmp_path, "live", [0, 1], effects=_INPUT_EFFECTS, input_ack=False)
    assert parts.controller.run(max_frames=2) == EXIT_ERROR
    assert parts.controller.exit_reason == "input_ack_failed:move"
    assert parts.detector.calls == 1


def test_release_failure_forces_nonzero_exit(tmp_path) -> None:
    """M6: 入力解放が確認できなければ正常終了でも非0にする。"""
    parts = _build(tmp_path, "live", [0], release_ok=False)
    assert parts.controller.run(max_frames=1) == EXIT_ERROR
    assert _stages(_rows(parts), "input_release")[0]["payload"] == {"released": False}


def test_arm_resets_temporal_state_and_marks_episode_start(tmp_path) -> None:
    """arm は state machine を明示 arm し、HUD/policy の時系列状態を捨てる。"""
    parts = _build(tmp_path, "shadow", [0, 1])
    parts.controller.arm(
        campaign_run_mode=CampaignRunMode.OPERATOR_DEBUG_RESTART, run_id="run-1", gameplay_attempt_id="a-1"
    )
    assert parts.controller.run(max_frames=2) == EXIT_OK
    assert parts.sm.armed[0]["run_id"] == "run-1"
    assert (parts.hud.resets, parts.runtime.resets) == (1, 1)
    assert parts.runtime.episode_flags == [True, False]
