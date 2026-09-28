"""Survivors controller の bounded pipeline 本体(shadow/live 共通経路)。

capture → detector → tracker → HUD parser → obs assembler → policy → state machine →
effect の順に、1本のループで各 stage を呼び出して telemetry へ記録します。
shadow mode は OS 入力 adapter を一切持たず、effect を「proposed」として記録するだけです。
live mode だけが ``execute_effect`` で実際の入力を送ります。それ以外の経路はすべて同じです。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import hashlib
import queue
import time
from typing import Any, Literal

import numpy as np

from reinbalance_survivors_contracts.deploy_obs import DeployObsSchema, DeployObservation

from ..capture.window_locator import TargetWindowStateError
from ..vision.entity_tracker import TrackedWorldStateV1
from .health_monitor import HealthMonitor, HealthVerdict, observation_is_valid
from .state_machine import CampaignRunMode, ControllerState
from .telemetry import TelemetryWriter
from .ui_navigation import Effect, build_ui_action_telemetry, execute_effect

EXIT_OK = 0
EXIT_HEALTH_STOP = 2
EXIT_ERROR = 3
EXIT_TERMINAL_FAILURE = 4

# execute_effect が OS 入力として実行する effect 種別(live mode だけが渡す)。
_INPUT_EFFECTS = frozenset({"move", "ui_click", "ui_key", "release_all"})

# hud_parser 行へ書く HudStateV1 の信頼度と判定理由(parser validity/reasons)。
_HUD_TELEMETRY_FIELDS = (
    "screen_state_confidence", "screen_state_reason",
    "timer_confidence", "timer_reason",
    "hp_confidence", "hp_reason",
    "xp_confidence", "xp_reason",
    "level_confidence", "level_reason",
    "inventory_confidence",
    "capability_confidence", "capability_reason",
)


def _obs_quality_summary(obs: DeployObservation) -> dict[str, Any]:
    """DeployObs の validity/age 配列を telemetry 用の少数の数値へ要約する。

    生配列は大きいので書かず、欠損要素数(validity<1)・validity 平均・age 平均/最大だけを残します。
    どの tick で観測が劣化したかを後から追えるようにするための要約です。
    """
    validity = np.asarray(obs.validity, dtype=np.float64)
    age = np.asarray(obs.age, dtype=np.float64)
    return {
        "obs_invalid_count": int(np.count_nonzero(validity < 1.0)),
        "obs_validity_mean": float(validity.mean()) if validity.size else 1.0,
        "obs_age_mean": float(age.mean()) if age.size else 0.0,
        "obs_age_max": float(age.max()) if age.size else 0.0,
    }


class _StageError(Exception):
    """perception/policy stage の例外を stage 名付きで包む内部例外。

    どの stage で推論が失敗したかを health monitor と telemetry に渡すためだけに使います。
    元の例外は ``__cause__`` に残ります。
    """

    def __init__(self, stage: str) -> None:
        """失敗した stage 名を保持する。

        telemetry の error record と health の detail に同じ名前を使います。
        """
        super().__init__(stage)
        self.stage = stage


def observation_hash(observation: DeployObservation) -> str:
    """DeployObservation の三平面・schema・時刻から sha256 を計算する。

    effect がどの観測値から生まれたかを後から照合するための指紋です。
    同じ観測なら必ず同じ hash になります。
    """
    digest = hashlib.sha256()
    digest.update(observation.schema_hash.encode("ascii"))
    digest.update(str(observation.timestamp_ns).encode("ascii"))
    for plane in (observation.values, observation.validity, observation.age):
        digest.update(np.ascontiguousarray(plane, dtype=np.float32).tobytes())
    return digest.hexdigest()


class SurvivorsController:
    """capture から effect 実行までを束ねる単一ループの controller。

    ``run()`` を呼ぶと、新しい frame を1枚ずつ取り出して全 stage を順番に処理します。
    frame は常に最新の1枚だけを使い(latest-only)、古い frame や順序の逆転した frame は
    捨てます。各 stage の結果は同じ frame の correlation id で telemetry に残ります。
    health STOP・例外・terminal effect のどれで止まっても、capture 停止 → queue drain →
    入力解放 → telemetry 確定 の順で片付けてから終了コードを返します。
    単一スレッドで動くため、stage 間の bounded queue は capture の LatestFrameQueue(maxsize=1)だけです。
    """

    def __init__(
        self,
        *,
        mode: Literal["shadow", "live"],
        session_id: str,
        capture: Any,
        detector: Any,
        tracker: Any,
        hud_parser: Any,
        assembler: Any,
        runtime: Any,
        state_machine: Any,
        health: HealthMonitor,
        telemetry: TelemetryWriter,
        schema: DeployObsSchema,
        model_hashes: Mapping[str, str],
        input_controller: Any = None,
        ui_config: Any = None,
        class_map_path: Any = None,
        viewport: tuple[int, int] = (1920, 1080),
        score_threshold: float = 0.5,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
        sleep: Callable[[float], None] = time.sleep,
        idle_sleep_s: float = 0.001,
        drain_timeout_ns: int = 500_000_000,
    ) -> None:
        """全 stage の依存を受け取り、mode と入力 adapter の組み合わせを検証する。

        shadow mode に入力 adapter を渡すと即座に拒否します(M3: shadow は入力を持たない)。
        live mode は入力 adapter が必須です。adapter の生成・破棄は呼び出し側(CLI)の責務です。
        """
        if mode not in ("shadow", "live"):
            raise ValueError("mode must be shadow or live")
        if mode == "shadow" and input_controller is not None:
            raise ValueError("shadow mode must not hold an input adapter")
        if mode == "live" and input_controller is None:
            raise ValueError("live mode requires an input adapter")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a non-empty string")
        if not isinstance(model_hashes, Mapping) or not model_hashes:
            raise ValueError("model_hashes must be a non-empty mapping")
        self.mode = mode
        self.session_id = session_id
        self._capture = capture
        self._detector = detector
        self._tracker = tracker
        self._hud_parser = hud_parser
        self._assembler = assembler
        self._runtime = runtime
        self._state_machine = state_machine
        self._health = health
        self._telemetry = telemetry
        self._schema = schema
        self._model_hashes = dict(model_hashes)
        self._input = input_controller
        self._ui_config = ui_config
        self._class_map_path = class_map_path
        self._viewport = viewport
        self._score_threshold = score_threshold
        self._clock_ns = clock_ns
        self._sleep = sleep
        self._idle_sleep_s = idle_sleep_s
        self._drain_timeout_ns = drain_timeout_ns
        self._control_id = f"{session_id}:controller"
        self._last_frame_index = -1
        self._last_obs_valid = False
        self._episode_start = False
        self._exit: tuple[int, str] | None = None
        self.errors: list[dict[str, str]] = []
        self.shutdown_steps: list[str] = []

    @property
    def exit_code(self) -> int | None:
        """確定した終了コード(未確定なら None)。

        最初に確定した理由だけが有効で、後から上書きされません。
        """
        return None if self._exit is None else self._exit[0]

    @property
    def exit_reason(self) -> str | None:
        """確定した終了理由(未確定なら None)。

        telemetry の shutdown record と同じ文字列です。
        """
        return None if self._exit is None else self._exit[1]

    def arm(self, *, campaign_run_mode: CampaignRunMode, run_id: str, gameplay_attempt_id: str) -> None:
        """state machine を明示的に arm し、HUD/policy の時系列状態を新 run 用に初期化する。

        画面認識だけで自己武装しないよう、arm は必ず呼び出し側(CLI の arm toggle)が行います。
        HUD parser の時系列状態と policy の LSTM 状態もここで捨て、次の decide を episode 開始扱いにします。
        """
        now_ns = self._clock_ns()
        self._state_machine.arm(
            campaign_run_mode=campaign_run_mode,
            run_id=run_id,
            gameplay_attempt_id=gameplay_attempt_id,
            now_ns=now_ns,
        )
        self._hud_parser.reset_temporal_state()
        self._runtime.reset_episode()
        self._episode_start = True
        self._write(
            "arm",
            self._control_id,
            payload={
                "campaign_run_mode": campaign_run_mode.value,
                "run_id": run_id,
                "gameplay_attempt_id": gameplay_attempt_id,
            },
        )

    def run(self, *, max_frames: int | None = None, should_stop: Callable[[], bool] | None = None) -> int:
        """capture を開始し、停止条件まで frame を処理して終了コードを返す。

        ``max_frames`` 枚処理するか ``should_stop()`` が True を返すと正常終了(0)します。
        health STOP は 2、例外は 3、formal terminal failure は 4 を返します。
        どの経路でも ``_shutdown()`` を必ず通ります。``KeyboardInterrupt`` などは
        片付けの後にそのまま再送出されます。
        """
        processed = 0
        try:
            self._capture.start()
            while self._exit is None:
                if should_stop is not None and should_stop():
                    self._finish(EXIT_OK, "stop_requested")
                    break
                if max_frames is not None and processed >= max_frames:
                    self._finish(EXIT_OK, "max_frames")
                    break
                frame = self._next_frame()
                if frame is None:
                    if self._health.poll(self._clock_ns()) is HealthVerdict.STOP:
                        self._health_stop()
                    else:
                        self._sleep(self._idle_sleep_s)
                    continue
                processed += 1
                self._process_frame(frame)
        except Exception as exc:
            self._record_error("controller", exc, self._control_id)
            self._finish(EXIT_ERROR, f"exception:{type(exc).__name__}")
        finally:
            if self._exit is None:
                self._finish(EXIT_ERROR, "interrupted")
            self._shutdown()
        return self._exit[0]

    def _finish(self, code: int, reason: str) -> None:
        """終了コードと理由を最初の1回だけ確定する。

        複数の停止要因が同じ tick に重なっても、最初に起きたものを正とします。
        """
        if self._exit is None:
            self._exit = (code, reason)

    def _write(
        self, stage: str, correlation_id: str, *, latency_ns: int = 0, queue_depth: int = 0,
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        """stage event を現在時刻で telemetry へ1行書く。

        書き込み失敗は例外のまま呼び出し側へ伝え、握り潰しません。
        """
        self._telemetry.write_stage(
            stage,
            correlation_id=correlation_id,
            timestamp_ns=self._clock_ns(),
            latency_ns=latency_ns,
            queue_depth=queue_depth,
            payload=payload,
        )

    def _record_error(self, stage: str, exc: BaseException, correlation_id: str) -> None:
        """例外を structured error として保持し、可能なら telemetry にも書く。

        telemetry 自体が壊れていても元の例外情報を失わないよう、先に ``errors`` へ積みます。
        telemetry への書き込みが失敗した場合はその失敗も ``errors`` に追記します。
        """
        record = {"stage": stage, "type": type(exc).__name__, "message": str(exc)}
        self.errors.append(record)
        try:
            self._write("error", correlation_id, payload=record)
        except Exception as write_exc:
            self.errors.append(
                {"stage": "telemetry", "type": type(write_exc).__name__, "message": str(write_exc)}
            )

    def _health_stop(self) -> None:
        """health STOP の report を telemetry へ書き、終了コード 2 を確定する。

        実際の入力解放は ``_shutdown()`` が M2 の順序で行います(M6)。
        """
        report = self._health.report()
        self._write("health", self._control_id, payload=report.to_wire())
        first = report.first_failure
        self._finish(EXIT_HEALTH_STOP, f"health_stop:{first.reason.value if first else 'unknown'}")

    def _timed(self, stage: str, fn: Callable[[], Any]) -> tuple[Any, int]:
        """stage を1回実行し、結果と所要時間(ns)を返す。

        stage 内の例外は ``_StageError`` に包んで、どの stage で失敗したかを伝えます。
        """
        started = self._clock_ns()
        try:
            result = fn()
        except Exception as exc:
            raise _StageError(stage) from exc
        return result, self._clock_ns() - started

    def _next_frame(self) -> Any:
        """capture から最新 frame を1枚取り、古い/逆順の frame は捨てる。

        ``capture_next()`` が queue へ積んだ最新1枚だけを取り出します(latest-only)。
        前回処理した frame 以前の番号なら discard として記録して None を返し、
        番号が飛んでいればその枚数を drop として capture record に残します。
        """
        # 実 CaptureSession は一時的な focus 喪失を例外にせず内部 pause して None を返す。
        # pause に入った tick だけを focus_lost として記録する(fake capture は paused を持たない)。
        was_paused = getattr(self._capture, "paused", False)
        try:
            captured = self._capture.capture_next()
            if not was_paused and getattr(self._capture, "paused", False):
                self._health.record_focus_lost(
                    now_ns=self._clock_ns(), detail="target window lost foreground"
                )
                return None
        except TargetWindowStateError as exc:
            # ここに来るのは identity/geometry 変化など回復不能な状態異常だけ
            # (一時的な focus 喪失は上の paused 立ち上がり検知で処理済み)。
            # STOP の記録だけ行い、_health_stop() は run() の poll 分岐に一本化する(二重に呼ばないため)。
            self._health.record_focus_lost(now_ns=self._clock_ns(), detail=str(exc))
            return None
        if captured is None:
            return None
        try:
            frame = self._capture.frames.get_latest_nowait()
        except queue.Empty:
            return None
        index = frame.session_frame_index
        cid = f"{self.session_id}:{index}"
        if index <= self._last_frame_index:
            self._write(
                "discard", cid,
                payload={"frame_index": index, "last_frame_index": self._last_frame_index, "reason": "stale_frame"},
            )
            return None
        dropped = index - self._last_frame_index - 1
        self._last_frame_index = index
        self._write(
            "capture", cid,
            latency_ns=max(0, self._clock_ns() - frame.captured_monotonic_ns),
            queue_depth=1,
            payload={
                "frame_index": index,
                "captured_ns": frame.captured_monotonic_ns,
                "foreground": frame.foreground,
                "dropped_frames": dropped,
                "target_profile_hash": frame.target_profile_hash,
                "game_build_id": frame.game_build_id,
            },
        )
        return frame

    def _process_frame(self, frame: Any) -> None:
        """1 frame 分の perception → health → policy → state machine → effect を実行する。

        policy と state machine は assembler が snapshot を返した tick(15 Hz)だけ動きます。
        perception/policy の例外は推論エラーとして health を即 STOP させます。
        health が STOP を返したら policy へ進まず、その tick の effect は出しません。
        """
        index = frame.session_frame_index
        captured_ns = frame.captured_monotonic_ns
        cid = f"{self.session_id}:{index}"
        try:
            detection, latency = self._timed(
                "detector",
                lambda: self._detector.infer(
                    np.ascontiguousarray(frame.frame_bgra[:, :, :3]), score_threshold=self._score_threshold
                ),
            )
            self._write("detector", cid, latency_ns=latency, payload={"detections": len(detection.scores)})
            world, latency = self._timed(
                "tracker",
                lambda: TrackedWorldStateV1.from_state(
                    self._tracker.update(detection, index, captured_ns), index, captured_ns, self._class_map_path
                ),
            )
            self._write("tracker", cid, latency_ns=latency, payload={"tracks": len(world.tracks)})
            hud, latency = self._timed(
                "hud_parser",
                lambda: self._hud_parser.parse(
                    frame.frame_bgra, session_id=self.session_id, frame_index=index, captured_monotonic_ns=captured_ns
                ),
            )
            self._write(
                "hud_parser", cid, latency_ns=latency,
                payload={
                    "screen_state": hud.screen_state,
                    "parser_artifact_hash": hud.parser_artifact_hash,
                    **{name: getattr(hud, name) for name in _HUD_TELEMETRY_FIELDS},
                },
            )
            snapshot, latency = self._timed(
                "obs",
                lambda: self._assembler.assemble(hud, world, self._schema, self._viewport, self._ui_config),
            )
            obs_hash = None
            if snapshot is not None:
                self._last_obs_valid = observation_is_valid(snapshot.deploy_obs, self._schema)
                obs_hash = observation_hash(snapshot.deploy_obs)
            self._write(
                "obs", cid, latency_ns=latency,
                payload={"emitted": False} if snapshot is None else {
                    "emitted": True,
                    "snapshot_id": snapshot.snapshot_id,
                    "frame_id": snapshot.frame_id,
                    "screen_state": snapshot.screen_state,
                    "ui_state_key": snapshot.ui_state_key,
                    "source_content_hash": snapshot.source_content_hash,
                    "obs_hash": obs_hash,
                    "obs_valid": self._last_obs_valid,
                    "obs_timestamp_ns": snapshot.deploy_obs.timestamp_ns,
                    **_obs_quality_summary(snapshot.deploy_obs),
                    "ui_schema_hash": snapshot.ui_presentation.schema_hash,
                    "ui_candidate_set_hash": snapshot.ui_presentation.candidate_set_hash,
                    "ui_inventory_hash": snapshot.ui_presentation.inventory_hash,
                },
            )
            now_ns = self._clock_ns()
            verdict = self._health.ingest(
                now_ns=now_ns,
                capture_timestamp_ns=captured_ns,
                window_focused=frame.foreground,
                perception_latency_ns=max(0, now_ns - captured_ns),
                observation_valid=self._last_obs_valid,
                controller_state=self._state_machine.context.state,
            )
            if verdict is HealthVerdict.STOP:
                self._health_stop()
                return
            if snapshot is None:
                return
            decision, latency = self._timed(
                "policy",
                lambda: self._runtime.decide(snapshot, now_ns=now_ns, episode_start=self._episode_start),
            )
        except _StageError as err:
            cause = err.__cause__ or err
            self._record_error(err.stage, cause, cid)
            self._health.record_inference_error(
                now_ns=self._clock_ns(), detail=f"{err.stage}: {type(cause).__name__}: {cause}"
            )
            self._health_stop()
            return
        self._episode_start = False
        decision_hash = decision.decision_hash()
        self._write(
            "policy", cid, latency_ns=latency,
            payload={**decision.to_wire(), "decision_hash": decision_hash, "model_hashes": self._model_hashes},
        )
        from_state = self._state_machine.context.state
        effects = self._state_machine.step(snapshot, decision, now_ns=now_ns, window_focused=frame.foreground)
        self._write(
            "state_machine", cid,
            payload={
                "from_state": from_state.value,
                "to_state": self._state_machine.context.state.value,
                "effects": [effect.kind for effect in effects],
            },
        )
        source = {
            "frame_index": index,
            "captured_ns": captured_ns,
            "snapshot_id": snapshot.snapshot_id,
            "frame_id": snapshot.frame_id,
            "source_content_hash": snapshot.source_content_hash,
            "obs_hash": obs_hash,
            "decision_id": decision.decision_id,
            "decision_hash": decision_hash,
            "model_hashes": self._model_hashes,
        }
        for effect in effects:
            self._apply_effect(effect, cid, snapshot, source)

    def _apply_effect(self, effect: Effect, cid: str, snapshot: Any, source: Mapping[str, Any]) -> None:
        """effect を1件処理し、source frame/obs/model hash と一緒に telemetry へ記録する。

        ``combat_reset``/``controller_stop``/``process_terminate`` は controller 自身が処理します(I7)。
        入力系 effect は live mode だけ ``execute_effect`` へ渡し、shadow では proposed として記録するだけです(M3)。
        入力の ack が False なら fail-closed に異常終了させます。
        """
        ack: bool | None = None
        if effect.kind == "combat_reset":
            self._runtime.reset_episode()
            disposition = "handled"
        elif effect.kind in ("controller_stop", "process_terminate"):
            # process_terminate の実際のプロセス終了は run() の戻り値を受けた CLI が行う。
            completed = self._state_machine.context.terminal_state is ControllerState.COMPLETE
            self._finish(EXIT_OK if completed else EXIT_TERMINAL_FAILURE, f"{effect.kind}:{effect.reason}")
            disposition = "handled"
        elif effect.kind in _INPUT_EFFECTS and self.mode == "live":
            ack = execute_effect(effect, self._input).ack
            disposition = "executed"
        elif effect.kind in _INPUT_EFFECTS:
            disposition = "proposed"
        else:
            raise ValueError(f"unknown effect kind: {effect.kind!r}")
        payload: dict[str, Any] = {
            "kind": effect.kind,
            "disposition": disposition,
            "ack": ack,
            "action_index": effect.action_index,
            "key": effect.key,
            "mode": effect.mode,
            "reason": effect.reason,
            "source": dict(source),
        }
        if effect.intent is not None and effect.target is not None and effect.mode is not None:
            payload["ui_action"] = dict(build_ui_action_telemetry(effect.intent, effect.target, effect.mode, snapshot))
        self._write("effect", cid, payload=payload)
        if ack is False:
            self._finish(EXIT_ERROR, f"input_ack_failed:{effect.kind}")

    def _shutdown(self) -> None:
        """capture stop → queue drain(期限付き) → input release → artifact finalize の順で片付ける(M2)。

        各段の失敗は ``errors`` に積んで次の段へ進み、最後に終了コードを非0へ倒します。
        live mode の入力解放が失敗/未確認のときも非0にします。telemetry は最後に閉じます。
        """
        code, reason = self._exit
        failed = False

        def step(name: str, fn: Callable[[], Mapping[str, Any]]) -> None:
            nonlocal failed
            try:
                payload = fn()
            except Exception as exc:
                failed = True
                self._record_error(name, exc, self._control_id)
                payload = {"failed": True}
            self.shutdown_steps.append(name)
            if name == "artifact_finalize":
                return
            try:
                self._write(name, self._control_id, payload=payload)
            except Exception as exc:
                failed = True
                self.errors.append({"stage": "telemetry", "type": type(exc).__name__, "message": str(exc)})

        def capture_stop() -> Mapping[str, Any]:
            self._capture.close()
            return {}

        def queue_drain() -> Mapping[str, Any]:
            deadline = self._clock_ns() + self._drain_timeout_ns
            drained = 0
            while self._clock_ns() <= deadline:
                try:
                    self._capture.frames.get_latest_nowait()
                except queue.Empty:
                    return {"drained": drained, "timed_out": False}
                drained += 1
            return {"drained": drained, "timed_out": True}

        def input_release() -> Mapping[str, Any]:
            nonlocal failed
            if self._input is None:
                return {"released": None}
            released = bool(self._input.emergency_release())
            failed = failed or not released
            # helper の release audit と同じ時計(time.monotonic_ns/time.time_ns)で記録し、
            # telemetry 時計(既定 perf_counter_ns)とのドメイン差なしに audit と相関できるようにする。
            return {
                "released": released,
                "release_timestamp_ns": time.time_ns(),
                "release_monotonic_ns": time.monotonic_ns(),
            }

        def artifact_finalize() -> Mapping[str, Any]:
            nonlocal code
            if failed and code == EXIT_OK:
                code = EXIT_ERROR
            try:
                self._write(
                    "shutdown", self._control_id,
                    payload={
                        "exit_code": code,
                        "reason": reason,
                        "steps": list(self.shutdown_steps),
                        "errors": list(self.errors),
                        "health": self._health.report().to_wire(),
                    },
                )
            finally:
                self._telemetry.close()
            return {}

        step("capture_stop", capture_stop)
        step("queue_drain", queue_drain)
        step("input_release", input_release)
        step("artifact_finalize", artifact_finalize)
        if failed and code == EXIT_OK:
            code = EXIT_ERROR
        self._exit = (code, reason)
