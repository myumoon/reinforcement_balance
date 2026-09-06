"""AgentRuntime: screen_state ルーティング・時間ゲート・scheduler 統合を検証する。

RealObsAssembler が生成した実 PerceptionSnapshot を使い、typed screen_state に
よる経路分岐、snapshot age / global validity / inference timeout の各 gate、
episode 境界での LSTM state 破棄、15 Hz cadence の強制、そして plan 05-01 の
AgentDecision wire 契約を確認する。OS input が生成されないことも検証する。
"""
from __future__ import annotations

import dataclasses
import inspect
import json

import numpy as np
import pytest

from reinbalance_survivors_contracts.deploy_obs import DeployObservation, DeployObsSchema
from reinbalance_survivors_contracts.ui_intent import (
    DecisionOwner,
    UiIntentKind,
    UiIntentV1,
)
from reinbalance_survivors_contracts.ui_policy import ScreenState
from survivors.real_obs_assembler import RealObsAssembler
from survivors.vision.entity_tracker import (
    PlayerAnchorState,
    TrackedEntityV1,
    TrackedWorldStateV1,
)
from survivors.vision.hud_parser import HudStateV1, ParsedCard

from survivors.runtime import agent_runtime as agent_runtime_module
from survivors.runtime.agent_runtime import (
    AGENT_DECISION_SCHEMA_VERSION,
    AgentDecision,
    AgentRuntime,
    decisions_to_jsonl,
)
from survivors.runtime.artifact_bundle import RuntimeBundle
from survivors.runtime.decision_scheduler import DecisionScheduler

TICK_NS = DecisionScheduler().interval_ns


def _hud_world(screen_state: str = "gameplay", *, ts: int = 1_000_000_000):
    """テスト用 HUD + world state を返す。"""
    card = ParsedCard(0, "whip", "weapon", 2, .99, "ok", (100, 100, 400, 500))
    hud = HudStateV1(
        "hud_state.v1", "session", 4, ts, "a" * 64, screen_state, .9, "ok",
        20., .9, "ok", False, .75, .9, "ok", .5, .9, "ok", 4, .9, "ok",
        ("whip",) + (None,) * 11, .9, "b" * 64, (card,), "c" * 64, (),
        False, False, False, .9, "ok",
    )
    visible = TrackedEntityV1(1, 2, "enemy_normal", "enemy", .9, 1, 4, .7, .5, .2, 0., 0., 0., True, False)
    world = TrackedWorldStateV1(4, ts, [visible], PlayerAnchorState(.5, .5, .9, False))
    return hud, world


def _snap(screen_state: str = "gameplay", *, ts: int = 1_000_000_000):
    """RealObsAssembler で PerceptionSnapshot を作る。"""
    schema = DeployObsSchema.default_v1()
    hud, world = _hud_world(screen_state, ts=ts)
    snap = RealObsAssembler().assemble(hud, world, schema, (1000, 1000))
    assert snap is not None, f"assembler returned None for screen_state={screen_state!r}"
    return snap


def _fresh_now(snapshot) -> int:
    """snapshot が age gate を通る monotonic 時刻を返す。"""
    return snapshot.captured_ns + 5_000_000


@pytest.fixture()
def runtime(golden_combat_policy) -> AgentRuntime:
    """ItemSelector を持たない golden bundle の runtime。"""
    return AgentRuntime(RuntimeBundle.from_golden_fixture(golden_combat_policy))


class TestAgentDecisionContract:
    """[指摘10] plan 05-01 の AgentDecision 契約と canonical wire を検証する。"""

    def test_field_names_match_plan_contract(self):
        names = [f.name for f in dataclasses.fields(AgentDecision)]
        for expected in (
            "decision_id", "kind", "action_index", "ui_intent", "confidence", "reason",
            "source_snapshot_id", "source_frame_id", "source_content_hash",
            "snapshot_timestamp_ns", "inference_started_ns", "inference_finished_ns",
        ):
            assert expected in names, expected

    def test_move_requires_action_index(self):
        with pytest.raises(ValueError, match="action_index"):
            AgentDecision(
                decision_id="x", kind="move", action_index=None, ui_intent=None,
                confidence=0.5, reason="r", source_snapshot_id="s", source_frame_id="f",
                source_content_hash="c", snapshot_timestamp_ns=0,
                inference_started_ns=0, inference_finished_ns=0,
            )

    def test_ui_requires_intent(self):
        with pytest.raises(ValueError, match="ui_intent"):
            AgentDecision(
                decision_id="x", kind="ui", action_index=None, ui_intent=None,
                confidence=0.5, reason="r", source_snapshot_id="s", source_frame_id="f",
                source_content_hash="c", snapshot_timestamp_ns=0,
                inference_started_ns=0, inference_finished_ns=0,
            )

    def test_no_op_must_not_have_action_index(self):
        with pytest.raises(ValueError, match="no_op"):
            AgentDecision(
                decision_id="x", kind="no_op", action_index=3, ui_intent=None,
                confidence=0.0, reason="r", source_snapshot_id="s", source_frame_id="f",
                source_content_hash="c", snapshot_timestamp_ns=0,
                inference_started_ns=0, inference_finished_ns=0,
            )

    def test_confidence_out_of_range_rejected(self):
        with pytest.raises(ValueError, match="confidence"):
            AgentDecision(
                decision_id="x", kind="no_op", action_index=None, ui_intent=None,
                confidence=1.5, reason="r", source_snapshot_id="s", source_frame_id="f",
                source_content_hash="c", snapshot_timestamp_ns=0,
                inference_started_ns=0, inference_finished_ns=0,
            )

    def test_empty_reason_rejected(self):
        with pytest.raises(ValueError, match="reason"):
            AgentDecision(
                decision_id="x", kind="no_op", action_index=None, ui_intent=None,
                confidence=0.0, reason="", source_snapshot_id="s", source_frame_id="f",
                source_content_hash="c", snapshot_timestamp_ns=0,
                inference_started_ns=0, inference_finished_ns=0,
            )

    def test_wire_roundtrip_preserves_hash(self, runtime):
        snap = _snap("gameplay")
        decision = runtime.decide(snap, now_ns=_fresh_now(snap), episode_start=True)
        restored = AgentDecision.from_wire(json.loads(decision.canonical_bytes()))
        assert restored.decision_hash() == decision.decision_hash()

    def test_wire_is_byte_stable(self, runtime):
        snap = _snap("gameplay")
        decision = runtime.decide(snap, now_ns=_fresh_now(snap), episode_start=True)
        assert decision.canonical_bytes() == decision.canonical_bytes()
        assert decision.to_wire()["schema_version"] == AGENT_DECISION_SCHEMA_VERSION

    def test_ui_decision_wire_embeds_intent(self, runtime):
        snap = _snap("chest", ts=2_000_000_000)
        decision = runtime.decide(snap, now_ns=_fresh_now(snap), episode_start=True)
        assert decision.kind == "ui"
        wire = decision.to_wire()
        assert wire["ui_intent"]["kind"] == UiIntentKind.ACK_CHEST.value

    def test_jsonl_has_one_line_per_decision(self, runtime):
        snap = _snap("gameplay")
        first = runtime.decide(snap, now_ns=_fresh_now(snap), episode_start=True)
        chest = _snap("chest", ts=2_000_000_000)
        second = runtime.decide(chest, now_ns=_fresh_now(chest) + TICK_NS)
        payload = decisions_to_jsonl([first, second])
        lines = payload.splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["decision_id"] == first.decision_id

    def test_decision_has_no_os_input_fields(self, runtime):
        snap = _snap("gameplay")
        decision = runtime.decide(snap, now_ns=_fresh_now(snap), episode_start=True)
        for forbidden in ("key", "mouse_click", "send_input", "roi", "click"):
            assert not hasattr(decision, forbidden)


class TestTypedScreenStateRouting:
    """[指摘4] raw state の重複表ではなく ui_policy_input.screen_state で route する。"""

    def test_no_duplicated_raw_screen_state_table(self):
        """runtime が raw state 名の集合を独自に持たないこと。"""
        source = inspect.getsource(agent_runtime_module)
        assert "_NON_MODEL_UI_SCREEN_STATES" not in source
        assert "_ITEM_SCREEN_STATES" not in source

    def test_target_reached_transition_produces_confirm(self, runtime):
        """assembler の raw state は target_reached_transition。confirm intent になる。"""
        snap = _snap("target_reached_transition", ts=3_000_000_000)
        assert snap.ui_policy_input.screen_state == ScreenState.CONFIRM
        decision = runtime.decide(snap, now_ns=_fresh_now(snap), episode_start=True)
        assert decision.kind == "ui"
        assert decision.ui_intent is not None
        assert decision.ui_intent.kind == UiIntentKind.CONFIRM

    def test_confirm_intent_is_not_no_op(self, runtime):
        """レビュー再現: confirm intent が常に no_op になってはいけない。"""
        snap = _snap("target_reached_transition", ts=3_500_000_000)
        decision = runtime.decide(snap, now_ns=_fresh_now(snap), episode_start=True)
        assert decision.kind != "no_op"

    def test_chest_produces_ack_chest(self, runtime):
        snap = _snap("chest", ts=4_000_000_000)
        decision = runtime.decide(snap, now_ns=_fresh_now(snap), episode_start=True)
        assert decision.kind == "ui"
        assert decision.ui_intent.kind == UiIntentKind.ACK_CHEST

    def test_gameplay_produces_move(self, runtime):
        snap = _snap("gameplay")
        decision = runtime.decide(snap, now_ns=_fresh_now(snap), episode_start=True)
        assert decision.kind == "move"
        assert isinstance(decision.action_index, int)
        assert 0 <= decision.action_index < 9
        assert decision.ui_intent is None

    def test_effect_owner_is_ui_state_machine(self, runtime):
        snap = _snap("chest", ts=4_500_000_000)
        decision = runtime.decide(snap, now_ns=_fresh_now(snap), episode_start=True)
        assert decision.ui_intent.effect_owner == "ui_state_machine_v1"

    def test_non_snapshot_returns_stop(self, runtime):
        assert runtime.decide("not-a-snapshot").kind == "stop"


class TestSnapshotTimeAndValidityGates:
    """[指摘5] captured_ns の age・global validity・timeout を検証する。"""

    def test_zero_captured_ns_does_not_return_move(self, runtime):
        """レビュー再現: captured_ns=0 でも move を返してはいけない。"""
        snap = dataclasses.replace(_snap("gameplay"), captured_ns=0)
        decision = runtime.decide(snap, now_ns=2_000_000_000, episode_start=True)
        assert decision.kind == "no_op"
        assert decision.action_index is None
        assert "age gate" in decision.reason

    def test_stale_snapshot_does_not_return_move(self, runtime):
        snap = _snap("gameplay")
        stale_now = snap.captured_ns + 10 * TICK_NS
        decision = runtime.decide(snap, now_ns=stale_now, episode_start=True)
        assert decision.kind == "no_op"
        assert "age gate" in decision.reason

    def test_future_snapshot_does_not_return_move(self, runtime):
        snap = _snap("gameplay")
        decision = runtime.decide(
            snap, now_ns=snap.captured_ns - 1_000_000, episode_start=True
        )
        assert decision.kind == "no_op"

    def test_low_global_validity_does_not_return_move(self, runtime):
        """観測が信用できないときは movement action を返さない。"""
        snap = _snap("gameplay")
        observation = snap.deploy_obs
        degraded = DeployObservation(
            values=np.asarray(observation.values),
            validity=np.zeros_like(np.asarray(observation.validity)),
            age=np.asarray(observation.age),
            schema_hash=observation.schema_hash,
            timestamp_ns=observation.timestamp_ns,
            provenance=observation.provenance,
        )
        snap = dataclasses.replace(snap, deploy_obs=degraded)
        decision = runtime.decide(snap, now_ns=_fresh_now(snap), episode_start=True)
        assert decision.kind == "no_op"
        assert "validity gate" in decision.reason

    def test_inference_timeout_downgrades_move(self, golden_combat_policy):
        """締切超過の推論結果は move にせず no_op へ倒す。"""
        bundle = RuntimeBundle.from_golden_fixture(golden_combat_policy)
        runtime = AgentRuntime(bundle, scheduler=DecisionScheduler(inference_timeout_ns=1))
        snap = _snap("gameplay")
        decision = runtime.decide(snap, now_ns=_fresh_now(snap), episode_start=True)
        assert decision.kind == "no_op"
        assert "timeout" in decision.reason


class TestEpisodeBoundaryReset:
    """[指摘5] death / result / unknown gap で recurrent state を破棄する。"""

    @pytest.mark.parametrize("boundary_state", ["death", "result", "unknown"])
    def test_boundary_state_resets_recurrent_state(self, runtime, boundary_state):
        gameplay = _snap("gameplay")
        runtime.decide(gameplay, now_ns=_fresh_now(gameplay), episode_start=True)
        assert runtime.combat_session.lstm_state_copy() is not None

        boundary = _snap(boundary_state, ts=5_000_000_000)
        decision = runtime.decide(boundary, now_ns=_fresh_now(boundary) + TICK_NS)
        assert decision.kind == "no_op"
        assert runtime.combat_session.lstm_state_copy() is None
        assert runtime.combat_session.episode_start_pending is True

    def test_hidden_state_does_not_leak_into_next_run(self, runtime):
        """レビュー再現: death 後に hidden state が残ってはいけない。"""
        first = _snap("gameplay")
        now = _fresh_now(first)
        baseline = runtime.decide(first, now_ns=now, episode_start=True)

        # 同じ観測を続けて state を進める
        for step in range(1, 4):
            follow = _snap("gameplay", ts=1_000_000_000 + step)
            runtime.decide(follow, now_ns=now + step * TICK_NS)

        death = _snap("death", ts=6_000_000_000)
        runtime.decide(death, now_ns=now + 4 * TICK_NS)

        revived = _snap("gameplay", ts=7_000_000_000)
        after = runtime.decide(revived, now_ns=_fresh_now(revived))
        assert after.kind == "move"
        # reset 済みなので episode 先頭と同じ action になる
        assert after.action_index == baseline.action_index

    def test_paused_does_not_reset_state(self, runtime):
        gameplay = _snap("gameplay")
        runtime.decide(gameplay, now_ns=_fresh_now(gameplay), episode_start=True)
        paused = _snap("paused", ts=8_000_000_000)
        decision = runtime.decide(paused, now_ns=_fresh_now(paused) + TICK_NS)
        assert decision.kind == "no_op"
        assert runtime.combat_session.lstm_state_copy() is not None

    def test_reset_episode_clears_state(self, runtime):
        gameplay = _snap("gameplay")
        runtime.decide(gameplay, now_ns=_fresh_now(gameplay), episode_start=True)
        runtime.reset_episode()
        assert runtime.combat_session.lstm_state_copy() is None


class TestSchedulerIntegration:
    """[指摘7] DecisionScheduler が実際の decide flow に組み込まれている。"""

    def test_runtime_exposes_scheduler(self, runtime):
        assert isinstance(runtime.scheduler, DecisionScheduler)
        assert runtime.scheduler.hz == 15

    def test_off_cadence_call_is_skipped(self, runtime):
        snap = _snap("gameplay")
        now = _fresh_now(snap)
        first = runtime.decide(snap, now_ns=now, episode_start=True)
        assert first.kind == "move"
        second = runtime.decide(snap, now_ns=now + 1_000_000)
        assert second.kind == "no_op"
        assert "off-cadence" in second.reason

    def test_next_tick_is_served(self, runtime):
        snap = _snap("gameplay")
        now = _fresh_now(snap)
        runtime.decide(snap, now_ns=now, episode_start=True)
        follow = _snap("gameplay", ts=snap.captured_ns + TICK_NS)
        decision = runtime.decide(follow, now_ns=now + TICK_NS + 1)
        assert decision.kind == "move"

    def test_backlog_skips_are_recorded(self, runtime):
        snap = _snap("gameplay")
        now = _fresh_now(snap)
        runtime.decide(snap, now_ns=now, episode_start=True)
        late = _snap("gameplay", ts=snap.captured_ns + 6 * TICK_NS)
        runtime.decide(late, now_ns=now + 6 * TICK_NS)
        assert runtime.scheduler.skipped_tick_count >= 4

    def test_decision_count_matches_served_ticks(self, runtime):
        snap = _snap("gameplay")
        now = _fresh_now(snap)
        runtime.decide(snap, now_ns=now, episode_start=True)
        for step in range(1, 5):
            follow = _snap("gameplay", ts=snap.captured_ns + step * TICK_NS)
            runtime.decide(follow, now_ns=now + step * TICK_NS)
            # 同一 tick 内の追加呼び出しは skip される
            runtime.decide(follow, now_ns=now + step * TICK_NS + 1_000)
        assert runtime.scheduler.decision_count == 5

    def test_timing_metadata_is_recorded(self, runtime):
        snap = _snap("gameplay")
        runtime.decide(snap, now_ns=_fresh_now(snap), episode_start=True)
        timing = runtime.last_timing
        assert timing is not None
        assert timing.inference_latency_ms >= 0.0

    def test_recorded_sequence_is_reproducible(self, golden_combat_policy):
        """同一 snapshot 列 / 同一 tick 列から同じ decision 列を再生する。"""
        def run() -> list[tuple[str, int | None]]:
            runtime = AgentRuntime(
                RuntimeBundle.from_golden_fixture(golden_combat_policy)
            )
            results: list[tuple[str, int | None]] = []
            base_ts = 1_000_000_000
            now = base_ts + 5_000_000
            states = ["gameplay", "gameplay", "chest", "gameplay", "death", "gameplay"]
            for step, state in enumerate(states):
                snap = _snap(state, ts=base_ts + step * TICK_NS)
                decision = runtime.decide(
                    snap, now_ns=now + step * TICK_NS, episode_start=(step == 0)
                )
                results.append((decision.kind, decision.action_index))
            return results

        assert run() == run()


class TestNoOsInputSideEffects:
    """runtime は typed decision を返すだけで OS input を送らない。"""

    def test_runtime_module_has_no_input_imports(self):
        source = inspect.getsource(agent_runtime_module)
        for forbidden in ("SendInput", "pyautogui", "win32api", "ctypes", "keyboard"):
            assert forbidden not in source

    def test_decision_kinds_are_limited(self, runtime):
        snap = _snap("gameplay")
        decision = runtime.decide(snap, now_ns=_fresh_now(snap), episode_start=True)
        assert decision.kind in ("move", "ui", "no_op", "stop")

    def test_invalid_bundle_rejected(self):
        with pytest.raises(ValueError, match="RuntimeBundle"):
            AgentRuntime(object())  # type: ignore[arg-type]
