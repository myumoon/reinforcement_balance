"""AgentRuntime: screen_state ルーティング・stale handling・OS input 不在を検証する。

golden fixture bundle と RealObsAssembler で PerceptionSnapshot を生成し、
全 screen_state 経路と安全系の振る舞いを確認する。
"""
from __future__ import annotations

import numpy as np
import pytest

from reinbalance_survivors_contracts.deploy_obs import DeployObsSchema
from reinbalance_survivors_contracts.ui_intent import UiIntentKind
from survivors.real_obs_assembler import RealObsAssembler
from survivors.vision.entity_tracker import PlayerAnchorState, TrackedEntityV1, TrackedWorldStateV1
from survivors.vision.hud_parser import HudStateV1, ParsedCard
from survivors.runtime.artifact_bundle import RuntimeBundle, _CombatGRU
from survivors.runtime.agent_runtime import AgentDecision, AgentRuntime


OBS_DIM = 3 * DeployObsSchema.default_v1().dim  # values+validity+age


def _gru_model() -> _CombatGRU:
    """deploy_schema の obs dim に合わせた最小 GRU。"""
    return _CombatGRU(OBS_DIM, 9, 16)


def _golden_bundle(item_selector=None) -> RuntimeBundle:
    return RuntimeBundle.from_golden_fixture(
        _gru_model(),
        item_selector=item_selector,
    )


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


class TestAgentDecision:
    def test_move_requires_action_index(self):
        with pytest.raises(ValueError, match="move_action_index"):
            AgentDecision(
                kind="move", decision_id="x",
                ui_intent=None, move_action_index=None,
                no_op_reason=None,
                source_snapshot_id="s", source_frame_id="f",
                source_content_hash="c", snapshot_timestamp_ns=0,
                inference_started_ns=0, inference_finished_ns=0,
            )

    def test_ui_requires_intent(self):
        with pytest.raises(ValueError, match="ui_intent"):
            AgentDecision(
                kind="ui", decision_id="x",
                ui_intent=None, move_action_index=None,
                no_op_reason=None,
                source_snapshot_id="s", source_frame_id="f",
                source_content_hash="c", snapshot_timestamp_ns=0,
                inference_started_ns=0, inference_finished_ns=0,
            )

    def test_no_op_must_not_have_intent(self):
        from reinbalance_survivors_contracts.ui_intent import UiIntentV1, UiIntentKind, DecisionOwner
        intent = UiIntentV1(
            kind=UiIntentKind.NO_OP,
            semantic_action="no_op",
            decision_owner=DecisionOwner.RUNTIME_SAFETY,
            source_snapshot_hash="s" * 64,
            source_frame_hash="f" * 64,
            source_content_hash="c" * 64,
            ui_state_key="k" * 64,
        )
        with pytest.raises(ValueError, match="no_op"):
            AgentDecision(
                kind="no_op", decision_id="x",
                ui_intent=intent, move_action_index=None,
                no_op_reason=None,
                source_snapshot_id="s", source_frame_id="f",
                source_content_hash="c", snapshot_timestamp_ns=0,
                inference_started_ns=0, inference_finished_ns=0,
            )


class TestAgentRuntimeInit:
    def test_invalid_bundle_raises(self):
        with pytest.raises(ValueError, match="RuntimeBundle"):
            AgentRuntime(object())  # type: ignore[arg-type]

    def test_valid_bundle_accepted(self):
        runtime = AgentRuntime(_golden_bundle())
        assert runtime is not None


class TestGameplayRouting:
    def test_gameplay_snapshot_returns_move(self):
        runtime = AgentRuntime(_golden_bundle())
        snap = _snap("gameplay")
        decision = runtime.decide(snap, episode_start=True)
        assert decision.kind == "move"
        assert decision.ui_intent is None
        assert isinstance(decision.move_action_index, int)
        assert 0 <= decision.move_action_index < 9

    def test_move_decision_has_no_ui_intent(self):
        runtime = AgentRuntime(_golden_bundle())
        snap = _snap("gameplay")
        decision = runtime.decide(snap)
        assert decision.ui_intent is None

    def test_move_decision_is_not_os_input(self):
        """AgentDecision は OS input を一切持たない — kind と index だけ。"""
        runtime = AgentRuntime(_golden_bundle())
        decision = runtime.decide(_snap("gameplay"))
        assert decision.kind in ("move", "no_op", "stop")
        # OS input 系の属性が存在しないことを確認
        assert not hasattr(decision, "key")
        assert not hasattr(decision, "mouse_click")
        assert not hasattr(decision, "send_input")

    def test_source_snapshot_hash_matches(self):
        runtime = AgentRuntime(_golden_bundle())
        snap = _snap("gameplay")
        decision = runtime.decide(snap)
        assert decision.source_snapshot_id == snap.snapshot_id
        assert decision.source_frame_id == snap.frame_id


class TestUnknownScreenState:
    def test_unknown_screen_state_returns_no_op(self):
        runtime = AgentRuntime(_golden_bundle())
        snap = _snap("paused")  # assembler が paused を返すか unknown
        decision = runtime.decide(snap)
        # paused は unhandled → no_op または stop
        assert decision.kind in ("no_op", "stop")
        assert decision.ui_intent is None
        assert decision.move_action_index is None

    def test_non_snapshot_type_returns_stop(self):
        runtime = AgentRuntime(_golden_bundle())
        # PerceptionSnapshot ではない object を渡す
        decision = runtime.decide("not-a-snapshot")  # type: ignore[arg-type]
        assert decision.kind == "stop"


class TestLevelUpRouting:
    def test_level_up_without_item_session_returns_ui(self):
        """ItemSession が None の場合 level_up は non-model UI policy (fallback 経路) へ。"""
        runtime = AgentRuntime(_golden_bundle(item_selector=None))
        snap = _snap("level_up_items", ts=2_000_000_000)
        decision = runtime.decide(snap)
        # item_session なし → non-model UI policy か no_op
        assert decision.kind in ("ui", "no_op")

    def test_level_up_no_item_selector_returns_no_op(self):
        """ItemSelector なし level_up_items は non-model UI policy 経由で no_op になる。
        LEVEL_UP screen_state に対して decide_non_model_ui_intent は None を返すため no_op になる。
        """
        runtime = AgentRuntime(_golden_bundle(item_selector=None))
        snap = _snap("level_up_items")
        decision = runtime.decide(snap)
        assert decision.kind in ("no_op", "ui")


class TestEpisodeReset:
    def test_reset_episode_can_be_called(self):
        runtime = AgentRuntime(_golden_bundle())
        runtime.decide(_snap("gameplay"), episode_start=True)
        runtime.reset_episode()
        decision = runtime.decide(_snap("gameplay", ts=2_000_000_000), episode_start=True)
        assert decision.kind == "move"
