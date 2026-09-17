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
from reinbalance_survivors_contracts.ui_policy import NonModelUiPolicyConfigV1, ScreenState
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
from survivors.runtime.artifact_bundle import BundleLoadError, RuntimeBundle
from survivors.runtime.decision_scheduler import DecisionScheduler

TICK_NS = DecisionScheduler().interval_ns


class _TickingClock:
    """呼び出しごとに指定 timestamp を返す test clock。

    やさしい説明: 実時間に依存せず inference timeout を全経路で再現します。
    """

    def __init__(self, *values: int) -> None:
        """返却値の列を記録する。

        やさしい説明: runtime が想定以上に時計を読むと StopIteration でテストを失敗させます。
        """
        self._values = iter(values)

    def __call__(self) -> int:
        """次の timestamp を返す。

        やさしい説明: 一回の推論時間を任意の差分へ固定します。
        """
        return next(self._values)


class _ItemSelector:
    """item runtime 経路を通す最小 selector double。

    やさしい説明: 外部 ONNX 実行を避けつつ ItemSession の実入力組み立てを通します。
    """

    nmax = 3
    feature_schema = "context_only_v1"
    temperature = 1.0
    confidence_threshold = 0.0
    ui_policy_config = NonModelUiPolicyConfigV1.load_default()

    def predict(self, context: np.ndarray, candidates: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """先頭の valid card を選ぶ logits を返す。

        やさしい説明: shape/mask は ItemSession が組み立てた実値のまま受け取ります。
        """
        return np.array([[9.0, 0.0, 0.0]], dtype=np.float32)


def _hud_world(screen_state: str = "gameplay", *, ts: int = 1_000_000_000):
    """テスト用 HUD + world state を返す。

    やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
    """
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
    """RealObsAssembler で PerceptionSnapshot を作る。

    やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
    """
    schema = DeployObsSchema.default_v1()
    hud, world = _hud_world(screen_state, ts=ts)
    snap = RealObsAssembler().assemble(hud, world, schema, (1000, 1000))
    assert snap is not None, f"assembler returned None for screen_state={screen_state!r}"
    return snap


def _fresh_now(snapshot) -> int:
    """snapshot が age gate を通る monotonic 時刻を返す。

    やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
    """
    return snapshot.captured_ns + 5_000_000


@pytest.fixture()
def runtime(golden_combat_policy) -> AgentRuntime:
    """ItemSelector を持たない golden bundle の runtime。

    やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
    """
    return AgentRuntime(RuntimeBundle.from_golden_fixture(golden_combat_policy))


class TestAgentDecisionContract:
    """[指摘10] plan 05-01 の AgentDecision 契約と canonical wire を検証する。

    やさしい説明: このまとまりが担当する境界を分かりやすく示します。
    """

    def test_field_names_match_plan_contract(self):
        """test_field_names_match_plan_contract の契約を検証する。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        names = [f.name for f in dataclasses.fields(AgentDecision)]
        for expected in (
            "decision_id", "kind", "action_index", "ui_intent", "confidence", "reason",
            "source_snapshot_id", "source_frame_id", "source_content_hash",
            "snapshot_timestamp_ns", "inference_started_ns", "inference_finished_ns",
        ):
            assert expected in names, expected

    def test_move_requires_action_index(self):
        """test_move_requires_action_index の契約を検証する。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        with pytest.raises(ValueError, match="action_index"):
            AgentDecision(
                decision_id="x", kind="move", action_index=None, ui_intent=None,
                confidence=0.5, reason="r", source_snapshot_id="s", source_frame_id="f",
                source_content_hash="c", snapshot_timestamp_ns=0,
                inference_started_ns=0, inference_finished_ns=0,
            )

    def test_ui_requires_intent(self):
        """test_ui_requires_intent の契約を検証する。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        with pytest.raises(ValueError, match="ui_intent"):
            AgentDecision(
                decision_id="x", kind="ui", action_index=None, ui_intent=None,
                confidence=0.5, reason="r", source_snapshot_id="s", source_frame_id="f",
                source_content_hash="c", snapshot_timestamp_ns=0,
                inference_started_ns=0, inference_finished_ns=0,
            )

    def test_no_op_must_not_have_action_index(self):
        """test_no_op_must_not_have_action_index の契約を検証する。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        with pytest.raises(ValueError, match="no_op"):
            AgentDecision(
                decision_id="x", kind="no_op", action_index=3, ui_intent=None,
                confidence=0.0, reason="r", source_snapshot_id="s", source_frame_id="f",
                source_content_hash="c", snapshot_timestamp_ns=0,
                inference_started_ns=0, inference_finished_ns=0,
            )

    def test_confidence_out_of_range_rejected(self):
        """test_confidence_out_of_range_rejected の契約を検証する。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        with pytest.raises(ValueError, match="confidence"):
            AgentDecision(
                decision_id="x", kind="no_op", action_index=None, ui_intent=None,
                confidence=1.5, reason="r", source_snapshot_id="s", source_frame_id="f",
                source_content_hash="c", snapshot_timestamp_ns=0,
                inference_started_ns=0, inference_finished_ns=0,
            )

    def test_empty_reason_rejected(self):
        """test_empty_reason_rejected の契約を検証する。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        with pytest.raises(ValueError, match="reason"):
            AgentDecision(
                decision_id="x", kind="no_op", action_index=None, ui_intent=None,
                confidence=0.0, reason="", source_snapshot_id="s", source_frame_id="f",
                source_content_hash="c", snapshot_timestamp_ns=0,
                inference_started_ns=0, inference_finished_ns=0,
            )

    @pytest.mark.parametrize("action_index", [True, -1, 9])
    def test_move_action_index_must_be_strict_and_in_range(self, action_index) -> None:
        """move の action index に bool・負数・9以上を許さない。

        やさしい説明: 共有9 action map に存在しない番号を downstream へ渡しません。
        """
        with pytest.raises(ValueError, match="action_index"):
            AgentDecision(
                decision_id="x",
                kind="move",
                action_index=action_index,
                ui_intent=None,
                confidence=0.5,
                reason="r",
                source_snapshot_id="s",
                source_frame_id="f",
                source_content_hash="c",
                snapshot_timestamp_ns=0,
                inference_started_ns=0,
                inference_finished_ns=0,
            )

    def test_inference_timestamps_must_be_ordered(self) -> None:
        """推論終了時刻が開始時刻より前の decision を拒否する。

        やさしい説明: latency が負になる壊れた wire を telemetry へ残しません。
        """
        with pytest.raises(ValueError, match="inference timestamps"):
            AgentDecision(
                decision_id="x",
                kind="no_op",
                action_index=None,
                ui_intent=None,
                confidence=0.0,
                reason="r",
                source_snapshot_id="s",
                source_frame_id="f",
                source_content_hash="c",
                snapshot_timestamp_ns=0,
                inference_started_ns=2,
                inference_finished_ns=1,
            )

    def test_wire_roundtrip_preserves_hash(self, runtime):
        """test_wire_roundtrip_preserves_hash の契約を検証する。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        snap = _snap("gameplay")
        decision = runtime.decide(snap, now_ns=_fresh_now(snap), episode_start=True)
        restored = AgentDecision.from_wire(json.loads(decision.canonical_bytes()))
        assert restored.decision_hash() == decision.decision_hash()

    def test_wire_is_byte_stable(self, runtime):
        """test_wire_is_byte_stable の契約を検証する。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        snap = _snap("gameplay")
        decision = runtime.decide(snap, now_ns=_fresh_now(snap), episode_start=True)
        assert decision.canonical_bytes() == decision.canonical_bytes()
        assert decision.to_wire()["schema_version"] == AGENT_DECISION_SCHEMA_VERSION

    def test_ui_decision_wire_embeds_intent(self, runtime):
        """test_ui_decision_wire_embeds_intent の契約を検証する。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        snap = _snap("chest", ts=2_000_000_000)
        decision = runtime.decide(snap, now_ns=_fresh_now(snap), episode_start=True)
        assert decision.kind == "ui"
        wire = decision.to_wire()
        assert wire["ui_intent"]["kind"] == UiIntentKind.ACK_CHEST.value

    def test_ui_intent_source_binding_must_match_decision(self, runtime):
        """UiIntent と AgentDecision の三つの source identity を一致させる。

        やさしい説明: 別frameのintentを現在のdecisionへ差し替えるcandidate remapを拒否します。
        """
        snap = _snap("chest", ts=2_100_000_000)
        decision = runtime.decide(snap, now_ns=_fresh_now(snap), episode_start=True)
        assert decision.ui_intent is not None
        remapped = dataclasses.replace(decision.ui_intent, source_frame_hash="other-frame")

        with pytest.raises(ValueError, match="source binding"):
            dataclasses.replace(decision, ui_intent=remapped)

    def test_jsonl_has_one_line_per_decision(self, runtime):
        """test_jsonl_has_one_line_per_decision の契約を検証する。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        snap = _snap("gameplay")
        first = runtime.decide(snap, now_ns=_fresh_now(snap), episode_start=True)
        chest = _snap("chest", ts=2_000_000_000)
        second = runtime.decide(chest, now_ns=_fresh_now(chest) + TICK_NS)
        payload = decisions_to_jsonl([first, second])
        lines = payload.splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["decision_id"] == first.decision_id

    def test_decision_has_no_os_input_fields(self, runtime):
        """test_decision_has_no_os_input_fields の契約を検証する。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        snap = _snap("gameplay")
        decision = runtime.decide(snap, now_ns=_fresh_now(snap), episode_start=True)
        for forbidden in ("key", "mouse_click", "send_input", "roi", "click"):
            assert not hasattr(decision, forbidden)


class TestTypedScreenStateRouting:
    """[指摘4] raw state の重複表ではなく ui_policy_input.screen_state で route する。

    やさしい説明: このまとまりが担当する境界を分かりやすく示します。
    """

    def test_no_duplicated_raw_screen_state_table(self):
        """runtime が raw state 名の集合を独自に持たないこと。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        source = inspect.getsource(agent_runtime_module)
        assert "_NON_MODEL_UI_SCREEN_STATES" not in source
        assert "_ITEM_SCREEN_STATES" not in source

    def test_target_reached_transition_produces_confirm(self, runtime):
        """assembler の raw state は target_reached_transition。confirm intent になる。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        snap = _snap("target_reached_transition", ts=3_000_000_000)
        assert snap.ui_policy_input.screen_state == ScreenState.CONFIRM
        decision = runtime.decide(snap, now_ns=_fresh_now(snap), episode_start=True)
        assert decision.kind == "ui"
        assert decision.ui_intent is not None
        assert decision.ui_intent.kind == UiIntentKind.CONFIRM

    def test_confirm_intent_is_not_no_op(self, runtime):
        """レビュー再現: confirm intent が常に no_op になってはいけない。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        snap = _snap("target_reached_transition", ts=3_500_000_000)
        decision = runtime.decide(snap, now_ns=_fresh_now(snap), episode_start=True)
        assert decision.kind != "no_op"

    def test_chest_produces_ack_chest(self, runtime):
        """test_chest_produces_ack_chest の契約を検証する。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        snap = _snap("chest", ts=4_000_000_000)
        decision = runtime.decide(snap, now_ns=_fresh_now(snap), episode_start=True)
        assert decision.kind == "ui"
        assert decision.ui_intent.kind == UiIntentKind.ACK_CHEST

    def test_gameplay_produces_move(self, runtime):
        """test_gameplay_produces_move の契約を検証する。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        snap = _snap("gameplay")
        decision = runtime.decide(snap, now_ns=_fresh_now(snap), episode_start=True)
        assert decision.kind == "move"
        assert isinstance(decision.action_index, int)
        assert 0 <= decision.action_index < 9
        assert decision.ui_intent is None

    @pytest.mark.parametrize(
        ("screen_state", "field", "value"),
        [
            ("gameplay", "source_snapshot_hash", "x" * 64),
            ("gameplay", "source_frame_hash", "x" * 64),
            ("gameplay", "source_content_hash", "x" * 64),
            ("gameplay", "ui_state_key", "x" * 64),
            ("chest", "screen_state", ScreenState.GAMEPLAY),
            ("level_up_items", "screen_state", ScreenState.GAMEPLAY),
            ("target_reached_transition", "screen_state", ScreenState.GAMEPLAY),
        ],
    )
    def test_unbound_ui_policy_input_is_rejected_before_routing(
        self, runtime, screen_state, field, value
    ):
        """外側 snapshot と一致しない typed UI 入力を経路選択前に拒否する。

        やさしい説明: 別 frame の状態を使って move/ui を選ぶ回帰を全 binding で捕えます。
        """
        snapshot = _snap(screen_state)
        assert snapshot.ui_policy_input is not None
        unbound_input = dataclasses.replace(
            snapshot.ui_policy_input, **{field: value}
        )
        unbound_snapshot = dataclasses.replace(
            snapshot, ui_policy_input=unbound_input
        )

        decision = runtime.decide(
            unbound_snapshot,
            now_ns=_fresh_now(unbound_snapshot),
            episode_start=True,
        )

        assert decision.kind in {"no_op", "stop"}
        assert "binding" in decision.reason
        assert runtime.screen_scheduler.evaluation_count == 0
        assert runtime.combat_session.recurrent_state_copy() is None

    def test_effect_owner_is_ui_state_machine(self, runtime):
        """test_effect_owner_is_ui_state_machine の契約を検証する。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        snap = _snap("chest", ts=4_500_000_000)
        decision = runtime.decide(snap, now_ns=_fresh_now(snap), episode_start=True)
        assert decision.ui_intent.effect_owner == "ui_state_machine_v1"

    def test_non_snapshot_returns_stop(self, runtime):
        """test_non_snapshot_returns_stop の契約を検証する。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        assert runtime.decide("not-a-snapshot").kind == "stop"


class TestSnapshotTimeAndValidityGates:
    """[指摘5] captured_ns の age・global validity・timeout を検証する。

    やさしい説明: このまとまりが担当する境界を分かりやすく示します。
    """

    def test_zero_captured_ns_does_not_return_move(self, runtime):
        """レビュー再現: captured_ns=0 でも move を返してはいけない。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        snap = dataclasses.replace(_snap("gameplay"), captured_ns=0)
        decision = runtime.decide(snap, now_ns=2_000_000_000, episode_start=True)
        assert decision.kind == "no_op"
        assert decision.action_index is None
        assert "age gate" in decision.reason

    def test_stale_snapshot_does_not_return_move(self, runtime):
        """test_stale_snapshot_does_not_return_move の契約を検証する。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        snap = _snap("gameplay")
        stale_now = snap.captured_ns + 10 * TICK_NS
        decision = runtime.decide(snap, now_ns=stale_now, episode_start=True)
        assert decision.kind == "no_op"
        assert "age gate" in decision.reason
        assert runtime.screen_scheduler.evaluation_count == 0

    def test_future_snapshot_does_not_return_move(self, runtime):
        """test_future_snapshot_does_not_return_move の契約を検証する。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        snap = _snap("gameplay")
        decision = runtime.decide(
            snap, now_ns=snap.captured_ns - 1_000_000, episode_start=True
        )
        assert decision.kind == "no_op"

    def test_future_snapshot_does_not_poison_next_valid_snapshot(self, runtime):
        """未来 timestamp の拒否後も次の正常 snapshot を処理する。

        やさしい説明: age gate 前に replay cursor を進めて正常 frame まで stale 扱いする回帰を捕えます。
        """
        future = _snap("gameplay", ts=2_000_000_000)
        rejected = runtime.decide(
            future, now_ns=1_000_000_000, episode_start=True
        )
        assert rejected.kind == "no_op"
        assert "age gate" in rejected.reason

        valid = _snap("gameplay", ts=1_033_000_000)
        accepted = runtime.decide(valid, now_ns=_fresh_now(valid))
        assert accepted.kind == "move"

    def test_low_global_validity_does_not_return_move(self, runtime):
        """観測が信用できないときは movement action を返さない。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
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
        """締切超過の推論結果は move にせず no_op へ倒す。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        bundle = RuntimeBundle.from_golden_fixture(golden_combat_policy)
        runtime = AgentRuntime(bundle, scheduler=DecisionScheduler(inference_timeout_ns=1))
        snap = _snap("gameplay")
        decision = runtime.decide(snap, now_ns=_fresh_now(snap), episode_start=True)
        assert decision.kind == "no_op"
        assert "timeout" in decision.reason

    def test_inference_timeout_does_not_commit_recurrent_state(
        self, golden_combat_policy
    ):
        """timeout した combat 推論は recurrent state を確定しない。

        やさしい説明: 破棄した action の内部記憶だけが次 tick へ残る回帰を捕えます。
        """
        bundle = RuntimeBundle.from_golden_fixture(golden_combat_policy)
        runtime = AgentRuntime(
            bundle,
            scheduler=DecisionScheduler(inference_timeout_ns=5),
            clock_ns=_TickingClock(100, 110, 120),
        )
        snapshot = _snap("gameplay")

        decision = runtime.decide(
            snapshot, now_ns=_fresh_now(snapshot), episode_start=True
        )

        assert decision.kind == "no_op"
        assert decision.reason == "inference timeout gate failed"
        assert runtime.combat_session.recurrent_state_copy() is None
        assert runtime.combat_session.episode_start_pending is True


class TestEpisodeBoundaryReset:
    """[指摘5] death / result / unknown gap で recurrent state を破棄する。

    やさしい説明: このまとまりが担当する境界を分かりやすく示します。
    """

    @pytest.mark.parametrize("boundary_state", ["death", "result", "unknown"])
    def test_boundary_state_resets_recurrent_state(self, runtime, boundary_state):
        """test_boundary_state_resets_recurrent_state の契約を検証する。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        gameplay = _snap("gameplay")
        runtime.decide(gameplay, now_ns=_fresh_now(gameplay), episode_start=True)
        assert runtime.combat_session.lstm_state_copy() is not None

        boundary = _snap(boundary_state, ts=5_000_000_000)
        decision = runtime.decide(boundary, now_ns=_fresh_now(boundary) + TICK_NS)
        assert decision.kind == "no_op"
        assert runtime.combat_session.lstm_state_copy() is None
        assert runtime.combat_session.episode_start_pending is True

    def test_hidden_state_does_not_leak_into_next_run(self, runtime):
        """レビュー再現: death 後に hidden state が残ってはいけない。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
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
        """test_paused_does_not_reset_state の契約を検証する。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        gameplay = _snap("gameplay")
        runtime.decide(gameplay, now_ns=_fresh_now(gameplay), episode_start=True)
        paused = _snap("paused", ts=8_000_000_000)
        decision = runtime.decide(paused, now_ns=_fresh_now(paused) + TICK_NS)
        assert decision.kind == "no_op"
        assert runtime.combat_session.lstm_state_copy() is not None

    def test_reset_episode_clears_state(self, runtime):
        """test_reset_episode_clears_state の契約を検証する。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        gameplay = _snap("gameplay")
        runtime.decide(gameplay, now_ns=_fresh_now(gameplay), episode_start=True)
        runtime.reset_episode()
        assert runtime.combat_session.lstm_state_copy() is None


class TestSchedulerIntegration:
    """[指摘7] DecisionScheduler が実際の decide flow に組み込まれている。

    やさしい説明: このまとまりが担当する境界を分かりやすく示します。
    """

    def test_runtime_exposes_scheduler(self, runtime):
        """test_runtime_exposes_scheduler の契約を検証する。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        assert isinstance(runtime.scheduler, DecisionScheduler)
        assert runtime.scheduler.hz == 15

    def test_off_cadence_call_is_skipped(self, runtime):
        """test_off_cadence_call_is_skipped の契約を検証する。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        snap = _snap("gameplay")
        now = _fresh_now(snap)
        first = runtime.decide(snap, now_ns=now, episode_start=True)
        assert first.kind == "move"
        follow = _snap("gameplay", ts=snap.captured_ns + 1_000_000)
        second = runtime.decide(follow, now_ns=now + 1_000_000)
        assert second.kind == "no_op"
        assert "off-cadence" in second.reason

    def test_replayed_snapshot_is_rejected_before_any_decision_path(self, runtime):
        """同一 capture binding の再処理を stale として拒否する。

        やさしい説明: 前 frame を再送して combat/UI 判断を二重実行する回帰を捕えます。
        """
        snap = _snap("gameplay")
        now = _fresh_now(snap)
        assert runtime.decide(snap, now_ns=now, episode_start=True).kind == "move"

        replay = runtime.decide(snap, now_ns=now + 1_000_000)

        assert replay.kind == "no_op"
        assert replay.reason == "stale snapshot gate failed"

    def test_next_tick_is_served(self, runtime):
        """test_next_tick_is_served の契約を検証する。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        snap = _snap("gameplay")
        now = _fresh_now(snap)
        runtime.decide(snap, now_ns=now, episode_start=True)
        follow = _snap("gameplay", ts=snap.captured_ns + TICK_NS)
        decision = runtime.decide(follow, now_ns=now + TICK_NS + 1)
        assert decision.kind == "move"

    def test_backlog_skips_are_recorded(self, runtime):
        """test_backlog_skips_are_recorded の契約を検証する。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        snap = _snap("gameplay")
        now = _fresh_now(snap)
        runtime.decide(snap, now_ns=now, episode_start=True)
        late = _snap("gameplay", ts=snap.captured_ns + 6 * TICK_NS)
        runtime.decide(late, now_ns=now + 6 * TICK_NS)
        assert runtime.scheduler.skipped_tick_count >= 4

    def test_decision_count_matches_served_ticks(self, runtime):
        """test_decision_count_matches_served_ticks の契約を検証する。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
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
        """test_timing_metadata_is_recorded の契約を検証する。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        snap = _snap("gameplay")
        runtime.decide(snap, now_ns=_fresh_now(snap), episode_start=True)
        timing = runtime.last_timing
        assert timing is not None
        assert timing.inference_latency_ms >= 0.0

    def test_recorded_sequence_is_reproducible(self, golden_combat_policy):
        """同一 snapshot 列 / 同一 tick 列から同じ decision 列を再生する。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        def run() -> list[tuple[str, int | None]]:
            """run の契約を検証する。

            やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
            """
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
    """runtime は typed decision を返すだけで OS input を送らない。

    やさしい説明: このまとまりが担当する境界を分かりやすく示します。
    """

    def test_runtime_module_has_no_input_imports(self):
        """test_runtime_module_has_no_input_imports の契約を検証する。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        source = inspect.getsource(agent_runtime_module)
        for forbidden in ("SendInput", "pyautogui", "win32api", "ctypes", "keyboard"):
            assert forbidden not in source

    def test_decision_kinds_are_limited(self, runtime):
        """test_decision_kinds_are_limited の契約を検証する。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        snap = _snap("gameplay")
        decision = runtime.decide(snap, now_ns=_fresh_now(snap), episode_start=True)
        assert decision.kind in ("move", "ui", "no_op", "stop")

    def test_invalid_bundle_rejected(self):
        """test_invalid_bundle_rejected の契約を検証する。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        with pytest.raises(ValueError, match="RuntimeBundle"):
            AgentRuntime(object())  # type: ignore[arg-type]

    def test_live_startup_uses_bundle_eligibility_gate(self, golden_combat_policy):
        """live 起動は RuntimeBundle 自身の trust gate に委譲する。

        やさしい説明: golden fixture を本番 runtime として使おうとすると起動前に止まります。
        """
        bundle = RuntimeBundle.from_golden_fixture(golden_combat_policy)
        with pytest.raises(BundleLoadError, match="development_only"):
            AgentRuntime(bundle, require_live=True)


class TestPr2CadenceAndSymmetricGates:
    """PR2 の UI cadence 分離と対称 safety gate を検証する。

    やさしい説明: combat だけを 15 Hz に制限し、UI は 30 Hz frame ごとに評価します。
    """

    def test_ui_transition_is_processed_inside_combat_interval(self, runtime: AgentRuntime) -> None:
        """combat tick 間の chest frame も UI decision になる。

        やさしい説明: 33 ms 後の UI transition を 66 ms combat gate が捨てる回帰を捕えます。
        """
        gameplay = _snap("gameplay", ts=1_000_000_000)
        now = _fresh_now(gameplay)
        assert runtime.decide(gameplay, now_ns=now, episode_start=True).kind == "move"
        chest = _snap("chest", ts=gameplay.captured_ns + 33_000_000)
        decision = runtime.decide(chest, now_ns=now + 33_000_000)
        assert decision.kind == "ui"
        assert decision.ui_intent is not None
        assert decision.ui_intent.kind == UiIntentKind.ACK_CHEST
        assert runtime.screen_scheduler.evaluation_count == 2
        assert runtime.scheduler.decision_count == 1

    @pytest.mark.parametrize("screen_state", ["gameplay", "level_up_items", "chest"])
    def test_snapshot_age_gate_is_shared_by_all_decision_paths(
        self, runtime: AgentRuntime, screen_state: str
    ) -> None:
        """combat/item/non-model UI の stale snapshot を同じ age gate で拒否する。

        やさしい説明: UI だけ古い frame を操作できる非対称を防ぎます。
        """
        snapshot = _snap(screen_state)
        decision = runtime.decide(
            snapshot,
            now_ns=snapshot.captured_ns + 10 * TICK_NS,
            episode_start=True,
        )
        assert decision.kind == "no_op"
        assert decision.reason.startswith("snapshot age gate failed")

    @pytest.mark.parametrize("screen_state", ["gameplay", "level_up_items", "chest"])
    def test_inference_timeout_is_shared_by_all_decision_paths(
        self, golden_combat_policy, screen_state: str
    ) -> None:
        """combat/item/non-model UI の遅い結果を同じ timeout gate で破棄する。

        やさしい説明: model を使わない UI 経路も古い画面への intent を返しません。
        """
        selector = _ItemSelector() if screen_state == "level_up_items" else None
        bundle = RuntimeBundle.from_golden_fixture(
            golden_combat_policy,
            item_selector=selector,
        )
        runtime = AgentRuntime(
            bundle,
            scheduler=DecisionScheduler(inference_timeout_ns=5),
            clock_ns=_TickingClock(100, 110, 120),
        )
        snapshot = _snap(screen_state)
        decision = runtime.decide(snapshot, now_ns=_fresh_now(snapshot), episode_start=True)
        assert decision.kind == "no_op"
        assert decision.reason == "inference timeout gate failed"

    def test_unknown_capability_owner_blocks_startup(self, golden_combat_policy) -> None:
        """owner のない meta capability を runtime 起動時に拒否する。

        やさしい説明: 誰が決定するか不明な semantic を実 frame 到着まで持ち越しません。
        """
        config = dataclasses.replace(
            NonModelUiPolicyConfigV1.load_default(),
            meta_policy_enabled=True,
            meta_priority=("unowned_capability",),
        )
        bundle = RuntimeBundle.from_golden_fixture(
            golden_combat_policy,
            ui_policy_config=config,
        )
        with pytest.raises(ValueError, match="capability owner"):
            AgentRuntime(bundle)
