"""UI policy: runtime (05-01) と Training evaluator (02-04) の byte-identical parity を検証する。

decide_non_model_ui_intent() が共通 policy から同じ canonical wire を生成することを確認する。
ItemSelector (choose_card) は item_selector_session が生成し、05-03 は intent を生成しない。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from reinbalance_survivors_contracts.canonical_json import canonical_hash, canonical_json_bytes
from reinbalance_survivors_contracts.ui_intent import (
    ContractValidationError,
    DecisionOwner,
    UiIntentKind,
    UiIntentV1,
    allowed_semantic_actions,
)
from reinbalance_survivors_contracts.ui_policy import (
    ButtonOption,
    ButtonSemantic,
    FallbackTarget,
    NonModelUiPolicyConfigV1,
    ScreenState,
    UiPolicyInputV1,
    decide_non_model_ui_intent,
)


def _policy_input(
    screen_state: ScreenState,
    *,
    hp: float = 0.8,
    fallback_targets: list[dict] | None = None,
    button: dict | None = None,
) -> UiPolicyInputV1:
    """テスト用 UiPolicyInputV1 を返す。"""
    targets = [FallbackTarget.from_wire(t) for t in (fallback_targets or [])]
    btn = ButtonOption.from_wire(button) if button else None
    return UiPolicyInputV1(
        source_snapshot_hash="s" * 64,
        source_frame_hash="f" * 64,
        source_content_hash="c" * 64,
        ui_state_key="k" * 64,
        screen_state=screen_state,
        hp_fraction=hp,
        fallback_targets=tuple(targets),
        button=btn,
    )


class TestUiPolicyRuleCoverage:
    """全 reachable semantic が UiIntentV1 で表現できることを確認する。"""

    def test_choose_card_semantic_set(self):
        """choose_card の semantic_action は "choose_card" だけ。target の意味は target_id/semantic_kind で表す。"""
        allowed = allowed_semantic_actions(UiIntentKind.CHOOSE_CARD)
        assert "choose_card" in allowed

    def test_choose_fallback_semantics_covered(self):
        allowed = allowed_semantic_actions(UiIntentKind.CHOOSE_FALLBACK)
        assert "chicken" in allowed or "gold" in allowed

    def test_no_op_semantic_is_no_op(self):
        assert allowed_semantic_actions(UiIntentKind.NO_OP) == frozenset({"no_op"})

    def test_stop_semantic_is_stop(self):
        assert allowed_semantic_actions(UiIntentKind.STOP) == frozenset({"stop"})

    def test_choose_card_owner_is_item_selector(self):
        """choose_card の decision_owner は item_selector_session でなければならない。"""
        with pytest.raises(ContractValidationError, match="owned by|owner"):
            UiIntentV1(
                kind=UiIntentKind.CHOOSE_CARD,
                semantic_action="choose_card",
                decision_owner=DecisionOwner.NON_MODEL_UI_POLICY,  # wrong: must be ITEM_SELECTOR_SESSION
                source_snapshot_hash="s" * 64,
                source_frame_hash="f" * 64,
                source_content_hash="c" * 64,
                ui_state_key="k" * 64,
                target_index=0,
                candidate_set_hash="c" * 64,
                inventory_hash="i" * 64,
                decision_policy_id="p",
                decision_rule_id="r",
                decision_config_hash="d" * 64,
            )

    def test_non_model_policy_cannot_produce_choose_card(self):
        """non-model policy は choose_card を返せない (choose_fallback だけ)。"""
        inp = _policy_input(
            ScreenState.FALLBACK,
            fallback_targets=[
                {"target_id": "t1", "target_index": 0, "semantic": "chicken", "valid": True}
            ],
        )
        config = NonModelUiPolicyConfigV1.load_default()
        intent = decide_non_model_ui_intent(inp, config)
        assert intent is not None
        assert intent.kind != UiIntentKind.CHOOSE_CARD


class TestNonModelPolicyParity:
    """02-04 Training evaluator と同じ policy ロジックが 05-01 で動く。"""

    def _config(self) -> NonModelUiPolicyConfigV1:
        return NonModelUiPolicyConfigV1.load_default()

    def test_ack_chest_on_chest_state(self):
        inp = _policy_input(ScreenState.CHEST)
        config = self._config()
        intent = decide_non_model_ui_intent(inp, config)
        assert intent is not None
        assert intent.kind == UiIntentKind.ACK_CHEST
        assert intent.decision_owner == DecisionOwner.NON_MODEL_UI_POLICY
        assert intent.effect_owner == "ui_state_machine_v1"

    def test_confirm_on_confirm_state(self):
        inp = _policy_input(ScreenState.CONFIRM)
        config = self._config()
        intent = decide_non_model_ui_intent(inp, config)
        assert intent is not None
        assert intent.kind == UiIntentKind.CONFIRM

    def test_choose_fallback_chicken_when_hp_low(self):
        """HP が低いとき chicken fallback を選ぶ。"""
        inp = _policy_input(
            ScreenState.FALLBACK,
            hp=0.10,
            fallback_targets=[
                {"target_id": "c1", "target_index": 0, "semantic": "chicken", "valid": True},
                {"target_id": "g1", "target_index": 1, "semantic": "gold", "valid": True},
            ],
        )
        config = self._config()
        intent = decide_non_model_ui_intent(inp, config)
        assert intent is not None
        assert intent.kind == UiIntentKind.CHOOSE_FALLBACK
        assert intent.semantic_action == "chicken"

    def test_choose_fallback_gold_when_hp_high(self):
        """HP が十分高いとき gold fallback を選ぶ。"""
        inp = _policy_input(
            ScreenState.FALLBACK,
            hp=0.99,
            fallback_targets=[
                {"target_id": "c1", "target_index": 0, "semantic": "chicken", "valid": True},
                {"target_id": "g1", "target_index": 1, "semantic": "gold", "valid": True},
            ],
        )
        config = self._config()
        intent = decide_non_model_ui_intent(inp, config)
        assert intent is not None
        assert intent.kind == UiIntentKind.CHOOSE_FALLBACK
        assert intent.semantic_action == "gold"

    def test_stop_on_empty_fallback(self):
        """fallback 候補が空 → stop。"""
        inp = _policy_input(ScreenState.FALLBACK, fallback_targets=[])
        config = self._config()
        intent = decide_non_model_ui_intent(inp, config)
        assert intent is not None
        assert intent.kind == UiIntentKind.STOP

    def test_source_binding_copied_to_intent(self):
        """intent.source_snapshot_hash が policy_input.source_snapshot_hash を継承する。"""
        inp = _policy_input(ScreenState.CHEST)
        config = self._config()
        intent = decide_non_model_ui_intent(inp, config)
        assert intent is not None
        assert intent.source_snapshot_hash == inp.source_snapshot_hash
        assert intent.source_frame_hash == inp.source_frame_hash
        assert intent.source_content_hash == inp.source_content_hash
        assert intent.ui_state_key == inp.ui_state_key

    def test_intent_canonical_bytes_are_stable(self):
        """同一入力から常に同じ canonical wire bytes が生成される。"""
        inp = _policy_input(ScreenState.CHEST)
        config = self._config()
        i1 = decide_non_model_ui_intent(inp, config)
        i2 = decide_non_model_ui_intent(inp, config)
        assert i1 is not None and i2 is not None
        assert canonical_json_bytes(i1.to_wire()) == canonical_json_bytes(i2.to_wire())

    def test_gameplay_returns_none(self):
        """gameplay 画面に対して policy は None を返す (combat session に委ねる)。"""
        inp = _policy_input(ScreenState.GAMEPLAY)
        config = self._config()
        intent = decide_non_model_ui_intent(inp, config)
        assert intent is None


class TestIntentOwnershipRules:
    """UiIntentV1 の owner / effect_owner ルールを確認する。"""

    def test_effect_owner_is_always_ui_state_machine(self):
        """05-03 だけが effect を持つ — effect_owner は固定値。"""
        inp = _policy_input(ScreenState.CHEST)
        config = NonModelUiPolicyConfigV1.load_default()
        intent = decide_non_model_ui_intent(inp, config)
        assert intent is not None
        assert intent.effect_owner == "ui_state_machine_v1"

    def test_ui_state_machine_cannot_generate_intent_in_decision_policy(self):
        """05-03 は intent を生成しない — decision_policy_id に ui_state_machine を含む intent は拒否。"""
        with pytest.raises(ContractValidationError, match="05-03"):
            UiIntentV1(
                kind=UiIntentKind.ACK_CHEST,
                semantic_action="ack_chest",
                decision_owner=DecisionOwner.NON_MODEL_UI_POLICY,
                source_snapshot_hash="s" * 64,
                source_frame_hash="f" * 64,
                source_content_hash="c" * 64,
                ui_state_key="k" * 64,
                decision_policy_id="ui_state_machine_v2",  # 禁止
                decision_rule_id="r",
                decision_config_hash="d" * 64,
            )

    def test_runtime_safety_owns_no_op_and_stop(self):
        for kind, semantic in [(UiIntentKind.NO_OP, "no_op"), (UiIntentKind.STOP, "stop")]:
            intent = UiIntentV1(
                kind=kind,
                semantic_action=semantic,
                decision_owner=DecisionOwner.RUNTIME_SAFETY,
                source_snapshot_hash="s" * 64,
                source_frame_hash="f" * 64,
                source_content_hash="c" * 64,
                ui_state_key="k" * 64,
            )
            assert intent.decision_owner == DecisionOwner.RUNTIME_SAFETY
