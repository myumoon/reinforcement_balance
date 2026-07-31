"""``UiIntentV1`` の golden および one-of バリデーションのテスト。"""

import json
from pathlib import Path

import pytest

from reinbalance_survivors_contracts import (
    CHOOSE_FALLBACK_SEMANTICS,
    ContractValidationError,
    DecisionOwner,
    EFFECT_OWNER_UI_STATE_MACHINE,
    FallbackSemantic,
    UiIntentKind,
    UiIntentV1,
    allowed_semantic_actions,
    canonical_json_bytes,
)

FIXTURE = Path(__file__).parent / "fixtures" / "ui_intents_v1.json"


def _load():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_fixture_covers_every_kind():
    data = _load()
    kinds = {c["wire"]["kind"] for c in data["valid"]}
    assert kinds == {k.value for k in UiIntentKind}


@pytest.mark.parametrize("case", _load()["valid"], ids=lambda c: c["name"])
def test_valid_intent_roundtrip_and_golden_hash(case):
    intent = UiIntentV1.from_wire(case["wire"])
    # wire が byte 単位で往復する
    assert intent.to_wire() == case["wire"]
    # golden な内容ハッシュが安定している
    assert intent.intent_hash() == case["intent_hash"]
    # canonical バイト列をデコードすると等価な intent に戻る
    reparsed = UiIntentV1.from_wire(json.loads(intent.canonical_bytes()))
    assert reparsed == intent


@pytest.mark.parametrize("case", _load()["invalid"], ids=lambda c: c["name"])
def test_invalid_intent_rejected(case):
    with pytest.raises(ContractValidationError) as exc:
        UiIntentV1.from_wire(case["wire"])
    assert case["error_contains"] in str(exc.value)


def test_effect_owner_defaults_to_state_machine():
    intent = UiIntentV1(
        kind=UiIntentKind.NO_OP,
        semantic_action="no_op",
        decision_owner=DecisionOwner.RUNTIME_SAFETY,
        source_snapshot_hash="s",
        source_frame_hash="f",
        source_content_hash="c",
        ui_state_key="k",
    )
    assert intent.effect_owner == EFFECT_OWNER_UI_STATE_MACHINE


def test_state_machine_cannot_be_decision_source():
    with pytest.raises(ContractValidationError):
        UiIntentV1(
            kind=UiIntentKind.CONFIRM,
            semantic_action="confirm",
            decision_owner=DecisionOwner.NON_MODEL_UI_POLICY,
            source_snapshot_hash="s",
            source_frame_hash="f",
            source_content_hash="c",
            ui_state_key="k",
            decision_policy_id="ui_state_machine_v1",
            decision_rule_id="r",
            decision_config_hash="h",
        )


def test_canonical_bytes_omits_absent_optionals():
    intent = UiIntentV1(
        kind=UiIntentKind.STOP,
        semantic_action="stop",
        decision_owner=DecisionOwner.RUNTIME_SAFETY,
        source_snapshot_hash="s",
        source_frame_hash="f",
        source_content_hash="c",
        ui_state_key="k",
    )
    wire = json.loads(intent.canonical_bytes())
    assert "target_id" not in wire
    assert "candidate_set_hash" not in wire
    assert wire["effect_owner"] == EFFECT_OWNER_UI_STATE_MACHINE


def test_fallback_semantics_in_sync():
    """choose_fallback の許可 semantic は ui_policy.FallbackSemantic と一致しなければならない。"""
    assert CHOOSE_FALLBACK_SEMANTICS == {s.value for s in FallbackSemantic}
    assert allowed_semantic_actions(UiIntentKind.CHOOSE_FALLBACK) == (
        CHOOSE_FALLBACK_SEMANTICS
    )


def test_choose_fallback_rejects_kind_name_as_semantic():
    with pytest.raises(ContractValidationError):
        UiIntentV1(
            kind=UiIntentKind.CHOOSE_FALLBACK,
            semantic_action="choose_fallback",  # 退化形：chicken/gold でなければならない
            decision_owner=DecisionOwner.NON_MODEL_UI_POLICY,
            source_snapshot_hash="s",
            source_frame_hash="f",
            source_content_hash="c",
            ui_state_key="k",
            target_id="chicken_0",
            target_index=0,
            decision_policy_id="non_model_ui_policy_v1",
            decision_rule_id="fallback_heuristic_v1",
            decision_config_hash="h",
        )


def test_meta_kind_rejects_mismatched_semantic():
    with pytest.raises(ContractValidationError):
        UiIntentV1(
            kind=UiIntentKind.REROLL,
            semantic_action="skip",  # "reroll" と一致しなければならない
            decision_owner=DecisionOwner.NON_MODEL_UI_POLICY,
            source_snapshot_hash="s",
            source_frame_hash="f",
            source_content_hash="c",
            ui_state_key="k",
            decision_policy_id="non_model_ui_policy_v1",
            decision_rule_id="meta_choice_heuristic_v1",
            decision_config_hash="h",
        )


def test_bytes_are_sorted_and_compact():
    intent = UiIntentV1(
        kind=UiIntentKind.NO_OP,
        semantic_action="no_op",
        decision_owner=DecisionOwner.RUNTIME_SAFETY,
        source_snapshot_hash="s",
        source_frame_hash="f",
        source_content_hash="c",
        ui_state_key="k",
    )
    assert intent.canonical_bytes() == canonical_json_bytes(intent.to_wire())
