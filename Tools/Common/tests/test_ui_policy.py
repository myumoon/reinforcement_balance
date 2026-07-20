"""共有の非モデル UI policy の golden 決定表テスト。"""

import json
from pathlib import Path

import pytest

from reinbalance_survivors_contracts import (
    ButtonOption,
    ContractValidationError,
    FallbackTarget,
    NonModelUiPolicyConfigV1,
    ScreenState,
    UiIntentKind,
    UiPolicyInputV1,
    canonical_json_bytes,
    decide_non_model_ui_intent,
    sha256_hex,
)

FIXTURE = Path(__file__).parent / "fixtures" / "ui_policy_cases_v1.json"


def _load():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _default_config():
    return NonModelUiPolicyConfigV1.default_config()


def test_default_config_hash_matches_fixture():
    data = _load()
    assert _default_config().config_hash == data["config_hash"]


def test_default_config_initial_values():
    cfg = _default_config()
    assert cfg.meta_policy_enabled is False
    assert cfg.hp_chicken_threshold == pytest.approx(0.70)
    assert cfg.meta_priority == ()


@pytest.mark.parametrize("case", _load()["cases"], ids=lambda c: c["name"])
def test_decision_matches_golden(case):
    cfg = _default_config()
    inp = UiPolicyInputV1.from_wire(case["input"])
    intent = decide_non_model_ui_intent(inp, cfg)
    if case["expected"] is None:
        assert intent is None
    else:
        assert intent is not None
        assert intent.to_wire() == case["expected"]


def test_cross_package_jsonl_golden():
    """全ケースにわたる canonical JSONL は、commit 済みの golden sha と一致しなければならない。

    Training と Deployment の smoke テストは、同じ fixture からこの JSONL を再計算する。
    sha が一致することで、パッケージ間で byte-identical な intent であることを保証する。
    """
    data = _load()
    cfg = _default_config()
    lines = []
    for case in data["cases"]:
        inp = UiPolicyInputV1.from_wire(case["input"])
        intent = decide_non_model_ui_intent(inp, cfg)
        expected = intent.to_wire() if intent is not None else None
        lines.append(canonical_json_bytes({"case": case["name"], "intent": expected}))
    jsonl = b"\n".join(lines)
    assert sha256_hex(jsonl) == data["expected_jsonl_sha256"]


def _fallback_input(hp, targets):
    return UiPolicyInputV1(
        source_snapshot_hash="s",
        source_frame_hash="f",
        source_content_hash="c",
        ui_state_key="k",
        screen_state=ScreenState.FALLBACK,
        hp_fraction=hp,
        fallback_targets=targets,
    )


def test_meta_enabled_emits_priority_button():
    cfg = NonModelUiPolicyConfigV1(
        meta_policy_enabled=True, meta_priority=("reroll", "skip", "banish")
    )
    inp = UiPolicyInputV1(
        source_snapshot_hash="s",
        source_frame_hash="f",
        source_content_hash="c",
        ui_state_key="k",
        screen_state=ScreenState.LEVEL_UP,
        hp_fraction=1.0,
        button=ButtonOption("reroll", capability=True),
    )
    intent = decide_non_model_ui_intent(inp, cfg)
    assert intent is not None
    assert intent.kind is UiIntentKind.REROLL
    assert intent.decision_config_hash == cfg.config_hash


def test_meta_enabled_declines_when_not_capable():
    cfg = NonModelUiPolicyConfigV1(
        meta_policy_enabled=True, meta_priority=("reroll",)
    )
    inp = UiPolicyInputV1(
        source_snapshot_hash="s",
        source_frame_hash="f",
        source_content_hash="c",
        ui_state_key="k",
        screen_state=ScreenState.LEVEL_UP,
        hp_fraction=1.0,
        button=ButtonOption("reroll", capability=False),
    )
    assert decide_non_model_ui_intent(inp, cfg) is None


def test_meta_enabled_declines_when_not_in_priority():
    cfg = NonModelUiPolicyConfigV1(meta_policy_enabled=True, meta_priority=("skip",))
    inp = UiPolicyInputV1(
        source_snapshot_hash="s",
        source_frame_hash="f",
        source_content_hash="c",
        ui_state_key="k",
        screen_state=ScreenState.LEVEL_UP,
        hp_fraction=1.0,
        button=ButtonOption("reroll", capability=True),
    )
    assert decide_non_model_ui_intent(inp, cfg) is None


def test_input_rejects_unknown_field():
    with pytest.raises(ContractValidationError):
        UiPolicyInputV1.from_wire(
            {
                "schema_version": "ui_policy_input.v1",
                "source_snapshot_hash": "s",
                "source_frame_hash": "f",
                "source_content_hash": "c",
                "ui_state_key": "k",
                "screen_state": "gameplay",
                "hp_fraction": 1.0,
                "roi": [1, 2, 3, 4],
            }
        )


@pytest.mark.parametrize("hp", [-0.1, 1.1])
def test_input_rejects_hp_out_of_range(hp):
    with pytest.raises(ContractValidationError):
        _fallback_input(hp, ())


def _valid_input_wire():
    return {
        "schema_version": "ui_policy_input.v1",
        "source_snapshot_hash": "s",
        "source_frame_hash": "f",
        "source_content_hash": "c",
        "ui_state_key": "k",
        "screen_state": "gameplay",
        "hp_fraction": 1.0,
    }


def test_input_rejects_missing_schema_version():
    wire = _valid_input_wire()
    del wire["schema_version"]
    with pytest.raises(ContractValidationError):
        UiPolicyInputV1.from_wire(wire)


def test_input_rejects_string_hp_fraction():
    wire = {**_valid_input_wire(), "hp_fraction": "1.0"}
    with pytest.raises(ContractValidationError):
        UiPolicyInputV1.from_wire(wire)


def test_input_rejects_non_bool_fallback_valid():
    wire = {
        **_valid_input_wire(),
        "screen_state": "fallback",
        "hp_fraction": 0.5,
        "fallback_targets": [
            {"target_id": "chicken_0", "target_index": 0, "semantic": "chicken", "valid": "true"}
        ],
    }
    with pytest.raises(ContractValidationError):
        UiPolicyInputV1.from_wire(wire)


def test_fallback_target_rejects_unknown_field():
    wire = {
        **_valid_input_wire(),
        "screen_state": "fallback",
        "hp_fraction": 0.5,
        "fallback_targets": [
            {
                "target_id": "chicken_0",
                "target_index": 0,
                "semantic": "chicken",
                "valid": True,
                "roi": [1, 2, 3, 4],
            }
        ],
    }
    with pytest.raises(ContractValidationError):
        UiPolicyInputV1.from_wire(wire)


def test_button_rejects_unknown_field():
    wire = {
        **_valid_input_wire(),
        "screen_state": "level_up",
        "button": {"semantic": "reroll", "capability": True, "roi": [0, 0, 1, 1]},
    }
    with pytest.raises(ContractValidationError):
        UiPolicyInputV1.from_wire(wire)


def _valid_config_wire():
    return {
        "schema_version": "non_model_ui_policy_config.v1",
        "policy_id": "non_model_ui_policy_v1",
        "fallback_rule_id": "fallback_heuristic_v1",
        "meta_rule_id": "meta_choice_heuristic_v1",
        "ack_confirm_rule_id": "ack_confirm_rule_v1",
        "hp_chicken_threshold": 0.7,
        "meta_policy_enabled": False,
        "meta_priority": [],
    }


def test_config_from_wire_rejects_missing_schema_version():
    wire = _valid_config_wire()
    del wire["schema_version"]
    with pytest.raises(ContractValidationError):
        NonModelUiPolicyConfigV1.from_wire(wire)


def test_config_from_wire_rejects_unknown_field():
    wire = {**_valid_config_wire(), "extra_knob": 1}
    with pytest.raises(ContractValidationError):
        NonModelUiPolicyConfigV1.from_wire(wire)


def test_button_option_direct_construction_validates():
    with pytest.raises(ContractValidationError):
        ButtonOption("reroll", capability="false")  # str は bool ではない


def test_fallback_target_direct_construction_validates():
    with pytest.raises(ContractValidationError):
        FallbackTarget("chicken_0", 0, "chicken", "true")  # valid は bool でなければならない


def test_config_direct_construction_validates():
    with pytest.raises(ContractValidationError):
        NonModelUiPolicyConfigV1(meta_policy_enabled="false")  # str は bool ではない
