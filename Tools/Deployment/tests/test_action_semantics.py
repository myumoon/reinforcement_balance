import pytest

from survivors.action_semantics import ActionContract, load_action_contract


def test_canonical_nine_actions_and_cadence():
    contract = load_action_contract()
    assert [r.name for r in contract.rows] == ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "idle"]
    assert len({r.sim_vector for r in contract.rows}) == 9
    assert len({r.screen_direction for r in contract.rows}) == 9
    assert len({r.wasd_chord for r in contract.rows}) == 9
    assert contract.physics_hz == 60 and contract.frame_skip == 4 and contract.decision_hz == 15
    assert contract.rows[2].sim_vector[0] > 0 and contract.rows[2].screen_direction == "right"
    assert contract.rows[6].sim_vector[0] < 0 and contract.rows[6].screen_direction == "left"


def test_cadence_and_golden_parity_are_blocking():
    data = load_action_contract().to_wire()
    data["decision_hz"] = 30
    with pytest.raises(ValueError):
        ActionContract.from_wire(data)
    data = load_action_contract().to_wire()
    data["rows"][2]["screen_direction"] = "left"
    with pytest.raises(ValueError):
        ActionContract.from_wire(data)
