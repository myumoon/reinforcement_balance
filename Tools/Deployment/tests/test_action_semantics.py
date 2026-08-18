"""action 仕様(9 アクション変位・C++ 照合・golden fixture)の検証テスト。

9 種類の操作の向き・大きさがゲーム本体と一致し、正解表とズレないかを自動テストで確認します。
"""

import pytest

from survivors.action_semantics import ActionContract, load_action_contract, validate_cpp_source, validate_golden_fixture


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
    data = load_action_contract().to_wire(); data["source_descriptor"]["physics_dt"]="1/30"
    with pytest.raises(ValueError): ActionContract.from_wire(data)

def test_measured_fixture_east_west_drift_is_blocking(tmp_path):
    import json
    contract=load_action_contract(); fixture={"schema_version":"survivors_action_telemetry.v1","measurement":"dry run","samples":[]}
    path=tmp_path/"fixture.json"; path.write_text(json.dumps(fixture))
    with pytest.raises(ValueError):validate_golden_fixture(contract,path)
    real=__import__("pathlib").Path(__file__).parents[1]/"configs"/"golden"/"action_displacement_wasd_v1.json"
    forged=json.loads(real.read_text()); forged["attestation"]["capture_evidence_hash"]="0"*64
    path.write_text(json.dumps(forged))
    with pytest.raises(ValueError): validate_golden_fixture(contract,path)
    data = load_action_contract().to_wire()
    data["rows"][2]["screen_direction"] = "left"
    with pytest.raises(ValueError):
        ActionContract.from_wire(data)

def test_cpp_apply_action_is_parsed_and_drift_is_blocking(tmp_path):
    contract=load_action_contract()
    source=contract.to_wire()["source_descriptor"]["path"]
    from pathlib import Path
    real=Path(__file__).parents[3]/source
    validate_cpp_source(contract,real)
    drift=tmp_path/"SurvivorsGameLogic.cpp"
    source_text=real.read_text(encoding="utf-8")
    drift.write_text(source_text.replace("case 2: MoveDir=FVector2D(1.f,0.f)","case 2: MoveDir=FVector2D(-1.f,0.f)"),encoding="utf-8")
    with pytest.raises(ValueError): validate_cpp_source(contract,drift)
    drift.write_text(source_text.replace("case 2: MoveDir=FVector2D(1.f,0.f)","case 2: MoveDir=FVector2D(1.5f,0.f)"),encoding="utf-8")
    with pytest.raises(ValueError): validate_cpp_source(contract,drift)
