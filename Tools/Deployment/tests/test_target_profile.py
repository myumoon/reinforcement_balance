import pytest

from survivors.target_profile import TargetProfile, SuccessObservation, load_target_profile
from reinbalance_survivors_contracts.canonical_json import canonical_hash


def test_profile_identity_and_unknown_fields():
    profile = load_target_profile()
    assert profile.target_hash == TargetProfile.from_wire(profile.to_wire()).target_hash
    data = profile.to_wire(); data["unknown"] = 1
    with pytest.raises(ValueError): TargetProfile.from_wire(data)


def test_success_requires_run_bound_post_30_observation(tmp_path):
    profile = load_target_profile()
    assert profile.success_state(1799, None) == "RUNNING"
    assert profile.success_state(1800, None) == "TARGET_REACHED_PENDING_TRANSITION"
    assert profile.success_state(1800, True) == "TARGET_REACHED_PENDING_TRANSITION"
    event={"run_id":"run-1","event":"result_screen","observed_at_seconds":1801}
    evidence=tmp_path/"post30-evidence.yaml"; evidence.write_text("run_id: run-1\nevent: result_screen\nobserved_at_seconds: 1801\n")
    evidence_hash=canonical_hash({"bytes_hex":evidence.read_bytes().hex()})
    telemetry=tmp_path/"post30.yaml"; telemetry.write_text(f"run_id: run-1\nevent: result_screen\nobserved_at_seconds: 1801\nevidence_hash: {evidence_hash}\n")
    observation=SuccessObservation.from_telemetry("run-1",telemetry,evidence)
    assert profile.success_state(1800, observation) == "TARGET_REACHED_PENDING_TRANSITION"
    assert profile.success_state(1800, observation,"run-2") == "TARGET_REACHED_PENDING_TRANSITION"
    assert profile.success_state(1800, observation,"run-1") == "TARGET_REACHED_CONFIRMED"
    telemetry.write_text(telemetry.read_text().replace("1801","1802"))
    with pytest.raises(ValueError): SuccessObservation.from_telemetry("run-1",telemetry,evidence)
    with pytest.raises(ValueError): SuccessObservation.from_telemetry("run-1",telemetry)

@pytest.mark.parametrize(("section","field","value"),[("choice_taxonomy","states",["future"]),("choice_taxonomy","level_up_card_counts",[5]),("progression","reroll_count",-1),("input","physics_hz",30)])
def test_nested_taxonomy_is_closed(section,field,value):
    data=load_target_profile().to_wire(); data[section][field]=value
    with pytest.raises(ValueError): TargetProfile.from_wire(data)

@pytest.mark.parametrize(("field","value"),[("unlocked_items",["future_item"]),("collection_pool",["whip","future_item"]),("purchased_power_ups",["might"]),("unlocked_characters",[1])])
def test_progression_item_vocabulary_is_closed(field,value):
    data=load_target_profile().to_wire(); data["progression"][field]=value
    with pytest.raises(ValueError): TargetProfile.from_wire(data)
