import pytest

from survivors.target_profile import TargetProfile, load_target_profile


def test_profile_identity_and_unknown_fields():
    profile = load_target_profile()
    assert profile.target_hash == TargetProfile.from_wire(profile.to_wire()).target_hash
    data = profile.to_wire(); data["unknown"] = 1
    with pytest.raises(ValueError): TargetProfile.from_wire(data)


def test_success_requires_post_30_transition():
    profile = load_target_profile()
    assert profile.success_state(1799, None) == "RUNNING"
    assert profile.success_state(1800, None) == "TARGET_REACHED_PENDING_TRANSITION"
    assert profile.success_state(1800, True) == "TARGET_REACHED_PENDING_TRANSITION"
    assert profile.success_state(1800, {"event":"result_screen","observed_at_seconds":1801}) == "TARGET_REACHED_CONFIRMED"

@pytest.mark.parametrize(("section","field","value"),[("choice_taxonomy","states",["future"]),("choice_taxonomy","level_up_card_counts",[5]),("progression","reroll_count",-1),("input","physics_hz",30)])
def test_nested_taxonomy_is_closed(section,field,value):
    data=load_target_profile().to_wire(); data[section][field]=value
    with pytest.raises(ValueError): TargetProfile.from_wire(data)
