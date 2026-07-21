import pytest

from survivors.target_profile import TargetProfile, load_target_profile


def test_profile_identity_and_unknown_fields():
    profile = load_target_profile()
    assert profile.target_hash == TargetProfile.from_wire(profile.to_wire()).target_hash
    data = profile.to_wire(); data["unknown"] = 1
    with pytest.raises(ValueError): TargetProfile.from_wire(data)


def test_success_requires_post_30_transition():
    profile = load_target_profile()
    assert profile.success_state(1799, False) == "RUNNING"
    assert profile.success_state(1800, False) == "TARGET_REACHED_PENDING_TRANSITION"
    assert profile.success_state(1800, True) == "TARGET_REACHED_CONFIRMED"
