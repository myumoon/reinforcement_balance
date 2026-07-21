import pytest

from survivors.obs_observability import OBS_SEGMENTS, classify_segments, validate_release_observable, validate_exact_schema


def test_all_segments_are_classified_exactly_once():
    classified = classify_segments()
    assert set().union(*classified.values()) == OBS_SEGMENTS
    assert sum(map(len, classified.values())) == len(OBS_SEGMENTS)
    assert len(OBS_SEGMENTS) == 38
    validate_exact_schema(OBS_SEGMENTS)
    with pytest.raises(ValueError): validate_exact_schema(set(OBS_SEGMENTS)-{"player_hp"})


@pytest.mark.parametrize("segment", ["enemy_hp", "enemy_count", "enemy_density_near_16dir", "gem_density_all_16dir"])
def test_hidden_or_offscreen_state_is_not_release_observable(segment):
    with pytest.raises(ValueError): validate_release_observable({segment})
