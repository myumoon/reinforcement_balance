import pytest

from survivors.obs_observability import OBS_SEGMENTS, classify_segments, validate_release_observable


def test_all_segments_are_classified_exactly_once():
    classified = classify_segments()
    assert set().union(*classified.values()) == OBS_SEGMENTS
    assert sum(map(len, classified.values())) == len(OBS_SEGMENTS)


@pytest.mark.parametrize("segment", ["offscreen_entities", "hidden_enemy_hp", "hidden_cooldowns", "global_state_count", "global_density"])
def test_hidden_or_offscreen_state_is_not_release_observable(segment):
    with pytest.raises(ValueError): validate_release_observable({segment})
