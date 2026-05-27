import sys
from pathlib import Path

_TRAINING_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(_TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAINING_ROOT))

from games.survivors.survivors_difficulty import (
    _PARAM_DIFFICULTY_MAX,
    _PARAM_DIFFICULTY_MIN,
    compute_difficulty_score,
)


def test_score_at_min_params_is_zero():
    score = compute_difficulty_score(dict(_PARAM_DIFFICULTY_MIN))
    assert abs(score) < 1e-6


def test_score_at_max_params_is_one():
    max_params = dict(_PARAM_DIFFICULTY_MAX)
    max_params["time_scaling"] = True
    score = compute_difficulty_score(max_params)
    assert abs(score - 1.0) < 1e-6


def test_score_beyond_max_exceeds_one():
    hard_params = dict(_PARAM_DIFFICULTY_MAX)
    hard_params["max_enemies"] = 300
    hard_params["enemy_hp_scale"] = 4.0
    score = compute_difficulty_score(hard_params)
    assert score > 1.0


def test_score_monotone_max_enemies():
    base = dict(_PARAM_DIFFICULTY_MIN)
    base["max_enemies"] = 6
    score_low = compute_difficulty_score(base)
    base["max_enemies"] = 100
    score_high = compute_difficulty_score(base)
    assert score_high > score_low


def test_bool_time_scaling_accepted():
    params = dict(_PARAM_DIFFICULTY_MIN)
    params["time_scaling"] = True
    score = compute_difficulty_score(params)
    assert score > 0.0


def test_missing_key_uses_min():
    score = compute_difficulty_score({})
    assert abs(score) < 1e-6
