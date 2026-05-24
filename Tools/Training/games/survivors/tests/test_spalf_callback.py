"""SpalfCallback のユニットテスト。"""
import math
import sys
from pathlib import Path
import numpy as np
import pytest

_TRAINING_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(_TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAINING_ROOT))

from games.survivors.spalf_callback import (
    SpalfCallback,
    _PARAM_BOUNDS,
    _PARAM_KEYS,
    _N_PARAMS,
    _PHASE0_PARAMS,
)


def _make_cb(**kwargs) -> SpalfCallback:
    defaults = dict(raw_env=None, frame_skip=4, alive_reward=0.001,
                    r_b=0.1, alpha=0.2, max_score=2250.0,
                    buffer_size=200, warmup_episodes=50)
    defaults.update(kwargs)
    return SpalfCallback(**defaults)


class TestParamsToVec:
    def test_phase0_in_range(self):
        cb = _make_cb()
        vec = cb._params_to_vec(_PHASE0_PARAMS)
        assert vec.shape == (_N_PARAMS,)
        assert np.all(vec >= 0.0) and np.all(vec <= 1.0)

    def test_roundtrip_phase0(self):
        cb = _make_cb()
        vec = cb._params_to_vec(_PHASE0_PARAMS)
        params = cb._vec_to_params(vec)
        assert params["min_enemies"] == _PHASE0_PARAMS["min_enemies"]
        assert params["max_enemies"] == _PHASE0_PARAMS["max_enemies"]
        assert abs(params["speed_mult"] - _PHASE0_PARAMS["speed_mult"]) < 0.01

    def test_max_params_vec_is_one(self):
        cb = _make_cb()
        max_params = {k: hi for k, (lo, hi) in _PARAM_BOUNDS.items()}
        vec = cb._params_to_vec(max_params)
        for i, key in enumerate(_PARAM_KEYS):
            if key == "time_scaling":
                assert vec[i] == pytest.approx(1.0, abs=0.01)
            else:
                assert vec[i] == pytest.approx(1.0, abs=0.01)

    def test_min_params_vec_is_zero(self):
        cb = _make_cb()
        min_params = {k: lo for k, (lo, hi) in _PARAM_BOUNDS.items()}
        min_params["time_scaling"] = False
        vec = cb._params_to_vec(min_params)
        assert np.allclose(vec, 0.0, atol=0.01)

    def test_max_enemies_constraint(self):
        cb = _make_cb()
        # min_enemies==max_enemies のとき max_enemies+1 に補正されること
        params = cb._vec_to_params(cb._params_to_vec({"min_enemies": 10, "max_enemies": 10,
                                                        "speed_mult": 1.0, "spawn_rate_mult": 1.0,
                                                        "max_enemy_type_id": 5, "enemy_hp_scale": 1.0,
                                                        "enemy_damage_scale": 1.0, "time_scaling": False}))
        assert params["max_enemies"] > params["min_enemies"]

    def test_time_scaling_bool(self):
        cb = _make_cb()
        p_true = {"time_scaling": True, "min_enemies": 4, "max_enemies": 6,
                  "speed_mult": 1.0, "spawn_rate_mult": 1.0, "max_enemy_type_id": 5,
                  "enemy_hp_scale": 1.0, "enemy_damage_scale": 1.0}
        p_false = dict(p_true, time_scaling=False)
        assert cb._vec_to_params(cb._params_to_vec(p_true))["time_scaling"] is True
        assert cb._vec_to_params(cb._params_to_vec(p_false))["time_scaling"] is False


class TestFAlpha:
    def test_zero_input(self):
        cb = _make_cb(alpha=0.2)
        assert cb._f_alpha(0.0) == pytest.approx(0.0, abs=1e-9)

    def test_positive_input_negative_output(self):
        cb = _make_cb(alpha=0.2)
        # f_α(x) = -α(1 - exp(x/α)): x > 0 → f > 0
        val = cb._f_alpha(0.2)
        assert val > 0.0

    def test_large_input_clipped(self):
        cb = _make_cb(alpha=0.2)
        # x/α = 500 のとき overflow しないこと
        val = cb._f_alpha(100.0)
        assert math.isfinite(val)

    def test_large_negative_input_clipped(self):
        cb = _make_cb(alpha=0.2)
        val = cb._f_alpha(-100.0)
        assert math.isfinite(val)

    def test_monotone_increasing(self):
        cb = _make_cb(alpha=0.2)
        xs = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]
        vals = [cb._f_alpha(x) for x in xs]
        assert all(a <= b for a, b in zip(vals, vals[1:]))


class TestComputeAlp:
    def test_alp_nonneg_spalf_mode(self):
        cb = _make_cb(r_b=0.1, alpha=0.2)
        cb._use_spalf_mode = True
        alp = cb._compute_alp(0.5, 0.0)
        assert alp >= 0.0

    def test_alp_nonneg_vanilla_mode(self):
        cb = _make_cb()
        cb._use_spalf_mode = False
        alp = cb._compute_alp(0.3, 0.1)
        assert alp == pytest.approx(abs(0.3 - 0.1), abs=1e-9)

    def test_alp_for_episode_no_history(self):
        cb = _make_cb()
        vec = np.zeros(_N_PARAMS, dtype=np.float32)
        alp = cb._compute_alp_for_episode(0.5, vec)
        assert alp >= 0.0


class TestExportImportState:
    def test_roundtrip_empty(self):
        cb = _make_cb()
        state = cb.export_state()
        cb2 = _make_cb()
        cb2.import_state(state)
        assert cb2._total_episodes == 0

    def test_roundtrip_with_data(self):
        cb = _make_cb()
        # バッファに手動でデータを追加
        vec = np.full(_N_PARAMS, 0.5, dtype=np.float32)
        cb._reward_history.append((vec, 0.3))
        cb._alp_buffer.append((vec, 0.1))
        cb._recent_reward_buffer.append(0.3)
        cb._total_episodes = 10
        cb._current_params = dict(_PHASE0_PARAMS)
        cb._use_spalf_mode = True

        state = cb.export_state()
        cb2 = _make_cb()
        cb2.import_state(state)

        assert cb2._total_episodes == 10
        assert len(cb2._reward_history) == 1
        assert len(cb2._alp_buffer) == 1
        assert cb2._use_spalf_mode is True

    def test_current_params_preserved(self):
        cb = _make_cb()
        cb._current_params = {"min_enemies": 20, "max_enemies": 40,
                               "speed_mult": 1.1, "spawn_rate_mult": 2.0,
                               "max_enemy_type_id": 8, "enemy_hp_scale": 2.0,
                               "enemy_damage_scale": 1.5, "time_scaling": True}
        state = cb.export_state()
        cb2 = _make_cb()
        cb2.import_state(state)
        assert cb2._current_params["min_enemies"] == 20
        assert cb2._current_params["time_scaling"] is True


class TestSaveStatus:
    def test_saves_to_file(self, tmp_path):
        path = tmp_path / "log" / "spalf_state.json"
        cb = _make_cb(status_path=str(path))
        cb._save_status()
        import json
        data = json.loads(path.read_text())
        assert "total_episodes" in data
        assert "current_params" in data

    def test_no_error_when_path_none(self):
        cb = _make_cb(status_path=None)
        cb._save_status()  # エラーにならないこと
