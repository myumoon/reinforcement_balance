"""state_modules のユニットテスト。"""
import sys
from pathlib import Path
import numpy as np
import pytest

_TRAINING_ROOT = Path(__file__).resolve().parent.parent
if str(_TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAINING_ROOT))

from games.survivors.state_modules import EpisodeScoreTracker, SpalfStateModule, _PHASE0_PARAMS


class TestEpisodeScoreTracker:
    def test_basic_single_env(self):
        tracker = EpisodeScoreTracker(frame_skip=4, alive_reward=0.001)
        tracker.reset(1)
        infos = [{"base_reward": 10.0, "episode": {"l": 100, "r": 10.0}}]
        results = tracker.process(infos)
        assert len(results) == 1
        env_idx, active_score, ep_len = results[0]
        assert env_idx == 0
        assert ep_len == 100
        alive_total = 0.001 * 4 * 100
        expected = max(0.0, 10.0 - alive_total)
        assert abs(active_score - expected) < 1e-6

    def test_n_envs(self):
        tracker = EpisodeScoreTracker(frame_skip=1, alive_reward=0.0)
        tracker.reset(3)
        infos = [
            {"base_reward": 5.0, "episode": {"l": 50, "r": 5.0}},
            {"base_reward": 3.0},  # episode not done
            {"base_reward": 8.0, "episode": {"l": 80, "r": 8.0}},
        ]
        results = tracker.process(infos)
        assert len(results) == 2
        assert results[0][0] == 0
        assert results[1][0] == 2
        assert abs(results[0][1] - 5.0) < 1e-6
        assert abs(results[1][1] - 8.0) < 1e-6

    def test_accumulation_across_steps(self):
        tracker = EpisodeScoreTracker(frame_skip=1, alive_reward=0.0)
        tracker.reset(1)
        tracker.process([{"base_reward": 3.0}])
        results = tracker.process([{"base_reward": 7.0, "episode": {"l": 2, "r": 10.0}}])
        assert len(results) == 1
        assert abs(results[0][1] - 10.0) < 1e-6


class TestSpalfStateModuleRoundtrip:
    def test_export_import_empty(self):
        m = SpalfStateModule(r_b=0.1, alpha=0.2, max_score=2250.0,
                              buffer_size=200, warmup_episodes=50)
        state = m.export_state()
        m2 = SpalfStateModule(r_b=0.1, alpha=0.2, max_score=2250.0,
                               buffer_size=200, warmup_episodes=50)
        m2.import_state(state)
        assert m2._total_episodes == 0
        assert m2._use_spalf_mode is True

    def test_export_import_with_data(self):
        m = SpalfStateModule(r_b=0.1, alpha=0.2, max_score=2250.0,
                              buffer_size=200, warmup_episodes=50)
        vec = np.full(8, 0.5, dtype=np.float32)
        m._reward_history.append((vec, 0.3))
        m._alp_buffer.append((vec, 0.1))
        m._recent_reward_buffer.append(0.3)
        m._total_episodes = 10
        state = m.export_state()
        m2 = SpalfStateModule(r_b=0.1, alpha=0.2, max_score=2250.0,
                               buffer_size=200, warmup_episodes=50)
        m2.import_state(state)
        assert m2._total_episodes == 10
        assert len(m2._reward_history) == 1
        assert len(m2._alp_buffer) == 1

    def test_params_to_vec_and_back(self):
        m = SpalfStateModule(r_b=0.1, alpha=0.2, max_score=2250.0,
                              buffer_size=200, warmup_episodes=50)
        vec = m.params_to_vec(_PHASE0_PARAMS)
        assert vec.shape == (8,)
        assert np.all(vec >= 0.0) and np.all(vec <= 1.0)
        params_back = m.vec_to_params(vec)
        assert params_back["min_enemies"] == _PHASE0_PARAMS["min_enemies"]
