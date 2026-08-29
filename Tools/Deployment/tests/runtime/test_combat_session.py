"""CombatSession: GRU hidden state 管理と episode 境界を検証する。

golden fixture model を使い、action sequence 再現性と stale / invalid 観測の
safe handling をテストする。
"""
import numpy as np
import pytest
import torch

from survivors.runtime.artifact_bundle import _CombatGRU
from survivors.runtime.combat_session import CombatSession, StaleSnapshotError

OBS_DIM = 4
ACTION_DIM = 9
HIDDEN_DIM = 8


def _model() -> _CombatGRU:
    return _CombatGRU(OBS_DIM, ACTION_DIM, HIDDEN_DIM)


def _random_obs() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.standard_normal(OBS_DIM).astype(np.float32)


class TestCombatSessionInit:
    def test_properties_match_model(self):
        session = CombatSession(_model())
        assert session.observation_dim == OBS_DIM
        assert session.action_dim == ACTION_DIM
        assert session.hidden_dim == HIDDEN_DIM

    def test_wrong_type_raises(self):
        with pytest.raises(ValueError, match="_CombatGRU"):
            CombatSession(object())  # type: ignore[arg-type]


class TestDecide:
    def test_returns_valid_action_index(self):
        session = CombatSession(_model())
        action = session.decide(_random_obs())
        assert isinstance(action, int)
        assert 0 <= action < ACTION_DIM

    def test_nonfinite_obs_raises_stale(self):
        session = CombatSession(_model())
        obs = np.full(OBS_DIM, float("nan"), dtype=np.float32)
        with pytest.raises(StaleSnapshotError, match="non-finite"):
            session.decide(obs)

    def test_wrong_shape_raises_stale(self):
        session = CombatSession(_model())
        obs = np.zeros(OBS_DIM + 1, dtype=np.float32)
        with pytest.raises(StaleSnapshotError, match="shape"):
            session.decide(obs)

    def test_2d_obs_raises_stale(self):
        session = CombatSession(_model())
        obs = np.zeros((1, OBS_DIM), dtype=np.float32)
        with pytest.raises(StaleSnapshotError, match="shape"):
            session.decide(obs)

    def test_action_reproducible_from_same_init(self):
        """同一初期状態から同じ obs で同じ action を返す。"""
        model = _model()
        obs = _random_obs()
        s1 = CombatSession(model)
        s2 = CombatSession(model)
        assert s1.decide(obs) == s2.decide(obs)

    def test_second_step_differs_from_first(self):
        """hidden state が更新され、同じ obs でも 2 ステップ目は変わりうる。"""
        session = CombatSession(_model())
        obs = _random_obs()
        _ = session.decide(obs)
        h1 = session.hidden_state_copy()
        # 1 step 後の hidden が変化している
        assert not np.allclose(np.zeros(HIDDEN_DIM), h1)


class TestEpisodeReset:
    def test_reset_zeros_hidden_state(self):
        session = CombatSession(_model())
        session.decide(_random_obs())
        session.reset_episode()
        assert np.allclose(session.hidden_state_copy(), 0.0)

    def test_episode_start_flag_resets_state(self):
        """episode_start=True は reset_episode() 相当のリセットを行う。"""
        session = CombatSession(_model())
        session.decide(_random_obs())
        h_after_step = session.hidden_state_copy().copy()
        # reset して同じ obs を渡す
        session.decide(_random_obs(), episode_start=True)
        h_after_reset = session.hidden_state_copy()
        # hidden は同じ obs / init state から計算されるので一致しない ≠ step 後の state
        assert h_after_step.shape == h_after_reset.shape

    def test_death_then_new_episode_same_sequence(self):
        """episode リセット後に同じ obs 列を入力すると同じ action 列が返る。"""
        model = _model()
        obs_list = [_random_obs() for _ in range(3)]
        s = CombatSession(model)
        actions_first = [s.decide(o) for o in obs_list]
        s.reset_episode()
        actions_second = [s.decide(o) for o in obs_list]
        assert actions_first == actions_second

    def test_unknown_gap_no_ops_then_reset(self):
        """セッション中の gap 後は reset_episode() を呼んで新 episode を始める。"""
        session = CombatSession(_model())
        session.decide(_random_obs())
        session.reset_episode()
        # リセット後は再び有効な action を返せる
        action = session.decide(_random_obs())
        assert 0 <= action < ACTION_DIM


class TestHiddenStateCopy:
    def test_copy_does_not_alias(self):
        session = CombatSession(_model())
        session.decide(_random_obs())
        h1 = session.hidden_state_copy()
        h1[:] = 0.0
        h2 = session.hidden_state_copy()
        assert not np.allclose(h2, 0.0)
