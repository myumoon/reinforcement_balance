"""CombatSession: saved recurrent actor との parity と episode 境界を検証する。

known obs sequence に対して、offline RecurrentPPO `predict` ループと
CombatSession が同一の action 列と actor LSTM state を返すことを確認する。
episode reset / death gap で state が破棄されることも併せて検証する。
"""
from __future__ import annotations

import numpy as np
import pytest

from survivors.runtime.combat_session import (
    CombatDecision,
    CombatSession,
    StaleSnapshotError,
)

from . import _runtime_fixtures as fx


def _obs_sequence(length: int = 6, *, seed: int = 7) -> list[np.ndarray]:
    """決定的な観測列を返す。parity 比較の入力に使う。"""
    rng = np.random.default_rng(seed)
    return [
        rng.normal(size=fx.DEPLOY_OBS_DIM).astype(np.float32) for _ in range(length)
    ]


def _offline_reference(policy, observations):
    """SB3 RecurrentPPO を素で回した参照 action / state 列を返す。

    CombatSession を一切使わない独立経路。これと一致することが parity の定義。
    """
    model = policy.model
    vecnormalize = policy.vecnormalize
    state = None
    starts = np.array([True])
    actions: list[int] = []
    states: list[tuple[np.ndarray, np.ndarray]] = []
    for obs in observations:
        normalized = np.asarray(
            vecnormalize.normalize_obs(obs.reshape(1, -1)), dtype=np.float32
        )
        action, state = model.predict(
            normalized, state=state, episode_start=starts, deterministic=True
        )
        starts = np.array([False])
        actions.append(int(np.asarray(action).reshape(-1)[0]))
        states.append((np.asarray(state[0]).copy(), np.asarray(state[1]).copy()))
    return actions, states


class TestRecurrentActorParity:
    """[指摘9] saved recurrent actor と action / LSTM state が一致する。"""

    def test_actions_match_offline_predict(self, golden_combat_policy):
        observations = _obs_sequence()
        session = CombatSession(golden_combat_policy)
        session.reset_episode()
        actual = [session.decide(obs).action_index for obs in observations]
        expected, _ = _offline_reference(golden_combat_policy, observations)
        assert actual == expected

    def test_lstm_states_match_offline_predict(self, golden_combat_policy):
        observations = _obs_sequence()
        session = CombatSession(golden_combat_policy)
        session.reset_episode()
        actual_states = []
        for obs in observations:
            session.decide(obs)
            actual_states.append(session.lstm_state_copy())
        _, expected_states = _offline_reference(golden_combat_policy, observations)
        for actual, expected in zip(actual_states, expected_states):
            assert actual is not None
            np.testing.assert_array_equal(actual[0], expected[0])
            np.testing.assert_array_equal(actual[1], expected[1])

    def test_lstm_state_shape_is_layers_one_hidden(self, golden_combat_policy):
        """plan 指定の `[n_layers, 1, hidden]` 形状を保持する。"""
        session = CombatSession(golden_combat_policy)
        session.decide(_obs_sequence(1)[0])
        state = session.lstm_state_copy()
        assert state is not None
        expected = golden_combat_policy.lstm_state_shape
        assert state[0].shape == expected
        assert state[1].shape == expected

    def test_vecnormalize_is_applied(self, golden_combat_policy):
        """正規化前後で値が変わり、保存統計が実際に使われていること。"""
        session = CombatSession(golden_combat_policy)
        raw = np.full(fx.DEPLOY_OBS_DIM, 3.0, dtype=np.float32)
        normalized = session.normalize_observation(raw)
        assert not np.allclose(normalized.reshape(-1), raw)

    def test_decide_returns_combat_decision(self, golden_combat_policy):
        session = CombatSession(golden_combat_policy)
        decision = session.decide(_obs_sequence(1)[0])
        assert isinstance(decision, CombatDecision)
        assert 0 <= decision.action_index < golden_combat_policy.action_dim
        assert 0.0 <= decision.confidence <= 1.0


class TestEpisodeReset:
    """episode 境界で actor LSTM state が破棄される。"""

    def test_reset_clears_lstm_state(self, golden_combat_policy):
        session = CombatSession(golden_combat_policy)
        session.decide(_obs_sequence(1)[0])
        assert session.lstm_state_copy() is not None
        session.reset_episode()
        assert session.lstm_state_copy() is None
        assert session.episode_start_pending is True

    def test_reset_restores_first_action(self, golden_combat_policy):
        """reset 後は同じ観測に対して episode 先頭と同じ action を返す。"""
        observations = _obs_sequence(4)
        session = CombatSession(golden_combat_policy)
        session.reset_episode()
        first = session.decide(observations[0]).action_index
        for obs in observations[1:]:
            session.decide(obs)
        session.reset_episode()
        assert session.decide(observations[0]).action_index == first

    def test_episode_start_flag_resets_mid_sequence(self, golden_combat_policy):
        observations = _obs_sequence(3)
        session = CombatSession(golden_combat_policy)
        first = session.decide(observations[0], episode_start=True).action_index
        session.decide(observations[1])
        again = session.decide(observations[0], episode_start=True).action_index
        assert again == first

    def test_state_persists_without_reset(self, golden_combat_policy):
        """reset しなければ記憶が残り、同じ観測でも state が進む。"""
        observations = _obs_sequence(2)
        session = CombatSession(golden_combat_policy)
        session.decide(observations[0], episode_start=True)
        first_state = session.lstm_state_copy()
        session.decide(observations[0])
        second_state = session.lstm_state_copy()
        assert first_state is not None and second_state is not None
        assert not np.array_equal(first_state[0], second_state[0])


class TestInvalidObservations:
    """stale / invalid 観測は StaleSnapshotError にする。"""

    def test_wrong_shape_raises(self, golden_combat_policy):
        session = CombatSession(golden_combat_policy)
        with pytest.raises(StaleSnapshotError, match="obs shape"):
            session.decide(np.zeros(3, dtype=np.float32))

    def test_non_finite_raises(self, golden_combat_policy):
        session = CombatSession(golden_combat_policy)
        obs = np.zeros(fx.DEPLOY_OBS_DIM, dtype=np.float32)
        obs[0] = np.nan
        with pytest.raises(StaleSnapshotError, match="non-finite"):
            session.decide(obs)

    def test_infinite_value_raises(self, golden_combat_policy):
        session = CombatSession(golden_combat_policy)
        obs = np.zeros(fx.DEPLOY_OBS_DIM, dtype=np.float32)
        obs[1] = np.inf
        with pytest.raises(StaleSnapshotError, match="non-finite"):
            session.decide(obs)

    def test_non_policy_rejected(self):
        with pytest.raises(ValueError, match="RecurrentCombatPolicy"):
            CombatSession(object())  # type: ignore[arg-type]


class TestLongRunStability:
    """長時間の連続推論で NaN / 範囲外 action を出さない。"""

    def test_sustained_actions_stay_in_range_and_finite(self, golden_combat_policy):
        # 15 Hz × 30 分 = 27000 tick。CI 時間の都合で代表 1500 tick を回す。
        session = CombatSession(golden_combat_policy)
        session.reset_episode()
        rng = np.random.default_rng(11)
        for _ in range(1500):
            obs = rng.normal(size=fx.DEPLOY_OBS_DIM).astype(np.float32)
            decision = session.decide(obs)
            assert 0 <= decision.action_index < golden_combat_policy.action_dim
            assert np.isfinite(decision.confidence)
        state = session.lstm_state_copy()
        assert state is not None
        assert np.all(np.isfinite(state[0])) and np.all(np.isfinite(state[1]))
