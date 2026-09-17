"""combat session の recurrent state・action parity・入力境界を検証する。

やさしい説明: 03-05 の保存済み GRU actor を直接進めた参照結果と session の結果を比べ、
episode 境界で `[1, 1, hidden]` の記憶と episode_start が必ず一緒に戻ることを確かめます。
"""
from __future__ import annotations

import numpy as np
import pytest
import torch as th

from reinbalance_survivors_contracts.target_action import ActionSemantics
from survivors.runtime.artifact_bundle import CombatPolicy
from survivors.runtime.combat_session import CombatDecision, CombatSession, StaleSnapshotError

from . import _runtime_fixtures as fx


def _obs_sequence(length: int = 6, *, seed: int = 7) -> list[np.ndarray]:
    """決定的な観測列を返す。

    やさしい説明: action/state parity の期待値を毎回同じ入力から作れるようにします。
    """
    rng = np.random.default_rng(seed)
    return [rng.normal(size=fx.DEFAULT_OBSERVATION_DIM).astype(np.float32) for _ in range(length)]


def _offline_reference(
    policy: CombatPolicy, observations: list[np.ndarray]
) -> tuple[list[int], list[np.ndarray]]:
    """保存 actor を session 外で直接進めた action と state を返す。

    やさしい説明: runtime wrapper を使わない独立経路を正解とし、再帰状態の更新ずれを検出します。
    """
    hidden = policy.initial_hidden_state()
    actions: list[int] = []
    states: list[np.ndarray] = []
    with th.inference_mode():
        for obs in observations:
            logits, _, hidden = policy.model.step(th.from_numpy(obs).reshape(1, -1), hidden)
            actions.append(int(th.argmax(logits, dim=1).item()))
            states.append(hidden.detach().cpu().numpy()[None, ...].copy())
    return actions, states


class TestRecurrentActorParity:
    """保存 actor と session の action/state parity を検証する。

    やさしい説明: package から復元された同じ重みなら逐次決定が一切 drift しないことを見ます。
    """

    def test_actions_and_states_match_offline_actor(self, golden_combat_policy: CombatPolicy) -> None:
        """known observation sequence の全 action/state が一致する。

        やさしい説明: action だけ偶然一致する退化を避けるため、各 tick の内部記憶も比較します。
        """
        observations = _obs_sequence()
        expected_actions, expected_states = _offline_reference(golden_combat_policy, observations)
        session = CombatSession(golden_combat_policy)
        actual_actions: list[int] = []
        actual_states: list[np.ndarray] = []
        for obs in observations:
            actual_actions.append(session.decide(obs).action_index)
            state = session.recurrent_state_copy()
            assert state is not None
            actual_states.append(state)
        assert actual_actions == expected_actions
        for actual, expected in zip(actual_states, expected_states, strict=True):
            np.testing.assert_array_equal(actual, expected)

    def test_state_shape_is_layers_batch_hidden(self, golden_combat_policy: CombatPolicy) -> None:
        """actor state は `[n_layers, 1, hidden]` を保つ。

        やさしい説明: batch 軸や layer 軸を取り違えた状態を次 tick へ渡さないことを確認します。
        """
        session = CombatSession(golden_combat_policy)
        session.decide(_obs_sequence(1)[0])
        state = session.recurrent_state_copy()
        assert state is not None
        assert state.shape == (1, 1, golden_combat_policy.hidden_dim)

    def test_action_semantics_map_all_nine_indices(self, golden_combat_policy: CombatPolicy) -> None:
        """9 action index が共有 semantic へ一意に写る。

        やさしい説明: 方向 action の index と共有 action map の対応を session 境界で固定します。
        """
        semantics = ActionSemantics.default_v1().actions
        assert semantics == (
            "move_dx-1_dy-1", "move_dx0_dy-1", "move_dx1_dy-1",
            "move_dx-1_dy0", "move_dx0_dy0", "move_dx1_dy0",
            "move_dx-1_dy1", "move_dx0_dy1", "move_dx1_dy1",
        )
        session = CombatSession(golden_combat_policy)
        assert tuple(session.semantic_for(index) for index in range(9)) == semantics
        assert len(set(semantics)) == 9


class TestEpisodeReset:
    """episode 境界で state と episode_start を対称に戻す。

    やさしい説明: death/result/unknown gap/new run の入口が呼ぶ一つの reset を検証します。
    """

    @pytest.mark.parametrize("trigger", ["episode_reset", "unknown_gap", "death", "result", "new_run"])
    def test_reset_trigger_clears_state_and_sets_episode_start(
        self, golden_combat_policy: CombatPolicy, trigger: str
    ) -> None:
        """全 reset trigger で state と flag が同時に初期化される。

        やさしい説明: trigger 名は呼び出し側の経路一覧で、session 内では同じ原子的 reset を使います。
        """
        session = CombatSession(golden_combat_policy)
        session.decide(_obs_sequence(1)[0])
        session.reset_episode(trigger)
        assert session.recurrent_state_copy() is None
        assert session.episode_start_pending is True

    def test_reset_restores_first_action_and_state(self, golden_combat_policy: CombatPolicy) -> None:
        """reset 後の同じ観測は最初と同じ結果になる。

        やさしい説明: 前 episode の隠れ状態が新 run へ漏れていないことを出力から確認します。
        """
        obs = _obs_sequence(1)[0]
        session = CombatSession(golden_combat_policy)
        first = session.decide(obs)
        first_state = session.recurrent_state_copy()
        session.decide(obs)
        session.reset_episode("new_run")
        again = session.decide(obs)
        assert again.action_index == first.action_index
        np.testing.assert_array_equal(session.recurrent_state_copy(), first_state)


class TestInvalidObservations:
    """不正 observation と policy 出力を fail-closed にする。

    やさしい説明: shape・NaN・型違いを actor へ渡さず、呼び出し側が安全決定へ落とせる例外にします。
    """

    @pytest.mark.parametrize("obs", [np.zeros(3, dtype=np.float32), np.full(fx.DEFAULT_OBSERVATION_DIM, np.nan, dtype=np.float32)])
    def test_invalid_observation_raises(self, golden_combat_policy: CombatPolicy, obs: np.ndarray) -> None:
        """wrong shape と non-finite 値を拒否する。

        やさしい説明: actor state を進める前に入力境界で止めます。
        """
        with pytest.raises(StaleSnapshotError):
            CombatSession(golden_combat_policy).decide(obs)

    def test_non_policy_rejected(self) -> None:
        """検証済み CombatPolicy 以外を拒否する。

        やさしい説明: package 検証を迂回した任意モデルを session へ差し込ませません。
        """
        with pytest.raises(ValueError, match="CombatPolicy"):
            CombatSession(object())  # type: ignore[arg-type]

    def test_decision_is_finite_and_in_range(self, golden_combat_policy: CombatPolicy) -> None:
        """通常推論は有限 confidence と範囲内 action を返す。

        やさしい説明: downstream が追加の action 補正をせず使える最小出力契約を確認します。
        """
        decision = CombatSession(golden_combat_policy).decide(_obs_sequence(1)[0])
        assert isinstance(decision, CombatDecision)
        assert 0 <= decision.action_index < 9
        assert np.isfinite(decision.confidence) and 0.0 <= decision.confidence <= 1.0
