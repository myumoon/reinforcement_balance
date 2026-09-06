"""RecurrentPPO combat session: actor LSTM 状態を episode 単位で管理する。

保存済み SB3 RecurrentPPO をそのまま推論に使い、`[n_layers, 1, hidden]` 形状の
actor LSTM state を episode 境界でだけ reset する。観測は deploy VecNormalize の
保存統計 (training=False / norm_reward=False) で訓練時と同じ条件へ正規化する。
Deployment 側で policy architecture を再定義しないため、saved recurrent actor と
action / state が drift しない。OS input には触れず action index だけを返す。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch as th

from .artifact_bundle import RecurrentCombatPolicy


class StaleSnapshotError(ValueError):
    """観測が stale または無効で inference を進められない場合の例外。

    invalid / unknown snapshot に対して no_op / stop を選ぶよう caller に通知する。
    """


@dataclass(frozen=True)
class CombatDecision:
    """combat session が 1 tick で返す action と確信度。

    action_index は 9 direction/idle の index、confidence は deterministic policy が
    その action へ割り当てた確率。telemetry と安全 gate の両方で使う。
    """

    action_index: int
    confidence: float


def _as_state_tuple(states: object) -> tuple[np.ndarray, np.ndarray]:
    """SB3 が返す LSTM state を (h, c) の numpy tuple へ正規化する。

    SB3 は tuple / RNNStates など複数表現を返し得るため、runtime 側で 1 つに揃える。
    """
    if states is None:
        raise StaleSnapshotError("recurrent policy returned no LSTM state")
    if isinstance(states, tuple) and len(states) == 2:
        hidden, cell = states
    else:  # pragma: no cover - SB3 の別表現に対する防御
        hidden = getattr(states, "pi", (None, None))[0]
        cell = getattr(states, "pi", (None, None))[1]
    if hidden is None or cell is None:
        raise StaleSnapshotError("recurrent policy LSTM state is incomplete")
    return (
        np.asarray(hidden, dtype=np.float32).copy(),
        np.asarray(cell, dtype=np.float32).copy(),
    )


class CombatSession:
    """episode 単位の actor LSTM 状態を保持する combat 推論セッション。

    deploy VecNormalize で観測を正規化してから RecurrentPPO へ渡す。
    episode 境界 (新 run / death / result / unknown gap) では必ず
    reset_episode() を呼び、前 episode の記憶を次 run へ持ち越さない。
    """

    def __init__(self, combat_policy: RecurrentCombatPolicy) -> None:
        """検証済み policy を受け取り LSTM state をゼロ初期化する。

        policy architecture は SB3 が所有する。ここでは推論条件だけを固定する。
        """
        if not isinstance(combat_policy, RecurrentCombatPolicy):
            raise ValueError("combat_policy must be RecurrentCombatPolicy")
        self._policy = combat_policy
        self._model = combat_policy.model
        self._vecnormalize = combat_policy.vecnormalize
        self._lstm_states: tuple[np.ndarray, np.ndarray] | None = None
        self._episode_start = True

    @property
    def observation_dim(self) -> int:
        """期待する観測 vector の次元を返す。"""
        return self._policy.observation_dim

    @property
    def action_dim(self) -> int:
        """action 空間の大きさを返す。"""
        return self._policy.action_dim

    @property
    def lstm_state_shape(self) -> tuple[int, int, int]:
        """actor LSTM state の shape `[n_layers, 1, hidden]` を返す。"""
        return self._policy.lstm_state_shape

    @property
    def episode_start_pending(self) -> bool:
        """次の decide() が episode 先頭として扱われるかを返す。"""
        return self._episode_start

    def reset_episode(self) -> None:
        """エピソード境界で actor LSTM 状態を破棄する。

        death / result / unknown gap / 新 run で必ず呼ぶ。
        呼ばなければ前 episode の記憶が次 episode に混入する。
        """
        self._lstm_states = None
        self._episode_start = True

    def normalize_observation(self, obs_vector: np.ndarray) -> np.ndarray:
        """deploy VecNormalize の保存統計で観測を正規化する。

        訓練時と同じ統計を使い、統計自体は更新しない (training=False)。
        """
        obs = np.asarray(obs_vector, dtype=np.float32)
        if obs.ndim != 1 or obs.shape[0] != self._policy.observation_dim:
            raise StaleSnapshotError(
                f"obs shape {obs.shape} != ({self._policy.observation_dim},)"
            )
        if not np.all(np.isfinite(obs)):
            raise StaleSnapshotError("combat observation contains non-finite values")
        normalized = np.asarray(
            self._vecnormalize.normalize_obs(obs.reshape(1, -1)), dtype=np.float32
        )
        if normalized.shape != (1, self._policy.observation_dim):
            raise StaleSnapshotError("deploy VecNormalize changed observation shape")
        if not np.all(np.isfinite(normalized)):
            raise StaleSnapshotError("normalized observation contains non-finite values")
        return normalized

    def decide(self, obs_vector: np.ndarray, *, episode_start: bool = False) -> CombatDecision:
        """観測 vector から action index と確信度を返す。

        episode_start=True の場合は LSTM 状態をリセットしてから推論する。
        観測または policy 出力が非有限なら StaleSnapshotError を送出する。
        """
        if episode_start:
            self.reset_episode()

        normalized = self.normalize_observation(obs_vector)
        starts = np.array([bool(self._episode_start)], dtype=bool)
        previous_states = self._lstm_states

        confidence = self._action_confidence(normalized, previous_states, starts)
        try:
            actions, new_states = self._model.predict(
                normalized,
                state=previous_states,
                episode_start=starts,
                deterministic=True,
            )
        except Exception as exc:  # noqa: BLE001  # SB3 は多様な例外型を送出する
            raise StaleSnapshotError(f"recurrent policy inference failed: {exc}") from exc

        self._lstm_states = _as_state_tuple(new_states)
        self._episode_start = False

        action_index = int(np.asarray(actions).reshape(-1)[0])
        if not 0 <= action_index < self._policy.action_dim:
            raise StaleSnapshotError("combat model returned out-of-range action index")
        return CombatDecision(action_index=action_index, confidence=confidence)

    def _action_confidence(
        self,
        normalized_obs: np.ndarray,
        lstm_states: tuple[np.ndarray, np.ndarray] | None,
        episode_starts: np.ndarray,
    ) -> float:
        """deterministic action へ policy が割り当てた確率を返す。

        LSTM state は進めない。predict() と同じ入力から分布だけを取り出すため、
        action / state の parity には影響しない。
        """
        policy = self._model.policy
        obs_tensor, _ = policy.obs_to_tensor(normalized_obs)
        if lstm_states is None:
            shape = self._policy.lstm_state_shape
            state_tensors = (
                th.zeros(shape, dtype=th.float32),
                th.zeros(shape, dtype=th.float32),
            )
        else:
            state_tensors = (
                th.as_tensor(lstm_states[0], dtype=th.float32),
                th.as_tensor(lstm_states[1], dtype=th.float32),
            )
        starts_tensor = th.as_tensor(episode_starts, dtype=th.float32)
        with th.no_grad():
            distribution, _ = policy.get_distribution(obs_tensor, state_tensors, starts_tensor)
            probabilities = distribution.distribution.probs
        probs = np.asarray(probabilities.detach().cpu().numpy(), dtype=np.float64).reshape(-1)
        if probs.size != self._policy.action_dim or not np.all(np.isfinite(probs)):
            raise StaleSnapshotError("combat policy produced non-finite action probabilities")
        return float(probs.max())

    def lstm_state_copy(self) -> tuple[np.ndarray, np.ndarray] | None:
        """現在の actor LSTM 状態のコピーを返す。

        shadow / telemetry / parity test 用。状態本体は変更しない。
        未推論 (episode 先頭) の場合は None を返す。
        """
        if self._lstm_states is None:
            return None
        return (self._lstm_states[0].copy(), self._lstm_states[1].copy())
