"""Survivors ゲームのスタブ環境（UE5 接続なし dry-run 用）。"""

import numpy as np
import gymnasium as gym

_OBS_DIM = 183
_NUM_ACTIONS = 5


class DummySurvivorsEnv(gym.Env):
    """ランダム観測を返すスタブ。SB3 の訓練ループ疎通確認用。"""

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(_OBS_DIM,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(_NUM_ACTIONS)
        self._step_count = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        obs = np.zeros(_OBS_DIM, dtype=np.float32)
        obs[12] = 1.0  # player_hp = 1.0 (満タン)
        return obs, {}

    def step(self, action):
        self._step_count += 1
        obs = np.zeros(_OBS_DIM, dtype=np.float32)
        obs[12] = max(0.0, 1.0 - self._step_count / 1000.0)  # HP 線形減少
        reward = 0.005  # AliveReward
        done = self._step_count >= 500
        return obs, reward, done, False, {"base_reward": reward, "shaped_reward": 0.0}

    def set_params(self, **kwargs) -> bool:
        return True

    def close(self):
        super().close()
