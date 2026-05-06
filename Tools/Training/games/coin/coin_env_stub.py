"""コインゲームのスタブ環境（UE5 接続なし dry-run 用）。"""

import numpy as np
import gymnasium as gym

_OBS_DIM = 116
_NUM_ACTIONS = 5


class DummyCoinEnv(gym.Env):
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
        return self.observation_space.sample(), {}

    def step(self, action):
        self._step_count += 1
        obs = self.observation_space.sample()
        reward = float(self.np_random.integers(0, 2))
        done = self._step_count >= 500
        return obs, reward, done, False, {}

    def close(self):
        super().close()
