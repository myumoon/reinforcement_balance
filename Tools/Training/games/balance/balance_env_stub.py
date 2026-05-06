import numpy as np
import gymnasium as gym


class DummyBalanceEnv(gym.Env):
    """UE5 なしで SB3 訓練ループを検証するためのスタブ環境。
    ランダムな観測値を返し、物理シミュレーションは行わない。
    """

    metadata = {"render_modes": []}

    _OBS_LOW = np.array(
        [-4.8, -np.inf, -np.deg2rad(60), -np.inf, -np.deg2rad(60), -np.inf],
        dtype=np.float32,
    )
    _OBS_HIGH = np.array(
        [4.8, np.inf, np.deg2rad(60), np.inf, np.deg2rad(60), np.inf],
        dtype=np.float32,
    )

    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=self._OBS_LOW, high=self._OBS_HIGH, dtype=np.float32
        )
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self._step_count = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._step_count = 0
        obs = self.observation_space.sample() * 0.05
        return obs, {}

    def step(self, action):
        self._step_count += 1
        obs = self.observation_space.sample() * 0.05
        reward = 1.0
        done = self._step_count >= 200
        return obs, reward, done, False, {}
