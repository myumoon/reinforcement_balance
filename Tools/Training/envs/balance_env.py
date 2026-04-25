"""UE5 BalancePole gymnasium 環境ラッパー。"""

import numpy as np
import gymnasium as gym
from .base_ue5_env import BaseUE5Env


class BalanceEnv(BaseUE5Env):
    """UE5 BalancePole の gymnasium ラッパー。

    行動空間: Box(-1, 1, shape=(1,)) — カートへの正規化力
    観測空間: Box(6,) — カート pos/vel + ポール角度/角速度 × 2
    """

    _OBS_LOW = np.array(
        [-4.8, -np.inf, -np.deg2rad(60), -np.inf, -np.deg2rad(60), -np.inf],
        dtype=np.float32,
    )
    _OBS_HIGH = np.array(
        [4.8, np.inf, np.deg2rad(60), np.inf, np.deg2rad(60), np.inf],
        dtype=np.float32,
    )

    def __init__(self, host: str = "127.0.0.1", port: int = 8765, connect_timeout: int = 120):
        super().__init__(host=host, port=port, connect_timeout=connect_timeout)
        self.observation_space = gym.spaces.Box(
            low=self._OBS_LOW, high=self._OBS_HIGH, dtype=np.float32
        )
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def _action_to_payload(self, action) -> dict:
        force = float(np.clip(action, -1.0, 1.0))
        return {"action": [force]}
