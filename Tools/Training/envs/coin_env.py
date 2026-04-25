"""UE5 コインゲーム gymnasium 環境ラッパー。"""

import numpy as np
import gymnasium as gym
from .base_ue5_env import BaseUE5Env

_OBS_DIM = 116
_NUM_ACTIONS = 5


class CoinEnv(BaseUE5Env):
    """UE5 コイン収集 + 敵回避ゲームの gymnasium ラッパー。

    行動空間: Discrete(5) — 0=上(+Y), 1=下(-Y), 2=左(-X), 3=右(+X), 4=静止
    観測空間: Box(116,) — プレイヤー pos/vel, 壁距離, 敵情報等
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8766, connect_timeout: int = 120):
        super().__init__(host=host, port=port, connect_timeout=connect_timeout)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(_OBS_DIM,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(_NUM_ACTIONS)

    def _action_to_payload(self, action) -> dict:
        return {"action": [float(int(action))]}
