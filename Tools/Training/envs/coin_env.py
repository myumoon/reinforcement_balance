"""UE5 コインゲーム gymnasium 環境ラッパー。"""

import numpy as np
import gymnasium as gym
from .base_ue5_env import BaseUE5Env

_NUM_ACTIONS = 5

# 観測次元: 10 + NumCoinObs*2 + MaxEnemyObs*5
# デフォルト: NumCoinObs=3, MaxEnemyObs=20 → 116
def _calc_obs_dim(num_coin_obs: int = 3, max_enemy_obs: int = 20) -> int:
    return 10 + num_coin_obs * 2 + max_enemy_obs * 5


class CoinEnv(BaseUE5Env):
    """UE5 コイン収集 + 敵回避ゲームの gymnasium ラッパー。

    行動空間: Discrete(5) — 0=上(+Y), 1=下(-Y), 2=左(-X), 3=右(+X), 4=静止
    観測空間: Box(obs_dim,) — UE5側の NumCoinObs に合わせて指定する

    Args:
        num_coin_obs: UE5側 ACoinGame.NumCoinObs と一致させること（デフォルト 3）
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8766,
        connect_timeout: int = 120,
        num_coin_obs: int = 3,
    ):
        super().__init__(host=host, port=port, connect_timeout=connect_timeout)
        obs_dim = _calc_obs_dim(num_coin_obs)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(_NUM_ACTIONS)

    def _action_to_payload(self, action) -> dict:
        return {"action": [float(int(action))]}
