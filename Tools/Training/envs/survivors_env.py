"""UE5 Survivors ゲーム gymnasium 環境ラッパー。"""

from typing import Callable

import numpy as np
import gymnasium as gym
import requests
from .base_ue5_env import BaseUE5Env

_NUM_ACTIONS = 5


class SurvivorsEnv(BaseUE5Env):
    """UE5 Survivors ゲームの gymnasium ラッパー。

    行動空間: Discrete(5) — 0=上(+Y), 1=下(-Y), 2=左(-X), 3=右(+X), 4=静止
    観測空間: UE5 の /obs_schema エンドポイントから自動取得する

    _reward_fn に reward_shaping(obs, prev_obs, base_reward) -> float を設定すると
    EUREKA 型報酬シェーピングが有効になる。None のときは base_reward をそのまま返す。
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8767, connect_timeout: int = 120,
                 shaping_weight: float = 1.0):
        super().__init__(host=host, port=port, connect_timeout=connect_timeout)
        self.action_space = gym.spaces.Discrete(_NUM_ACTIONS)
        self._expected_schema_hash: str | None = None
        self._reward_fn: Callable | None = None
        self._prev_obs: np.ndarray | None = None
        self.shaping_weight = shaping_weight
        self._offsets: dict[str, int] = {}

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
        )
        self._wait_for_server()

    def _on_server_connected(self):
        resp = self.session.get(f"{self.base_url}/obs_schema", timeout=10)
        resp.raise_for_status()
        schema = resp.json()

        total_dim = schema["total_dim"]
        self._expected_schema_hash = schema["obs_schema_hash"]

        offset = 0
        for seg in schema["segments"]:
            self._offsets[seg["name"]] = offset
            offset += seg["dim"]

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_dim,), dtype=np.float32
        )
        segments = [(s["name"], s["dim"]) for s in schema["segments"]]
        print(f"[INFO] obs_schema 取得完了: total_dim={total_dim}, hash={self._expected_schema_hash}")
        print(f"[INFO]   segments: {segments}")

    def _on_reset(self, data: dict):
        received_hash = data.get("obs_schema_hash", "")
        if self._expected_schema_hash and received_hash != self._expected_schema_hash:
            raise RuntimeError(
                f"obs_schema_hash が一致しません。UE5 側が変更された可能性があります。\n"
                f"  期待値: {self._expected_schema_hash}\n"
                f"  受信値: {received_hash}"
            )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = super().reset(seed=seed, options=options)
        self._prev_obs = obs
        return obs, info

    def step(self, action):
        obs, base_reward, done, truncated, _ = super().step(action)

        shaped = 0.0
        if self._reward_fn is not None:
            try:
                shaped = float(self._reward_fn(obs, self._prev_obs, base_reward))
                shaped = float(np.clip(shaped, -1.0, 1.0)) * self.shaping_weight
            except Exception as e:
                print(f"[WARN] reward_fn エラー: {e}")
                shaped = 0.0

        info = {"base_reward": base_reward, "shaped_reward": shaped}
        self._prev_obs = obs
        return obs, base_reward + shaped, done, truncated, info

    def set_params(self, **kwargs) -> bool:
        """カリキュラム用パラメータを /params エンドポイントで更新する。

        Args:
            MaxActiveEnemies (int): 同時存在できる最大敵数
            EnemySpeedMult   (float): 敵速度の倍率
            SpawnInterval    (float): 敵スポーン間隔 [秒]
        Returns:
            True if successful
        """
        try:
            resp = self.session.post(f"{self.base_url}/params", json=kwargs, timeout=5)
            resp.raise_for_status()
            return True
        except Exception as e:
            print(f"[WARN] /params 更新失敗: {e}")
            return False

    def _action_to_payload(self, action) -> dict:
        return {"action": [float(int(action))]}
