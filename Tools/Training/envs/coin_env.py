"""UE5 コインゲーム gymnasium 環境ラッパー。"""

import numpy as np
import gymnasium as gym
from .base_ue5_env import BaseUE5Env

_NUM_ACTIONS = 5


class CoinEnv(BaseUE5Env):
    """UE5 コイン収集 + 敵回避ゲームの gymnasium ラッパー。

    行動空間: Discrete(5) — 0=上(+Y), 1=下(-Y), 2=左(-X), 3=右(+X), 4=静止
    観測空間: UE5 の /obs_schema エンドポイントから自動取得する

    obs_schema_hash を /reset レスポンスで検証するため、
    UE5側で NumCoinObs を変更してモデルと不一致になった場合は即座にエラーになる。
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8766, connect_timeout: int = 120):
        super().__init__(host=host, port=port, connect_timeout=connect_timeout)
        self.action_space = gym.spaces.Discrete(_NUM_ACTIONS)
        self._expected_schema_hash: str | None = None

        # SB3 が env をラップする前に observation_space を確定させる必要があるため、
        # __init__ 時点でサーバーに接続して /obs_schema から正しい shape を取得する
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32  # 仮; 直後に上書き
        )
        self._wait_for_server()

    def _on_server_connected(self):
        """接続後に /obs_schema を取得して observation_space を確定する。"""
        resp = self.session.get(f"{self.base_url}/obs_schema", timeout=10)
        resp.raise_for_status()
        schema = resp.json()

        total_dim = schema["total_dim"]
        self._expected_schema_hash = schema["obs_schema_hash"]

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_dim,), dtype=np.float32
        )

        segments = [(s["name"], s["dim"]) for s in schema["segments"]]
        print(f"[INFO] obs_schema 取得完了: total_dim={total_dim}, hash={self._expected_schema_hash}")
        print(f"[INFO]   segments: {segments}")

    def _on_reset(self, data: dict):
        """reset レスポンスのハッシュを検証する。"""
        received_hash = data.get("obs_schema_hash", "")
        if self._expected_schema_hash and received_hash != self._expected_schema_hash:
            raise RuntimeError(
                f"obs_schema_hash が一致しません。UE5側の NumCoinObs が変更された可能性があります。\n"
                f"  期待値: {self._expected_schema_hash}\n"
                f"  受信値: {received_hash}\n"
                f"  → モデルを再訓練するか、NumCoinObs を元に戻してください。"
            )

    def _action_to_payload(self, action) -> dict:
        return {"action": [float(int(action))]}
