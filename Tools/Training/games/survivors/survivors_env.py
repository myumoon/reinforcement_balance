"""UE5 Survivors ゲーム gymnasium 環境ラッパー。"""

from typing import Callable

import numpy as np
import gymnasium as gym
from base.base_ue5_env import BaseUE5Env

_NUM_ACTIONS = 5


class SurvivorsEnv(BaseUE5Env):
    """UE5 Survivors ゲームの gymnasium ラッパー。

    行動空間: Discrete(5) — 0=上(+Y), 1=下(-Y), 2=左(-X), 3=右(+X), 4=静止
    観測空間: UE5 の /obs_schema エンドポイントから自動取得する

    _reward_fn に reward_shaping(obs, prev_obs, base_reward) -> float を設定すると
    EUREKA 型報酬シェーピングが有効になる。None のときは base_reward をそのまま返す。
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8767, connect_timeout: int = 120,
                 shaping_weight: float = 1.0, frame_skip: int = 1):
        super().__init__(host=host, port=port, connect_timeout=connect_timeout, frame_skip=frame_skip)
        self.action_space = gym.spaces.Discrete(_NUM_ACTIONS)
        self._expected_schema_hash: str | None = None
        self._reward_fn: Callable | None = None
        self._prev_obs: np.ndarray | None = None
        self.shaping_weight = shaping_weight
        self._offsets: dict[str, int] = {}
        self._obs_schema: list[dict] = []

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
        )
        self._wait_for_server()

    def _on_server_connected(self):
        schema = self._get_json("/obs_schema", timeout=10, retries=3)

        total_dim = schema["total_dim"]
        self._expected_schema_hash = schema["obs_schema_hash"]
        self._obs_schema = schema["segments"]

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
        obs, base_reward, done, truncated, ue_info = super().step(action)

        # 永続 HP ダメージペナルティ（敵接触シグナルを強化）
        # HP は [0,1] に正規化済み（÷100）。1 HP ダメージ = -1.0（クリップ前）
        hp_i = self._offsets.get("player_hp", 12)
        hp_penalty = 0.0
        if self._prev_obs is not None:
            hp_delta = max(0.0, float(self._prev_obs[hp_i]) - float(obs[hp_i]))
            hp_penalty = float(np.clip(-hp_delta * 100.0, -1.0, 0.0))

        shaped = 0.0
        if self._reward_fn is not None:
            try:
                shaped = float(self._reward_fn(obs, self._prev_obs, base_reward))
                shaped = float(np.clip(shaped, -1.0, 1.0)) * self.shaping_weight
            except Exception as e:
                print(f"[WARN] reward_fn エラー: {e}")
                shaped = 0.0

        info = dict(ue_info)
        info.update({"base_reward": base_reward, "shaped_reward": shaped, "hp_penalty": hp_penalty})
        info.update(self._extract_training_metrics(obs))
        self._prev_obs = obs
        return obs, base_reward + shaped + hp_penalty, done, truncated, info

    def _extract_training_metrics(self, obs: np.ndarray) -> dict:
        def offset(name: str, default: int) -> int:
            return self._offsets.get(name, default)

        def nearest_distance(segment_name: str, default_offset: int, count: int) -> float | None:
            start = offset(segment_name, default_offset)
            best = None
            for i in range(count):
                dx = float(obs[start + i * 2]) * 30.0
                dy = float(obs[start + i * 2 + 1]) * 30.0
                dist = float(np.sqrt(dx * dx + dy * dy))
                if dist < 0.01 and i > 0:
                    continue
                if best is None or dist < best:
                    best = dist
            return best

        enemy_rel_i = offset("enemy_rel_pos", 63)
        contact_enemy_count = 0
        for i in range(20):
            dx = float(obs[enemy_rel_i + i * 2]) * 30.0
            dy = float(obs[enemy_rel_i + i * 2 + 1]) * 30.0
            dist = float(np.sqrt(dx * dx + dy * dy))
            if dist < 0.01 and i > 0:
                continue
            if dist < 0.7:
                contact_enemy_count += 1

        return {
            "player_hp": float(obs[offset("player_hp", 12)]),
            "xp_progress": float(obs[offset("xp_progress", 21)]),
            "observed_enemy_count": float(obs[offset("enemy_count", 20)]),
            "nearest_gem_distance": nearest_distance("gem_rel_pos", 23, 20),
            "nearest_enemy_distance": nearest_distance("enemy_rel_pos", 63, 20),
            "contact_enemy_count": contact_enemy_count,
        }

    def set_params(self, **kwargs) -> bool:
        """カリキュラム用パラメータを /params エンドポイントで更新する。

        Args:
            MinActiveEnemies   (int):   毎ステップ即時維持する最小敵数
            MaxActiveEnemies   (int):   同時存在できる最大敵数
            EnemySpeedMult     (float): 敵速度の倍率
            SpawnRateMult      (float): スポーンレートの倍率（通常スポーン部分）
            MaxEnemyTypeId     (int):   スポーン可能な敵 TypeId の上限 (0-10)
            EnemyHPScale       (float): 敵HP倍率 (0.1-10.0, TimeScaling と乗算合成)
            EnemyDamageScale   (float): 敵接触ダメージ倍率 (0.1-10.0, TimeScaling と乗算合成)
            TimeScalingEnabled (bool):  時間経過による HP/ダメージ増加の有効化
        Returns:
            True if successful
        """
        try:
            self._post_json("/params", kwargs, timeout=5, retries=2)
            return True
        except Exception as e:
            print(f"[WARN] /params 更新失敗: {e}")
            return False

    def _action_to_payload(self, action) -> dict:
        return {"action": [float(int(action))]}
