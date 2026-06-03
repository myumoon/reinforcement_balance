"""UE5 Survivors ゲーム gymnasium 環境ラッパー。"""

from typing import Callable

import numpy as np
import gymnasium as gym
from base.base_ue5_env import BaseUE5Env

_NUM_ACTIONS = 9


class SurvivorsEnv(BaseUE5Env):
    """UE5 Survivors ゲームの gymnasium ラッパー。

    行動空間: Discrete(9) — 0=北(+Y), 1=北東, 2=東(+X), 3=南東, 4=南(-Y),
                             5=南西, 6=西(-X), 7=北西, 8=静止
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
        self._last_params: dict = {}
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
        # HP は [0,1] に正規化済み。1 HP ダメージ = -1.0（クリップ前）
        # schema ベースのオフセット（player_hp が常に存在するセグメント）
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

        def schema_dim(name: str, default: int) -> int:
            """セグメント名から次元数を取得する。obs_schema が利用可能な場合に使用。"""
            for seg in self._obs_schema:
                if seg["name"] == name:
                    return seg["dim"]
            return default

        def nearest_distance(segment_name: str, default_offset: int, default_count: int) -> float | None:
            """セグメント名ベースで最近傍距離を計算する（obs インデックスハードコードを廃止）。"""
            start = offset(segment_name, default_offset)
            # dim から entity 数を計算（各エンティティは dx,dy の 2 次元）
            seg_dim = schema_dim(segment_name, default_count * 2)
            count = seg_dim // 2
            best = None
            for i in range(count):
                idx = start + i * 2
                if idx + 1 >= len(obs):
                    break
                dx = float(obs[idx]) * 30.0
                dy = float(obs[idx + 1]) * 30.0
                dist = float(np.sqrt(dx * dx + dy * dy))
                if dist < 0.01 and i > 0:
                    continue
                if best is None or dist < best:
                    best = dist
            return best

        # enemy_rel_pos のセグメント情報からエンティティ数を取得（新スキーマ対応）
        enemy_rel_i = offset("enemy_rel_pos", -1)
        enemy_count_from_schema = schema_dim("enemy_rel_pos", 64) // 2  # 新スキーマでは 32体
        contact_enemy_count = 0
        if enemy_rel_i >= 0:
            for i in range(enemy_count_from_schema):
                idx = enemy_rel_i + i * 2
                if idx + 1 >= len(obs):
                    break
                dx = float(obs[idx]) * 30.0
                dy = float(obs[idx + 1]) * 30.0
                dist = float(np.sqrt(dx * dx + dy * dy))
                if dist < 0.01 and i > 0:
                    continue
                if dist < 0.7:
                    contact_enemy_count += 1

        vel_i = offset("player_vel", 2)
        vx, vy = float(obs[vel_i]), float(obs[vel_i + 1])
        move_speed = float(np.sqrt(vx * vx + vy * vy))

        wall_ray_i = offset("wall_rays", 4)
        wall_min = float(np.min(obs[wall_ray_i:wall_ray_i + 8]))

        # 最近傍ジェム距離: 新スキーマでは red_gem_rel_pos / green_gem_rel_pos / blue_gem_rel_pos に分割
        # 後方互換: 旧スキーマの gem_rel_pos も参照する
        nearest_gem = None
        for gem_seg in ("red_gem_rel_pos", "green_gem_rel_pos", "blue_gem_rel_pos", "gem_rel_pos"):
            d = nearest_distance(gem_seg, -1, 10)
            if d is not None and (nearest_gem is None or d < nearest_gem):
                nearest_gem = d

        return {
            "player_hp": float(obs[offset("player_hp", 12)]),
            "xp_progress": float(obs[offset("xp_progress", -1)]) if offset("xp_progress", -1) >= 0 else 0.0,
            "observed_enemy_count": float(obs[offset("enemy_count", -1)]) if offset("enemy_count", -1) >= 0 else 0.0,
            "nearest_gem_distance": nearest_gem,
            "nearest_enemy_distance": nearest_distance("enemy_rel_pos", -1, enemy_count_from_schema),
            "contact_enemy_count": contact_enemy_count,
            "move_speed": move_speed,
            "is_stationary": int(move_speed < 0.003),
            "is_wall_near": int(wall_min < 0.08),
        }

    def get_obs_schema(self) -> list:
        """SubprocVecEnv の env_method 経由での取得用。"""
        return self._obs_schema

    def get_offsets(self) -> dict:
        """SubprocVecEnv の env_method 経由での取得用。"""
        return self._offsets

    def get_obs_schema_hash(self) -> str:
        """全 env 間の obs_schema 一致確認用。"""
        return self._expected_schema_hash or ""

    def get_shaping_weight(self) -> float:
        """SubprocVecEnv / マルチenv の env_method 経由での取得用。"""
        return self.shaping_weight

    def set_shaping_weight(self, weight: float) -> None:
        """SubprocVecEnv / マルチenv の env_method 経由での設定用。"""
        self.shaping_weight = weight

    def clear_reward_fn(self) -> None:
        """SubprocVecEnv / マルチenv の env_method 経由でアニーリング完了時に無効化。"""
        self._reward_fn = None

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
            self._last_params.update(kwargs)
            return True
        except Exception as e:
            print(f"[WARN] /params 更新失敗: {e}")
            return False

    def get_params(self) -> dict:
        """最後に set_params で適用したパラメータを返す。eval_env との同期用。"""
        return dict(self._last_params)

    def _action_to_payload(self, action) -> dict:
        return {"action": [float(int(action))]}


from stable_baselines3.common.monitor import Monitor


class SurvivorsMonitor(Monitor):
    """Monitor + SurvivorsEnv 固有メソッドの明示フォワード。

    gymnasium の __getattr__ 経由のフォワードは v1.0 で廃止されるため
    必要なメソッドを明示定義して非推奨警告を抑制する。
    DummyVecEnv / SubprocVecEnv 両対応。
    """

    def set_params(self, **kwargs) -> bool:
        return self.env.set_params(**kwargs)

    def get_params(self) -> dict:
        return self.env.get_params()

    def set_shaping_weight(self, weight: float) -> None:
        self.env.set_shaping_weight(weight)

    def get_shaping_weight(self) -> float:
        return self.env.get_shaping_weight()

    def clear_reward_fn(self) -> None:
        self.env.clear_reward_fn()

    def get_obs_schema_hash(self) -> str:
        return self.env.get_obs_schema_hash()

    def get_offsets(self):
        return self.env.get_offsets()

    def get_obs_schema(self):
        return self.env.get_obs_schema()
