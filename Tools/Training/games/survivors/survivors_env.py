"""UE5 Survivors ゲーム gymnasium 環境ラッパー。

通常 action と保留中 level-up choice を分離し、paired rollout 用 outcome metrics を返す。
"""

from collections.abc import Collection
from typing import Callable

import numpy as np
import gymnasium as gym
from base.base_ue5_env import BaseUE5Env
from games.survivors.choice_preview import (
    SurvivorsLevelUpPreview,
    parse_level_up_preview,
)
from games.survivors.choice_branch_rollout import BRANCH_RNG_SCHEMA_VERSION

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
        """HTTP 接続・schema offsets・action/observation spaces を初期化する。"""
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
        """live observation schema を取得し、segment offsets を固定する。"""
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
        """reset response の observation schema binding を検証する。"""
        received_hash = data.get("obs_schema_hash", "")
        if self._expected_schema_hash and received_hash != self._expected_schema_hash:
            raise RuntimeError(
                f"obs_schema_hash が一致しません。UE5 側が変更された可能性があります。\n"
                f"  期待値: {self._expected_schema_hash}\n"
                f"  受信値: {received_hash}"
            )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """episode を reset し、HP penalty 用の直前 observation を更新する。"""
        obs, info = super().reset(seed=seed, options=options)
        self._prev_obs = obs
        return obs, info

    def step(self, action):
        """通常 reward semantics を保ったまま rollout outcome info を追加する。"""
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
        info.update(
            self._extract_rollout_outcome_metrics(
                obs,
                ue_info,
                done=done,
                truncated=truncated,
            )
        )
        self._prev_obs = obs
        return obs, base_reward + shaped + hp_penalty, done, truncated, info

    def _extract_training_metrics(self, obs: np.ndarray) -> dict:
        """observation から既存 training diagnostics を読み出す。"""
        def offset(name: str, default: int) -> int:
            """segment name の offset を後方互換 default 付きで返す。"""
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
            if start < 0:
                # セグメントが obs_schema に存在しない場合はスキップ
                return None
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

        weapon_slot_i = offset("weapon_slots", -1)
        weapon_types: list[int] = []
        if weapon_slot_i >= 0:
            for s in range(6):
                idx = weapon_slot_i + s * 3
                if idx + 2 >= len(obs):
                    break
                tn = float(obs[idx])
                if tn > 1e-4:
                    wtype_id = int(round(tn * 64.0))
                    if wtype_id > 0:
                        weapon_types.append(wtype_id)

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
            "weapon_types": weapon_types,
        }

    def _extract_rollout_outcome_metrics(
        self,
        obs: np.ndarray,
        ue_info: dict,
        *,
        done: bool,
        truncated: bool,
    ) -> dict:
        """paired rollout ground truth 用の read-only step metrics を返す。

        新 producer の raw counters を優先し、旧 producer では observation と terminal flag から
        後方互換値を補う。reward の計算・返却値には触れない。
        """
        elapsed_offset = self._offsets.get("elapsed_time", -1)
        level_offset = self._offsets.get("player_level", -1)
        fallback_elapsed = (
            float(obs[elapsed_offset]) * 1800.0
            if 0 <= elapsed_offset < len(obs)
            else 0.0
        )
        fallback_level = (
            int(round(float(obs[level_offset]) * 100.0))
            if 0 <= level_offset < len(obs)
            else 0
        )
        alive = bool(ue_info.get("alive", not done))
        return {
            "elapsed": float(ue_info.get("elapsed", fallback_elapsed)),
            "level": int(ue_info.get("level", fallback_level)),
            "gems": int(ue_info.get("gems", 0)),
            "kills": int(ue_info.get("kills", 0)),
            "alive": alive,
            "stage_clear": bool(
                ue_info.get("stage_clear", truncated and alive)
            ),
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
            MinActiveEnemies    (int):   毎ステップ即時維持する最小敵数
            MaxActiveEnemies    (int):   同時存在できる最大敵数
            EnemySpeedMult      (float): 敵速度の倍率
            SpawnRateMult       (float): スポーンレートの倍率（通常スポーン部分）
            MaxEnemyTypeId      (int):   スポーン可能な敵 TypeId の上限 (0-10)
            EnemyHPScale        (float): 敵HP倍率 (0.1-10.0, TimeScaling と乗算合成)
            EnemyDamageScale    (float): 敵接触ダメージ倍率 (0.1-10.0, TimeScaling と乗算合成)
            TimeScalingEnabled  (bool):  時間経過による HP/ダメージ増加の有効化
            weapon_pool_mode    (str):   武器プールモード（"garlic_only" / "fixed_subset" /
                                         "weighted" / "all_base" / "all_with_evolutions"）
            allowed_weapon_types (list[int]): 使用可能な武器タイプのリスト（EWeaponType integer 値）
            enable_passives     (bool):  パッシブアイテムの有効化
            enable_evolutions   (bool):  進化武器の有効化
            replay_old_phase_fraction (float): 旧フェーズリプレイの割合 (0.0-1.0)
            starting_weapon_mode (str):  開始武器選択モード（"pool_random" など）
            weapon_weights      (dict):  武器タイプ int → 重み float の辞書（"weighted" モード用）
            item_selection_mode (str):   "auto"（既定）または "external"
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

    def choose_level_up(
        self, decision_id: str, choice_id: str
    ) -> tuple[np.ndarray, dict]:
        """保留中の level-up choice を exactly-once endpoint へ送る。

        通信タイムアウト時は同じ ID を再送し、UE5 の idempotency 応答で二重適用を防ぐ。
        """
        if not isinstance(decision_id, str) or not decision_id:
            raise ValueError("decision_id must be a non-empty string")
        if not isinstance(choice_id, str) or not choice_id:
            raise ValueError("choice_id must be a non-empty string")

        data = self._post_json(
            "/level_up_choice",
            {"decision_id": decision_id, "choice_id": choice_id},
            timeout=10,
            retries=2,
        )
        if data.get("status") != "applied":
            raise RuntimeError(f"unexpected level-up response: {data!r}")
        if data.get("decision_id") != decision_id or data.get("choice_id") != choice_id:
            raise RuntimeError("level-up acknowledgement IDs do not match request")

        received_hash = data.get("obs_schema_hash", "")
        if self._expected_schema_hash and received_hash != self._expected_schema_hash:
            raise RuntimeError(
                "obs_schema_hash が一致しません。"
                f" expected={self._expected_schema_hash}, received={received_hash}"
            )
        info = data.get("info")
        if (
            not isinstance(info, dict)
            or not isinstance(info.get("level_up_pending"), bool)
            or not isinstance(info.get("level_up_choices"), list)
        ):
            raise RuntimeError("level-up response info contract is malformed")

        obs = np.array(data["obs"], dtype=np.float32)
        self._prev_obs = obs
        return obs, info

    def preview_level_up(
        self, decision_id: str, expected_choice_ids: Collection[str]
    ) -> SurvivorsLevelUpPreview:
        """pending全候補のproduction適用直後raw observationを取得する。

        Python では値を予測せず、decision・schema・choice ID 集合だけを検証する。
        """

        if not isinstance(decision_id, str) or not decision_id:
            raise ValueError("decision_id must be a non-empty string")
        if not self._expected_schema_hash:
            raise ValueError("obs_schema_hash is not initialized")
        shape = getattr(self.observation_space, "shape", None)
        if (
            not isinstance(shape, tuple)
            or len(shape) != 1
            or isinstance(shape[0], bool)
            or not isinstance(shape[0], int)
            or shape[0] <= 0
        ):
            raise ValueError("observation_space must expose one positive dimension")
        segment_names: list[str] = []
        schema_dim = 0
        for index, segment in enumerate(self._obs_schema):
            if not isinstance(segment, dict):
                raise ValueError(f"obs_schema segment {index} must be an object")
            name = segment.get("name")
            dim = segment.get("dim")
            if not isinstance(name, str) or not name or name in segment_names:
                raise ValueError(f"obs_schema segment {index} has invalid name")
            if isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0:
                raise ValueError(f"obs_schema segment {index} has invalid dim")
            segment_names.append(name)
            schema_dim += dim
        if schema_dim != shape[0]:
            raise ValueError(
                "obs_schema dimension mismatch: "
                f"segments={schema_dim}, observation_space={shape[0]}"
            )
        data = self._post_json(
            "/preview_level_up",
            {"decision_id": decision_id},
            timeout=10,
            retries=2,
        )
        return parse_level_up_preview(
            data,
            expected_decision_id=decision_id,
            expected_schema_hash=self._expected_schema_hash,
            expected_choice_ids=expected_choice_ids,
            obs_dim=shape[0],
            schema_segment_names=segment_names,
        )

    def activate_validation_branch_rng(
        self,
        replication_key: str,
        stream_seed: int,
    ) -> dict:
        """semantic replay 後の validation-only post-decision stream を有効化する。

        candidate ID を送らず、Python で固定した replication key/seed binding を応答で再確認する。
        """
        if (
            not isinstance(replication_key, str)
            or len(replication_key) != 64
            or any(
                character not in "0123456789abcdef"
                for character in replication_key
            )
        ):
            raise ValueError("replication_key must be lowercase sha256")
        if (
            isinstance(stream_seed, bool)
            or not isinstance(stream_seed, int)
            or not -(2**31) <= stream_seed < 2**31
        ):
            raise ValueError("stream_seed must fit signed int32")
        data = self._post_json(
            "/validation_branch_rng",
            {
                "schema_version": BRANCH_RNG_SCHEMA_VERSION,
                "replication_key": replication_key,
                "stream_seed": stream_seed,
            },
            timeout=10,
            retries=2,
        )
        expected = {
            "schema_version": BRANCH_RNG_SCHEMA_VERSION,
            "replication_key": replication_key,
            "stream_seed": stream_seed,
            "validation_only": True,
        }
        if any(data.get(field) != value for field, value in expected.items()):
            raise RuntimeError("validation branch RNG acknowledgement mismatch")
        initial_state = data.get("initial_stream_state")
        if isinstance(initial_state, bool) or not isinstance(
            initial_state, (int, float)
        ):
            raise RuntimeError("validation branch RNG state is malformed")
        return dict(data)

    def get_params(self) -> dict:
        """最後に set_params で適用したパラメータを返す。eval_env との同期用。"""
        return dict(self._last_params)

    def _action_to_payload(self, action) -> dict:
        """discrete action を UE5 /step wire payload へ変換する。"""
        return {"action": [float(int(action))]}


# 旧ドキュメント名を canonical SurvivorsEnv へ結び付ける互換 alias。
# SurvivorsUE5Env を import する既存コードでも同じ choice API を利用できる。
SurvivorsUE5Env = SurvivorsEnv


from stable_baselines3.common.monitor import Monitor


class SurvivorsMonitor(Monitor):
    """Monitor + SurvivorsEnv 固有メソッドの明示フォワード。

    gymnasium の __getattr__ 経由のフォワードは v1.0 で廃止されるため
    必要なメソッドを明示定義して非推奨警告を抑制する。
    DummyVecEnv / SubprocVecEnv 両対応。
    """

    def set_params(self, **kwargs) -> bool:
        """wrapped environment へ runtime parameters を転送する。"""
        return self.env.set_params(**kwargs)

    def get_params(self) -> dict:
        """wrapped environment の直近 parameters を返す。"""
        return self.env.get_params()

    def set_shaping_weight(self, weight: float) -> None:
        """wrapped environment の shaping weight を更新する。"""
        self.env.set_shaping_weight(weight)

    def get_shaping_weight(self) -> float:
        """wrapped environment の shaping weight を返す。"""
        return self.env.get_shaping_weight()

    def clear_reward_fn(self) -> None:
        """wrapped environment の reward shaping function を解除する。"""
        self.env.clear_reward_fn()

    def get_obs_schema_hash(self) -> str:
        """wrapped environment の observation schema hash を返す。"""
        return self.env.get_obs_schema_hash()

    def get_offsets(self):
        """wrapped environment の observation offsets を返す。"""
        return self.env.get_offsets()

    def get_obs_schema(self):
        """wrapped environment の observation schema を返す。"""
        return self.env.get_obs_schema()

    def choose_level_up(
        self, decision_id: str, choice_id: str
    ) -> tuple[np.ndarray, dict]:
        """wrapped SurvivorsEnv の外部 level-up API を明示的に転送する。"""
        return self.env.choose_level_up(decision_id, choice_id)

    def preview_level_up(
        self, decision_id: str, expected_choice_ids: Collection[str]
    ) -> SurvivorsLevelUpPreview:
        """wrapped SurvivorsEnv のtyped preview APIを明示的に転送する。

        VecEnv から呼ぶ場合も同じ wire 検証を必ず通す。
        """

        return self.env.preview_level_up(decision_id, expected_choice_ids)

    def activate_validation_branch_rng(
        self,
        replication_key: str,
        stream_seed: int,
    ) -> dict:
        """wrapped SurvivorsEnv の validation branch RNG API を転送する。"""
        return self.env.activate_validation_branch_rng(
            replication_key,
            stream_seed,
        )
