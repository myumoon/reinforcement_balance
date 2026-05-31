"""UE5 env への set_params 送信を担当するモジュール。

SpalfCallback / CurriculumCallback / HybridCallback が共通使用する。
training_env は _on_training_start 後に set_training_env() で設定する。
"""

from __future__ import annotations

from typing import Optional


class ParamApplier:
    """UE5 env への set_params 送信を担当するモジュール。

    SpalfCallback / CurriculumCallback / HybridCallback が共通使用する。
    training_env は _on_training_start 後に set_training_env() で設定する。
    """

    # Survivors 内部 key → UE5 API key のマッピング
    _KEY_MAP: dict[str, str] = {
        "min_enemies":        "MinActiveEnemies",
        "max_enemies":        "MaxActiveEnemies",
        "speed_mult":         "EnemySpeedMult",
        "spawn_rate_mult":    "SpawnRateMult",
        "max_enemy_type_id":  "MaxEnemyTypeId",
        "enemy_hp_scale":     "EnemyHPScale",
        "enemy_damage_scale": "EnemyDamageScale",
        "time_scaling":       "TimeScalingEnabled",
    }

    def __init__(self, raw_env=None):
        self._raw_env = raw_env
        self._training_env = None

    def set_training_env(self, training_env) -> None:
        """BaseCallback._on_training_start() 後に呼ぶ。"""
        self._training_env = training_env

    def apply(self, params: dict, env_idx: Optional[int] = None) -> bool:
        """params を UE5 env に送信する。

        Args:
            params:   内部 key 形式の dict（"min_enemies" など）
            env_idx:  None = 全 env、int = 指定 env のみ

        Returns:
            成功した場合 True
        """
        ue5_params = {self._KEY_MAP.get(k, k): v for k, v in params.items()}

        # training_env（DummyVecEnv/SubprocVecEnv）を優先する。
        # n_envs=1 でも VecNormalize/Monitor wrapper 経由のため training_env を使う。
        # raw_env は training_env が未設定（_on_training_start 前）の場合のみ使用。
        if self._training_env is not None:
            indices = [env_idx] if env_idx is not None else None
            results = self._training_env.env_method("set_params", indices=indices, **ue5_params)
            failed = [i for i, r in enumerate(results or []) if r is False]
            if failed:
                print(f"[ParamApplier][ERROR] set_params 失敗: env index {failed}")
            return len(failed) == 0
        elif self._raw_env is not None:
            return self._raw_env.set_params(**ue5_params)
        return False

    def get_current_params(self) -> Optional[dict]:
        """training_env から現在のパラメータを取得する（eval sync 用）。"""
        if self._training_env is None:
            return None
        params_list = self._training_env.env_method("get_params")
        params_list = [p for p in (params_list or []) if p]
        return params_list[0] if params_list else None
