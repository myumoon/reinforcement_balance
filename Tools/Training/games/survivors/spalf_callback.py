"""SPALF (Self-Paced Absolute Learning Progress) による連続パラメータカリキュラム。

ALP-GMM をベースに正則化関数 f_α で ALP を変換し、エージェントの成熟度に応じて
正則化の on/off を切り替える Self-Paced 制御を実装する。

パラメータ空間 (8次元):
  min_enemies, max_enemies, speed_mult, spawn_rate_mult,
  max_enemy_type_id, enemy_hp_scale, enemy_damage_scale, time_scaling

使用例:
  python train.py --game survivors --spalf --total-steps 5000000
  python train.py --game survivors --spalf --resume runs/survivors/v06/train/run-spalf
"""

import json
import math
import random
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from games.survivors.survivors_difficulty import compute_difficulty_score

# パラメータ空間の定義（PHASES 最大値を超えた領域も探索）
_PARAM_BOUNDS: dict[str, tuple] = {
    "min_enemies":        (4,    80),
    "max_enemies":        (6,   300),
    "speed_mult":         (0.8,  1.2),   # VS 本編でも高難易度で敵は遅いため上限は 1.2
    "spawn_rate_mult":    (1.0,  6.0),
    "max_enemy_type_id":  (1,    10),
    "enemy_hp_scale":     (0.5,  4.0),
    "enemy_damage_scale": (0.5,  4.0),
    "time_scaling":       (0.0,  1.0),   # 0.5 以上で True
}

_PARAM_KEYS = list(_PARAM_BOUNDS.keys())
_N_PARAMS = len(_PARAM_KEYS)

# warm-up 期間に適用する Phase 0 相当パラメータ（内部 dict キー名で定義）
_PHASE0_PARAMS: dict = {
    "min_enemies":        4,
    "max_enemies":        6,
    "speed_mult":         0.8,
    "spawn_rate_mult":    1.0,
    "max_enemy_type_id":  1,
    "enemy_hp_scale":     0.5,
    "enemy_damage_scale": 0.5,
    "time_scaling":       False,
}


class SpalfCallback(BaseCallback):
    """SPALF 連続パラメータカリキュラムコールバック。

    Args:
        raw_env:           SurvivorsEnv インスタンス（n_envs==1 時の set_params 用）
        frame_skip:        train.py の --frame-skip と同じ値
        alive_reward:      生存ボーナス係数（train.py の --curriculum-alive-reward と同じ値）
        r_b:               正規化後バッファ平均報酬の境界値（< r_b で SPALF モード）
        alpha:             正則化強度 α（正規化スコア空間）
        max_score:         スコア正規化スケール。score_norm = active_score / max_score が
                           常に [0, 1] に収まるよう、訓練中の期待最大スコアより大きく設定すること。
                           小さすぎると score_norm > 1.0 となり ALP が破綻する。
        buffer_size:       (params, score) / (params, ALP) バッファサイズ（エピソード数）
        warmup_episodes:   Phase 0 固定で動作する初期エピソード数
        status_path:       spalf_state.json の出力先（None = 保存なし）
    """

    def __init__(
        self,
        raw_env,
        frame_skip: int = 4,
        alive_reward: float = 0.001,
        r_b: float = 0.1,
        alpha: float = 0.2,
        max_score: float = 2250.0,
        buffer_size: int = 200,
        warmup_episodes: int = 50,
        status_path=None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self._raw_env = raw_env
        self._frame_skip = frame_skip
        self._alive_reward = alive_reward
        self._r_b = r_b
        self._alpha = alpha
        self._max_score = max(max_score, 1e-8)
        self._buffer_size = buffer_size
        self._warmup_episodes = warmup_episodes
        self._status_path = status_path

        # バッファ
        self._reward_history: deque = deque(maxlen=buffer_size)  # (param_vec, score_norm)
        self._alp_buffer: deque = deque(maxlen=buffer_size)       # (param_vec, alp)
        self._recent_reward_buffer: deque = deque(maxlen=buffer_size)  # score_norm

        # 状態
        self._total_episodes: int = 0
        self._gmm = None
        self._component_mean_alp: np.ndarray = np.array([])
        self._use_spalf_mode: bool = True
        self._current_params: dict = dict(_PHASE0_PARAMS)
        self._current_param_vec: np.ndarray = self._params_to_vec(_PHASE0_PARAMS)
        self._ep_base_per_env: list[float] = []
        # env ごとに「現在エピソード開始時点のパラメータ vec」を記録する。
        # n_envs > 1 では params 変更が全 env に即時適用されるため、
        # エピソード途中の env は旧 params で走り続ける。
        # ALP 計算時はこの per-env 値を使い、スコアを正しいパラメータに帰属させる。
        self._ep_start_param_vec_per_env: list[np.ndarray] = []

    # ------------------------------------------------------------------
    # SB3 BaseCallback インターフェース
    # ------------------------------------------------------------------

    def _on_training_start(self) -> None:
        n = self.training_env.num_envs if self.training_env is not None else 1
        self._ep_base_per_env = [0.0] * n
        # resume 時は import_state で復元済みの _current_params を適用する。
        # 新規開始時は __init__ で _current_params = _PHASE0_PARAMS に初期化済み。
        self._apply_params(self._current_params)
        # 全 env はこの時点から同一パラメータで開始する
        self._ep_start_param_vec_per_env = [self._current_param_vec.copy() for _ in range(n)]

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        dones = self.locals.get("dones", [False] * len(infos))
        n_envs = len(infos)

        if len(self._ep_base_per_env) != n_envs:
            self._ep_base_per_env = [0.0] * n_envs
        if len(self._ep_start_param_vec_per_env) != n_envs:
            self._ep_start_param_vec_per_env = [self._current_param_vec.copy() for _ in range(n_envs)]

        ep_active_scores: list[float] = []
        ep_score_norms: list[float] = []

        for env_idx, info in enumerate(infos):
            self._ep_base_per_env[env_idx] += info.get("base_reward", 0.0)

            if "episode" not in info:
                continue

            ep_len = info["episode"]["l"]
            ep_base = self._ep_base_per_env[env_idx]
            alive_total = self._alive_reward * self._frame_skip * ep_len
            active_score = max(0.0, ep_base - alive_total)
            score_norm = active_score / self._max_score
            self._ep_base_per_env[env_idx] = 0.0
            self._total_episodes += 1

            # SPALF モード切替判定
            self._recent_reward_buffer.append(score_norm)
            if self._recent_reward_buffer:
                mean_norm = sum(self._recent_reward_buffer) / len(self._recent_reward_buffer)
                self._use_spalf_mode = mean_norm < self._r_b

            # warm-up 期間はバッファ更新・パラメータ変更をスキップ
            if self._total_episodes <= self._warmup_episodes:
                ep_active_scores.append(active_score)
                ep_score_norms.append(score_norm)
                continue

            # ALP 計算: このエピソードが「開始した時点のパラメータ」を使う。
            # n_envs > 1 では他 env のエピソード終了により params が途中変更されるため、
            # _current_param_vec（最後にサンプルしたパラメータ）ではなく
            # _ep_start_param_vec_per_env[env_idx]（このエピソード開始時点のパラメータ）を使う。
            ep_param_vec = self._ep_start_param_vec_per_env[env_idx]
            alp = self._compute_alp_for_episode(score_norm, ep_param_vec)

            # バッファ更新（スコアをそのエピソードのパラメータに帰属）
            self._reward_history.append((ep_param_vec.copy(), score_norm))
            self._alp_buffer.append((ep_param_vec.copy(), alp))

            # GMM 再フィット（5 エピソードごと）
            # _total_episodes は warmup 分も含むため ALP バッファサイズと乖離するが、
            # len(self._alp_buffer) >= 10 のガードが実質的な最小サンプル数を保証する。
            if self._total_episodes % 5 == 0 and len(self._alp_buffer) >= 10:
                self._fit_gmm()

            # 次のパラメータをサンプリングして env_idx のみに適用する。
            # 他 env はエピソード途中のため送信不要（自身のエピソード終了時に受け取る）。
            new_params, new_vec = self._sample_next_params()
            self._apply_params(new_params, env_idx=env_idx)
            self._current_param_vec = new_vec
            # env_idx の次エピソードは new_vec で開始する。
            # 他 env はまだ旧パラメータのエピソード途中なので _ep_start_param_vec を変えない。
            self._ep_start_param_vec_per_env[env_idx] = new_vec.copy()

            ep_active_scores.append(active_score)
            ep_score_norms.append(score_norm)

            # spalf_state.json 保存（50 エピソードごと）
            if self._total_episodes % 50 == 0 and self._status_path is not None:
                self._save_status()

        # W&B ログ: 同一 num_timesteps で複数回呼ばれないよう、ステップ末尾に 1 回だけ実行
        if ep_active_scores:
            self._log_wandb_per_step(ep_active_scores, ep_score_norms)

        return True

    # ------------------------------------------------------------------
    # 正規化・変換ユーティリティ
    # ------------------------------------------------------------------

    def _params_to_vec(self, params: dict) -> np.ndarray:
        """env パラメータ dict → [0,1] 正規化ベクトル (8次元)。"""
        vec = np.zeros(_N_PARAMS, dtype=np.float32)
        for i, key in enumerate(_PARAM_KEYS):
            lo, hi = _PARAM_BOUNDS[key]
            val = params.get(key, lo)
            if isinstance(val, bool):
                val = 1.0 if val else 0.0
            vec[i] = (float(val) - lo) / (hi - lo)
        return np.clip(vec, 0.0, 1.0)

    def _vec_to_params(self, vec: np.ndarray) -> dict:
        """[0,1] ベクトル → env パラメータ dict。制約を適用してから返す。"""
        params: dict = {}
        for i, key in enumerate(_PARAM_KEYS):
            lo, hi = _PARAM_BOUNDS[key]
            val = lo + float(vec[i]) * (hi - lo)
            if key in ("min_enemies", "max_enemies", "max_enemy_type_id"):
                val = int(round(val))
            elif key == "time_scaling":
                val = float(vec[i]) >= 0.5
            params[key] = val
        # 制約: max_enemies >= min_enemies + 1
        if params["max_enemies"] < params["min_enemies"] + 1:
            params["max_enemies"] = params["min_enemies"] + 1
        return params

    def _f_alpha(self, x: float) -> float:
        """正則化関数 f_α(x) = -α × (1 - exp(x/α))。x/α を [-500, 500] にクリップ。"""
        clamped = min(max(x / self._alpha, -500.0), 500.0)
        return -self._alpha * (1.0 - math.exp(clamped))

    def _compute_alp(self, score_norm_new: float, score_norm_old: float) -> float:
        if self._use_spalf_mode:
            return abs(self._f_alpha(score_norm_new) - self._f_alpha(score_norm_old))
        return abs(score_norm_new - score_norm_old)

    def _compute_alp_for_episode(self, score_norm: float, param_vec: np.ndarray) -> float:
        """バッファ最近傍から r_old を取得して ALP を計算する。"""
        if not self._reward_history:
            r_old = 0.0
        else:
            vecs = np.array([b[0] for b in self._reward_history])
            dists = np.linalg.norm(vecs - param_vec, axis=1)
            nearest_idx = int(np.argmin(dists))
            r_old = list(self._reward_history)[nearest_idx][1]
        return self._compute_alp(score_norm, r_old)

    # ------------------------------------------------------------------
    # GMM フィッティング
    # ------------------------------------------------------------------

    def _fit_gmm(self) -> None:
        try:
            from sklearn.mixture import GaussianMixture
        except ImportError:
            print("[SPALF][WARN] scikit-learn がインストールされていません。GMM をスキップします。")
            return

        X = np.array([b[0] for b in self._alp_buffer])
        alp_values = np.array([b[1] for b in self._alp_buffer])

        K_MAX = min(10, len(self._alp_buffer) // 5)
        best_gmm, best_bic = None, float("inf")
        for k in range(1, K_MAX + 1):
            try:
                gmm = GaussianMixture(
                    n_components=k, covariance_type="full", n_init=3, random_state=42
                )
                gmm.fit(X)
                bic = gmm.bic(X)
                if bic < best_bic:
                    best_bic, best_gmm = bic, gmm
            except Exception:
                continue

        if best_gmm is None:
            return

        labels = best_gmm.predict(X)
        self._component_mean_alp = np.array([
            alp_values[labels == k].mean() if (labels == k).any() else 0.0
            for k in range(best_gmm.n_components)
        ])
        self._gmm = best_gmm

    # ------------------------------------------------------------------
    # サンプリング
    # ------------------------------------------------------------------

    def _sample_next_params(self) -> tuple[dict, np.ndarray]:
        use_random = (
            len(self._alp_buffer) < 50
            or self._gmm is None
            or random.random() >= 0.8
        )
        if use_random:
            vec = np.random.uniform(0.0, 1.0, size=_N_PARAMS).astype(np.float32)
        else:
            weights = np.maximum(self._component_mean_alp, 0.0) + 1e-6
            weights /= weights.sum()
            k = np.random.choice(len(weights), p=weights)
            mean = self._gmm.means_[k]
            cov = self._gmm.covariances_[k]
            vec = np.random.multivariate_normal(mean, cov).astype(np.float32)
            vec = np.clip(vec, 0.0, 1.0)

        params = self._vec_to_params(vec)
        return params, self._params_to_vec(params)

    # ------------------------------------------------------------------
    # パラメータ適用
    # ------------------------------------------------------------------

    def _apply_params(self, params: dict, env_idx: Optional[int] = None) -> None:
        """env_idx を指定すると該当 env のみに送信する。None の場合は全 env に送信する。"""
        ue5_params = dict(
            MinActiveEnemies=params["min_enemies"],
            MaxActiveEnemies=params["max_enemies"],
            EnemySpeedMult=params["speed_mult"],
            SpawnRateMult=params["spawn_rate_mult"],
            MaxEnemyTypeId=params["max_enemy_type_id"],
            EnemyHPScale=params["enemy_hp_scale"],
            EnemyDamageScale=params["enemy_damage_scale"],
            TimeScalingEnabled=params["time_scaling"],
        )
        if self.training_env is not None and self.training_env.num_envs > 1:
            indices = [env_idx] if env_idx is not None else None
            results = self.training_env.env_method("set_params", indices=indices, **ue5_params)
            failed = [i for i, r in enumerate(results) if not r]
            if failed:
                actual_indices = [env_idx] if env_idx is not None else list(range(self.training_env.num_envs))
                print(f"[SPALF][ERROR] /params 適用失敗: env index {[actual_indices[i] for i in failed]}")
        else:
            if self._raw_env is not None:
                ok = self._raw_env.set_params(**ue5_params)
                if not ok:
                    print("[SPALF][ERROR] /params 適用失敗: single env")
        self._current_params = params
        if self.verbose >= 1:
            print(
                f"[SPALF] params 更新: "
                f"敵数 {params['min_enemies']}-{params['max_enemies']}, "
                f"速度x{params['speed_mult']:.2f}, "
                f"スポーンx{params['spawn_rate_mult']:.2f}, "
                f"TypeId<={params['max_enemy_type_id']}, "
                f"HP x{params['enemy_hp_scale']:.2f}, "
                f"Dmg x{params['enemy_damage_scale']:.2f}, "
                f"TimeScaling={'ON' if params['time_scaling'] else 'OFF'}"
            )

    # ------------------------------------------------------------------
    # W&B ロギング
    # ------------------------------------------------------------------

    def get_wandb_progress_metrics(self) -> dict:
        """副作用なしで W&B ログ用メトリクス dict を返す。"""
        mean_norm = (
            sum(self._recent_reward_buffer) / len(self._recent_reward_buffer)
            if self._recent_reward_buffer
            else 0.0
        )
        alp_mean = (
            float(np.mean([b[1] for b in self._alp_buffer]))
            if self._alp_buffer
            else 0.0
        )
        mode = 0 if self._total_episodes <= self._warmup_episodes else (1 if self._use_spalf_mode else 2)
        return {
            "spalf/mode":                       mode,
            "spalf/mean_reward_normalized":     mean_norm,
            "spalf/buffer_size":                len(self._alp_buffer),
            "spalf/alp_mean":                   alp_mean,
            "spalf/current_min_enemies":        self._current_params.get("min_enemies", 0),
            "spalf/current_max_enemies":        self._current_params.get("max_enemies", 0),
            "spalf/current_speed_mult":         self._current_params.get("speed_mult", 0.0),
            "spalf/current_spawn_rate_mult":    self._current_params.get("spawn_rate_mult", 0.0),
            "spalf/current_max_enemy_type_id":  self._current_params.get("max_enemy_type_id", 0),
            "spalf/current_enemy_hp_scale":     self._current_params.get("enemy_hp_scale", 0.0),
            "spalf/current_enemy_damage_scale": self._current_params.get("enemy_damage_scale", 0.0),
            "spalf/current_time_scaling":       int(bool(self._current_params.get("time_scaling", False))),
            "survivors/difficulty_score":       compute_difficulty_score(self._current_params),
            "spalf/score_norm_over1_ratio":     (
                sum(1 for s in self._recent_reward_buffer if s > 1.0) / len(self._recent_reward_buffer)
                if self._recent_reward_buffer else 0.0
            ),
        }

    def _log_wandb_per_step(
        self, active_scores: list[float], score_norms: list[float]
    ) -> None:
        try:
            import wandb
            if not wandb.run:
                return
            metrics = self.get_wandb_progress_metrics()
            metrics.update({
                "survivors/active_score":     sum(active_scores) / len(active_scores),
                "survivors/score_normalized": sum(score_norms) / len(score_norms),
                "global_step": self.num_timesteps,
            })
            wandb.log(metrics, step=self.num_timesteps)
        except ImportError:
            pass

    # ------------------------------------------------------------------
    # 状態の export / import（resume サポート）
    # ------------------------------------------------------------------

    def export_state(self) -> dict:
        wandb_run_id: Optional[str] = None
        try:
            import wandb
            if wandb.run:
                wandb_run_id = wandb.run.id
        except ImportError:
            pass

        def _serialize_params(p: dict) -> dict:
            return {k: (bool(v) if isinstance(v, (bool, np.bool_)) else v) for k, v in p.items()}

        return {
            "reward_history": [
                [b[0].tolist(), float(b[1])] for b in self._reward_history
            ],
            "alp_buffer": [
                [b[0].tolist(), float(b[1])] for b in self._alp_buffer
            ],
            "recent_reward_buffer": list(self._recent_reward_buffer),
            "total_episodes": self._total_episodes,
            "current_params": _serialize_params(self._current_params),
            "use_spalf_mode": bool(self._use_spalf_mode),
            "wandb_run_id": wandb_run_id,
        }

    def import_state(self, state: dict) -> None:
        self._reward_history = deque(
            [(np.array(b[0], dtype=np.float32), float(b[1])) for b in state.get("reward_history", [])],
            maxlen=self._buffer_size,
        )
        self._alp_buffer = deque(
            [(np.array(b[0], dtype=np.float32), float(b[1])) for b in state.get("alp_buffer", [])],
            maxlen=self._buffer_size,
        )
        self._recent_reward_buffer = deque(
            [float(v) for v in state.get("recent_reward_buffer", [])],
            maxlen=self._buffer_size,
        )
        self._total_episodes = int(state.get("total_episodes", 0))
        self._current_params = state.get("current_params", dict(_PHASE0_PARAMS))
        self._use_spalf_mode = bool(state.get("use_spalf_mode", True))
        self._current_param_vec = self._params_to_vec(self._current_params)

        # バッファが十分あれば GMM を再フィット
        if len(self._alp_buffer) >= 10:
            self._fit_gmm()

    def _save_status(self) -> None:
        if self._status_path is None:
            return
        Path(self._status_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self._status_path).write_text(
            json.dumps(self.export_state(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
