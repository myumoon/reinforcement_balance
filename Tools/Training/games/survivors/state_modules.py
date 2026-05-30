"""Survivors 訓練用ステートモジュール群。
EpisodeScoreTracker / SpalfStateModule / CurriculumStateModule を提供する。
"""

from __future__ import annotations

import json
import math
import random
from abc import ABC, abstractmethod
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

from games.survivors.survivors_difficulty import PARAM_BOUNDS, compute_difficulty_score

from abc import ABC, abstractmethod

class BaseStateModule(ABC):
    """SpalfStateModule / CurriculumStateModule の共通インターフェース。"""

    @abstractmethod
    def get_wandb_metrics(self) -> dict:
        """W&B ログ用メトリクス dict を返す（副作用なし）。"""

    @abstractmethod
    def export_state(self) -> dict:
        """resume 用の状態を dict で返す。"""

    @abstractmethod
    def import_state(self, state: dict) -> None:
        """export_state() の出力から状態を復元する。"""


class EpisodeScoreTracker:
    """per-env の base_reward 蓄積と active_score 計算。

    SpalfCallback と CurriculumCallback で重複していた計算ロジックを統一する。
    BaseStateModule は継承しない（計算ユーティリティのため）。
    """

    def __init__(self, frame_skip: int, alive_reward: float):
        self.frame_skip = frame_skip
        self.alive_reward = alive_reward
        self._ep_base_per_env: list[float] = []

    def reset(self, n_envs: int) -> None:
        self._ep_base_per_env = [0.0] * n_envs

    def process(self, infos: list[dict]) -> list[tuple[int, float, int]]:
        """infos を処理して終了したエピソードの (env_idx, active_score, ep_len) を返す。"""
        n = len(infos)
        if len(self._ep_base_per_env) != n:
            self._ep_base_per_env = [0.0] * n
        results: list[tuple[int, float, int]] = []
        for env_idx, info in enumerate(infos):
            self._ep_base_per_env[env_idx] += info.get("base_reward", 0.0)
            if "episode" not in info:
                continue
            ep_len = info["episode"]["l"]
            alive_total = self.alive_reward * self.frame_skip * ep_len
            active_score = max(0.0, self._ep_base_per_env[env_idx] - alive_total)
            self._ep_base_per_env[env_idx] = 0.0
            results.append((env_idx, active_score, ep_len))
        return results


_PARAM_KEYS = list(PARAM_BOUNDS.keys())
_N_PARAMS = len(_PARAM_KEYS)

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


class SpalfStateModule(BaseStateModule):
    """SPALF ALP-GMM ロジック・bounds 正規化・W&B ログ・状態管理を担当する。"""

    def __init__(self, r_b: float, alpha: float, max_score: float,
                 buffer_size: int, warmup_episodes: int,
                 status_path=None, initial_bounds=None):
        self._r_b = r_b
        self._alpha = alpha
        self._max_score = max(max_score, 1e-8)
        self._buffer_size = buffer_size
        self._warmup_episodes = warmup_episodes
        self._status_path = status_path
        self._active_bounds = initial_bounds or PARAM_BOUNDS
        self._reward_history: deque = deque(maxlen=buffer_size)
        self._alp_buffer: deque = deque(maxlen=buffer_size)
        self._recent_reward_buffer: deque = deque(maxlen=buffer_size)
        self._total_episodes: int = 0
        self._gmm = None
        self._component_mean_alp: np.ndarray = np.array([])
        self._use_spalf_mode: bool = True
        self._current_params: dict = dict(_PHASE0_PARAMS)
        self._current_param_vec: np.ndarray = self.params_to_vec(_PHASE0_PARAMS)

    def set_bounds(self, bounds: dict) -> None:
        self._active_bounds = bounds

    def reset_buffers(self) -> None:
        self._reward_history.clear()
        self._alp_buffer.clear()
        self._recent_reward_buffer.clear()
        self._total_episodes = 0
        self._gmm = None
        self._component_mean_alp = np.array([])

    def params_to_vec(self, params: dict) -> np.ndarray:
        vec = np.zeros(_N_PARAMS, dtype=np.float32)
        for i, key in enumerate(_PARAM_KEYS):
            lo, hi = self._active_bounds[key]
            val = params.get(key, lo)
            if isinstance(val, bool):
                val = 1.0 if val else 0.0
            vec[i] = (float(val) - lo) / (hi - lo)
        return np.clip(vec, 0.0, 1.0)

    def vec_to_params(self, vec: np.ndarray) -> dict:
        params: dict = {}
        for i, key in enumerate(_PARAM_KEYS):
            lo, hi = self._active_bounds[key]
            val = lo + float(vec[i]) * (hi - lo)
            if key in ("min_enemies", "max_enemies", "max_enemy_type_id"):
                val = int(round(val))
            elif key == "time_scaling":
                val = float(vec[i]) >= 0.5
            params[key] = val
        if params["max_enemies"] < params["min_enemies"] + 1:
            params["max_enemies"] = params["min_enemies"] + 1
        return params

    def _f_alpha(self, x: float) -> float:
        clamped = min(max(x / self._alpha, -500.0), 500.0)
        return -self._alpha * (1.0 - math.exp(clamped))

    def _compute_alp(self, score_norm_new: float, score_norm_old: float) -> float:
        if self._use_spalf_mode:
            return abs(self._f_alpha(score_norm_new) - self._f_alpha(score_norm_old))
        return abs(score_norm_new - score_norm_old)

    def _compute_alp_for_episode(self, score_norm: float, param_vec: np.ndarray) -> float:
        if not self._reward_history:
            r_old = 0.0
        else:
            vecs = np.array([b[0] for b in self._reward_history])
            dists = np.linalg.norm(vecs - param_vec, axis=1)
            nearest_idx = int(np.argmin(dists))
            r_old = list(self._reward_history)[nearest_idx][1]
        return self._compute_alp(score_norm, r_old)

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
                gmm = GaussianMixture(n_components=k, covariance_type="full", n_init=3, random_state=42)
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

    def sample_next_params(self) -> tuple[dict, np.ndarray]:
        use_random = (len(self._alp_buffer) < 50 or self._gmm is None or random.random() >= 0.8)
        if use_random:
            vec = np.random.uniform(0.0, 1.0, size=_N_PARAMS).astype(np.float32)
        else:
            weights = np.maximum(self._component_mean_alp, 0.0) + 1e-6
            weights /= weights.sum()
            k = np.random.choice(len(weights), p=weights)
            mean_ = self._gmm.means_[k]
            cov = self._gmm.covariances_[k]
            vec = np.random.multivariate_normal(mean_, cov).astype(np.float32)
            vec = np.clip(vec, 0.0, 1.0)
        params = self.vec_to_params(vec)
        return params, self.params_to_vec(params)

    def on_episode_end(self, env_idx: int, active_score: float, ep_start_param_vec: np.ndarray) -> bool:
        """戻り値: warmup 中か否か（True=warmup）"""
        score_norm = active_score / self._max_score
        self._total_episodes += 1
        self._recent_reward_buffer.append(score_norm)
        if self._recent_reward_buffer:
            mean_norm = sum(self._recent_reward_buffer) / len(self._recent_reward_buffer)
            self._use_spalf_mode = mean_norm < self._r_b
        if self._total_episodes <= self._warmup_episodes:
            return True
        alp = self._compute_alp_for_episode(score_norm, ep_start_param_vec)
        self._reward_history.append((ep_start_param_vec.copy(), score_norm))
        self._alp_buffer.append((ep_start_param_vec.copy(), alp))
        if self._total_episodes % 5 == 0 and len(self._alp_buffer) >= 10:
            self._fit_gmm()
        return False

    def get_wandb_metrics(self) -> dict:
        mean_norm = (sum(self._recent_reward_buffer) / len(self._recent_reward_buffer)
            if self._recent_reward_buffer else 0.0)
        alp_mean = (float(np.mean([b[1] for b in self._alp_buffer]))
            if self._alp_buffer else 0.0)
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

    def export_state(self) -> dict:
        wandb_run_id: Optional[str] = None
        try:
            import wandb
            if wandb.run:
                wandb_run_id = wandb.run.id
        except ImportError:
            pass
        def _ser(p_: dict) -> dict:
            return {k: (bool(v) if isinstance(v, (bool, np.bool_)) else v) for k, v in p_.items()}
        return {
            "reward_history": [[b[0].tolist(), float(b[1])] for b in self._reward_history],
            "alp_buffer": [[b[0].tolist(), float(b[1])] for b in self._alp_buffer],
            "recent_reward_buffer": list(self._recent_reward_buffer),
            "total_episodes": self._total_episodes,
            "current_params": _ser(self._current_params),
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
        self._current_param_vec = self.params_to_vec(self._current_params)
        if len(self._alp_buffer) >= 10:
            self._fit_gmm()

    def save_status(self, path=None) -> None:
        target = path or self._status_path
        if target is None:
            return
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        Path(target).write_text(
            json.dumps(self.export_state(), ensure_ascii=False, indent=2), encoding="utf-8"
        )


