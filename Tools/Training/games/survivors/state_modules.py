"""Survivors 訓練用ステートモジュール群。
EpisodeScoreTracker / SpalfStateModule / CurriculumStateModule /
WeaponPhaseAutoStateModule を提供する。
"""

from __future__ import annotations

import json
import math
import random
from abc import ABC, abstractmethod
from collections import deque
from pathlib import Path
from typing import Callable, Optional

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
        results: list[tuple[int, float, int, float]] = []
        for env_idx, info in enumerate(infos):
            self._ep_base_per_env[env_idx] += info.get("base_reward", 0.0)
            if "episode" not in info:
                continue
            ep_len = info["episode"]["l"]
            ep_base = self._ep_base_per_env[env_idx]
            alive_total = self.alive_reward * self.frame_skip * ep_len
            active_score = max(0.0, ep_base - alive_total)
            self._ep_base_per_env[env_idx] = 0.0
            results.append((env_idx, active_score, ep_len, ep_base))
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
        self._phase_warmup_remaining: int = 0

    def set_bounds(self, bounds: dict) -> None:
        self._active_bounds = bounds

    def reset_buffers(self) -> None:
        self._reward_history.clear()
        self._alp_buffer.clear()
        self._recent_reward_buffer.clear()
        self._total_episodes = 0
        self._gmm = None
        self._component_mean_alp = np.array([])

    def reset_buffers_for_phase_change(self) -> None:
        """フェーズ切り替え時に ALP バッファのみリセットする。

        _recent_reward_buffer / _total_episodes / _use_spalf_mode は維持する
        （前フェーズの習熟度を引き継ぐため）。
        SpalfStateModule.reset_buffers() は全バッファをクリアするため、
        Hybrid のフェーズ切り替えではこちらを使用すること。
        """
        self._reward_history.clear()
        self._alp_buffer.clear()
        self._gmm = None
        self._component_mean_alp = np.array([])

    def set_phase_warmup(self, n_episodes: int) -> None:
        """フェーズ切り替え後の per-phase warmup を設定する。

        n_episodes エピソードの間は ALP 計算をスキップしてランダムサンプリングを行う。
        _total_episodes ベースではなく、呼び出しからのカウントダウンで管理する。
        """
        self._phase_warmup_remaining = n_episodes

    def params_to_vec(self, params: dict) -> np.ndarray:
        vec = np.zeros(_N_PARAMS, dtype=np.float32)
        for i, key in enumerate(_PARAM_KEYS):
            lo, hi = self._active_bounds[key]
            val = params.get(key, lo)
            if isinstance(val, bool):
                val = 1.0 if val else 0.0
            if hi <= lo:          # ゼロ幅 bounds（固定値）→ 中央値 0.5 でマーク
                vec[i] = 0.5
                continue
            vec[i] = (float(val) - lo) / (hi - lo)
        return np.clip(vec, 0.0, 1.0)

    def vec_to_params(self, vec: np.ndarray) -> dict:
        params: dict = {}
        for i, key in enumerate(_PARAM_KEYS):
            lo, hi = self._active_bounds[key]
            if hi <= lo:          # ゼロ幅 bounds → lo 固定
                val = lo
            else:
                val = lo + float(vec[i]) * (hi - lo)
            if key in ("min_enemies", "max_enemies", "max_enemy_type_id"):
                val = int(round(val))
            elif key == "time_scaling":
                val = float(val) >= 0.5  # bounds 適用後の val を bool 化（ゼロ幅 bounds 時も lo を使う）
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

        # per-phase warmup 中は ALP/GMM 更新のみスキップ（統計量は上記で更新済み）
        if self._phase_warmup_remaining > 0:
            self._phase_warmup_remaining -= 1
            return True
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




class CurriculumStateModule(BaseStateModule):
    """カリキュラムのフェーズ管理・昇格・ロールバックロジックを担当するステートモジュール。

    CurriculumCallback からフェーズ遷移ロジックを分離したクラス。
    フェーズ定義 (PHASES) と判定ロジックは survivors_curriculum から参照する。
    """

    def __init__(
        self,
        window: int,
        threshold_mult: float,
        rollback_patience: int = 3,
        status_path=None,
    ):
        from games.survivors.survivors_curriculum import PHASES
        self._PHASES = PHASES
        self.window = window
        self.threshold_mult = threshold_mult
        self.rollback_patience = rollback_patience
        # rollback 判定に必要な最低 episode 数。window の半分を基本とし、window を超えない。
        # max(5, ...) を使うと window < 10 では rollback_min_episodes > window となり
        # 必要数を永遠に満たせなくなるため、window を上限にする。
        self.rollback_min_episodes = min(max(1, window // 2), window)
        self._status_path = status_path

        self._phase_idx: int = 0
        self._scores: list[float] = []
        self._rollback_bad_windows: int = 0
        self._episode_scores: list[float] = []
        self._episode_lengths: list[int] = []
        self._episode_base_rewards: list[float] = []
        self._episode_alive_rewards: list[float] = []
        self._terminated_count: int = 0
        self._truncated_count: int = 0
        self._phase_events: list[dict] = []
        self._steps_in_phase: int = 0
        self._episodes_in_phase: int = 0
        self._missing_episode_info_count: int = 0

        self._completion_window: int = window
        self._completion_min_episodes: int = window
        self._completion_min_score_ratio: float = 1.0
        self._completion_min_episode_len_ratio: float = 1.0

        # probe window（HybridPromotionProbeCallback が書き込む昇格判定専用バッファ）
        self._promotion_scores: list[float] = []
        self._promotion_episode_lengths: list[int] = []
        self._promotion_base_rewards: list[float] = []
        self._promotion_events: list[dict] = []

    @property
    def current_phase(self) -> int:
        return self._phase_idx

    @property
    def is_final_phase(self) -> bool:
        return self._phase_idx >= len(self._PHASES) - 1

    def configure_completion(self, window, min_episodes, min_score_ratio, min_episode_len_ratio):
        self._completion_window = max(1, int(window))
        self._completion_min_episodes = max(1, int(min_episodes))
        self._completion_min_score_ratio = max(0.0, float(min_score_ratio))
        self._completion_min_episode_len_ratio = max(0.0, float(min_episode_len_ratio))

    def on_episode_end(self, active_score: float, ep_len: int,
                       base_reward: float = 0.0, alive_reward: float = 0.0,
                       terminated: bool = True) -> None:
        self._scores.append(active_score)
        self._episode_scores.append(active_score)
        self._episode_lengths.append(ep_len)
        self._episode_base_rewards.append(base_reward)
        self._episode_alive_rewards.append(alive_reward)
        if terminated:
            self._terminated_count += 1
        else:
            self._truncated_count += 1
        self._episodes_in_phase += 1

    def check_phase_transition(self, *, allow_promotion: bool = True, promotion_source: str = "train"):
        from base.curriculum import mean as _mean, stdev as _stdev, percentile as _percentile
        phase = self._PHASES[self._phase_idx]
        effective_threshold = (phase.threshold or float("inf")) * self.threshold_mult
        recent_scores = self._scores[-min(len(self._scores), self.window):]
        recent_lengths = self._episode_lengths[-len(recent_scores):] if recent_scores else []
        mean = _mean(recent_scores)
        mean_len = _mean([float(v) for v in recent_lengths])
        score_min = min(recent_scores) if recent_scores else 0.0
        score_std = _stdev(recent_scores)
        self.save_status()
        if len(self._scores) < self.window:
            return None
        rollback, reason = self._should_rollback(phase, mean, mean_len, effective_threshold, len(recent_scores))
        if rollback:
            self._rollback_bad_windows += 1
            if self._rollback_bad_windows >= self.rollback_patience:
                self._rollback_phase(mean, effective_threshold, mean_len, reason)
                return "rollback"
        else:
            self._rollback_bad_windows = 0
        if not allow_promotion:
            return None
        can_promote, promotion_reason = self._promotion_judgment(phase, mean, mean_len, score_min, score_std, effective_threshold, len(recent_scores), recent_scores)
        if can_promote:
            self._advance_phase(mean, effective_threshold, promotion_reason)
            return "advance"
        return None

    def on_promotion_probe_episode_end(
        self, active_score: float, ep_len: int, base_reward: float = 0.0
    ) -> None:
        """probe episode の結果を promotion 判定専用バッファに追加する。"""
        self._promotion_scores.append(active_score)
        self._promotion_episode_lengths.append(ep_len)
        self._promotion_base_rewards.append(base_reward)

    def check_promotion_transition(self, *, promotion_source: str = "probe") -> str | None:
        """probe scores のみで昇格判定を行う（rollback は発生しない）。

        probe window が window 数に満たない場合、または昇格条件未達の場合は None を返す。
        """
        from base.curriculum import mean as _mean, stdev as _stdev, percentile as _percentile
        phase = self._PHASES[self._phase_idx]
        if phase.threshold is None:
            return None
        effective_threshold = phase.threshold * self.threshold_mult
        recent_scores = self._promotion_scores[-min(len(self._promotion_scores), self.window):]
        recent_lengths = self._promotion_episode_lengths[-len(recent_scores):] if recent_scores else []
        if len(recent_scores) < self.window:
            return None
        mean = _mean(recent_scores)
        mean_len = _mean([float(v) for v in recent_lengths])
        score_min = min(recent_scores) if recent_scores else 0.0
        score_std = _stdev(recent_scores)
        can_promote, promotion_reason = self._promotion_judgment(
            phase, mean, mean_len, score_min, score_std,
            effective_threshold, len(recent_scores), recent_scores
        )
        if can_promote:
            self._advance_phase(mean, effective_threshold, f"probe:{promotion_reason}")
            return "advance"
        return None

    def clear_promotion_probe_window(self) -> None:
        """probe window をクリアする（phase advance / rollback 時に呼ばれる）。"""
        self._promotion_scores.clear()
        self._promotion_episode_lengths.clear()
        self._promotion_base_rewards.clear()

    def get_promotion_probe_diagnostics(self) -> dict:
        """probe window の現状を返す（W&B ログ・デバッグ用）。"""
        from base.curriculum import mean as _mean, stdev as _stdev, percentile as _percentile
        phase = self._PHASES[self._phase_idx]
        n = len(self._promotion_scores)
        recent = self._promotion_scores[-min(n, self.window):]
        score_mean = _mean(recent) if recent else 0.0
        score_std = _stdev(recent) if recent else 0.0
        score_cv = score_std / max(score_mean, 1e-8) if recent and score_mean > 0.0 else None
        threshold = (phase.threshold or float("inf")) * self.threshold_mult if phase.threshold else None
        return {
            "n_probe_episodes": n,
            "window_required": self.window,
            "score_mean": round(score_mean, 4),
            "score_min": round(min(recent), 4) if recent else None,
            "score_p10": round(_percentile(recent, 10.0), 4) if recent else None,
            "score_cv": round(score_cv, 4) if score_cv is not None else None,
            "ep_length_mean": round(_mean([float(v) for v in self._promotion_episode_lengths[-len(recent):]]), 1) if recent else None,
            "threshold": round(threshold, 4) if threshold is not None else None,
            "promotion_ready": threshold is not None and len(recent) >= self.window and score_mean >= threshold,
        }

    def _should_rollback(self, phase, mean, mean_len, effective_threshold, recent_count):
        if self._phase_idx <= 0 or recent_count < self.rollback_min_episodes:
            return False, ""
        score_floor, length_floor = self._rollback_floors(phase, effective_threshold)
        if score_floor is not None and mean < score_floor:
            return True, f"score={mean:.3f} < rollback_score_floor={score_floor:.3f}"
        if length_floor is not None and mean_len < length_floor:
            return True, f"ep_len={mean_len:.1f} < rollback_length_floor={length_floor:.1f}"
        return False, ""

    def _rollback_floors(self, phase, effective_threshold):
        if phase.threshold is not None:
            score_floor = effective_threshold * phase.rollback_score_ratio
        elif phase.rollback_reference_threshold is not None:
            score_floor = phase.rollback_reference_threshold * self.threshold_mult * phase.rollback_score_ratio
        else:
            score_floor = None
        length_floor = (phase.min_episode_steps * phase.rollback_length_ratio if phase.min_episode_steps > 0 else None)
        return score_floor, length_floor

    def _promotion_judgment(self, phase, mean, mean_len, score_min, score_std, effective_threshold, recent_count, recent_scores=None):
        from base.curriculum import percentile as _percentile
        if phase.threshold is None or recent_count < self.window:
            return False, ""
        if mean < effective_threshold:
            return False, f"score_mean={mean:.3f} < threshold={effective_threshold:.3f}"
        if mean_len < phase.min_episode_steps:
            return False, f"ep_len={mean_len:.1f} < min_episode_steps={phase.min_episode_steps}"
        if phase.promotion_score_stat == "percentile" and recent_scores is not None:
            low_score = _percentile(recent_scores, phase.promotion_score_percentile)
            low_stat_label = f"score_p{int(phase.promotion_score_percentile)}"
        else:
            low_score = score_min
            low_stat_label = "score_min"
        min_score_floor = effective_threshold * phase.promotion_min_score_ratio
        if low_score < min_score_floor:
            return False, f"{low_stat_label}={low_score:.3f} < promotion_min_score_floor={min_score_floor:.3f}"
        score_cv = score_std / max(mean, 1e-8)
        if score_cv > phase.promotion_max_score_cv:
            return False, f"score_cv={score_cv:.3f} > promotion_max_score_cv={phase.promotion_max_score_cv:.3f}"
        return True, f"{low_stat_label}={low_score:.3f} >= {min_score_floor:.3f}, score_cv={score_cv:.3f} <= {phase.promotion_max_score_cv:.3f}"

    def _advance_phase(self, mean, threshold, reason):
        prev_name = self._PHASES[self._phase_idx].name
        prev_idx = self._phase_idx
        self._phase_idx = min(self._phase_idx + 1, len(self._PHASES) - 1)
        self._scores.clear()
        self._rollback_bad_windows = 0
        self._steps_in_phase = 0
        self._episodes_in_phase = 0
        self.clear_promotion_probe_window()
        next_phase = self._PHASES[self._phase_idx]
        self._phase_events.append({"event": "advance", "from_phase_idx": prev_idx, "from_phase_name": prev_name, "to_phase_idx": self._phase_idx, "to_phase_name": next_phase.name, "active_score_mean": round(mean, 4), "threshold": round(threshold, 4), "reason": reason})
        msg = "[Curriculum] Phase " + str(self._phase_idx) + " 昇格: " + prev_name + " -> " + next_phase.name + " (score=" + str(round(mean, 3)) + " >= " + str(round(threshold, 1)) + ", " + reason + ")"
        print(msg)

    def _rollback_phase(self, mean, threshold, mean_len, reason):
        prev_name = self._PHASES[self._phase_idx].name
        prev_idx = self._phase_idx
        self._phase_idx = max(self._phase_idx - 1, 0)
        self._scores.clear()
        self._rollback_bad_windows = 0
        self._steps_in_phase = 0
        self._episodes_in_phase = 0
        self.clear_promotion_probe_window()
        next_phase = self._PHASES[self._phase_idx]
        self._phase_events.append({"event": "rollback", "from_phase_idx": prev_idx, "from_phase_name": prev_name, "to_phase_idx": self._phase_idx, "to_phase_name": next_phase.name, "active_score_mean": round(mean, 4), "episode_length_mean": round(mean_len, 1), "threshold": round(threshold, 4), "reason": reason})
        msg = "[Curriculum] Phase " + str(self._phase_idx) + " 降格: " + prev_name + " -> " + next_phase.name + " (" + reason + ", score=" + str(round(mean, 3)) + ", ep_len=" + str(round(mean_len, 1)) + ")"
        print(msg)

    def rollback_one_phase(self, reason: str = "external") -> None:
        """外部トリガーによる1フェーズ強制降格（スコア判定なし）。

        WeaponPhaseAutoStateModule など外部モジュールから武器フェーズ昇格に伴い呼ばれる。
        phase_idx のみを更新するため、敵難易度パラメータ適用は呼び出し元が担う。
        """
        if self._phase_idx <= 0:
            print(f"[Curriculum] 強制降格スキップ: 既に最初のフェーズです ({reason})")
            return
        prev_name = self._PHASES[self._phase_idx].name
        prev_idx = self._phase_idx
        self._phase_idx = max(self._phase_idx - 1, 0)
        self._scores.clear()
        self._rollback_bad_windows = 0
        self._steps_in_phase = 0
        self._episodes_in_phase = 0
        self.clear_promotion_probe_window()
        next_phase = self._PHASES[self._phase_idx]
        self._phase_events.append({
            "event": "forced_rollback",
            "from_phase_idx": prev_idx,
            "from_phase_name": prev_name,
            "to_phase_idx": self._phase_idx,
            "to_phase_name": next_phase.name,
            "reason": reason,
        })
        print(f"[Curriculum] 強制降格: {prev_name} -> {next_phase.name} ({reason})")

    def get_wandb_metrics(self) -> dict:
        from base.curriculum import mean as _mean, stdev as _stdev, percentile as _percentile
        phase = self._PHASES[self._phase_idx]
        base_threshold = phase.threshold
        effective_threshold = (base_threshold or float("inf")) * self.threshold_mult
        recent_count = min(len(self._scores), self.window)
        recent_scores = self._scores[-recent_count:] if recent_count else []
        score_mean = _mean(recent_scores)
        threshold_ratio = (score_mean / effective_threshold if base_threshold is not None and effective_threshold > 0.0 and recent_scores else None)
        score_cv = (_stdev(recent_scores) / max(score_mean, 1e-8) if recent_scores and score_mean > 0.0 else None)
        if phase.promotion_score_stat == "percentile" and recent_scores:
            promotion_low_score = _percentile(recent_scores, phase.promotion_score_percentile)
        else:
            promotion_low_score = min(recent_scores) if recent_scores else 0.0
        promotion_low_score_floor = (effective_threshold * phase.promotion_min_score_ratio if base_threshold is not None else None)
        completion = self.get_completion_diagnostics()
        return {
            "curriculum/phase_idx": self._phase_idx,
            "curriculum/phase_progress_ratio": threshold_ratio,
            "curriculum/score_mean": score_mean,
            "curriculum/score_min": min(recent_scores) if recent_scores else None,
            "curriculum/score_cv": score_cv,
            "curriculum/threshold_ratio": threshold_ratio,
            "curriculum/steps_in_phase": self._steps_in_phase,
            "curriculum/episodes_in_phase": self._episodes_in_phase,
            "curriculum/rollback_bad_windows": self._rollback_bad_windows,
            "curriculum/window_episode_count": recent_count,
            "curriculum/is_final_phase": int(bool(completion.get("is_final_phase"))),
            "curriculum/completion_ready": int(bool(completion.get("complete"))),
            "curriculum/completion_score_mean": completion.get("active_score_mean"),
            "curriculum/completion_ep_len_mean": completion.get("episode_length_mean"),
            "curriculum/promotion_low_score": round(promotion_low_score, 4),
            "curriculum/promotion_low_score_floor": (round(promotion_low_score_floor, 4) if promotion_low_score_floor is not None else None),
            "curriculum/promotion_low_score_ok": int(promotion_low_score_floor is not None and promotion_low_score >= promotion_low_score_floor),
            "survivors/difficulty_score": compute_difficulty_score({
                "min_enemies": phase.min_enemies,
                "max_enemies": phase.max_enemies,
                "speed_mult": phase.speed_mult,
                "spawn_rate_mult": phase.spawn_rate_mult,
                "max_enemy_type_id": phase.max_enemy_type_id,
                "enemy_hp_scale": phase.enemy_hp_scale,
                "enemy_damage_scale": phase.enemy_damage_scale,
                "time_scaling": phase.time_scaling,
            }),
        }

    def get_diagnostics(self, num_timesteps: int = 0) -> dict:
        """CurriculumCallback.get_diagnostics() と同等。callback 側から委譲される。"""
        from base.curriculum import mean as _mean, stdev as _stdev, percentile as _percentile
        phase = self._PHASES[self._phase_idx]
        base_threshold = phase.threshold
        effective_threshold = (base_threshold or float("inf")) * self.threshold_mult
        recent_count = min(len(self._scores), self.window)
        recent_scores = self._scores[-recent_count:] if recent_count else []
        recent_lengths = self._episode_lengths[-recent_count:] if recent_count else []
        score_mean = _mean(recent_scores)
        score_std = _stdev(recent_scores)
        length_mean = _mean([float(v) for v in recent_lengths])
        score_floor, length_floor = self._rollback_floors(phase, effective_threshold)
        threshold_ratio = (
            score_mean / effective_threshold
            if base_threshold is not None and effective_threshold > 0.0 and recent_scores
            else None
        )
        score_min = min(recent_scores) if recent_scores else None
        score_cv = (
            score_std / max(score_mean, 1e-8)
            if recent_scores and score_mean > 0.0 else None
        )
        if phase.promotion_score_stat == "percentile" and recent_scores:
            promotion_low_score = _percentile(recent_scores, phase.promotion_score_percentile)
        else:
            promotion_low_score = score_min if score_min is not None else 0.0
        promotion_low_score_floor = (
            effective_threshold * phase.promotion_min_score_ratio
            if base_threshold is not None else None
        )
        promotion_low_score_ok = (
            promotion_low_score_floor is not None
            and promotion_low_score >= promotion_low_score_floor
        )
        promotion_stable = (
            recent_count >= self.window and base_threshold is not None
            and promotion_low_score_floor is not None and promotion_low_score_ok
            and score_cv is not None and score_cv <= phase.promotion_max_score_cv
        )
        return {
            "timestep": num_timesteps,
            "phase_idx": self._phase_idx,
            "phase_name": phase.name,
            "window": self.window,
            "threshold_mult": self.threshold_mult,
            "rollback_patience": self.rollback_patience,
            "rollback_min_episodes": self.rollback_min_episodes,
            "rollback_bad_windows": self._rollback_bad_windows,
            "episodes_total": len(self._episode_scores),
            "terminated_episodes": self._terminated_count,
            "truncated_episodes": self._truncated_count,
            "current_window": {
                "episodes": recent_count,
                "required_episodes": self.window,
                "active_score_mean": round(score_mean, 4),
                "active_score_min": round(score_min, 4) if score_min is not None else None,
                "active_score_max": round(max(recent_scores), 4) if recent_scores else None,
                "active_score_std": round(score_std, 4),
                "active_score_cv": round(score_cv, 4) if score_cv is not None else None,
                "active_score_p10": round(_percentile(recent_scores, 10.0), 4) if recent_scores else None,
                "active_score_p20": round(_percentile(recent_scores, 20.0), 4) if recent_scores else None,
                "episode_length_mean": round(length_mean, 1),
                "min_episode_steps": phase.min_episode_steps,
                "base_threshold": base_threshold,
                "effective_threshold": round(effective_threshold, 4) if base_threshold is not None else None,
                "threshold_ratio": round(threshold_ratio, 4) if threshold_ratio is not None else None,
                "promotion_min_score_ratio": phase.promotion_min_score_ratio,
                "promotion_max_score_cv": phase.promotion_max_score_cv,
                "promotion_score_stat": phase.promotion_score_stat,
                "promotion_score_percentile": phase.promotion_score_percentile,
                "promotion_low_score": round(promotion_low_score, 4),
                "promotion_low_score_floor": round(promotion_low_score_floor, 4) if promotion_low_score_floor is not None else None,
                "promotion_low_score_ok": promotion_low_score_ok,
                "promotion_stability_ok": promotion_stable,
                "rollback_score_floor": round(score_floor, 4) if score_floor is not None else None,
                "rollback_length_floor": round(length_floor, 1) if length_floor is not None else None,
                "rollback_bad_windows": self._rollback_bad_windows,
                "ready_for_phase_judgment": (
                    recent_count >= self.window and base_threshold is not None
                    and score_mean >= effective_threshold
                    and length_mean >= phase.min_episode_steps and promotion_stable
                ),
            },
            "overall": {
                "active_score_mean": round(_mean(self._episode_scores), 4),
                "episode_length_mean": round(_mean([float(v) for v in self._episode_lengths]), 1),
                "base_reward_mean": round(_mean(self._episode_base_rewards), 4),
                "alive_reward_mean": round(_mean(self._episode_alive_rewards), 4),
            },
            "phase_events": self._phase_events,
            "completion": self.get_completion_diagnostics(),
        }

    def _compute_blocker_category(self, window_dict: dict, phase_threshold_is_none: bool) -> int:
        """CurriculumCallback._compute_blocker_category() と同等。callback 側から委譲される。"""
        from games.survivors.survivors_curriculum import (
            BLOCKER_NONE, BLOCKER_NOT_ENOUGH_EP, BLOCKER_SCORE_MEAN_LOW,
            BLOCKER_EP_LEN_LOW, BLOCKER_SCORE_MIN_LOW, BLOCKER_SCORE_CV_HIGH,
        )
        if phase_threshold_is_none or self.is_final_phase:
            return BLOCKER_NONE
        recent_count = int(window_dict.get("episodes", 0) or 0)
        required = int(window_dict.get("required_episodes", 1) or 1)
        if recent_count < required:
            return BLOCKER_NOT_ENOUGH_EP
        score_mean = float(window_dict.get("active_score_mean") or 0.0)
        threshold = float(window_dict.get("effective_threshold") or float("inf"))
        if score_mean < threshold:
            return BLOCKER_SCORE_MEAN_LOW
        length_mean = float(window_dict.get("episode_length_mean") or 0.0)
        min_ep_steps = int(window_dict.get("min_episode_steps") or 0)
        if length_mean < min_ep_steps:
            return BLOCKER_EP_LEN_LOW
        promotion_low_score = window_dict.get("promotion_low_score")
        min_score_floor = window_dict.get("promotion_low_score_floor")
        if (promotion_low_score is not None and min_score_floor is not None
                and promotion_low_score < min_score_floor):
            return BLOCKER_SCORE_MIN_LOW
        score_cv = window_dict.get("active_score_cv")
        max_cv = window_dict.get("promotion_max_score_cv")
        if score_cv is not None and max_cv is not None and score_cv > max_cv:
            return BLOCKER_SCORE_CV_HIGH
        return BLOCKER_NONE

    def get_wandb_progress_metrics(self, num_timesteps: int = 0) -> dict:
        """CurriculumCallback.get_wandb_progress_metrics() と同等。callback 側から委譲される。"""
        diagnostics = self.get_diagnostics(num_timesteps)
        window = diagnostics.get("current_window", {})
        overall = diagnostics.get("overall", {})
        completion = diagnostics.get("completion", {})
        episodes_total = max(int(diagnostics.get("episodes_total", 0) or 0), 1)
        terminated = int(diagnostics.get("terminated_episodes", 0) or 0)
        truncated = int(diagnostics.get("truncated_episodes", 0) or 0)
        phase = self._PHASES[self._phase_idx]
        blocker = self._compute_blocker_category(
            window_dict=window, phase_threshold_is_none=(phase.threshold is None)
        )
        return {
            "survivors/active_score_mean": window.get("active_score_mean"),
            "survivors/active_score_min": window.get("active_score_min"),
            "survivors/active_score_std": window.get("active_score_std"),
            "survivors/active_score_cv": window.get("active_score_cv"),
            "survivors/episode_length_mean": window.get("episode_length_mean"),
            "survivors/base_reward_mean": overall.get("base_reward_mean"),
            "survivors/alive_reward_mean": overall.get("alive_reward_mean"),
            "survivors/terminated_ratio": terminated / episodes_total,
            "survivors/truncated_ratio": truncated / episodes_total,
            "curriculum/phase_idx": diagnostics.get("phase_idx"),
            "curriculum/phase_progress_ratio": window.get("threshold_ratio"),
            "curriculum/score_mean": window.get("active_score_mean"),
            "curriculum/score_min": window.get("active_score_min"),
            "curriculum/score_cv": window.get("active_score_cv"),
            "curriculum/threshold_ratio": window.get("threshold_ratio"),
            "curriculum/steps_in_phase": self._steps_in_phase,
            "curriculum/episodes_in_phase": self._episodes_in_phase,
            "curriculum/promotion_blocker": blocker,
            "curriculum/ready_for_phase_judgment": int(bool(window.get("ready_for_phase_judgment"))),
            "curriculum/promotion_stability_ok": int(bool(window.get("promotion_stability_ok"))),
            "curriculum/active_score_p10": window.get("active_score_p10"),
            "curriculum/active_score_p20": window.get("active_score_p20"),
            "curriculum/promotion_low_score": window.get("promotion_low_score"),
            "curriculum/promotion_low_score_floor": window.get("promotion_low_score_floor"),
            "curriculum/promotion_low_score_ok": int(bool(window.get("promotion_low_score_ok"))),
            "curriculum/promotion_score_percentile": window.get("promotion_score_percentile"),
            "curriculum/rollback_bad_windows": diagnostics.get("rollback_bad_windows"),
            "curriculum/window_episode_count": window.get("episodes"),
            "curriculum/is_final_phase": int(bool(completion.get("is_final_phase"))),
            "curriculum/completion_ready": int(bool(completion.get("complete"))),
            "curriculum/completion_score_mean": completion.get("active_score_mean"),
            "curriculum/completion_ep_len_mean": completion.get("episode_length_mean"),
        }

    def _completion_base_threshold(self):
        phase = self._PHASES[self._phase_idx]
        if phase.threshold is not None:
            return phase.threshold
        if self._phase_idx <= 0:
            return None
        return self._PHASES[self._phase_idx - 1].threshold

    def get_completion_diagnostics(self, window=None, min_episodes=None, min_score_ratio=None, min_episode_len_ratio=None):
        from base.curriculum import mean as _mean, stdev as _stdev
        window = self._completion_window if window is None else max(1, int(window))
        min_episodes = self._completion_min_episodes if min_episodes is None else max(1, int(min_episodes))
        min_score_ratio = self._completion_min_score_ratio if min_score_ratio is None else max(0.0, float(min_score_ratio))
        min_episode_len_ratio = self._completion_min_episode_len_ratio if min_episode_len_ratio is None else max(0.0, float(min_episode_len_ratio))
        phase = self._PHASES[self._phase_idx]
        recent_count = min(len(self._scores), window)
        recent_scores = self._scores[-recent_count:] if recent_count else []
        recent_lengths = self._episode_lengths[-recent_count:] if recent_count else []
        score_mean = _mean(recent_scores)
        score_min = min(recent_scores) if recent_scores else None
        score_std = _stdev(recent_scores)
        score_cv = (score_std / max(score_mean, 1e-8) if recent_scores and score_mean > 0.0 else None)
        length_mean = _mean([float(v) for v in recent_lengths])
        base_threshold = self._completion_base_threshold()
        score_floor = (base_threshold * self.threshold_mult * min_score_ratio if base_threshold is not None else None)
        length_floor = phase.min_episode_steps * min_episode_len_ratio
        reasons = []
        if not self.is_final_phase:
            reasons.append("not_final_phase")
        if recent_count < min_episodes:
            reasons.append(f"episodes={recent_count} < min_episodes={min_episodes}")
        if score_floor is not None and score_mean < score_floor:
            reasons.append(f"score_mean={score_mean:.3f} < completion_score_floor={score_floor:.3f}")
        if length_mean < length_floor:
            reasons.append(f"ep_len={length_mean:.1f} < completion_length_floor={length_floor:.1f}")
        if score_min is not None and score_floor is not None and score_min < score_floor * phase.promotion_min_score_ratio:
            reasons.append(f"score_min={score_min:.3f} < completion_min_score_floor={score_floor * phase.promotion_min_score_ratio:.3f}")
        if score_cv is not None and score_cv > phase.promotion_max_score_cv:
            reasons.append(f"score_cv={score_cv:.3f} > promotion_max_score_cv={phase.promotion_max_score_cv:.3f}")
        return {"complete": len(reasons) == 0, "reasons": reasons, "is_final_phase": self.is_final_phase, "phase_idx": self._phase_idx, "phase_name": phase.name, "window": window, "episodes": recent_count, "min_episodes": min_episodes, "active_score_mean": round(score_mean, 4), "active_score_min": round(score_min, 4) if score_min is not None else None, "active_score_std": round(score_std, 4), "active_score_cv": round(score_cv, 4) if score_cv is not None else None, "episode_length_mean": round(length_mean, 1), "completion_base_threshold": base_threshold, "completion_score_floor": round(score_floor, 4) if score_floor is not None else None, "completion_length_floor": round(length_floor, 1), "min_score_ratio": min_score_ratio, "min_episode_len_ratio": min_episode_len_ratio}

    def export_state(self) -> dict:
        phase_idx = self._phase_idx
        return {
            "phase_idx": phase_idx,
            "phase_name": self._PHASES[phase_idx].name,
            "scores_window": self._scores,
            "rollback_bad_windows": self._rollback_bad_windows,
            "ep_base": 0.0,
            "episode_scores": self._episode_scores,
            "episode_lengths": self._episode_lengths,
            "episode_base_rewards": self._episode_base_rewards,
            "episode_alive_rewards": self._episode_alive_rewards,
            "terminated_count": self._terminated_count,
            "truncated_count": self._truncated_count,
            "phase_events": self._phase_events,
            "steps_in_phase": self._steps_in_phase,
            "episodes_in_phase": self._episodes_in_phase,
            "promotion_scores": self._promotion_scores,
            "promotion_episode_lengths": self._promotion_episode_lengths,
            "promotion_base_rewards": self._promotion_base_rewards,
            "promotion_events": self._promotion_events,
        }

    def import_state(self, state: dict) -> None:
        phase_idx = int(state.get("phase_idx", 0))
        self._phase_idx = max(0, min(phase_idx, len(self._PHASES) - 1))
        phase_name = state.get("phase_name")
        if phase_name:
            for idx, phase in enumerate(self._PHASES):
                if phase.name == phase_name:
                    self._phase_idx = idx
                    break
        self._scores = [float(v) for v in state.get("scores_window", [])]
        self._rollback_bad_windows = int(state.get("rollback_bad_windows", 0))
        self._episode_scores = [float(v) for v in state.get("episode_scores", [])]
        self._episode_lengths = [int(v) for v in state.get("episode_lengths", [])]
        self._episode_base_rewards = [float(v) for v in state.get("episode_base_rewards", [])]
        self._episode_alive_rewards = [float(v) for v in state.get("episode_alive_rewards", [])]
        self._terminated_count = int(state.get("terminated_count", 0))
        self._truncated_count = int(state.get("truncated_count", 0))
        self._phase_events = list(state.get("phase_events", []))
        self._steps_in_phase = int(state.get("steps_in_phase", 0))
        self._episodes_in_phase = int(state.get("episodes_in_phase", 0))
        self._promotion_scores = [float(v) for v in state.get("promotion_scores", [])]
        self._promotion_episode_lengths = [int(v) for v in state.get("promotion_episode_lengths", [])]
        self._promotion_base_rewards = [float(v) for v in state.get("promotion_base_rewards", [])]
        self._promotion_events = list(state.get("promotion_events", []))

    def save_status(self, path=None) -> None:
        import json
        from pathlib import Path
        target = path or self._status_path
        if target is None:
            return
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        Path(target).write_text(json.dumps(self.export_state(), ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# WeaponPhaseAutoStateModule
# ---------------------------------------------------------------------------

# 循環インポートを避けるため実行時にのみロード
def _get_weapon_phases():
    from games.survivors.survivors_weapon_curriculum import WEAPON_PHASES, get_params_for_phase
    return WEAPON_PHASES, get_params_for_phase

WEAPON_PHASE_AUTO_SEQUENCE: list[str] = [
    "W1", "W2", "W3", "W4", "W5",
]


class WeaponPhaseAutoStateModule(BaseStateModule):
    """武器フェーズ自動昇格の状態管理モジュール。

    固定フェーズ（W1/W2 等）では CurriculumStateModule の current_phase を監視し、
    stagnation_steps ステップ以上変化しなかった場合に次フェーズへ進める。
    遷移フェーズ（W1_to_W2 等）では カリキュラム状態に関わらず transition_steps
    経過後に自動で次の固定フェーズへ移行する（補間を必ず完走させるため）。

    設計上の分離:
    - 本モジュールは「状態管理」のみを担う（set_params 送信は担わない）
    - カリキュラム降格は rollback_fn（外部注入）に委譲する
      - --curriculum 時: CurriculumCallback.rollback_one_phase（state + enemy params 適用）
      - --curriculum-spalf 時: HybridCurriculumSpalfCallback.rollback_one_phase
        （state + SPALF更新 + params 適用）
    - 武器パラメータの set_params 送信は WeaponPhaseAutoCallback（SB3コールバック側）が担う
    - on_step() が新しい武器フェーズキーを返した時点で、コールバックが set_params を呼ぶ
    - 状態永続化は train_status_{step}_steps.json 経由のみ（ファイル直接書き込みは行わない）
    """

    def __init__(
        self,
        curriculum: "CurriculumStateModule",
        stagnation_steps: int = 500_000,
        rollback_fn: "Callable[[str], None] | None" = None,
    ) -> None:
        """
        Args:
            curriculum: 停滞検出に使用する CurriculumStateModule（read-only）。
            stagnation_steps: 固定フェーズでカリキュラムが変化しない場合に停滞とみなすステップ数。
                遷移フェーズでは使用されない（transition_steps を使う）。
            rollback_fn: 武器フェーズ昇格時に呼ぶカリキュラム降格関数。
                None の場合は curriculum.rollback_one_phase を使用（--curriculum 単体向け）。
                --curriculum-spalf では HybridCurriculumSpalfCallback.rollback_one_phase を渡すこと。
        """
        self._curriculum = curriculum
        self._stagnation_steps = max(stagnation_steps, 1)
        self._rollback_fn: Callable[[str], None] = (
            rollback_fn if rollback_fn is not None else curriculum.rollback_one_phase
        )

        self._weapon_phase_seq_idx: int = 0
        self._last_curriculum_phase: int = -1  # 初回 on_step でリセットされる（後方互換用、未使用）
        self._max_curriculum_phase: int = self._curriculum.current_phase  # 最大到達フェーズ（rollback でリセットしない）
        self._stagnation_start_step: int = 0
        self._phase_start_step: int = 0        # 現在の武器フェーズ開始時の num_timesteps
        self._state_restored: bool = False     # import_state() で復元済みか否か

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def current_phase_key(self) -> str:
        """現在の武器フェーズキー（WEAPON_PHASE_AUTO_SEQUENCE のいずれか）。"""
        idx = min(self._weapon_phase_seq_idx, len(WEAPON_PHASE_AUTO_SEQUENCE) - 1)
        return WEAPON_PHASE_AUTO_SEQUENCE[idx]

    @property
    def is_final_weapon_phase(self) -> bool:
        return self._weapon_phase_seq_idx >= len(WEAPON_PHASE_AUTO_SEQUENCE) - 1

    @property
    def is_transition(self) -> bool:
        """現在が遷移フェーズ（weapon_pool_mode="weighted"）かどうか。"""
        WEAPON_PHASES, _ = _get_weapon_phases()
        return WEAPON_PHASES.get(self.current_phase_key, {}).get("weapon_pool_mode") == "weighted"

    def get_transition_elapsed(self, num_timesteps: int) -> int:
        """現在の武器フェーズ開始からの経過ステップ数（遷移フェーズの補間に使用）。"""
        return max(0, num_timesteps - self._phase_start_step)

    # ------------------------------------------------------------------ #
    # Core logic                                                           #
    # ------------------------------------------------------------------ #

    def reset_stagnation_timer(self, num_timesteps: int) -> None:
        """停滞タイマーを現在ステップでリセットする（resume 時の _on_training_start で呼ぶ）。"""
        self._last_curriculum_phase = self._curriculum.current_phase
        self._max_curriculum_phase = self._curriculum.current_phase
        self._stagnation_start_step = num_timesteps

    def on_step(self, num_timesteps: int) -> "str | None":
        """毎ステップ呼ぶ。武器フェーズが昇格した場合は新フェーズキーを返す。

        - 遷移フェーズ中（W1_to_W2 等）: transition_steps 経過で次フェーズへ自動進行。
          カリキュラム停滞判定は行わない（補間を必ず完走させるため）。
        - 固定フェーズ中（W1/W2 等）: カリキュラム current_phase が stagnation_steps
          ステップ変化しない場合に次フェーズへ進める。

        Returns:
            新しい武器フェーズキー（コールバック側が set_params を呼ぶ）、変化なければ None。
        """
        if self.is_final_weapon_phase:
            return None

        if self.is_transition:
            # 遷移フェーズ: transition_steps 経過で次のフェーズへ自動進行
            WEAPON_PHASES, _ = _get_weapon_phases()
            transition_steps = WEAPON_PHASES.get(self.current_phase_key, {}).get(
                "transition_steps", 2_000_000
            )
            if self.get_transition_elapsed(num_timesteps) >= transition_steps:
                return self._advance(num_timesteps, reason="transition_complete")
            return None

        # 固定フェーズ: カリキュラム最大到達フェーズが更新された時のみタイマーリセット
        # rollback でカリキュラムフェーズが下がってもタイマーはリセットしない
        # （Phase 10/11 振動でタイマーが永遠にリセットされるのを防ぐ）
        curr = self._curriculum.current_phase
        if curr > self._max_curriculum_phase:
            self._max_curriculum_phase = curr
            self._stagnation_start_step = num_timesteps
            return None
        if num_timesteps - self._stagnation_start_step >= self._stagnation_steps:
            return self._advance(num_timesteps, reason="curriculum_stagnation")
        return None

    def _advance(self, num_timesteps: int, reason: str = "") -> str:
        old_key = self.current_phase_key
        self._weapon_phase_seq_idx += 1
        new_key = self.current_phase_key
        self._phase_start_step = num_timesteps

        print(
            f"[WeaponPhaseAuto] 武器フェーズ昇格: {old_key} -> {new_key} ({reason})"
        )

        # カリキュラム降格（state + enemy params 適用は rollback_fn に委譲）
        self._rollback_fn(f"weapon_phase_advanced_to_{new_key}")

        # 降格後のフェーズを記録し、停滞タイマーをリセット（固定フェーズ用）
        # _max_curriculum_phase もリセット（新しい武器フェーズで停滞計測をやり直す）
        self._last_curriculum_phase = self._curriculum.current_phase
        self._max_curriculum_phase = self._curriculum.current_phase
        self._stagnation_start_step = num_timesteps

        return new_key

    # ------------------------------------------------------------------ #
    # BaseStateModule interface                                            #
    # ------------------------------------------------------------------ #

    def get_wandb_metrics(self) -> dict:
        """BaseStateModule インターフェース実装。stagnation_countdown は -1（不明）を返す。
        正確な値が必要な場合は get_wandb_metrics_with_step() を使うこと。
        """
        return self.get_wandb_metrics_with_step(num_timesteps=self._stagnation_start_step)

    def get_wandb_metrics_with_step(self, num_timesteps: int) -> dict:
        """num_timesteps を受け取って stagnation_countdown を正確に計算するメトリクスを返す。"""
        WEAPON_PHASES, _ = _get_weapon_phases()
        phase_key = self.current_phase_key
        phase_idx = self._weapon_phase_seq_idx
        # weapon_curriculum/use_weapon_num: 現フェーズの allowed_weapon_types の武器種類数
        use_weapon_num = len(WEAPON_PHASES.get(phase_key, {}).get("allowed_weapon_types", []))
        # weapon_curriculum/phase_stagnation_countdown: 遷移フェーズは -1、固定フェーズは残りステップ数
        if self.is_transition:
            stagnation_countdown = -1
        else:
            stagnation_countdown = max(
                0, self._stagnation_steps - (num_timesteps - self._stagnation_start_step)
            )
        return {
            "weapon_auto/phase_seq_idx": phase_idx,
            "weapon_auto/phase_key": phase_key,
            "weapon_curriculum/phase_idx": phase_idx,
            "weapon_curriculum/use_weapon_num": use_weapon_num,
            "weapon_curriculum/phase_stagnation_countdown": stagnation_countdown,
        }

    def export_state(self) -> dict:
        return {
            "weapon_phase_seq_idx": self._weapon_phase_seq_idx,
            "phase_start_step": self._phase_start_step,
            "max_curriculum_phase": self._max_curriculum_phase,
            "stagnation_start_step": self._stagnation_start_step,
        }

    def import_state(self, state: dict) -> None:
        self._weapon_phase_seq_idx = int(state.get("weapon_phase_seq_idx", 0))
        self._phase_start_step = int(state.get("phase_start_step", 0))
        self._max_curriculum_phase = int(
            state.get("max_curriculum_phase", self._curriculum.current_phase)
        )
        self._stagnation_start_step = int(state.get("stagnation_start_step", 0))
        self._state_restored = True  # 復元済みフラグを立てる

    def save_status(self, path=None) -> None:
        pass  # 状態永続化は train_status_{step}_steps.json に委譲する

