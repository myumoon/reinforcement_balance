"""SPALF コールバック（リファクタリング版）。"""
from __future__ import annotations
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from games.survivors.survivors_difficulty import PARAM_BOUNDS
from games.survivors.param_applier import ParamApplier
from games.survivors.state_modules import EpisodeScoreTracker, SpalfStateModule, _PHASE0_PARAMS
_PARAM_BOUNDS = PARAM_BOUNDS
_PARAM_KEYS = list(_PARAM_BOUNDS.keys())
_N_PARAMS = len(_PARAM_KEYS)
class SpalfCallback(BaseCallback):
    """SPALF 連続パラメータカリキュラムコールバック（リファクタリング版）。

    Args:
        raw_env:           SurvivorsEnv インスタンス（n_envs==1 時の set_params 用）
        frame_skip:        train.py の --frame-skip と同じ値
        alive_reward:      生存ボーナス係数
        r_b:               正規化後バッファ平均報酬の境界値
        alpha:             正則化強度 α
        max_score:         スコア正規化スケール
        buffer_size:       バッファサイズ（エピソード数）
        warmup_episodes:   Phase 0 固定で動作する初期エピソード数
        status_path:       spalf_state.json の出力先（None = 保存なし）
        wandb_logger:      WandbLogger インスタンス（None の場合は W&B ログなし）
    """

    def __init__(self, raw_env, frame_skip: int = 4, alive_reward: float = 0.001,
                 r_b: float = 0.1, alpha: float = 0.2, max_score: float = 2250.0,
                 buffer_size: int = 200, warmup_episodes: int = 50,
                 status_path=None, wandb_logger=None, verbose: int = 0):
        super().__init__(verbose)
        self._param_applier = ParamApplier(raw_env)
        self._score_tracker = EpisodeScoreTracker(frame_skip, alive_reward)
        self._spalf = SpalfStateModule(r_b, alpha, max_score, buffer_size,
                                        warmup_episodes, status_path)
        self._wandb_logger = wandb_logger
        if wandb_logger:
            wandb_logger.add_metric_prefix("survivors/")
            wandb_logger.add_metric_prefix("spalf/")
        self._ep_start_param_vec_per_env: list[np.ndarray] = []

    def _on_training_start(self) -> None:
        n = self.training_env.num_envs if self.training_env is not None else 1
        self._param_applier.set_training_env(self.training_env)
        self._score_tracker.reset(n)
        # resume 時は import_state() で _current_params が復元済みのため上書きしない
        if self._spalf._current_params is None:
            self._spalf._current_params = dict(_PHASE0_PARAMS)
        initial_params = self._spalf._current_params
        self._param_applier.apply(initial_params)
        initial_vec = self._spalf.params_to_vec(initial_params)
        self._ep_start_param_vec_per_env = [initial_vec.copy() for _ in range(n)]
        if self._spalf._current_param_vec is None:
            self._spalf._current_param_vec = initial_vec

    def _on_step(self) -> bool:
        n_envs = len(self.locals["infos"])
        if len(self._ep_start_param_vec_per_env) != n_envs:
            initial_vec = self._spalf.params_to_vec(self._spalf._current_params)
            self._ep_start_param_vec_per_env = [initial_vec.copy() for _ in range(n_envs)]
        episode_results = self._score_tracker.process(self.locals["infos"])
        ep_active_scores: list[float] = []
        ep_score_norms: list[float] = []
        ep_has_warmup: bool = False
        for env_idx, active_score, ep_len, _ep_base in episode_results:
            ep_param_vec = self._ep_start_param_vec_per_env[env_idx]
            is_warmup = self._spalf.on_episode_end(env_idx, active_score, ep_param_vec)
            score_norm = active_score / self._spalf._max_score
            ep_active_scores.append(active_score)
            ep_score_norms.append(score_norm)
            if is_warmup:
                ep_has_warmup = True
                continue
            new_params, new_vec = self._spalf.sample_next_params()
            self._param_applier.apply(new_params, env_idx=env_idx)
            self._spalf._current_params = new_params
            self._spalf._current_param_vec = new_vec
            self._ep_start_param_vec_per_env[env_idx] = new_vec.copy()
            if self._spalf._total_episodes % 50 == 0:
                self._spalf.save_status()
        if ep_active_scores:
            self._log_per_step(ep_active_scores, ep_score_norms, ep_has_warmup)
        return True

    def _log_per_step(self, active_scores: list, score_norms: list, has_warmup: bool) -> None:
        if not self._wandb_logger or not self._wandb_logger.enabled:
            return
        metrics = self._spalf.get_wandb_metrics()
        if has_warmup:
            metrics["spalf/mode"] = 0
        metrics.update({
            "survivors/active_score":     sum(active_scores) / len(active_scores),
            "survivors/score_normalized": sum(score_norms) / len(score_norms),
        })
        self._wandb_logger.log(metrics, step=self.num_timesteps)

    # 後方互換 proxy
    def reset_spalf_buffers(self) -> None:
        self._spalf.reset_buffers()

    def get_wandb_progress_metrics(self) -> dict:
        return self._spalf.get_wandb_metrics()

    def export_state(self) -> dict:
        return self._spalf.export_state()

    def import_state(self, state: dict) -> None:
        self._spalf.import_state(state)
        if self._spalf._current_params:
            self._spalf._current_param_vec = self._spalf.params_to_vec(self._spalf._current_params)

    def _save_status(self) -> None:
        self._spalf.save_status()

    # 旧 API 互換（direct wandb アクセス削除、_log_wandb_per_step は _log_per_step に統合済み）
    def _log_wandb_per_step(self, active_scores, score_norms, has_warmup=False) -> None:
        self._log_per_step(active_scores, score_norms, has_warmup)

    # 旧 API 互換: _params_to_vec / _vec_to_params
    def _params_to_vec(self, params: dict) -> np.ndarray:
        return self._spalf.params_to_vec(params)

    def _vec_to_params(self, vec: np.ndarray) -> dict:
        return self._spalf.vec_to_params(vec)
