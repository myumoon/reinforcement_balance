"""HybridPromotionProbeCallback: Hybrid SPALF の昇格判定を固定 Phase probe に分離するコールバック。

train path での昇格判定を排除し、固定 Phase params の probe episode 結果のみで昇格を判定する。
"""
from __future__ import annotations

import copy

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize

from games.survivors.hybrid_callback import HybridCurriculumSpalfCallback
from games.survivors.param_applier import ParamApplier


class HybridPromotionProbeCallback(BaseCallback):
    """固定 Phase params で probe episodes を実行して昇格判定を行うコールバック。

    probe_freq ごとに current phase の固定パラメータで eval_env を実行し、
    active_score を HybridCurriculumSpalfCallback.on_promotion_probe_results() に渡す。
    昇格判定は train path から完全に分離されており、rollback は発生しない。
    """

    def __init__(
        self,
        hybrid_cb: HybridCurriculumSpalfCallback,
        eval_env,
        probe_freq: int,
        n_probe_episodes: int,
        frame_skip: int,
        alive_reward: float,
        deterministic: bool = True,
        wandb_logger=None,
    ):
        super().__init__(verbose=0)
        self._hybrid_cb = hybrid_cb
        self._eval_env = eval_env
        self.probe_freq = probe_freq
        self.n_probe_episodes = n_probe_episodes
        self.frame_skip = frame_skip
        self.alive_reward = alive_reward
        self.deterministic = deterministic
        self._wandb_logger = wandb_logger
        self._last_probe_step: int = 0

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if self.num_timesteps - self._last_probe_step < self.probe_freq:
            return
        self._last_probe_step = self.num_timesteps
        self._run_probe()

    def _sync_vecnormalize(self) -> None:
        """訓練側 VecNormalize の obs_rms/ret_rms を eval_env へコピーする。"""
        train_vecnorm: VecNormalize | None = None
        cur = self.training_env
        while cur is not None:
            if isinstance(cur, VecNormalize):
                train_vecnorm = cur
                break
            cur = getattr(cur, "venv", None)

        eval_vecnorm: VecNormalize | None = None
        cur = self._eval_env
        while cur is not None:
            if isinstance(cur, VecNormalize):
                eval_vecnorm = cur
                break
            cur = getattr(cur, "venv", None)

        if train_vecnorm is not None and eval_vecnorm is not None:
            eval_vecnorm.obs_rms = copy.deepcopy(train_vecnorm.obs_rms)
            eval_vecnorm.ret_rms = copy.deepcopy(train_vecnorm.ret_rms)

    def _apply_phase_params(self) -> None:
        """current phase の固定 params を eval_env に適用する。"""
        # _KEY_MAP: 内部 key → UE5 API key
        _KEY_MAP = {
            "min_enemies":        "MinActiveEnemies",
            "max_enemies":        "MaxActiveEnemies",
            "speed_mult":         "EnemySpeedMult",
            "spawn_rate_mult":    "SpawnRateMult",
            "max_enemy_type_id":  "MaxEnemyTypeId",
            "enemy_hp_scale":     "EnemyHPScale",
            "enemy_damage_scale": "EnemyDamageScale",
            "time_scaling":       "TimeScalingEnabled",
        }
        phase_params = self._hybrid_cb.get_current_phase_params()
        ue5_params = {_KEY_MAP.get(k, k): v for k, v in phase_params.items()}
        self._eval_env.env_method("set_params", **ue5_params)

    def _run_probe(self) -> None:
        """probe episodes を実行して昇格判定に渡す。"""
        self._sync_vecnormalize()
        self._apply_phase_params()

        was_training = getattr(self._eval_env, "training", None)
        if was_training is not None:
            self._eval_env.training = False

        obs = self._eval_env.reset()
        episode_results: list[dict] = []

        for _ in range(self.n_probe_episodes):
            done = np.array([False])
            lstm_states = None
            episode_starts = np.ones((self._eval_env.num_envs,), dtype=bool)
            ep_base = 0.0
            ep_steps = 0

            while not done[0]:
                action, lstm_states = self.model.predict(
                    obs,
                    state=lstm_states,
                    episode_start=episode_starts,
                    deterministic=self.deterministic,
                )
                obs, _reward, done, info = self._eval_env.step(action)
                episode_starts = done
                ep_steps += 1

                si = info[0] if info else {}
                ep_base += float(si.get("base_reward", 0.0) or 0.0)

            alive_total = self.alive_reward * self.frame_skip * ep_steps
            active_score = max(0.0, ep_base - alive_total)

            episode_results.append({
                "active_score": active_score,
                "ep_len": ep_steps,
                "base_reward": ep_base,
            })

        if was_training is not None:
            self._eval_env.training = was_training

        event = self._hybrid_cb.on_promotion_probe_results(episode_results)

        self._log_wandb(episode_results, event)

    def _log_wandb(self, results: list[dict], event: str | None) -> None:
        if not self._wandb_logger or not self._wandb_logger.enabled:
            return
        scores = [r["active_score"] for r in results]
        ep_lens = [r["ep_len"] for r in results]
        if not scores:
            return

        diag = self._hybrid_cb._curriculum.get_promotion_probe_diagnostics()
        threshold = diag.get("threshold")

        # percentile 計算（numpy 使用）
        scores_arr = np.array(scores, dtype=np.float64)
        score_mean = float(np.mean(scores_arr))
        score_min = float(np.min(scores_arr))
        score_p10 = float(np.percentile(scores_arr, 10))
        score_cv = float(np.std(scores_arr) / max(score_mean, 1e-8)) if score_mean > 0.0 else 0.0

        metrics = {
            "curriculum_probe/active_score_mean": score_mean,
            "curriculum_probe/active_score_min": score_min,
            "curriculum_probe/active_score_p10": score_p10,
            "curriculum_probe/score_cv": score_cv,
            "curriculum_probe/ep_length_mean": float(np.mean(ep_lens)),
            "curriculum_probe/n_episodes": len(results),
            "curriculum_probe/phase_idx": self._hybrid_cb._curriculum.current_phase,
            "curriculum_probe/threshold": threshold,
            "curriculum_probe/promotion_ready": int(bool(diag.get("promotion_ready"))),
            "curriculum_probe/event": int(event == "advance"),
        }
        self._wandb_logger.log(metrics, step=self.num_timesteps)
