"""Survivors deterministic 評価rollout コールバック。"""

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

# active_score 計算で使う定数（C++ Source of Truth と合わせる）
_KILL_REWARD = 2.0


class SurvivorsEvalCallback(BaseCallback):
    """deterministic policy で評価rolloutを実行し、W&B と SB3 logger へ記録する。

    UE5 は1インスタンスのため、訓練環境をそのまま評価に転用する。
    ``_on_rollout_end``（rollout収集後・gradient更新前）でトリガーされ、
    ``eval_freq`` ステップごとに ``n_eval_episodes`` エピソードを
    ``deterministic=True`` で実行する。

    評価中は VecNormalize の training フラグを False にして統計更新を止める。
    評価後は model._last_obs を最後の auto-reset obs で更新し、
    次の rollout collection が clean state から始まれるよう保証する。

    Args:
        eval_freq:        評価間隔（timesteps）。0 で無効。
        n_eval_episodes:  評価エピソード数
        frame_skip:       train.py の --frame-skip 値（active_score・kills推定に使用）
        alive_reward:     alive_reward 値（active_score 計算に使用）
        verbose:          1 で評価完了時にコンソールへ要約を表示
    """

    def __init__(
        self,
        eval_freq: int = 50_000,
        n_eval_episodes: int = 5,
        frame_skip: int = 1,
        alive_reward: float = 0.001,
        verbose: int = 1,
    ):
        super().__init__(verbose=verbose)
        self.eval_freq = max(1, eval_freq)
        self.n_eval_episodes = max(1, n_eval_episodes)
        self.frame_skip = frame_skip
        self.alive_reward = alive_reward
        self._last_eval_step: int = 0

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if self.num_timesteps - self._last_eval_step < self.eval_freq:
            return

        self._last_eval_step = self.num_timesteps
        env = self.training_env
        model = self.model

        # 評価中は VecNormalize の running stats を更新しない
        was_training = getattr(env, "training", None)
        if was_training is not None:
            env.training = False

        # 最初の eval episode 用に env をリセット
        obs = env.reset()
        episode_results: list[dict] = []

        for _ in range(self.n_eval_episodes):
            done = np.array([False])
            lstm_states = None
            episode_starts = np.ones((env.num_envs,), dtype=bool)

            ep_base = 0.0
            ep_shaped = 0.0
            ep_damage = 0.0
            ep_hp: list[float] = []
            ep_gem_pickups = 0
            ep_kills = 0
            ep_prev_xp: float | None = None
            ep_xp_sum = 0.0
            ep_gem_dist_sum = 0.0
            ep_gem_dist_count = 0
            ep_enemy_dist_sum = 0.0
            ep_enemy_dist_count = 0
            ep_contact_sum = 0.0
            ep_speed_sum = 0.0
            ep_stationary_steps = 0
            ep_steps = 0
            is_truncated = False

            while not done[0]:
                action, lstm_states = model.predict(
                    obs,
                    state=lstm_states,
                    episode_start=episode_starts,
                    deterministic=True,
                )
                obs, _reward, done, info = env.step(action)
                # done 後の obs は DummyVecEnv の auto-reset obs（次エピソード開始点）
                episode_starts = done
                ep_steps += 1

                si = info[0] if info else {}

                base_r = float(si.get("base_reward", 0.0) or 0.0)
                ep_base += base_r
                ep_shaped += float(si.get("shaped_reward", 0.0) or 0.0)
                ep_damage += abs(float(si.get("hp_penalty", 0.0) or 0.0))

                hp = si.get("player_hp")
                if hp is not None:
                    ep_hp.append(float(hp))

                xp = float(si.get("xp_progress", 0.0) or 0.0)
                ep_xp_sum += xp
                if ep_prev_xp is not None and xp > ep_prev_xp + 0.005:
                    ep_gem_pickups += 1
                ep_prev_xp = xp

                # Kill推定: KillReward=2.0 相当の報酬スパイクをカウント
                alive_step = self.alive_reward * self.frame_skip
                if base_r - alive_step >= 1.9:
                    ep_kills += max(1, int((base_r - alive_step + 0.05) / _KILL_REWARD))

                gd = si.get("nearest_gem_distance")
                if gd is not None:
                    ep_gem_dist_sum += float(gd)
                    ep_gem_dist_count += 1
                ed = si.get("nearest_enemy_distance")
                if ed is not None:
                    ep_enemy_dist_sum += float(ed)
                    ep_enemy_dist_count += 1

                ep_contact_sum += float(si.get("contact_enemy_count", 0.0) or 0.0)
                ep_speed_sum += float(si.get("move_speed", 0.0) or 0.0)
                if si.get("is_stationary"):
                    ep_stationary_steps += 1

                if done[0]:
                    is_truncated = bool(si.get("TimeLimit.truncated", False))

            # active_score: alive_reward 分を除いた実績スコア（カリキュラムと同定義）
            alive_total = self.alive_reward * self.frame_skip * ep_steps
            active_score = max(0.0, ep_base - alive_total)

            episode_results.append({
                "ep_length":      ep_steps,
                "active_score":   active_score,
                "base_reward":    ep_base,
                "shaped_reward":  ep_shaped,
                "hp":             float(np.mean(ep_hp)) if ep_hp else 0.0,
                "hp_min":         float(np.min(ep_hp)) if ep_hp else 0.0,
                "damage_taken":   ep_damage,
                "gem_pickups":    ep_gem_pickups,
                "kills":          ep_kills,
                "xp_progress":    ep_xp_sum / max(ep_steps, 1),
                "gem_dist":       ep_gem_dist_sum / max(ep_gem_dist_count, 1),
                "enemy_dist":     ep_enemy_dist_sum / max(ep_enemy_dist_count, 1),
                "contact":        ep_contact_sum / max(ep_steps, 1),
                "move_speed":     ep_speed_sum / max(ep_steps, 1),
                "stationary":     ep_stationary_steps / max(ep_steps, 1),
                "terminated":     int(not is_truncated),
            })
            # obs は done 時の auto-reset obs。次 episode はここから開始するので reset() 不要。

        # VecNormalize を訓練モードに戻す
        if was_training is not None:
            env.training = was_training

        # model._last_obs を更新（最後の episode の auto-reset obs を次 rollout の起点に）
        model._last_obs = obs
        model._last_episode_starts = np.ones((env.num_envs,), dtype=bool)

        metrics = self._aggregate(episode_results)
        self._log_results(metrics)

    def _aggregate(self, results: list[dict]) -> dict:
        scalar_keys = [
            "ep_length", "active_score", "base_reward", "shaped_reward",
            "hp", "hp_min", "damage_taken",
            "gem_pickups", "kills", "xp_progress",
            "gem_dist", "enemy_dist", "contact",
            "move_speed", "stationary",
        ]
        agg: dict = {}
        for k in scalar_keys:
            vals = [r[k] for r in results]
            agg[k] = float(np.mean(vals))
        agg["terminated_ratio"] = float(np.mean([r["terminated"] for r in results]))
        agg["n_episodes"] = len(results)
        return agg

    def _log_results(self, metrics: dict) -> None:
        payload = {
            "eval/ep_length":        metrics["ep_length"],
            "eval/active_score":     metrics["active_score"],
            "eval/base_reward":      metrics["base_reward"],
            "eval/shaped_reward":    metrics["shaped_reward"],
            "eval/hp_mean":          metrics["hp"],
            "eval/hp_min":           metrics["hp_min"],
            "eval/damage_taken":     metrics["damage_taken"],
            "eval/gem_pickups_est":  metrics["gem_pickups"],
            "eval/kills_est":        metrics["kills"],
            "eval/xp_progress":      metrics["xp_progress"],
            "eval/gem_dist":         metrics["gem_dist"],
            "eval/enemy_dist":       metrics["enemy_dist"],
            "eval/contact":          metrics["contact"],
            "eval/move_speed":       metrics["move_speed"],
            "eval/stationary_ratio": metrics["stationary"],
            "eval/terminated_ratio": metrics["terminated_ratio"],
            "eval/n_episodes":       metrics["n_episodes"],
        }

        if self.verbose >= 1:
            print(
                f"\n[Eval] step={self.num_timesteps:,} | "
                f"active_score={metrics['active_score']:.1f} | "
                f"ep_len={metrics['ep_length']:.0f} | "
                f"gem={metrics['gem_pickups']:.1f} | "
                f"kill={metrics['kills']:.1f} | "
                f"hp={metrics['hp']:.3f} | "
                f"stationary={metrics['stationary']:.2f} | "
                f"n_ep={metrics['n_episodes']}"
            )

        if _WANDB_AVAILABLE and wandb.run:
            payload["global_step"] = self.num_timesteps
            wandb.log(payload, step=self.num_timesteps)

        if getattr(self, "model", None) is not None:
            for k, v in payload.items():
                self.logger.record(k, v)
