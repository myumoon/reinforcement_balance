"""Survivors deterministic 評価rollout コールバック。"""

from __future__ import annotations

import copy
import json
from typing import Callable

import numpy as np
from base.base_eval_callback import BaseEvalCallback
from stable_baselines3.common.vec_env import VecNormalize

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

# active_score 計算で使う定数（C++ Source of Truth と合わせる）
_KILL_REWARD = 2.0


def run_survivors_eval_episodes(
    *,
    model,
    env,
    n_eval_episodes: int,
    frame_skip: int,
    alive_reward: float,
    deterministic: bool = True,
) -> tuple[list[dict], dict, object]:
    """Survivors 評価エピソードを実行して結果を返す。

    Args:
        model:            SB3 モデル
        env:              評価用 VecEnv
        n_eval_episodes:  評価エピソード数
        frame_skip:       フレームスキップ数
        alive_reward:     alive_reward 値（active_score 計算に使用）
        deterministic:    deterministic policy を使うか

    Returns:
        (episode_results, aggregate_metrics, last_obs)
        episode_results の各 dict は ep_length キーを持つ。
        last_obs は最終エピソード終了時の obs（n_envs=1 旧互換パスで model._last_obs に使用）。
    """
    was_training = getattr(env, "training", None)
    if was_training is not None:
        env.training = False

    try:
        obs = env.reset()
        episode_results: list[dict] = []

        for _ in range(n_eval_episodes):
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
                    deterministic=deterministic,
                )
                obs, _reward, done, info = env.step(action)
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

                alive_step = alive_reward * frame_skip
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

            alive_total = alive_reward * frame_skip * ep_steps
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

    finally:
        # VecNormalize を元の状態に必ず戻す
        if was_training is not None:
            env.training = was_training

    metrics = _aggregate_eval_results(episode_results)
    return episode_results, metrics, obs


def _aggregate_eval_results(results: list[dict]) -> dict:
    """episode_results を集約して metrics dict を返す。"""
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


class SurvivorsEvalCallback(BaseEvalCallback):
    """deterministic policy で評価rolloutを実行し、W&B と SB3 logger へ記録する。

    eval_env が指定されている場合（n_envs > 1）:
        訓練環境とは独立した専用の VecEnv で評価を実施する。
        model._last_obs / _last_episode_starts には一切触れない。
        評価直前に訓練側 VecNormalize の obs_rms/ret_rms を eval 側へコピーして同期する。

    eval_env が None の場合（n_envs == 1 旧互換モード）:
        self.training_env をそのまま評価に使用する（旧来の動作）。
        評価後に model._last_obs / _last_episode_starts を更新して次 rollout の起点を保証する。

    Args:
        eval_env:             評価専用 VecEnv（n_envs > 1 時）。None なら training_env を使用。
        eval_freq:            評価間隔（timesteps）。0 で無効。
        n_eval_episodes:      評価エピソード数
        frame_skip:           train.py の --frame-skip 値（active_score・kills推定に使用）
        alive_reward:         alive_reward 値（active_score 計算に使用）
        params_provider:      eval 直前に呼び出して phase params を取得する callable。
                              None の場合は _sync_env_params() を使って train env から同期する。
        after_eval_callback:  eval 完了後に (episode_results, metrics) で呼び出す callable。
        verbose:              1 で評価完了時にコンソールへ要約を表示
    """

    def __init__(
        self,
        eval_env=None,
        eval_freq: int = 50_000,
        n_eval_episodes: int = 5,
        frame_skip: int = 1,
        alive_reward: float = 0.001,
        wandb_logger=None,
        params_provider: Callable[[], dict] | None = None,
        after_eval_callback: Callable[[list[dict], dict], None] | None = None,
        verbose: int = 1,
    ):
        super().__init__(
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            wandb_logger=wandb_logger,
            verbose=verbose,
        )
        self.frame_skip = frame_skip
        self.alive_reward = alive_reward
        self.params_provider = params_provider
        self.after_eval_callback = after_eval_callback

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
        cur = self.eval_env
        while cur is not None:
            if isinstance(cur, VecNormalize):
                eval_vecnorm = cur
                break
            cur = getattr(cur, "venv", None)

        if train_vecnorm is not None and eval_vecnorm is not None:
            eval_vecnorm.obs_rms = copy.deepcopy(train_vecnorm.obs_rms)
            eval_vecnorm.ret_rms = copy.deepcopy(train_vecnorm.ret_rms)

    def _sync_shaping(self) -> None:
        """訓練側の shaping_weight を eval_env へ同期する。

        _AnnealingShapingCallback によってアニーリングが進んでいる場合、
        eval env も同じ shaping_weight で評価する必要がある。
        weight が 0 の場合は clear_reward_fn も実行して eval shaped_reward を 0 に揃える。
        """
        if self.eval_env is None:
            return
        weights = self.training_env.env_method("get_shaping_weight")
        if not weights:
            return
        w = float(weights[0])
        self.eval_env.env_method("set_shaping_weight", w)
        if w == 0.0:
            self.eval_env.env_method("clear_reward_fn")

    def _sync_env_params(self) -> None:
        """訓練側の curriculum phase params を eval_env へ同期する。

        curriculum が training env に適用した UE5 パラメータ（敵数・速度等）を
        eval env に転送して、公平な評価条件を保つ。
        """
        if self.eval_env is None:
            return
        params_list = self.training_env.env_method("get_params")
        params_list = [p for p in params_list if p]
        if not params_list:
            print("[WARN] Eval env params sync skipped: training env returned empty params.")
            return

        if len({json.dumps(p, sort_keys=True) for p in params_list}) > 1:
            print("[WARN] training env params differ across envs; using env[0] for eval sync.")

        params = params_list[0]
        self.eval_env.env_method("set_params", **params)

        if self.verbose >= 1:
            summary = ", ".join(
                f"{k}={v}" for k, v in params.items()
                if k in ("MinActiveEnemies", "MaxActiveEnemies", "SpawnRateMult",
                         "MaxEnemyTypeId", "EnemyHPScale", "EnemyDamageScale")
            )
            print(f"[Eval] synced env params: {summary}")

    def _sync_before_eval(self) -> None:
        """評価前の同期処理（BaseEvalCallback のオーバーライド）。

        VecNormalize と shaping は params_provider の有無にかかわらず常に同期する。
        params の同期は params_provider がある場合はそれを使い、ない場合は _sync_env_params() を呼ぶ。

        params_provider がある場合（curriculum-spalf 使用時）:
            training env から現在の全 params（weapon params を含む）を取得し、
            params_provider() の difficulty params で上書きして merge する。
            これにより weapon params が eval env に届かない問題を解消する。
        """
        if self.eval_env is not None:
            # VecNormalize と shaping は常に同期する
            self._sync_vecnormalize()
            self._sync_shaping()

            # params のみ分岐
            if self.params_provider is not None:
                # Step 1: training env から現在の全 params（weapon params 含む）を取得
                merged_params: dict = {}
                try:
                    params_list = self.training_env.env_method("get_params")
                    params_list = [p for p in params_list if p]
                    if params_list:
                        merged_params = dict(params_list[0])
                except Exception as exc:
                    print(f"[Eval] training env から全 params 取得に失敗（weapon params なしで続行）: {exc}")

                # Step 2: params_provider() の difficulty params で上書きして merge
                difficulty_params = self.params_provider()  # 内部 key (min_enemies など)
                from games.survivors.param_applier import ParamApplier
                ue5_difficulty_params = {
                    ParamApplier._KEY_MAP.get(k, k): v for k, v in difficulty_params.items()
                }
                merged_params.update(ue5_difficulty_params)

                # Step 3: merge した params を eval env に送信
                self.eval_env.env_method("set_params", **merged_params)
            else:
                self._sync_env_params()

    def _run_eval_and_log(self) -> None:
        # eval_env が None の場合は training_env を使う（n_envs=1 旧互換）
        use_training_env = self.eval_env is None
        env = self.training_env if use_training_env else self.eval_env
        model = self.model

        episode_results, metrics, last_obs = run_survivors_eval_episodes(
            model=model,
            env=env,
            n_eval_episodes=self.n_eval_episodes,
            frame_skip=self.frame_skip,
            alive_reward=self.alive_reward,
            deterministic=True,
        )

        if use_training_env:
            # n_envs=1 旧互換: model._last_obs を更新して次 rollout の起点を保証
            # last_obs は最終エピソードの done 直後の obs（元実装と同等、追加 reset なし）
            model._last_obs = last_obs
            model._last_episode_starts = np.ones((env.num_envs,), dtype=bool)
        # eval_env 使用時は訓練環境（model._last_obs 等）には一切触れない

        self._log_results(metrics)

        if self.after_eval_callback is not None:
            self.after_eval_callback(episode_results, metrics)

    def _aggregate(self, results: list[dict]) -> dict:
        return _aggregate_eval_results(results)

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

        if self._wandb_logger:
            self._wandb_logger.log(payload, step=self.num_timesteps)

        if getattr(self, "model", None) is not None:
            for k, v in payload.items():
                self.logger.record(k, v)
