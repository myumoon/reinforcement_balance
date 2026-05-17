"""Survivors専用の訓練メトリクス出力callback。"""

import math

from stable_baselines3.common.callbacks import BaseCallback

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


class SurvivorsMetricsCallback(BaseCallback):
    """Survivorsウィンドウ統計とエピソード単位ログを W&B へ出力する。

    n_envs > 1 対応: 全 env の infos を集計してウィンドウ統計を計算する。
    エピソードログは env ごとに独立してトラッキングし、done 時に出力する。

    Args:
        log_freq:   ウィンドウ統計を出力するステップ間隔
        frame_skip: train.py の --frame-skip と同じ値（kills推定に使用）
    """

    def __init__(self, log_freq: int = 5_000, frame_skip: int = 1):
        super().__init__(verbose=0)
        self.log_freq = log_freq
        self.frame_skip = frame_skip
        self._last_log = 0
        self._ep_per_env: list[dict] = []
        self._reset_window()

    def _reset_window(self) -> None:
        self._samples = 0
        self._hp_sum = 0.0
        self._hp_min: float | None = None
        self._nearest_gem_sum = 0.0
        self._nearest_gem_count = 0
        self._nearest_enemy_sum = 0.0
        self._nearest_enemy_count = 0
        self._contact_sum = 0.0
        self._contact_nonzero_steps = 0
        self._enemy_count_sum = 0.0
        self._xp_sum = 0.0
        self._done_count = 0
        self._speed_sum = 0.0
        self._stationary_steps = 0
        self._wall_near_steps = 0

    def _new_ep_state(self) -> dict:
        return {
            "base_reward": 0.0,
            "shaped_reward": 0.0,
            "damage": 0.0,
            "hp_sum": 0.0,
            "hp_min": None,
            "hp_steps": 0,
            "gem_pickups_est": 0,
            "kills_est": 0,
            "prev_xp": None,
            "steps": 0,
        }

    def _on_training_start(self) -> None:
        n = self.training_env.num_envs
        self._ep_per_env = [self._new_ep_state() for _ in range(n)]

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        dones = self.locals["dones"]
        n = len(infos)

        # resume 直後などで _ep_per_env が未初期化の場合は補完
        if len(self._ep_per_env) != n:
            self._ep_per_env = [self._new_ep_state() for _ in range(n)]

        for i, (info, done) in enumerate(zip(infos, dones)):
            ep = self._ep_per_env[i]
            ep["steps"] += 1
            self._samples += 1

            # HP
            hp = info.get("player_hp")
            if hp is not None:
                hp = float(hp)
                self._hp_sum += hp
                ep["hp_sum"] += hp
                ep["hp_steps"] += 1
                self._hp_min = hp if self._hp_min is None else min(self._hp_min, hp)
                ep["hp_min"] = hp if ep["hp_min"] is None else min(ep["hp_min"], hp)

            # 距離系
            nearest_gem = info.get("nearest_gem_distance")
            if nearest_gem is not None:
                self._nearest_gem_sum += float(nearest_gem)
                self._nearest_gem_count += 1
            nearest_enemy = info.get("nearest_enemy_distance")
            if nearest_enemy is not None:
                self._nearest_enemy_sum += float(nearest_enemy)
                self._nearest_enemy_count += 1

            contact = float(info.get("contact_enemy_count", 0.0) or 0.0)
            self._contact_sum += contact
            if contact > 0:
                self._contact_nonzero_steps += 1

            self._enemy_count_sum += float(info.get("observed_enemy_count", 0.0) or 0.0)
            self._xp_sum += float(info.get("xp_progress", 0.0) or 0.0)

            # 行動品質
            speed = float(info.get("move_speed", 0.0) or 0.0)
            self._speed_sum += speed
            if info.get("is_stationary"):
                self._stationary_steps += 1
            if info.get("is_wall_near"):
                self._wall_near_steps += 1

            # エピソード報酬追跡
            base_r = float(info.get("base_reward", 0.0) or 0.0)
            shaped_r = float(info.get("shaped_reward", 0.0) or 0.0)
            hp_pen = float(info.get("hp_penalty", 0.0) or 0.0)
            ep["base_reward"] += base_r
            ep["shaped_reward"] += shaped_r
            ep["damage"] += abs(hp_pen)

            # Gem pickup 推定: XP の増加をカウント（レベルアップは XP 低下で無視）
            xp_now = float(info.get("xp_progress", 0.0) or 0.0)
            if ep["prev_xp"] is not None and xp_now > ep["prev_xp"] + 0.005:
                ep["gem_pickups_est"] += 1
            ep["prev_xp"] = xp_now

            # Kill 推定: KillReward=2.0 相当のスパイクを検出
            alive_step = 0.001 * self.frame_skip
            if base_r - alive_step >= 1.9:
                ep["kills_est"] += max(1, int((base_r - alive_step + 0.05) / 2.0))

            # エピソード終了時: per-episode ログ
            if done:
                self._done_count += 1
                ep_info = info.get("episode")
                ep_len = int(ep_info["l"]) if ep_info else ep["steps"]
                is_truncated = bool(info.get("TimeLimit.truncated", False))

                if _WANDB_AVAILABLE and wandb.run:
                    ep_base = ep["base_reward"]
                    ep_shaped = ep["shaped_reward"]
                    wandb.log({
                        "episode/length":          ep_len,
                        "episode/base_reward":     ep_base,
                        "episode/shaped_reward":   ep_shaped,
                        "episode/shaping_ratio":   abs(ep_shaped) / max(abs(ep_base), 1e-8),
                        "episode/hp_mean":         ep["hp_sum"] / max(ep["hp_steps"], 1),
                        "episode/hp_min":          ep["hp_min"] if ep["hp_min"] is not None else 0.0,
                        "episode/damage_taken":    ep["damage"],
                        "episode/gem_pickups_est": ep["gem_pickups_est"],
                        "episode/kills_est":       ep["kills_est"],
                        "episode/terminated":      int(not is_truncated),
                        "episode/truncated":       int(is_truncated),
                        "global_step":             self.num_timesteps,
                    }, step=self.num_timesteps)

                self._ep_per_env[i] = self._new_ep_state()

        # ウィンドウ統計ログ
        if self.num_timesteps - self._last_log < self.log_freq or self._samples <= 0:
            return True

        self._last_log = self.num_timesteps
        payload = {
            "survivors/player_hp_mean":              self._hp_sum / max(self._samples, 1),
            "survivors/player_hp_min":               self._hp_min,
            "survivors/nearest_gem_distance_mean":   self._nearest_gem_sum / max(self._nearest_gem_count, 1),
            "survivors/nearest_enemy_distance_mean": self._nearest_enemy_sum / max(self._nearest_enemy_count, 1),
            "survivors/contact_enemy_count_mean":    self._contact_sum / max(self._samples, 1),
            "survivors/observed_enemy_count_mean":   self._enemy_count_sum / max(self._samples, 1),
            "survivors/xp_progress_mean":            self._xp_sum / max(self._samples, 1),
            "survivors/episodes_per_window":         self._done_count,
            "behavior/move_speed_mean":              self._speed_sum / max(self._samples, 1),
            "behavior/stationary_ratio":             self._stationary_steps / max(self._samples, 1),
            "behavior/wall_near_ratio":              self._wall_near_steps / max(self._samples, 1),
            "behavior/contact_ratio":                self._contact_nonzero_steps / max(self._samples, 1),
        }
        print(
            "[INFO] Survivors metrics: "
            f"hp_mean={payload['survivors/player_hp_mean']:.3f}, "
            f"hp_min={(payload['survivors/player_hp_min'] if payload['survivors/player_hp_min'] is not None else 0.0):.3f}, "
            f"gem_dist={payload['survivors/nearest_gem_distance_mean']:.2f}, "
            f"speed_mean={payload['behavior/move_speed_mean']:.3f}, "
            f"stationary={payload['behavior/stationary_ratio']:.2f}, "
            f"episodes={self._done_count}"
        )
        if _WANDB_AVAILABLE and wandb.run:
            payload["global_step"] = self.num_timesteps
            wandb.log(payload, step=self.num_timesteps)
        self._reset_window()
        return True


class ActionDistributionCallback(BaseCallback):
    """行動分布とエントロピーを追跡するコールバック。

    Survivors の 9方向行動（0=北 〜 7=北西、8=静止）に対して
    各アクションの選択率とエントロピーを W&B へ出力する。
    n_envs > 1 対応: 全 env のアクションを集計する。

    Args:
        n_actions: アクション数（デフォルト: 9）
        log_freq:  出力間隔ステップ数
    """

    _ACTION_NAMES = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "idle"]

    def __init__(self, n_actions: int = 9, log_freq: int = 5_000):
        super().__init__(verbose=0)
        self.n_actions = n_actions
        self.log_freq = log_freq
        self._last_log = 0
        self._counts = [0] * n_actions
        self._total = 0

    def _on_step(self) -> bool:
        actions = self.locals["actions"]
        for action in actions:
            a = int(action)
            if 0 <= a < self.n_actions:
                self._counts[a] += 1
        self._total += len(actions)

        if self.num_timesteps - self._last_log < self.log_freq or self._total <= 0:
            return True

        self._last_log = self.num_timesteps
        total = max(self._total, 1)
        probs = [c / total for c in self._counts]
        entropy = -sum(p * math.log(p + 1e-10) for p in probs)
        entropy_max = math.log(self.n_actions)

        payload: dict = {
            "behavior/action_entropy":          entropy,
            "behavior/action_entropy_ratio":    entropy / entropy_max if entropy_max > 0 else 0.0,
            "behavior/stationary_action_ratio": probs[8] if self.n_actions > 8 else 0.0,
        }
        names = self._ACTION_NAMES[:self.n_actions]
        for i, (name, prob) in enumerate(zip(names, probs)):
            payload[f"behavior/action_{i}_{name}"] = prob

        if _WANDB_AVAILABLE and wandb.run:
            payload["global_step"] = self.num_timesteps
            wandb.log(payload, step=self.num_timesteps)

        self._counts = [0] * self.n_actions
        self._total = 0
        return True


class SurvivorsCurriculumProgressMetricsCallback(BaseCallback):
    """Survivorsカリキュラムの進捗・停滞判断に必要なwindow統計をW&Bへ出力する。"""

    def __init__(self, curriculum_cb, log_freq: int = 5_000):
        super().__init__(verbose=0)
        self._curriculum_cb = curriculum_cb
        self.log_freq = max(1, log_freq)
        self._last_log = 0

    def _on_step(self) -> bool:
        should_log = any(self.locals["dones"]) or self.num_timesteps - self._last_log >= self.log_freq
        if not should_log or not (_WANDB_AVAILABLE and wandb.run):
            return True

        payload = self._curriculum_cb.get_wandb_progress_metrics()
        filtered = {k: v for k, v in payload.items() if v is not None}
        filtered["global_step"] = self.num_timesteps
        wandb.log(filtered, step=self.num_timesteps)
        self._last_log = self.num_timesteps
        return True
