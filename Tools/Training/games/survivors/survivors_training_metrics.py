"""Survivors専用の訓練メトリクス出力callback。"""

from stable_baselines3.common.callbacks import BaseCallback

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


class SurvivorsMetricsCallback(BaseCallback):
    def __init__(self, log_freq: int = 5_000):
        super().__init__(verbose=0)
        self.log_freq = log_freq
        self._last_log = 0
        self._reset()

    def _reset(self) -> None:
        self._samples = 0
        self._hp_sum = 0.0
        self._hp_min = None
        self._nearest_gem_sum = 0.0
        self._nearest_gem_count = 0
        self._nearest_enemy_sum = 0.0
        self._nearest_enemy_count = 0
        self._contact_sum = 0.0
        self._enemy_count_sum = 0.0
        self._xp_sum = 0.0
        self._done_count = 0

    def _on_step(self) -> bool:
        info = self.locals["infos"][0]
        self._samples += 1
        hp = info.get("player_hp")
        if hp is not None:
            hp = float(hp)
            self._hp_sum += hp
            self._hp_min = hp if self._hp_min is None else min(self._hp_min, hp)
        nearest_gem = info.get("nearest_gem_distance")
        if nearest_gem is not None:
            self._nearest_gem_sum += float(nearest_gem)
            self._nearest_gem_count += 1
        nearest_enemy = info.get("nearest_enemy_distance")
        if nearest_enemy is not None:
            self._nearest_enemy_sum += float(nearest_enemy)
            self._nearest_enemy_count += 1
        self._contact_sum += float(info.get("contact_enemy_count", 0.0) or 0.0)
        self._enemy_count_sum += float(info.get("observed_enemy_count", 0.0) or 0.0)
        self._xp_sum += float(info.get("xp_progress", 0.0) or 0.0)
        if self.locals["dones"][0]:
            self._done_count += 1

        if self.num_timesteps - self._last_log < self.log_freq or self._samples <= 0:
            return True
        self._last_log = self.num_timesteps
        payload = {
            "survivors/player_hp_mean": self._hp_sum / max(self._samples, 1),
            "survivors/player_hp_min": self._hp_min,
            "survivors/nearest_gem_distance_mean": self._nearest_gem_sum / max(self._nearest_gem_count, 1),
            "survivors/nearest_enemy_distance_mean": self._nearest_enemy_sum / max(self._nearest_enemy_count, 1),
            "survivors/contact_enemy_count_mean": self._contact_sum / max(self._samples, 1),
            "survivors/observed_enemy_count_mean": self._enemy_count_sum / max(self._samples, 1),
            "survivors/xp_progress_mean": self._xp_sum / max(self._samples, 1),
            "survivors/episodes_per_window": self._done_count,
        }
        print(
            "[INFO] Survivors metrics: "
            f"hp_mean={payload['survivors/player_hp_mean']:.3f}, "
            f"hp_min={(payload['survivors/player_hp_min'] if payload['survivors/player_hp_min'] is not None else 0.0):.3f}, "
            f"gem_dist={payload['survivors/nearest_gem_distance_mean']:.2f}, "
            f"enemy_dist={payload['survivors/nearest_enemy_distance_mean']:.2f}, "
            f"contact={payload['survivors/contact_enemy_count_mean']:.2f}, "
            f"episodes={self._done_count}"
        )
        if _WANDB_AVAILABLE and wandb.run:
            wandb.log(payload, step=self.num_timesteps)
        self._reset()
        return True


class SurvivorsCurriculumProgressMetricsCallback(BaseCallback):
    """Survivorsカリキュラムの進捗・停滞判断に必要なwindow統計をW&Bへ出力する。"""

    def __init__(self, curriculum_cb, log_freq: int = 5_000):
        super().__init__(verbose=0)
        self._curriculum_cb = curriculum_cb
        self.log_freq = max(1, log_freq)
        self._last_log = 0

    def _on_step(self) -> bool:
        should_log = self.locals["dones"][0] or self.num_timesteps - self._last_log >= self.log_freq
        if not should_log or not (_WANDB_AVAILABLE and wandb.run):
            return True

        payload = self._curriculum_cb.get_wandb_progress_metrics()
        wandb.log({k: v for k, v in payload.items() if v is not None}, step=self.num_timesteps)
        self._last_log = self.num_timesteps
        return True
