import json
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback


class CurriculumCallback(BaseCallback):
    """直近 window エピソードの active_score 平均が threshold を超えたら難易度を上げる。

    active_score = base_reward_ep - alive_reward * frame_skip * ep_len
    生存ボーナスを除いた能動的行動（撃破・収集）による実質スコア。
    """

    def __init__(self, raw_env, frame_skip: int = 1, window: int = 20,
                 threshold: float = 5.0, alive_reward: float = 0.001,
                 status_path=None, verbose: int = 0):
        super().__init__(verbose)
        self._raw_env = raw_env
        self.frame_skip = frame_skip
        self.window = window
        self.threshold = threshold
        self.alive_reward = alive_reward
        self._status_path = status_path
        self._scores = []
        self._stage = 0
        self._ep_base = 0.0

    def _on_step(self) -> bool:
        info = self.locals["infos"][0]
        self._ep_base += info.get("base_reward", 0.0)

        if "episode" in info:
            ep_len = info["episode"]["l"]
            alive_total = self.alive_reward * self.frame_skip * ep_len
            score = max(0.0, self._ep_base - alive_total)
            self._scores.append(score)
            self._ep_base = 0.0

            if len(self._scores) >= self.window:
                mean = sum(self._scores[-self.window:]) / self.window
                if self._status_path is not None:
                    status = {
                        "timestep": self.num_timesteps,
                        "curriculum_stage": self._stage,
                        "active_score_mean": round(mean, 4),
                        "threshold": self.threshold,
                        "episodes_in_window": min(len(self._scores), self.window),
                    }
                    Path(self._status_path).write_text(
                        json.dumps(status, ensure_ascii=False, indent=2)
                    )
                if mean >= self.threshold:
                    self._advance_stage(mean)
        return True

    def _advance_stage(self, mean: float):
        self._stage += 1
        self._scores.clear()
        enemies = min(6 + self._stage, 20)
        speed   = min(1.0 + self._stage * 0.2, 3.0)
        spawn   = max(5.0 - self._stage * 0.5, 2.0)
        ok = self._raw_env.set_params(
            MaxActiveEnemies=enemies,
            EnemySpeedMult=speed,
            SpawnInterval=spawn,
        )
        if ok:
            print(f"[Curriculum] Stage {self._stage} に昇格 "
                  f"(active_score_mean={mean:.3f}) — "
                  f"敵数={enemies}, 速度×{speed:.1f}, スポーン{spawn:.1f}s")
        else:
            print(f"[Curriculum] Stage {self._stage} 昇格試行 — /params 更新失敗")
