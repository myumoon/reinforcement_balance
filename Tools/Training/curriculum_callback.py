"""VSスタイル7段階カリキュラムコールバック。

各フェーズで直近 window エピソードの active_score 平均が
フェーズ閾値 × threshold_mult に達したら次フェーズへ昇格する。

  active_score = base_reward_ep - alive_reward * frame_skip * ep_len
  生存ボーナスを除いた撃破・収集による実質スコア。

フェーズ設計 (Mad Forest 準拠):
  Phase 0: バット入門   -- 2敵,  速度x0.5, TypeId<=0 (Bat)
  Phase 1: 雑魚2種     -- 4敵,  速度x0.7, TypeId<=2 (+ Zombie/Skeleton)
  Phase 2: 速敵追加    -- 6敵,  速度x0.9, TypeId<=4 (+ Ghost/Werewolf)
  Phase 3: 標準密度    -- 8敵,  速度x1.0, TypeId<=6 (+ Mummy/Plant)
  Phase 4: 高密度      -- 12敵, 速度x1.1, TypeId<=9 (全通常敵)
  Phase 5: 時間強化    -- 15敵, 速度x1.2, TimeScaling ON
  Phase 6: フルゲーム  -- 20敵, 速度x1.5, GiantBat込み (最終段)
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from stable_baselines3.common.callbacks import BaseCallback


@dataclass(frozen=True)
class _Phase:
    name: str
    max_enemies: int
    speed_mult: float
    spawn_rate_mult: float
    max_enemy_type_id: int
    time_scaling: bool
    threshold: Optional[float]  # None = 最終段（昇格なし）


PHASES: list[_Phase] = [
    _Phase("バット入門",  2,  0.5, 0.5,  0, False,  3.0),
    _Phase("雑魚2種",    4,  0.7, 0.7,  2, False,  5.0),
    _Phase("速敵追加",   6,  0.9, 1.0,  4, False,  7.0),
    _Phase("標準密度",   8,  1.0, 1.0,  6, False,  9.0),
    _Phase("高密度",    12,  1.1, 1.2,  9, False, 10.0),
    _Phase("時間強化",  15,  1.2, 1.5,  9, True,  11.0),
    _Phase("フルゲーム", 20, 1.5, 2.0, 10, True,   None),
]


class CurriculumCallback(BaseCallback):
    """VSスタイル7段階カリキュラム。

    Args:
        raw_env:        SurvivorsUE5Env インスタンス（set_params を呼ぶ）
        frame_skip:     train.py の --frame-skip と同じ値
        window:         昇格判定に使うエピソード数（推奨: 20）
        threshold_mult: 各フェーズ閾値への乗数（1.0 = デフォルト難易度）
        alive_reward:   生存ボーナス係数（train.py の AliveReward と合わせる）
        status_path:    curriculum_status.json の出力先パス
    """

    def __init__(
        self,
        raw_env,
        frame_skip: int = 1,
        window: int = 20,
        threshold_mult: float = 1.0,
        alive_reward: float = 0.001,
        status_path=None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self._raw_env = raw_env
        self.frame_skip = frame_skip
        self.window = window
        self.threshold_mult = threshold_mult
        self.alive_reward = alive_reward
        self._status_path = status_path
        self._scores: list[float] = []
        self._phase_idx = 0
        self._ep_base = 0.0

    def _on_training_start(self) -> None:
        self._apply_phase(PHASES[0], initial=True)

    def _on_step(self) -> bool:
        info = self.locals["infos"][0]
        self._ep_base += info.get("base_reward", 0.0)

        if "episode" not in info:
            return True

        ep_len = info["episode"]["l"]
        alive_total = self.alive_reward * self.frame_skip * ep_len
        score = max(0.0, self._ep_base - alive_total)
        self._scores.append(score)
        self._ep_base = 0.0

        if len(self._scores) < self.window:
            return True

        mean = sum(self._scores[-self.window:]) / self.window
        phase = PHASES[self._phase_idx]
        effective_threshold = (phase.threshold or float("inf")) * self.threshold_mult
        self._save_status(mean, effective_threshold)

        if phase.threshold is not None and mean >= effective_threshold:
            self._advance_phase(mean, effective_threshold)

        return True

    def _advance_phase(self, mean: float, threshold: float) -> None:
        prev_name = PHASES[self._phase_idx].name
        self._phase_idx = min(self._phase_idx + 1, len(PHASES) - 1)
        self._scores.clear()
        next_phase = PHASES[self._phase_idx]
        print(
            f"\n[Curriculum] Phase {self._phase_idx} 昇格: "
            f"{prev_name} -> {next_phase.name} "
            f"(score={mean:.3f} >= {threshold:.1f})"
        )
        self._apply_phase(next_phase)
        self._log_wandb(mean, next_phase)

    def _apply_phase(self, phase: _Phase, initial: bool = False) -> None:
        ok = self._raw_env.set_params(
            MaxActiveEnemies=phase.max_enemies,
            EnemySpeedMult=phase.speed_mult,
            SpawnRateMult=phase.spawn_rate_mult,
            MaxEnemyTypeId=phase.max_enemy_type_id,
            TimeScalingEnabled=phase.time_scaling,
        )
        label = "初期設定" if initial else f"Phase {self._phase_idx}"
        status = "適用" if ok else "失敗 (/params 更新エラー)"
        print(
            f"[Curriculum] {label} {status}: {phase.name} -- "
            f"敵数={phase.max_enemies}, 速度x{phase.speed_mult:.1f}, "
            f"スポーンx{phase.spawn_rate_mult:.1f}, "
            f"TypeId<={phase.max_enemy_type_id}, "
            f"TimeScaling={'ON' if phase.time_scaling else 'OFF'}"
        )

    def _save_status(self, mean: float, threshold: float) -> None:
        if self._status_path is None:
            return
        phase = PHASES[self._phase_idx]
        status = {
            "timestep": self.num_timesteps,
            "phase_idx": self._phase_idx,
            "phase_name": phase.name,
            "active_score_mean": round(mean, 4),
            "threshold": round(threshold, 4),
            "episodes_in_window": min(len(self._scores), self.window),
            "params": {
                "max_enemies": phase.max_enemies,
                "speed_mult": phase.speed_mult,
                "spawn_rate_mult": phase.spawn_rate_mult,
                "max_enemy_type_id": phase.max_enemy_type_id,
                "time_scaling": phase.time_scaling,
            },
        }
        Path(self._status_path).write_text(
            json.dumps(status, ensure_ascii=False, indent=2)
        )

    def _log_wandb(self, mean: float, phase: _Phase) -> None:
        try:
            import wandb
            if wandb.run:
                wandb.log(
                    {
                        "curriculum/phase_idx": self._phase_idx,
                        "curriculum/phase_name": phase.name,
                        "curriculum/score_mean": mean,
                        "curriculum/max_enemies": phase.max_enemies,
                        "curriculum/speed_mult": phase.speed_mult,
                        "curriculum/spawn_rate_mult": phase.spawn_rate_mult,
                        "curriculum/max_enemy_type_id": phase.max_enemy_type_id,
                        "curriculum/time_scaling": int(phase.time_scaling),
                    },
                    step=self.num_timesteps,
                )
        except ImportError:
            pass
