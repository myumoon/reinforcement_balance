"""VSスタイル6段階カリキュラムコールバック。

各フェーズで直近 window エピソードの active_score 平均が
フェーズ閾値 × threshold_mult に達したら次フェーズへ昇格する。

  active_score = base_reward_ep - alive_reward * frame_skip * ep_len
  生存ボーナスを除いた撃破・収集による実質スコア
  （= kills × KillReward + gems × ItemReward）

設計方針:
  - VS 本家に倣い「敵は常に一定数以上いる」状態を維持する
  - MinActiveEnemies の補充スポーンにより敵数 0 の訓練空白をなくす
  - 初期フェーズは敵を弱く（HPScale=0.5, DamageScale=0.5）して即死を防ぐ
  - 難易度は「敵種構成 → HP/ダメージ倍率 → 密度・速度 → TimeScaling」の順で段階上昇

フェーズ設計:
  Phase 0: 入門        -- Min=4,  Max=6,   TypeId<=1, HP=0.50, Dmg=0.50, Speed=0.8
  Phase 1: Gem回収開始 -- Min=6,  Max=10,  TypeId<=2, HP=0.75, Dmg=0.75, Speed=0.9
  Phase 2: 通常序盤    -- Min=8,  Max=20,  TypeId<=4, HP=1.00, Dmg=1.00, Speed=1.0
  Phase 3: 囲まれ対応  -- Min=12, Max=40,  TypeId<=6, HP=1.25, Dmg=1.25, Speed=1.0
  Phase 4: 群れ対応    -- Min=20, Max=80,  TypeId<=9, HP=1.50, Dmg=1.50, Speed=1.1, TimeScaling ON
  Phase 5: Mad Forest  -- Min=40, Max=150, TypeId<=10, HP=2.00, Dmg=2.00, Speed=1.2, TimeScaling ON
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from stable_baselines3.common.callbacks import BaseCallback


@dataclass(frozen=True)
class _Phase:
    name: str
    min_enemies: int
    max_enemies: int
    speed_mult: float
    spawn_rate_mult: float
    max_enemy_type_id: int
    enemy_hp_scale: float
    enemy_damage_scale: float
    time_scaling: bool
    threshold: Optional[float]  # None = 最終段（昇格なし）


PHASES: list[_Phase] = [
    _Phase("入門",        4,   6,  0.8, 1.0,  1, 0.50, 0.50, False,  30.0),
    _Phase("Gem回収開始", 6,  10,  0.9, 1.5,  2, 0.75, 0.75, False,  60.0),
    _Phase("通常序盤",    8,  20,  1.0, 2.0,  4, 1.00, 1.00, False, 100.0),
    _Phase("囲まれ対応", 12,  40,  1.0, 2.5,  6, 1.25, 1.25, False, 140.0),
    _Phase("群れ対応",   20,  80,  1.1, 3.0,  9, 1.50, 1.50, True,  180.0),
    _Phase("Mad Forest", 40, 150,  1.2, 4.0, 10, 2.00, 2.00, True,   None),
]


class CurriculumCallback(BaseCallback):
    """VSスタイル6段階カリキュラム。

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
            MinActiveEnemies=phase.min_enemies,
            MaxActiveEnemies=phase.max_enemies,
            EnemySpeedMult=phase.speed_mult,
            SpawnRateMult=phase.spawn_rate_mult,
            MaxEnemyTypeId=phase.max_enemy_type_id,
            EnemyHPScale=phase.enemy_hp_scale,
            EnemyDamageScale=phase.enemy_damage_scale,
            TimeScalingEnabled=phase.time_scaling,
        )
        label = "初期設定" if initial else f"Phase {self._phase_idx}"
        status = "適用" if ok else "失敗 (/params 更新エラー)"
        print(
            f"[Curriculum] {label} {status}: {phase.name} -- "
            f"敵数 Min={phase.min_enemies}/Max={phase.max_enemies}, "
            f"速度x{phase.speed_mult:.1f}, スポーンx{phase.spawn_rate_mult:.1f}, "
            f"TypeId<={phase.max_enemy_type_id}, "
            f"HPx{phase.enemy_hp_scale:.2f}, Dmgx{phase.enemy_damage_scale:.2f}, "
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
                "min_enemies": phase.min_enemies,
                "max_enemies": phase.max_enemies,
                "speed_mult": phase.speed_mult,
                "spawn_rate_mult": phase.spawn_rate_mult,
                "max_enemy_type_id": phase.max_enemy_type_id,
                "enemy_hp_scale": phase.enemy_hp_scale,
                "enemy_damage_scale": phase.enemy_damage_scale,
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
                        "curriculum/min_enemies": phase.min_enemies,
                        "curriculum/max_enemies": phase.max_enemies,
                        "curriculum/speed_mult": phase.speed_mult,
                        "curriculum/spawn_rate_mult": phase.spawn_rate_mult,
                        "curriculum/max_enemy_type_id": phase.max_enemy_type_id,
                        "curriculum/enemy_hp_scale": phase.enemy_hp_scale,
                        "curriculum/enemy_damage_scale": phase.enemy_damage_scale,
                        "curriculum/time_scaling": int(phase.time_scaling),
                    },
                    step=self.num_timesteps,
                )
        except ImportError:
            pass
