"""VSスタイルの段階式カリキュラムコールバック。

各フェーズで直近 window エピソードの active_score 平均が
フェーズ閾値 × threshold_mult に達したら次フェーズへ昇格する。
直近 window のスコアまたはエピソード長が大きく崩れた場合は、
前フェーズへ降格して学習破綻した難易度に留まり続けないようにする。

  active_score = base_reward_ep - alive_reward * frame_skip * ep_len
  生存ボーナスを除いた撃破・収集による実質スコア
  （= kills × KillReward + gems × ItemReward）

設計方針:
  - VS 本家に倣い「敵は常に一定数以上いる」状態を維持する
  - MinActiveEnemies の補充スポーンにより敵数 0 の訓練空白をなくす
  - 初期フェーズは敵を弱く（HPScale=0.5, DamageScale=0.5）して即死を防ぐ
  - 難易度は「敵種構成 → HP/ダメージ倍率 → 密度・速度 → TimeScaling」の順で段階上昇

フェーズ設計:
  - Gem回収フェーズで移動と回収の基礎を作る
  - 通常序盤から包囲入門A/B/B+/Cへ小刻みに上げる
  - 群れ対応もA/B/Cへ分け、密度・速度・TimeScalingを段階的に上げる
  - Mad Forest を最終フェーズとして扱う
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from base.curriculum import (
    CurriculumTuningInput,
    WindowThresholdRecommendationPolicy,
    mean as _mean,
    stdev as _stdev,
)
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
    min_episode_steps: int
    threshold: Optional[float]  # None = 最終段（昇格なし）
    rollback_score_ratio: float = 0.35
    rollback_length_ratio: float = 0.45
    promotion_min_score_ratio: float = 0.85
    promotion_max_score_cv: float = 0.20


PHASES: list[_Phase] = [
    _Phase("入門",              4,   6, 0.8, 1.0,  1, 0.50, 0.50, False,  600,   30.0),
    _Phase("Gem回収開始",       6,  10, 0.9, 1.4,  2, 0.75, 0.75, False, 1200,  100.0),
    _Phase("Gem追従強化",       7,  14, 0.9, 1.7,  3, 0.85, 0.85, False, 1800,  250.0),
    _Phase("通常序盤",          8,  20, 1.0, 2.0,  4, 1.00, 1.00, False, 2400,  800.0),
    _Phase("包囲入門A",         9,  18, 1.0, 2.0,  4, 1.00, 0.80, False, 2400,  950.0),
    _Phase("包囲入門B",        10,  22, 1.0, 2.1,  4, 1.00, 0.90, False, 2400,  940.0),
    _Phase("包囲入門B+",       10,  24, 1.0, 2.15, 4, 1.00, 0.95, False, 2400, 1100.0),
    _Phase("包囲入門C",        10,  25, 1.0, 2.2,  4, 1.05, 1.00, False, 2400, 1150.0),
    _Phase("包囲対応",         12,  30, 1.0, 2.4,  5, 1.10, 1.10, False, 2400, 1350.0),
    _Phase("多敵対応",         14,  40, 1.0, 2.6,  6, 1.20, 1.20, False, 2400, 1550.0),
    _Phase("群れ対応A",        16,  50, 1.0, 2.7,  7, 1.25, 1.25, False, 2400, 1700.0),
    _Phase("群れ対応B",        18,  65, 1.05, 2.8,  8, 1.35, 1.35, True,  2400, 1850.0),
    _Phase("群れ対応C",        20,  80, 1.1, 3.0,  9, 1.50, 1.50, True,  2400, 2000.0),
    _Phase("Mad Forest",       40, 150, 1.2, 4.0, 10, 2.00, 2.00, True,  2400,   None),
]


class CurriculumCallback(BaseCallback):
    """VSスタイル段階式カリキュラム。

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
        self.rollback_patience = 2
        self.rollback_min_episodes = max(5, window // 2)
        self._status_path = status_path
        self._scores: list[float] = []
        self._phase_idx = 0
        self._rollback_bad_windows = 0
        self._ep_base = 0.0
        self._episode_scores: list[float] = []
        self._episode_lengths: list[int] = []
        self._episode_base_rewards: list[float] = []
        self._episode_alive_rewards: list[float] = []
        self._terminated_count = 0
        self._truncated_count = 0
        self._phase_events: list[dict] = []
        self._recommendation_policy = WindowThresholdRecommendationPolicy()

    def _on_training_start(self) -> None:
        self._apply_phase(PHASES[self._phase_idx], initial=True)

    @property
    def is_final_phase(self) -> bool:
        return self._phase_idx >= len(PHASES) - 1

    def export_state(self) -> dict:
        return {
            "phase_idx": self._phase_idx,
            "phase_name": PHASES[self._phase_idx].name,
            "scores_window": self._scores,
            "rollback_bad_windows": self._rollback_bad_windows,
            "ep_base": self._ep_base,
            "episode_scores": self._episode_scores,
            "episode_lengths": self._episode_lengths,
            "episode_base_rewards": self._episode_base_rewards,
            "episode_alive_rewards": self._episode_alive_rewards,
            "terminated_count": self._terminated_count,
            "truncated_count": self._truncated_count,
            "phase_events": self._phase_events,
        }

    def import_state(self, state: dict) -> None:
        phase_idx = int(state.get("phase_idx", 0))
        self._phase_idx = max(0, min(phase_idx, len(PHASES) - 1))
        phase_name = state.get("phase_name")
        if phase_name:
            for idx, phase in enumerate(PHASES):
                if phase.name == phase_name:
                    self._phase_idx = idx
                    break
        self._scores = [float(v) for v in state.get("scores_window", [])]
        self._rollback_bad_windows = int(state.get("rollback_bad_windows", 0))
        self._ep_base = float(state.get("ep_base", 0.0))
        self._episode_scores = [float(v) for v in state.get("episode_scores", [])]
        self._episode_lengths = [int(v) for v in state.get("episode_lengths", [])]
        self._episode_base_rewards = [float(v) for v in state.get("episode_base_rewards", [])]
        self._episode_alive_rewards = [float(v) for v in state.get("episode_alive_rewards", [])]
        self._terminated_count = int(state.get("terminated_count", 0))
        self._truncated_count = int(state.get("truncated_count", 0))
        self._phase_events = list(state.get("phase_events", []))

    def _on_step(self) -> bool:
        info = self.locals["infos"][0]
        self._ep_base += info.get("base_reward", 0.0)

        if "episode" not in info:
            return True

        ep_len = info["episode"]["l"]
        alive_total = self.alive_reward * self.frame_skip * ep_len
        score = max(0.0, self._ep_base - alive_total)
        self._scores.append(score)
        self._episode_scores.append(score)
        self._episode_lengths.append(ep_len)
        self._episode_base_rewards.append(self._ep_base)
        self._episode_alive_rewards.append(alive_total)
        if info.get("TimeLimit.truncated", False) or info.get("spawn_debug", {}).get("truncated", False):
            self._truncated_count += 1
        else:
            self._terminated_count += 1
        self._ep_base = 0.0

        phase = PHASES[self._phase_idx]
        effective_threshold = (phase.threshold or float("inf")) * self.threshold_mult
        recent_scores = self._scores[-min(len(self._scores), self.window):]
        recent_lengths = self._episode_lengths[-len(recent_scores):] if recent_scores else []
        mean = _mean(recent_scores)
        mean_len = _mean([float(v) for v in recent_lengths])
        score_min = min(recent_scores) if recent_scores else 0.0
        score_std = _stdev(recent_scores)
        self._save_status(mean, effective_threshold)

        if len(self._scores) < self.window:
            return True

        rollback, reason = self._should_rollback(
            phase=phase,
            mean=mean,
            mean_len=mean_len,
            effective_threshold=effective_threshold,
            recent_count=len(recent_scores),
        )
        if rollback:
            self._rollback_bad_windows += 1
            if self._rollback_bad_windows >= self.rollback_patience:
                self._rollback_phase(mean, effective_threshold, mean_len, reason)
                return True
        else:
            self._rollback_bad_windows = 0

        can_promote, promotion_reason = self._promotion_judgment(
            phase=phase,
            mean=mean,
            mean_len=mean_len,
            score_min=score_min,
            score_std=score_std,
            effective_threshold=effective_threshold,
            recent_count=len(recent_scores),
        )
        if can_promote:
            self._advance_phase(mean, effective_threshold, promotion_reason)

        return True

    def _advance_phase(self, mean: float, threshold: float, reason: str) -> None:
        prev_name = PHASES[self._phase_idx].name
        prev_idx = self._phase_idx
        self._phase_idx = min(self._phase_idx + 1, len(PHASES) - 1)
        self._scores.clear()
        self._rollback_bad_windows = 0
        next_phase = PHASES[self._phase_idx]
        self._phase_events.append({
            "event": "advance",
            "timestep": self.num_timesteps,
            "from_phase_idx": prev_idx,
            "from_phase_name": prev_name,
            "to_phase_idx": self._phase_idx,
            "to_phase_name": next_phase.name,
            "active_score_mean": round(mean, 4),
            "threshold": round(threshold, 4),
            "reason": reason,
        })
        print(
            f"\n[Curriculum] Phase {self._phase_idx} 昇格: "
            f"{prev_name} -> {next_phase.name} "
            f"(score={mean:.3f} >= {threshold:.1f}, {reason})"
        )
        self._apply_phase(next_phase)
        self._log_wandb(mean, next_phase, event="advance")

    def _promotion_judgment(
        self,
        phase: _Phase,
        mean: float,
        mean_len: float,
        score_min: float,
        score_std: float,
        effective_threshold: float,
        recent_count: int,
    ) -> tuple[bool, str]:
        if phase.threshold is None or recent_count < self.window:
            return False, ""
        if mean < effective_threshold:
            return False, f"score_mean={mean:.3f} < threshold={effective_threshold:.3f}"
        if mean_len < phase.min_episode_steps:
            return False, f"ep_len={mean_len:.1f} < min_episode_steps={phase.min_episode_steps}"

        min_score_floor = effective_threshold * phase.promotion_min_score_ratio
        if score_min < min_score_floor:
            return False, (
                f"score_min={score_min:.3f} < promotion_min_score_floor={min_score_floor:.3f}"
            )

        score_cv = score_std / max(mean, 1e-8)
        if score_cv > phase.promotion_max_score_cv:
            return False, (
                f"score_cv={score_cv:.3f} > promotion_max_score_cv={phase.promotion_max_score_cv:.3f}"
            )

        return True, (
            f"score_min={score_min:.3f} >= {min_score_floor:.3f}, "
            f"score_cv={score_cv:.3f} <= {phase.promotion_max_score_cv:.3f}"
        )

    def _should_rollback(
        self,
        phase: _Phase,
        mean: float,
        mean_len: float,
        effective_threshold: float,
        recent_count: int,
    ) -> tuple[bool, str]:
        if self._phase_idx <= 0 or recent_count < self.rollback_min_episodes:
            return False, ""

        score_floor, length_floor = self._rollback_floors(phase, effective_threshold)
        if score_floor is not None and mean < score_floor:
            return True, f"score={mean:.3f} < rollback_score_floor={score_floor:.3f}"
        if length_floor is not None and mean_len < length_floor:
            return True, f"ep_len={mean_len:.1f} < rollback_length_floor={length_floor:.1f}"
        return False, ""

    def _rollback_phase(
        self,
        mean: float,
        threshold: float,
        mean_len: float,
        reason: str,
    ) -> None:
        prev_name = PHASES[self._phase_idx].name
        prev_idx = self._phase_idx
        self._phase_idx = max(self._phase_idx - 1, 0)
        self._scores.clear()
        self._rollback_bad_windows = 0
        next_phase = PHASES[self._phase_idx]
        self._phase_events.append({
            "event": "rollback",
            "timestep": self.num_timesteps,
            "from_phase_idx": prev_idx,
            "from_phase_name": prev_name,
            "to_phase_idx": self._phase_idx,
            "to_phase_name": next_phase.name,
            "active_score_mean": round(mean, 4),
            "episode_length_mean": round(mean_len, 1),
            "threshold": round(threshold, 4),
            "reason": reason,
        })
        print(
            f"\n[Curriculum] Phase {self._phase_idx} 降格: "
            f"{prev_name} -> {next_phase.name} "
            f"({reason}, score={mean:.3f}, ep_len={mean_len:.1f})"
        )
        self._apply_phase(next_phase)
        self._log_wandb(mean, next_phase, event="rollback", mean_len=mean_len, reason=reason)

    def _rollback_floors(
        self,
        phase: _Phase,
        effective_threshold: float,
    ) -> tuple[Optional[float], Optional[float]]:
        score_floor = (
            effective_threshold * phase.rollback_score_ratio
            if phase.threshold is not None
            else None
        )
        length_floor = (
            phase.min_episode_steps * phase.rollback_length_ratio
            if phase.min_episode_steps > 0
            else None
        )
        return score_floor, length_floor

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
            f"TimeScaling={'ON' if phase.time_scaling else 'OFF'}, "
            f"MinEpisodeSteps={phase.min_episode_steps}"
        )

    def _save_status(self, mean: float, threshold: float) -> None:
        if self._status_path is None:
            return
        status = self.get_diagnostics()
        status["current_window"]["active_score_mean"] = round(mean, 4)
        status["current_window"]["effective_threshold"] = round(threshold, 4)
        Path(self._status_path).write_text(
            json.dumps(status, ensure_ascii=False, indent=2)
        )

    def get_diagnostics(self) -> dict:
        phase = PHASES[self._phase_idx]
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
        promotion_min_score_floor = (
            effective_threshold * phase.promotion_min_score_ratio
            if base_threshold is not None
            else None
        )
        score_cv = (
            score_std / max(score_mean, 1e-8)
            if recent_scores and score_mean > 0.0
            else None
        )
        promotion_stable = (
            recent_count >= self.window
            and base_threshold is not None
            and score_min is not None
            and promotion_min_score_floor is not None
            and score_min >= promotion_min_score_floor
            and score_cv is not None
            and score_cv <= phase.promotion_max_score_cv
        )
        return {
            "timestep": self.num_timesteps,
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
                "episode_length_mean": round(length_mean, 1),
                "min_episode_steps": phase.min_episode_steps,
                "base_threshold": base_threshold,
                "effective_threshold": round(effective_threshold, 4) if base_threshold is not None else None,
                "threshold_ratio": round(threshold_ratio, 4) if threshold_ratio is not None else None,
                "promotion_min_score_ratio": phase.promotion_min_score_ratio,
                "promotion_min_score_floor": (
                    round(promotion_min_score_floor, 4)
                    if promotion_min_score_floor is not None
                    else None
                ),
                "promotion_max_score_cv": phase.promotion_max_score_cv,
                "promotion_stability_ok": promotion_stable,
                "rollback_score_floor": round(score_floor, 4) if score_floor is not None else None,
                "rollback_length_floor": round(length_floor, 1) if length_floor is not None else None,
                "rollback_bad_windows": self._rollback_bad_windows,
                "ready_for_phase_judgment": (
                    recent_count >= self.window
                    and base_threshold is not None
                    and score_mean >= effective_threshold
                    and length_mean >= phase.min_episode_steps
                    and promotion_stable
                ),
            },
            "overall": {
                "active_score_mean": round(_mean(self._episode_scores), 4),
                "episode_length_mean": round(_mean([float(v) for v in self._episode_lengths]), 1),
                "base_reward_mean": round(_mean(self._episode_base_rewards), 4),
                "alive_reward_mean": round(_mean(self._episode_alive_rewards), 4),
            },
            "phase_events": self._phase_events,
            "params": {
                "min_enemies": phase.min_enemies,
                "max_enemies": phase.max_enemies,
                "speed_mult": phase.speed_mult,
                "spawn_rate_mult": phase.spawn_rate_mult,
                "max_enemy_type_id": phase.max_enemy_type_id,
                "enemy_hp_scale": phase.enemy_hp_scale,
                "enemy_damage_scale": phase.enemy_damage_scale,
                "time_scaling": phase.time_scaling,
                "min_episode_steps": phase.min_episode_steps,
                "rollback_score_ratio": phase.rollback_score_ratio,
                "rollback_length_ratio": phase.rollback_length_ratio,
                "promotion_min_score_ratio": phase.promotion_min_score_ratio,
                "promotion_max_score_cv": phase.promotion_max_score_cv,
            },
            "recommendation": self.recommend_next_settings(),
        }

    def recommend_next_settings(self) -> dict:
        phase = PHASES[self._phase_idx]
        recent_count = min(len(self._scores), self.window)
        recent_scores = self._scores[-recent_count:] if recent_count else []
        recent_lengths = self._episode_lengths[-recent_count:] if recent_count else []
        length_mean = _mean([float(v) for v in recent_lengths])
        if (
            recent_count >= self.window
            and phase.min_episode_steps > 0
            and length_mean < phase.min_episode_steps * 0.5
        ):
            return {
                "suggested_curriculum_threshold": self.threshold_mult,
                "suggested_curriculum_window": self.window,
                "failure_class": "episode_length_collapse",
                "threshold_reason": (
                    "episode length collapsed far below the current phase requirement; "
                    "do not lower threshold. Prefer rollback, phase subdivision, or resume from a pre-collapse checkpoint."
                ),
                "window_reason": "current window has enough episode samples",
                "observed_episode_length_mean": round(length_mean, 1),
                "required_min_episode_steps": phase.min_episode_steps,
            }
        return self._recommendation_policy.recommend(
            CurriculumTuningInput(
                threshold_mult=self.threshold_mult,
                window=self.window,
                episode_count=len(self._episode_scores),
                recent_scores=recent_scores,
                base_threshold=phase.threshold,
            )
        )

    def _log_wandb(
        self,
        mean: float,
        phase: _Phase,
        event: str,
        mean_len: Optional[float] = None,
        reason: Optional[str] = None,
    ) -> None:
        try:
            import wandb
            if wandb.run:
                payload = {
                    "curriculum/phase_idx": self._phase_idx,
                    "curriculum/phase_name": phase.name,
                    "curriculum/event": event,
                    "curriculum/score_mean": mean,
                    "curriculum/min_enemies": phase.min_enemies,
                    "curriculum/max_enemies": phase.max_enemies,
                    "curriculum/speed_mult": phase.speed_mult,
                    "curriculum/spawn_rate_mult": phase.spawn_rate_mult,
                    "curriculum/max_enemy_type_id": phase.max_enemy_type_id,
                    "curriculum/enemy_hp_scale": phase.enemy_hp_scale,
                    "curriculum/enemy_damage_scale": phase.enemy_damage_scale,
                    "curriculum/time_scaling": int(phase.time_scaling),
                    "curriculum/min_episode_steps": phase.min_episode_steps,
                    "curriculum/rollback_score_ratio": phase.rollback_score_ratio,
                    "curriculum/rollback_length_ratio": phase.rollback_length_ratio,
                    "curriculum/promotion_min_score_ratio": phase.promotion_min_score_ratio,
                    "curriculum/promotion_max_score_cv": phase.promotion_max_score_cv,
                }
                if mean_len is not None:
                    payload["curriculum/episode_length_mean"] = mean_len
                if reason:
                    payload["curriculum/rollback_reason"] = reason
                wandb.log(payload, step=self.num_timesteps)
        except ImportError:
            pass
