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
    percentile as _percentile,
)
from stable_baselines3.common.callbacks import BaseCallback

from games.survivors.survivors_difficulty import compute_difficulty_score
from games.survivors.state_modules import CurriculumStateModule, EpisodeScoreTracker
from games.survivors.param_applier import ParamApplier


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
    rollback_score_ratio: float = 0.60
    rollback_length_ratio: float = 0.60
    promotion_min_score_ratio: float = 0.85
    promotion_max_score_cv: float = 0.20
    promotion_score_stat: str = "min"       # "min" | "percentile"
    promotion_score_percentile: float = 10.0
    # threshold=None の最終フェーズ用: ロールバック score floor の基準となる参照閾値
    rollback_reference_threshold: Optional[float] = None


# promotion_blocker カテゴリ定数（W&B グラフで停滞原因を数値で識別する）
BLOCKER_NONE = 0           # 昇格可能（または最終フェーズ）
BLOCKER_NOT_ENOUGH_EP = 1  # ウィンドウ内エピソード数不足
BLOCKER_SCORE_MEAN_LOW = 2 # score_mean < effective_threshold
BLOCKER_EP_LEN_LOW = 3     # episode_length_mean < min_episode_steps
BLOCKER_SCORE_MIN_LOW = 4  # promotion_low_score (score_min or score_p10) < promotion_low_score_floor
BLOCKER_SCORE_CV_HIGH = 5  # score_cv > promotion_max_score_cv

PHASES: list[_Phase] = [
    _Phase("入門",              4,   6, 0.8, 1.0,  1, 0.50, 0.50, False,  600,   30.0),
    _Phase("Gem回収開始",       6,  10, 0.9, 1.4,  2, 0.75, 0.75, False, 1200,  100.0),
    _Phase("Gem追従強化",       7,  14, 0.9, 1.7,  3, 0.85, 0.85, False, 1800,  250.0),
    _Phase("通常序盤",          8,  20, 1.0, 2.0,  4, 1.00, 1.00, False, 2400,  800.0),
    _Phase("包囲入門A",         9,  18, 1.0, 2.0,  4, 1.00, 0.80, False, 2400,  950.0,
           promotion_min_score_ratio=0.70, promotion_max_score_cv=0.25,
           promotion_score_stat="percentile", promotion_score_percentile=10.0),
    _Phase("包囲入門B",        10,  22, 1.0, 2.1,  4, 1.00, 0.90, False, 2400,  940.0,
           promotion_min_score_ratio=0.70, promotion_max_score_cv=0.25,
           promotion_score_stat="percentile", promotion_score_percentile=10.0),
    _Phase("包囲入門B+",       10,  24, 1.0, 2.15, 4, 1.00, 0.95, False, 2400, 1100.0,
           promotion_min_score_ratio=0.70, promotion_max_score_cv=0.25,
           promotion_score_stat="percentile", promotion_score_percentile=10.0),
    _Phase("包囲入門C",        10,  25, 1.0, 2.2,  4, 1.05, 1.00, False, 2400, 1150.0,
           promotion_min_score_ratio=0.70, promotion_max_score_cv=0.25,
           promotion_score_stat="percentile", promotion_score_percentile=10.0),
    _Phase("包囲対応",         12,  30, 1.0, 2.4,  5, 1.10, 1.10, False, 2400, 1350.0,
           promotion_min_score_ratio=0.70, promotion_max_score_cv=0.25,
           promotion_score_stat="percentile", promotion_score_percentile=10.0),
    _Phase("多敵対応",         14,  40, 1.0, 2.6,  6, 1.20, 1.20, False, 2400, 1550.0,
           promotion_min_score_ratio=0.70, promotion_max_score_cv=0.25,
           promotion_score_stat="percentile", promotion_score_percentile=10.0),
    _Phase("群れ対応A",        16,  50, 1.0, 2.7,  7, 1.25, 1.25, False, 2400, 1700.0,
           promotion_min_score_ratio=0.70, promotion_max_score_cv=0.25,
           promotion_score_stat="percentile", promotion_score_percentile=10.0),
    _Phase("群れ対応B",        18,  65, 1.05, 2.8,  8, 1.35, 1.35, True,  2400, 1850.0,
           promotion_min_score_ratio=0.70, promotion_max_score_cv=0.25,
           promotion_score_stat="percentile", promotion_score_percentile=10.0),
    _Phase("群れ対応C",        20,  80, 1.1,  3.0,  9, 1.50, 1.50, True,  2400, 2000.0,
           promotion_min_score_ratio=0.70, promotion_max_score_cv=0.25,
           promotion_score_stat="percentile", promotion_score_percentile=10.0),
    _Phase("Mad Forest 入門",  27, 100, 1.13, 3.3, 10, 1.67, 1.67, True,  2400, 2100.0,
           promotion_min_score_ratio=0.70, promotion_max_score_cv=0.25,
           promotion_score_stat="percentile", promotion_score_percentile=10.0),
    _Phase("Mad Forest 中級",  33, 125, 1.17, 3.7, 10, 1.83, 1.83, True,  2400, 2250.0,
           promotion_min_score_ratio=0.70, promotion_max_score_cv=0.25,
           promotion_score_stat="percentile", promotion_score_percentile=10.0),
    _Phase("Mad Forest",       40, 150, 1.2,  4.0, 10, 2.00, 2.00, True,  2400,   None,
           rollback_reference_threshold=2250.0),
]




def _phase_to_params(phase_idx: int) -> dict:
    phase = PHASES[phase_idx]
    return {
        "min_enemies":        phase.min_enemies,
        "max_enemies":        phase.max_enemies,
        "speed_mult":         phase.speed_mult,
        "spawn_rate_mult":    phase.spawn_rate_mult,
        "max_enemy_type_id":  phase.max_enemy_type_id,
        "enemy_hp_scale":     phase.enemy_hp_scale,
        "enemy_damage_scale": phase.enemy_damage_scale,
        "time_scaling":       phase.time_scaling,
    }

class CurriculumCallback(BaseCallback):
    """VSスタイル段階式カリキュラム（リファクタリング版）。"""

    def __init__(
        self,
        raw_env,
        frame_skip: int = 1,
        window: int = 20,
        threshold_mult: float = 1.0,
        alive_reward: float = 0.001,
        status_path=None,
        wandb_logger=None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self._param_applier = ParamApplier(raw_env)
        self._score_tracker = EpisodeScoreTracker(frame_skip, alive_reward)
        self._curriculum = CurriculumStateModule(
            window=window,
            threshold_mult=threshold_mult,
            rollback_patience=3,
            status_path=status_path,
        )
        self._wandb_logger = wandb_logger
        if wandb_logger:
            wandb_logger.add_metric_prefix("curriculum/")
        self.frame_skip = frame_skip
        self.window = window
        self.threshold_mult = threshold_mult
        self.alive_reward = alive_reward
        self.rollback_patience = 3
        self.rollback_min_episodes = max(5, window // 2)
        self._status_path = status_path
        self._recommendation_policy = WindowThresholdRecommendationPolicy()
        self._missing_episode_info_count = 0

    def _on_training_start(self) -> None:
        n = self.training_env.num_envs if self.training_env is not None else 1
        self._param_applier.set_training_env(self.training_env)
        self._score_tracker.reset(n)
        self._param_applier.apply(_phase_to_params(self._curriculum.current_phase))

    @property
    def is_final_phase(self) -> bool:
        return self._curriculum.is_final_phase

    def configure_completion(self, window, min_episodes, min_score_ratio, min_episode_len_ratio):
        self._curriculum.configure_completion(window, min_episodes, min_score_ratio, min_episode_len_ratio)

    def export_state(self) -> dict:
        return self._curriculum.export_state()

    def import_state(self, state: dict) -> None:
        self._curriculum.import_state(state)

    def _save_status(self, *a):
        self._curriculum.save_status()

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        dones = self.locals.get("dones", [False] * len(infos))
        episode_results = self._score_tracker.process(infos)
        for env_idx, active_score, ep_len in episode_results:
            info = infos[env_idx] if env_idx < len(infos) else {}
            base_r = float(info.get("base_reward_ep", 0.0) or 0.0)
            alive_r = self.alive_reward * self.frame_skip * ep_len
            is_truncated = bool(info.get("TimeLimit.truncated", False))
            self._curriculum.on_episode_end(
                active_score, ep_len,
                base_reward=base_r,
                alive_reward=alive_r,
                terminated=not is_truncated,
            )
        if episode_results:
            event = self._curriculum.check_phase_transition()
            if event in ("advance", "rollback"):
                self._param_applier.apply(_phase_to_params(self._curriculum.current_phase))
                if self._wandb_logger:
                    self._wandb_logger.log(
                        self._curriculum.get_wandb_metrics(),
                        step=self.num_timesteps,
                    )
        self._curriculum._steps_in_phase += 1
        return True

    def get_completion_diagnostics(self, window=None, min_episodes=None, min_score_ratio=None, min_episode_len_ratio=None):
        return self._curriculum.get_completion_diagnostics(window, min_episodes, min_score_ratio, min_episode_len_ratio)

    def get_diagnostics(self) -> dict:
        phase = PHASES[self._curriculum.current_phase]
        base_threshold = phase.threshold
        effective_threshold = (base_threshold or float("inf")) * self.threshold_mult
        recent_count = min(len(self._curriculum._scores), self.window)
        recent_scores = self._curriculum._scores[-recent_count:] if recent_count else []
        recent_lengths = self._curriculum._episode_lengths[-recent_count:] if recent_count else []
        score_mean = _mean(recent_scores)
        score_std = _stdev(recent_scores)
        length_mean = _mean([float(v) for v in recent_lengths])
        score_floor, length_floor = self._curriculum._rollback_floors(phase, effective_threshold)
        threshold_ratio = (score_mean / effective_threshold if base_threshold is not None and effective_threshold > 0.0 and recent_scores else None)
        score_min = min(recent_scores) if recent_scores else None
        score_cv = (score_std / max(score_mean, 1e-8) if recent_scores and score_mean > 0.0 else None)
        if phase.promotion_score_stat == "percentile" and recent_scores:
            promotion_low_score = _percentile(recent_scores, phase.promotion_score_percentile)
        else:
            promotion_low_score = score_min if score_min is not None else 0.0
        promotion_low_score_floor = (effective_threshold * phase.promotion_min_score_ratio if base_threshold is not None else None)
        promotion_low_score_ok = (promotion_low_score_floor is not None and promotion_low_score >= promotion_low_score_floor)
        promotion_stable = (recent_count >= self.window and base_threshold is not None and promotion_low_score_floor is not None and promotion_low_score_ok and score_cv is not None and score_cv <= phase.promotion_max_score_cv)
        c = self._curriculum
        return {
            "timestep": self.num_timesteps,
            "phase_idx": c._phase_idx,
            "phase_name": phase.name,
            "window": self.window,
            "threshold_mult": self.threshold_mult,
            "rollback_patience": self.rollback_patience,
            "rollback_min_episodes": self.rollback_min_episodes,
            "rollback_bad_windows": c._rollback_bad_windows,
            "episodes_total": len(c._episode_scores),
            "terminated_episodes": c._terminated_count,
            "truncated_episodes": c._truncated_count,
            "current_window": {
                "episodes": recent_count,
                "required_episodes": self.window,
                "active_score_mean": round(score_mean, 4),
                "active_score_min": round(score_min, 4) if score_min is not None else None,
                "active_score_max": round(max(recent_scores), 4) if recent_scores else None,
                "active_score_std": round(score_std, 4),
                "active_score_cv": round(score_cv, 4) if score_cv is not None else None,
                "active_score_p10": round(_percentile(recent_scores, 10.0), 4) if recent_scores else None,
                "active_score_p20": round(_percentile(recent_scores, 20.0), 4) if recent_scores else None,
                "episode_length_mean": round(length_mean, 1),
                "min_episode_steps": phase.min_episode_steps,
                "base_threshold": base_threshold,
                "effective_threshold": round(effective_threshold, 4) if base_threshold is not None else None,
                "threshold_ratio": round(threshold_ratio, 4) if threshold_ratio is not None else None,
                "promotion_min_score_ratio": phase.promotion_min_score_ratio,
                "promotion_min_score_floor": round(promotion_low_score_floor, 4) if promotion_low_score_floor is not None else None,
                "promotion_max_score_cv": phase.promotion_max_score_cv,
                "promotion_score_stat": phase.promotion_score_stat,
                "promotion_score_percentile": phase.promotion_score_percentile,
                "promotion_low_score": round(promotion_low_score, 4),
                "promotion_low_score_floor": round(promotion_low_score_floor, 4) if promotion_low_score_floor is not None else None,
                "promotion_low_score_ok": promotion_low_score_ok,
                "promotion_stability_ok": promotion_stable,
                "rollback_score_floor": round(score_floor, 4) if score_floor is not None else None,
                "rollback_length_floor": round(length_floor, 1) if length_floor is not None else None,
                "rollback_bad_windows": c._rollback_bad_windows,
                "ready_for_phase_judgment": (recent_count >= self.window and base_threshold is not None and score_mean >= effective_threshold and length_mean >= phase.min_episode_steps and promotion_stable),
            },
            "overall": {
                "active_score_mean": round(_mean(c._episode_scores), 4),
                "episode_length_mean": round(_mean([float(v) for v in c._episode_lengths]), 1),
                "base_reward_mean": round(_mean(c._episode_base_rewards), 4),
                "alive_reward_mean": round(_mean(c._episode_alive_rewards), 4),
            },
            "phase_events": c._phase_events,
            "completion": self.get_completion_diagnostics(),
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
                "promotion_score_stat": phase.promotion_score_stat,
                "promotion_score_percentile": phase.promotion_score_percentile,
            },
            "recommendation": self.recommend_next_settings(),
        }

    def recommend_next_settings(self) -> dict:
        phase = PHASES[self._curriculum.current_phase]
        c = self._curriculum
        recent_count = min(len(c._scores), self.window)
        recent_scores = c._scores[-recent_count:] if recent_count else []
        recent_lengths = c._episode_lengths[-recent_count:] if recent_count else []
        length_mean = _mean([float(v) for v in recent_lengths])
        if (recent_count >= self.window and phase.min_episode_steps > 0 and length_mean < phase.min_episode_steps * 0.5):
            return {"suggested_curriculum_threshold": self.threshold_mult, "suggested_curriculum_window": self.window, "failure_class": "episode_length_collapse", "threshold_reason": "episode length collapsed", "window_reason": "current window has enough episode samples", "observed_episode_length_mean": round(length_mean, 1), "required_min_episode_steps": phase.min_episode_steps}
        return self._recommendation_policy.recommend(
            CurriculumTuningInput(
                threshold_mult=self.threshold_mult,
                window=self.window,
                episode_count=len(c._episode_scores),
                recent_scores=recent_scores,
                base_threshold=phase.threshold,
            )
        )

    def _compute_blocker_category(self, window_dict: dict, phase_threshold_is_none: bool) -> int:
        if phase_threshold_is_none or self.is_final_phase:
            return BLOCKER_NONE
        recent_count = int(window_dict.get("episodes", 0) or 0)
        required = int(window_dict.get("required_episodes", 1) or 1)
        if recent_count < required:
            return BLOCKER_NOT_ENOUGH_EP
        score_mean = float(window_dict.get("active_score_mean") or 0.0)
        threshold = float(window_dict.get("effective_threshold") or float("inf"))
        if score_mean < threshold:
            return BLOCKER_SCORE_MEAN_LOW
        length_mean = float(window_dict.get("episode_length_mean") or 0.0)
        min_ep_steps = int(window_dict.get("min_episode_steps") or 0)
        if length_mean < min_ep_steps:
            return BLOCKER_EP_LEN_LOW
        promotion_low_score = window_dict.get("promotion_low_score")
        min_score_floor = window_dict.get("promotion_low_score_floor")
        if promotion_low_score is not None and min_score_floor is not None and promotion_low_score < min_score_floor:
            return BLOCKER_SCORE_MIN_LOW
        score_cv = window_dict.get("active_score_cv")
        max_cv = window_dict.get("promotion_max_score_cv")
        if score_cv is not None and max_cv is not None and score_cv > max_cv:
            return BLOCKER_SCORE_CV_HIGH
        return BLOCKER_NONE

    def get_wandb_progress_metrics(self) -> dict:
        diagnostics = self.get_diagnostics()
        window = diagnostics.get("current_window", {})
        overall = diagnostics.get("overall", {})
        completion = diagnostics.get("completion", {})
        episodes_total = max(int(diagnostics.get("episodes_total", 0) or 0), 1)
        terminated = int(diagnostics.get("terminated_episodes", 0) or 0)
        truncated = int(diagnostics.get("truncated_episodes", 0) or 0)
        phase = PHASES[self._curriculum.current_phase]
        blocker = self._compute_blocker_category(window=window, phase_threshold_is_none=(phase.threshold is None))
        return {
            "survivors/active_score_mean": window.get("active_score_mean"),
            "survivors/active_score_min": window.get("active_score_min"),
            "survivors/active_score_std": window.get("active_score_std"),
            "survivors/active_score_cv": window.get("active_score_cv"),
            "survivors/episode_length_mean": window.get("episode_length_mean"),
            "survivors/base_reward_mean": overall.get("base_reward_mean"),
            "survivors/alive_reward_mean": overall.get("alive_reward_mean"),
            "survivors/terminated_ratio": terminated / episodes_total,
            "survivors/truncated_ratio": truncated / episodes_total,
            "curriculum/phase_idx": diagnostics.get("phase_idx"),
            "curriculum/phase_progress_ratio": window.get("threshold_ratio"),
            "curriculum/score_mean": window.get("active_score_mean"),
            "curriculum/score_min": window.get("active_score_min"),
            "curriculum/score_cv": window.get("active_score_cv"),
            "curriculum/threshold_ratio": window.get("threshold_ratio"),
            "curriculum/steps_in_phase": self._curriculum._steps_in_phase,
            "curriculum/episodes_in_phase": self._curriculum._episodes_in_phase,
            "curriculum/promotion_blocker": blocker,
            "curriculum/ready_for_phase_judgment": int(bool(window.get("ready_for_phase_judgment"))),
            "curriculum/promotion_stability_ok": int(bool(window.get("promotion_stability_ok"))),
            "curriculum/active_score_p10": window.get("active_score_p10"),
            "curriculum/active_score_p20": window.get("active_score_p20"),
            "curriculum/promotion_low_score": window.get("promotion_low_score"),
            "curriculum/promotion_low_score_floor": window.get("promotion_low_score_floor"),
            "curriculum/promotion_low_score_ok": int(bool(window.get("promotion_low_score_ok"))),
            "curriculum/promotion_score_percentile": window.get("promotion_score_percentile"),
            "curriculum/rollback_bad_windows": diagnostics.get("rollback_bad_windows"),
            "curriculum/window_episode_count": window.get("episodes"),
            "curriculum/is_final_phase": int(bool(completion.get("is_final_phase"))),
            "curriculum/completion_ready": int(bool(completion.get("complete"))),
            "curriculum/completion_score_mean": completion.get("active_score_mean"),
            "curriculum/completion_ep_len_mean": completion.get("episode_length_mean"),
        }

    def _log_wandb(self, mean, phase, event, mean_len=None, reason=None):
        try:
            import wandb
            if wandb.run:
                payload = {"curriculum/phase_idx": self._curriculum.current_phase, "curriculum/phase_name": phase.name, "curriculum/event": event, "curriculum/score_mean": mean, "curriculum/min_enemies": phase.min_enemies, "curriculum/max_enemies": phase.max_enemies, "curriculum/speed_mult": phase.speed_mult, "curriculum/spawn_rate_mult": phase.spawn_rate_mult, "curriculum/max_enemy_type_id": phase.max_enemy_type_id, "curriculum/enemy_hp_scale": phase.enemy_hp_scale, "curriculum/enemy_damage_scale": phase.enemy_damage_scale, "curriculum/time_scaling": int(phase.time_scaling), "curriculum/min_episode_steps": phase.min_episode_steps, "curriculum/rollback_score_ratio": phase.rollback_score_ratio, "curriculum/rollback_length_ratio": phase.rollback_length_ratio, "curriculum/promotion_min_score_ratio": phase.promotion_min_score_ratio, "curriculum/promotion_max_score_cv": phase.promotion_max_score_cv, "survivors/difficulty_score": compute_difficulty_score({"min_enemies": phase.min_enemies, "max_enemies": phase.max_enemies, "speed_mult": phase.speed_mult, "spawn_rate_mult": phase.spawn_rate_mult, "max_enemy_type_id": phase.max_enemy_type_id, "enemy_hp_scale": phase.enemy_hp_scale, "enemy_damage_scale": phase.enemy_damage_scale, "time_scaling": phase.time_scaling})}
                if mean_len is not None:
                    payload["curriculum/episode_length_mean"] = mean_len
                if reason:
                    payload["curriculum/rollback_reason"] = reason
                payload["global_step"] = self.num_timesteps
                wandb.log(payload, step=self.num_timesteps)
        except ImportError:
            pass

def promotion_judgment(phase, mean: float, mean_len: float, score_min: float, score_std: float,
                        effective_threshold: float, window: int, recent_count: int,
                        recent_scores=None) -> tuple:
    from base.curriculum import percentile as _percentile
    if phase.threshold is None or recent_count < window:
        return False, ""
    if mean < effective_threshold:
        return False, f"score_mean={mean:.3f} < threshold={effective_threshold:.3f}"
    if mean_len < phase.min_episode_steps:
        return False, f"ep_len={mean_len:.1f} < min_episode_steps={phase.min_episode_steps}"
    if phase.promotion_score_stat == "percentile" and recent_scores is not None:
        low_score = _percentile(recent_scores, phase.promotion_score_percentile)
        low_stat_label = f"score_p{int(phase.promotion_score_percentile)}"
    else:
        low_score = score_min
        low_stat_label = "score_min"
    min_score_floor = effective_threshold * phase.promotion_min_score_ratio
    if low_score < min_score_floor:
        return False, f"{low_stat_label}={low_score:.3f} < promotion_min_score_floor={min_score_floor:.3f}"
    score_cv = score_std / max(mean, 1e-8)
    if score_cv > phase.promotion_max_score_cv:
        return False, f"score_cv={score_cv:.3f} > promotion_max_score_cv={phase.promotion_max_score_cv:.3f}"
    return True, f"{low_stat_label}={low_score:.3f} >= {min_score_floor:.3f}, score_cv={score_cv:.3f} <= {phase.promotion_max_score_cv:.3f}"


def rollback_floors(phase, effective_threshold: float, threshold_mult: float) -> tuple:
    if phase.threshold is not None:
        score_floor = effective_threshold * phase.rollback_score_ratio
    elif phase.rollback_reference_threshold is not None:
        score_floor = (
            phase.rollback_reference_threshold
            * threshold_mult
            * phase.rollback_score_ratio
        )
    else:
        score_floor = None
    length_floor = (
        phase.min_episode_steps * phase.rollback_length_ratio
        if phase.min_episode_steps > 0
        else None
    )
    return score_floor, length_floor


def should_rollback(phase, phase_idx: int, mean: float, mean_len: float,
                    effective_threshold: float, threshold_mult: float,
                    rollback_min_episodes: int, recent_count: int) -> tuple:
    if phase_idx <= 0 or recent_count < rollback_min_episodes:
        return False, ""
    score_floor, length_floor = rollback_floors(phase, effective_threshold, threshold_mult)
    if score_floor is not None and mean < score_floor:
        return True, f"score={mean:.3f} < rollback_score_floor={score_floor:.3f}"
    if length_floor is not None and mean_len < length_floor:
        return True, f"ep_len={mean_len:.1f} < rollback_length_floor={length_floor:.1f}"
    return False, ""
