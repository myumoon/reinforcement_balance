"""TaskCellSamplerStateModule: タスクセル（武器×敵難度）のサンプリング状態管理。"""
from __future__ import annotations
import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING
import numpy as np
from games.survivors.modules.state_modules import BaseStateModule
from games.survivors.survivors_weapon_table import (
    WEAPON_UNLOCK_ORDER,
    get_unlocked_startable_weapon_ids,
)

if TYPE_CHECKING:
    from games.survivors.modules.weapon_unlock_module import WeaponUnlockAdvanceEvent


@dataclass(frozen=True)
class TaskCell:
    weapon_unlock_stage_key: str
    first_weapon_id: int
    enemy_phase_idx: int

    def key(self) -> str:
        return f"{self.weapon_unlock_stage_key}/{self.first_weapon_id}/{self.enemy_phase_idx}"


@dataclass
class TaskCellStats:
    cell: TaskCell
    episode_count: int = 0
    recent_scores: deque = field(default_factory=lambda: deque(maxlen=40))
    recent_episode_lengths: deque = field(default_factory=lambda: deque(maxlen=40))
    terminated_count: int = 0
    truncated_count: int = 0
    previous_score_mean: float | None = None
    active_score_mean: float = 0.0
    active_score_p10: float = 0.0
    active_score_cv: float = 0.0
    episode_length_mean: float = 0.0
    terminated_rate: float = 0.0
    learning_progress: float = 0.0
    regression_score: float = 0.0
    last_sample_step: int = 0
    blocked_until_step: int = 0


class TaskCellSamplerStateModule(BaseStateModule):
    """タスクセル（武器×敵難度）の選択・統計管理。

    サンプリング重み: min_exposure_bonus + weakness_bonus + learning_progress_weight
                    + regression_replay_weight + recency_weight + random_floor
    """

    def __init__(
        self,
        min_episodes_per_cell: int = 30,
        target_p10: float = 300.0,
        progress_scale: float = 300.0,
        random_floor: float = 0.05,
        recency_scale: int = 500_000,
        unlearnable_min_episodes: int = 10,
        unlearnable_score_floor: float = 50.0,
        emergency_length_floor_ratio: float = 0.25,
        blocked_steps: int = 200_000,
        enemy_phase_backtrack: int = 1,
    ) -> None:
        self._min_episodes = min_episodes_per_cell
        self._target_p10 = target_p10
        self._progress_scale = progress_scale
        self._random_floor = random_floor
        self._recency_scale = recency_scale
        self._unlearnable_min_episodes = unlearnable_min_episodes
        self._unlearnable_score_floor = unlearnable_score_floor
        self._emergency_length_floor_ratio = emergency_length_floor_ratio
        self._blocked_steps = blocked_steps
        self._enemy_phase_backtrack = enemy_phase_backtrack

        self._current_stage_key: str = "WU0"
        self._max_unlocked_enemy_phase_idx: int = 0
        self._min_episode_steps_by_phase: dict[int, int] = {}  # phase_idx -> min_episode_steps

        # セル統計: cell.key() -> TaskCellStats
        self._stats: dict[str, TaskCellStats] = {}
        # 候補セルリスト（現在サンプリング対象）
        self._candidate_cells: list[TaskCell] = []

    def rebuild_candidate_cells(
        self,
        stage_key: str,
        max_unlocked_enemy_phase_idx: int,
        min_episode_steps_by_phase: dict[int, int],
    ) -> None:
        """候補セルリストを再構築する。既存 stats は保持する。"""
        self._current_stage_key = stage_key
        self._max_unlocked_enemy_phase_idx = max_unlocked_enemy_phase_idx
        self._min_episode_steps_by_phase = dict(min_episode_steps_by_phase)

        startable_ids = get_unlocked_startable_weapon_ids(stage_key)
        lo_phase = max(0, max_unlocked_enemy_phase_idx - self._enemy_phase_backtrack)
        hi_phase = max_unlocked_enemy_phase_idx

        new_cells: list[TaskCell] = []
        for wid in startable_ids:
            for phase_idx in range(lo_phase, hi_phase + 1):
                cell = TaskCell(
                    weapon_unlock_stage_key=stage_key,
                    first_weapon_id=wid,
                    enemy_phase_idx=phase_idx,
                )
                new_cells.append(cell)
                if cell.key() not in self._stats:
                    self._stats[cell.key()] = TaskCellStats(cell=cell)

        self._candidate_cells = new_cells

    def on_episode_end(
        self,
        cell: TaskCell,
        active_score: float,
        ep_len: int,
        terminated: bool,
        num_timesteps: int,
    ) -> None:
        """エピソード終了時に統計を更新する。"""
        k = cell.key()
        if k not in self._stats:
            self._stats[k] = TaskCellStats(cell=cell)
        stats = self._stats[k]

        stats.episode_count += 1
        stats.recent_scores.append(active_score)
        stats.recent_episode_lengths.append(ep_len)
        if terminated:
            stats.terminated_count += 1
        else:
            stats.truncated_count += 1

        scores = list(stats.recent_scores)
        if scores:
            new_mean = float(np.mean(scores))
            # previous_score_mean は recent_scores が maxlen に達するたびに更新
            if len(stats.recent_scores) == stats.recent_scores.maxlen:
                stats.previous_score_mean = stats.active_score_mean if stats.active_score_mean > 0 else new_mean
            stats.active_score_mean = new_mean
            stats.active_score_p10 = float(np.percentile(scores, 10))
            std = float(np.std(scores))
            stats.active_score_cv = std / max(new_mean, 1e-8) if new_mean > 0 else 0.0
        else:
            stats.active_score_mean = 0.0
            stats.active_score_p10 = 0.0
            stats.active_score_cv = 0.0

        lengths = list(stats.recent_episode_lengths)
        stats.episode_length_mean = float(np.mean(lengths)) if lengths else 0.0
        ep_total = stats.terminated_count + stats.truncated_count
        stats.terminated_rate = stats.terminated_count / ep_total if ep_total > 0 else 0.0

        prev = stats.previous_score_mean
        if prev is not None:
            stats.learning_progress = abs(stats.active_score_mean - prev)
            stats.regression_score = max(0.0, prev - stats.active_score_mean)
        else:
            stats.learning_progress = 0.0
            stats.regression_score = 0.0

        # ブロック判定
        min_ep_steps = self._min_episode_steps_by_phase.get(cell.enemy_phase_idx, 0)
        emergency_floor = min_ep_steps * self._emergency_length_floor_ratio
        is_unlearnable = (
            stats.episode_count >= self._unlearnable_min_episodes
            and stats.active_score_mean < self._unlearnable_score_floor
            and (min_ep_steps == 0 or stats.episode_length_mean < emergency_floor)
        )
        if is_unlearnable:
            stats.blocked_until_step = num_timesteps + self._blocked_steps
            print(
                f"[TaskCellSampler] セルをブロック: {k} "
                f"(mean={stats.active_score_mean:.1f}, ep_len={stats.episode_length_mean:.1f}, "
                f"blocked_until={stats.blocked_until_step:,})"
            )

    def sample_cell(self, num_timesteps: int) -> TaskCell:
        """サンプリング重みに基づいてセルを選択する。"""
        if not self._candidate_cells:
            raise RuntimeError("候補セルが空です。rebuild_candidate_cells() を先に呼んでください。")

        active = [c for c in self._candidate_cells
                  if self._stats[c.key()].blocked_until_step <= num_timesteps]
        if not active:
            # 全セルブロック中: 最もブロック解除が早いセルを返す
            active = sorted(
                self._candidate_cells,
                key=lambda c: self._stats[c.key()].blocked_until_step
            )[:1]

        weights = np.array([self._compute_weight(c, num_timesteps) for c in active], dtype=float)
        weights /= weights.sum()
        idx = int(np.random.choice(len(active), p=weights))
        chosen = active[idx]
        self._stats[chosen.key()].last_sample_step = num_timesteps
        return chosen

    def _compute_weight(self, cell: TaskCell, num_timesteps: int) -> float:
        stats = self._stats[cell.key()]
        min_ep_steps = self._min_episode_steps_by_phase.get(cell.enemy_phase_idx, 0)
        emergency_floor = min_ep_steps * self._emergency_length_floor_ratio

        # min_exposure_bonus
        min_exposure_bonus = 1.0 if stats.episode_count < self._min_episodes else 0.0

        # weakness_bonus
        if stats.active_score_p10 < self._target_p10 and stats.episode_length_mean >= emergency_floor:
            weakness_bonus = float(np.clip(
                (self._target_p10 - stats.active_score_p10) / max(self._target_p10, 1e-8),
                0.0, 1.0
            ))
        else:
            weakness_bonus = 0.0

        # learning_progress_weight
        learning_progress_weight = float(np.clip(
            stats.learning_progress / max(self._progress_scale, 1e-8),
            0.0, 1.0
        ))

        # regression_replay_weight
        regression_replay_weight = 0.5 * float(np.clip(
            stats.regression_score / max(self._progress_scale, 1e-8),
            0.0, 1.0
        ))

        # recency_weight
        age = num_timesteps - stats.last_sample_step
        recency_weight = min(age / max(self._recency_scale, 1), 1.0) * 0.1

        return (
            min_exposure_bonus
            + weakness_bonus
            + learning_progress_weight
            + regression_replay_weight
            + recency_weight
            + self._random_floor
        )

    def get_cell_stats(self, first_weapon_id: int, enemy_phase_idx: int) -> TaskCellStats | None:
        """現在の stage_key でセルの統計を返す。存在しない場合は None。"""
        cell = TaskCell(
            weapon_unlock_stage_key=self._current_stage_key,
            first_weapon_id=first_weapon_id,
            enemy_phase_idx=enemy_phase_idx,
        )
        return self._stats.get(cell.key())

    def on_weapon_unlock_advanced(self, event: "WeaponUnlockAdvanceEvent") -> None:
        """武器アンロック進行時に候補セルを再構築する。"""
        self.rebuild_candidate_cells(
            stage_key=event.to_stage_key,
            max_unlocked_enemy_phase_idx=self._max_unlocked_enemy_phase_idx,
            min_episode_steps_by_phase=self._min_episode_steps_by_phase,
        )

    def get_wandb_metrics(self) -> dict:
        metrics: dict = {
            "task_cell_sampler/blocked_cell_count": sum(
                1 for s in self._stats.values() if s.blocked_until_step > 0
            ),
        }
        # セル別メトリクス
        for k, stats in self._stats.items():
            cell = stats.cell
            # weapon_name: WeaponUnlockOrder から key を引く
            weapon_name = next(
                (e.key for e in WEAPON_UNLOCK_ORDER if e.weapon_id == cell.first_weapon_id),
                str(cell.first_weapon_id),
            )
            prefix = f"task_cell/{cell.weapon_unlock_stage_key}/{weapon_name}/enemy_phase/{cell.enemy_phase_idx}"
            metrics[f"{prefix}/active_score_mean"] = stats.active_score_mean
            metrics[f"{prefix}/active_score_p10"] = stats.active_score_p10
            metrics[f"{prefix}/active_score_cv"] = stats.active_score_cv
            metrics[f"{prefix}/episode_count"] = stats.episode_count
            metrics[f"{prefix}/terminated_rate"] = stats.terminated_rate
            metrics[f"{prefix}/episode_length_mean"] = stats.episode_length_mean
        return metrics

    def export_state(self) -> dict:
        stats_list = []
        for k, s in self._stats.items():
            stats_list.append({
                "cell_key": k,
                "cell": {
                    "weapon_unlock_stage_key": s.cell.weapon_unlock_stage_key,
                    "first_weapon_id": s.cell.first_weapon_id,
                    "enemy_phase_idx": s.cell.enemy_phase_idx,
                },
                "episode_count": s.episode_count,
                "recent_scores": list(s.recent_scores),
                "recent_episode_lengths": list(s.recent_episode_lengths),
                "terminated_count": s.terminated_count,
                "truncated_count": s.truncated_count,
                "previous_score_mean": s.previous_score_mean,
                "active_score_mean": s.active_score_mean,
                "active_score_p10": s.active_score_p10,
                "active_score_cv": s.active_score_cv,
                "episode_length_mean": s.episode_length_mean,
                "terminated_rate": s.terminated_rate,
                "learning_progress": s.learning_progress,
                "regression_score": s.regression_score,
                "last_sample_step": s.last_sample_step,
                "blocked_until_step": s.blocked_until_step,
            })
        return {
            "current_stage_key": self._current_stage_key,
            "max_unlocked_enemy_phase_idx": self._max_unlocked_enemy_phase_idx,
            "min_episode_steps_by_phase": {str(k): v for k, v in self._min_episode_steps_by_phase.items()},
            "stats": stats_list,
        }

    def import_state(self, state: dict) -> None:
        self._current_stage_key = state.get("current_stage_key", "WU0")
        self._max_unlocked_enemy_phase_idx = int(state.get("max_unlocked_enemy_phase_idx", 0))
        self._min_episode_steps_by_phase = {
            int(k): int(v) for k, v in state.get("min_episode_steps_by_phase", {}).items()
        }
        self._stats = {}
        for s in state.get("stats", []):
            cell = TaskCell(
                weapon_unlock_stage_key=s["cell"]["weapon_unlock_stage_key"],
                first_weapon_id=int(s["cell"]["first_weapon_id"]),
                enemy_phase_idx=int(s["cell"]["enemy_phase_idx"]),
            )
            stats = TaskCellStats(cell=cell)
            stats.episode_count = int(s.get("episode_count", 0))
            stats.recent_scores = deque(s.get("recent_scores", []), maxlen=40)
            stats.recent_episode_lengths = deque(s.get("recent_episode_lengths", []), maxlen=40)
            stats.terminated_count = int(s.get("terminated_count", 0))
            stats.truncated_count = int(s.get("truncated_count", 0))
            stats.previous_score_mean = s.get("previous_score_mean")
            stats.active_score_mean = float(s.get("active_score_mean", 0.0))
            stats.active_score_p10 = float(s.get("active_score_p10", 0.0))
            stats.active_score_cv = float(s.get("active_score_cv", 0.0))
            stats.episode_length_mean = float(s.get("episode_length_mean", 0.0))
            stats.terminated_rate = float(s.get("terminated_rate", 0.0))
            stats.learning_progress = float(s.get("learning_progress", 0.0))
            stats.regression_score = float(s.get("regression_score", 0.0))
            stats.last_sample_step = int(s.get("last_sample_step", 0))
            stats.blocked_until_step = int(s.get("blocked_until_step", 0))
            self._stats[cell.key()] = stats
        # 候補セルをstage_keyから再構築
        if self._current_stage_key:
            self.rebuild_candidate_cells(
                self._current_stage_key,
                self._max_unlocked_enemy_phase_idx,
                self._min_episode_steps_by_phase,
            )
