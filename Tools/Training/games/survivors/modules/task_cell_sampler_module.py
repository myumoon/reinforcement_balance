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
    WeaponEntry,
    get_unlocked_startable_weapon_ids,
)

if TYPE_CHECKING:
    from games.survivors.modules.weapon_unlock_module import WeaponUnlockAdvanceEvent
    from games.survivors.modules.weapon_bootstrap_module import WeaponBootstrapStateModule


@dataclass(frozen=True)
class TaskCell:
    weapon_unlock_stage_key: str
    first_weapon_id: int
    enemy_phase_idx: int
    task_kind: str = "wave_main"
    build_policy: str = ""

    def key(self) -> str:
        return f"{self.task_kind}/{self.weapon_unlock_stage_key}/{self.first_weapon_id}/{self.enemy_phase_idx}"


# selected lane を index 化する際のマッピング（wandb 数値 metric 用）
LANE_INDEX: dict[str, int] = {
    "solo_bootstrap": 0,
    "weak_cells": 1,
    "maintenance": 2,
    "integration": 3,
    "fallback": 4,
}


@dataclass
class TaskCellSampleDecision:
    """sample_cell_with_lane_mix() が下した lane 選択の内訳。

    selected_lane:        実際に選ばれた lane 名。fallback パスに落ちた場合は "fallback"。
                          候補が全く無い場合は None。
    active_lanes:         lane 名 -> その lane の候補セル数。
    lane_probabilities:   lane 名 -> 正規化後の選択確率。
    selected_task_kind:   選ばれた cell の task_kind。
    selected_weapon_id:   選ばれた cell の first_weapon_id。
    selected_enemy_phase_idx: 選ばれた cell の enemy_phase_idx。
    fallback_reason:      fallback / None 選択に落ちた理由（診断用）。
    """

    selected_lane: str | None
    active_lanes: dict[str, int]
    lane_probabilities: dict[str, float]
    selected_task_kind: str | None
    selected_weapon_id: int | None
    selected_enemy_phase_idx: int | None
    fallback_reason: str | None = None


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
    episode_length_p10: float = 0.0
    short_episode_rate: float = 0.0
    recent_terminated_rate: float = 0.0
    recent_terminated_flags: deque = field(default_factory=lambda: deque(maxlen=40))


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
        weapon_unlock_order: list[WeaponEntry] | None = None,
        short_episode_steps: int = 600,
        integration_target_p10: float = 300.0,
        maintenance_regression_ratio: float = 0.35,
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
        self._weapon_unlock_order: list[WeaponEntry] = weapon_unlock_order if weapon_unlock_order is not None else WEAPON_UNLOCK_ORDER
        self._short_episode_steps = short_episode_steps
        self._integration_target_p10 = integration_target_p10
        self._maintenance_regression_ratio = maintenance_regression_ratio

        self._current_stage_key: str = "WU0"
        self._max_unlocked_enemy_phase_idx: int = 0
        self._min_episode_steps_by_phase: dict[int, int] = {}  # phase_idx -> min_episode_steps

        # セル統計: cell.key() -> TaskCellStats
        self._stats: dict[str, TaskCellStats] = {}
        # 候補セルリスト（現在サンプリング対象）
        self._candidate_cells: list[TaskCell] = []
        # 武器アンロック判定に使う enemy_phase（候補セルに必ず含める）
        self._readiness_cap_phase: int | None = None
        # 直近の sample_cell_with_lane_mix() の lane 選択内訳（selected lane logging 用）
        self._last_sample_decision: TaskCellSampleDecision | None = None

    @property
    def last_sample_decision(self) -> "TaskCellSampleDecision | None":
        """直近の sample_cell_with_lane_mix() が下した lane 選択内訳（read-only）。

        legacy sample_cell() は lane を持たないため更新しない。
        """
        return self._last_sample_decision

    def rebuild_candidate_cells(
        self,
        stage_key: str,
        max_unlocked_enemy_phase_idx: int,
        min_episode_steps_by_phase: dict[int, int],
        readiness_cap_phase: int | None = None,
    ) -> None:
        """候補セルリストを再構築する。既存 stats は保持する。

        Args:
            readiness_cap_phase: 武器アンロック判定に使う enemy_phase。
                backtrack 範囲外でも候補セルに強制追加し、stats を確実に収集する。
                None の場合は追加しない。
        """
        self._current_stage_key = stage_key
        self._max_unlocked_enemy_phase_idx = max_unlocked_enemy_phase_idx
        self._min_episode_steps_by_phase = dict(min_episode_steps_by_phase)
        if readiness_cap_phase is not None:
            self._readiness_cap_phase = readiness_cap_phase

        startable_ids = get_unlocked_startable_weapon_ids(stage_key, self._weapon_unlock_order)
        lo_phase = max(0, max_unlocked_enemy_phase_idx - self._enemy_phase_backtrack)
        hi_phase = max_unlocked_enemy_phase_idx

        # メインの候補 phase 範囲
        candidate_phases = list(range(lo_phase, hi_phase + 1))
        # readiness_cap_phase が範囲外なら強制追加（武器アンロード判定のためのstats収集）
        if (self._readiness_cap_phase is not None
                and self._readiness_cap_phase < lo_phase
                and self._readiness_cap_phase <= hi_phase):
            candidate_phases = [self._readiness_cap_phase] + candidate_phases

        new_cells: list[TaskCell] = []
        for wid in startable_ids:
            for phase_idx in candidate_phases:
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
        stats.recent_terminated_flags.append(terminated)
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
        stats.episode_length_p10 = float(np.percentile(lengths, 10)) if lengths else 0.0
        stats.short_episode_rate = sum(1 for l in lengths if l < self._short_episode_steps) / len(lengths) if lengths else 0.0
        recent_flags = list(stats.recent_terminated_flags)
        stats.recent_terminated_rate = sum(recent_flags) / len(recent_flags) if recent_flags else 0.0
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

    def get_stats_for_cell(self, cell: TaskCell) -> "TaskCellStats | None":
        """指定されたセルの統計を返す。存在しない場合は None。"""
        return self._stats.get(cell.key())

    def get_cell_stats(self, first_weapon_id: int, enemy_phase_idx: int) -> "TaskCellStats | None":
        """現在の stage_key でセルの統計を返す（wave_main互換用）。存在しない場合は None。"""
        cell = TaskCell(
            weapon_unlock_stage_key=self._current_stage_key,
            first_weapon_id=first_weapon_id,
            enemy_phase_idx=enemy_phase_idx,
            task_kind="wave_main",
        )
        return self._stats.get(cell.key())

    def on_weapon_unlock_advanced(self, event: "WeaponUnlockAdvanceEvent") -> None:
        """武器アンロック進行時に候補セルを再構築する。"""
        self.rebuild_candidate_cells(
            stage_key=event.to_stage_key,
            max_unlocked_enemy_phase_idx=self._max_unlocked_enemy_phase_idx,
            min_episode_steps_by_phase=self._min_episode_steps_by_phase,
            readiness_cap_phase=self._readiness_cap_phase,
        )

    def rebuild_bootstrap_candidate_cells(
        self,
        *,
        stage_key: str,
        max_unlocked_enemy_phase_idx: int,
        min_episode_steps_by_phase: dict[int, int],
        weapon_bootstrap: "WeaponBootstrapStateModule",
    ) -> None:
        """Bootstrap lane用の候補セルリストを再構築する。既存 stats は保持する。

        solo_bootstrap/integration/maintenance 武器ごとにセルを生成する。
        """
        self._current_stage_key = stage_key
        self._max_unlocked_enemy_phase_idx = max_unlocked_enemy_phase_idx
        self._min_episode_steps_by_phase = dict(min_episode_steps_by_phase)

        new_cells: list[TaskCell] = []

        from games.survivors.survivors_weapon_curriculum import WeaponType
        garlic_id = WeaponType.GARLIC

        # solo_bootstrap 武器: phase 0..min(max_enemy_phase, 2) のセル
        for state in weapon_bootstrap.get_weapons_by_status("solo_bootstrap"):
            max_phase = min(max_unlocked_enemy_phase_idx, 2)
            for phase_idx in range(0, max_phase + 1):
                cell = TaskCell(
                    weapon_unlock_stage_key=stage_key,
                    first_weapon_id=state.weapon_id,
                    enemy_phase_idx=phase_idx,
                    task_kind="solo_bootstrap",
                    build_policy="target_only",
                )
                new_cells.append(cell)
                if cell.key() not in self._stats:
                    self._stats[cell.key()] = TaskCellStats(cell=cell)

        # integration 武器: phase2のセル
        garlic_bootstrap_status = weapon_bootstrap.get_garlic_bootstrap_status()
        from games.survivors.survivors_weapon_table import get_unlocked_weapon_ids
        unlocked_ids = get_unlocked_weapon_ids(stage_key, self._weapon_unlock_order)
        garlic_in_unlocked = garlic_id in unlocked_ids

        for state in weapon_bootstrap.get_weapons_by_status("integration"):
            if max_unlocked_enemy_phase_idx < 2:
                # phase 2 が未解禁の場合は利用可能な最大フェーズを使う
                phase_idx = max_unlocked_enemy_phase_idx
            else:
                phase_idx = 2

            # Garlicがmaintenanceかつアンロック済みなら target_plus_anchor_if_unlocked
            if (garlic_bootstrap_status == "maintenance" and garlic_in_unlocked):
                build_policy = "target_plus_anchor_if_unlocked"
            else:
                build_policy = "target_only"

            cell = TaskCell(
                weapon_unlock_stage_key=stage_key,
                first_weapon_id=state.weapon_id,
                enemy_phase_idx=phase_idx,
                task_kind="integration",
                build_policy=build_policy,
            )
            new_cells.append(cell)
            if cell.key() not in self._stats:
                self._stats[cell.key()] = TaskCellStats(cell=cell)

        # maintenance 武器: phase2のセル
        for state in weapon_bootstrap.get_weapons_by_status("maintenance"):
            if max_unlocked_enemy_phase_idx < 2:
                phase_idx = max_unlocked_enemy_phase_idx
            else:
                phase_idx = 2

            cell = TaskCell(
                weapon_unlock_stage_key=stage_key,
                first_weapon_id=state.weapon_id,
                enemy_phase_idx=phase_idx,
                task_kind="maintenance",
                build_policy="target_plus_anchor_if_unlocked",
            )
            new_cells.append(cell)
            if cell.key() not in self._stats:
                self._stats[cell.key()] = TaskCellStats(cell=cell)

        self._candidate_cells = new_cells

    def sample_cell_with_lane_mix(
        self,
        *,
        num_timesteps: int,
        weapon_bootstrap: "WeaponBootstrapStateModule",
        sample_mix: dict[str, float],
    ) -> "TaskCell":
        """レーン別サンプリング比率でセルを選択する。

        2段階サンプリング:
        1. sample_mix の比率でレーンを選択
        2. 選ばれたレーン内の候補セルを重み付き選択
        """
        if not self._candidate_cells:
            # 候補が全く無い場合は decision を初期化してから raise する
            # （呼び出し側が last_sample_decision を参照しても None 相当の空 lane が読める）
            self._last_sample_decision = TaskCellSampleDecision(
                selected_lane=None,
                active_lanes={},
                lane_probabilities={},
                selected_task_kind=None,
                selected_weapon_id=None,
                selected_enemy_phase_idx=None,
                fallback_reason="no_candidate_cells",
            )
            raise RuntimeError("候補セルが空です。rebuild_bootstrap_candidate_cells() を先に呼んでください。")

        # weak_cells の判定
        def _is_weak(cell: TaskCell) -> bool:
            if cell.task_kind not in {"integration", "maintenance", "wave_main_probe"}:
                return False
            stats = self._stats.get(cell.key())
            if stats is None or stats.episode_count < self._min_episodes:
                return False
            if stats.blocked_until_step > num_timesteps:
                return False
            if cell.task_kind == "integration":
                return stats.active_score_p10 < self._integration_target_p10
            elif cell.task_kind == "maintenance":
                state = weapon_bootstrap._states.get(cell.first_weapon_id)
                if state is None or state.best_phase2_p10 <= 0:
                    return False
                target = state.best_phase2_p10 * (1 - self._maintenance_regression_ratio)
                return stats.active_score_p10 < target
            return False

        # レーン別セルリストを構築
        lanes: dict[str, list[TaskCell]] = {
            "solo_bootstrap": [],
            "weak_cells": [],
            "maintenance": [],
            "integration": [],
        }

        for cell in self._candidate_cells:
            stats = self._stats.get(cell.key())
            if stats is None:
                continue
            if stats.blocked_until_step > num_timesteps:
                continue

            if cell.task_kind == "solo_bootstrap":
                lanes["solo_bootstrap"].append(cell)
            elif cell.task_kind == "integration":
                lanes["integration"].append(cell)
                if _is_weak(cell):
                    lanes["weak_cells"].append(cell)
            elif cell.task_kind == "maintenance":
                lanes["maintenance"].append(cell)
                if _is_weak(cell):
                    lanes["weak_cells"].append(cell)

        # 有効なレーンのみで再正規化
        active_lanes = {k: v for k, v in lanes.items() if v and k in sample_mix and sample_mix[k] > 0}
        # active lane の候補数（selected lane logging 用）
        active_lane_counts = {k: len(v) for k, v in active_lanes.items()}
        if not active_lanes:
            # 全レーンが空: 最もブロック解除が早いか全候補から選ぶ
            active = [c for c in self._candidate_cells
                      if self._stats[c.key()].blocked_until_step <= num_timesteps]
            if not active:
                fallback_reason = "all_lanes_empty_no_unblocked_cells"
                active = sorted(
                    self._candidate_cells,
                    key=lambda c: self._stats[c.key()].blocked_until_step
                )[:1]
            else:
                fallback_reason = "all_lanes_empty"
            weights = np.array([self._compute_weight(c, num_timesteps) for c in active], dtype=float)
            weights /= weights.sum()
            idx = int(np.random.choice(len(active), p=weights))
            chosen = active[idx]
            self._stats[chosen.key()].last_sample_step = num_timesteps
            self._last_sample_decision = TaskCellSampleDecision(
                selected_lane="fallback",
                active_lanes=active_lane_counts,
                lane_probabilities={},
                selected_task_kind=chosen.task_kind,
                selected_weapon_id=chosen.first_weapon_id,
                selected_enemy_phase_idx=chosen.enemy_phase_idx,
                fallback_reason=fallback_reason,
            )
            return chosen

        # レーンの比率を正規化
        total_weight = sum(sample_mix.get(k, 0.0) for k in active_lanes)
        lane_weights = {k: sample_mix.get(k, 0.0) / total_weight for k in active_lanes}

        # レーン選択
        lane_keys = list(active_lanes.keys())
        lane_probs = np.array([lane_weights[k] for k in lane_keys], dtype=float)
        lane_probs /= lane_probs.sum()
        selected_lane_key = lane_keys[int(np.random.choice(len(lane_keys), p=lane_probs))]
        lane_cells = active_lanes[selected_lane_key]

        # レーン内でセル選択
        weights = np.array([self._compute_weight(c, num_timesteps) for c in lane_cells], dtype=float)
        weights /= weights.sum()
        idx = int(np.random.choice(len(lane_cells), p=weights))
        chosen = lane_cells[idx]
        self._stats[chosen.key()].last_sample_step = num_timesteps
        self._last_sample_decision = TaskCellSampleDecision(
            selected_lane=selected_lane_key,
            active_lanes=active_lane_counts,
            lane_probabilities={k: float(lane_weights[k]) for k in lane_keys},
            selected_task_kind=chosen.task_kind,
            selected_weapon_id=chosen.first_weapon_id,
            selected_enemy_phase_idx=chosen.enemy_phase_idx,
            fallback_reason=None,
        )
        return chosen

    def get_wandb_metrics(self, num_timesteps: int = 0) -> dict:
        metrics: dict = {
            "task_cell_sampler/blocked_cell_count": sum(
                1 for s in self._stats.values() if s.blocked_until_step > num_timesteps
            ),
        }
        # セル別メトリクス
        for k, stats in self._stats.items():
            cell = stats.cell
            # weapon_name: WeaponUnlockOrder から key を引く
            weapon_name = next(
                (e.key for e in self._weapon_unlock_order if e.weapon_id == cell.first_weapon_id),
                str(cell.first_weapon_id),
            )
            prefix = f"task_cell/{cell.task_kind}/{cell.weapon_unlock_stage_key}/{weapon_name}/enemy_phase/{cell.enemy_phase_idx}"
            metrics[f"{prefix}/active_score_mean"] = stats.active_score_mean
            metrics[f"{prefix}/active_score_p10"] = stats.active_score_p10
            metrics[f"{prefix}/active_score_cv"] = stats.active_score_cv
            metrics[f"{prefix}/episode_count"] = stats.episode_count
            metrics[f"{prefix}/terminated_rate"] = stats.terminated_rate
            metrics[f"{prefix}/episode_length_mean"] = stats.episode_length_mean
            metrics[f"{prefix}/episode_length_p10"] = stats.episode_length_p10
            metrics[f"{prefix}/short_episode_rate"] = stats.short_episode_rate
            metrics[f"{prefix}/recent_terminated_rate"] = stats.recent_terminated_rate
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
                    "task_kind": s.cell.task_kind,
                    "build_policy": s.cell.build_policy,
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
                "episode_length_p10": s.episode_length_p10,
                "short_episode_rate": s.short_episode_rate,
                "recent_terminated_rate": s.recent_terminated_rate,
                "recent_terminated_flags": list(s.recent_terminated_flags),
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
            # 旧形式keyの変換: "/" が2つ (WU0/7/2 形式) なら wave_main/ プレフィックスを付加
            raw_key = s.get("cell_key", "")
            if raw_key.count("/") == 2:
                raw_key = "wave_main/" + raw_key

            cell_data = s["cell"]
            task_kind = cell_data.get("task_kind", "wave_main")
            build_policy = cell_data.get("build_policy", "")
            cell = TaskCell(
                weapon_unlock_stage_key=cell_data["weapon_unlock_stage_key"],
                first_weapon_id=int(cell_data["first_weapon_id"]),
                enemy_phase_idx=int(cell_data["enemy_phase_idx"]),
                task_kind=task_kind,
                build_policy=build_policy,
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
            # 新フィールドが旧 state に無い場合は保存済みの履歴から再計算して乖離を防ぐ
            lengths = list(stats.recent_episode_lengths)
            if "episode_length_p10" in s:
                stats.episode_length_p10 = float(s["episode_length_p10"])
            else:
                stats.episode_length_p10 = float(np.percentile(lengths, 10)) if lengths else 0.0
            if "short_episode_rate" in s:
                stats.short_episode_rate = float(s["short_episode_rate"])
            else:
                stats.short_episode_rate = (
                    sum(1 for l in lengths if l < self._short_episode_steps) / len(lengths)
                    if lengths else 0.0
                )
            stats.recent_terminated_flags = deque(s.get("recent_terminated_flags", []), maxlen=40)
            if "recent_terminated_rate" in s:
                stats.recent_terminated_rate = float(s["recent_terminated_rate"])
            else:
                recent_flags = list(stats.recent_terminated_flags)
                if recent_flags:
                    stats.recent_terminated_rate = sum(recent_flags) / len(recent_flags)
                else:
                    # flags が旧 state に無い場合は全体 terminated_rate に fallback
                    stats.recent_terminated_rate = stats.terminated_rate
            self._stats[cell.key()] = stats
        # 候補セルをstage_keyから再構築
        if self._current_stage_key:
            self.rebuild_candidate_cells(
                self._current_stage_key,
                self._max_unlocked_enemy_phase_idx,
                self._min_episode_steps_by_phase,
            )
