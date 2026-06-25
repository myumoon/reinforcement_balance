"""WeaponUnlockStateModule: タスクセルサンプラー用の武器アンロック状態管理。

CurriculumStateModule（敵Phase管理）とは独立した武器アンロック軸を管理する。
/params は送信しない。候補セルへの追加は TaskCellSamplerCallback が担う。
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, runtime_checkable
from games.survivors.modules.state_modules import BaseStateModule
from games.survivors.survivors_weapon_table import WEAPON_UNLOCK_ORDER


@dataclass
class WeaponUnlockAdvanceEvent:
    from_stage_key: str
    to_stage_key: str
    new_weapon_id: int
    step: int


@runtime_checkable
class TaskCellStatsProvider(Protocol):
    def get_cell_stats(self, first_weapon_id: int, enemy_phase_idx: int): ...


class WeaponUnlockStateModule(BaseStateModule):
    """武器アンロック段階の状態管理。

    maybe_advance() はタスクセル統計を参照して次武器の解禁可否を判定する。
    進行した場合は WeaponUnlockAdvanceEvent を返す。
    """

    def __init__(
        self,
        initial_stage_key: str = "WU0",
        weapon_unlock_min_episodes: int = 30,
        weapon_unlock_target_p10: float = 300.0,
        weapon_unlock_max_terminated_rate: float = 0.5,
        weapon_unlock_min_steps: int = 100_000,
        weapon_unlock_readiness_enemy_phase_cap: int = 2,
    ) -> None:
        self._stage_order: int = self._key_to_order(initial_stage_key)
        self._min_episodes = weapon_unlock_min_episodes
        self._target_p10 = weapon_unlock_target_p10
        self._max_terminated_rate = weapon_unlock_max_terminated_rate
        self._min_steps = weapon_unlock_min_steps
        self._readiness_enemy_phase_cap = weapon_unlock_readiness_enemy_phase_cap
        self._last_advance_step: int | None = None
        self._start_step: int = 0  # training開始ステップ（import_state前に設定）
        self._events: list[dict] = []

    @staticmethod
    def _key_to_order(stage_key: str) -> int:
        for e in WEAPON_UNLOCK_ORDER:
            if e.unlock_stage_key == stage_key:
                return e.unlock_order
        raise ValueError(f"Unknown stage_key: {stage_key!r}")

    @staticmethod
    def _order_to_key(order: int) -> str:
        return WEAPON_UNLOCK_ORDER[order].unlock_stage_key

    @property
    def current_stage_key(self) -> str:
        return self._order_to_key(self._stage_order)

    @property
    def current_stage_order(self) -> int:
        return self._stage_order

    @property
    def is_final_stage(self) -> bool:
        return self._stage_order >= len(WEAPON_UNLOCK_ORDER) - 1

    def set_start_step(self, num_timesteps: int) -> None:
        """訓練開始時のステップ数を記録する（min_steps 判定に使用）。"""
        self._start_step = num_timesteps

    def maybe_advance(
        self,
        stats_provider: TaskCellStatsProvider,
        num_timesteps: int,
        max_unlocked_enemy_phase_idx: int,
    ) -> WeaponUnlockAdvanceEvent | None:
        """次武器の解禁条件を確認し、条件を満たせば進行してイベントを返す。"""
        if self.is_final_stage:
            return None

        next_order = self._stage_order + 1
        next_entry = WEAPON_UNLOCK_ORDER[next_order]
        new_weapon_id = next_entry.weapon_id

        # 最低ステップ数チェック
        ref_step = self._last_advance_step if self._last_advance_step is not None else self._start_step
        if num_timesteps - ref_step < self._min_steps:
            return None

        # 対象 enemy_phase: max_unlocked_enemy_phase_idx と cap の小さい方
        target_phase = min(max_unlocked_enemy_phase_idx, self._readiness_enemy_phase_cap)
        stats = stats_provider.get_cell_stats(new_weapon_id, target_phase)
        if stats is None:
            return None

        ready = (
            stats.episode_count >= self._min_episodes
            and stats.active_score_p10 >= self._target_p10
            and stats.terminated_rate <= self._max_terminated_rate
        )
        if not ready:
            return None

        from_key = self.current_stage_key
        self._stage_order = next_order
        self._last_advance_step = num_timesteps
        to_key = self.current_stage_key

        event = WeaponUnlockAdvanceEvent(
            from_stage_key=from_key,
            to_stage_key=to_key,
            new_weapon_id=new_weapon_id,
            step=num_timesteps,
        )
        self._events.append({
            "from_stage_key": from_key,
            "to_stage_key": to_key,
            "new_weapon_id": new_weapon_id,
            "step": num_timesteps,
        })
        print(
            f"[WeaponUnlock] 武器アンロック: {from_key} -> {to_key} "
            f"(weapon_id={new_weapon_id}, step={num_timesteps:,})"
        )
        return event

    def get_wandb_metrics(self) -> dict:
        entry = WEAPON_UNLOCK_ORDER[self._stage_order]
        return {
            "weapon_unlock/stage_order": self._stage_order,
            "weapon_unlock/stage_key_id": self._stage_order,
            "weapon_unlock/unlocked_weapon_count": self._stage_order + 1,
            "weapon_unlock/current_added_weapon_id": entry.weapon_id,
        }

    def export_state(self) -> dict:
        return {
            "stage_order": self._stage_order,
            "stage_key": self.current_stage_key,
            "last_advance_step": self._last_advance_step,
            "start_step": self._start_step,
            "events": self._events,
        }

    def import_state(self, state: dict) -> None:
        self._stage_order = int(state.get("stage_order", 0))
        self._last_advance_step = state.get("last_advance_step")
        self._start_step = int(state.get("start_step", 0))
        self._events = list(state.get("events", []))
