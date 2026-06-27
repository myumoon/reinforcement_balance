"""WeaponUnlockStateModule: タスクセルサンプラー用の武器アンロック状態管理。

CurriculumStateModule（敵Phase管理）とは独立した武器アンロック軸を管理する。
/params は送信しない。候補セルへの追加は TaskCellSamplerCallback が担う。
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, runtime_checkable
from games.survivors.modules.state_modules import BaseStateModule
from games.survivors.survivors_weapon_table import WEAPON_UNLOCK_ORDER, WeaponEntry, get_added_weapon_id


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
        weapon_unlock_max_terminated_rate: float | None = 0.5,
        weapon_unlock_min_steps: int = 100_000,
        weapon_unlock_readiness_enemy_phase_cap: int = 2,
        weapon_unlock_order: list[WeaponEntry] | None = None,
        weapon_unlock_min_ep_len_p10_by_phase: dict[int, int] | None = None,
        weapon_unlock_max_short_episode_rate: float | None = 0.15,
    ) -> None:
        self._weapon_unlock_order: list[WeaponEntry] = weapon_unlock_order if weapon_unlock_order is not None else WEAPON_UNLOCK_ORDER
        self._stage_order: int = self._key_to_order(initial_stage_key)
        self._min_episodes = weapon_unlock_min_episodes
        self._target_p10 = weapon_unlock_target_p10
        self._max_terminated_rate: float | None = weapon_unlock_max_terminated_rate
        self._min_steps = weapon_unlock_min_steps
        self._readiness_enemy_phase_cap = weapon_unlock_readiness_enemy_phase_cap
        self._min_ep_len_p10_by_phase: dict[int, int] = (
            weapon_unlock_min_ep_len_p10_by_phase
            if weapon_unlock_min_ep_len_p10_by_phase is not None
            else {0: 600, 1: 900, 2: 1200, 3: 1200}
        )
        self._max_short_episode_rate = weapon_unlock_max_short_episode_rate
        self._last_advance_step: int | None = None
        self._start_step: int = 0  # training開始ステップ（import_state前に設定）
        self._events: list[dict] = []
        self._state_restored: bool = False  # import_state() 後に True になる

    def _key_to_order(self, stage_key: str) -> int:
        for e in self._weapon_unlock_order:
            if e.unlock_stage_key == stage_key:
                return e.unlock_order
        raise ValueError(f"Unknown stage_key: {stage_key!r}")

    def _order_to_key(self, order: int) -> str:
        return self._weapon_unlock_order[order].unlock_stage_key

    @property
    def current_stage_key(self) -> str:
        return self._order_to_key(self._stage_order)

    @property
    def current_stage_order(self) -> int:
        return self._stage_order

    @property
    def is_final_stage(self) -> bool:
        return self._stage_order >= len(self._weapon_unlock_order) - 1

    def set_start_step(self, num_timesteps: int) -> None:
        """訓練開始時のステップ数を記録する（min_steps 判定に使用）。resume 時はスキップする。"""
        if self._state_restored:
            return
        self._start_step = num_timesteps

    def maybe_advance(
        self,
        stats_provider: TaskCellStatsProvider,
        num_timesteps: int,
        max_unlocked_enemy_phase_idx: int,
        target_enemy_phase: int | None = None,
    ) -> WeaponUnlockAdvanceEvent | None:
        """次武器の解禁条件を確認し、条件を満たせば進行してイベントを返す。

        Args:
            target_enemy_phase: 解禁判定に使う enemy_phase。省略時は
                min(max_unlocked_enemy_phase_idx, readiness_cap) を使う。
                呼び出し側でバックトラック範囲を考慮した値を渡すことを推奨する。
        """
        if self.is_final_stage:
            return None

        # 現在ステージの武器（= 候補セルに必ず存在する）の stats を確認
        current_weapon_id = get_added_weapon_id(self.current_stage_key, self._weapon_unlock_order)

        # 最低ステップ数チェック
        ref_step = self._last_advance_step if self._last_advance_step is not None else self._start_step
        if num_timesteps - ref_step < self._min_steps:
            return None

        # 対象 enemy_phase: 呼び出し側が指定しない場合は cap と max の小さい方
        if target_enemy_phase is None:
            target_enemy_phase = min(max_unlocked_enemy_phase_idx, self._readiness_enemy_phase_cap)
        stats = stats_provider.get_cell_stats(current_weapon_id, target_enemy_phase)
        if stats is None:
            return None

        # min_ep_len_p10 を target_enemy_phase から決定（定義外は最大定義phaseの値にfallback）
        defined_phases = sorted(self._min_ep_len_p10_by_phase.keys())
        if not defined_phases:
            raise ValueError("weapon_unlock_min_ep_len_p10_by_phase が空です")
        if target_enemy_phase in self._min_ep_len_p10_by_phase:
            min_ep_len_p10 = self._min_ep_len_p10_by_phase[target_enemy_phase]
        else:
            min_ep_len_p10 = self._min_ep_len_p10_by_phase[max(defined_phases)]

        ready = (
            stats.episode_count >= self._min_episodes
            and stats.active_score_p10 >= self._target_p10
            and stats.episode_length_p10 >= min_ep_len_p10
            and (self._max_short_episode_rate is None or stats.short_episode_rate <= self._max_short_episode_rate)
            and (self._max_terminated_rate is None or stats.recent_terminated_rate <= self._max_terminated_rate)
        )
        if not ready:
            cell_key = f"{self.current_stage_key}/{current_weapon_id}/{target_enemy_phase}"
            reasons = []
            if stats.episode_count < self._min_episodes:
                reasons.append(f"  episodes={stats.episode_count} < {self._min_episodes}")
            if stats.active_score_p10 < self._target_p10:
                reasons.append(f"  p10={stats.active_score_p10:.1f} < {self._target_p10:.1f}")
            if stats.episode_length_p10 < min_ep_len_p10:
                reasons.append(f"  ep_len_p10={stats.episode_length_p10:.0f} < {min_ep_len_p10}")
            if self._max_short_episode_rate is not None and stats.short_episode_rate > self._max_short_episode_rate:
                reasons.append(f"  short_rate={stats.short_episode_rate:.2f} > {self._max_short_episode_rate:.2f}")
            if self._max_terminated_rate is not None and stats.recent_terminated_rate > self._max_terminated_rate:
                reasons.append(f"  recent_terminated_rate={stats.recent_terminated_rate:.2f} > {self._max_terminated_rate:.2f}")
            if reasons:
                print(f"[WeaponUnlock] not ready {cell_key}:\n" + "\n".join(reasons))
            return None

        from_key = self.current_stage_key
        next_order = self._stage_order + 1
        next_entry = self._weapon_unlock_order[next_order]
        self._stage_order = next_order
        self._last_advance_step = num_timesteps
        to_key = self.current_stage_key

        event = WeaponUnlockAdvanceEvent(
            from_stage_key=from_key,
            to_stage_key=to_key,
            new_weapon_id=next_entry.weapon_id,
            step=num_timesteps,
        )
        self._events.append({
            "from_stage_key": from_key,
            "to_stage_key": to_key,
            "new_weapon_id": next_entry.weapon_id,
            "step": num_timesteps,
        })
        print(
            f"[WeaponUnlock] 武器アンロック: {from_key} -> {to_key} "
            f"(new_weapon_id={next_entry.weapon_id}, step={num_timesteps:,})"
        )
        return event

    def get_wandb_metrics(self) -> dict:
        entry = self._weapon_unlock_order[self._stage_order]
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
        self._state_restored = True
