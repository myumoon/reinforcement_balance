"""WeaponBootstrapStateModule: 武器別 Solo Bootstrap Lane と Maintenance の状態管理。"""
from __future__ import annotations
from dataclasses import dataclass, field
from games.survivors.modules.state_modules import BaseStateModule
from games.survivors.survivors_weapon_table import WeaponEntry, get_added_weapon_id
from games.survivors.survivors_weapon_curriculum import WeaponType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from games.survivors.modules.task_cell_sampler_module import TaskCell, TaskCellSamplerStateModule
    from games.survivors.modules.weapon_unlock_module import WeaponUnlockStateModule, WeaponUnlockAdvanceEvent


@dataclass
class WeaponBootstrapState:
    weapon_id: int
    status: str  # "locked", "solo_bootstrap", "integration", "maintenance"
    best_phase2_p10: float = 0.0
    regression_from_best: float | None = None
    last_probe_step: int = 0
    regression_count: int = 0


class WeaponBootstrapStateModule(BaseStateModule):
    """武器別 Bootstrap Lane と Maintenance の状態管理。

    各武器の Bootstrap 進行状態を管理し、solo_bootstrap → integration → maintenance の
    ステータス遷移を行う。
    """

    def __init__(
        self,
        *,
        weapon_unlock_order: list[WeaponEntry],
        initial_status: dict[str, str] | None = None,
        initial_best_phase2_p10: dict[str, float] | None = None,
        solo_bootstrap_target_p10: float = 300.0,
        solo_bootstrap_min_ep_len_p10: float = 1200.0,
        solo_bootstrap_max_short_episode_rate: float = 0.15,
        solo_bootstrap_min_episodes: int = 40,
        integration_target_p10: float = 300.0,
        integration_max_regression_from_solo: float = 0.35,
        integration_min_episodes: int = 40,
        maintenance_regression_ratio: float = 0.35,
        maintenance_min_probe_episodes: int = 20,
    ) -> None:
        self._weapon_unlock_order = weapon_unlock_order
        self._solo_bootstrap_target_p10 = solo_bootstrap_target_p10
        self._solo_bootstrap_min_ep_len_p10 = solo_bootstrap_min_ep_len_p10
        self._solo_bootstrap_max_short_episode_rate = solo_bootstrap_max_short_episode_rate
        self._solo_bootstrap_min_episodes = solo_bootstrap_min_episodes
        self._integration_target_p10 = integration_target_p10
        self._integration_max_regression_from_solo = integration_max_regression_from_solo
        self._integration_min_episodes = integration_min_episodes
        self._maintenance_regression_ratio = maintenance_regression_ratio
        self._maintenance_min_probe_episodes = maintenance_min_probe_episodes

        # 全武器の状態を weapon_unlock_order順に初期化
        self._states: dict[int, WeaponBootstrapState] = {}
        for entry in weapon_unlock_order:
            # initial_status のキーは entry.key と同じ文字列 (例: "garlic", "king_bible")
            status = "locked"
            if initial_status is not None and entry.key in initial_status:
                status = initial_status[entry.key]

            best_p10 = 0.0
            if initial_best_phase2_p10 is not None and entry.key in initial_best_phase2_p10:
                best_p10 = initial_best_phase2_p10[entry.key]

            self._states[entry.weapon_id] = WeaponBootstrapState(
                weapon_id=entry.weapon_id,
                status=status,
                best_phase2_p10=best_p10,
            )

    def on_episode_end(
        self,
        cell: "TaskCell",
        stats_provider: "TaskCellSamplerStateModule",
        current_stage_key: str,
        num_timesteps: int,
    ) -> None:
        """エピソード終了時にブートストラップ状態を更新する。"""
        weapon_id = cell.first_weapon_id
        state = self._states.get(weapon_id)
        if state is None:
            return

        stats = stats_provider.get_stats_for_cell(cell)
        if stats is None:
            return

        # 更新前の best_phase2_p10 を保存（regression計算に使用）
        prev_best_phase2_p10 = state.best_phase2_p10

        # phase2エピソードの場合のみ best_phase2_p10 を更新
        if cell.enemy_phase_idx == 2:
            if stats.active_score_p10 > state.best_phase2_p10:
                state.best_phase2_p10 = stats.active_score_p10

        # solo_bootstrap 完了判定 (phase2のみ)
        # phase0/1 のエピソード結果だけで integration へ昇格しないようにする
        if state.status == "solo_bootstrap" and cell.enemy_phase_idx == 2:
            ready = (
                stats.episode_count >= self._solo_bootstrap_min_episodes
                and stats.active_score_p10 >= self._solo_bootstrap_target_p10
                and stats.episode_length_p10 >= self._solo_bootstrap_min_ep_len_p10
                and stats.short_episode_rate <= self._solo_bootstrap_max_short_episode_rate
            )
            if ready:
                state.status = "integration"
                print(
                    f"[WeaponBootstrap] solo_bootstrap → integration: weapon_id={weapon_id}, "
                    f"p10={stats.active_score_p10:.1f}, ep_len_p10={stats.episode_length_p10:.0f}, "
                    f"short_rate={stats.short_episode_rate:.3f}"
                )

        # integration 完了判定
        elif state.status == "integration":
            solo_best = state.best_phase2_p10
            if solo_best > 0:
                regression_from_solo = 1.0 - stats.active_score_p10 / solo_best
            else:
                regression_from_solo = None

            ready = (
                stats.episode_count >= self._integration_min_episodes
                and stats.active_score_p10 >= self._integration_target_p10
                and solo_best > 0
                and regression_from_solo is not None
                and regression_from_solo <= self._integration_max_regression_from_solo
            )
            if ready:
                state.status = "maintenance"
                print(
                    f"[WeaponBootstrap] integration → maintenance: weapon_id={weapon_id}, "
                    f"p10={stats.active_score_p10:.1f}, regression_from_solo={regression_from_solo:.3f}"
                )

        # maintenance 状態で phase2エピソードの場合 regression_from_best を更新
        elif state.status == "maintenance":
            if cell.enemy_phase_idx == 2 and prev_best_phase2_p10 > 0:
                regression = 1.0 - stats.active_score_p10 / prev_best_phase2_p10
                state.regression_from_best = regression
                if regression > self._maintenance_regression_ratio:
                    state.regression_count += 1

    def maybe_advance_stage(
        self,
        *,
        weapon_unlock: "WeaponUnlockStateModule",
        num_timesteps: int,
    ) -> "WeaponUnlockAdvanceEvent | None":
        """現在stageの武器がmaintenanceになったら次stageへ進める。"""
        current_stage_key = weapon_unlock.current_stage_key
        current_weapon_id = get_added_weapon_id(current_stage_key, self._weapon_unlock_order)
        state = self._states.get(current_weapon_id)
        if state is None or state.status != "maintenance":
            return None
        return weapon_unlock.advance_to_next_stage(num_timesteps)

    def on_weapon_unlock_advanced(self, event: "WeaponUnlockAdvanceEvent") -> None:
        """武器アンロック進行時に新規武器のステータスを solo_bootstrap にする。

        既存武器のstatusは変更しない。
        """
        state = self._states.get(event.new_weapon_id)
        if state is not None and state.status == "locked":
            state.status = "solo_bootstrap"
            print(
                f"[WeaponBootstrap] locked → solo_bootstrap: weapon_id={event.new_weapon_id}, "
                f"stage={event.from_stage_key} → {event.to_stage_key}"
            )

    def get_weapon_snapshot(self, weapon_id: int) -> dict:
        """指定武器の状態スナップショットを返す。"""
        state = self._states.get(weapon_id)
        if state is None:
            return {}
        return {
            "weapon_id": state.weapon_id,
            "status": state.status,
            "best_phase2_p10": state.best_phase2_p10,
            "regression_from_best": state.regression_from_best,
            "regression_count": state.regression_count,
            "last_probe_step": state.last_probe_step,
        }

    def get_weapons_by_status(self, status: str) -> list[WeaponBootstrapState]:
        """指定ステータスの武器一覧を返す。"""
        return [s for s in self._states.values() if s.status == status]

    def get_garlic_bootstrap_status(self) -> str | None:
        """Garlic武器のブートストラップステータスを返す。"""
        garlic_id = WeaponType.GARLIC
        state = self._states.get(garlic_id)
        return state.status if state is not None else None

    def get_wandb_metrics(self) -> dict:
        metrics: dict = {}
        for weapon_id, state in self._states.items():
            entry = next((e for e in self._weapon_unlock_order if e.weapon_id == weapon_id), None)
            weapon_name = entry.key if entry else str(weapon_id)
            prefix = f"weapon_bootstrap/{weapon_name}"
            status_int = {"locked": 0, "solo_bootstrap": 1, "integration": 2, "maintenance": 3}.get(state.status, -1)
            metrics[f"{prefix}/status"] = status_int
            metrics[f"{prefix}/best_phase2_p10"] = state.best_phase2_p10
            metrics[f"{prefix}/regression_from_best"] = state.regression_from_best if state.regression_from_best is not None else 0.0
            metrics[f"{prefix}/regression_count"] = state.regression_count
        return metrics

    def export_state(self) -> dict:
        weapons = []
        for entry in self._weapon_unlock_order:
            state = self._states.get(entry.weapon_id)
            if state is None:
                continue
            weapons.append({
                "weapon_id": state.weapon_id,
                "weapon_key": entry.key,
                "status": state.status,
                "best_phase2_p10": state.best_phase2_p10,
                "regression_from_best": state.regression_from_best,
                "regression_count": state.regression_count,
                "last_probe_step": state.last_probe_step,
            })
        return {"weapons": weapons}

    def import_state(self, state: dict) -> None:
        # weapon_key または weapon_id でマッピング
        key_to_id = {e.key: e.weapon_id for e in self._weapon_unlock_order}
        for w in state.get("weapons", []):
            weapon_id = w.get("weapon_id")
            weapon_key = w.get("weapon_key")
            # weapon_key からのマッピングを優先
            if weapon_key is not None and weapon_key in key_to_id:
                weapon_id = key_to_id[weapon_key]
            if weapon_id is None:
                continue
            weapon_id = int(weapon_id)
            if weapon_id not in self._states:
                continue
            s = self._states[weapon_id]
            s.status = w.get("status", "locked")
            s.best_phase2_p10 = float(w.get("best_phase2_p10", 0.0))
            s.regression_from_best = w.get("regression_from_best")
            s.regression_count = int(w.get("regression_count", 0))
            s.last_probe_step = int(w.get("last_probe_step", 0))
