"""WeaponBootstrapStateModule: 武器別 Solo Bootstrap Lane と Maintenance の状態管理。"""
from __future__ import annotations

from dataclasses import dataclass, field
from games.survivors.modules.state_modules import BaseStateModule
from games.survivors.survivors_weapon_table import WeaponEntry, get_added_weapon_id, get_unlocked_weapon_ids, get_unlocked_startable_weapon_ids
from games.survivors.survivors_weapon_curriculum import WeaponType
from games.survivors.weapon_bootstrap_gate_policy import WeaponBootstrapGatePolicy, build_weapon_bootstrap_gate_policies
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
    # --- 追加: deterministic 結果を保持 ---
    last_regression_eval_step: int = -1  # regression_count を最後にインクリメントした eval_step（同一 eval で複数カウントを防ぐ）
    deterministic_p10: float | None = None  # 最新の deterministic eval active_score_p10
    deterministic_episode_length_p10: float | None = None  # 同 episode_length_p10
    deterministic_short_episode_rate: float | None = None  # 同 short_episode_rate
    deterministic_eval_step: int = 0  # eval が実施された timestep
    deterministic_task_kind: str | None = None  # 実施時の task_kind（"solo_bootstrap" 等）
    deterministic_enemy_phase_idx: int | None = None  # 実施時の enemy_phase_idx（gate 側で phase ずれ検知用）


class WeaponBootstrapStateModule(BaseStateModule):
    """武器別 Bootstrap Lane と Maintenance の状態管理。

    各武器の Bootstrap 進行状態を管理し、solo_bootstrap → integration → maintenance の
    ステータス遷移を行う。
    """

    def __init__(
        self,
        *,
        weapon_unlock_order: list[WeaponEntry],
        item_stage_key: str = "IS0",
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
        deterministic_gate_max_eval_age_steps: int = 0,  # 0 = 鮮度チェック無効（無期限）
        deterministic_gate_enemy_phase_idx: int = 2,    # gate が期待する enemy_phase_idx（現在は 2 固定）
        trait_bootstrap_weapon_keys: str | tuple[str, ...] | list[str] = "",
        trait_bootstrap_target_p10: float = 120.0,
        trait_bootstrap_min_ep_len_p10: float = 3000.0,
        trait_bootstrap_max_short_episode_rate: float = 0.05,
    ) -> None:
        self._weapon_unlock_order = weapon_unlock_order
        self._item_stage_key = item_stage_key
        self._solo_bootstrap_target_p10 = solo_bootstrap_target_p10
        self._solo_bootstrap_min_ep_len_p10 = solo_bootstrap_min_ep_len_p10
        self._solo_bootstrap_max_short_episode_rate = solo_bootstrap_max_short_episode_rate
        self._solo_bootstrap_min_episodes = solo_bootstrap_min_episodes
        self._integration_target_p10 = integration_target_p10
        self._integration_max_regression_from_solo = integration_max_regression_from_solo
        self._integration_min_episodes = integration_min_episodes
        self._maintenance_regression_ratio = maintenance_regression_ratio
        self._maintenance_min_probe_episodes = maintenance_min_probe_episodes
        self._deterministic_gate_max_eval_age_steps = deterministic_gate_max_eval_age_steps
        self._deterministic_gate_enemy_phase_idx = deterministic_gate_enemy_phase_idx

        self._gate_policies: dict[int, WeaponBootstrapGatePolicy] = build_weapon_bootstrap_gate_policies(
            weapon_unlock_order=weapon_unlock_order,
            score_solo_target_p10=solo_bootstrap_target_p10,
            score_solo_min_ep_len_p10=solo_bootstrap_min_ep_len_p10,
            score_solo_max_short_episode_rate=solo_bootstrap_max_short_episode_rate,
            integration_target_p10=integration_target_p10,
            trait_weapon_keys=trait_bootstrap_weapon_keys,
            trait_target_p10=trait_bootstrap_target_p10,
            trait_min_ep_len_p10=trait_bootstrap_min_ep_len_p10,
            trait_max_short_episode_rate=trait_bootstrap_max_short_episode_rate,
        )

        self._states: dict[int, WeaponBootstrapState] = {}
        for entry in weapon_unlock_order:
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

    def set_deterministic_result(
        self,
        weapon_id: int,
        task_kind: str,
        enemy_phase_idx: int,
        p10: float,
        episode_length_p10: float,
        short_episode_rate: float,
        num_timesteps: int,
    ) -> None:
        """BootstrapGateEvalCallback から deterministic eval 結果を注入する。

        このメソッドは BootstrapGateEvalCallback から呼ばれる（Phase B で実装）。
        Phase A では set_deterministic_result() を呼ぶ本番経路は存在しないが、
        on_episode_end() 内の gate 判定はこのメソッドで注入された結果を参照する。
        """
        state = self._states.get(weapon_id)
        if state is None:
            return
        state.deterministic_p10 = p10
        state.deterministic_episode_length_p10 = episode_length_p10
        state.deterministic_short_episode_rate = short_episode_rate
        state.deterministic_eval_step = num_timesteps
        state.deterministic_task_kind = task_kind
        state.deterministic_enemy_phase_idx = enemy_phase_idx

    def get_eval_build_policy(
        self,
        task_kind: str,
        *,
        current_stage_key: str | None = None,
    ) -> str:
        """eval 用の build_policy を task_kind と現在ステージに応じて返す。

        integration 中は garlic が maintenance 済みかつアンロック済みなら
        target_plus_anchor_if_unlocked を返す。
        solo_bootstrap / maintenance は固定。
        """
        if task_kind == "integration" and current_stage_key is not None:
            garlic_id = WeaponType.GARLIC
            garlic_status = self.get_garlic_bootstrap_status()
            unlocked_ids = get_unlocked_weapon_ids(current_stage_key, self._weapon_unlock_order)
            garlic_unlocked = garlic_id in unlocked_ids
            if garlic_status == "maintenance" and garlic_unlocked:
                return "target_plus_anchor_if_unlocked"
            return "target_only"

        # solo_bootstrap / maintenance は固定
        _POLICY = {
            "solo_bootstrap": "target_only",
            "maintenance": "target_plus_anchor_if_unlocked",
        }
        return _POLICY.get(task_kind, "target_only")

    def on_episode_end(
        self,
        cell: "TaskCell",
        stats_provider: "TaskCellSamplerStateModule",
        current_stage_key: str,
        num_timesteps: int,
    ) -> bool:
        """エピソード終了時にブートストラップ状態を更新する。

        Returns:
            bool: status 遷移が発生したかどうか（True なら候補セル再構築が必要）。
        """
        weapon_id = cell.first_weapon_id
        state = self._states.get(weapon_id)
        if state is None:
            return False

        stats = stats_provider.get_stats_for_cell(cell)
        if stats is None:
            return False

        status_changed = False

        # solo_bootstrap 完了判定 (task_kind="solo_bootstrap" かつ phase2のみ)
        # phase0/1 のエピソード結果だけで integration へ昇格しないようにする
        # 判定は deterministic 結果のみを使用（stochastic stats は episode_count のみ）
        if (state.status == "solo_bootstrap"
                and cell.task_kind == "solo_bootstrap"
                and cell.enemy_phase_idx == 2):

            det_p10 = state.deterministic_p10
            det_ep_len = state.deterministic_episode_length_p10
            det_short = state.deterministic_short_episode_rate
            det_step = state.deterministic_eval_step

            # 鮮度チェック: None ガード + task_kind チェック + phase ずれ検知
            det_fresh = (
                det_p10 is not None
                and det_ep_len is not None
                and det_short is not None
                and state.deterministic_task_kind == "solo_bootstrap"
                and state.deterministic_enemy_phase_idx == self._deterministic_gate_enemy_phase_idx
                and (
                    self._deterministic_gate_max_eval_age_steps == 0  # 0 = 鮮度チェック無効
                    or (num_timesteps - det_step) <= self._deterministic_gate_max_eval_age_steps
                )
            )

            # exposure 数のみ stochastic（生存品質は deterministic 側で判定）
            # 武器別 gate policy を参照（trait exposure gate は低 p10 閾値 + 生存品質重視）
            _policy = self._gate_policies.get(weapon_id)
            _solo_p10 = _policy.solo_target_p10 if _policy is not None else self._solo_bootstrap_target_p10
            _solo_ep_len = _policy.solo_min_ep_len_p10 if _policy is not None else self._solo_bootstrap_min_ep_len_p10
            _solo_short = _policy.solo_max_short_episode_rate if _policy is not None else self._solo_bootstrap_max_short_episode_rate
            ready = (
                stats.episode_count >= self._solo_bootstrap_min_episodes
                and det_fresh
                and det_p10 >= _solo_p10
                and det_ep_len >= _solo_ep_len
                and det_short <= _solo_short
            )
            if ready:
                state.status = "integration"
                status_changed = True
                # gate 通過時に best_phase2_p10 を deterministic 実力値に設定する
                # （solo_bootstrap 中の stochastic 更新は行わないため、ここが唯一の設定タイミング）
                state.best_phase2_p10 = det_p10
                print(
                    f"[WeaponBootstrap] solo_bootstrap → integration: weapon_id={weapon_id}, "
                    f"det_p10={det_p10:.1f}, det_ep_len={det_ep_len:.0f}, "
                    f"det_short={det_short:.3f}"
                )

        # integration 完了判定 (task_kind="integration" かつ phase2のみ)
        elif (state.status == "integration"
              and cell.task_kind == "integration"
              and cell.enemy_phase_idx == 2):

            det_p10 = state.deterministic_p10
            det_ep_len = state.deterministic_episode_length_p10
            det_short = state.deterministic_short_episode_rate
            det_step = state.deterministic_eval_step

            # 鮮度チェック: None ガード + task_kind チェック + phase ずれ検知
            det_fresh = (
                det_p10 is not None
                and det_ep_len is not None
                and det_short is not None
                and state.deterministic_task_kind == "integration"
                and state.deterministic_enemy_phase_idx == self._deterministic_gate_enemy_phase_idx
                and (
                    self._deterministic_gate_max_eval_age_steps == 0
                    or (num_timesteps - det_step) <= self._deterministic_gate_max_eval_age_steps
                )
            )

            solo_best = state.best_phase2_p10
            regression_from_solo: float | None
            if solo_best > 0 and det_fresh and det_p10 is not None:
                regression_from_solo = 1.0 - det_p10 / solo_best
            else:
                regression_from_solo = None

            _int_policy = self._gate_policies.get(weapon_id)
            _int_p10 = _int_policy.integration_target_p10 if _int_policy is not None else self._integration_target_p10
            ready = (
                stats.episode_count >= self._integration_min_episodes
                and det_fresh
                and det_p10 is not None
                and det_p10 >= _int_p10
                and solo_best > 0
                and regression_from_solo is not None
                and regression_from_solo <= self._integration_max_regression_from_solo
            )
            if ready:
                state.status = "maintenance"
                status_changed = True
                # integration → maintenance 遷移時に best_phase2_p10 を更新（高い方を採用）
                state.best_phase2_p10 = max(state.best_phase2_p10, det_p10)
                print(
                    f"[WeaponBootstrap] integration → maintenance: weapon_id={weapon_id}, "
                    f"det_p10={det_p10:.1f}, regression_from_solo={regression_from_solo:.3f}"
                )

        # maintenance 状態で task_kind="maintenance" かつ phase2 の場合のみ regression_from_best を更新
        elif (state.status == "maintenance"
              and cell.task_kind == "maintenance"
              and cell.enemy_phase_idx == 2):

            det_p10 = state.deterministic_p10
            det_step = state.deterministic_eval_step
            det_ep_len = state.deterministic_episode_length_p10
            det_short = state.deterministic_short_episode_rate

            det_fresh = (
                det_p10 is not None
                and det_ep_len is not None
                and det_short is not None
                and state.deterministic_task_kind == "maintenance"
                and state.deterministic_enemy_phase_idx == self._deterministic_gate_enemy_phase_idx
                and (
                    self._deterministic_gate_max_eval_age_steps == 0
                    or (num_timesteps - det_step) <= self._deterministic_gate_max_eval_age_steps
                )
            )

            if state.best_phase2_p10 > 0 and det_fresh:
                if stats.episode_count >= self._maintenance_min_probe_episodes:
                    # det_p10 が best を上回った場合は high-water mark を更新
                    state.best_phase2_p10 = max(state.best_phase2_p10, det_p10)
                    regression = 1.0 - det_p10 / state.best_phase2_p10
                    state.regression_from_best = regression
                    if (regression > self._maintenance_regression_ratio
                            and det_step != state.last_regression_eval_step):
                        # P2 fix: 同一 eval 結果で複数カウントを防ぐ
                        state.regression_count += 1
                        state.last_regression_eval_step = det_step
                else:
                    # episode_count が不足の場合は古い値をクリアして判定保留
                    state.regression_from_best = None
            elif not det_fresh:
                # task_kind/phase ずれ or 鮮度切れ: regression 判定不可
                state.regression_from_best = None

        return status_changed

    def is_complete_through_stage(self, target_stage_key: str) -> bool:
        unlocked_ids = get_unlocked_startable_weapon_ids(target_stage_key, self._weapon_unlock_order)
        return all(
            self._states[weapon_id].status == "maintenance"
            for weapon_id in unlocked_ids
            if weapon_id in self._states
        )

    def get_completion_snapshot(self, target_stage_key: str) -> dict:
        unlocked_ids = get_unlocked_startable_weapon_ids(target_stage_key, self._weapon_unlock_order)
        unfinished = []
        weapons = []
        for entry in self._weapon_unlock_order:
            if entry.weapon_id not in unlocked_ids:
                continue
            state = self._states.get(entry.weapon_id)
            if state is None:
                continue
            policy = self._gate_policies.get(entry.weapon_id)
            row = {
                "weapon_key": entry.key,
                "weapon_id": entry.weapon_id,
                "status": state.status,
                "gate_goal": policy.gate_goal if policy is not None else "score",
            }
            weapons.append(row)
            if state.status != "maintenance":
                unfinished.append(row)
        return {
            "target_stage_key": target_stage_key,
            "complete": len(unfinished) == 0,
            "weapons": weapons,
            "unfinished_weapons": unfinished,
        }

    def get_regressed_weapons(self, max_regression_count: int) -> list[dict]:
        rows = []
        for entry in self._weapon_unlock_order:
            state = self._states.get(entry.weapon_id)
            if state is None:
                continue
            if state.regression_count > max_regression_count:
                rows.append({
                    "weapon_key": entry.key,
                    "weapon_id": entry.weapon_id,
                    "status": state.status,
                    "regression_count": state.regression_count,
                    "regression_from_best": state.regression_from_best,
                })
        return rows

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
            "deterministic_p10": state.deterministic_p10,
            "deterministic_episode_length_p10": state.deterministic_episode_length_p10,
            "deterministic_short_episode_rate": state.deterministic_short_episode_rate,
            "deterministic_eval_step": state.deterministic_eval_step,
            "deterministic_task_kind": state.deterministic_task_kind,
            "deterministic_enemy_phase_idx": state.deterministic_enemy_phase_idx,
            "last_regression_eval_step": state.last_regression_eval_step,
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
            metrics[f"{prefix}/deterministic_p10"] = state.deterministic_p10 if state.deterministic_p10 is not None else 0.0
            metrics[f"{prefix}/deterministic_ep_len_p10"] = state.deterministic_episode_length_p10 if state.deterministic_episode_length_p10 is not None else 0.0
            metrics[f"{prefix}/deterministic_short_rate"] = state.deterministic_short_episode_rate if state.deterministic_short_episode_rate is not None else 0.0
            metrics[f"{prefix}/deterministic_eval_step"] = state.deterministic_eval_step
            metrics[f"{prefix}/deterministic_enemy_phase_idx"] = state.deterministic_enemy_phase_idx if state.deterministic_enemy_phase_idx is not None else -1
            metrics[f"{prefix}/last_regression_eval_step"] = state.last_regression_eval_step if state.last_regression_eval_step is not None else -1
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
                "deterministic_p10": state.deterministic_p10,
                "deterministic_episode_length_p10": state.deterministic_episode_length_p10,
                "deterministic_short_episode_rate": state.deterministic_short_episode_rate,
                "deterministic_eval_step": state.deterministic_eval_step,
                "deterministic_task_kind": state.deterministic_task_kind,
                "deterministic_enemy_phase_idx": state.deterministic_enemy_phase_idx,
                "last_regression_eval_step": state.last_regression_eval_step,
            })
        return {"item_stage_key": self._item_stage_key, "weapons": weapons}

    def import_state(self, state: dict) -> bool:
        """resume state を取り込む。

        Returns:
            bool: state が適用されたら True。item_stage_key mismatch で無視した
            場合は False（呼び出し側はコンストラクタで適用済みの initial_status を
            フォールバックとして利用でき、全 locked による候補セル空を防げる）。
        """
        # item stage が異なる state は流用しない（IS0 の完了状態を IS1 に持ち込まない）
        saved_item_stage = state.get("item_stage_key", "IS0")
        if saved_item_stage != self._item_stage_key:
            print(
                "[INFO] weapon_bootstrap state ignored because item_stage_key differs: "
                f"saved={saved_item_stage}, current={self._item_stage_key}"
            )
            return False

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
            # 新フィールド（欠損しても KeyError にしない）
            s.deterministic_p10 = w.get("deterministic_p10")
            s.deterministic_episode_length_p10 = w.get("deterministic_episode_length_p10")
            s.deterministic_short_episode_rate = w.get("deterministic_short_episode_rate")
            s.deterministic_eval_step = int(w.get("deterministic_eval_step", 0))
            s.deterministic_task_kind = w.get("deterministic_task_kind")  # None をデフォルト
            s.deterministic_enemy_phase_idx = w.get("deterministic_enemy_phase_idx")  # None をデフォルト
            s.last_regression_eval_step = int(w.get("last_regression_eval_step", -1))  # -1 = 未カウント
        return True
