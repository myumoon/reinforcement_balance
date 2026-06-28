"""TaskCellSamplerCallback: タスクセルサンプラーの SB3 コールバック。

episode単位で TaskCell を選択し、UE5 /params を送信する。
HybridCurriculumSpalfCallback と並列で動作するが、
CurriculumStateModule.on_episode_end() は Hybrid 側が担当する。
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import TYPE_CHECKING
from stable_baselines3.common.callbacks import BaseCallback
from games.survivors.modules.state_modules import EpisodeScoreTracker
from games.survivors.modules.task_cell_sampler_module import TaskCell, TaskCellSamplerStateModule
from games.survivors.modules.weapon_unlock_module import WeaponUnlockStateModule
from games.survivors.param_applier import ParamApplier
from games.survivors.survivors_weapon_table import (
    WEAPON_UNLOCK_ORDER,
    build_weapon_params_for_cell,
    resolve_weapon_unlock_order,
)
from games.survivors.survivors_curriculum import get_enemy_params_for_phase, PHASES

if TYPE_CHECKING:
    from games.survivors.hybrid_callback import HybridCurriculumSpalfCallback
    from common.wandb_logger import WandbLogger
    from games.survivors.modules.weapon_bootstrap_module import WeaponBootstrapStateModule


class TaskCellSamplerCallback(BaseCallback):
    """タスクセル選択・/params 送信・episode統計収集を担当する。

    - episode終了を独自の EpisodeScoreTracker で検出する
    - CurriculumStateModule.on_episode_end() は呼ばない（Hybrid 側が担当）
    - done=True 検出時:
        1. active_cell の stats を更新
        2. pending -> active に昇格
        3. weapon_unlock.maybe_advance() を呼ぶ
        4. 進行したら task_cell_sampler.on_weapon_unlock_advanced(event) を呼ぶ
        5. 新セルをサンプルして /params 送信、pending に設定
    """

    def __init__(
        self,
        hybrid_cb: "HybridCurriculumSpalfCallback",
        task_cell_sampler: TaskCellSamplerStateModule,
        weapon_unlock: WeaponUnlockStateModule,
        param_applier: ParamApplier,
        frame_skip: int = 4,
        alive_reward: float = 0.001,
        item_stage_key: str = "IS0",
        enemy_param_mode: str = "phase_fixed",
        pool_policy: str = "target_plus_anchor",
        wandb_logger: "WandbLogger | None" = None,
        wandb_log_freq: int = 10_000,
        log_dir: "Path | str | None" = None,
        weapon_unlock_table_name: str = "default_v1",
        weapon_bootstrap: "WeaponBootstrapStateModule | None" = None,
        weapon_bootstrap_sample_mix: "dict[str, float] | None" = None,
    ) -> None:
        super().__init__(verbose=0)
        self._hybrid_cb = hybrid_cb
        self._tcs = task_cell_sampler
        self._weapon_unlock = weapon_unlock
        self._param_applier = param_applier
        self._item_stage_key = item_stage_key
        self._enemy_param_mode = enemy_param_mode
        self._pool_policy = pool_policy
        self._wandb_logger = wandb_logger
        self._wandb_log_freq = wandb_log_freq
        self._last_wandb_log: int = -1
        self._log_dir = Path(log_dir) if log_dir is not None else None
        self._jsonl_path: Path | None = (
            self._log_dir / "task_cell_episode_metrics.jsonl" if self._log_dir else None
        )
        self._weapon_unlock_table_name = weapon_unlock_table_name
        self._weapon_bootstrap = weapon_bootstrap
        self._weapon_bootstrap_sample_mix: dict[str, float] = (
            weapon_bootstrap_sample_mix
            if weapon_bootstrap_sample_mix is not None
            else {"solo_bootstrap": 0.35, "weak_cells": 0.30, "maintenance": 0.20, "integration": 0.15}
        )

        self._score_tracker = EpisodeScoreTracker(frame_skip=frame_skip, alive_reward=alive_reward)
        self._active_cell_by_env: dict[int, TaskCell | None] = {}
        self._active_params_by_env: dict[int, dict | None] = {}
        self._pending_cell_by_env: dict[int, TaskCell | None] = {}
        self._pending_params_by_env: dict[int, dict | None] = {}
        self._status_path: Path | None = (
            self._log_dir / "task_cell_sampler_status.json" if self._log_dir else None
        )
        self._last_status_save: int = -1
        self._status_save_freq: int = 10_000

    def _on_training_start(self) -> None:
        n = self.training_env.num_envs
        self._param_applier.set_training_env(self.training_env)
        self._score_tracker.reset(n)
        self._weapon_unlock.set_start_step(self.num_timesteps)

        # 候補セルを初期化
        max_phase = self._hybrid_cb.current_phase
        min_ep_steps = {i: PHASES[i].min_episode_steps for i in range(len(PHASES))}

        if self._weapon_bootstrap is None:
            readiness_cap = min(max_phase, self._weapon_unlock._readiness_enemy_phase_cap)
            self._tcs.rebuild_candidate_cells(
                stage_key=self._weapon_unlock.current_stage_key,
                max_unlocked_enemy_phase_idx=max_phase,
                min_episode_steps_by_phase=min_ep_steps,
                readiness_cap_phase=readiness_cap,
            )
        else:
            self._tcs.rebuild_bootstrap_candidate_cells(
                stage_key=self._weapon_unlock.current_stage_key,
                max_unlocked_enemy_phase_idx=max_phase,
                min_episode_steps_by_phase=min_ep_steps,
                weapon_bootstrap=self._weapon_bootstrap,
            )

        if self._log_dir is not None:
            self._log_dir.mkdir(parents=True, exist_ok=True)

        # 初期セルをサンプルして pending に設定（active は None から始める）
        for env_idx in range(n):
            if self._weapon_bootstrap is None:
                cell = self._tcs.sample_cell(self.num_timesteps)
            else:
                cell = self._tcs.sample_cell_with_lane_mix(
                    num_timesteps=self.num_timesteps,
                    weapon_bootstrap=self._weapon_bootstrap,
                    sample_mix=self._weapon_bootstrap_sample_mix,
                )
            self._active_cell_by_env[env_idx] = None
            self._active_params_by_env[env_idx] = None
            self._pending_cell_by_env[env_idx] = cell
            params = self._build_params_for_cell(cell)
            self._pending_params_by_env[env_idx] = params
            self._param_applier.apply(params, env_idx=env_idx)

    def _on_step(self) -> bool:
        episode_results = self._score_tracker.process(self.locals["infos"])
        infos = self.locals["infos"]

        # 敵フェーズ変化検出（probe 昇格・rollback を TCS に反映）
        current_enemy_phase = self._hybrid_cb.current_phase
        if current_enemy_phase != self._tcs._max_unlocked_enemy_phase_idx:
            self._on_enemy_phase_changed(current_enemy_phase)

        for env_idx, active_score, ep_len, ep_base in episode_results:
            info = infos[env_idx] if env_idx < len(infos) else {}
            is_truncated = bool(info.get("TimeLimit.truncated", False))
            terminated = not is_truncated

            # 1. 終了 episode の active_cell で stats を更新
            active_cell = self._active_cell_by_env.get(env_idx)
            active_params = self._active_params_by_env.get(env_idx)
            if active_cell is not None:
                self._tcs.on_episode_end(
                    cell=active_cell,
                    active_score=active_score,
                    ep_len=ep_len,
                    terminated=terminated,
                    num_timesteps=self.num_timesteps,
                )
                if self._weapon_bootstrap is not None:
                    status_changed = self._weapon_bootstrap.on_episode_end(
                        cell=active_cell,
                        stats_provider=self._tcs,
                        current_stage_key=self._weapon_unlock.current_stage_key,
                        num_timesteps=self.num_timesteps,
                    )
                    # status 遷移が発生したら候補セルを再構築
                    if status_changed:
                        min_ep_steps = {i: PHASES[i].min_episode_steps for i in range(len(PHASES))}
                        self._tcs.rebuild_bootstrap_candidate_cells(
                            stage_key=self._weapon_unlock.current_stage_key,
                            max_unlocked_enemy_phase_idx=self._hybrid_cb.current_phase,
                            min_episode_steps_by_phase=min_ep_steps,
                            weapon_bootstrap=self._weapon_bootstrap,
                        )
                self._write_jsonl(env_idx, active_cell, active_score, ep_len, terminated, ep_base, active_params)

            # 2. pending -> active に昇格（pending が None の場合も active = None に昇格）
            self._active_cell_by_env[env_idx] = self._pending_cell_by_env.get(env_idx)
            self._active_params_by_env[env_idx] = self._pending_params_by_env.get(env_idx)

            # 3-4. 武器アンロック判定
            max_phase = self._hybrid_cb.current_phase
            if self._weapon_bootstrap is None:
                # target_phase: cap が候補セルに強制追加されているので min(max_phase, cap) を使える
                target_phase = min(max_phase, self._weapon_unlock._readiness_enemy_phase_cap)
                event = self._weapon_unlock.maybe_advance(
                    stats_provider=self._tcs,
                    num_timesteps=self.num_timesteps,
                    max_unlocked_enemy_phase_idx=max_phase,
                    target_enemy_phase=target_phase,
                )
            else:
                event = self._weapon_bootstrap.maybe_advance_stage(
                    weapon_unlock=self._weapon_unlock,
                    num_timesteps=self.num_timesteps,
                )
            if event is not None:
                self._tcs.on_weapon_unlock_advanced(event)
                if self._weapon_bootstrap is not None:
                    self._weapon_bootstrap.on_weapon_unlock_advanced(event)
                    # bootstrap lane の場合は候補セルを再構築
                    min_ep_steps = {i: PHASES[i].min_episode_steps for i in range(len(PHASES))}
                    self._tcs.rebuild_bootstrap_candidate_cells(
                        stage_key=self._weapon_unlock.current_stage_key,
                        max_unlocked_enemy_phase_idx=self._hybrid_cb.current_phase,
                        min_episode_steps_by_phase=min_ep_steps,
                        weapon_bootstrap=self._weapon_bootstrap,
                    )

            # 5. 次 episode 用の新セルをサンプルして pending に設定
            if self._weapon_bootstrap is None:
                next_cell = self._tcs.sample_cell(self.num_timesteps)
            else:
                next_cell = self._tcs.sample_cell_with_lane_mix(
                    num_timesteps=self.num_timesteps,
                    weapon_bootstrap=self._weapon_bootstrap,
                    sample_mix=self._weapon_bootstrap_sample_mix,
                )
            self._pending_cell_by_env[env_idx] = next_cell
            next_params = self._build_params_for_cell(next_cell)
            self._pending_params_by_env[env_idx] = next_params
            self._param_applier.apply(next_params, env_idx=env_idx)

        # status JSON 定期保存
        if (self._status_path is not None
                and self.num_timesteps - self._last_status_save >= self._status_save_freq):
            self._save_status()
            self._last_status_save = self.num_timesteps

        # W&B ログ
        if (episode_results and self._wandb_logger and self._wandb_logger.enabled
                and self.num_timesteps - self._last_wandb_log >= self._wandb_log_freq):
            self._last_wandb_log = self.num_timesteps
            metrics: dict = {}
            metrics.update(self._tcs.get_wandb_metrics(self.num_timesteps))
            metrics.update(self._weapon_unlock.get_wandb_metrics())
            if self._weapon_bootstrap is not None:
                metrics.update(self._weapon_bootstrap.get_wandb_metrics())
            # 直近サンプルされたセルのメトリクス
            if self._active_cell_by_env:
                sample_cell = next(iter(v for v in self._active_cell_by_env.values() if v is not None), None)
                if sample_cell is not None:
                    metrics["task_cell_sampler/selected_enemy_phase_idx"] = sample_cell.enemy_phase_idx
                    metrics["task_cell_sampler/selected_first_weapon_id"] = sample_cell.first_weapon_id
            self._wandb_logger.log(metrics, step=self.num_timesteps)

        return True

    def _on_enemy_phase_changed(self, new_max_phase: int) -> None:
        """敵フェーズ変化時に候補セルを再構築する。"""
        old_phase = self._tcs._max_unlocked_enemy_phase_idx
        min_ep_steps = {i: PHASES[i].min_episode_steps for i in range(len(PHASES))}
        if self._weapon_bootstrap is None:
            readiness_cap = min(new_max_phase, self._weapon_unlock._readiness_enemy_phase_cap)
            self._tcs.rebuild_candidate_cells(
                stage_key=self._weapon_unlock.current_stage_key,
                max_unlocked_enemy_phase_idx=new_max_phase,
                min_episode_steps_by_phase=min_ep_steps,
                readiness_cap_phase=readiness_cap,
            )
        else:
            self._tcs.rebuild_bootstrap_candidate_cells(
                stage_key=self._weapon_unlock.current_stage_key,
                max_unlocked_enemy_phase_idx=new_max_phase,
                min_episode_steps_by_phase=min_ep_steps,
                weapon_bootstrap=self._weapon_bootstrap,
            )
        print(
            f"[TaskCellSampler] 敵フェーズ変化を検出: {old_phase} → {new_max_phase}, "
            f"候補セルを再構築しました"
        )

    def _save_status(self) -> None:
        """task_cell_sampler_status.json に現在状態を書き出す。"""
        if self._status_path is None:
            return
        state = {
            "task_cell_sampler": self._tcs.export_state(),
            "weapon_unlock": self._weapon_unlock.export_state(),
        }
        self._status_path.write_text(
            json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        if self._weapon_bootstrap is not None and self._log_dir is not None:
            bs_path = self._log_dir / "weapon_bootstrap_status.json"
            bs_path.write_text(
                json.dumps(self._weapon_bootstrap.export_state(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def _effective_pool_policy_for_cell(self, cell: TaskCell) -> str:
        """セルの有効なpool_policyを返す。cell.build_policyが空の場合はデフォルトを使用。"""
        return cell.build_policy or self._pool_policy

    def _build_params_for_cell(self, cell: TaskCell) -> dict:
        """セルから UE5 /params 送信用 dict を構築する。"""
        weapon_params = build_weapon_params_for_cell(
            first_weapon_id=cell.first_weapon_id,
            max_unlocked_stage_key=cell.weapon_unlock_stage_key,
            item_stage_key=self._item_stage_key,
            pool_policy=self._effective_pool_policy_for_cell(cell),
            weapon_unlock_order=self._tcs._weapon_unlock_order,
        )
        if self._enemy_param_mode == "phase_fixed":
            enemy_params = get_enemy_params_for_phase(cell.enemy_phase_idx)
        else:
            raise ValueError(f"Unknown enemy_param_mode: {self._enemy_param_mode!r}")
        return {**weapon_params, **enemy_params}

    def _write_jsonl(
        self,
        env_idx: int,
        cell: TaskCell,
        active_score: float,
        ep_len: int,
        terminated: bool,
        ep_base: float,
        params: dict | None = None,
    ) -> None:
        if self._jsonl_path is None:
            return
        weapon_entry = next((e for e in self._tcs._weapon_unlock_order if e.weapon_id == cell.first_weapon_id), None)
        weapon_name = weapon_entry.key if weapon_entry else str(cell.first_weapon_id)
        enemy_params = get_enemy_params_for_phase(cell.enemy_phase_idx)
        record = {
            "step": self.num_timesteps,
            "env_idx": env_idx,
            "weapon_unlock_stage_key": cell.weapon_unlock_stage_key,
            "weapon_pool_policy": self._pool_policy,
            "weapon_item_stage_key": self._item_stage_key,
            "first_weapon_id": cell.first_weapon_id,
            "first_weapon_name": weapon_name,
            "enemy_phase_idx": cell.enemy_phase_idx,
            "enemy_params": {k: (bool(v) if isinstance(v, bool) else v) for k, v in enemy_params.items()},
            "active_score": active_score,
            "ep_len": ep_len,
            "terminated": terminated,
            "truncated": not terminated,
            "is_short_episode": ep_len < self._tcs._short_episode_steps,
            "task_cell_key": cell.key(),
            "weapon_unlock_table": self._weapon_unlock_table_name,
            "initial_weapon_slots": params.get("initial_weapon_slots") if params else None,
            "allowed_weapon_types": params.get("allowed_weapon_types") if params else None,
            # bootstrap 関連フィールド
            "task_kind": cell.task_kind,
            "build_policy": cell.build_policy,
            "effective_pool_policy": self._effective_pool_policy_for_cell(cell),
            "weapon_bootstrap_status": (
                self._weapon_bootstrap.get_weapon_snapshot(cell.first_weapon_id).get("status")
                if self._weapon_bootstrap is not None else None
            ),
            "bootstrap_lane": cell.task_kind if self._weapon_bootstrap is not None else None,
            "regression_from_best": (
                self._weapon_bootstrap.get_weapon_snapshot(cell.first_weapon_id).get("regression_from_best")
                if self._weapon_bootstrap is not None else None
            ),
        }
        with open(self._jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
