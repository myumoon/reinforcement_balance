from __future__ import annotations

from stable_baselines3.common.callbacks import BaseCallback

from games.survivors.run_event_logger import JsonlEventLogger


class SurvivorsRunSupervisorCallback(BaseCallback):
    def __init__(
        self,
        *,
        weapon_unlock,
        weapon_bootstrap,
        target_stage_key: str = "WU12",
        item_stage_key: str = "IS0",
        post_bootstrap_mode: str = "stop",
        check_freq: int = 2048,
        stage_timeout_steps: int = 2_000_000,
        max_regression_count: int = 3,
        event_logger: JsonlEventLogger | None = None,
    ) -> None:
        super().__init__(verbose=0)
        self._weapon_unlock = weapon_unlock
        self._weapon_bootstrap = weapon_bootstrap
        self._target_stage_key = target_stage_key
        self._item_stage_key = item_stage_key
        self._post_bootstrap_mode = post_bootstrap_mode
        self._check_freq = max(int(check_freq), 1)
        self._stage_timeout_steps = int(stage_timeout_steps)
        self._max_regression_count = int(max_regression_count)
        self._event_logger = event_logger.child("survivors_supervisor") if event_logger else None
        self._last_stage_key = weapon_unlock.current_stage_key
        self._stage_entered_step = 0
        self._bootstrap_complete = False
        self._post_bootstrap_transition_requested = False
        self._exit_reason: str | None = None
        self._exit_payload: dict | None = None
        # no-progress タイムアウト用トラッキング。
        # 「同一 stage への滞在時間」ではなく「未完了武器に進捗がない時間」で判定する。
        self._last_progress_step = 0
        self._last_progress_signature: tuple | None = None
        self._last_progress_snapshot: dict | None = None

    def _build_progress_snapshot(self, current_stage: str) -> dict:
        snapshot = self._weapon_bootstrap.get_unfinished_progress_snapshot(self._target_stage_key)
        snapshot["current_stage_key"] = current_stage
        return snapshot

    def _progress_signature(self, snapshot: dict) -> tuple:
        rows = []
        for row in snapshot.get("unfinished_weapons", []):
            rows.append((
                int(row.get("weapon_id", -1)),
                str(row.get("status", "")),
                row.get("best_phase2_p10"),
                row.get("deterministic_p10"),
                row.get("deterministic_episode_length_p10"),
                row.get("deterministic_short_episode_rate"),
                int(row.get("deterministic_eval_step", 0) or 0),
                row.get("deterministic_task_kind"),
                row.get("deterministic_enemy_phase_idx"),
            ))
        return (
            snapshot.get("target_stage_key"),
            snapshot.get("current_stage_key"),
            tuple(rows),
        )

    def _mark_progress(self, current_stage: str) -> None:
        snapshot = self._build_progress_snapshot(current_stage)
        self._last_progress_snapshot = snapshot
        self._last_progress_signature = self._progress_signature(snapshot)
        self._last_progress_step = self.num_timesteps

    @property
    def post_bootstrap_transition_requested(self) -> bool:
        return self._post_bootstrap_transition_requested

    @property
    def target_stage_key(self) -> str:
        return self._target_stage_key

    def _write_event(self, event: str, payload: dict) -> None:
        if self._event_logger is None:
            return
        self._event_logger.write(event, step=self.num_timesteps, payload=payload)

    def _on_training_start(self) -> None:
        self._last_stage_key = self._weapon_unlock.current_stage_key
        self._stage_entered_step = self.num_timesteps
        self._mark_progress(self._last_stage_key)

    def _on_step(self) -> bool:
        # n_calls は _on_step() 呼び出し回数のため VecEnv の n_envs 増分に依存しない
        if self.n_calls % self._check_freq != 0:
            return True

        current_stage = self._weapon_unlock.current_stage_key
        if current_stage != self._last_stage_key:
            self._write_event(
                "stage_entered",
                {
                    "from_stage_key": self._last_stage_key,
                    "to_stage_key": current_stage,
                    "stage_age_steps": self.num_timesteps - self._stage_entered_step,
                },
            )
            self._last_stage_key = current_stage
            self._stage_entered_step = self.num_timesteps
            # stage 変化は進捗の一種として扱い、no-progress タイマーをリセットする。
            self._mark_progress(current_stage)

        regressed = self._weapon_bootstrap.get_regressed_weapons(self._max_regression_count)
        if regressed:
            self._exit_reason = "bootstrap_regression_limit"
            self._exit_payload = {"regressed_weapons": regressed}
            self._write_event("blocked", {"reason": self._exit_reason, **self._exit_payload})
            return False

        # completion を timeout より先に評価する（完走済みなら timeout は無視）
        snapshot = self._weapon_bootstrap.get_completion_snapshot(self._target_stage_key)
        if snapshot["complete"]:
            snapshot["item_stage_key"] = self._item_stage_key
            self._bootstrap_complete = True
            self._write_event("bootstrap_complete", snapshot)
            if self._post_bootstrap_mode == "stop":
                self._exit_reason = "bootstrap_complete"
                self._exit_payload = snapshot
                return False
            if self._post_bootstrap_mode in ("combination_smoke", "passive_item_stage"):
                # post-bootstrap lane モードでは training を停止せず、遷移フラグのみ立てる。
                # 同一 iteration 内で TaskCellSamplerCallback が
                # post_bootstrap_transition_requested を参照し当該 lane へ切り替える
                # （callback 登録順は supervisor → TCS）。
                # イベントは初回のみ書き、以降は継続する。
                if not self._post_bootstrap_transition_requested:
                    self._post_bootstrap_transition_requested = True
                    self._write_event(
                        f"{self._post_bootstrap_mode}_transition_requested", snapshot
                    )
                return True

        # 未完了武器の進捗スナップショットを取得し、前回から変化があれば進捗ありとみなす。
        progress_snapshot = self._build_progress_snapshot(current_stage)
        progress_signature = self._progress_signature(progress_snapshot)
        if progress_signature != self._last_progress_signature:
            previous_progress_step = self._last_progress_step
            self._last_progress_snapshot = progress_snapshot
            self._last_progress_signature = progress_signature
            self._last_progress_step = self.num_timesteps
            self._write_event(
                "progress_observed",
                {
                    "stage_key": current_stage,
                    "previous_progress_step": previous_progress_step,
                    "last_progress_step": self._last_progress_step,
                    "stage_age_steps": self.num_timesteps - self._stage_entered_step,
                    "progress": progress_snapshot,
                },
            )

        stage_age = self.num_timesteps - self._stage_entered_step
        no_progress_steps = self.num_timesteps - self._last_progress_step
        if self._stage_timeout_steps > 0 and no_progress_steps > self._stage_timeout_steps:
            self._exit_reason = "bootstrap_no_progress_timeout"
            self._exit_payload = {
                "stage_key": current_stage,
                "stage_age_steps": stage_age,
                "last_progress_step": self._last_progress_step,
                "no_progress_steps": no_progress_steps,
                "timeout_steps": self._stage_timeout_steps,
                "completion": snapshot,
                "progress": progress_snapshot,
            }
            self._write_event("blocked", {"reason": self._exit_reason, **self._exit_payload})
            return False

        return True

    def export_state(self) -> dict:
        return {
            "target_stage_key": self._target_stage_key,
            "item_stage_key": self._item_stage_key,
            "post_bootstrap_mode": self._post_bootstrap_mode,
            "last_stage_key": self._last_stage_key,
            "stage_entered_step": self._stage_entered_step,
            "stage_age_steps": self.num_timesteps - self._stage_entered_step,
            "bootstrap_complete": self._bootstrap_complete,
            "post_bootstrap_transition_requested": self._post_bootstrap_transition_requested,
            "exit_reason": self._exit_reason,
            "exit_payload": self._exit_payload,
            "last_progress_step": self._last_progress_step,
            "no_progress_steps": self.num_timesteps - self._last_progress_step,
            "last_progress_snapshot": self._last_progress_snapshot,
        }
