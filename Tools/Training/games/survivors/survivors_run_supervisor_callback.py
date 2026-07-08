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

    @property
    def post_bootstrap_transition_requested(self) -> bool:
        return self._post_bootstrap_transition_requested

    def _write_event(self, event: str, payload: dict) -> None:
        if self._event_logger is None:
            return
        self._event_logger.write(event, step=self.num_timesteps, payload=payload)

    def _on_training_start(self) -> None:
        self._last_stage_key = self._weapon_unlock.current_stage_key
        self._stage_entered_step = self.num_timesteps

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

        regressed = self._weapon_bootstrap.get_regressed_weapons(self._max_regression_count)
        if regressed:
            self._exit_reason = "bootstrap_regression_limit"
            self._exit_payload = {"regressed_weapons": regressed}
            self._write_event("blocked", {"reason": self._exit_reason, **self._exit_payload})
            return False

        # completion を timeout より先に評価する（完走済みなら timeout は無視）
        snapshot = self._weapon_bootstrap.get_completion_snapshot(self._target_stage_key)
        if snapshot["complete"]:
            self._bootstrap_complete = True
            self._write_event("bootstrap_complete", snapshot)
            if self._post_bootstrap_mode == "stop":
                self._exit_reason = "bootstrap_complete"
                self._exit_payload = snapshot
                return False
            if self._post_bootstrap_mode == "combination_smoke":
                # Phase B が未実装のため training を停止する。
                # train.py は exit_reason と post_bootstrap_transition_requested を確認し遷移を実施する。
                self._post_bootstrap_transition_requested = True
                self._exit_reason = "bootstrap_complete_combination_smoke_requested"
                self._exit_payload = snapshot
                return False

        stage_age = self.num_timesteps - self._stage_entered_step
        if self._stage_timeout_steps > 0 and stage_age > self._stage_timeout_steps:
            self._exit_reason = "bootstrap_stage_timeout"
            self._exit_payload = {
                "stage_key": current_stage,
                "stage_age_steps": stage_age,
                "timeout_steps": self._stage_timeout_steps,
                "completion": snapshot,
            }
            self._write_event("blocked", {"reason": self._exit_reason, **self._exit_payload})
            return False

        return True

    def export_state(self) -> dict:
        return {
            "target_stage_key": self._target_stage_key,
            "post_bootstrap_mode": self._post_bootstrap_mode,
            "last_stage_key": self._last_stage_key,
            "stage_entered_step": self._stage_entered_step,
            "stage_age_steps": self.num_timesteps - self._stage_entered_step,
            "bootstrap_complete": self._bootstrap_complete,
            "post_bootstrap_transition_requested": self._post_bootstrap_transition_requested,
            "exit_reason": self._exit_reason,
            "exit_payload": self._exit_payload,
        }
