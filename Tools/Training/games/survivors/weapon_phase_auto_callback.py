"""
WeaponPhaseAutoCallback: WeaponPhaseAutoStateModule を SB3 コールバックとして動かす薄いラッパー。

--weapon-phase auto 指定時のみ train.py から登録される。
状態管理（停滞検出・フェーズ昇格・カリキュラム降格）は WeaponPhaseAutoStateModule が担い、
本クラスは SB3 の _on_step フックと set_params 送信のみを担当する。

resume 対応:
  train.py が train_status_{step}_steps.json から "weapon_phase_auto" キーを読み取り、
  module.import_state() を呼んだ後でコールバックを登録する。
  _on_training_start では restore は行わず、停滞タイマーのリセットと初回 set_params のみ実行する。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from stable_baselines3.common.callbacks import BaseCallback

from games.survivors.survivors_weapon_curriculum import get_params_for_phase
from games.survivors.state_modules import WeaponPhaseAutoStateModule

if TYPE_CHECKING:
    from common.wandb_logger import WandbLogger


class WeaponPhaseAutoCallback(BaseCallback):
    """WeaponPhaseAutoStateModule の SB3 コールバックラッパー。

    - _on_training_start: 停滞タイマーリセット → 初回 set_params
    - _on_step:           module.on_step() を呼び、昇格イベント時に set_params を送信
                          遷移フェーズ中は weapon_update_freq ステップごとに weapon_weights を更新
    """

    def __init__(
        self,
        module: WeaponPhaseAutoStateModule,
        weapon_update_freq: int = 10_000,
        wandb_logger: "WandbLogger | None" = None,
        wandb_log_freq: int = 10_000,
    ):
        """
        Args:
            module: WeaponPhaseAutoStateModule インスタンス（状態管理を委譲）。
            weapon_update_freq: 遷移フェーズ中に weapon_weights を更新するステップ間隔。
            wandb_logger: W&B ログ用ロガー（None の場合はログなし）。
            wandb_log_freq: 武器カリキュラムメトリクスの定期ログ間隔（ステップ数）。
        """
        super().__init__(verbose=0)
        self._module = module
        self._weapon_update_freq = max(weapon_update_freq, 1)
        self._last_weapon_update: int = 0
        self._wandb_logger = wandb_logger
        self._wandb_log_freq = max(wandb_log_freq, 1)
        self._last_wandb_log: int = -1
        if wandb_logger:
            wandb_logger.add_metric_prefix("weapon_curriculum/")

    # ------------------------------------------------------------------ #
    # SB3 callbacks                                                        #
    # ------------------------------------------------------------------ #

    def _on_training_start(self) -> None:
        # module の状態は train.py が import_state() で復元済みの場合がある。
        # 復元済み（resume）の場合は max_curriculum_phase / stagnation_start_step を
        # 上書きしないようにタイマーリセットをスキップする。
        # 新規訓練時のみ停滞タイマーをリセットする。
        if not self._module._state_restored:
            self._module.reset_stagnation_timer(self.num_timesteps)
        self._last_weapon_update = self.num_timesteps

        # 遷移フェーズは phase_start_step からの elapsed で補間位置を復元
        elapsed = self._module.get_transition_elapsed(self.num_timesteps)
        self._apply_weapon_phase(elapsed)

        # 初回 W&B ログ
        self._log_wandb_metrics()

        print(
            f"[WeaponPhaseAuto] 有効 phase={self._module.current_phase_key}, "
            f"stagnation_steps={self._module._stagnation_steps:,}, "
            f"weapon_update_freq={self._weapon_update_freq:,}"
        )

    def _on_step(self) -> bool:
        # v09: current_curriculum_phase を curriculum から取得して on_step() に渡す
        current_curriculum_phase = self._module._curriculum.current_phase
        new_phase_key = self._module.on_step(
            self.num_timesteps, current_curriculum_phase=current_curriculum_phase
        )
        if new_phase_key is not None:
            # 武器フェーズ昇格 → 新フェーズの初期パラメータを送信
            self._apply_weapon_phase(elapsed=0)
            self._last_weapon_update = self.num_timesteps
            # フェーズ遷移時は W&B にログを記録
            self._log_wandb_metrics()
        elif (self._module.is_transition
              and self.num_timesteps - self._last_weapon_update >= self._weapon_update_freq):
            # v09 では遷移フェーズは廃止されているため通常ここには到達しない
            # backward compatibility のため残す
            elapsed = self._module.get_transition_elapsed(self.num_timesteps)
            self._apply_weapon_phase(elapsed)
            self._last_weapon_update = self.num_timesteps

        # 定期ログ（wandb_log_freq ステップごと）
        if self._last_wandb_log < 0 or self.num_timesteps - self._last_wandb_log >= self._wandb_log_freq:
            self._log_wandb_metrics()

        return True

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _log_wandb_metrics(self) -> None:
        """武器カリキュラムメトリクスを W&B に記録する。"""
        if not self._wandb_logger or not self._wandb_logger.enabled:
            return
        metrics = self._module.get_wandb_metrics_with_step(self.num_timesteps)
        self._wandb_logger.log(metrics, step=self.num_timesteps)
        self._last_wandb_log = self.num_timesteps

    def _apply_weapon_phase(self, elapsed: int) -> None:
        phase_key = self._module.current_phase_key
        params = get_params_for_phase(phase_key, global_step=elapsed)
        try:
            results = self.training_env.env_method("set_params", **params)
            if any(r is False for r in (results or [])):
                n_fail = sum(1 for r in results if r is False)
                print(
                    f"[WeaponPhaseAuto] WARN: {n_fail} env で set_params が拒否されました "
                    f"(phase={phase_key})"
                )
        except Exception as exc:
            print(f"[WeaponPhaseAuto] WARN: set_params 送信失敗 (phase={phase_key}): {exc}")

        if self._module.is_transition:
            weights = params.get("weapon_weights", {})
            print(
                f"[WeaponPhaseAuto] phase={phase_key}, elapsed={elapsed}, "
                f"weapon_weights={weights}"
            )
