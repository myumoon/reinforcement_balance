"""
WeaponSlotSamplerCallback: エピソード終了時に次エピソード用の
制約付きランダム武器スロットを各 env に設定する SB3 コールバック。
既存の WeaponCurriculumCallback / HybridCurriculumSpalfCallback とは独立。
--rsi-mode auto で有効化される。
"""
from __future__ import annotations

import random
from typing import Callable, Optional

from stable_baselines3.common.callbacks import BaseCallback

from games.survivors.weapon_slot_sampler import (
    get_initial_elapsed_time,
    sample_weapon_slots,
)


class WeaponSlotSamplerCallback(BaseCallback):
    def __init__(
        self,
        get_phase_name_fn: Callable[[], str],
        seed: Optional[int] = None,
    ):
        super().__init__(verbose=0)
        self._get_phase_name = get_phase_name_fn
        self._rng = random.Random(seed)

    def _on_training_start(self) -> None:
        self._apply_all_envs()  # 初期送信（ep1 の reset 前にセット）

    def _on_rollout_start(self) -> None:
        # ロールアウト開始時に全 env へ RSI パラメータを再送する。
        # _on_step() での done 検出サンプルが次エピソードのリセット前に
        # 届かない場合のフォールバックとして機能する。
        self._apply_all_envs()

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        for i, done in enumerate(dones):
            if done:
                # done=True 検出時は SB3 がすでに次エピソードの reset() を完了しているため、
                # ここで送信したサンプルは「次の次の」エピソード開始時に適用される。
                # 1 エピソードのズレは _on_training_start / _on_rollout_start でカバーする。
                self._apply_single_env(i)
        return True

    def _apply_all_envs(self) -> None:
        for i in range(self.training_env.num_envs):
            self._apply_single_env(i)

    def _apply_single_env(self, env_idx: int) -> None:
        phase_name   = self._get_phase_name()
        elapsed_time = get_initial_elapsed_time(phase_name)
        slots        = sample_weapon_slots(elapsed_time, rng=self._rng)
        if slots is None:
            self.training_env.env_method(
                "set_params", indices=[env_idx], clear_initial_override=True,
            )
            return
        self.training_env.env_method(
            "set_params", indices=[env_idx],
            initial_elapsed_time=elapsed_time,
            initial_weapon_slots=slots,
            MaxEpisodeTime=elapsed_time + 300.0,  # RSI 後に 300 秒のエピソードを保証
        )
