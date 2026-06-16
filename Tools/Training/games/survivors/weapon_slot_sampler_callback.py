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
        self._apply_all_envs()

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        for i, done in enumerate(dones):
            if done:
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
        )
