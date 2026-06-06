"""
WeaponPhaseAutoCallback: カリキュラム停滞を検出して weapon_phase を自動昇格するコールバック。

--weapon-phase auto 指定時のみ train.py から登録される。
HybridCurriculumSpalfCallback の current_phase を監視し、
stagnation_steps ステップ以上変化しなかった場合に次の武器フェーズへ進める。

武器フェーズ昇格時はカリキュラムを1フェーズ強制降格（hybrid_cb.rollback_one_phase() 経由）。

resume 時: weapon_phase_seq_idx を復元し、stagnation_start_step はリセットする。
"""
from __future__ import annotations

import json
import os

from stable_baselines3.common.callbacks import BaseCallback

from games.survivors.survivors_weapon_curriculum import WEAPON_PHASES, get_params_for_phase

PHASE_SEQUENCE: list[str] = [
    "W0", "W0_to_W1", "W1", "W1_to_W2", "W2", "W3", "W4", "W5", "W6",
]

_STATUS_FILENAME = "weapon_phase_auto_status.json"


class WeaponPhaseAutoCallback(BaseCallback):
    """武器フェーズ自動昇格コールバック。

    カリキュラムの current_phase が stagnation_steps ステップ変化しない場合に
    PHASE_SEQUENCE を1段階進める。遷移フェーズ（W0_to_W1 等）では
    weapon_update_freq ステップごとに weapon_weights を線形補間で更新する。
    """

    def __init__(
        self,
        hybrid_cb,
        stagnation_steps: int = 500_000,
        output_dir: str = "",
        checkpoint_dir: str = "",
        weapon_update_freq: int = 10_000,
    ):
        super().__init__(verbose=0)
        self._hybrid_cb = hybrid_cb
        self._stagnation_steps = max(stagnation_steps, 1)
        self._output_dir = output_dir
        self._checkpoint_dir = checkpoint_dir
        self._weapon_update_freq = max(weapon_update_freq, 1)

        self._weapon_phase_seq_idx: int = 0
        self._last_curriculum_phase: int = 0
        self._stagnation_start_step: int = 0
        self._phase_start_step: int = 0   # 遷移フェーズ内の経過ステップ計算用
        self._last_weapon_update: int = 0
        self._is_transition: bool = False

        self._status_file = os.path.join(output_dir, _STATUS_FILENAME) if output_dir else ""

    # ------------------------------------------------------------------ #
    # SB3 callbacks                                                        #
    # ------------------------------------------------------------------ #

    def _on_training_start(self) -> None:
        # (a) 同一run resume: output_dir の status から weapon_phase_seq_idx と phase_start_step を復元
        if self._status_file and os.path.exists(self._status_file):
            try:
                with open(self._status_file) as f:
                    status = json.load(f)
                self._weapon_phase_seq_idx = int(status.get("weapon_phase_seq_idx", 0))
                self._phase_start_step = int(status.get("phase_start_step", self.num_timesteps))
                print(
                    f"[WeaponPhaseAuto] resume 復元 (output_dir): "
                    f"weapon_phase={self._current_phase_key()}, "
                    f"phase_start_step={self._phase_start_step}"
                )
            except (json.JSONDecodeError, KeyError, OSError, ValueError):
                self._phase_start_step = self.num_timesteps
        # (b) branch resume: checkpoint_dir から復元を試みる
        elif self._checkpoint_dir:
            ckpt_status = os.path.join(self._checkpoint_dir, _STATUS_FILENAME)
            if os.path.exists(ckpt_status):
                try:
                    with open(ckpt_status) as f:
                        status = json.load(f)
                    self._weapon_phase_seq_idx = int(status.get("weapon_phase_seq_idx", 0))
                    self._phase_start_step = int(status.get("phase_start_step", self.num_timesteps))
                    print(
                        f"[WeaponPhaseAuto] branch resume 復元 (checkpoint_dir): "
                        f"weapon_phase={self._current_phase_key()}, "
                        f"phase_start_step={self._phase_start_step}"
                    )
                except (json.JSONDecodeError, KeyError, OSError, ValueError):
                    self._phase_start_step = self.num_timesteps
        else:
            self._phase_start_step = self.num_timesteps

        # stagnation タイマーは常にリセット（resume 時も含む）
        self._last_curriculum_phase = self._hybrid_cb.current_phase
        self._stagnation_start_step = self.num_timesteps
        self._last_weapon_update = self.num_timesteps

        phase_key = self._current_phase_key()
        phase_def = WEAPON_PHASES.get(phase_key, {})
        self._is_transition = phase_def.get("weapon_pool_mode") == "weighted"

        # 初回 set_params 送信（遷移フェーズはフェーズ開始からの elapsed で補間位置を復元）
        elapsed = self.num_timesteps - self._phase_start_step
        self._apply_weapon_phase(elapsed=elapsed)
        self._save_status()
        print(
            f"[WeaponPhaseAuto] 有効 phase={phase_key}, "
            f"stagnation_steps={self._stagnation_steps:,}, "
            f"weapon_update_freq={self._weapon_update_freq:,}"
        )

    def _on_step(self) -> bool:
        curr = self._hybrid_cb.current_phase

        if curr != self._last_curriculum_phase:
            # カリキュラムフェーズが変化 → 停滞タイマーをリセット
            self._last_curriculum_phase = curr
            self._stagnation_start_step = self.num_timesteps
        elif (
            self.num_timesteps - self._stagnation_start_step >= self._stagnation_steps
            and self._weapon_phase_seq_idx < len(PHASE_SEQUENCE) - 1
        ):
            # 停滞検出 → 武器フェーズ昇格
            self._advance_weapon_phase()

        # 遷移フェーズ中: weapon_weights を定期更新
        if self._is_transition and self.num_timesteps - self._last_weapon_update >= self._weapon_update_freq:
            elapsed = self.num_timesteps - self._phase_start_step
            self._apply_weapon_phase(elapsed=elapsed)
            self._last_weapon_update = self.num_timesteps

        return True

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _current_phase_key(self) -> str:
        idx = min(self._weapon_phase_seq_idx, len(PHASE_SEQUENCE) - 1)
        return PHASE_SEQUENCE[idx]

    def _advance_weapon_phase(self) -> None:
        old_key = self._current_phase_key()
        self._weapon_phase_seq_idx += 1
        new_key = self._current_phase_key()

        print(
            f"[WeaponPhaseAuto] 武器フェーズ昇格: {old_key} -> {new_key} "
            f"(停滞 {self._stagnation_steps:,} step 検出)"
        )

        # 遷移フェーズかどうか更新
        phase_def = WEAPON_PHASES.get(new_key, {})
        self._is_transition = phase_def.get("weapon_pool_mode") == "weighted"
        self._phase_start_step = self.num_timesteps
        self._last_weapon_update = self.num_timesteps

        # 新しい武器フェーズのパラメータを送信（遷移フェーズは elapsed=0 から開始）
        self._apply_weapon_phase(elapsed=0)

        # カリキュラムを1フェーズ降格
        self._hybrid_cb.rollback_one_phase(f"weapon_phase_advanced_to_{new_key}")
        # 降格後のフェーズで停滞タイマーをリセット
        self._last_curriculum_phase = self._hybrid_cb.current_phase
        self._stagnation_start_step = self.num_timesteps

        self._save_status()

    def _apply_weapon_phase(self, elapsed: int) -> None:
        phase_key = self._current_phase_key()
        params = get_params_for_phase(phase_key, global_step=elapsed)
        try:
            results = self.training_env.env_method("set_params", **params)
            if any(r is False for r in (results or [])):
                n_fail = sum(1 for r in results if r is False)
                print(f"[WeaponPhaseAuto] WARN: {n_fail} env で set_params が拒否されました (phase={phase_key})")
        except Exception as exc:
            print(f"[WeaponPhaseAuto] WARN: set_params 送信失敗 (phase={phase_key}): {exc}")

        if self._is_transition:
            weights = params.get("weapon_weights", {})
            print(f"[WeaponPhaseAuto] phase={phase_key}, elapsed={elapsed}, weapon_weights={weights}")

    def _save_status(self) -> None:
        if not self._status_file:
            return
        os.makedirs(self._output_dir, exist_ok=True)
        with open(self._status_file, "w") as f:
            json.dump({
                "weapon_phase_seq_idx": self._weapon_phase_seq_idx,
                "phase_start_step": self._phase_start_step,
            }, f)
