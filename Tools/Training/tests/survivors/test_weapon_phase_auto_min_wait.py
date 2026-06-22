"""WeaponPhaseAutoStateModule の待機ロジック回帰テスト。

テスト対象のシナリオ:
1. ゲート条件成立後の待機: gate 条件成立後すぐには昇格せず、min_wait_steps 経過後に昇格する
2. 待機後昇格: min_wait_steps ちょうど経過したステップで昇格する
3. rollback リセット: ゲート成立→curriculum rollback でゲート未満になったら
   _gate_first_met_step がリセットされ、再成立から改めてカウントする
4. export/import 後の再開: export_state() → import_state() で _gate_first_met_step が
   正しく復元され、待機が中断されず継続する
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

_TRAINING_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAINING_ROOT))

import pytest

from games.survivors.state_modules import (
    WeaponPhaseAutoStateModule,
    CurriculumStateModule,
    WEAPON_PHASE_CURRICULUM_GATES,
    WEAPON_PHASE_AUTO_SEQUENCE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_curriculum_mock(phase: int) -> MagicMock:
    """指定したフェーズを返す CurriculumStateModule モック。"""
    m = MagicMock(spec=CurriculumStateModule)
    m.current_phase = phase
    return m


def _make_module(
    curriculum_phase: int = 0,
    min_wait_steps: int = 100_000,
    advance_fn=None,
) -> WeaponPhaseAutoStateModule:
    """WeaponPhaseAutoStateModule のインスタンスを生成するヘルパー。"""
    curriculum = _make_curriculum_mock(curriculum_phase)
    return WeaponPhaseAutoStateModule(
        curriculum=curriculum,
        min_wait_steps=min_wait_steps,
        on_weapon_phase_advance_fn=advance_fn,
    )


def _get_gate_for_seq_idx(seq_idx: int) -> int:
    """シーケンスインデックスに対応するゲート値を返す。"""
    phase_key = WEAPON_PHASE_AUTO_SEQUENCE[seq_idx]
    gate = WEAPON_PHASE_CURRICULUM_GATES[phase_key]
    assert gate is not None, f"{phase_key} は最終フェーズ"
    return gate


# ---------------------------------------------------------------------------
# テスト 1: ゲート条件成立後の待機
# ---------------------------------------------------------------------------

class TestGateWaitBeforeAdvance:
    """ゲート条件が成立してもすぐには昇格しないこと。"""

    def test_no_advance_immediately_after_gate(self):
        """ゲート成立直後のステップでは昇格しないこと。"""
        min_wait = 100_000
        mod = _make_module(min_wait_steps=min_wait)

        gate = _get_gate_for_seq_idx(0)  # W0 のゲート値
        # ゲート成立: curriculum が gate 以上を返す
        mod._curriculum.current_phase = gate

        # ゲート成立直後（待機 0 step）
        result = mod.on_step(num_timesteps=1_000_000, current_curriculum_phase=gate)
        assert result is None, "ゲート成立直後（待機 0 step）は昇格しないこと"

        # _gate_first_met_step が記録されていること
        assert mod._gate_first_met_step == 1_000_000

    def test_no_advance_before_min_wait_expires(self):
        """min_wait_steps - 1 step では昇格しないこと。"""
        min_wait = 50_000
        mod = _make_module(min_wait_steps=min_wait)

        gate = _get_gate_for_seq_idx(0)
        gate_met_step = 500_000

        # ゲート条件成立を記録
        mod.on_step(num_timesteps=gate_met_step, current_curriculum_phase=gate)
        assert mod._gate_first_met_step == gate_met_step

        # min_wait_steps - 1 step経過したステップ（昇格しないこと）
        result = mod.on_step(
            num_timesteps=gate_met_step + min_wait - 1,
            current_curriculum_phase=gate,
        )
        assert result is None, f"待機 {min_wait - 1} step では昇格しないこと"


# ---------------------------------------------------------------------------
# テスト 2: 待機後昇格
# ---------------------------------------------------------------------------

class TestAdvanceAfterMinWait:
    """min_wait_steps ちょうど経過したステップで昇格すること。"""

    def test_advance_exactly_at_min_wait(self):
        """min_wait_steps ちょうどのステップで昇格すること。"""
        min_wait = 50_000
        advance_called = []
        mod = _make_module(
            min_wait_steps=min_wait,
            advance_fn=lambda: advance_called.append(True),
        )

        gate = _get_gate_for_seq_idx(0)
        gate_met_step = 200_000
        initial_seq_idx = mod._weapon_phase_seq_idx

        # ゲート条件成立
        mod.on_step(num_timesteps=gate_met_step, current_curriculum_phase=gate)
        assert mod._gate_first_met_step == gate_met_step

        # ちょうど min_wait_steps 経過
        result = mod.on_step(
            num_timesteps=gate_met_step + min_wait,
            current_curriculum_phase=gate,
        )
        assert result is not None, "min_wait_steps 経過後は昇格すること"
        assert mod._weapon_phase_seq_idx == initial_seq_idx + 1
        assert len(advance_called) == 1, "on_weapon_phase_advance_fn が呼ばれること"

    def test_advance_beyond_min_wait(self):
        """min_wait_steps 超過後のステップでも昇格すること。"""
        min_wait = 30_000
        mod = _make_module(min_wait_steps=min_wait)

        gate = _get_gate_for_seq_idx(0)
        gate_met_step = 100_000
        initial_seq_idx = mod._weapon_phase_seq_idx

        # ゲート条件成立
        mod.on_step(num_timesteps=gate_met_step, current_curriculum_phase=gate)

        # min_wait を大幅に超過
        result = mod.on_step(
            num_timesteps=gate_met_step + min_wait + 99_999,
            current_curriculum_phase=gate,
        )
        assert result is not None
        assert mod._weapon_phase_seq_idx == initial_seq_idx + 1

    def test_gate_first_met_step_reset_after_advance(self):
        """昇格後に _gate_first_met_step が None にリセットされること。"""
        min_wait = 10_000
        mod = _make_module(min_wait_steps=min_wait)

        gate = _get_gate_for_seq_idx(0)
        gate_met_step = 50_000

        mod.on_step(num_timesteps=gate_met_step, current_curriculum_phase=gate)
        mod.on_step(
            num_timesteps=gate_met_step + min_wait,
            current_curriculum_phase=gate,
        )
        # 昇格後は _gate_first_met_step がリセットされていること
        assert mod._gate_first_met_step is None


# ---------------------------------------------------------------------------
# テスト 3: rollback リセット
# ---------------------------------------------------------------------------

class TestRollbackResetsGateTimer:
    """curriculum rollback でゲート未満になったら _gate_first_met_step がリセットされ、
    再成立から改めてカウントすること。"""

    def test_gate_first_met_step_reset_on_rollback(self):
        """curriculum が gate 未満に戻ったとき _gate_first_met_step が None になること。"""
        min_wait = 100_000
        mod = _make_module(min_wait_steps=min_wait)

        gate = _get_gate_for_seq_idx(0)
        gate_met_step = 300_000

        # ゲート条件成立
        mod.on_step(num_timesteps=gate_met_step, current_curriculum_phase=gate)
        assert mod._gate_first_met_step == gate_met_step

        # curriculum rollback: ゲート未満に戻る
        result = mod.on_step(
            num_timesteps=gate_met_step + 10_000,
            current_curriculum_phase=gate - 1,
        )
        assert result is None
        assert mod._gate_first_met_step is None, "rollback 後は _gate_first_met_step がリセットされること"

    def test_recount_from_new_gate_after_rollback(self):
        """rollback 後に再度ゲート成立したとき、そこから改めてカウントすること。"""
        min_wait = 50_000
        advance_called = []
        mod = _make_module(
            min_wait_steps=min_wait,
            advance_fn=lambda: advance_called.append(True),
        )

        gate = _get_gate_for_seq_idx(0)
        gate_met_step_1 = 200_000
        rollback_step = gate_met_step_1 + 20_000
        gate_met_step_2 = rollback_step + 5_000

        # 1回目ゲート成立（min_wait の途中まで待機）
        mod.on_step(num_timesteps=gate_met_step_1, current_curriculum_phase=gate)
        assert mod._gate_first_met_step == gate_met_step_1

        # rollback: ゲート未満に戻る
        mod.on_step(num_timesteps=rollback_step, current_curriculum_phase=gate - 1)
        assert mod._gate_first_met_step is None

        # 2回目ゲート成立
        mod.on_step(num_timesteps=gate_met_step_2, current_curriculum_phase=gate)
        assert mod._gate_first_met_step == gate_met_step_2

        # 2回目ゲート成立から min_wait - 1 step: まだ昇格しないこと
        no_advance_step = gate_met_step_2 + min_wait - 1
        result = mod.on_step(num_timesteps=no_advance_step, current_curriculum_phase=gate)
        assert result is None
        assert len(advance_called) == 0

        # 2回目ゲート成立から min_wait step: 昇格すること
        advance_step = gate_met_step_2 + min_wait
        result = mod.on_step(num_timesteps=advance_step, current_curriculum_phase=gate)
        assert result is not None
        assert len(advance_called) == 1

    def test_old_wait_time_not_carried_over_after_rollback(self):
        """rollback 前の待機時間が次のゲート成立時に引き継がれないこと。

        1回目に min_wait - 1 step 待機 → rollback → 再成立直後に昇格しないことを確認する。
        """
        min_wait = 80_000
        mod = _make_module(min_wait_steps=min_wait)

        gate = _get_gate_for_seq_idx(0)
        gate_met_step_1 = 100_000
        # min_wait の直前まで待機
        just_before_advance = gate_met_step_1 + min_wait - 1

        mod.on_step(num_timesteps=gate_met_step_1, current_curriculum_phase=gate)
        mod.on_step(num_timesteps=just_before_advance, current_curriculum_phase=gate)

        # rollback
        mod.on_step(num_timesteps=just_before_advance + 1, current_curriculum_phase=gate - 1)
        assert mod._gate_first_met_step is None

        # 再成立: 1 step 後 → 以前の待機時間は引き継がれないので昇格しないこと
        gate_met_step_2 = just_before_advance + 2
        result = mod.on_step(num_timesteps=gate_met_step_2, current_curriculum_phase=gate)
        assert result is None, "rollback 後の再成立直後（1 step）は昇格しないこと"
        assert mod._gate_first_met_step == gate_met_step_2


# ---------------------------------------------------------------------------
# テスト 4: export/import 後の再開
# ---------------------------------------------------------------------------

class TestExportImportResumesWait:
    """export_state() → import_state() で _gate_first_met_step が正しく復元され、
    待機が中断されず継続すること。"""

    def test_export_includes_gate_first_met_step(self):
        """export_state に _gate_first_met_step が含まれること。"""
        mod = _make_module()
        gate = _get_gate_for_seq_idx(0)
        mod.on_step(num_timesteps=500_000, current_curriculum_phase=gate)
        assert mod._gate_first_met_step == 500_000

        state = mod.export_state()
        assert "gate_first_met_step" in state
        assert state["gate_first_met_step"] == 500_000

    def test_import_restores_gate_first_met_step(self):
        """import_state で _gate_first_met_step が復元されること。"""
        min_wait = 60_000
        mod = _make_module(min_wait_steps=min_wait)
        gate = _get_gate_for_seq_idx(0)
        gate_met_step = 400_000

        # ゲート成立（待機開始）
        mod.on_step(num_timesteps=gate_met_step, current_curriculum_phase=gate)
        assert mod._gate_first_met_step == gate_met_step

        # export してから新しいモジュールに import
        state = mod.export_state()
        curriculum_mock = _make_curriculum_mock(gate)
        mod2 = WeaponPhaseAutoStateModule(
            curriculum=curriculum_mock,
            min_wait_steps=min_wait,
        )
        mod2.import_state(state)

        assert mod2._gate_first_met_step == gate_met_step, (
            "import_state で _gate_first_met_step が復元されること"
        )

    def test_wait_continues_after_import(self):
        """import 後の on_step で待機が継続すること（min_wait_steps - 1 では昇格しない）。"""
        min_wait = 60_000
        mod = _make_module(min_wait_steps=min_wait)
        gate = _get_gate_for_seq_idx(0)
        gate_met_step = 400_000

        # ゲート成立
        mod.on_step(num_timesteps=gate_met_step, current_curriculum_phase=gate)
        state = mod.export_state()

        curriculum_mock = _make_curriculum_mock(gate)
        mod2 = WeaponPhaseAutoStateModule(
            curriculum=curriculum_mock,
            min_wait_steps=min_wait,
        )
        mod2.import_state(state)

        # import 後、min_wait - 1 step のタイムスタンプ: 昇格しないこと
        result = mod2.on_step(
            num_timesteps=gate_met_step + min_wait - 1,
            current_curriculum_phase=gate,
        )
        assert result is None, "import 後 min_wait - 1 step では昇格しないこと"

    def test_advance_happens_at_correct_step_after_import(self):
        """import 後の on_step で min_wait_steps ちょうどのステップに昇格すること。"""
        min_wait = 60_000
        advance_called = []
        gate = _get_gate_for_seq_idx(0)
        gate_met_step = 400_000

        # 元のモジュールでゲート成立
        mod = _make_module(min_wait_steps=min_wait)
        mod.on_step(num_timesteps=gate_met_step, current_curriculum_phase=gate)
        state = mod.export_state()

        # import して別のモジュールに復元
        curriculum_mock = _make_curriculum_mock(gate)
        mod2 = WeaponPhaseAutoStateModule(
            curriculum=curriculum_mock,
            min_wait_steps=min_wait,
            on_weapon_phase_advance_fn=lambda: advance_called.append(True),
        )
        mod2.import_state(state)
        initial_seq_idx = mod2._weapon_phase_seq_idx

        # ちょうど min_wait_steps 経過したステップ
        result = mod2.on_step(
            num_timesteps=gate_met_step + min_wait,
            current_curriculum_phase=gate,
        )
        assert result is not None, "import 後 min_wait_steps 経過で昇格すること"
        assert mod2._weapon_phase_seq_idx == initial_seq_idx + 1
        assert len(advance_called) == 1

    def test_export_import_with_no_gate_met(self):
        """ゲート未成立のまま export/import しても問題ないこと。"""
        min_wait = 50_000
        mod = _make_module(min_wait_steps=min_wait)
        # ゲート成立なし
        assert mod._gate_first_met_step is None

        state = mod.export_state()
        assert state["gate_first_met_step"] is None

        curriculum_mock = _make_curriculum_mock(0)
        mod2 = WeaponPhaseAutoStateModule(
            curriculum=curriculum_mock,
            min_wait_steps=min_wait,
        )
        mod2.import_state(state)
        assert mod2._gate_first_met_step is None
