"""tests/survivors/test_weapon_unlock_module.py"""
from __future__ import annotations
import sys
from pathlib import Path

_TRAINING_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAINING_ROOT))

import pytest
from games.survivors.modules.weapon_unlock_module import WeaponUnlockStateModule, WeaponUnlockAdvanceEvent
from games.survivors.survivors_weapon_curriculum import WeaponType


class MockStats:
    def __init__(self, episode_count, active_score_p10, terminated_rate):
        self.episode_count = episode_count
        self.active_score_p10 = active_score_p10
        self.terminated_rate = terminated_rate


class MockStatsProvider:
    def __init__(self, stats=None):
        self._stats = stats

    def get_cell_stats(self, first_weapon_id, enemy_phase_idx):
        return self._stats


def test_no_advance_if_stats_none():
    m = WeaponUnlockStateModule(weapon_unlock_min_steps=0)
    provider = MockStatsProvider(None)
    result = m.maybe_advance(provider, num_timesteps=100_000, max_unlocked_enemy_phase_idx=2)
    assert result is None


def test_no_advance_if_conditions_not_met():
    m = WeaponUnlockStateModule(weapon_unlock_min_steps=0)
    provider = MockStatsProvider(MockStats(episode_count=5, active_score_p10=100.0, terminated_rate=0.8))
    result = m.maybe_advance(provider, num_timesteps=100_000, max_unlocked_enemy_phase_idx=2)
    assert result is None


def test_advance_when_conditions_met():
    m = WeaponUnlockStateModule(
        weapon_unlock_min_episodes=30,
        weapon_unlock_target_p10=300.0,
        weapon_unlock_max_terminated_rate=0.5,
        weapon_unlock_min_steps=0,
    )
    provider = MockStatsProvider(MockStats(episode_count=35, active_score_p10=350.0, terminated_rate=0.3))
    event = m.maybe_advance(provider, num_timesteps=0, max_unlocked_enemy_phase_idx=2)
    assert event is not None
    assert isinstance(event, WeaponUnlockAdvanceEvent)
    assert event.from_stage_key == "WU0"
    assert event.to_stage_key == "WU1"
    assert event.new_weapon_id == WeaponType.KING_BIBLE


def test_stage_order_increments():
    m = WeaponUnlockStateModule(weapon_unlock_min_steps=0)
    assert m.current_stage_order == 0
    provider = MockStatsProvider(MockStats(episode_count=35, active_score_p10=350.0, terminated_rate=0.3))
    m.maybe_advance(provider, num_timesteps=0, max_unlocked_enemy_phase_idx=2)
    assert m.current_stage_order == 1
    assert m.current_stage_key == "WU1"


def test_export_import():
    m = WeaponUnlockStateModule(weapon_unlock_min_steps=0)
    provider = MockStatsProvider(MockStats(episode_count=35, active_score_p10=350.0, terminated_rate=0.3))
    m.maybe_advance(provider, num_timesteps=12345, max_unlocked_enemy_phase_idx=2)
    state = m.export_state()
    m2 = WeaponUnlockStateModule(weapon_unlock_min_steps=0)
    m2.import_state(state)
    assert m2.current_stage_key == m.current_stage_key
    assert m2.current_stage_order == m.current_stage_order


def test_set_start_step_skipped_after_resume():
    """resume 時は set_start_step() が start_step を上書きしないことを確認する。"""
    m = WeaponUnlockStateModule(weapon_unlock_min_steps=100_000)
    m.set_start_step(0)
    assert m._start_step == 0

    state = m.export_state()
    m2 = WeaponUnlockStateModule(weapon_unlock_min_steps=100_000)
    m2.import_state(state)
    # resume 後は set_start_step を呼んでも上書きされない
    m2.set_start_step(500_000)
    assert m2._start_step == 0  # 復元値を保持


def test_target_enemy_phase_parameter():
    """target_enemy_phase を明示指定したとき、その phase の stats が参照されることを確認する。"""
    seen_phases: list[int] = []

    class TrackingProvider:
        def get_cell_stats(self, first_weapon_id, enemy_phase_idx):
            seen_phases.append(enemy_phase_idx)
            return MockStats(episode_count=35, active_score_p10=350.0, terminated_rate=0.3)

    m = WeaponUnlockStateModule(
        weapon_unlock_min_steps=0,
        weapon_unlock_readiness_enemy_phase_cap=2,
    )
    m.maybe_advance(
        TrackingProvider(),
        num_timesteps=0,
        max_unlocked_enemy_phase_idx=4,
        target_enemy_phase=3,  # backtrack 考慮済みの値を呼び出し側が指定
    )
    assert seen_phases == [3]  # cap=2 ではなく指定した 3 が使われること
