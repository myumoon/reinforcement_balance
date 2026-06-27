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
    def __init__(
        self,
        episode_count,
        active_score_p10,
        terminated_rate,
        episode_length_p10: float = 1200.0,
        short_episode_rate: float = 0.0,
        recent_terminated_rate: float = 0.0,
    ):
        self.episode_count = episode_count
        self.active_score_p10 = active_score_p10
        self.terminated_rate = terminated_rate
        self.episode_length_p10 = episode_length_p10
        self.short_episode_rate = short_episode_rate
        self.recent_terminated_rate = recent_terminated_rate


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


def test_no_block_if_max_terminated_rate_none():
    """weapon_unlock_max_terminated_rate=None の場合、terminated rate でブロックされない。"""
    m = WeaponUnlockStateModule(
        weapon_unlock_min_episodes=30,
        weapon_unlock_target_p10=300.0,
        weapon_unlock_max_terminated_rate=None,
        weapon_unlock_min_steps=0,
    )
    # recent_terminated_rate が高くても unlock される
    provider = MockStatsProvider(MockStats(
        episode_count=35,
        active_score_p10=350.0,
        terminated_rate=0.99,
        recent_terminated_rate=0.99,
    ))
    event = m.maybe_advance(provider, num_timesteps=0, max_unlocked_enemy_phase_idx=2)
    assert event is not None


def test_block_if_active_score_p10_not_met():
    """active_score_p10 未達ならunlockしない。"""
    m = WeaponUnlockStateModule(
        weapon_unlock_min_episodes=30,
        weapon_unlock_target_p10=300.0,
        weapon_unlock_max_terminated_rate=None,
        weapon_unlock_min_steps=0,
    )
    provider = MockStatsProvider(MockStats(
        episode_count=35,
        active_score_p10=250.0,  # target_p10=300.0 未達
        terminated_rate=0.0,
        episode_length_p10=1200.0,
        short_episode_rate=0.0,
        recent_terminated_rate=0.0,
    ))
    result = m.maybe_advance(provider, num_timesteps=0, max_unlocked_enemy_phase_idx=2)
    assert result is None


def test_block_if_ep_len_p10_not_met():
    """ep_len_p10 未達ならunlockしない。"""
    m = WeaponUnlockStateModule(
        weapon_unlock_min_episodes=30,
        weapon_unlock_target_p10=300.0,
        weapon_unlock_max_terminated_rate=None,
        weapon_unlock_max_short_episode_rate=None,
        weapon_unlock_min_steps=0,
        weapon_unlock_min_ep_len_p10_by_phase={0: 600, 1: 900, 2: 1200, 3: 1200},
    )
    # target_enemy_phase=2 なので min_ep_len_p10=1200 が必要
    provider = MockStatsProvider(MockStats(
        episode_count=35,
        active_score_p10=350.0,
        terminated_rate=0.0,
        episode_length_p10=800.0,  # 1200 未達
        short_episode_rate=0.0,
        recent_terminated_rate=0.0,
    ))
    result = m.maybe_advance(
        provider,
        num_timesteps=0,
        max_unlocked_enemy_phase_idx=2,
        target_enemy_phase=2,
    )
    assert result is None


def test_block_if_short_episode_rate_exceeded():
    """short_episode_rate 超過ならunlockしない。"""
    m = WeaponUnlockStateModule(
        weapon_unlock_min_episodes=30,
        weapon_unlock_target_p10=300.0,
        weapon_unlock_max_terminated_rate=None,
        weapon_unlock_max_short_episode_rate=0.15,
        weapon_unlock_min_steps=0,
    )
    provider = MockStatsProvider(MockStats(
        episode_count=35,
        active_score_p10=350.0,
        terminated_rate=0.0,
        episode_length_p10=1200.0,
        short_episode_rate=0.30,  # 0.15 超過
        recent_terminated_rate=0.0,
    ))
    result = m.maybe_advance(provider, num_timesteps=0, max_unlocked_enemy_phase_idx=2)
    assert result is None


def test_advance_when_all_conditions_met():
    """全条件が揃ったらunlockする（新判定ロジック）。"""
    m = WeaponUnlockStateModule(
        weapon_unlock_min_episodes=30,
        weapon_unlock_target_p10=300.0,
        weapon_unlock_max_terminated_rate=0.5,
        weapon_unlock_max_short_episode_rate=0.15,
        weapon_unlock_min_steps=0,
        weapon_unlock_min_ep_len_p10_by_phase={0: 600, 1: 900, 2: 1200, 3: 1200},
    )
    provider = MockStatsProvider(MockStats(
        episode_count=35,
        active_score_p10=350.0,
        terminated_rate=0.3,
        episode_length_p10=1300.0,
        short_episode_rate=0.10,
        recent_terminated_rate=0.20,
    ))
    event = m.maybe_advance(
        provider,
        num_timesteps=0,
        max_unlocked_enemy_phase_idx=2,
        target_enemy_phase=2,
    )
    assert event is not None
    assert isinstance(event, WeaponUnlockAdvanceEvent)
    assert event.from_stage_key == "WU0"
    assert event.to_stage_key == "WU1"


def test_ep_len_p10_fallback_for_high_phase():
    """target_enemy_phase が min_ep_len_p10_by_phase に無い場合、最大定義phaseの値へfallback する。"""
    # phase 0, 1, 2 のみ定義。target_enemy_phase=5 は定義外
    m = WeaponUnlockStateModule(
        weapon_unlock_min_episodes=30,
        weapon_unlock_target_p10=300.0,
        weapon_unlock_max_terminated_rate=None,
        weapon_unlock_max_short_episode_rate=None,
        weapon_unlock_min_steps=0,
        weapon_unlock_min_ep_len_p10_by_phase={0: 600, 1: 900, 2: 1200},
    )
    # fallback は max(defined_phases)=2 → 1200
    # ep_len_p10=1300 >= 1200 なのでunlock成功
    provider = MockStatsProvider(MockStats(
        episode_count=35,
        active_score_p10=350.0,
        terminated_rate=0.0,
        episode_length_p10=1300.0,
        short_episode_rate=0.0,
        recent_terminated_rate=0.0,
    ))
    event = m.maybe_advance(
        provider,
        num_timesteps=0,
        max_unlocked_enemy_phase_idx=5,
        target_enemy_phase=5,
    )
    assert event is not None

    # ep_len_p10=1100 < 1200 なのでunlock失敗
    m2 = WeaponUnlockStateModule(
        weapon_unlock_min_episodes=30,
        weapon_unlock_target_p10=300.0,
        weapon_unlock_max_terminated_rate=None,
        weapon_unlock_max_short_episode_rate=None,
        weapon_unlock_min_steps=0,
        weapon_unlock_min_ep_len_p10_by_phase={0: 600, 1: 900, 2: 1200},
    )
    provider2 = MockStatsProvider(MockStats(
        episode_count=35,
        active_score_p10=350.0,
        terminated_rate=0.0,
        episode_length_p10=1100.0,
        short_episode_rate=0.0,
        recent_terminated_rate=0.0,
    ))
    result2 = m2.maybe_advance(
        provider2,
        num_timesteps=0,
        max_unlocked_enemy_phase_idx=5,
        target_enemy_phase=5,
    )
    assert result2 is None


def test_few_episodes_no_unlock_even_if_ep_len_p10_computed():
    """40件未満のrecent windowでも episode_length_p10 は計算されるが、episode_count < min_episodes ではunlockしない。"""
    m = WeaponUnlockStateModule(
        weapon_unlock_min_episodes=30,
        weapon_unlock_target_p10=300.0,
        weapon_unlock_max_terminated_rate=None,
        weapon_unlock_max_short_episode_rate=None,
        weapon_unlock_min_steps=0,
        weapon_unlock_min_ep_len_p10_by_phase={0: 600},
    )
    # episode_count=10（< min_episodes=30）
    provider = MockStatsProvider(MockStats(
        episode_count=10,
        active_score_p10=350.0,
        terminated_rate=0.0,
        episode_length_p10=1200.0,
        short_episode_rate=0.0,
        recent_terminated_rate=0.0,
    ))
    result = m.maybe_advance(provider, num_timesteps=0, max_unlocked_enemy_phase_idx=0)
    assert result is None
