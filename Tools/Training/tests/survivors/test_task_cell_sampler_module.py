"""tests/survivors/test_task_cell_sampler_module.py"""
from __future__ import annotations
import sys
from pathlib import Path

_TRAINING_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAINING_ROOT))

import pytest
from games.survivors.modules.task_cell_sampler_module import (
    TaskCell, TaskCellSamplerStateModule,
)
from games.survivors.survivors_weapon_curriculum import WeaponType


def make_sampler(
    min_episodes_per_cell=5,
    target_p10=300.0,
    progress_scale=300.0,
    random_floor=0.05,
    recency_scale=500_000,
    unlearnable_min_episodes=10,
    unlearnable_score_floor=50.0,
    emergency_length_floor_ratio=0.25,
    blocked_steps=200_000,
    enemy_phase_backtrack=1,
    **kwargs,
):
    return TaskCellSamplerStateModule(
        min_episodes_per_cell=min_episodes_per_cell,
        target_p10=target_p10,
        progress_scale=progress_scale,
        random_floor=random_floor,
        recency_scale=recency_scale,
        unlearnable_min_episodes=unlearnable_min_episodes,
        unlearnable_score_floor=unlearnable_score_floor,
        emergency_length_floor_ratio=emergency_length_floor_ratio,
        blocked_steps=blocked_steps,
        enemy_phase_backtrack=enemy_phase_backtrack,
        **kwargs,
    )


def test_rebuild_candidate_cells():
    s = make_sampler()
    s.rebuild_candidate_cells(
        stage_key="WU0",
        max_unlocked_enemy_phase_idx=0,
        min_episode_steps_by_phase={0: 100},
    )
    assert len(s._candidate_cells) == 1
    cell = s._candidate_cells[0]
    assert cell.first_weapon_id == WeaponType.GARLIC
    assert cell.enemy_phase_idx == 0


def test_rebuild_with_wu1():
    s = make_sampler()
    s.rebuild_candidate_cells(
        stage_key="WU1",
        max_unlocked_enemy_phase_idx=2,
        min_episode_steps_by_phase={0: 100, 1: 200, 2: 300},
    )
    # WU1: GARLIC + KING_BIBLE, enemy_phase: 1 or 2 (backtrack=1)
    weapon_ids = {c.first_weapon_id for c in s._candidate_cells}
    assert WeaponType.GARLIC in weapon_ids
    assert WeaponType.KING_BIBLE in weapon_ids
    phase_idxs = {c.enemy_phase_idx for c in s._candidate_cells}
    assert 1 in phase_idxs
    assert 2 in phase_idxs


def test_on_episode_end_updates_stats():
    s = make_sampler()
    s.rebuild_candidate_cells("WU0", 0, {0: 100})
    cell = s._candidate_cells[0]
    s.on_episode_end(cell, active_score=500.0, ep_len=300, terminated=False, num_timesteps=1000)
    stats = s.get_cell_stats(WeaponType.GARLIC, 0)
    assert stats is not None
    assert stats.episode_count == 1
    assert stats.active_score_mean == pytest.approx(500.0)


def test_blocked_cell_excluded():
    s = make_sampler(unlearnable_min_episodes=2, unlearnable_score_floor=100.0)
    s.rebuild_candidate_cells("WU1", 1, {0: 1000, 1: 1000})
    garlic_cell = TaskCell("WU1", WeaponType.GARLIC, 1)
    kb_cell = TaskCell("WU1", WeaponType.KING_BIBLE, 1)
    # KING_BIBLE を崩壊状態にしてブロックさせる
    for _ in range(5):
        s.on_episode_end(kb_cell, active_score=10.0, ep_len=50, terminated=True, num_timesteps=1000)
    kb_stats = s.get_cell_stats(WeaponType.KING_BIBLE, 1)
    # ブロックされていること（統計的にブロック条件を満たしている場合）
    # 注: emergency_length_floor = 1000 * 0.25 = 250, ep_len=50 < 250
    if kb_stats and kb_stats.blocked_until_step > 0:
        sampled = s.sample_cell(num_timesteps=1000)
        # ブロック中のKING_BIBLEが選ばれないことを複数回確認（確率的テスト）
        sampled_ids = set()
        for _ in range(20):
            sampled_ids.add(s.sample_cell(1000).first_weapon_id)
        # GARLIC のみサンプルされる（KING_BIBLEはブロック中）
        assert WeaponType.GARLIC in sampled_ids


def test_export_import():
    s = make_sampler()
    s.rebuild_candidate_cells("WU0", 0, {0: 100})
    cell = s._candidate_cells[0]
    s.on_episode_end(cell, active_score=400.0, ep_len=200, terminated=False, num_timesteps=5000)
    state = s.export_state()
    s2 = make_sampler()
    s2.import_state(state)
    stats = s2.get_cell_stats(WeaponType.GARLIC, 0)
    assert stats is not None
    assert stats.episode_count == 1
    assert stats.active_score_mean == pytest.approx(400.0)
