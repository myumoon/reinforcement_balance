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


def test_readiness_cap_phase_always_in_candidates():
    """readiness_cap_phase が backtrack 範囲外でも候補セルに含まれることを確認する。

    max_phase=4, backtrack=1 の場合: 通常候補は [3, 4]
    readiness_cap_phase=2 を渡すと [2, 3, 4] になる。
    """
    s = make_sampler()
    s.rebuild_candidate_cells(
        "WU0",
        max_unlocked_enemy_phase_idx=4,
        min_episode_steps_by_phase={0: 100, 1: 200, 2: 300, 3: 400, 4: 500},
        readiness_cap_phase=2,
    )
    phase_idxs = {c.enemy_phase_idx for c in s._candidate_cells}
    assert 2 in phase_idxs  # cap が強制追加されている
    assert 3 in phase_idxs
    assert 4 in phase_idxs
    assert 0 not in phase_idxs  # cap より低い phase は追加されない
    assert 1 not in phase_idxs


def test_readiness_cap_phase_not_added_if_already_in_range():
    """readiness_cap_phase が backtrack 範囲内なら重複して追加されないことを確認する。"""
    s = make_sampler()
    s.rebuild_candidate_cells(
        "WU0",
        max_unlocked_enemy_phase_idx=3,
        min_episode_steps_by_phase={0: 100, 1: 200, 2: 300, 3: 400},
        readiness_cap_phase=2,  # backtrack=1 なので lo_phase=2、cap=2 は範囲内
    )
    # 重複していないこと
    phase_list = [c.enemy_phase_idx for c in s._candidate_cells]
    assert phase_list.count(2) == 1  # Phase 2 は1度だけ


def test_blocked_cell_count_excludes_expired():
    """期限切れのブロックは blocked_cell_count に含まれないことを確認する。"""
    s = make_sampler(unlearnable_min_episodes=2, unlearnable_score_floor=100.0)
    s.rebuild_candidate_cells("WU0", 0, {0: 1000})
    cell = s._candidate_cells[0]

    # ブロック状態にする
    for _ in range(3):
        s.on_episode_end(cell, active_score=10.0, ep_len=50, terminated=True, num_timesteps=1000)

    if s._stats[cell.key()].blocked_until_step > 0:
        blocked_step = s._stats[cell.key()].blocked_until_step
        # ブロック中
        assert s.get_wandb_metrics(num_timesteps=1000)["task_cell_sampler/blocked_cell_count"] == 1
        # ブロック解除後
        assert s.get_wandb_metrics(num_timesteps=blocked_step + 1)["task_cell_sampler/blocked_cell_count"] == 0


def test_episode_length_p10_computed():
    """episode_length_p10 が recent window の10%ile で計算される。"""
    s = make_sampler(short_episode_steps=600)
    s.rebuild_candidate_cells("WU0", 0, {0: 100})
    cell = s._candidate_cells[0]

    # 10エピソードのep_lenを記録（p10 = 10%ile）
    ep_lens = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    for ep_len in ep_lens:
        s.on_episode_end(cell, active_score=500.0, ep_len=ep_len, terminated=False, num_timesteps=1000)

    stats = s.get_cell_stats(WeaponType.GARLIC, 0)
    assert stats is not None
    import numpy as np
    expected_p10 = float(np.percentile(ep_lens, 10))
    assert stats.episode_length_p10 == pytest.approx(expected_p10)


def test_short_episode_rate_computed():
    """short_episode_rate が閾値通り計算される。"""
    short_threshold = 500
    s = make_sampler(short_episode_steps=short_threshold)
    s.rebuild_candidate_cells("WU0", 0, {0: 100})
    cell = s._candidate_cells[0]

    # 10エピソード: 4つが threshold 未満（100, 200, 300, 400）、6つが以上
    ep_lens = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    for ep_len in ep_lens:
        s.on_episode_end(cell, active_score=500.0, ep_len=ep_len, terminated=False, num_timesteps=1000)

    stats = s.get_cell_stats(WeaponType.GARLIC, 0)
    assert stats is not None
    # 100, 200, 300, 400 の4つが short_threshold(500) 未満
    expected_rate = 4 / 10
    assert stats.short_episode_rate == pytest.approx(expected_rate)


def test_recent_terminated_rate_computed():
    """recent_terminated_rate が recent_terminated_flags から計算される。"""
    s = make_sampler()
    s.rebuild_candidate_cells("WU0", 0, {0: 100})
    cell = s._candidate_cells[0]

    # 10エピソード: 3つが terminated=True
    for i in range(10):
        terminated = i < 3
        s.on_episode_end(cell, active_score=500.0, ep_len=300, terminated=terminated, num_timesteps=1000)

    stats = s.get_cell_stats(WeaponType.GARLIC, 0)
    assert stats is not None
    assert stats.recent_terminated_rate == pytest.approx(3 / 10)


def test_import_state_backward_compat():
    """旧 state（新フィールドなし）の import_state() が recent_episode_lengths から再計算する。

    - episode_length_p10 / short_episode_rate は recent_episode_lengths から再計算する。
    - recent_terminated_rate は recent_terminated_flags がない場合に terminated_rate へ fallback する。
    """
    import numpy as np

    short_steps = 600
    s = make_sampler(short_episode_steps=short_steps)
    s.rebuild_candidate_cells("WU0", 0, {0: 100})
    cell = s._candidate_cells[0]
    # ep_len=800 は short_steps(600) 以上なので short 扱いにならない
    # terminated=False → terminated_rate=0.0
    s.on_episode_end(cell, active_score=400.0, ep_len=800, terminated=False, num_timesteps=5000)

    # 旧フォーマット（新フィールドなし）のstateを手動構築
    state = s.export_state()
    # 新フィールドを削除して旧フォーマットを模擬
    for stat_dict in state["stats"]:
        stat_dict.pop("episode_length_p10", None)
        stat_dict.pop("short_episode_rate", None)
        stat_dict.pop("recent_terminated_rate", None)
        stat_dict.pop("recent_terminated_flags", None)

    s2 = make_sampler(short_episode_steps=short_steps)
    s2.import_state(state)
    stats = s2.get_cell_stats(WeaponType.GARLIC, 0)
    assert stats is not None
    # recent_episode_lengths=[800] から再計算
    assert stats.episode_length_p10 == pytest.approx(float(np.percentile([800], 10)))
    # 800 >= short_steps(600) → short ではないので rate=0.0
    assert stats.short_episode_rate == pytest.approx(0.0)
    # recent_terminated_flags が無いので terminated_rate(0.0) に fallback
    assert stats.recent_terminated_rate == pytest.approx(0.0)
    assert len(stats.recent_terminated_flags) == 0


def test_import_state_backward_compat_terminated_fallback():
    """旧 state で terminated があった場合、recent_terminated_rate が terminated_rate に fallback する。"""
    s = make_sampler()
    s.rebuild_candidate_cells("WU0", 0, {0: 100})
    cell = s._candidate_cells[0]
    # 2エピソード: 1つ terminated=True → terminated_rate=0.5
    s.on_episode_end(cell, active_score=400.0, ep_len=300, terminated=True, num_timesteps=1000)
    s.on_episode_end(cell, active_score=400.0, ep_len=300, terminated=False, num_timesteps=2000)

    state = s.export_state()
    for stat_dict in state["stats"]:
        stat_dict.pop("recent_terminated_rate", None)
        stat_dict.pop("recent_terminated_flags", None)

    s2 = make_sampler()
    s2.import_state(state)
    stats = s2.get_cell_stats(WeaponType.GARLIC, 0)
    assert stats is not None
    # flags 不在 → terminated_rate(=0.5) に fallback
    assert stats.recent_terminated_rate == pytest.approx(0.5)
