"""tests/survivors/test_task_cell_sampler_callback.py - smoke tests"""
from __future__ import annotations
import sys
from pathlib import Path

_TRAINING_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAINING_ROOT))

import pytest
from unittest.mock import MagicMock, patch
from games.survivors.modules.task_cell_sampler_module import TaskCellSamplerStateModule
from games.survivors.modules.weapon_unlock_module import WeaponUnlockStateModule
from games.survivors.survivors_weapon_curriculum import WeaponType


def make_mock_hybrid_cb(current_phase=1):
    cb = MagicMock()
    cb.current_phase = current_phase
    return cb


def test_task_cell_sampler_callback_imports():
    """TaskCellSamplerCallback がインポートできること。"""
    from games.survivors.task_cell_sampler_callback import TaskCellSamplerCallback
    assert TaskCellSamplerCallback is not None


def test_active_pending_cell_lifecycle():
    """active/pending のライフサイクルが正しいことを確認する。

    training_start: active=None, pending=cellA, params=A送信
    episode 0 終了: active=None→skip, active=cellA, new cellB→pending+params送信
    episode 1 終了: active=cellA→stats更新, active=cellB, new cellC→pending+params送信
    """
    from games.survivors.modules.task_cell_sampler_module import TaskCell, TaskCellSamplerStateModule

    tcs = TaskCellSamplerStateModule(min_episodes_per_cell=5)
    tcs.rebuild_candidate_cells("WU0", 0, {0: 100})

    applied_cells = []  # apply() が呼ばれたセルを追跡

    # シミュレーション: training_start
    cell_0 = tcs.sample_cell(0)
    active_by_env = {0: None}
    pending_by_env = {0: cell_0}
    applied_cells.append(cell_0)  # params apply

    # シミュレーション: episode 0 終了
    active_cell = active_by_env[0]  # = None
    assert active_cell is None  # 最初のepisodeはstats更新をスキップ

    # pending -> active に昇格（pending=cell_0 → active=cell_0）
    active_by_env[0] = pending_by_env.get(0)
    cell_1 = tcs.sample_cell(100)
    pending_by_env[0] = cell_1
    applied_cells.append(cell_1)  # params apply

    # シミュレーション: episode 1 終了
    active_cell = active_by_env[0]  # = cell_0
    assert active_cell is not None
    assert active_cell == cell_0  # cell_0 で stats 更新される
    tcs.on_episode_end(active_cell, active_score=400.0, ep_len=200, terminated=False, num_timesteps=200)

    # pending -> active に昇格（pending=cell_1 → active=cell_1）
    active_by_env[0] = pending_by_env.get(0)
    cell_2 = tcs.sample_cell(200)
    pending_by_env[0] = cell_2
    applied_cells.append(cell_2)  # params apply

    # シミュレーション: episode 2 終了
    active_cell = active_by_env[0]  # = cell_1
    assert active_cell == cell_1

    # applied_cells と active セルの対応を確認:
    # applied[0]=cell_0 → episode 0 のパラメータ
    # episode 0 終了後: active=None（スキップ）、active=cell_0、applied[1]=cell_1
    # episode 1 終了後: active=cell_0→stats更新、active=cell_1、applied[2]=cell_2
    # episode 1 のスコアは cell_0 に記録される（cell_0 の params で動いたepisode）
    assert len(applied_cells) == 3

    # cell_0 の stats が更新されていることを確認
    stats_0 = tcs.get_cell_stats(cell_0.first_weapon_id, cell_0.enemy_phase_idx)
    assert stats_0 is not None
    assert stats_0.episode_count == 1
    assert stats_0.active_score_mean == pytest.approx(400.0)
