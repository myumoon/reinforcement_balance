"""tests/survivors/test_task_cell_sampler_bootstrap_lanes.py

TaskCellSamplerStateModule の Bootstrap Lane 機能テスト。
"""
from __future__ import annotations
import sys
from pathlib import Path

_TRAINING_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAINING_ROOT))

import pytest
from games.survivors.modules.task_cell_sampler_module import TaskCell, TaskCellSamplerStateModule, TaskCellStats
from games.survivors.modules.weapon_bootstrap_module import WeaponBootstrapStateModule
from games.survivors.survivors_weapon_table import WEAPON_UNLOCK_ORDER, build_weapon_params_for_cell
from games.survivors.survivors_weapon_curriculum import WeaponType


# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------

def make_tcs(**kwargs) -> TaskCellSamplerStateModule:
    defaults = dict(
        min_episodes_per_cell=5,
        target_p10=300.0,
        progress_scale=300.0,
        random_floor=0.05,
        weapon_unlock_order=WEAPON_UNLOCK_ORDER,
        integration_target_p10=300.0,
        maintenance_regression_ratio=0.35,
    )
    defaults.update(kwargs)
    return TaskCellSamplerStateModule(**defaults)


def make_bootstrap_module(initial_status: dict | None = None, initial_best: dict | None = None) -> WeaponBootstrapStateModule:
    return WeaponBootstrapStateModule(
        weapon_unlock_order=WEAPON_UNLOCK_ORDER,
        initial_status=initial_status,
        initial_best_phase2_p10=initial_best,
    )


# ---------------------------------------------------------------------------
# テスト: TaskCell.key()
# ---------------------------------------------------------------------------

def test_task_cell_key_format():
    """TaskCell.key()が '{task_kind}/{stage}/{weapon_id}/{phase}' 形式になる。"""
    cell = TaskCell(
        weapon_unlock_stage_key="WU0",
        first_weapon_id=WeaponType.GARLIC,
        enemy_phase_idx=2,
        task_kind="solo_bootstrap",
        build_policy="target_only",
    )
    assert cell.key() == f"solo_bootstrap/WU0/{WeaponType.GARLIC}/2"


def test_task_cell_default_task_kind():
    """TaskCellのデフォルト task_kind='wave_main', build_policy='' で既存コードが壊れない。"""
    cell = TaskCell(
        weapon_unlock_stage_key="WU0",
        first_weapon_id=WeaponType.GARLIC,
        enemy_phase_idx=2,
    )
    assert cell.task_kind == "wave_main"
    assert cell.build_policy == ""
    # keyが "wave_main/WU0/1/2" 形式
    assert cell.key() == f"wave_main/WU0/{WeaponType.GARLIC}/2"


# ---------------------------------------------------------------------------
# テスト: target_only ポリシー
# ---------------------------------------------------------------------------

def test_target_only_policy_builds_params():
    """target_onlyポリシーがbuild_weapon_params_for_cellで動作する。"""
    params = build_weapon_params_for_cell(
        first_weapon_id=WeaponType.KING_BIBLE,
        max_unlocked_stage_key="WU1",
        item_stage_key="IS0",
        pool_policy="target_only",
        weapon_unlock_order=WEAPON_UNLOCK_ORDER,
    )
    assert params["weapon_pool_mode"] == "fixed_subset"
    # target_only: first_weapon_id のみ
    assert params["allowed_weapon_types"] == [WeaponType.KING_BIBLE]
    # initial_slots は first_weapon_id のみ
    assert len(params["initial_weapon_slots"]) == 1
    assert params["initial_weapon_slots"][0]["weapon_id"] == WeaponType.KING_BIBLE


def test_target_only_policy_garlic():
    """target_only: Garlic でも Garlic のみが allowed になる。"""
    params = build_weapon_params_for_cell(
        first_weapon_id=WeaponType.GARLIC,
        max_unlocked_stage_key="WU0",
        item_stage_key="IS0",
        pool_policy="target_only",
        weapon_unlock_order=WEAPON_UNLOCK_ORDER,
    )
    assert params["allowed_weapon_types"] == [WeaponType.GARLIC]


# ---------------------------------------------------------------------------
# テスト: rebuild_bootstrap_candidate_cells
# ---------------------------------------------------------------------------

def test_rebuild_bootstrap_candidate_cells_creates_solo_bootstrap_cells():
    """rebuild_bootstrap_candidate_cells()がsolo_bootstrapセルを構築する。"""
    tcs = make_tcs()
    bs = make_bootstrap_module(initial_status={"garlic": "solo_bootstrap"})
    tcs.rebuild_bootstrap_candidate_cells(
        stage_key="WU0",
        max_unlocked_enemy_phase_idx=2,
        min_episode_steps_by_phase={0: 600, 1: 900, 2: 1200},
        weapon_bootstrap=bs,
    )
    solo_cells = [c for c in tcs._candidate_cells if c.task_kind == "solo_bootstrap"]
    assert len(solo_cells) > 0
    # solo_bootstrap は phase 0..2 のセル
    phases = [c.enemy_phase_idx for c in solo_cells]
    assert 0 in phases
    assert 2 in phases


def test_rebuild_bootstrap_candidate_cells_excludes_locked():
    """locked武器は候補セルに出ない。"""
    tcs = make_tcs()
    bs = make_bootstrap_module(initial_status={"garlic": "solo_bootstrap"})
    tcs.rebuild_bootstrap_candidate_cells(
        stage_key="WU0",
        max_unlocked_enemy_phase_idx=2,
        min_episode_steps_by_phase={0: 600, 1: 900, 2: 1200},
        weapon_bootstrap=bs,
    )
    # locked 武器 (king_bible など) が候補に含まれない
    locked_cells = [c for c in tcs._candidate_cells if c.first_weapon_id == WeaponType.KING_BIBLE]
    assert len(locked_cells) == 0


def test_rebuild_bootstrap_candidate_cells_phase2_always_included():
    """solo_bootstrapのphase2セルが常に候補に含まれる (max_enemy_phase >= 2の場合)。"""
    tcs = make_tcs()
    bs = make_bootstrap_module(initial_status={"garlic": "solo_bootstrap"})
    tcs.rebuild_bootstrap_candidate_cells(
        stage_key="WU0",
        max_unlocked_enemy_phase_idx=3,
        min_episode_steps_by_phase={0: 600, 1: 900, 2: 1200, 3: 1200},
        weapon_bootstrap=bs,
    )
    garlic_phase2 = [c for c in tcs._candidate_cells
                     if c.first_weapon_id == WeaponType.GARLIC
                     and c.task_kind == "solo_bootstrap"
                     and c.enemy_phase_idx == 2]
    assert len(garlic_phase2) >= 1


def test_rebuild_bootstrap_creates_integration_and_maintenance():
    """rebuild_bootstrap_candidate_cells()がintegration/maintenanceセルを構築する。"""
    tcs = make_tcs()
    bs = make_bootstrap_module(initial_status={
        "garlic": "maintenance",
        "king_bible": "integration",
    })
    tcs.rebuild_bootstrap_candidate_cells(
        stage_key="WU1",
        max_unlocked_enemy_phase_idx=2,
        min_episode_steps_by_phase={0: 600, 1: 900, 2: 1200},
        weapon_bootstrap=bs,
    )
    integration_cells = [c for c in tcs._candidate_cells if c.task_kind == "integration"]
    maintenance_cells = [c for c in tcs._candidate_cells if c.task_kind == "maintenance"]
    assert len(integration_cells) >= 1
    assert len(maintenance_cells) >= 1


# ---------------------------------------------------------------------------
# テスト: sample_cell_with_lane_mix
# ---------------------------------------------------------------------------

def test_sample_cell_with_lane_mix_returns_cell():
    """sample_cell_with_lane_mix()がレーン選択後にレーン内セルを選ぶ。"""
    tcs = make_tcs()
    bs = make_bootstrap_module(initial_status={"garlic": "solo_bootstrap"})
    tcs.rebuild_bootstrap_candidate_cells(
        stage_key="WU0",
        max_unlocked_enemy_phase_idx=2,
        min_episode_steps_by_phase={0: 600, 1: 900, 2: 1200},
        weapon_bootstrap=bs,
    )
    sample_mix = {"solo_bootstrap": 1.0, "weak_cells": 0.0, "maintenance": 0.0, "integration": 0.0}
    cell = tcs.sample_cell_with_lane_mix(
        num_timesteps=0,
        weapon_bootstrap=bs,
        sample_mix=sample_mix,
    )
    assert isinstance(cell, TaskCell)
    assert cell.task_kind == "solo_bootstrap"


def test_sample_cell_with_lane_mix_regression_in_weak_cells():
    """regressionしたmaintenanceセルがweak_cellsにも入る。"""
    tcs = make_tcs(min_episodes_per_cell=1)
    bs = make_bootstrap_module(
        initial_status={"garlic": "maintenance"},
        initial_best={"garlic": 400.0},
    )
    tcs.rebuild_bootstrap_candidate_cells(
        stage_key="WU0",
        max_unlocked_enemy_phase_idx=2,
        min_episode_steps_by_phase={0: 600, 1: 900, 2: 1200},
        weapon_bootstrap=bs,
    )
    # maintenance セルのstatsを設定（regression状態）
    maintenance_cell = next(c for c in tcs._candidate_cells if c.task_kind == "maintenance")
    stats = tcs._stats[maintenance_cell.key()]
    stats.episode_count = 10
    stats.active_score_p10 = 200.0  # 200/400 = 0.5 > 0.35 (regression)

    # weak_cells レーンのみでサンプル
    sample_mix = {"solo_bootstrap": 0.0, "weak_cells": 1.0, "maintenance": 0.0, "integration": 0.0}
    cell = tcs.sample_cell_with_lane_mix(
        num_timesteps=0,
        weapon_bootstrap=bs,
        sample_mix=sample_mix,
    )
    assert cell.task_kind == "maintenance"


# ---------------------------------------------------------------------------
# テスト: Garlic の bootstrap status が integration の build_policy に影響
# ---------------------------------------------------------------------------

def test_integration_target_only_when_garlic_not_maintenance():
    """Garlicがbootstrapでmaintenanceでないときintegrationにtarget_onlyが適用される。"""
    tcs = make_tcs()
    # garlic は solo_bootstrap、king_bible は integration
    bs = make_bootstrap_module(initial_status={
        "garlic": "solo_bootstrap",
        "king_bible": "integration",
    })
    tcs.rebuild_bootstrap_candidate_cells(
        stage_key="WU1",
        max_unlocked_enemy_phase_idx=2,
        min_episode_steps_by_phase={0: 600, 1: 900, 2: 1200},
        weapon_bootstrap=bs,
    )
    integration_cells = [c for c in tcs._candidate_cells
                         if c.task_kind == "integration"
                         and c.first_weapon_id == WeaponType.KING_BIBLE]
    assert len(integration_cells) >= 1
    for cell in integration_cells:
        assert cell.build_policy == "target_only"


def test_integration_target_plus_anchor_when_garlic_maintenance():
    """Garlicがmaintenanceかつアンロック済みのときintegrationにtarget_plus_anchor_if_unlockedが適用される。"""
    tcs = make_tcs()
    # garlic は maintenance、king_bible は integration (WU1 でアンロック)
    bs = make_bootstrap_module(initial_status={
        "garlic": "maintenance",
        "king_bible": "integration",
    })
    tcs.rebuild_bootstrap_candidate_cells(
        stage_key="WU1",  # king_bible がアンロックされているstage
        max_unlocked_enemy_phase_idx=2,
        min_episode_steps_by_phase={0: 600, 1: 900, 2: 1200},
        weapon_bootstrap=bs,
    )
    integration_cells = [c for c in tcs._candidate_cells
                         if c.task_kind == "integration"
                         and c.first_weapon_id == WeaponType.KING_BIBLE]
    assert len(integration_cells) >= 1
    for cell in integration_cells:
        assert cell.build_policy == "target_plus_anchor_if_unlocked"


# ---------------------------------------------------------------------------
# テスト: weapon_bootstrap=None の場合は従来ロジック
# ---------------------------------------------------------------------------

def test_weapon_bootstrap_none_uses_legacy_sample_cell():
    """weapon_bootstrap=Noneの場合は従来rebuild_candidate_cells()とsample_cell()が使われる。"""
    tcs = make_tcs()
    tcs.rebuild_candidate_cells(
        stage_key="WU0",
        max_unlocked_enemy_phase_idx=2,
        min_episode_steps_by_phase={0: 600, 1: 900, 2: 1200},
    )
    # task_kind がすべて wave_main
    for cell in tcs._candidate_cells:
        assert cell.task_kind == "wave_main"

    cell = tcs.sample_cell(0)
    assert isinstance(cell, TaskCell)
    assert cell.task_kind == "wave_main"


# ---------------------------------------------------------------------------
# テスト: import_state の旧形式key変換
# ---------------------------------------------------------------------------

def test_import_state_old_format_key_conversion():
    """旧形式key(WU0/7/2 形式: '/' が2つ)をimport_stateでwave_main/WU0/7/2として扱える。"""
    tcs = make_tcs()

    # 旧形式のstate (cell_key に '/' が2つのみ)
    old_state = {
        "current_stage_key": "WU0",
        "max_unlocked_enemy_phase_idx": 2,
        "min_episode_steps_by_phase": {"0": 600, "1": 900, "2": 1200},
        "stats": [
            {
                "cell_key": f"WU0/{WeaponType.GARLIC}/2",  # 旧形式
                "cell": {
                    "weapon_unlock_stage_key": "WU0",
                    "first_weapon_id": WeaponType.GARLIC,
                    "enemy_phase_idx": 2,
                    # task_kind と build_policy はなし（旧形式）
                },
                "episode_count": 100,
                "recent_scores": [300.0, 350.0],
                "recent_episode_lengths": [1200, 1300],
                "terminated_count": 50,
                "truncated_count": 50,
                "previous_score_mean": 300.0,
                "active_score_mean": 325.0,
                "active_score_p10": 300.0,
                "active_score_cv": 0.1,
                "episode_length_mean": 1250.0,
                "terminated_rate": 0.5,
                "learning_progress": 25.0,
                "regression_score": 0.0,
                "last_sample_step": 0,
                "blocked_until_step": 0,
                "episode_length_p10": 1200.0,
                "short_episode_rate": 0.05,
                "recent_terminated_rate": 0.3,
                "recent_terminated_flags": [True, False],
            }
        ],
    }

    tcs.import_state(old_state)

    # wave_main/WU0/{GARLIC}/2 形式で参照できる
    wave_main_cell = TaskCell(
        weapon_unlock_stage_key="WU0",
        first_weapon_id=WeaponType.GARLIC,
        enemy_phase_idx=2,
        task_kind="wave_main",
    )
    stats = tcs.get_stats_for_cell(wave_main_cell)
    assert stats is not None
    assert stats.episode_count == 100


# ---------------------------------------------------------------------------
# テスト: status 遷移後の候補セル再構築
# ---------------------------------------------------------------------------

def test_on_episode_end_returns_true_on_status_change():
    """on_episode_end() が solo_bootstrap→integration 遷移時に True を返す。"""
    bs = make_bootstrap_module(initial_status={"garlic": "solo_bootstrap"})
    tcs = make_tcs()

    # solo_bootstrap phase2 セルで条件を満たすように stats を設定
    cell = TaskCell(
        weapon_unlock_stage_key="WU0",
        first_weapon_id=WeaponType.GARLIC,
        enemy_phase_idx=2,
        task_kind="solo_bootstrap",
        build_policy="target_only",
    )
    # stats を TCS に登録
    tcs.rebuild_bootstrap_candidate_cells(
        stage_key="WU0",
        max_unlocked_enemy_phase_idx=2,
        min_episode_steps_by_phase={0: 600, 1: 900, 2: 1200},
        weapon_bootstrap=bs,
    )
    # stats を手動で設定（完了条件を満たす）
    stats = tcs._stats.get(cell.key())
    if stats is None:
        from games.survivors.modules.task_cell_sampler_module import TaskCellStats
        stats = TaskCellStats(cell=cell)
        tcs._stats[cell.key()] = stats
    stats.episode_count = 50
    stats.active_score_p10 = 350.0
    stats.episode_length_p10 = 1300.0
    stats.short_episode_rate = 0.05

    result = bs.on_episode_end(
        cell=cell,
        stats_provider=tcs,
        current_stage_key="WU0",
        num_timesteps=1000,
    )
    assert result is True
    assert bs._states[WeaponType.GARLIC].status == "integration"


def test_candidate_cells_rebuilt_after_status_change():
    """status 遷移後に候補セルが再構築されると integration セルが含まれる。"""
    bs = make_bootstrap_module(initial_status={"garlic": "solo_bootstrap"})
    tcs = make_tcs()

    # 初期候補セル（solo_bootstrap のみ）
    tcs.rebuild_bootstrap_candidate_cells(
        stage_key="WU0",
        max_unlocked_enemy_phase_idx=2,
        min_episode_steps_by_phase={0: 600, 1: 900, 2: 1200},
        weapon_bootstrap=bs,
    )
    initial_task_kinds = {c.task_kind for c in tcs._candidate_cells}
    assert "integration" not in initial_task_kinds

    # status を integration に変更
    bs._states[WeaponType.GARLIC].status = "integration"

    # 候補セルを再構築
    tcs.rebuild_bootstrap_candidate_cells(
        stage_key="WU0",
        max_unlocked_enemy_phase_idx=2,
        min_episode_steps_by_phase={0: 600, 1: 900, 2: 1200},
        weapon_bootstrap=bs,
    )
    task_kinds = {c.task_kind for c in tcs._candidate_cells}
    assert "integration" in task_kinds
