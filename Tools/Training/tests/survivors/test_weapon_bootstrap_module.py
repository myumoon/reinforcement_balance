"""tests/survivors/test_weapon_bootstrap_module.py

WeaponBootstrapStateModule のユニットテスト。
"""
from __future__ import annotations
import sys
from pathlib import Path

_TRAINING_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAINING_ROOT))

import pytest
from games.survivors.modules.weapon_bootstrap_module import WeaponBootstrapStateModule, WeaponBootstrapState
from games.survivors.modules.weapon_unlock_module import WeaponUnlockStateModule, WeaponUnlockAdvanceEvent
from games.survivors.modules.task_cell_sampler_module import TaskCell, TaskCellSamplerStateModule, TaskCellStats
from games.survivors.survivors_weapon_table import WEAPON_UNLOCK_ORDER, WeaponEntry
from games.survivors.survivors_weapon_curriculum import WeaponType


# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------

class MockStatsProvider:
    """TaskCellSamplerStateModule の get_stats_for_cell() のモック。"""

    def __init__(self):
        self._stats: dict[str, TaskCellStats] = {}

    def set_stats(self, cell: TaskCell, **kwargs):
        stats = TaskCellStats(cell=cell)
        for k, v in kwargs.items():
            setattr(stats, k, v)
        self._stats[cell.key()] = stats

    def get_stats_for_cell(self, cell: TaskCell) -> TaskCellStats | None:
        return self._stats.get(cell.key())


def make_module(
    initial_status: dict | None = None,
    initial_best_phase2_p10: dict | None = None,
    **kwargs,
) -> WeaponBootstrapStateModule:
    """テスト用のモジュールを作成するヘルパー。"""
    return WeaponBootstrapStateModule(
        weapon_unlock_order=WEAPON_UNLOCK_ORDER,
        initial_status=initial_status,
        initial_best_phase2_p10=initial_best_phase2_p10,
        solo_bootstrap_target_p10=300.0,
        solo_bootstrap_min_ep_len_p10=1200.0,
        solo_bootstrap_max_short_episode_rate=0.15,
        solo_bootstrap_min_episodes=40,
        integration_target_p10=300.0,
        integration_max_regression_from_solo=0.35,
        integration_min_episodes=40,
        maintenance_regression_ratio=0.35,
        maintenance_min_probe_episodes=20,
        **kwargs,
    )


def make_cell(weapon_id: int, phase: int = 2, task_kind: str = "solo_bootstrap") -> TaskCell:
    """テスト用のTaskCellを作成するヘルパー。"""
    return TaskCell(
        weapon_unlock_stage_key="WU0",
        first_weapon_id=weapon_id,
        enemy_phase_idx=phase,
        task_kind=task_kind,
        build_policy="target_only",
    )


# ---------------------------------------------------------------------------
# テスト: 初期化
# ---------------------------------------------------------------------------

def test_initial_status_locked_by_default():
    """initial_statusに記載のない武器がlockedになる。"""
    module = make_module()
    garlic_id = WeaponType.GARLIC
    state = module._states[garlic_id]
    assert state.status == "locked"


def test_initial_status_specified():
    """initial_statusに記載の武器が正しいステータスになる。"""
    module = make_module(initial_status={"garlic": "maintenance", "king_bible": "solo_bootstrap"})
    assert module._states[WeaponType.GARLIC].status == "maintenance"
    assert module._states[WeaponType.KING_BIBLE].status == "solo_bootstrap"
    # 他の武器は locked
    assert module._states[WeaponType.MAGIC_WAND].status == "locked"


def test_initial_best_phase2_p10():
    """initial_best_phase2_p10が正しく設定される。"""
    module = make_module(
        initial_status={"garlic": "maintenance"},
        initial_best_phase2_p10={"garlic": 590.0},
    )
    assert module._states[WeaponType.GARLIC].best_phase2_p10 == 590.0
    # 記載のない武器は 0.0
    assert module._states[WeaponType.KING_BIBLE].best_phase2_p10 == 0.0


# ---------------------------------------------------------------------------
# テスト: solo_bootstrap 完了判定
# ---------------------------------------------------------------------------

def test_solo_bootstrap_to_integration_on_condition():
    """solo_bootstrap完了条件でintegrationへ進む (phase2エピソードで条件を満たした場合)。"""
    module = make_module(initial_status={"garlic": "solo_bootstrap"})
    provider = MockStatsProvider()
    garlic_id = WeaponType.GARLIC
    cell = make_cell(garlic_id, phase=2, task_kind="solo_bootstrap")
    provider.set_stats(
        cell,
        episode_count=50,
        active_score_p10=350.0,
        episode_length_p10=1300.0,
        short_episode_rate=0.05,
    )

    module.on_episode_end(cell=cell, stats_provider=provider, current_stage_key="WU0", num_timesteps=1000)
    assert module._states[garlic_id].status == "integration"


def test_solo_bootstrap_not_ready_if_count_low():
    """solo_bootstrap: エピソード数が不足する場合は進まない。"""
    module = make_module(initial_status={"garlic": "solo_bootstrap"})
    provider = MockStatsProvider()
    garlic_id = WeaponType.GARLIC
    cell = make_cell(garlic_id, phase=2, task_kind="solo_bootstrap")
    provider.set_stats(
        cell,
        episode_count=10,  # 40未満
        active_score_p10=350.0,
        episode_length_p10=1300.0,
        short_episode_rate=0.05,
    )

    module.on_episode_end(cell=cell, stats_provider=provider, current_stage_key="WU0", num_timesteps=1000)
    assert module._states[garlic_id].status == "solo_bootstrap"


# ---------------------------------------------------------------------------
# テスト: best_phase2_p10 の更新タイミング
# ---------------------------------------------------------------------------

def test_best_phase2_p10_updated_only_on_phase2():
    """best_phase2_p10がphase_idx==2のepisodeでのみ更新される (phase0/1では更新されない)。"""
    module = make_module(initial_status={"garlic": "solo_bootstrap"})
    provider = MockStatsProvider()
    garlic_id = WeaponType.GARLIC

    # phase0 エピソード
    cell_phase0 = make_cell(garlic_id, phase=0, task_kind="solo_bootstrap")
    provider.set_stats(cell_phase0, episode_count=5, active_score_p10=200.0, episode_length_p10=1000.0, short_episode_rate=0.1)
    module.on_episode_end(cell=cell_phase0, stats_provider=provider, current_stage_key="WU0", num_timesteps=1000)
    assert module._states[garlic_id].best_phase2_p10 == 0.0  # 更新されない

    # phase1 エピソード
    cell_phase1 = make_cell(garlic_id, phase=1, task_kind="solo_bootstrap")
    provider.set_stats(cell_phase1, episode_count=5, active_score_p10=250.0, episode_length_p10=1100.0, short_episode_rate=0.1)
    module.on_episode_end(cell=cell_phase1, stats_provider=provider, current_stage_key="WU0", num_timesteps=2000)
    assert module._states[garlic_id].best_phase2_p10 == 0.0  # 更新されない

    # phase2 エピソード
    cell_phase2 = make_cell(garlic_id, phase=2, task_kind="solo_bootstrap")
    provider.set_stats(cell_phase2, episode_count=5, active_score_p10=400.0, episode_length_p10=1300.0, short_episode_rate=0.05)
    module.on_episode_end(cell=cell_phase2, stats_provider=provider, current_stage_key="WU0", num_timesteps=3000)
    assert module._states[garlic_id].best_phase2_p10 == 400.0  # 更新される


# ---------------------------------------------------------------------------
# テスト: integration 完了判定
# ---------------------------------------------------------------------------

def test_integration_to_maintenance():
    """integration完了条件でmaintenanceへ進む。"""
    module = make_module(initial_status={"garlic": "integration"})
    # best_phase2_p10 を手動設定
    module._states[WeaponType.GARLIC].best_phase2_p10 = 400.0
    provider = MockStatsProvider()
    garlic_id = WeaponType.GARLIC
    cell = make_cell(garlic_id, phase=2, task_kind="integration")
    provider.set_stats(
        cell,
        episode_count=50,
        active_score_p10=340.0,  # 340/400 = regression 0.15 <= 0.35
        episode_length_p10=1300.0,
        short_episode_rate=0.05,
    )

    module.on_episode_end(cell=cell, stats_provider=provider, current_stage_key="WU0", num_timesteps=1000)
    assert module._states[garlic_id].status == "maintenance"


def test_integration_not_ready_if_regression_high():
    """integration: regression_from_soloが高い場合は進まない。"""
    module = make_module(initial_status={"garlic": "integration"})
    module._states[WeaponType.GARLIC].best_phase2_p10 = 400.0
    provider = MockStatsProvider()
    garlic_id = WeaponType.GARLIC
    cell = make_cell(garlic_id, phase=2, task_kind="integration")
    provider.set_stats(
        cell,
        episode_count=50,
        active_score_p10=200.0,  # 200/400 = regression 0.5 > 0.35
        episode_length_p10=1300.0,
        short_episode_rate=0.05,
    )

    module.on_episode_end(cell=cell, stats_provider=provider, current_stage_key="WU0", num_timesteps=1000)
    assert module._states[garlic_id].status == "integration"


def test_regression_from_best_is_none_when_best_is_zero():
    """best_phase2_p10 <= 0 で regression_from_best が None になる。"""
    module = make_module(initial_status={"garlic": "maintenance"})
    module._states[WeaponType.GARLIC].best_phase2_p10 = 0.0  # 0
    provider = MockStatsProvider()
    garlic_id = WeaponType.GARLIC
    cell = make_cell(garlic_id, phase=2, task_kind="maintenance")
    provider.set_stats(cell, episode_count=30, active_score_p10=200.0)

    module.on_episode_end(cell=cell, stats_provider=provider, current_stage_key="WU0", num_timesteps=1000)
    # best_phase2_p10=0 なので regression_from_best は None のまま
    assert module._states[garlic_id].regression_from_best is None


# ---------------------------------------------------------------------------
# テスト: maintenance
# ---------------------------------------------------------------------------

def test_maintenance_regression_updates():
    """maintenance状態でphase2のregression_from_bestが更新される。"""
    module = make_module(initial_status={"garlic": "maintenance"})
    module._states[WeaponType.GARLIC].best_phase2_p10 = 400.0
    provider = MockStatsProvider()
    garlic_id = WeaponType.GARLIC
    cell = make_cell(garlic_id, phase=2, task_kind="maintenance")
    provider.set_stats(cell, episode_count=30, active_score_p10=200.0)

    module.on_episode_end(cell=cell, stats_provider=provider, current_stage_key="WU0", num_timesteps=1000)
    # regression = 1 - 200/400 = 0.5
    assert module._states[garlic_id].regression_from_best == pytest.approx(0.5)
    # 0.5 > 0.35 なので regression_count = 1
    assert module._states[garlic_id].regression_count == 1


# ---------------------------------------------------------------------------
# テスト: advance_to_next_stage
# ---------------------------------------------------------------------------

def test_advance_to_next_stage_updates_last_advance_step():
    """advance_to_next_stage()がlast_advance_stepをnum_timestepsで更新する。"""
    wu = WeaponUnlockStateModule(weapon_unlock_min_steps=0)
    event = wu.advance_to_next_stage(num_timesteps=50_000)
    assert event is not None
    assert wu._last_advance_step == 50_000
    assert event.from_stage_key == "WU0"
    assert event.to_stage_key == "WU1"


def test_advance_to_next_stage_on_final_stage_returns_none():
    """final_stage で advance_to_next_stage() が None を返す。"""
    wu = WeaponUnlockStateModule(weapon_unlock_min_steps=0)
    # 最終ステージまで進める
    for _ in range(len(WEAPON_UNLOCK_ORDER) - 1):
        wu.advance_to_next_stage(num_timesteps=0)
    assert wu.is_final_stage
    result = wu.advance_to_next_stage(num_timesteps=1000)
    assert result is None


# ---------------------------------------------------------------------------
# テスト: maybe_advance_stage
# ---------------------------------------------------------------------------

def test_maybe_advance_stage_advances_when_maintenance():
    """maybe_advance_stage()が現在stage武器がmaintenanceになった場合のみ次stageへ進める。"""
    wu = WeaponUnlockStateModule(weapon_unlock_min_steps=0)
    module = make_module(initial_status={"garlic": "maintenance"})

    event = module.maybe_advance_stage(weapon_unlock=wu, num_timesteps=10_000)
    assert event is not None
    assert event.from_stage_key == "WU0"


def test_maybe_advance_stage_no_advance_when_solo_bootstrap():
    """maybe_advance_stage()がsolo_bootstrapの場合は進めない。"""
    wu = WeaponUnlockStateModule(weapon_unlock_min_steps=0)
    module = make_module(initial_status={"garlic": "solo_bootstrap"})

    event = module.maybe_advance_stage(weapon_unlock=wu, num_timesteps=10_000)
    assert event is None


def test_maybe_advance_stage_no_advance_when_integration():
    """maybe_advance_stage()がintegrationの場合は進めない。"""
    wu = WeaponUnlockStateModule(weapon_unlock_min_steps=0)
    module = make_module(initial_status={"garlic": "integration"})

    event = module.maybe_advance_stage(weapon_unlock=wu, num_timesteps=10_000)
    assert event is None


# ---------------------------------------------------------------------------
# テスト: on_weapon_unlock_advanced
# ---------------------------------------------------------------------------

def test_on_weapon_unlock_advanced_sets_new_weapon_solo_bootstrap():
    """on_weapon_unlock_advanced()が新武器(locked)のみをsolo_bootstrapにする。"""
    module = make_module(initial_status={"garlic": "maintenance"})
    # 初期: garlic=maintenance, king_bible=locked
    assert module._states[WeaponType.KING_BIBLE].status == "locked"
    assert module._states[WeaponType.GARLIC].status == "maintenance"

    event = WeaponUnlockAdvanceEvent(
        from_stage_key="WU0",
        to_stage_key="WU1",
        new_weapon_id=WeaponType.KING_BIBLE,
        step=10_000,
    )
    module.on_weapon_unlock_advanced(event)

    # king_bible が solo_bootstrap になる
    assert module._states[WeaponType.KING_BIBLE].status == "solo_bootstrap"
    # garlic は変わらない
    assert module._states[WeaponType.GARLIC].status == "maintenance"


def test_on_weapon_unlock_advanced_does_not_change_non_locked():
    """on_weapon_unlock_advanced()が locked でない武器のステータスを変更しない。"""
    module = make_module(initial_status={"garlic": "integration", "king_bible": "solo_bootstrap"})

    event = WeaponUnlockAdvanceEvent(
        from_stage_key="WU1",
        to_stage_key="WU2",
        new_weapon_id=WeaponType.KING_BIBLE,  # solo_bootstrap
        step=10_000,
    )
    module.on_weapon_unlock_advanced(event)

    # solo_bootstrap のままで変わらない
    assert module._states[WeaponType.KING_BIBLE].status == "solo_bootstrap"


# ---------------------------------------------------------------------------
# テスト: export_state / import_state
# ---------------------------------------------------------------------------

def test_export_import_state_roundtrip():
    """export_state()/import_state()が状態を正しく往復できる。"""
    module = make_module(
        initial_status={"garlic": "maintenance", "king_bible": "integration"},
        initial_best_phase2_p10={"garlic": 590.0},
    )
    module._states[WeaponType.GARLIC].regression_from_best = 0.15
    module._states[WeaponType.GARLIC].regression_count = 3

    state = module.export_state()
    module2 = make_module()
    module2.import_state(state)

    assert module2._states[WeaponType.GARLIC].status == "maintenance"
    assert module2._states[WeaponType.GARLIC].best_phase2_p10 == 590.0
    assert module2._states[WeaponType.GARLIC].regression_from_best == pytest.approx(0.15)
    assert module2._states[WeaponType.GARLIC].regression_count == 3
    assert module2._states[WeaponType.KING_BIBLE].status == "integration"


# ---------------------------------------------------------------------------
# テスト: get_weapons_by_status
# ---------------------------------------------------------------------------

def test_get_weapons_by_status():
    """get_weapons_by_status()が正しく動作する。"""
    module = make_module(initial_status={
        "garlic": "maintenance",
        "king_bible": "solo_bootstrap",
        "magic_wand": "integration",
    })
    maintenance = module.get_weapons_by_status("maintenance")
    assert len(maintenance) == 1
    assert maintenance[0].weapon_id == WeaponType.GARLIC

    solo = module.get_weapons_by_status("solo_bootstrap")
    assert len(solo) == 1
    assert solo[0].weapon_id == WeaponType.KING_BIBLE

    locked = module.get_weapons_by_status("locked")
    assert all(s.weapon_id not in (WeaponType.GARLIC, WeaponType.KING_BIBLE, WeaponType.MAGIC_WAND)
               for s in locked)
