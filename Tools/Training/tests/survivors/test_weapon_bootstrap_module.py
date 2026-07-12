"""tests/survivors/test_weapon_bootstrap_module.py WeaponBootstrapStateModule のユニットテスト。"""
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
    """テスト用のモジュールを作成するヘルパー。kwargs でデフォルト値をオーバーライドできる。"""
    defaults = dict(
        solo_bootstrap_target_p10=300.0,
        solo_bootstrap_min_ep_len_p10=1200.0,
        solo_bootstrap_max_short_episode_rate=0.15,
        solo_bootstrap_min_episodes=40,
        integration_target_p10=300.0,
        integration_max_regression_from_solo=0.35,
        integration_min_episodes=40,
        maintenance_regression_ratio=0.35,
        maintenance_min_probe_episodes=20,
    )
    defaults.update(kwargs)
    return WeaponBootstrapStateModule(
        weapon_unlock_order=WEAPON_UNLOCK_ORDER,
        initial_status=initial_status,
        initial_best_phase2_p10=initial_best_phase2_p10,
        **defaults,
    )


def make_cell(weapon_id: int, phase: int, task_kind: str = "solo_bootstrap") -> TaskCell:
    return TaskCell(
        first_weapon_id=weapon_id,
        weapon_unlock_stage_key="WU0",
        task_kind=task_kind,
        enemy_phase_idx=phase,
    )


def _set_det(module: WeaponBootstrapStateModule, weapon_id: int, task_kind: str, *,
             p10: float = 350.0, ep_len: float = 1300.0, short_rate: float = 0.05,
             enemy_phase_idx: int = 2, num_timesteps: int = 1000) -> None:
    """テスト用のヘルパー: deterministic 結果を一括設定する。"""
    module.set_deterministic_result(
        weapon_id=weapon_id,
        task_kind=task_kind,
        enemy_phase_idx=enemy_phase_idx,
        p10=p10,
        episode_length_p10=ep_len,
        short_episode_rate=short_rate,
        num_timesteps=num_timesteps,
    )


# ---------------------------------------------------------------------------
# テスト: 初期状態
# ---------------------------------------------------------------------------

def test_initial_status_locked_by_default():
    """全武器のステータスが初期値 locked になる。"""
    module = make_module()
    for entry in WEAPON_UNLOCK_ORDER:
        state = module._states[entry.weapon_id]
        assert state.status == "locked"


def test_initial_status_specified():
    """initial_status で指定された武器のステータスが設定される。"""
    module = make_module(
        initial_status={"garlic": "maintenance", "king_bible": "solo_bootstrap"},
    )
    assert module._states[WeaponType.GARLIC].status == "maintenance"
    assert module._states[WeaponType.KING_BIBLE].status == "solo_bootstrap"
    assert module._states[WeaponType.MAGIC_WAND].status == "locked"


def test_initial_best_phase2_p10():
    """initial_best_phase2_p10 で指定された武器の best_phase2_p10 が設定される。"""
    module = make_module(
        initial_status={"garlic": "maintenance"},
        initial_best_phase2_p10={"garlic": 590.0},
    )
    assert module._states[WeaponType.GARLIC].best_phase2_p10 == 590.0
    assert module._states[WeaponType.KING_BIBLE].best_phase2_p10 == 0.0


# ---------------------------------------------------------------------------
# テスト: solo_bootstrap 完了判定（deterministic ゲート）
# ---------------------------------------------------------------------------

def test_solo_bootstrap_to_integration_on_condition():
    """solo_bootstrap: deterministic 結果あり + 全条件満足で integration へ昇格する。"""
    module = make_module(initial_status={"garlic": "solo_bootstrap"})
    garlic_id = WeaponType.GARLIC
    provider = MockStatsProvider()
    cell = make_cell(garlic_id, phase=2, task_kind="solo_bootstrap")
    provider.set_stats(
        cell,
        episode_count=50,
        active_score_p10=350.0,
        episode_length_p10=1300.0,
        short_episode_rate=0.05,
    )
    # deterministic 結果を注入
    _set_det(module, garlic_id, "solo_bootstrap", p10=350.0, ep_len=1300.0, short_rate=0.05)

    module.on_episode_end(cell=cell, stats_provider=provider, current_stage_key="WU0", num_timesteps=1000)
    assert module._states[garlic_id].status == "integration"


def test_solo_bootstrap_not_ready_if_count_low():
    """solo_bootstrap: エピソード数が不足する場合は進まない（deterministic 結果があっても）。"""
    module = make_module(initial_status={"garlic": "solo_bootstrap"})
    garlic_id = WeaponType.GARLIC
    provider = MockStatsProvider()
    cell = make_cell(garlic_id, phase=2, task_kind="solo_bootstrap")
    provider.set_stats(
        cell,
        episode_count=10,  # 40未満
        active_score_p10=350.0,
        episode_length_p10=1300.0,
        short_episode_rate=0.05,
    )
    _set_det(module, garlic_id, "solo_bootstrap", p10=350.0, ep_len=1300.0, short_rate=0.05)

    module.on_episode_end(cell=cell, stats_provider=provider, current_stage_key="WU0", num_timesteps=1000)
    assert module._states[garlic_id].status == "solo_bootstrap"


def test_solo_bootstrap_not_complete_on_phase0():
    """solo_bootstrap: phase0 のエピソードで全条件を満たしても integration に昇格しない。"""
    module = make_module(initial_status={"garlic": "solo_bootstrap"})
    garlic_id = WeaponType.GARLIC
    provider = MockStatsProvider()
    cell = make_cell(garlic_id, phase=0, task_kind="solo_bootstrap")
    provider.set_stats(
        cell,
        episode_count=50,
        active_score_p10=350.0,
        episode_length_p10=1300.0,
        short_episode_rate=0.05,
    )
    _set_det(module, garlic_id, "solo_bootstrap", p10=350.0, ep_len=1300.0, short_rate=0.05)

    module.on_episode_end(cell=cell, stats_provider=provider, current_stage_key="WU0", num_timesteps=1000)
    # phase0 なので完了判定はスキップ → solo_bootstrap のまま
    assert module._states[garlic_id].status == "solo_bootstrap"


def test_solo_bootstrap_not_complete_on_phase1():
    """solo_bootstrap: phase1 のエピソードで全条件を満たしても integration に昇格しない。"""
    module = make_module(initial_status={"garlic": "solo_bootstrap"})
    garlic_id = WeaponType.GARLIC
    provider = MockStatsProvider()
    cell = make_cell(garlic_id, phase=1, task_kind="solo_bootstrap")
    provider.set_stats(
        cell,
        episode_count=50,
        active_score_p10=350.0,
        episode_length_p10=1300.0,
        short_episode_rate=0.05,
    )
    _set_det(module, garlic_id, "solo_bootstrap", p10=350.0, ep_len=1300.0, short_rate=0.05)

    module.on_episode_end(cell=cell, stats_provider=provider, current_stage_key="WU0", num_timesteps=1000)
    # phase1 なので完了判定はスキップ → solo_bootstrap のまま
    assert module._states[garlic_id].status == "solo_bootstrap"


# ---------------------------------------------------------------------------
# テスト: best_phase2_p10 の更新タイミング（gate 通過時に deterministic 値で設定）
# ---------------------------------------------------------------------------

def test_best_phase2_p10_updated_only_on_phase2():
    """best_phase2_p10 は solo_bootstrap → integration gate 通過時に det_p10 で設定される。"""
    module = make_module(initial_status={"garlic": "solo_bootstrap"})
    garlic_id = WeaponType.GARLIC
    provider = MockStatsProvider()

    # phase0 エピソード（gate には関係しない）
    cell_phase0 = make_cell(garlic_id, phase=0, task_kind="solo_bootstrap")
    provider.set_stats(cell_phase0, episode_count=5, active_score_p10=200.0, episode_length_p10=1000.0, short_episode_rate=0.1)
    module.on_episode_end(cell=cell_phase0, stats_provider=provider, current_stage_key="WU0", num_timesteps=1000)
    assert module._states[garlic_id].best_phase2_p10 == 0.0  # 更新されない

    # phase1 エピソード（gate には関係しない）
    cell_phase1 = make_cell(garlic_id, phase=1, task_kind="solo_bootstrap")
    provider.set_stats(cell_phase1, episode_count=5, active_score_p10=250.0, episode_length_p10=1100.0, short_episode_rate=0.1)
    module.on_episode_end(cell=cell_phase1, stats_provider=provider, current_stage_key="WU0", num_timesteps=2000)
    assert module._states[garlic_id].best_phase2_p10 == 0.0  # 更新されない

    # phase2 エピソード + deterministic 結果設定 → gate 通過 → best_phase2_p10 が det_p10 に設定される
    cell_phase2 = make_cell(garlic_id, phase=2, task_kind="solo_bootstrap")
    provider.set_stats(cell_phase2, episode_count=50, active_score_p10=400.0, episode_length_p10=1300.0, short_episode_rate=0.05)
    _set_det(module, garlic_id, "solo_bootstrap", p10=350.0, ep_len=1300.0, short_rate=0.05, num_timesteps=3000)
    module.on_episode_end(cell=cell_phase2, stats_provider=provider, current_stage_key="WU0", num_timesteps=3000)
    # gate 通過時に det_p10=350.0 が best_phase2_p10 に設定される
    assert module._states[garlic_id].best_phase2_p10 == 350.0
    assert module._states[garlic_id].status == "integration"


# ---------------------------------------------------------------------------
# テスト: integration 完了判定（deterministic ゲート）
# ---------------------------------------------------------------------------

def test_integration_to_maintenance():
    """integration: deterministic 結果あり + 全条件満足で maintenance へ昇格する。"""
    module = make_module(initial_status={"garlic": "integration"})
    module._states[WeaponType.GARLIC].best_phase2_p10 = 400.0
    garlic_id = WeaponType.GARLIC
    provider = MockStatsProvider()
    cell = make_cell(garlic_id, phase=2, task_kind="integration")
    provider.set_stats(
        cell,
        episode_count=50,
        active_score_p10=340.0,
        episode_length_p10=1300.0,
        short_episode_rate=0.05,
    )
    # deterministic 結果: regression = 1 - 340/400 = 0.15 <= 0.35
    _set_det(module, garlic_id, "integration", p10=340.0, ep_len=1300.0, short_rate=0.05)

    module.on_episode_end(cell=cell, stats_provider=provider, current_stage_key="WU0", num_timesteps=1000)
    assert module._states[garlic_id].status == "maintenance"


def test_integration_not_ready_if_regression_high():
    """integration: regression_from_solo が高い場合は進まない。"""
    module = make_module(initial_status={"garlic": "integration"})
    module._states[WeaponType.GARLIC].best_phase2_p10 = 400.0
    garlic_id = WeaponType.GARLIC
    provider = MockStatsProvider()
    cell = make_cell(garlic_id, phase=2, task_kind="integration")
    provider.set_stats(
        cell,
        episode_count=50,
        active_score_p10=200.0,
        episode_length_p10=1300.0,
        short_episode_rate=0.05,
    )
    # deterministic: regression = 1 - 200/400 = 0.5 > 0.35 → 昇格しない
    _set_det(module, garlic_id, "integration", p10=200.0, ep_len=1300.0, short_rate=0.05)

    module.on_episode_end(cell=cell, stats_provider=provider, current_stage_key="WU0", num_timesteps=1000)
    assert module._states[garlic_id].status == "integration"


def test_regression_from_best_is_none_when_below_min_probe_episodes():
    """maintenance_min_probe_episodes 未満の episode_count では regression_from_best が None になる。

    新ロジックでは deterministic 未設定（det_fresh=False）のとき regression_from_best=None。
    """
    module = make_module(initial_status={"garlic": "maintenance"}, maintenance_min_probe_episodes=20)
    module._states[WeaponType.GARLIC].best_phase2_p10 = 400.0
    garlic_id = WeaponType.GARLIC
    provider = MockStatsProvider()
    cell = make_cell(garlic_id, phase=2, task_kind="maintenance")
    provider.set_stats(cell, episode_count=5, active_score_p10=200.0)
    # deterministic 結果を設定しない → det_fresh=False → regression_from_best=None

    module.on_episode_end(cell=cell, stats_provider=provider, current_stage_key="WU0", num_timesteps=1000)
    assert module._states[garlic_id].regression_from_best is None


# ---------------------------------------------------------------------------
# テスト: maintenance（deterministic ゲート）
# ---------------------------------------------------------------------------

def test_maintenance_regression_updates():
    """maintenance: deterministic 結果あり + regression > ratio で regression_count が増える。"""
    module = make_module(initial_status={"garlic": "maintenance"})
    module._states[WeaponType.GARLIC].best_phase2_p10 = 400.0
    garlic_id = WeaponType.GARLIC
    provider = MockStatsProvider()
    cell = make_cell(garlic_id, phase=2, task_kind="maintenance")
    provider.set_stats(cell, episode_count=30, active_score_p10=200.0)
    # deterministic 結果設定
    _set_det(module, garlic_id, "maintenance", p10=200.0, ep_len=1300.0, short_rate=0.05)

    module.on_episode_end(cell=cell, stats_provider=provider, current_stage_key="WU0", num_timesteps=1000)
    # regression = 1 - 200/400 = 0.5
    assert module._states[garlic_id].regression_from_best == pytest.approx(0.5)
    # 0.5 > 0.35 なので regression_count = 1
    assert module._states[garlic_id].regression_count == 1


# ---------------------------------------------------------------------------
# テスト: advance_to_next_stage
# ---------------------------------------------------------------------------

def test_advance_to_next_stage_updates_last_advance_step():
    """advance_to_next_stage() が last_advance_step を num_timesteps で更新する。"""
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
    """maybe_advance_stage() が現在 stage 武器が maintenance のとき次 stage へ進める。"""
    wu = WeaponUnlockStateModule(weapon_unlock_min_steps=0)
    module = make_module(initial_status={"garlic": "maintenance"})
    event = module.maybe_advance_stage(weapon_unlock=wu, num_timesteps=10_000)
    assert event is not None
    assert event.from_stage_key == "WU0"


def test_maybe_advance_stage_no_advance_when_solo_bootstrap():
    """maybe_advance_stage() が solo_bootstrap の場合は進めない。"""
    wu = WeaponUnlockStateModule(weapon_unlock_min_steps=0)
    module = make_module(initial_status={"garlic": "solo_bootstrap"})
    event = module.maybe_advance_stage(weapon_unlock=wu, num_timesteps=10_000)
    assert event is None


def test_maybe_advance_stage_no_advance_when_integration():
    """maybe_advance_stage() が integration の場合は進めない。"""
    wu = WeaponUnlockStateModule(weapon_unlock_min_steps=0)
    module = make_module(initial_status={"garlic": "integration"})
    event = module.maybe_advance_stage(weapon_unlock=wu, num_timesteps=10_000)
    assert event is None


# ---------------------------------------------------------------------------
# テスト: on_weapon_unlock_advanced
# ---------------------------------------------------------------------------

def test_on_weapon_unlock_advanced_sets_new_weapon_solo_bootstrap():
    """on_weapon_unlock_advanced() が新規武器のステータスを solo_bootstrap にする。"""
    module = make_module()
    assert module._states[WeaponType.KING_BIBLE].status == "locked"
    assert module._states[WeaponType.GARLIC].status == "locked"

    event = WeaponUnlockAdvanceEvent(
        from_stage_key="WU0",
        to_stage_key="WU1",
        new_weapon_id=WeaponType.KING_BIBLE,
        step=10_000,
    )
    module.on_weapon_unlock_advanced(event)
    assert module._states[WeaponType.KING_BIBLE].status == "solo_bootstrap"
    assert module._states[WeaponType.GARLIC].status == "locked"


def test_on_weapon_unlock_advanced_does_not_change_non_locked():
    """on_weapon_unlock_advanced() が locked 以外のステータスを変更しない。

    KING_BIBLE を integration 状態に初期化してから unlock イベントを発火しても
    ステータスが変化しないことを確認する（locked 以外は保護される）。
    """
    module = make_module(initial_status={"king_bible": "integration"})

    # king_bible を integration 状態に初期化されているはず
    assert module._states[WeaponType.KING_BIBLE].status == "integration"

    event = WeaponUnlockAdvanceEvent(
        from_stage_key="WU1",
        to_stage_key="WU2",
        new_weapon_id=WeaponType.KING_BIBLE,
        step=20_000,
    )
    module.on_weapon_unlock_advanced(event)
    # integration のまま変化しない
    assert module._states[WeaponType.KING_BIBLE].status == "integration"


# ---------------------------------------------------------------------------
# テスト: export_state / import_state の往復
# ---------------------------------------------------------------------------

def test_export_import_state_roundtrip():
    """export_state() / import_state() が状態を正しく往復できる。"""
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
    """get_weapons_by_status() が正しく動作する。"""
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


# ---------------------------------------------------------------------------
# テスト: on_episode_end の戻り値（status 変化フラグ）
# ---------------------------------------------------------------------------

def test_on_episode_end_returns_true_on_solo_to_integration():
    """solo_bootstrap → integration 遷移時に on_episode_end が True を返す。"""
    module = make_module(initial_status={"garlic": "solo_bootstrap"})
    garlic_id = WeaponType.GARLIC
    provider = MockStatsProvider()
    cell = make_cell(garlic_id, phase=2, task_kind="solo_bootstrap")
    provider.set_stats(
        cell,
        episode_count=50,
        active_score_p10=350.0,
        episode_length_p10=1300.0,
        short_episode_rate=0.05,
    )
    _set_det(module, garlic_id, "solo_bootstrap", p10=350.0, ep_len=1300.0, short_rate=0.05)

    result = module.on_episode_end(cell=cell, stats_provider=provider, current_stage_key="WU0", num_timesteps=1000)
    assert result is True
    assert module._states[garlic_id].status == "integration"


def test_on_episode_end_returns_false_when_no_transition():
    """status 遷移がない場合に on_episode_end が False を返す。"""
    module = make_module(initial_status={"garlic": "solo_bootstrap"})
    garlic_id = WeaponType.GARLIC
    provider = MockStatsProvider()
    cell = make_cell(garlic_id, phase=2, task_kind="solo_bootstrap")
    provider.set_stats(
        cell,
        episode_count=10,  # 40未満でready=False
        active_score_p10=350.0,
        episode_length_p10=1300.0,
        short_episode_rate=0.05,
    )

    result = module.on_episode_end(cell=cell, stats_provider=provider, current_stage_key="WU0", num_timesteps=1000)
    assert result is False
    assert module._states[garlic_id].status == "solo_bootstrap"


def test_on_episode_end_returns_true_on_integration_to_maintenance():
    """integration → maintenance 遷移時に on_episode_end が True を返す。"""
    module = make_module(initial_status={"garlic": "integration"})
    module._states[WeaponType.GARLIC].best_phase2_p10 = 400.0
    garlic_id = WeaponType.GARLIC
    provider = MockStatsProvider()
    cell = make_cell(garlic_id, phase=2, task_kind="integration")
    provider.set_stats(
        cell,
        episode_count=50,
        active_score_p10=340.0,
        episode_length_p10=1300.0,
        short_episode_rate=0.05,
    )
    _set_det(module, garlic_id, "integration", p10=340.0, ep_len=1300.0, short_rate=0.05)

    result = module.on_episode_end(cell=cell, stats_provider=provider, current_stage_key="WU0", num_timesteps=1000)
    assert result is True
    assert module._states[garlic_id].status == "maintenance"


# ---------------------------------------------------------------------------
# テスト: task_kind による判定の絞り込み
# ---------------------------------------------------------------------------

def test_integration_not_triggered_by_solo_bootstrap_cell():
    """status=integration のときでも task_kind=solo_bootstrap のセルでは integration 完了判定をしない。"""
    module = make_module(initial_status={"garlic": "integration"})
    module._states[WeaponType.GARLIC].best_phase2_p10 = 400.0
    garlic_id = WeaponType.GARLIC
    provider = MockStatsProvider()
    # task_kind が solo_bootstrap のままのセル（status 遷移後に古いセルが終了した場合を再現）
    cell = make_cell(garlic_id, phase=2, task_kind="solo_bootstrap")
    provider.set_stats(
        cell,
        episode_count=50,
        active_score_p10=340.0,  # 条件は満たしているが task_kind が違う
        episode_length_p10=1300.0,
        short_episode_rate=0.05,
    )
    _set_det(module, garlic_id, "integration", p10=340.0, ep_len=1300.0, short_rate=0.05)

    result = module.on_episode_end(cell=cell, stats_provider=provider, current_stage_key="WU0", num_timesteps=1000)
    assert result is False
    # solo_bootstrap セルでは integration 完了判定が実行されないため maintenance に進まない
    assert module._states[garlic_id].status == "integration"


def test_maintenance_not_triggered_by_integration_cell():
    """status=maintenance のときでも task_kind=integration のセルでは regression 更新をしない。"""
    module = make_module(initial_status={"garlic": "maintenance"})
    module._states[WeaponType.GARLIC].best_phase2_p10 = 400.0
    garlic_id = WeaponType.GARLIC
    provider = MockStatsProvider()
    # task_kind が integration のままのセル
    cell = make_cell(garlic_id, phase=2, task_kind="integration")
    provider.set_stats(cell, episode_count=30, active_score_p10=200.0)

    module.on_episode_end(cell=cell, stats_provider=provider, current_stage_key="WU0", num_timesteps=1000)
    # integration セルでは maintenance regression 更新がされないので regression_from_best は None のまま
    assert module._states[garlic_id].regression_from_best is None


# ---------------------------------------------------------------------------
# テスト: maintenance_min_probe_episodes による regression 保留
# ---------------------------------------------------------------------------

def test_maintenance_regression_none_when_below_min_probe_episodes():
    """maintenance_min_probe_episodes 未満では regression_from_best が None になる（deterministic 未設定）。"""
    module = make_module(
        initial_status={"garlic": "maintenance"},
        maintenance_min_probe_episodes=20,
    )
    module._states[WeaponType.GARLIC].best_phase2_p10 = 400.0
    garlic_id = WeaponType.GARLIC
    provider = MockStatsProvider()
    cell = make_cell(garlic_id, phase=2, task_kind="maintenance")
    # episode_count が maintenance_min_probe_episodes 未満
    provider.set_stats(cell, episode_count=10, active_score_p10=200.0)
    # deterministic 結果を設定しない → det_fresh=False → regression_from_best=None

    module.on_episode_end(cell=cell, stats_provider=provider, current_stage_key="WU0", num_timesteps=1000)
    assert module._states[garlic_id].regression_from_best is None
    assert module._states[garlic_id].regression_count == 0


def test_maintenance_regression_updated_when_above_min_probe_episodes():
    """maintenance_min_probe_episodes 以上 + deterministic 結果あり → regression_from_best が更新される。"""
    module = make_module(
        initial_status={"garlic": "maintenance"},
        maintenance_min_probe_episodes=20,
    )
    module._states[WeaponType.GARLIC].best_phase2_p10 = 400.0
    garlic_id = WeaponType.GARLIC
    provider = MockStatsProvider()
    cell = make_cell(garlic_id, phase=2, task_kind="maintenance")
    provider.set_stats(cell, episode_count=20, active_score_p10=200.0)
    # deterministic 結果設定
    _set_det(module, garlic_id, "maintenance", p10=200.0, ep_len=1300.0, short_rate=0.05)

    module.on_episode_end(cell=cell, stats_provider=provider, current_stage_key="WU0", num_timesteps=1000)
    # regression = 1 - 200/400 = 0.5
    assert module._states[garlic_id].regression_from_best == pytest.approx(0.5)
    assert module._states[garlic_id].regression_count == 1


def test_maintenance_regression_count_not_incremented_when_below_min_probe_episodes_with_det_fresh():
    """det_fresh=True でも episode_count < maintenance_min_probe_episodes なら regression_count が増えない。

    指摘2 の修正確認: deterministic 結果が先に入っても episode_count が不足していれば
    regression_from_best=None のまま（判定保留）で regression_count はインクリメントされない。
    """
    module = make_module(
        initial_status={"garlic": "maintenance"},
        maintenance_min_probe_episodes=20,
        maintenance_regression_ratio=0.35,
    )
    module._states[WeaponType.GARLIC].best_phase2_p10 = 400.0
    garlic_id = WeaponType.GARLIC
    provider = MockStatsProvider()
    cell = make_cell(garlic_id, phase=2, task_kind="maintenance")
    # episode_count が maintenance_min_probe_episodes 未満（1 や 0 を想定）
    provider.set_stats(cell, episode_count=5, active_score_p10=200.0)
    # deterministic 結果は有効（det_fresh=True になる条件を満たす）
    _set_det(module, garlic_id, "maintenance", p10=200.0, ep_len=1300.0, short_rate=0.05, num_timesteps=1000)

    module.on_episode_end(cell=cell, stats_provider=provider, current_stage_key="WU0", num_timesteps=1000)

    # episode_count < maintenance_min_probe_episodes なので判定保留
    # regression_from_best は None のまま
    assert module._states[garlic_id].regression_from_best is None
    # regression_count はインクリメントされない
    assert module._states[garlic_id].regression_count == 0


def test_maintenance_regression_from_best_cleared_when_below_min_probe_with_det_fresh():
    """det_fresh=True かつ episode_count < min_probe の場合、既存の regression_from_best がクリアされる。

    レビュー指摘（iteration 2 minor): 古い regression_from_best 値が残らないことを確認。
    """
    module = make_module(
        initial_status={"garlic": "maintenance"},
        maintenance_min_probe_episodes=20,
        maintenance_regression_ratio=0.35,
    )
    garlic_id = WeaponType.GARLIC
    module._states[garlic_id].best_phase2_p10 = 400.0
    # 既存の regression_from_best に非 None 値をセット（前回の判定結果が残っている状態）
    module._states[garlic_id].regression_from_best = 0.12

    provider = MockStatsProvider()
    cell = make_cell(garlic_id, phase=2, task_kind="maintenance")
    provider.set_stats(cell, episode_count=5, active_score_p10=350.0)  # episode_count < 20
    _set_det(module, garlic_id, "maintenance", p10=350.0, ep_len=1400.0, short_rate=0.03, num_timesteps=2000)

    module.on_episode_end(cell=cell, stats_provider=provider, current_stage_key="WU0", num_timesteps=2000)

    # episode_count < min_probe なので古い regression_from_best はクリアされる
    assert module._states[garlic_id].regression_from_best is None


# ---------------------------------------------------------------------------
# テスト: set_deterministic_result() （プラン 変更 2）
# ---------------------------------------------------------------------------

def test_set_deterministic_result_stores_all_fields():
    """set_deterministic_result() が全フィールドを正しく設定する。"""
    module = make_module(initial_status={"garlic": "solo_bootstrap"})
    garlic_id = WeaponType.GARLIC

    module.set_deterministic_result(
        weapon_id=garlic_id,
        task_kind="solo_bootstrap",
        enemy_phase_idx=2,
        p10=350.0,
        episode_length_p10=1350.0,
        short_episode_rate=0.04,
        num_timesteps=50_000,
    )

    state = module._states[garlic_id]
    assert state.deterministic_p10 == 350.0
    assert state.deterministic_episode_length_p10 == 1350.0
    assert state.deterministic_short_episode_rate == 0.04
    assert state.deterministic_eval_step == 50_000
    assert state.deterministic_task_kind == "solo_bootstrap"
    assert state.deterministic_enemy_phase_idx == 2


def test_set_deterministic_result_unknown_weapon_ignored():
    """set_deterministic_result() で未知の weapon_id は無視される（例外なし）。"""
    module = make_module()
    # 存在しない weapon_id → 例外なし
    module.set_deterministic_result(
        weapon_id=999999,
        task_kind="solo_bootstrap",
        enemy_phase_idx=2,
        p10=350.0,
        episode_length_p10=1300.0,
        short_episode_rate=0.05,
        num_timesteps=1000,
    )


# ---------------------------------------------------------------------------
# テスト: solo_bootstrap gate（プラン 変更 4 のテスト項目）
# ---------------------------------------------------------------------------

def test_solo_gate_opens_with_valid_det_and_sufficient_episodes():
    """テスト項目 6: score/ep_len/short_rate 全条件満足 + det_fresh で gate が開く。"""
    module = make_module(initial_status={"garlic": "solo_bootstrap"})
    garlic_id = WeaponType.GARLIC
    provider = MockStatsProvider()
    cell = make_cell(garlic_id, phase=2, task_kind="solo_bootstrap")
    provider.set_stats(cell, episode_count=50, active_score_p10=350.0,
                       episode_length_p10=1300.0, short_episode_rate=0.05)
    _set_det(module, garlic_id, "solo_bootstrap", p10=350.0, ep_len=1300.0, short_rate=0.05)

    module.on_episode_end(cell=cell, stats_provider=provider, current_stage_key="WU0", num_timesteps=1000)
    assert module._states[garlic_id].status == "integration"
    # gate 通過時に best_phase2_p10 が det_p10 で設定される
    assert module._states[garlic_id].best_phase2_p10 == 350.0


def test_solo_gate_does_not_open_without_det():
    """テスト項目 7: deterministic 結果なし（det_fresh=False）のとき gate が開かない。"""
    module = make_module(initial_status={"garlic": "solo_bootstrap"})
    garlic_id = WeaponType.GARLIC
    provider = MockStatsProvider()
    cell = make_cell(garlic_id, phase=2, task_kind="solo_bootstrap")
    provider.set_stats(cell, episode_count=50, active_score_p10=350.0,
                       episode_length_p10=1300.0, short_episode_rate=0.05)
    # deterministic 結果を設定しない

    module.on_episode_end(cell=cell, stats_provider=provider, current_stage_key="WU0", num_timesteps=1000)
    assert module._states[garlic_id].status == "solo_bootstrap"


def test_solo_gate_freshness_check_disabled_when_max_age_zero():
    """テスト項目 8: max_eval_age_steps=0 のとき鮮度チェックが行われない（古い結果でも通過できる）。"""
    module = make_module(
        initial_status={"garlic": "solo_bootstrap"},
        deterministic_gate_max_eval_age_steps=0,  # 無効化
    )
    garlic_id = WeaponType.GARLIC
    provider = MockStatsProvider()
    cell = make_cell(garlic_id, phase=2, task_kind="solo_bootstrap")
    provider.set_stats(cell, episode_count=50, active_score_p10=350.0,
                       episode_length_p10=1300.0, short_episode_rate=0.05)
    # 古い eval 結果（num_timesteps=0 で設定、現在の step=1_000_000）
    _set_det(module, garlic_id, "solo_bootstrap", p10=350.0, ep_len=1300.0,
             short_rate=0.05, num_timesteps=0)

    # max_eval_age_steps=0 なので鮮度チェック無効 → gate が開く
    module.on_episode_end(cell=cell, stats_provider=provider, current_stage_key="WU0", num_timesteps=1_000_000)
    assert module._states[garlic_id].status == "integration"


def test_solo_gate_freshness_check_fails_when_too_old():
    """テスト項目 8: max_eval_age_steps>0 のとき古すぎる結果では gate が開かない。"""
    module = make_module(
        initial_status={"garlic": "solo_bootstrap"},
        deterministic_gate_max_eval_age_steps=100_000,
    )
    garlic_id = WeaponType.GARLIC
    provider = MockStatsProvider()
    cell = make_cell(garlic_id, phase=2, task_kind="solo_bootstrap")
    provider.set_stats(cell, episode_count=50, active_score_p10=350.0,
                       episode_length_p10=1300.0, short_episode_rate=0.05)
    # 古すぎる eval 結果
    _set_det(module, garlic_id, "solo_bootstrap", p10=350.0, ep_len=1300.0,
             short_rate=0.05, num_timesteps=0)

    # num_timesteps=200_000 → age=200_000 > 100_000 → gate が開かない
    module.on_episode_end(cell=cell, stats_provider=provider, current_stage_key="WU0", num_timesteps=200_000)
    assert module._states[garlic_id].status == "solo_bootstrap"


def test_solo_gate_does_not_open_when_enemy_phase_mismatch():
    """テスト項目 14: deterministic_enemy_phase_idx が期待値と異なる場合に gate が開かない（P2 fix）。"""
    module = make_module(
        initial_status={"garlic": "solo_bootstrap"},
        deterministic_gate_enemy_phase_idx=2,
    )
    garlic_id = WeaponType.GARLIC
    provider = MockStatsProvider()
    cell = make_cell(garlic_id, phase=2, task_kind="solo_bootstrap")
    provider.set_stats(cell, episode_count=50, active_score_p10=350.0,
                       episode_length_p10=1300.0, short_episode_rate=0.05)
    # enemy_phase_idx=1 で結果を設定（期待値=2 とずれる）
    _set_det(module, garlic_id, "solo_bootstrap", p10=350.0, ep_len=1300.0,
             short_rate=0.05, enemy_phase_idx=1)

    module.on_episode_end(cell=cell, stats_provider=provider, current_stage_key="WU0", num_timesteps=1000)
    assert module._states[garlic_id].status == "solo_bootstrap"


def test_solo_gate_best_phase2_p10_set_to_det_p10_on_pass():
    """テスト項目 6: gate 通過時に best_phase2_p10 が deterministic 実力値に設定される。"""
    module = make_module(initial_status={"garlic": "solo_bootstrap"})
    garlic_id = WeaponType.GARLIC
    provider = MockStatsProvider()
    cell = make_cell(garlic_id, phase=2, task_kind="solo_bootstrap")
    provider.set_stats(cell, episode_count=50, active_score_p10=500.0,
                       episode_length_p10=1300.0, short_episode_rate=0.05)
    # det_p10=350.0（stochastic の 500.0 とは別値）
    _set_det(module, garlic_id, "solo_bootstrap", p10=350.0, ep_len=1300.0, short_rate=0.05)

    module.on_episode_end(cell=cell, stats_provider=provider, current_stage_key="WU0", num_timesteps=1000)
    # gate 通過時 best_phase2_p10 = det_p10 = 350.0
    assert module._states[garlic_id].best_phase2_p10 == 350.0


# ---------------------------------------------------------------------------
# テスト: integration gate（プラン 変更 5）
# ---------------------------------------------------------------------------

def test_integration_gate_does_not_open_when_det_missing():
    """integration gate: deterministic 結果なしのとき gate が開かない。"""
    module = make_module(initial_status={"garlic": "integration"})
    module._states[WeaponType.GARLIC].best_phase2_p10 = 400.0
    garlic_id = WeaponType.GARLIC
    provider = MockStatsProvider()
    cell = make_cell(garlic_id, phase=2, task_kind="integration")
    provider.set_stats(cell, episode_count=50, active_score_p10=340.0,
                       episode_length_p10=1300.0, short_episode_rate=0.05)
    # deterministic 結果なし

    module.on_episode_end(cell=cell, stats_provider=provider, current_stage_key="WU0", num_timesteps=1000)
    assert module._states[garlic_id].status == "integration"


def test_integration_gate_regression_check_from_solo():
    """テスト項目 12: regression_from_solo が integration_max_regression を超えると gate が開かない。"""
    module = make_module(initial_status={"garlic": "integration"})
    module._states[WeaponType.GARLIC].best_phase2_p10 = 400.0
    garlic_id = WeaponType.GARLIC
    provider = MockStatsProvider()
    cell = make_cell(garlic_id, phase=2, task_kind="integration")
    provider.set_stats(cell, episode_count=50, active_score_p10=200.0,
                       episode_length_p10=1300.0, short_episode_rate=0.05)
    # det_p10=200 → regression = 1-200/400=0.5 > 0.35 → gate 不通過
    _set_det(module, garlic_id, "integration", p10=200.0, ep_len=1300.0, short_rate=0.05)

    module.on_episode_end(cell=cell, stats_provider=provider, current_stage_key="WU0", num_timesteps=1000)
    assert module._states[garlic_id].status == "integration"


def test_integration_gate_best_phase2_updated_on_transition():
    """テスト項目 13: integration → maintenance 遷移時に best_phase2_p10 が max(old, det_p10) になる。"""
    module = make_module(initial_status={"garlic": "integration"})
    module._states[WeaponType.GARLIC].best_phase2_p10 = 300.0  # old value
    garlic_id = WeaponType.GARLIC
    provider = MockStatsProvider()
    cell = make_cell(garlic_id, phase=2, task_kind="integration")
    provider.set_stats(cell, episode_count=50, active_score_p10=350.0,
                       episode_length_p10=1300.0, short_episode_rate=0.05)
    # det_p10=350 → regression = 1-350/300... wait, solo_best=300, det=350 → regression negative → gate open
    # actually regression_from_solo = 1 - 350/300 = negative → <= 0.35, so gate opens
    _set_det(module, garlic_id, "integration", p10=350.0, ep_len=1300.0, short_rate=0.05)

    module.on_episode_end(cell=cell, stats_provider=provider, current_stage_key="WU0", num_timesteps=1000)
    assert module._states[garlic_id].status == "maintenance"
    # best_phase2_p10 = max(300.0, 350.0) = 350.0
    assert module._states[garlic_id].best_phase2_p10 == 350.0


def test_integration_gate_does_not_open_when_enemy_phase_mismatch():
    """テスト項目 14: integration gate - deterministic_enemy_phase_idx が期待値と異なる場合に gate が開かない（P2 fix）。"""
    module = make_module(
        initial_status={"garlic": "integration"},
        deterministic_gate_enemy_phase_idx=2,
    )
    module._states[WeaponType.GARLIC].best_phase2_p10 = 400.0
    garlic_id = WeaponType.GARLIC
    provider = MockStatsProvider()
    cell = make_cell(garlic_id, phase=2, task_kind="integration")
    provider.set_stats(cell, episode_count=50, active_score_p10=340.0,
                       episode_length_p10=1300.0, short_episode_rate=0.05)
    # enemy_phase_idx=0 で結果設定（期待値=2 とずれる）
    _set_det(module, garlic_id, "integration", p10=340.0, ep_len=1300.0,
             short_rate=0.05, enemy_phase_idx=0)

    module.on_episode_end(cell=cell, stats_provider=provider, current_stage_key="WU0", num_timesteps=1000)
    assert module._states[garlic_id].status == "integration"


# ---------------------------------------------------------------------------
# テスト: maintenance（プラン 変更 6）
# ---------------------------------------------------------------------------

def test_maintenance_det_fresh_true_regression_over_threshold():
    """テスト項目 15: det_fresh=True かつ regression が閾値超のとき regression_count がインクリメントされる。"""
    module = make_module(initial_status={"garlic": "maintenance"})
    module._states[WeaponType.GARLIC].best_phase2_p10 = 400.0
    garlic_id = WeaponType.GARLIC
    provider = MockStatsProvider()
    cell = make_cell(garlic_id, phase=2, task_kind="maintenance")
    provider.set_stats(cell, episode_count=30, active_score_p10=200.0)
    _set_det(module, garlic_id, "maintenance", p10=200.0, ep_len=1300.0, short_rate=0.05, num_timesteps=1000)

    module.on_episode_end(cell=cell, stats_provider=provider, current_stage_key="WU0", num_timesteps=1000)
    assert module._states[garlic_id].regression_from_best == pytest.approx(0.5)
    assert module._states[garlic_id].regression_count == 1


def test_maintenance_det_fresh_false_regression_is_none():
    """テスト項目 16: det_fresh=False のとき regression_from_best が None になる。"""
    module = make_module(initial_status={"garlic": "maintenance"})
    module._states[WeaponType.GARLIC].best_phase2_p10 = 400.0
    garlic_id = WeaponType.GARLIC
    provider = MockStatsProvider()
    cell = make_cell(garlic_id, phase=2, task_kind="maintenance")
    provider.set_stats(cell, episode_count=30, active_score_p10=200.0)
    # deterministic 結果なし → det_fresh=False

    module.on_episode_end(cell=cell, stats_provider=provider, current_stage_key="WU0", num_timesteps=1000)
    assert module._states[garlic_id].regression_from_best is None


def test_maintenance_best_phase2_zero_no_regression():
    """テスト項目 17: best_phase2_p10=0 のとき regression が計算されない。"""
    module = make_module(initial_status={"garlic": "maintenance"})
    assert module._states[WeaponType.GARLIC].best_phase2_p10 == 0.0
    garlic_id = WeaponType.GARLIC
    provider = MockStatsProvider()
    cell = make_cell(garlic_id, phase=2, task_kind="maintenance")
    provider.set_stats(cell, episode_count=30, active_score_p10=200.0)
    _set_det(module, garlic_id, "maintenance", p10=200.0, ep_len=1300.0, short_rate=0.05)

    module.on_episode_end(cell=cell, stats_provider=provider, current_stage_key="WU0", num_timesteps=1000)
    # best_phase2_p10=0 なので regression は計算されない
    assert module._states[garlic_id].regression_from_best is None


def test_maintenance_same_det_step_increments_regression_count_once():
    """テスト項目 18: 同一 det_step では regression_count が 1 回しかインクリメントされない（P2 fix: 過剰カウント防止）。"""
    module = make_module(initial_status={"garlic": "maintenance"})
    module._states[WeaponType.GARLIC].best_phase2_p10 = 400.0
    garlic_id = WeaponType.GARLIC
    provider = MockStatsProvider()
    cell = make_cell(garlic_id, phase=2, task_kind="maintenance")
    provider.set_stats(cell, episode_count=30, active_score_p10=200.0)
    # 同じ det_step=5000 で 3 回 on_episode_end を呼ぶ
    _set_det(module, garlic_id, "maintenance", p10=200.0, ep_len=1300.0, short_rate=0.05, num_timesteps=5000)

    module.on_episode_end(cell=cell, stats_provider=provider, current_stage_key="WU0", num_timesteps=5000)
    module.on_episode_end(cell=cell, stats_provider=provider, current_stage_key="WU0", num_timesteps=5000)
    module.on_episode_end(cell=cell, stats_provider=provider, current_stage_key="WU0", num_timesteps=5000)

    # 同一 det_step での呼び出しは 1 回しかカウントされない
    assert module._states[garlic_id].regression_count == 1


def test_maintenance_different_det_steps_increments_multiple():
    """テスト項目 18 の逆: 異なる det_step なら複数回インクリメントされる。"""
    module = make_module(initial_status={"garlic": "maintenance"})
    module._states[WeaponType.GARLIC].best_phase2_p10 = 400.0
    garlic_id = WeaponType.GARLIC
    provider = MockStatsProvider()
    cell = make_cell(garlic_id, phase=2, task_kind="maintenance")
    provider.set_stats(cell, episode_count=30, active_score_p10=200.0)

    # det_step=1000 で 1 回
    _set_det(module, garlic_id, "maintenance", p10=200.0, ep_len=1300.0, short_rate=0.05, num_timesteps=1000)
    module.on_episode_end(cell=cell, stats_provider=provider, current_stage_key="WU0", num_timesteps=1000)
    assert module._states[garlic_id].regression_count == 1

    # det_step=2000 で 1 回（異なる step）
    _set_det(module, garlic_id, "maintenance", p10=200.0, ep_len=1300.0, short_rate=0.05, num_timesteps=2000)
    module.on_episode_end(cell=cell, stats_provider=provider, current_stage_key="WU0", num_timesteps=2000)
    assert module._states[garlic_id].regression_count == 2


def test_maintenance_enemy_phase_mismatch_skips_regression():
    """テスト項目 19: deterministic_enemy_phase_idx が期待値と異なる場合に regression 判定がスキップされる（P2 fix）。"""
    module = make_module(
        initial_status={"garlic": "maintenance"},
        deterministic_gate_enemy_phase_idx=2,
    )
    module._states[WeaponType.GARLIC].best_phase2_p10 = 400.0
    garlic_id = WeaponType.GARLIC
    provider = MockStatsProvider()
    cell = make_cell(garlic_id, phase=2, task_kind="maintenance")
    provider.set_stats(cell, episode_count=30, active_score_p10=200.0)
    # enemy_phase_idx=1 で結果設定（期待値=2 とずれる）
    _set_det(module, garlic_id, "maintenance", p10=200.0, ep_len=1300.0,
             short_rate=0.05, enemy_phase_idx=1)

    module.on_episode_end(cell=cell, stats_provider=provider, current_stage_key="WU0", num_timesteps=1000)
    # phase ずれなので regression 判定スキップ → regression_from_best は None
    assert module._states[garlic_id].regression_from_best is None
    assert module._states[garlic_id].regression_count == 0


# ---------------------------------------------------------------------------
# テスト: export_state / import_state（プラン 変更 8）
# ---------------------------------------------------------------------------

def test_export_import_state_roundtrip_all_deterministic_fields():
    """テスト項目 20: export_state() / import_state() で全 7 フィールドが往復保持される。"""
    module = make_module(initial_status={"garlic": "maintenance"})
    garlic_id = WeaponType.GARLIC
    state = module._states[garlic_id]
    state.deterministic_p10 = 350.0
    state.deterministic_episode_length_p10 = 1300.0
    state.deterministic_short_episode_rate = 0.05
    state.deterministic_eval_step = 50_000
    state.deterministic_task_kind = "maintenance"
    state.deterministic_enemy_phase_idx = 2
    state.last_regression_eval_step = 49_000

    exported = module.export_state()
    module2 = make_module()
    module2.import_state(exported)

    s2 = module2._states[garlic_id]
    assert s2.deterministic_p10 == 350.0
    assert s2.deterministic_episode_length_p10 == 1300.0
    assert s2.deterministic_short_episode_rate == 0.05
    assert s2.deterministic_eval_step == 50_000
    assert s2.deterministic_task_kind == "maintenance"
    assert s2.deterministic_enemy_phase_idx == 2
    assert s2.last_regression_eval_step == 49_000


def test_import_state_old_checkpoint_no_key_error():
    """テスト項目 21: 旧 checkpoint（deterministic_p10 キーなし等）を import_state() しても KeyError が出ない。"""
    module = make_module()
    # 旧形式の state（新フィールドなし）
    old_state = {
        "weapons": [
            {
                "weapon_id": WeaponType.GARLIC,
                "weapon_key": "garlic",
                "status": "maintenance",
                "best_phase2_p10": 400.0,
                "regression_from_best": 0.1,
                "regression_count": 2,
                "last_probe_step": 10000,
                # deterministic_p10, last_regression_eval_step などは含まれていない
            }
        ]
    }
    # KeyError が出ないことを確認
    module.import_state(old_state)
    s = module._states[WeaponType.GARLIC]
    assert s.status == "maintenance"
    assert s.deterministic_p10 is None
    assert s.deterministic_episode_length_p10 is None
    assert s.deterministic_short_episode_rate is None
    assert s.deterministic_eval_step == 0
    assert s.deterministic_task_kind is None
    assert s.deterministic_enemy_phase_idx is None
    assert s.last_regression_eval_step == -1


# ---------------------------------------------------------------------------
# テスト: get_eval_build_policy()（プラン 変更 2）
# ---------------------------------------------------------------------------

def test_get_eval_build_policy_solo_bootstrap():
    """get_eval_build_policy() - solo_bootstrap は target_only を返す。"""
    module = make_module()
    assert module.get_eval_build_policy("solo_bootstrap", current_stage_key="WU0") == "target_only"


def test_get_eval_build_policy_maintenance():
    """get_eval_build_policy() - maintenance は target_plus_anchor_if_unlocked を返す。"""
    module = make_module()
    assert module.get_eval_build_policy("maintenance", current_stage_key="WU0") == "target_plus_anchor_if_unlocked"


def test_get_eval_build_policy_integration_garlic_not_maintenance():
    """get_eval_build_policy() - integration 中 garlic が maintenance でない → target_only。"""
    module = make_module(initial_status={"garlic": "solo_bootstrap"})
    # garlic は solo_bootstrap なので target_only
    policy = module.get_eval_build_policy("integration", current_stage_key="WU1")
    assert policy == "target_only"


def test_get_eval_build_policy_integration_garlic_maintenance_unlocked():
    """get_eval_build_policy() - integration 中 garlic が maintenance かつアンロック済み → target_plus_anchor_if_unlocked。"""
    module = make_module(initial_status={"garlic": "maintenance"})
    # WU1 以上では garlic がアンロック済みになっている（unlock_stage_key="WU0"）
    policy = module.get_eval_build_policy("integration", current_stage_key="WU1")
    assert policy == "target_plus_anchor_if_unlocked"


def test_get_eval_build_policy_no_stage_key():
    """get_eval_build_policy() - current_stage_key=None の integration は target_only を返す。"""
    module = make_module(initial_status={"garlic": "maintenance"})
    # stage_key がない場合は integration の条件分岐がスキップされる（フォールバック）
    policy = module.get_eval_build_policy("integration", current_stage_key=None)
    assert policy == "target_only"


def test_get_eval_build_policy_unknown_task_kind():
    """get_eval_build_policy() - 未知の task_kind は target_only を返す（デフォルト）。"""
    module = make_module()
    assert module.get_eval_build_policy("unknown_task", current_stage_key="WU0") == "target_only"


def test_is_complete_through_stage_requires_all_unlocked_weapons_maintenance():
    module = WeaponBootstrapStateModule(
        weapon_unlock_order=WEAPON_UNLOCK_ORDER,
        initial_status={
            "garlic": "maintenance",
            "king_bible": "maintenance",
            "magic_wand": "integration",
        },
    )

    assert module.is_complete_through_stage("WU1") is True
    assert module.is_complete_through_stage("WU2") is False


def test_get_completion_snapshot_contains_unfinished_weapons():
    module = WeaponBootstrapStateModule(
        weapon_unlock_order=WEAPON_UNLOCK_ORDER,
        initial_status={
            "garlic": "maintenance",
            "king_bible": "integration",
        },
    )

    snapshot = module.get_completion_snapshot("WU1")

    assert snapshot["target_stage_key"] == "WU1"
    assert snapshot["complete"] is False
    assert snapshot["unfinished_weapons"] == [
        {
            "weapon_key": "king_bible",
            "weapon_id": WeaponType.KING_BIBLE,
            "status": "integration",
            "gate_goal": "score",
        }
    ]


def test_get_regressed_weapons_filters_by_regression_count():
    module = WeaponBootstrapStateModule(
        weapon_unlock_order=WEAPON_UNLOCK_ORDER,
        initial_status={"garlic": "maintenance", "king_bible": "maintenance"},
    )
    module._states[WeaponType.GARLIC].regression_count = 1
    module._states[WeaponType.KING_BIBLE].regression_count = 3

    regressed = module.get_regressed_weapons(max_regression_count=2)

    assert regressed == [{
        "weapon_key": "king_bible",
        "weapon_id": WeaponType.KING_BIBLE,
        "status": "maintenance",
        "regression_count": 3,
        "regression_from_best": None,
    }]


# ---------------------------------------------------------------------------
# テスト: item stage scope（IS0/IS1 の bootstrap state 分離）
# ---------------------------------------------------------------------------

def test_import_state_ignores_mismatched_item_stage():
    module = WeaponBootstrapStateModule(
        weapon_unlock_order=WEAPON_UNLOCK_ORDER,
        item_stage_key="IS1",
        initial_status={"garlic": "solo_bootstrap"},
    )
    module.import_state({
        "item_stage_key": "IS0",
        "weapons": [
            {
                "weapon_id": WeaponType.GARLIC,
                "weapon_key": "garlic",
                "status": "maintenance",
                "best_phase2_p10": 999.0,
            }
        ],
    })
    snapshot = module.get_weapon_snapshot(WeaponType.GARLIC)
    assert snapshot["status"] == "solo_bootstrap"
    assert snapshot["best_phase2_p10"] == 0.0


def test_import_state_returns_false_on_item_stage_mismatch():
    """item_stage mismatch では import_state が False を返し、initial_status が残る。

    False を受けた呼び出し側はコンストラクタで適用済みの initial_status を
    フォールバックとして使え、全 locked による候補セル空を防げる。
    """
    module = WeaponBootstrapStateModule(
        weapon_unlock_order=WEAPON_UNLOCK_ORDER,
        item_stage_key="IS1",
        initial_status={"garlic": "solo_bootstrap"},
    )
    applied = module.import_state({
        "item_stage_key": "IS0",
        "weapons": [
            {
                "weapon_id": WeaponType.GARLIC,
                "weapon_key": "garlic",
                "status": "maintenance",
                "best_phase2_p10": 999.0,
            }
        ],
    })
    assert applied is False
    # initial_status がフォールバックとして保持され、全 locked にならない
    assert module.get_weapon_snapshot(WeaponType.GARLIC)["status"] == "solo_bootstrap"


def test_import_state_returns_true_on_matching_item_stage():
    """item_stage 一致では import_state が True を返す。"""
    module = WeaponBootstrapStateModule(
        weapon_unlock_order=WEAPON_UNLOCK_ORDER,
        item_stage_key="IS1",
        initial_status={"garlic": "solo_bootstrap"},
    )
    applied = module.import_state({
        "item_stage_key": "IS1",
        "weapons": [
            {
                "weapon_id": WeaponType.GARLIC,
                "weapon_key": "garlic",
                "status": "maintenance",
                "best_phase2_p10": 999.0,
            }
        ],
    })
    assert applied is True
    assert module.get_weapon_snapshot(WeaponType.GARLIC)["status"] == "maintenance"


def test_export_state_includes_item_stage_key():
    module = WeaponBootstrapStateModule(
        weapon_unlock_order=WEAPON_UNLOCK_ORDER,
        item_stage_key="IS1",
    )
    assert module.export_state()["item_stage_key"] == "IS1"


def test_import_state_accepts_matching_item_stage():
    module = WeaponBootstrapStateModule(
        weapon_unlock_order=WEAPON_UNLOCK_ORDER,
        item_stage_key="IS1",
        initial_status={"garlic": "solo_bootstrap"},
    )
    module.import_state({
        "item_stage_key": "IS1",
        "weapons": [
            {
                "weapon_id": WeaponType.GARLIC,
                "weapon_key": "garlic",
                "status": "maintenance",
                "best_phase2_p10": 999.0,
            }
        ],
    })
    snapshot = module.get_weapon_snapshot(WeaponType.GARLIC)
    assert snapshot["status"] == "maintenance"
    assert snapshot["best_phase2_p10"] == 999.0


def test_import_state_default_item_stage_is_is0():
    """item_stage_key を指定しないモジュールは IS0 とみなし、旧 state を受け入れる。"""
    module = WeaponBootstrapStateModule(
        weapon_unlock_order=WEAPON_UNLOCK_ORDER,
        initial_status={"garlic": "solo_bootstrap"},
    )
    # item_stage_key キーなしの旧 state（デフォルト IS0 とみなす）
    module.import_state({
        "weapons": [
            {
                "weapon_id": WeaponType.GARLIC,
                "weapon_key": "garlic",
                "status": "maintenance",
                "best_phase2_p10": 500.0,
            }
        ],
    })
    snapshot = module.get_weapon_snapshot(WeaponType.GARLIC)
    assert snapshot["status"] == "maintenance"
    assert snapshot["best_phase2_p10"] == 500.0


# ---------------------------------------------------------------------------
# テスト: trait exposure gate（Peachone / Ebony Wings 武器別 gate policy）
# ---------------------------------------------------------------------------

def test_trait_bootstrap_weapon_can_pass_solo_gate_with_low_score_if_survival_is_stable():
    """trait exposure gate: 低スコアでも生存品質が高ければ solo_bootstrap → integration に昇格する。"""
    module = make_module(
        initial_status={"peachone": "solo_bootstrap"},
        trait_bootstrap_weapon_keys=("peachone", "ebony_wings"),
        trait_bootstrap_target_p10=120.0,
        trait_bootstrap_min_ep_len_p10=3000.0,
        trait_bootstrap_max_short_episode_rate=0.05,
    )
    weapon_id = WeaponType.PEACHONE
    provider = MockStatsProvider()
    cell = make_cell(weapon_id, phase=2, task_kind="solo_bootstrap")
    provider.set_stats(
        cell,
        episode_count=50,
        active_score_p10=125.0,
        episode_length_p10=4300.0,
        short_episode_rate=0.0,
    )
    _set_det(
        module,
        weapon_id,
        "solo_bootstrap",
        p10=125.0,
        ep_len=4300.0,
        short_rate=0.0,
    )

    changed = module.on_episode_end(
        cell=cell,
        stats_provider=provider,
        current_stage_key="WU11",
        num_timesteps=10_000,
    )

    assert changed is True
    assert module._states[weapon_id].status == "integration"
    assert module._states[weapon_id].best_phase2_p10 == 125.0


def test_score_bootstrap_weapon_uses_default_threshold():
    """score gate 武器: trait gate の低閾値が適用されず、デフォルト閾値（300.0）で判定される。"""
    module = make_module(
        initial_status={"garlic": "solo_bootstrap"},
        trait_bootstrap_weapon_keys=("peachone", "ebony_wings"),
        trait_bootstrap_target_p10=120.0,
        trait_bootstrap_min_ep_len_p10=3000.0,
        trait_bootstrap_max_short_episode_rate=0.05,
    )
    weapon_id = WeaponType.GARLIC
    provider = MockStatsProvider()
    cell = make_cell(weapon_id, phase=2, task_kind="solo_bootstrap")
    provider.set_stats(
        cell,
        episode_count=50,
        active_score_p10=125.0,   # 300.0 未満なので score gate では開かない
        episode_length_p10=4300.0,
        short_episode_rate=0.0,
    )
    _set_det(
        module,
        weapon_id,
        "solo_bootstrap",
        p10=125.0,
        ep_len=4300.0,
        short_rate=0.0,
    )

    changed = module.on_episode_end(
        cell=cell,
        stats_provider=provider,
        current_stage_key="WU0",
        num_timesteps=10_000,
    )

    assert changed is False
    assert module._states[weapon_id].status == "solo_bootstrap"


def test_completion_snapshot_shows_trait_exposure_gate_goal():
    """get_completion_snapshot が trait_exposure 武器の gate_goal を正しく返す。"""
    module = make_module(
        initial_status={"peachone": "solo_bootstrap"},
        trait_bootstrap_weapon_keys=("peachone", "ebony_wings"),
        trait_bootstrap_target_p10=120.0,
        trait_bootstrap_min_ep_len_p10=3000.0,
        trait_bootstrap_max_short_episode_rate=0.05,
    )

    snapshot = module.get_completion_snapshot("WU11")
    unfinished = {w["weapon_key"]: w for w in snapshot["unfinished_weapons"]}

    assert "peachone" in unfinished
    assert unfinished["peachone"]["gate_goal"] == "trait_exposure"
