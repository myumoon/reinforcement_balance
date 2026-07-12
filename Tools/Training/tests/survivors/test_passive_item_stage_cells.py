"""tests/survivors/test_passive_item_stage_cells.py

IS1 passive coverage セル生成のユニットテスト。
全 startable weapon × 全 passive がいずれかのセルでカバーされること、
および決定論性を検証する。
"""
from __future__ import annotations
import sys
from pathlib import Path

_TRAINING_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAINING_ROOT))

from games.survivors.passive_item_stage_cells import build_passive_item_stage_cells
from games.survivors.survivors_vs_spec import PASSIVE_VALID_FOR_RSI
from games.survivors.survivors_weapon_table import WEAPON_UNLOCK_ORDER, get_unlocked_startable_weapon_ids


def test_passive_item_stage_cells_cover_weapons_and_passives():
    cells = build_passive_item_stage_cells("WU12", 2, 96, 123, WEAPON_UNLOCK_ORDER)
    covered_weapons = set()
    covered_passives = set()
    for cell in cells:
        covered_weapons.update(cell.allowed_weapon_ids)
        covered_passives.update(pid for pid, _ in cell.initial_passive_slots)
    assert set(get_unlocked_startable_weapon_ids("WU12", WEAPON_UNLOCK_ORDER)) <= covered_weapons
    assert set(PASSIVE_VALID_FOR_RSI) <= covered_passives


def test_passive_item_stage_cells_are_deterministic():
    a = build_passive_item_stage_cells("WU12", 2, 32, 99, WEAPON_UNLOCK_ORDER)
    b = build_passive_item_stage_cells("WU12", 2, 32, 99, WEAPON_UNLOCK_ORDER)
    assert [cell.key() for cell in a] == [cell.key() for cell in b]


def test_passive_item_stage_cells_metadata():
    """各セルが passive_item_stage task_kind / combination_slots build_policy /
    is1_ prefix combo_key / 非空 passive slots を持つ。"""
    cells = build_passive_item_stage_cells("WU12", 2, 96, 123, WEAPON_UNLOCK_ORDER)
    assert cells, "セルが 1 つ以上生成されること"
    for cell in cells:
        assert cell.task_kind == "passive_item_stage"
        assert cell.build_policy == "combination_slots"
        assert cell.combo_key.startswith("is1_")
        assert cell.initial_passive_slots, "passive slot が非空であること"
        assert cell.enemy_phase_idx == 2
        assert cell.weapon_unlock_stage_key == "WU12"


def test_passive_item_stage_cells_respect_max_cells():
    cells = build_passive_item_stage_cells("WU12", 2, 5, 7, WEAPON_UNLOCK_ORDER)
    assert len(cells) <= 5


def test_passive_item_stage_cells_passive_levels_in_range():
    """各 passive slot のレベルが 1..max_level の範囲内であること。"""
    from games.survivors.survivors_vs_spec import PASSIVE_MAX_LEVEL
    cells = build_passive_item_stage_cells("WU12", 2, 96, 123, WEAPON_UNLOCK_ORDER)
    for cell in cells:
        for pid, level in cell.initial_passive_slots:
            assert 1 <= level <= int(PASSIVE_MAX_LEVEL[pid])
