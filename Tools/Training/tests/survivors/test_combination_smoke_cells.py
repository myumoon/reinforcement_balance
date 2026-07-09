"""tests/survivors/test_combination_smoke_cells.py

combination_smoke セル生成のテスト。
"""
from __future__ import annotations
import sys
from pathlib import Path

_TRAINING_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAINING_ROOT))

from games.survivors.combination_smoke_cells import build_combination_smoke_cells
from games.survivors.survivors_weapon_table import WEAPON_UNLOCK_ORDER, get_unlocked_startable_weapon_ids


def test_combination_smoke_cells_cover_all_weapons():
    cells = build_combination_smoke_cells(
        stage_key="WU12",
        enemy_phase_idx=2,
        max_cells=64,
        seed=123,
        weapon_unlock_order=WEAPON_UNLOCK_ORDER,
    )
    assert len(cells) > 0
    covered = set()
    for cell in cells:
        covered.update(cell.allowed_weapon_ids)
    expected = set(get_unlocked_startable_weapon_ids("WU12", WEAPON_UNLOCK_ORDER))
    assert expected <= covered


def test_combination_smoke_cells_are_deterministic():
    a = build_combination_smoke_cells("WU12", 2, 16, 99, WEAPON_UNLOCK_ORDER)
    b = build_combination_smoke_cells("WU12", 2, 16, 99, WEAPON_UNLOCK_ORDER)
    assert [cell.key() for cell in a] == [cell.key() for cell in b]


def test_combination_smoke_cell_has_slot_params():
    cells = build_combination_smoke_cells("WU3", 2, 8, 1, WEAPON_UNLOCK_ORDER)
    assert len(cells) > 0
    cell = cells[0]
    assert cell.task_kind == "combination_smoke"
    assert cell.build_policy == "combination_slots"
    assert len(cell.initial_weapon_slots) >= 2
    assert set(wid for wid, _ in cell.initial_weapon_slots) <= set(cell.allowed_weapon_ids)
