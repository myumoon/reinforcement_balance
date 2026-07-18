from __future__ import annotations

import sys
from pathlib import Path

_TRAINING_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAINING_ROOT))

from games.survivors.evolution_stage_cells import build_evolution_stage_cells
from games.survivors.survivors_vs_spec import EVOLUTION_TABLE
from games.survivors.survivors_weapon_curriculum import WeaponType
from games.survivors.survivors_weapon_table import WEAPON_UNLOCK_ORDER, get_unlocked_startable_weapon_ids


def test_evolution_stage_cells_cover_evolution_table_base_weapons():
    cells = build_evolution_stage_cells("WU12", 2, 64, 123, WEAPON_UNLOCK_ORDER)
    covered_base = {cell.first_weapon_id for cell in cells}
    unlocked = set(get_unlocked_startable_weapon_ids("WU12", WEAPON_UNLOCK_ORDER))
    expected_base = {row["base"] for row in EVOLUTION_TABLE if row["passive"] != 0 and row["base"] in unlocked}

    assert expected_base <= covered_base


def test_evolution_stage_cells_have_required_passive_slots():
    cells = build_evolution_stage_cells("WU12", 2, 64, 123, WEAPON_UNLOCK_ORDER)

    assert cells
    for cell in cells:
        if cell.combo_key == "is2_union_w13_w14_to_vandalier":
            continue
        assert cell.initial_passive_slots
        assert cell.combo_key.startswith("is2_")
        assert cell.task_kind == "evolution_stage"


def test_evolution_stage_cells_include_vandalier_union():
    cells = build_evolution_stage_cells("WU12", 2, 64, 123, WEAPON_UNLOCK_ORDER)
    union = next(
        cell for cell in cells
        if cell.combo_key == "is2_union_w13_w14_to_vandalier"
    )

    assert union.task_kind == "evolution_stage"
    assert union.build_policy == "combination_slots"
    assert tuple(wid for wid, _ in union.initial_weapon_slots) == (
        WeaponType.PEACHONE,
        WeaponType.EBONY_WINGS,
    )
    assert union.initial_passive_slots == ()
    assert union.allowed_weapon_ids == (WeaponType.PEACHONE, WeaponType.EBONY_WINGS)
