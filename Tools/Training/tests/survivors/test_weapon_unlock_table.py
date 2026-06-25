"""tests/survivors/test_weapon_unlock_table.py"""
from __future__ import annotations
import sys
from pathlib import Path

_TRAINING_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAINING_ROOT))

import pytest
from games.survivors.survivors_weapon_table import (
    WEAPON_UNLOCK_ORDER, ITEM_SYSTEM_STAGES,
    get_unlocked_weapon_ids, get_unlocked_startable_weapon_ids, get_added_weapon_id,
    build_weapon_params_for_cell,
)
from games.survivors.survivors_weapon_curriculum import WeaponType


def test_weapon_ids_unique():
    ids = [e.weapon_id for e in WEAPON_UNLOCK_ORDER]
    assert len(ids) == len(set(ids)), "weapon_id が重複しています"


def test_unlock_order_sequential():
    for i, e in enumerate(WEAPON_UNLOCK_ORDER):
        assert e.unlock_order == i


def test_wu0_startable():
    ids = get_unlocked_startable_weapon_ids("WU0")
    assert ids == [WeaponType.GARLIC]


def test_wu1_contains_king_bible():
    ids = get_unlocked_startable_weapon_ids("WU1")
    assert WeaponType.KING_BIBLE in ids
    assert WeaponType.GARLIC in ids


def test_get_added_weapon_id():
    assert get_added_weapon_id("WU0") == WeaponType.GARLIC
    assert get_added_weapon_id("WU1") == WeaponType.KING_BIBLE


def test_build_weapon_params_garlic():
    params = build_weapon_params_for_cell(
        first_weapon_id=WeaponType.GARLIC,
        max_unlocked_stage_key="WU0",
        item_stage_key="IS0",
        pool_policy="target_plus_anchor",
    )
    assert params["allowed_weapon_types"] == [WeaponType.GARLIC]
    assert params["initial_weapon_slots"] == [{"weapon_id": WeaponType.GARLIC, "level": 1}]
    assert params["enable_passives"] is False


def test_build_weapon_params_king_bible():
    params = build_weapon_params_for_cell(
        first_weapon_id=WeaponType.KING_BIBLE,
        max_unlocked_stage_key="WU1",
        item_stage_key="IS0",
        pool_policy="target_plus_anchor",
    )
    assert WeaponType.KING_BIBLE in params["allowed_weapon_types"]
    assert WeaponType.GARLIC in params["allowed_weapon_types"]
    assert params["initial_weapon_slots"] == [{"weapon_id": WeaponType.KING_BIBLE, "level": 1}]


def test_item_system_stages():
    assert ITEM_SYSTEM_STAGES["IS0"].enable_passives is False
    assert ITEM_SYSTEM_STAGES["IS0"].enable_evolutions is False
    assert ITEM_SYSTEM_STAGES["IS1"].enable_passives is True
