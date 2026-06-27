"""tests/survivors/test_weapon_unlock_table.py"""
from __future__ import annotations
import sys
from pathlib import Path

_TRAINING_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAINING_ROOT))

import pytest
from games.survivors.survivors_weapon_table import (
    WEAPON_UNLOCK_ORDER, WEAPON_UNLOCK_ORDER_KINGBIBLE_FIRST_V1,
    WEAPON_UNLOCK_TABLES, ITEM_SYSTEM_STAGES,
    get_unlocked_weapon_ids, get_unlocked_startable_weapon_ids, get_added_weapon_id,
    build_weapon_params_for_cell, resolve_weapon_unlock_order,
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


# ============================================================
# kingbible_first_v1 テーブルのテスト
# ============================================================

def test_kingbible_first_v1_wu0_startable_only_king_bible():
    """WU0 で startable = KING_BIBLE のみ。"""
    order = resolve_weapon_unlock_order("kingbible_first_v1")
    ids = get_unlocked_startable_weapon_ids("WU0", order)
    assert ids == [WeaponType.KING_BIBLE]


def test_kingbible_first_v1_wu1_startable_has_both():
    """WU1 で startable = [KING_BIBLE, GARLIC]（両方含む）。"""
    order = resolve_weapon_unlock_order("kingbible_first_v1")
    ids = get_unlocked_startable_weapon_ids("WU1", order)
    assert WeaponType.KING_BIBLE in ids
    assert WeaponType.GARLIC in ids


def test_kingbible_first_v1_get_added_weapon_wu0():
    """WU0 の追加武器 = KING_BIBLE。"""
    order = resolve_weapon_unlock_order("kingbible_first_v1")
    assert get_added_weapon_id("WU0", order) == WeaponType.KING_BIBLE


def test_kingbible_first_v1_get_added_weapon_wu1():
    """WU1 の追加武器 = GARLIC。"""
    order = resolve_weapon_unlock_order("kingbible_first_v1")
    assert get_added_weapon_id("WU1", order) == WeaponType.GARLIC


# ============================================================
# target_plus_anchor_if_unlocked のテスト
# ============================================================

def test_target_plus_anchor_if_unlocked_wu0_no_garlic():
    """WU0/KING_BIBLE では garlic が allowed に含まれない。"""
    order = resolve_weapon_unlock_order("kingbible_first_v1")
    params = build_weapon_params_for_cell(
        first_weapon_id=WeaponType.KING_BIBLE,
        max_unlocked_stage_key="WU0",
        item_stage_key="IS0",
        pool_policy="target_plus_anchor_if_unlocked",
        weapon_unlock_order=order,
    )
    assert WeaponType.GARLIC not in params["allowed_weapon_types"]
    assert params["allowed_weapon_types"] == [WeaponType.KING_BIBLE]
    assert params["initial_weapon_slots"] == [{"weapon_id": WeaponType.KING_BIBLE, "level": 1}]


def test_target_plus_anchor_if_unlocked_wu1_king_bible_has_garlic():
    """WU1/KING_BIBLE では garlic がアンロック済みなので allowed に含まれる。"""
    order = resolve_weapon_unlock_order("kingbible_first_v1")
    params = build_weapon_params_for_cell(
        first_weapon_id=WeaponType.KING_BIBLE,
        max_unlocked_stage_key="WU1",
        item_stage_key="IS0",
        pool_policy="target_plus_anchor_if_unlocked",
        weapon_unlock_order=order,
    )
    assert WeaponType.GARLIC in params["allowed_weapon_types"]
    assert WeaponType.KING_BIBLE in params["allowed_weapon_types"]
    assert params["initial_weapon_slots"] == [{"weapon_id": WeaponType.KING_BIBLE, "level": 1}]


def test_target_plus_anchor_if_unlocked_wu1_garlic_no_duplicate():
    """WU1/GARLIC では target==anchor なので allowed = [GARLIC] のみ。"""
    order = resolve_weapon_unlock_order("kingbible_first_v1")
    params = build_weapon_params_for_cell(
        first_weapon_id=WeaponType.GARLIC,
        max_unlocked_stage_key="WU1",
        item_stage_key="IS0",
        pool_policy="target_plus_anchor_if_unlocked",
        weapon_unlock_order=order,
    )
    assert params["allowed_weapon_types"] == [WeaponType.GARLIC]
    assert params["initial_weapon_slots"] == [{"weapon_id": WeaponType.GARLIC, "level": 1}]


def test_resolve_weapon_unlock_order_unknown():
    """不明なテーブル名で ValueError。"""
    with pytest.raises(ValueError):
        resolve_weapon_unlock_order("unknown_table")


def test_default_v1_backward_compat():
    """default_v1 は既存の WEAPON_UNLOCK_ORDER と同一。"""
    assert resolve_weapon_unlock_order("default_v1") is WEAPON_UNLOCK_ORDER
