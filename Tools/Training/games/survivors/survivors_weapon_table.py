"""Survivors 武器アンロックテーブル（タスクセルサンプラー用）。

WeaponEntry, ItemSystemStage, WeaponUnlockTable v1 を定義する。
WEAPON_PHASES（legacy curriculum用）とは別軸で管理する。
"""
from __future__ import annotations
from dataclasses import dataclass
from games.survivors.survivors_weapon_curriculum import WeaponType


@dataclass(frozen=True)
class WeaponEntry:
    weapon_id: int
    key: str
    display_name: str
    category: str
    startable: bool
    unlock_stage_key: str
    unlock_order: int
    tags: tuple[str, ...]
    notes: str = ""


@dataclass(frozen=True)
class ItemSystemStage:
    key: str
    enable_passives: bool
    enable_evolutions: bool


# WeaponUnlockTable v1: 1段階1武器、WU0〜WU12
WEAPON_UNLOCK_ORDER: list[WeaponEntry] = [
    WeaponEntry(weapon_id=WeaponType.GARLIC,         key="garlic",         display_name="GARLIC",         category="melee_aoe",   startable=True, unlock_stage_key="WU0",  unlock_order=0,  tags=("melee", "aoe"),        notes="近距離AoE。ベースライン。"),
    WeaponEntry(weapon_id=WeaponType.KING_BIBLE,     key="king_bible",     display_name="KING_BIBLE",     category="melee_orbit", startable=True, unlock_stage_key="WU1",  unlock_order=1,  tags=("melee", "orbit"),      notes="GARLICに最も近い範囲武器。"),
    WeaponEntry(weapon_id=WeaponType.MAGIC_WAND,     key="magic_wand",     display_name="MAGIC_WAND",     category="projectile",  startable=True, unlock_stage_key="WU2",  unlock_order=2,  tags=("projectile", "target"),notes="近い敵を狙う自動投射物。"),
    WeaponEntry(weapon_id=WeaponType.FIRE_WAND,      key="fire_wand",      display_name="FIRE_WAND",      category="projectile",  startable=True, unlock_stage_key="WU3",  unlock_order=3,  tags=("projectile", "random"),notes="ランダム性あり投射物。"),
    WeaponEntry(weapon_id=WeaponType.LIGHTNING_RING, key="lightning_ring", display_name="LIGHTNING_RING", category="area",        startable=True, unlock_stage_key="WU4",  unlock_order=4,  tags=("area", "auto"),        notes="自動攻撃寄り。"),
    WeaponEntry(weapon_id=WeaponType.SANTA_WATER,    key="santa_water",    display_name="SANTA_WATER",    category="area_aoe",    startable=True, unlock_stage_key="WU5",  unlock_order=5,  tags=("area", "delayed"),     notes="エリア制圧・遅延AoE。"),
    WeaponEntry(weapon_id=WeaponType.WHIP,           key="whip",           display_name="WHIP",           category="directional", startable=True, unlock_stage_key="WU6",  unlock_order=6,  tags=("directional", "horizontal"), notes="水平範囲依存。"),
    WeaponEntry(weapon_id=WeaponType.CROSS,          key="cross",          display_name="CROSS",          category="directional", startable=True, unlock_stage_key="WU7",  unlock_order=7,  tags=("directional", "boomerang"), notes="往復軌道。"),
    WeaponEntry(weapon_id=WeaponType.RUNETRACER,     key="runetracer",     display_name="RUNETRACER",     category="bounce",      startable=True, unlock_stage_key="WU8",  unlock_order=8,  tags=("bounce", "reflect"),   notes="反射・地形依存。"),
    WeaponEntry(weapon_id=WeaponType.AXE,            key="axe",            display_name="AXE",            category="arc",         startable=True, unlock_stage_key="WU9",  unlock_order=9,  tags=("arc", "vertical"),     notes="放物線/縦方向。"),
    WeaponEntry(weapon_id=WeaponType.KNIFE,          key="knife",          display_name="KNIFE",          category="directional", startable=True, unlock_stage_key="WU10", unlock_order=10, tags=("directional", "aimed"),notes="移動方向依存。エイム要求最高。"),
    WeaponEntry(weapon_id=WeaponType.PEACHONE,       key="peachone",       display_name="PEACHONE",       category="periodic",    startable=True, unlock_stage_key="WU11", unlock_order=11, tags=("periodic", "delayed"), notes="遅延・周期攻撃。"),
    WeaponEntry(weapon_id=WeaponType.EBONY_WINGS,    key="ebony_wings",    display_name="EBONY_WINGS",    category="periodic",    startable=True, unlock_stage_key="WU12", unlock_order=12, tags=("periodic", "pair"),    notes="PEACHONE系対。"),
]

WEAPON_UNLOCK_ORDER_KINGBIBLE_FIRST_V1: list[WeaponEntry] = [
    WeaponEntry(weapon_id=WeaponType.KING_BIBLE,     key="king_bible",     display_name="KING_BIBLE",     category="melee_orbit", startable=True, unlock_stage_key="WU0",  unlock_order=0,  tags=("melee", "orbit"),       notes="GARLICに最も近い範囲武器。"),
    WeaponEntry(weapon_id=WeaponType.GARLIC,         key="garlic",         display_name="GARLIC",         category="melee_aoe",   startable=True, unlock_stage_key="WU1",  unlock_order=1,  tags=("melee", "aoe"),         notes="近距離AoE。ベースライン。"),
    WeaponEntry(weapon_id=WeaponType.MAGIC_WAND,     key="magic_wand",     display_name="MAGIC_WAND",     category="projectile",  startable=True, unlock_stage_key="WU2",  unlock_order=2,  tags=("projectile", "target"), notes="近い敵を狙う自動投射物。"),
    WeaponEntry(weapon_id=WeaponType.FIRE_WAND,      key="fire_wand",      display_name="FIRE_WAND",      category="projectile",  startable=True, unlock_stage_key="WU3",  unlock_order=3,  tags=("projectile", "random"), notes="ランダム性あり投射物。"),
    WeaponEntry(weapon_id=WeaponType.LIGHTNING_RING, key="lightning_ring", display_name="LIGHTNING_RING", category="area",        startable=True, unlock_stage_key="WU4",  unlock_order=4,  tags=("area", "auto"),         notes="自動攻撃寄り。"),
    WeaponEntry(weapon_id=WeaponType.SANTA_WATER,    key="santa_water",    display_name="SANTA_WATER",    category="area_aoe",    startable=True, unlock_stage_key="WU5",  unlock_order=5,  tags=("area", "delayed"),      notes="エリア制圧・遅延AoE。"),
    WeaponEntry(weapon_id=WeaponType.WHIP,           key="whip",           display_name="WHIP",           category="directional", startable=True, unlock_stage_key="WU6",  unlock_order=6,  tags=("directional", "horizontal"), notes="水平範囲依存。"),
    WeaponEntry(weapon_id=WeaponType.CROSS,          key="cross",          display_name="CROSS",          category="directional", startable=True, unlock_stage_key="WU7",  unlock_order=7,  tags=("directional", "boomerang"), notes="往復軌道。"),
    WeaponEntry(weapon_id=WeaponType.RUNETRACER,     key="runetracer",     display_name="RUNETRACER",     category="bounce",      startable=True, unlock_stage_key="WU8",  unlock_order=8,  tags=("bounce", "reflect"),    notes="反射・地形依存。"),
    WeaponEntry(weapon_id=WeaponType.AXE,            key="axe",            display_name="AXE",            category="arc",         startable=True, unlock_stage_key="WU9",  unlock_order=9,  tags=("arc", "vertical"),      notes="放物線/縦方向。"),
    WeaponEntry(weapon_id=WeaponType.KNIFE,          key="knife",          display_name="KNIFE",          category="directional", startable=True, unlock_stage_key="WU10", unlock_order=10, tags=("directional", "aimed"),  notes="移動方向依存。エイム要求最高。"),
    WeaponEntry(weapon_id=WeaponType.PEACHONE,       key="peachone",       display_name="PEACHONE",       category="periodic",    startable=True, unlock_stage_key="WU11", unlock_order=11, tags=("periodic", "delayed"),   notes="遅延・周期攻撃。"),
    WeaponEntry(weapon_id=WeaponType.EBONY_WINGS,    key="ebony_wings",    display_name="EBONY_WINGS",    category="periodic",    startable=True, unlock_stage_key="WU12", unlock_order=12, tags=("periodic", "pair"),      notes="PEACHONE系対。"),
]

WEAPON_UNLOCK_TABLES: dict[str, list[WeaponEntry]] = {
    "default_v1": WEAPON_UNLOCK_ORDER,
    "kingbible_first_v1": WEAPON_UNLOCK_ORDER_KINGBIBLE_FIRST_V1,
}


def resolve_weapon_unlock_order(table_name: str) -> list[WeaponEntry]:
    """テーブル名から WeaponEntry リストを返す。不明な名前は ValueError。"""
    if table_name not in WEAPON_UNLOCK_TABLES:
        raise ValueError(f"Unknown weapon_unlock_table: {table_name!r}")
    return WEAPON_UNLOCK_TABLES[table_name]


ITEM_SYSTEM_STAGES: dict[str, ItemSystemStage] = {
    "IS0": ItemSystemStage(key="IS0", enable_passives=False, enable_evolutions=False),
    "IS1": ItemSystemStage(key="IS1", enable_passives=True,  enable_evolutions=False),
    "IS2": ItemSystemStage(key="IS2", enable_passives=True,  enable_evolutions=True),
}

_STAGE_KEY_TO_ORDER: dict[str, int] = {e.unlock_stage_key: e.unlock_order for e in WEAPON_UNLOCK_ORDER}


def get_unlocked_weapon_ids(stage_key: str, weapon_unlock_order: list[WeaponEntry] = WEAPON_UNLOCK_ORDER) -> list[int]:
    """stage_key までにアンロックされた全 weapon_id を返す。"""
    stage_key_to_order = {e.unlock_stage_key: e.unlock_order for e in weapon_unlock_order}
    order = stage_key_to_order[stage_key]
    return [e.weapon_id for e in weapon_unlock_order if e.unlock_order <= order]


def get_unlocked_startable_weapon_ids(stage_key: str, weapon_unlock_order: list[WeaponEntry] = WEAPON_UNLOCK_ORDER) -> list[int]:
    """stage_key までにアンロックされた startable=True の weapon_id を返す。"""
    stage_key_to_order = {e.unlock_stage_key: e.unlock_order for e in weapon_unlock_order}
    order = stage_key_to_order[stage_key]
    return [e.weapon_id for e in weapon_unlock_order if e.unlock_order <= order and e.startable]


def get_added_weapon_id(stage_key: str, weapon_unlock_order: list[WeaponEntry] = WEAPON_UNLOCK_ORDER) -> int:
    """stage_key で新規追加される weapon_id を返す。"""
    for e in weapon_unlock_order:
        if e.unlock_stage_key == stage_key:
            return e.weapon_id
    raise ValueError(f"Unknown stage_key: {stage_key!r}")


def build_weapon_params_for_cell(
    *,
    first_weapon_id: int,
    max_unlocked_stage_key: str,
    item_stage_key: str,
    pool_policy: str,
    weapon_unlock_order: list[WeaponEntry] = WEAPON_UNLOCK_ORDER,
) -> dict:
    """セル定義から UE5 /params 用の武器パラメータ dict を返す。"""
    item_stage = ITEM_SYSTEM_STAGES.get(item_stage_key, ITEM_SYSTEM_STAGES["IS0"])

    if pool_policy == "target_plus_anchor":
        garlic_id = WeaponType.GARLIC
        if first_weapon_id == garlic_id:
            allowed = [garlic_id]
        else:
            allowed = [first_weapon_id, garlic_id]
        initial_slots = [{"weapon_id": first_weapon_id, "level": 1}]
    elif pool_policy == "target_plus_anchor_if_unlocked":
        garlic_id = WeaponType.GARLIC
        allowed = [first_weapon_id]
        unlocked_ids = get_unlocked_weapon_ids(max_unlocked_stage_key, weapon_unlock_order)
        if garlic_id in unlocked_ids and garlic_id != first_weapon_id:
            allowed.append(garlic_id)
        initial_slots = [{"weapon_id": first_weapon_id, "level": 1}]
    else:
        raise ValueError(f"Unknown pool_policy: {pool_policy!r}. Supported: 'target_plus_anchor', 'target_plus_anchor_if_unlocked'.")

    return {
        "weapon_pool_mode": "fixed_subset",
        "allowed_weapon_types": allowed,
        "enable_passives": item_stage.enable_passives,
        "enable_evolutions": item_stage.enable_evolutions,
        "replay_old_phase_fraction": 0.0,
        "starting_weapon_mode": "pool_random",
        "initial_weapon_slots": initial_slots,
    }
