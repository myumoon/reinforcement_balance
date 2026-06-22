"""
Vampire Survivors 仕様定数。
武器・パッシブ・進化テーブルなど、ゲーム仕様に由来する定数を集約する。
C++ 側の SurvivorsTypes.h / SurvivorsWikiSpec.h と整合させること。
"""
from __future__ import annotations

from games.survivors.survivors_weapon_curriculum import WeaponType  # 既存を再利用


class PassiveItemType:
    """EPassiveItemType の integer 値（C++ enum と一致させること）。"""
    NONE          = 0
    SPINACH       = 1
    ARMOR         = 2
    HOLLOW_HEART  = 3
    PUMMAROLA     = 4
    EMPTY_TOME    = 5
    CANDELABRADOR = 6
    BRACER        = 7
    SPELLBINDER   = 8
    DUPLICATOR    = 9
    WINGS         = 10
    ATTRACTORB    = 11
    CLOVER        = 12
    CROWN         = 13
    STONE_MASK    = 14  # MaxLevel=0（効果未実装）
    SKULL_O_MANIAC= 15
    TIRAJISU      = 16  # MaxLevel=2（リバイバル）
    TORRONAS_BOX  = 17  # MaxLevel=9


PASSIVE_MAX_LEVEL: dict[int, int] = {
    PassiveItemType.NONE:           0,
    PassiveItemType.SPINACH:        5,
    PassiveItemType.ARMOR:          5,
    PassiveItemType.HOLLOW_HEART:   5,
    PassiveItemType.PUMMAROLA:      5,
    PassiveItemType.EMPTY_TOME:     5,
    PassiveItemType.CANDELABRADOR:  5,
    PassiveItemType.BRACER:         5,
    PassiveItemType.SPELLBINDER:    5,
    PassiveItemType.DUPLICATOR:     2,
    PassiveItemType.WINGS:          5,
    PassiveItemType.ATTRACTORB:     5,
    PassiveItemType.CLOVER:         5,
    PassiveItemType.CROWN:          5,
    PassiveItemType.STONE_MASK:     0,
    PassiveItemType.SKULL_O_MANIAC: 5,
    PassiveItemType.TIRAJISU:       2,
    PassiveItemType.TORRONAS_BOX:   9,
}

EVOLUTION_TABLE: list[dict] = [
    {"base": WeaponType.GARLIC,        "passive": PassiveItemType.PUMMAROLA,    "evolved": WeaponType.SOUL_EATER},
    {"base": WeaponType.WHIP,          "passive": PassiveItemType.HOLLOW_HEART, "evolved": WeaponType.BLOODY_TEAR},
    {"base": WeaponType.MAGIC_WAND,    "passive": PassiveItemType.EMPTY_TOME,   "evolved": WeaponType.HOLY_WAND},
    {"base": WeaponType.KNIFE,         "passive": PassiveItemType.BRACER,       "evolved": WeaponType.THOUSAND_EDGE},
    {"base": WeaponType.AXE,           "passive": PassiveItemType.CANDELABRADOR,"evolved": WeaponType.DEATH_SPIRAL},
    {"base": WeaponType.CROSS,         "passive": PassiveItemType.CLOVER,       "evolved": WeaponType.HEAVEN_SWORD},
    {"base": WeaponType.KING_BIBLE,    "passive": PassiveItemType.SPELLBINDER,  "evolved": WeaponType.UNHOLY_VESPERS},
    {"base": WeaponType.FIRE_WAND,     "passive": PassiveItemType.SPINACH,      "evolved": WeaponType.HELLFIRE},
    {"base": WeaponType.SANTA_WATER,   "passive": PassiveItemType.ATTRACTORB,   "evolved": WeaponType.LA_BORRA},
    {"base": WeaponType.RUNETRACER,    "passive": PassiveItemType.ARMOR,        "evolved": WeaponType.NO_FUTURE},
    {"base": WeaponType.LIGHTNING_RING,"passive": PassiveItemType.DUPLICATOR,   "evolved": WeaponType.THUNDER_LOOP},
    {"base": WeaponType.PENTAGRAM,     "passive": PassiveItemType.CROWN,        "evolved": WeaponType.GORGEOUS_MOON},
    {"base": WeaponType.PEACHONE,      "passive": PassiveItemType.NONE,         "evolved": WeaponType.VANDALIER,
     "union_weapon": WeaponType.EBONY_WINGS},
]

WEAPON_MAX_LEVEL = 8
EVOLVED_MAX_LEVEL = 1

WEAPON_EXCLUDED_AS_STARTING = [
    WeaponType.PENTAGRAM,
    WeaponType.LAUREL,
    WeaponType.GORGEOUS_MOON,
]

WEAPON_VALID_FOR_RSI = [
    WeaponType.GARLIC, WeaponType.WHIP, WeaponType.MAGIC_WAND,
    WeaponType.KNIFE, WeaponType.AXE, WeaponType.CROSS,
    WeaponType.KING_BIBLE, WeaponType.FIRE_WAND, WeaponType.SANTA_WATER,
    WeaponType.RUNETRACER, WeaponType.LIGHTNING_RING,
    WeaponType.PEACHONE, WeaponType.EBONY_WINGS,
]

PASSIVE_VALID_FOR_RSI = [
    pid for pid, max_lv in PASSIVE_MAX_LEVEL.items()
    if pid != PassiveItemType.NONE and max_lv > 0
]
