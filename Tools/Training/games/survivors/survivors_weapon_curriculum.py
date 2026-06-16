"""
Vampire Survivors 武器カリキュラム定義 W0〜W5。
HybridCurriculumSpalf（敵難易度）と独立した別軸で制御する。

v09 変更:
  - W0 Garlic 単独フェーズを復活（v06 で Phase 11 突破実績あり）
  - W0_to_W1 / W1_to_W2 の遷移フェーズを廃止（停滞ベース → ゲートベース昇格）
  - W1-W4 をタスク類似性に基づく武器グループで再設計
  - W5 は全基本武器 + passive + evolution
"""

from __future__ import annotations

# EWeaponType の integer 値（C++ enum と一致させること）
class WeaponType:
    NONE          = 0
    GARLIC        = 1
    WHIP          = 2
    MAGIC_WAND    = 3
    KNIFE         = 4
    AXE           = 5
    CROSS         = 6
    KING_BIBLE    = 7
    FIRE_WAND     = 8
    SANTA_WATER   = 9
    RUNETRACER    = 10
    LIGHTNING_RING = 11
    PENTAGRAM     = 12
    PEACHONE      = 13
    EBONY_WINGS   = 14
    LAUREL        = 15
    SOUL_EATER    = 16
    BLOODY_TEAR   = 17
    HOLY_WAND     = 18
    THOUSAND_EDGE = 19
    DEATH_SPIRAL  = 20
    HEAVEN_SWORD  = 21
    UNHOLY_VESPERS = 22
    HELLFIRE      = 23
    LA_BORRA      = 24
    NO_FUTURE     = 25
    THUNDER_LOOP  = 26
    GORGEOUS_MOON = 27
    VANDALIER     = 28

# 全基本武器（Garlic〜Laurel）。参照・進化判定用。
# Laurel (15) は攻撃判定がない防衛武器（ComputeHits 未実装）。
ALL_BASE_WEAPONS = [
    WeaponType.GARLIC, WeaponType.WHIP, WeaponType.MAGIC_WAND,
    WeaponType.KNIFE, WeaponType.AXE, WeaponType.CROSS,
    WeaponType.KING_BIBLE, WeaponType.FIRE_WAND, WeaponType.SANTA_WATER,
    WeaponType.RUNETRACER, WeaponType.LIGHTNING_RING, WeaponType.PENTAGRAM,
    WeaponType.PEACHONE, WeaponType.EBONY_WINGS, WeaponType.LAUREL,
]

# Laurel を除外した攻撃可能基本武器リスト。
# UE5 の fixed_subset モードでは allowed_weapon_types が初期武器とレベルアップ候補の
# 両方に使われる（SurvivorsPlayerComponent.cpp BuildLevelUpChoices 参照）。
# 現在の C++ API に開始武器とレベルアップ候補を別管理する仕組みがないため、
# W5/BC では Laurel をレベルアップ候補からも除外するトレードオフを採用する。
# Laurel の進化形 NoFuture は enable_evolutions=True の W5 で進化システム経由で出現する。
ALL_BASE_ATTACK_WEAPONS = [w for w in ALL_BASE_WEAPONS if w != WeaponType.LAUREL]

# 進化武器リスト。enable_evolutions=False のフェーズでは出現しない。
# SoulEater は Garlic の進化形（Garlic + Pummarola パッシブで進化）。
# allowed_weapon_types に含めると fixed_subset モードで初期武器・レベルアップ選択肢に
# 出現してしまうため、enable_evolutions=False のフェーズには含めてはならない。
ALL_EVOLVED_WEAPONS = [
    WeaponType.SOUL_EATER, WeaponType.BLOODY_TEAR, WeaponType.HOLY_WAND,
    WeaponType.THOUSAND_EDGE, WeaponType.DEATH_SPIRAL, WeaponType.HEAVEN_SWORD,
    WeaponType.UNHOLY_VESPERS, WeaponType.HELLFIRE, WeaponType.LA_BORRA,
    WeaponType.NO_FUTURE, WeaponType.THUNDER_LOOP, WeaponType.GORGEOUS_MOON,
    WeaponType.VANDALIER,
]

WEAPON_PHASES: dict[str, dict] = {
    "W0": {
        # Garlic 単独フェーズ: 移動基礎スキル習得（v06 で Phase 11 突破実績）
        "weapon_pool_mode": "garlic_only",
        "allowed_weapon_types": [WeaponType.GARLIC],
        "enable_passives": False,
        "enable_evolutions": False,
        "replay_old_phase_fraction": 0.0,
    },
    "W1": {
        # 近距離範囲群: Garlic と戦略的類似性が最も高いグループ
        # SoulEater は Garlic の進化形（enable_evolutions=False なので含めない）
        "weapon_pool_mode": "fixed_subset",
        "allowed_weapon_types": [
            WeaponType.GARLIC, WeaponType.KING_BIBLE,
        ],
        "enable_passives": True,
        "enable_evolutions": False,
        "replay_old_phase_fraction": 0.2,
    },
    "W2": {
        # ターゲット追尾遠距離群: 「敵を向く」という戦略の延長
        # SoulEater は Garlic の進化形（enable_evolutions=False なので含めない）
        "weapon_pool_mode": "fixed_subset",
        "allowed_weapon_types": [
            WeaponType.GARLIC, WeaponType.KING_BIBLE,
            WeaponType.MAGIC_WAND, WeaponType.FIRE_WAND, WeaponType.LIGHTNING_RING,
        ],
        "enable_passives": True,
        "enable_evolutions": False,
        "replay_old_phase_fraction": 0.2,
    },
    "W3": {
        # ライン/エリア制圧群: 敵集団の制御が必要
        # SoulEater は Garlic の進化形（enable_evolutions=False なので含めない）
        "weapon_pool_mode": "fixed_subset",
        "allowed_weapon_types": [
            WeaponType.GARLIC, WeaponType.KING_BIBLE,
            WeaponType.MAGIC_WAND, WeaponType.FIRE_WAND, WeaponType.LIGHTNING_RING,
            WeaponType.WHIP, WeaponType.SANTA_WATER,
        ],
        "enable_passives": True,
        "enable_evolutions": False,
        "replay_old_phase_fraction": 0.2,
    },
    "W4": {
        # 方向型投射物群: エイム・位置取りが必要な最難グループ
        # SoulEater は Garlic の進化形（enable_evolutions=False なので含めない）
        "weapon_pool_mode": "fixed_subset",
        "allowed_weapon_types": [
            WeaponType.GARLIC, WeaponType.KING_BIBLE,
            WeaponType.MAGIC_WAND, WeaponType.FIRE_WAND, WeaponType.LIGHTNING_RING,
            WeaponType.WHIP, WeaponType.SANTA_WATER,
            WeaponType.KNIFE, WeaponType.AXE, WeaponType.CROSS, WeaponType.RUNETRACER,
        ],
        "enable_passives": True,
        "enable_evolutions": False,
        "replay_old_phase_fraction": 0.25,
    },
    "W5": {
        # 全基本武器 + passive + evolution
        # ALL_BASE_ATTACK_WEAPONS (Laurel除外) を使用。
        # Laurel は攻撃不能なため初期武器・レベルアップ候補の両方から除外する。
        # Laurel の進化形 NoFuture は enable_evolutions=True の進化システム経由で出現する。
        "weapon_pool_mode": "fixed_subset",
        "allowed_weapon_types": ALL_BASE_ATTACK_WEAPONS,
        "enable_passives": True,
        "enable_evolutions": True,
        "replay_old_phase_fraction": 0.3,
    },
}

TRANSITION_WEAPON_WEIGHTS: dict[str, dict] = {}


# BC 専用固定 preset（W5 相当: 全基本武器+パッシブ+進化）
# ALL_BASE_ATTACK_WEAPONS (Laurel除外) を使用。W5 と同様の設計方針。
BC_WEAPON_PRESET: dict = {
    "weapon_pool_mode": "fixed_subset",
    "allowed_weapon_types": ALL_BASE_ATTACK_WEAPONS,
    "enable_passives": True,
    "enable_evolutions": True,
    "replay_old_phase_fraction": 0.0,
    "starting_weapon_mode": "pool_random",
}


def get_weapon_weights(phase_key: str, phase_progress: float) -> dict[int, float]:
    """v09 では遷移フェーズ（weighted mode）を廃止したため、常に空 dict を返す。"""
    if phase_key not in TRANSITION_WEAPON_WEIGHTS:
        return {}
    table = TRANSITION_WEAPON_WEIGHTS[phase_key]
    start = table["start"]
    end = table["end"]
    all_keys = set(start.keys()) | set(end.keys())
    weights = {}
    for k in all_keys:
        sw = start.get(k, 0.0)
        ew = end.get(k, 0.0)
        weights[k] = sw + (ew - sw) * phase_progress
    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}
    return weights


def get_params_for_phase(phase_key: str, global_step: int = 0) -> dict:
    """指定フェーズの武器カリキュラムパラメータを返す。

    v09 では全フェーズが fixed_subset / garlic_only / all_base のいずれかであり、
    weighted（遷移）フェーズは廃止されている。global_step は互換性のため残すが使用しない。

    Args:
        phase_key: WEAPON_PHASES のキー（例: "W0", "W1"）。
        global_step: 使用しない（backward compatibility のために残す）。

    Returns:
        UE5 の /params エンドポイントに送信するパラメータ辞書。

    Note:
        UE5 側の /params が以下のキーをサポートしている必要があります:
        - weapon_pool_mode, allowed_weapon_types, enable_passives, enable_evolutions,
          replay_old_phase_fraction, starting_weapon_mode
        UE5 側が未対応の場合、ゲームは Garlic-only で動作します。
    """
    if phase_key not in WEAPON_PHASES:
        raise ValueError(f"Unknown weapon phase: {phase_key}")
    phase = WEAPON_PHASES[phase_key]
    params: dict = {
        "weapon_pool_mode": phase.get("weapon_pool_mode", "garlic_only"),
        "allowed_weapon_types": phase.get("allowed_weapon_types", [WeaponType.GARLIC]),
        "enable_passives": phase.get("enable_passives", False),
        "enable_evolutions": phase.get("enable_evolutions", False),
        "replay_old_phase_fraction": phase.get("replay_old_phase_fraction", 0.0),
        "starting_weapon_mode": "pool_random",
    }
    # v09: weighted フェーズは廃止。weapon_pool_mode=="weighted" は通常発生しない。
    if phase.get("weapon_pool_mode") == "weighted":
        transition_steps = phase.get("transition_steps", 2_000_000)
        progress = min(global_step / max(transition_steps, 1), 1.0)
        weights = get_weapon_weights(phase_key, progress)
        params["weapon_weights"] = weights
        params["allowed_weapon_types"] = [
            wid for wid, w in weights.items() if w > 1e-6
        ]
    return params
