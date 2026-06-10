"""
Vampire Survivors 武器カリキュラム定義 W0〜W6。
HybridCurriculumSpalf（敵難易度）と独立した別軸で制御する。
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

ALL_BASE_WEAPONS = [
    WeaponType.GARLIC, WeaponType.WHIP, WeaponType.MAGIC_WAND,
    WeaponType.KNIFE, WeaponType.AXE, WeaponType.CROSS,
    WeaponType.KING_BIBLE, WeaponType.FIRE_WAND, WeaponType.SANTA_WATER,
    WeaponType.RUNETRACER, WeaponType.LIGHTNING_RING, WeaponType.PENTAGRAM,
    WeaponType.PEACHONE, WeaponType.EBONY_WINGS, WeaponType.LAUREL,
]

ALL_EVOLVED_WEAPONS = [
    WeaponType.SOUL_EATER, WeaponType.BLOODY_TEAR, WeaponType.HOLY_WAND,
    WeaponType.THOUSAND_EDGE, WeaponType.DEATH_SPIRAL, WeaponType.HEAVEN_SWORD,
    WeaponType.UNHOLY_VESPERS, WeaponType.HELLFIRE, WeaponType.LA_BORRA,
    WeaponType.NO_FUTURE, WeaponType.THUNDER_LOOP, WeaponType.GORGEOUS_MOON,
    WeaponType.VANDALIER,
]

WEAPON_PHASES: dict[str, dict] = {
    "W0": {
        "weapon_pool_mode": "garlic_only",
        "allowed_weapon_types": [WeaponType.GARLIC],
        "enable_passives": False,
        "enable_evolutions": False,
        "replay_old_phase_fraction": 0.0,
        "net_arch": [512, 256],
        "log_combination_rewards": True,
    },
    "W0_to_W1": {
        "weapon_pool_mode": "weighted",
        "allowed_weapon_types": [
            WeaponType.GARLIC, WeaponType.KING_BIBLE, WeaponType.SANTA_WATER,
        ],
        "enable_passives": False,
        "enable_evolutions": False,
        "replay_old_phase_fraction": 0.3,
        "transition_steps": 2_000_000,
        "from_phase": "W0",
        "to_phase": "W1",
        "log_combination_rewards": True,
    },
    "W1": {
        "weapon_pool_mode": "fixed_subset",
        "allowed_weapon_types": [
            WeaponType.GARLIC, WeaponType.KING_BIBLE, WeaponType.SANTA_WATER,
        ],
        "enable_passives": True,
        "enable_evolutions": False,
        "replay_old_phase_fraction": 0.0,  # v08: W0廃止により古いフェーズが存在しないため0.0に変更
        "log_combination_rewards": True,
    },
    "W1_to_W2": {
        "weapon_pool_mode": "weighted",
        "allowed_weapon_types": [
            WeaponType.GARLIC, WeaponType.WHIP, WeaponType.MAGIC_WAND,
            WeaponType.KNIFE, WeaponType.AXE, WeaponType.CROSS,
            WeaponType.KING_BIBLE, WeaponType.SANTA_WATER,
        ],
        "enable_passives": True,
        "enable_evolutions": False,
        "replay_old_phase_fraction": 0.25,
        "transition_steps": 2_000_000,
        "from_phase": "W1",
        "to_phase": "W2",
        "log_combination_rewards": True,
    },
    "W2": {
        "weapon_pool_mode": "fixed_subset",
        "allowed_weapon_types": [
            WeaponType.GARLIC, WeaponType.WHIP, WeaponType.MAGIC_WAND,
            WeaponType.KNIFE, WeaponType.AXE, WeaponType.CROSS,
            WeaponType.KING_BIBLE, WeaponType.SANTA_WATER,
        ],
        "enable_passives": True,
        "enable_evolutions": False,
        "replay_old_phase_fraction": 0.2,
        "log_combination_rewards": True,
    },
    "W3": {
        "weapon_pool_mode": "all_base",
        "allowed_weapon_types": ALL_BASE_WEAPONS,
        "enable_passives": True,
        "enable_evolutions": True,
        "replay_old_phase_fraction": 0.25,
        "log_combination_rewards": True,
    },
    "W4": {
        "weapon_pool_mode": "all_base",
        "allowed_weapon_types": ALL_BASE_WEAPONS,
        "enable_passives": True,
        "enable_evolutions": True,
        "replay_old_phase_fraction": 0.3,
        "log_combination_rewards": True,
    },
    "W5": {
        "weapon_pool_mode": "all_with_evolutions",
        "allowed_weapon_types": ALL_BASE_WEAPONS + ALL_EVOLVED_WEAPONS,
        "enable_passives": True,
        "enable_evolutions": True,
        "replay_old_phase_fraction": 0.3,
        "log_combination_rewards": True,
    },
}

TRANSITION_WEAPON_WEIGHTS: dict[str, dict] = {
    "W0_to_W1": {
        "start": {WeaponType.GARLIC: 1.0},
        "end": {
            WeaponType.GARLIC: 0.5,
            WeaponType.KING_BIBLE: 0.25,
            WeaponType.SANTA_WATER: 0.25,
        },
    },
    "W1_to_W2": {
        "start": {
            WeaponType.GARLIC: 0.5,
            WeaponType.KING_BIBLE: 0.25,
            WeaponType.SANTA_WATER: 0.25,
        },
        "end": {
            WeaponType.GARLIC: 0.2,
            WeaponType.WHIP: 0.1,
            WeaponType.MAGIC_WAND: 0.1,
            WeaponType.KNIFE: 0.1,
            WeaponType.AXE: 0.1,
            WeaponType.CROSS: 0.1,
            WeaponType.KING_BIBLE: 0.15,
            WeaponType.SANTA_WATER: 0.15,
        },
    },
}


# BC 専用固定 preset（W5 相当: 全基本武器+パッシブ+進化、weighted transition なし）
BC_WEAPON_PRESET: dict = {
    "weapon_pool_mode": "all_base",
    "allowed_weapon_types": ALL_BASE_WEAPONS,
    "enable_passives": True,
    "enable_evolutions": True,
    "replay_old_phase_fraction": 0.0,
    "starting_weapon_mode": "pool_random",
}


def get_weapon_weights(phase_key: str, phase_progress: float) -> dict[int, float]:
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

    weighted フェーズ（W0_to_W1 / W1_to_W2）では global_step に応じて
    weapon_weights を線形補間する。fixed フェーズでは weapon_weights は返さない。

    Args:
        phase_key: WEAPON_PHASES のキー（例: "W0", "W0_to_W1"）。
        global_step: 現在のフェーズ内経過ステップ数。weighted フェーズでの補間に使用。
            ※ 累積 num_timesteps ではなく、フェーズ開始からの相対ステップを渡すこと。

    Returns:
        UE5 の /params エンドポイントに送信するパラメータ辞書。

    Note:
        UE5 側の /params が以下のキーをサポートしている必要があります（Task A PR で実装）:
        - weapon_pool_mode, allowed_weapon_types, enable_passives, enable_evolutions,
          replay_old_phase_fraction, starting_weapon_mode, weapon_weights
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
    if phase.get("weapon_pool_mode") == "weighted":
        transition_steps = phase.get("transition_steps", 2_000_000)
        progress = min(global_step / max(transition_steps, 1), 1.0)
        weights = get_weapon_weights(phase_key, progress)
        params["weapon_weights"] = weights
        # 重み > 0 の武器 ID のみ allowed_weapon_types に含める
        # （重み=0 の武器がプール入りすると遷移設計が崩れるため除外する）
        params["allowed_weapon_types"] = [
            wid for wid, w in weights.items() if w > 1e-6
        ]
    return params
