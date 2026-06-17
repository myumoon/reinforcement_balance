"""
制約付きランダム武器・パッシブスロット生成（RSI Phase 1）。
既存の WeaponCurriculumCallback / HybridCurriculumSpalfCallback とは独立。
"""
from __future__ import annotations

import random
from typing import Optional

from games.survivors.survivors_vs_spec import (
    PassiveItemType,
    PASSIVE_MAX_LEVEL,
    PASSIVE_VALID_FOR_RSI,
    WEAPON_MAX_LEVEL,
    WEAPON_VALID_FOR_RSI,
    EVOLUTION_TABLE,
)

_PHASE_RSI: dict[str, dict] = {
    # weapon_lv: weapon_num本の武器全体への合計レベルのmin/max（_distributeで分配）
    # passive_lv: passive_num個のパッシブ全体への合計レベルのmin/max（_distribute_variable_maxで分配）
    "群れ対応A": {
        "elapsed_time": 300.0,
        "weapon_num": (2, 3), "weapon_lv": (4, 8),
        "passive_num": (0, 2), "passive_lv": (1, 6),
        "allow_evolved": False,
    },
    "群れ対応B": {
        "elapsed_time": 420.0,
        "weapon_num": (3, 4), "weapon_lv": (7, 12),
        "passive_num": (1, 3), "passive_lv": (2, 10),
        "allow_evolved": False,
    },
    "群れ対応C": {
        "elapsed_time": 600.0,
        "weapon_num": (4, 5), "weapon_lv": (10, 20),
        "passive_num": (2, 4), "passive_lv": (5, 18),
        "allow_evolved": False,
    },
    "Mad Forest 入門": {
        "elapsed_time": 600.0,
        "weapon_num": (4, 5), "weapon_lv": (14, 32),
        "passive_num": (3, 5), "passive_lv": (7, 22),
        "allow_evolved": False,
    },
    "Mad Forest 中級": {
        "elapsed_time": 900.0,
        "weapon_num": (5, 6), "weapon_lv": (30, 48),  # 各武器Lv6〜8相当（5〜6本 × Lv6〜8）
        "passive_num": (4, 6), "passive_lv": (12, 28),
        "allow_evolved": True,
    },
    "Mad Forest": {
        "elapsed_time": 900.0,
        "weapon_num": (5, 6), "weapon_lv": (30, 48),  # 各武器Lv6〜8相当（5〜6本 × Lv6〜8）
        "passive_num": (4, 6), "passive_lv": (12, 28),
        "allow_evolved": True,
    },
}


def get_initial_elapsed_time(phase_name: str) -> float:
    """フェーズ名から initial_elapsed_time を返す。RSI非対象フェーズは 0.0。"""
    entry = _PHASE_RSI.get(phase_name)
    return entry["elapsed_time"] if entry else 0.0


def _get_phase_rsi(phase_name: str) -> Optional[dict]:
    return _PHASE_RSI.get(phase_name)


def _distribute(total: int, n: int, max_per: int, rng: random.Random) -> list[int]:
    """total を n 個に分配（各最低1、最大 max_per）。武器スロット用。"""
    total = max(total, n)
    levels = [1] * n
    remaining = total - n
    indices = list(range(n))
    rng.shuffle(indices)
    for i in indices:
        add = min(remaining, max_per - 1)
        levels[i] += add
        remaining -= add
        if remaining <= 0:
            break
    return levels


def _distribute_variable_max(total: int, max_levels: list[int], rng: random.Random) -> list[int]:
    """total を len(max_levels) 個に分配（各最低1、各最大は max_levels[i]）。パッシブスロット用。"""
    n = len(max_levels)
    if n == 0:
        return []
    total = max(total, n)
    total = min(total, sum(max_levels))
    levels = [1] * n
    remaining = total - n
    indices = list(range(n))
    rng.shuffle(indices)
    for i in indices:
        add = min(remaining, max_levels[i] - 1)
        levels[i] += add
        remaining -= add
        if remaining <= 0:
            break
    return levels


def _sample_evolved_set(rng: random.Random) -> Optional[tuple[dict, dict, list[int]]]:
    """進化テーブルからランダムに1セットを返す。
    Returns: (evolved_weapon_slot, passive_slot, excluded_weapon_ids) または None
    """
    candidates = [e for e in EVOLUTION_TABLE if e["passive"] != PassiveItemType.NONE]
    if not candidates:
        return None
    entry = rng.choice(candidates)
    weapon_slot = {"weapon_id": entry["evolved"], "level": 1}
    passive_slot = {"passive_id": entry["passive"], "level": PASSIVE_MAX_LEVEL[entry["passive"]]}
    # 元武器IDを除外リストとして返す（進化後スロットと重複しないよう基本武器・合成武器を除外）
    excluded = [entry["base"]]
    if "union_weapon" in entry:
        excluded.append(entry["union_weapon"])
    return weapon_slot, passive_slot, excluded


def sample_weapon_slots(
    phase_name: str,
    rng: Optional[random.Random] = None,
) -> Optional[dict]:
    """フェーズ名に対応する制約付きランダム武器・パッシブスロットを生成する。
    RSI 不要フェーズは None を返す。

    Returns:
        None または以下のキーを持つ dict:
          "initial_elapsed_time": float
          "initial_weapon_slots": list[{"weapon_id": int, "level": int}]
          "initial_passive_slots": list[{"passive_id": int, "level": int}]
          "MaxEpisodeTime": float
    """
    constraints = _get_phase_rsi(phase_name)
    if constraints is None:
        return None
    elapsed_time = constraints["elapsed_time"]
    if elapsed_time <= 0.0:
        return None

    _rng = rng or random.Random()

    num_w    = _rng.randint(*constraints["weapon_num"])
    total_lv = _rng.randint(*constraints["weapon_lv"])

    weapon_slots: list[dict] = []
    passive_slots: list[dict] = []
    used_passives: set[int] = set()
    excluded_bases: list[int] = []

    if constraints.get("allow_evolved") and _rng.random() < 0.5:
        result = _sample_evolved_set(_rng)
        if result:
            ew_slot, ep_slot, excluded_bases = result
            weapon_slots.append(ew_slot)
            passive_slots.append(ep_slot)
            used_passives.add(ep_slot["passive_id"])
            num_w = max(num_w - 1, 0)

    # 残りの基本武器（進化後武器スロット使用済み＋進化元武器も除外）
    already_used_weapons = {s["weapon_id"] for s in weapon_slots} | set(excluded_bases)
    remaining_weapons = [w for w in WEAPON_VALID_FOR_RSI if w not in already_used_weapons]
    num_w = min(num_w, len(remaining_weapons))
    if num_w > 0:
        chosen = _rng.sample(remaining_weapons, num_w)
        base_total = max(total_lv - len(weapon_slots), num_w)
        levels = _distribute(base_total, num_w, WEAPON_MAX_LEVEL, _rng)
        weapon_slots.extend({"weapon_id": w, "level": lv} for w, lv in zip(chosen, levels))

    num_p_min, num_p_max = constraints["passive_num"]
    total_p_lv = _rng.randint(*constraints["passive_lv"])
    num_p = _rng.randint(num_p_min, num_p_max)

    available_passives = [p for p in PASSIVE_VALID_FOR_RSI if p not in used_passives]
    num_p = min(num_p - len(passive_slots), len(available_passives))
    if num_p > 0:
        chosen_p = _rng.sample(available_passives, num_p)
        max_lvs  = [PASSIVE_MAX_LEVEL[pid] for pid in chosen_p]
        p_levels = _distribute_variable_max(total_p_lv, max_lvs, _rng)
        passive_slots.extend(
            {"passive_id": pid, "level": lv}
            for pid, lv in zip(chosen_p, p_levels)
        )

    return {
        "initial_elapsed_time":   elapsed_time,
        "initial_weapon_slots":   weapon_slots,
        "initial_passive_slots":  passive_slots,
        "MaxEpisodeTime":         elapsed_time + 300.0,
    }
