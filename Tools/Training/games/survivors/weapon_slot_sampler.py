"""
制約付きランダム武器スロット生成（RSI Phase 1）。
既存の WeaponCurriculumCallback / HybridCurriculumSpalfCallback とは独立。
"""
from __future__ import annotations

import random
from typing import Optional

VALID_WEAPONS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14]

_TIME_CONSTRAINTS: list[tuple[float, dict]] = [
    (900.0, {"num": (5, 6), "total_lv": (14, 22)}),
    (600.0, {"num": (4, 5), "total_lv": (11, 16)}),
    (420.0, {"num": (3, 4), "total_lv": (7,  12)}),
    (300.0, {"num": (2, 3), "total_lv": (4,   8)}),
]

_PHASE_ELAPSED: list[tuple[str, float]] = [
    ("Mad Forest",   900.0),
    ("群れ対応C",    600.0),
    ("群れ対応B",    420.0),
    ("群れ対応A",    300.0),
]


def get_initial_elapsed_time(phase_name: str) -> float:
    for keyword, elapsed in _PHASE_ELAPSED:
        if keyword in phase_name:
            return elapsed
    return 0.0


def _distribute(total: int, n: int, rng: random.Random) -> list[int]:
    total = max(total, n)
    levels = [1] * n
    remaining = total - n
    indices = list(range(n))
    rng.shuffle(indices)
    for i in indices:
        add = min(remaining, 7)
        levels[i] += add
        remaining -= add
        if remaining <= 0:
            break
    return levels


def sample_weapon_slots(
    elapsed_time: float,
    rng: Optional[random.Random] = None,
) -> Optional[list[dict]]:
    if elapsed_time <= 0.0:
        return None
    _rng = rng or random.Random()
    constraints = None
    for threshold, c in _TIME_CONSTRAINTS:
        if elapsed_time >= threshold:
            constraints = c
            break
    if constraints is None:
        return None
    num      = _rng.randint(*constraints["num"])
    total_lv = _rng.randint(*constraints["total_lv"])
    weapons  = _rng.sample(VALID_WEAPONS, min(num, len(VALID_WEAPONS)))
    levels   = _distribute(total_lv, len(weapons), _rng)
    return [{"weapon_id": w, "level": lv} for w, lv in zip(weapons, levels)]
