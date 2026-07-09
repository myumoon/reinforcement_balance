"""combination_smoke セル生成。

Phase B (combination smoke) 用のタスクセルを構築する。bootstrap 完了後、
解禁済み startable 武器の 2 武器組み合わせを走査し、全武器をカバーするよう
決定論的にペアセルを生成する。
"""
from __future__ import annotations

import random

from games.survivors.modules.task_cell_sampler_module import TaskCell
from games.survivors.survivors_weapon_table import (
    WEAPON_UNLOCK_ORDER,
    WeaponEntry,
    get_unlocked_startable_weapon_ids,
)

# initial slot のデフォルトレベル
_DEFAULT_SLOT_LEVEL = 4


def _combo_key(weapon_ids: tuple[int, ...]) -> str:
    return "w" + "_".join(str(wid) for wid in sorted(weapon_ids))


def build_combination_smoke_cells(
    stage_key: str,
    enemy_phase_idx: int,
    max_cells: int,
    seed: int,
    weapon_unlock_order: list[WeaponEntry] = WEAPON_UNLOCK_ORDER,
) -> list[TaskCell]:
    """stage_key までに解禁された startable 武器の 2 武器ペアセルを生成する。

    - anchor 武器（unlock_order が最小の startable 武器）と他の全武器のペアを必ず作り、
      全武器がいずれかのセルに含まれるようカバレッジを保証する。
    - anchor ペアで max_cells に満たなければ、残りの武器同士のペアを決定論的に追加する。
    - `random.Random(seed)` でシードを固定して決定論的にする。
    """
    startable_ids = get_unlocked_startable_weapon_ids(stage_key, weapon_unlock_order)
    if len(startable_ids) < 2:
        # ペアを作れない場合は空を返す
        return []

    rng = random.Random(seed)

    anchor_id = startable_ids[0]
    others = startable_ids[1:]

    # anchor ペア（カバレッジ保証）: (anchor, other) を全 other について作る
    pairs: list[tuple[int, int]] = [(anchor_id, other) for other in others]

    # 全武器カバレッジには anchor ペアの数（= len(startable_ids) - 1）が必要。
    coverage_min = len(others)
    if max_cells < coverage_min:
        raise ValueError(
            f"max_cells={max_cells} は全武器カバレッジに必要な {coverage_min} を下回ります。"
            f" max_cells >= {coverage_min} を指定してください（stage_key={stage_key!r}）。"
        )

    if len(pairs) > max_cells:
        pairs = pairs[:max_cells]
    else:
        # 残り枠に anchor 以外の武器ペアを決定論的に追加する
        extra: list[tuple[int, int]] = []
        for i in range(len(others)):
            for j in range(i + 1, len(others)):
                extra.append((others[i], others[j]))
        rng.shuffle(extra)
        remaining = max_cells - len(pairs)
        if remaining > 0:
            pairs.extend(extra[:remaining])

    # 重複除去（sorted タプルで一意化）しつつ順序を保つ
    seen: set[tuple[int, int]] = set()
    unique_pairs: list[tuple[int, int]] = []
    for a, b in pairs:
        norm = (min(a, b), max(a, b))
        if norm in seen:
            continue
        seen.add(norm)
        unique_pairs.append(norm)

    cells: list[TaskCell] = []
    for a, b in unique_pairs:
        weapon_ids = (a, b)
        slots = tuple((wid, _DEFAULT_SLOT_LEVEL) for wid in weapon_ids)
        cell = TaskCell(
            weapon_unlock_stage_key=stage_key,
            first_weapon_id=a,
            enemy_phase_idx=enemy_phase_idx,
            task_kind="combination_smoke",
            build_policy="combination_slots",
            combo_key=_combo_key(weapon_ids),
            initial_weapon_slots=slots,
            initial_passive_slots=(),
            allowed_weapon_ids=weapon_ids,
        )
        cells.append(cell)

    return cells
