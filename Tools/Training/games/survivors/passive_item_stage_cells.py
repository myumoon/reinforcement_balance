"""IS1 passive item stage セル生成。

Phase 04 (Passive Item Stage) 用のタスクセルを構築する。IS1（passive 有効 /
evolution 無効）で全 startable 武器 × 全 passive をカバーするよう決定論的に
combination_slots セルを生成する。
"""
from __future__ import annotations

import random

from games.survivors.modules.task_cell_sampler_module import TaskCell
from games.survivors.survivors_vs_spec import PASSIVE_MAX_LEVEL, PASSIVE_VALID_FOR_RSI
from games.survivors.survivors_weapon_table import (
    WEAPON_UNLOCK_ORDER,
    WeaponEntry,
    get_unlocked_startable_weapon_ids,
)

# target 武器 slot のデフォルトレベル
_DEFAULT_WEAPON_SLOT_LEVEL = 3


def _passive_slot(pid: int, rng: random.Random) -> tuple[int, int]:
    """passive id に対して 1..max_level のランダムなレベルを持つ slot を返す。"""
    max_level = int(PASSIVE_MAX_LEVEL[pid])
    return (int(pid), rng.randint(1, max_level))


def build_passive_item_stage_cells(
    stage_key: str,
    enemy_phase_idx: int,
    max_cells: int,
    seed: int,
    weapon_unlock_order: list[WeaponEntry] = WEAPON_UNLOCK_ORDER,
) -> list[TaskCell]:
    """stage_key までに解禁された startable 武器 × passive の IS1 セルを生成する。

    - 各 startable 武器につき 1 セルを作り、全武器をカバーする。
    - 各セルには 1〜2 個の passive slot を割り当て、全 passive をカバーする。
    - 全武器を割り当てた後、passive の未カバー分を追加セルで補う。
    - `random.Random(seed)` でシードを固定して決定論的にする。

    Returns:
        最大 max_cells 個の TaskCell。
    """
    rng = random.Random(seed)
    weapons = get_unlocked_startable_weapon_ids(stage_key, weapon_unlock_order)
    passives = list(PASSIVE_VALID_FOR_RSI)
    if not weapons or not passives:
        return []

    if max_cells < len(weapons):
        raise ValueError(
            f"max_cells={max_cells} is less than the number of startable weapons ({len(weapons)}). "
            "All startable weapons must be covered."
        )

    cells: list[TaskCell] = []

    # 1) 各 startable 武器について 1 セル（全武器カバレッジ保証）。
    #    各セルに 1〜2 個の passive を割り当て、全 passive を広くカバーする。
    for idx, wid in enumerate(weapons):
        pids = [passives[idx % len(passives)]]
        if len(passives) > 1:
            pids.append(passives[(idx + 3) % len(passives)])
        passive_slots = tuple(_passive_slot(pid, rng) for pid in pids)
        cells.append(TaskCell(
            weapon_unlock_stage_key=stage_key,
            first_weapon_id=wid,
            enemy_phase_idx=enemy_phase_idx,
            task_kind="passive_item_stage",
            build_policy="combination_slots",
            combo_key="is1_w{}_p{}".format(wid, "_".join(str(pid) for pid in pids)),
            initial_weapon_slots=((wid, _DEFAULT_WEAPON_SLOT_LEVEL),),
            initial_passive_slots=passive_slots,
            allowed_weapon_ids=(wid,),
        ))

    # 2) passive の未カバー分を追加セルで補う（max_cells の範囲で）。
    passive_index = 0
    while len(cells) < max_cells and passive_index < len(passives):
        wid = weapons[passive_index % len(weapons)]
        pid = passives[passive_index]
        cells.append(TaskCell(
            weapon_unlock_stage_key=stage_key,
            first_weapon_id=wid,
            enemy_phase_idx=enemy_phase_idx,
            task_kind="passive_item_stage",
            build_policy="combination_slots",
            combo_key=f"is1_cover_w{wid}_p{pid}",
            initial_weapon_slots=((wid, _DEFAULT_WEAPON_SLOT_LEVEL),),
            initial_passive_slots=(_passive_slot(pid, rng),),
            allowed_weapon_ids=(wid,),
        ))
        passive_index += 1

    return cells[:max_cells]
