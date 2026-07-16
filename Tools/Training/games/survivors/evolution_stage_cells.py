"""IS2 evolution stage セル生成。

Phase 05 (Evolution Stage) 用のタスクセルを構築する。IS2（passive 有効 /
evolution 有効）で `survivors_vs_spec.EVOLUTION_TABLE` を走査し、通常進化は
base weapon + required passive(max level) + evolved weapon の coverage cell、
Vandalier は Peachone + Ebony Wings の union coverage cell として決定論的に
combination_slots セルを生成する。
"""
from __future__ import annotations

import random

from games.survivors.modules.task_cell_sampler_module import TaskCell
from games.survivors.survivors_vs_spec import EVOLUTION_TABLE, PASSIVE_MAX_LEVEL, PassiveItemType
from games.survivors.survivors_weapon_curriculum import WeaponType
from games.survivors.survivors_weapon_table import WeaponEntry, get_unlocked_startable_weapon_ids


def build_evolution_stage_cells(
    stage_key: str,
    enemy_phase_idx: int,
    max_cells: int,
    seed: int,
    weapon_unlock_order: list[WeaponEntry],
) -> list[TaskCell]:
    rng = random.Random(seed)
    unlocked = set(get_unlocked_startable_weapon_ids(stage_key, weapon_unlock_order))
    cells: list[TaskCell] = []

    for row in EVOLUTION_TABLE:
        base_id = int(row["base"])
        passive_id = int(row["passive"])
        evolved_id = int(row["evolved"])
        if base_id not in unlocked:
            continue
        if passive_id == PassiveItemType.NONE:
            union_weapon = row.get("union_weapon")
            if union_weapon is None:
                continue
            union_id = int(union_weapon)
            if union_id not in unlocked:
                continue
            combo_key = (
                "is2_union_w13_w14_to_vandalier"
                if evolved_id == WeaponType.VANDALIER
                else f"is2_union_w{base_id}_w{union_id}_to_w{evolved_id}"
            )
            cells.append(TaskCell(
                weapon_unlock_stage_key=stage_key,
                first_weapon_id=base_id,
                enemy_phase_idx=enemy_phase_idx,
                task_kind="evolution_stage",
                build_policy="combination_slots",
                combo_key=combo_key,
                initial_weapon_slots=((base_id, 8), (union_id, 8)),
                initial_passive_slots=(),
                allowed_weapon_ids=(base_id, union_id),
            ))
            if len(cells) >= max_cells:
                break
            continue
        passive_level = int(PASSIVE_MAX_LEVEL[passive_id])
        weapon_level = rng.randint(6, 8)
        cells.append(TaskCell(
            weapon_unlock_stage_key=stage_key,
            first_weapon_id=base_id,
            enemy_phase_idx=enemy_phase_idx,
            task_kind="evolution_stage",
            build_policy="combination_slots",
            combo_key=f"is2_w{base_id}_p{passive_id}",
            initial_weapon_slots=((base_id, weapon_level),),
            initial_passive_slots=((passive_id, passive_level),),
            allowed_weapon_ids=(base_id,),
        ))
        if len(cells) >= max_cells:
            break

    return cells
