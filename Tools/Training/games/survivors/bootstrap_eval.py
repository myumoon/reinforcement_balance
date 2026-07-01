"""Bootstrap cell の deterministic eval 用 cell spec parser と eval params builder。

KingBible solo phase2 / Garlic maintenance phase2 のような bootstrap cell を
固定評価するための薄いヘルパー。既存の武器パラメータ／敵パラメータ生成を再利用する。

cell spec 例:
    "king_bible:solo_bootstrap:2"
    "garlic:maintenance:2"
    "garlic:integration:2"

出力 params は UE5 /params の key（MinActiveEnemies など）に変換済みの full dict。
env_method("set_params", **params) にそのまま渡せる。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from games.survivors.param_applier import ParamApplier
from games.survivors.survivors_curriculum import PHASES, get_enemy_params_for_phase
from games.survivors.survivors_weapon_table import (
    WEAPON_UNLOCK_ORDER,
    WeaponEntry,
    build_weapon_params_for_cell,
)

# maintenance / integration cell で使う build_policy
_TASK_KIND_BUILD_POLICY: dict[str, str] = {
    "solo_bootstrap": "target_only",
    "integration": "target_only",
    "maintenance": "target_plus_anchor_if_unlocked",
    "wave_main": "target_plus_anchor",
}

_KNOWN_TASK_KINDS = tuple(_TASK_KIND_BUILD_POLICY.keys())


def _weapon_key_to_id(weapon_unlock_order: list[WeaponEntry]) -> dict[str, int]:
    return {e.key: e.weapon_id for e in weapon_unlock_order}


def _weapon_id_to_entry(weapon_unlock_order: list[WeaponEntry]) -> dict[int, WeaponEntry]:
    return {e.weapon_id: e for e in weapon_unlock_order}


@dataclass(frozen=True)
class BootstrapEvalCell:
    """parse 済みの bootstrap eval cell 定義。"""

    spec: str
    weapon_id: int
    weapon_key: str
    task_kind: str
    enemy_phase_idx: int


def parse_cell_spec(
    spec: str,
    *,
    weapon_unlock_order: list[WeaponEntry] = WEAPON_UNLOCK_ORDER,
) -> BootstrapEvalCell:
    """cell spec 文字列を parse する。

    形式: "<weapon_key>:<task_kind>:<enemy_phase_idx>"
    例:   "king_bible:solo_bootstrap:2"

    unknown weapon / task_kind / phase は明確な ValueError にする。
    """
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError(
            f"cell spec の形式が不正です: {spec!r}。"
            f"想定形式: '<weapon_key>:<task_kind>:<enemy_phase_idx>' "
            f"(例: 'king_bible:solo_bootstrap:2')"
        )
    weapon_key_raw, task_kind_raw, phase_raw = (p.strip() for p in parts)

    weapon_key = weapon_key_raw.lower()
    key_to_id = _weapon_key_to_id(weapon_unlock_order)
    if weapon_key not in key_to_id:
        raise ValueError(
            f"未知の weapon key: {weapon_key_raw!r}。"
            f"利用可能: {sorted(key_to_id.keys())}"
        )
    weapon_id = key_to_id[weapon_key]

    task_kind = task_kind_raw.lower()
    if task_kind not in _KNOWN_TASK_KINDS:
        raise ValueError(
            f"未知の task_kind: {task_kind_raw!r}。"
            f"利用可能: {list(_KNOWN_TASK_KINDS)}"
        )

    try:
        enemy_phase_idx = int(phase_raw)
    except ValueError as exc:
        raise ValueError(f"enemy_phase_idx が整数ではありません: {phase_raw!r}") from exc
    if not (0 <= enemy_phase_idx < len(PHASES)):
        raise ValueError(
            f"enemy_phase_idx が範囲外です: {enemy_phase_idx}。"
            f"有効範囲: 0..{len(PHASES) - 1}"
        )

    return BootstrapEvalCell(
        spec=spec,
        weapon_id=weapon_id,
        weapon_key=weapon_key,
        task_kind=task_kind,
        enemy_phase_idx=enemy_phase_idx,
    )


def build_eval_params(
    cell: BootstrapEvalCell,
    *,
    stage_key: str | None = None,
    item_stage_key: str = "IS0",
    weapon_unlock_order: list[WeaponEntry] = WEAPON_UNLOCK_ORDER,
) -> dict:
    """eval cell から UE5 /params の full dict（UE key 形式）を組み立てる。

    weapon params（allowed_weapon_types 等）と enemy phase params（MinActiveEnemies 等）を
    マージし、敵 params の内部 key は ParamApplier._KEY_MAP で UE key に変換する。

    Args:
        stage_key: 武器プールのアンロック段階。None の場合は対象武器がちょうど
                   アンロックされる段階（entry.unlock_stage_key）を使う。
    """
    entry = _weapon_id_to_entry(weapon_unlock_order).get(cell.weapon_id)
    if entry is None:
        raise ValueError(f"weapon_id={cell.weapon_id} が weapon_unlock_order に存在しません。")

    if stage_key is None:
        stage_key = entry.unlock_stage_key

    build_policy = _TASK_KIND_BUILD_POLICY[cell.task_kind]

    weapon_params = build_weapon_params_for_cell(
        first_weapon_id=cell.weapon_id,
        max_unlocked_stage_key=stage_key,
        item_stage_key=item_stage_key,
        pool_policy=build_policy,
        weapon_unlock_order=weapon_unlock_order,
    )

    # 敵 params は内部 key（min_enemies など）で返るため UE key に変換する
    enemy_params_internal = get_enemy_params_for_phase(cell.enemy_phase_idx)
    enemy_params_ue = {
        ParamApplier._KEY_MAP.get(k, k): v for k, v in enemy_params_internal.items()
    }

    return {**weapon_params, **enemy_params_ue}


def summarize_eval_results(
    cell_spec: str,
    episode_results: list[dict],
    *,
    deterministic: bool,
    model_path: str,
    global_timestep: int | None = None,
    short_episode_steps: int = 600,
) -> dict:
    """episode_results から eval cell の summary dict を組み立てる。

    episode 数不足（0 件）でも落ちないようにする。
    """
    scores = [float(r.get("active_score", 0.0)) for r in episode_results]
    lengths = [float(r.get("ep_length", 0.0)) for r in episode_results]

    def _p(values: list[float], q: float) -> float:
        if not values:
            return 0.0
        return float(np.percentile(values, q))

    def _mean(values: list[float]) -> float:
        if not values:
            return 0.0
        return float(np.mean(values))

    termination_counts: dict[str, int] = {}
    for r in episode_results:
        # terminated=1 は死亡（terminated）、0 は truncated（time limit）
        term = int(r.get("terminated", 0))
        key = "terminated" if term else "truncated"
        termination_counts[key] = termination_counts.get(key, 0) + 1

    short_rate = (
        sum(1 for l in lengths if l < short_episode_steps) / len(lengths)
        if lengths else 0.0
    )

    return {
        "cell": cell_spec,
        "episodes": len(episode_results),
        "deterministic": deterministic,
        "active_score_p10": _p(scores, 10),
        "active_score_p50": _p(scores, 50),
        "active_score_mean": _mean(scores),
        "episode_length_p10": _p(lengths, 10),
        "episode_length_mean": _mean(lengths),
        "short_episode_rate": short_rate,
        "termination_counts": termination_counts,
        "model_path": model_path,
        "global_timestep": global_timestep,
    }
