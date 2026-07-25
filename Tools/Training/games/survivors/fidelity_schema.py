"""C++ `/content_schema` export に監査 annotation だけを付与する。

武器・item・level・進化 pair は入力 JSON から読み取り、Python 側へ同じ集合を
ハードコードしません。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class AnnotatedFidelitySchema:
    """C++ schema と target relevance の不変な組。

    content identity を改変せず保持し、監査固有の注釈を別 object に分離します。
    """
    cpp_schema: Mapping[str, Any]
    annotations: Mapping[str, Mapping[str, Mapping[str, Any]]]


def _exact_object(value: Any, keys: set[str], label: str) -> Mapping[str, Any]:
    """object の必須 key と未知 key を同時に検証する。

    一部だけ読める壊れた C++ export を注釈済み schema として扱いません。
    """
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{label} keys mismatch")
    return value


def _finite_number(value: Any, label: str, *, positive: bool = False) -> float:
    """bool を除く有限数を検証する。

    JSON の NaN/Infinity や数値 field への真偽値混入を監査前に拒否します。
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite numeric")
    if positive and value <= 0:
        raise ValueError(f"{label} must be positive")
    return float(value)


def _positive_int(value: Any, label: str) -> int:
    """正の strict integer を検証する。

    level、slot、step 数へ bool や小数が入ることを防ぎます。
    """
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _validate_cpp_export(export: Mapping[str, Any]) -> None:
    """C++ content/action export の全必須構造を fail-closed 検証する。

    annotation が利用する ID だけでなく、level、XP、進化 pair、時間/action 契約まで
    一つの入力境界で検証します。
    """
    top = _exact_object(export, {"schema_version", "content", "action_time"}, "content schema top-level")
    if top["schema_version"] != "survivors.content_schema.v1":
        raise ValueError("unsupported content schema")
    content = _exact_object(
        top["content"],
        {"weapons", "passives", "gems", "xp_curve", "level_cadence", "offer", "slots", "evolutions", "chest"},
        "content",
    )
    for collection in ("weapons", "passives"):
        rows = content[collection]
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"{collection} must be a non-empty array")
        seen: set[str] = set()
        for row in rows:
            item = _exact_object(row, {"id", "max_level"}, f"{collection} row")
            if not isinstance(item["id"], str) or not item["id"] or item["id"] in seen:
                raise ValueError(f"{collection} ids must be non-empty and unique")
            seen.add(item["id"])
            _positive_int(item["max_level"], f"{collection}.max_level")
    gems = content["gems"]
    if not isinstance(gems, list) or len(gems) != 3:
        raise ValueError("gems must contain exactly three rows")
    gem_ids: set[str] = set()
    for row in gems:
        item = _exact_object(row, {"id", "xp"}, "gems row")
        if not isinstance(item["id"], str) or not item["id"] or item["id"] in gem_ids:
            raise ValueError("gem ids must be non-empty and unique")
        gem_ids.add(item["id"])
        _finite_number(item["xp"], "gems.xp", positive=True)
    if gem_ids != {"blue", "green", "red"}:
        raise ValueError("gem ids must be blue, green, and red")
    xp_curve = content["xp_curve"]
    if not isinstance(xp_curve, list) or not xp_curve:
        raise ValueError("xp_curve must be a non-empty array")
    for value in xp_curve:
        _finite_number(value, "xp_curve entry", positive=True)
    if not isinstance(content["level_cadence"], str) or not content["level_cadence"]:
        raise ValueError("level_cadence must be non-empty")
    offer = _exact_object(content["offer"], {"count", "fallback"}, "offer")
    _positive_int(offer["count"], "offer.count")
    if not isinstance(offer["fallback"], str) or not offer["fallback"]:
        raise ValueError("offer.fallback must be non-empty")
    slots = _exact_object(content["slots"], {"weapon", "passive"}, "slots")
    _positive_int(slots["weapon"], "slots.weapon")
    _positive_int(slots["passive"], "slots.passive")
    evolutions = content["evolutions"]
    if not isinstance(evolutions, list):
        raise ValueError("evolutions must be an array")
    evolution_rows: set[tuple[str, str, str, str]] = set()
    evolution_keys = {"base_id", "evolved_id", "passive_id", "union_partner_id"}
    for row in evolutions:
        item = _exact_object(row, evolution_keys, "evolution row")
        if any(not isinstance(item[key], str) or not item[key] for key in evolution_keys):
            raise ValueError("evolution ids must be non-empty strings")
        identity = tuple(item[key] for key in sorted(evolution_keys))
        if identity in evolution_rows:
            raise ValueError("duplicate evolution row")
        evolution_rows.add(identity)
    chest = _exact_object(content["chest"], {"boss_drop", "evolution_enabled_by_config"}, "chest")
    if type(chest["boss_drop"]) is not bool or type(chest["evolution_enabled_by_config"]) is not bool:
        raise ValueError("chest flags must be bool")

    action = _exact_object(
        top["action_time"],
        {"physics_dt", "decision_steps", "directions", "move_speed", "screen_displacement_per_step",
         "pause_during_level_up", "level_up_timing"},
        "action_time",
    )
    _finite_number(action["physics_dt"], "physics_dt", positive=True)
    _positive_int(action["decision_steps"], "decision_steps")
    _finite_number(action["move_speed"], "move_speed", positive=True)
    _finite_number(action["screen_displacement_per_step"], "screen_displacement_per_step", positive=True)
    if type(action["pause_during_level_up"]) is not bool:
        raise ValueError("pause_during_level_up must be bool")
    if not isinstance(action["level_up_timing"], str) or not action["level_up_timing"]:
        raise ValueError("level_up_timing must be non-empty")
    directions = action["directions"]
    if not isinstance(directions, list) or len(directions) != 9:
        raise ValueError("directions must contain exactly nine rows")
    direction_ids: set[int] = set()
    for row in directions:
        item = _exact_object(row, {"id", "x", "y"}, "direction row")
        if type(item["id"]) is not int or item["id"] not in range(9) or item["id"] in direction_ids:
            raise ValueError("direction ids must be unique integers 0..8")
        direction_ids.add(item["id"])
        _finite_number(item["x"], "direction.x")
        _finite_number(item["y"], "direction.y")


def annotate_cpp_content_schema(export: Mapping[str, Any], annotations: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> AnnotatedFidelitySchema:
    """C++ export の id 集合に対する annotation を厳格に付与する。

    annotation に未知 id があれば拒否し、ids/levels/pairs を Python から生成しません。
    """
    if not isinstance(export, Mapping):
        raise ValueError("content schema must be an object")
    _validate_cpp_export(export)
    content = export["content"]
    ids: dict[str, set[str]] = {}
    for collection in ("weapons", "passives", "gems"):
        values = content.get(collection)
        collection_ids: set[str] = set()
        for row in values:
            if row["id"] in collection_ids:
                raise ValueError(f"duplicate {collection} id: {row['id']}")
            collection_ids.add(row["id"])
        ids[collection] = collection_ids
    if not isinstance(annotations, Mapping) or not set(annotations) <= set(ids):
        raise ValueError("annotations contain unknown C++ content collection")
    checked: dict[str, dict[str, dict[str, Any]]] = {}
    for collection, collection_annotations in annotations.items():
        if not isinstance(collection_annotations, Mapping) or not set(collection_annotations) <= ids[collection]:
            raise ValueError(f"annotations contain unknown C++ {collection} ids")
        checked[collection] = {}
        for key, value in collection_annotations.items():
            if not isinstance(value, Mapping) or set(value) - {"target_relevant", "note"}:
                raise ValueError(f"invalid annotation for {collection}:{key}")
            if type(value.get("target_relevant")) is not bool:
                raise ValueError(f"target_relevant must be bool for {collection}:{key}")
            checked[collection][key] = dict(value)
    return AnnotatedFidelitySchema(dict(export), checked)
