"""Survivors content coverage の厳格な manifest 検証。

初心者向け:
C++ が出力した content schema を唯一の定義として読み、YAML に書かれた
シナリオ証跡が全コンテンツに揃っているかを安全側に倒して確認します。
"""

from __future__ import annotations

import json
import importlib.util
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from games.survivors.survivors_weapon_table import build_content_training_cells

try:
    import yaml
except ImportError:  # pragma: no cover - CLI が分かりやすいエラーを返すための境界
    yaml = None

_CANONICAL_JSON_PATH = (
    Path(__file__).resolve().parents[3]
    / "Common/src/reinbalance_survivors_contracts/canonical_json.py"
)
_CANONICAL_SPEC = importlib.util.spec_from_file_location("_survivors_canonical_json", _CANONICAL_JSON_PATH)
ensure_loader = _CANONICAL_SPEC is not None and _CANONICAL_SPEC.loader is not None
if not ensure_loader:  # pragma: no cover - repository 欠損時だけ到達する import 境界
    raise ImportError(f"cannot load canonical JSON helper: {_CANONICAL_JSON_PATH}")
_canonical_module = importlib.util.module_from_spec(_CANONICAL_SPEC)
_CANONICAL_SPEC.loader.exec_module(_canonical_module)
canonical_json_bytes = _canonical_module.canonical_json_bytes


class ContractValidationError(ValueError):
    """coverage wire contract の fail-closed 違反。

    初心者向け:
    未知の項目や欠けた証跡を通常の実行エラーと区別し、監査を必ず失敗にします。
    """


_TOP_KEYS = frozenset({"schema_version", "content", "action_time"})
_CONTENT_KEYS = frozenset({
    "max_level", "weapons", "passives", "gems", "xp_curve", "level_cadence",
    "offer", "slots", "evolutions", "chest",
})
_ACTION_KEYS = frozenset({
    "physics_dt", "decision_steps", "directions", "move_speed", "screen_displacement_per_step",
    "pause_during_level_up", "level_up_timing",
})
_ANNOTATION_TOP_KEYS = frozenset({"schema_version", "collections", "combinations", "intentional_exclusions"})
_COLLECTIONS = frozenset({"weapons", "passives", "gems", "evolutions", "enemies"})
_TRACE_KEYS = frozenset({
    "target_relevant", "implemented", "reachable", "observed", "scenario", "trained", "evaluated",
    "effect_handler", "obs_category",
})
_COMBINATION_KEYS = frozenset({"kind", "members", "scenario", "eval_assertion"})
_EXCLUSION_KEYS = frozenset({"collection", "id", "scope", "reason", "alternative_coverage"})
_REQUIRED_COMBINATIONS = frozenset({
    "pair_evolution_union", "weak_defensive_weapon", "instant_kill_resistance", "boss_interaction",
})
_ENEMY_IDS = frozenset(str(value) for value in range(11))
_GATES = ("implemented", "reachable", "observed", "trained", "evaluated")
_STARTING_EXCLUSION_IDS = frozenset({"12", "15", "27"})
_COMBINATION_EVIDENCE = {
    "pair_evolution_union": (
        frozenset({"weapon:13", "weapon:14", "weapon:28"}),
        "vandalier_consumes_partner_slot", "one_vandalier_slot_and_one_empty_slot",
    ),
    "weak_defensive_weapon": (
        frozenset({"weapon:15", "enemy:4"}),
        "laurel_shield_under_contact", "shield_absorbs_hit_and_observation_changes",
    ),
    "instant_kill_resistance": (
        frozenset({"weapon:12", "enemy:10"}),
        "pentagram_against_resistant_boss", "boss_survives_instant_kill",
    ),
    "boss_interaction": (
        frozenset({"enemy:10", "evolution:12:27"}),
        "gorgeous_moon_boss_chest", "boss_drop_and_evolution_are_observed",
    ),
}
_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
_LOGIC_PATH = _REPOSITORY_ROOT / "ReinBalance/Source/ReinBalanceLogic/Private/Survivors/SurvivorsGameLogic.cpp"
_TYPES_PATH = _REPOSITORY_ROOT / "ReinBalance/Source/ReinBalanceLogic/Public/Survivors/SurvivorsTypes.h"
_CONSTANTS_PATH = _REPOSITORY_ROOT / "ReinBalance/Source/ReinBalanceLogic/Public/Survivors/SurvivorsGameConstants.h"
_LLT_PATH = _REPOSITORY_ROOT / (
    "ReinBalance/Source/Programs/ReinBalanceLogicTests/Private/Survivors/SurvivorsWeaponTests.cpp"
)


def ensure(condition: bool, message: str) -> None:
    """契約条件を ContractValidationError として強制する。

    初心者向け:
    条件を満たさない入力を途中まで利用せず、その場で監査失敗にします。
    """
    if not condition:
        raise ContractValidationError(message)


def _exact_object(value: Any, keys: frozenset[str], label: str) -> Mapping[str, Any]:
    """object の必須 key と未知 key を同時検証する。

    初心者向け:
    項目の不足だけでなく余分な項目も拒否し、設定の書き間違いを見逃しません。
    """
    ensure(isinstance(value, Mapping), f"{label} must be an object")
    ensure(frozenset(value) == keys, f"{label} keys mismatch")
    return value


def _strict_id(value: Any, label: str) -> str:
    """文字列 ID を空文字なしで検証する。

    初心者向け:
    数値と文字列が混ざって同じ ID を別物として扱う事故を防ぎます。
    """
    ensure(isinstance(value, str) and bool(value), f"{label} must be a non-empty string")
    return value


def _positive_int(value: Any, label: str) -> int:
    """bool を除く正整数を検証する。

    初心者向け:
    レベルや枠数に小数、真偽値、ゼロ以下が入ることを防ぎます。
    """
    ensure(type(value) is int and value > 0, f"{label} must be a positive integer")
    return value


def _finite_number(value: Any, label: str, *, nonnegative: bool = False) -> float:
    """有限の数値を検証する。

    初心者向け:
    NaN や Infinity が XP 計算へ流れ込まないようにします。
    """
    ensure(type(value) in (int, float) and math.isfinite(value), f"{label} must be finite")
    ensure(not nonnegative or value >= 0, f"{label} must be non-negative")
    return float(value)


@dataclass(frozen=True)
class ContentManifest:
    """canonical schema と coverage annotation の検証済み組。

    初心者向け:
    C++ の定義値と YAML の証跡を混ぜずに保持し、公開定数は schema から計算します。
    """

    schema: Mapping[str, Any]
    annotations: Mapping[str, Any]
    ids: Mapping[str, frozenset[str]]
    max_levels: Mapping[str, Mapping[str, int]]
    evolution_pairs: tuple[tuple[str, str, str, str], ...]
    canonical_schema_bytes: bytes
    resolved_gates: Mapping[str, Mapping[str, Mapping[str, bool]]]


def _expected_scenario(collection: str, item_id: str) -> frozenset[str]:
    """content 行を LLT で実行される決定論シナリオ ID へ解決する。

    初心者向け:
    YAML に好きな名前を書いて成功扱いせず、表駆動 LLT が実際に扱う ID だけを許可します。
    """
    if collection == "weapons":
        special = {
            "12": "pentagram_acquisition_upgrade", "13": "peachone_union_prerequisite",
            "14": "ebony_union_prerequisite", "15": "laurel_acquisition_upgrade",
            "27": "gorgeous_moon_evolution", "28": "vandalier_union_slot_consumption",
        }
        return frozenset({special.get(item_id, (
            f"weapon_{item_id}_reset_step" if int(item_id) <= 15
            else f"evolution_{int(item_id) - 15}_{item_id}"
        ))})
    if collection == "passives":
        return frozenset({"stone_mask_no_combat_semantics" if item_id == "14" else f"passive_{item_id}_max_effect"})
    if collection == "gems":
        return frozenset({f"gem_{item_id}_pickup"})
    if collection == "evolutions":
        base, evolved = item_id.split(":")
        special = {"12:27": "gorgeous_moon_12_27", "13:28": "vandalier_union_13_28"}
        return frozenset({special.get(item_id, f"evolve_{base}_{evolved}")})
    return frozenset({"enemy_10_boss_reset_step" if item_id == "10" else f"enemy_{item_id}_reset_step"})


def _enum_id_names(source: str, enum_name: str) -> dict[str, str]:
    """C++ enum の明示 ID と名前を機械的に抽出する。

    初心者向け:
    Python に武器名やパッシブ名を複製せず、ゲーム本体の型宣言から対応を読み取ります。
    """
    match = re.search(rf"enum\s+class\s+{re.escape(enum_name)}\b[^{{]*{{(?P<body>.*?)}};", source, re.DOTALL)
    ensure(match is not None, f"{enum_name} enum missing")
    pairs = re.findall(r"^\s*([A-Za-z_]\w*)\s*=\s*(\d+)\s*,?", match.group("body"), re.MULTILINE)
    result = {item_id: name for name, item_id in pairs if item_id != "0"}
    ensure(len(result) == len(pairs) - sum(item_id == "0" for _, item_id in pairs), f"duplicate {enum_name} id")
    return result


def _function_body(source: str, signature: str) -> str:
    """C++ 関数の波括弧範囲を簡易 lexer で切り出す。

    初心者向け:
    関数名が残っているだけでは成功させず、その関数の中に実装があるかを調べます。
    """
    start = source.find(signature)
    ensure(start >= 0, f"production function missing: {signature}")
    brace = source.find("{", start)
    ensure(brace >= 0, f"production function body missing: {signature}")
    depth = 0
    for index in range(brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[brace + 1:index]
    raise ContractValidationError(f"unterminated production function: {signature}")


def _coverage_cells(source: str, array_name: str) -> dict[str, tuple[str, str]]:
    """LLT の機械可読 cell registry を exact-set map として解決する。

    初心者向け:
    各 content の scenario と評価名が実際に LLT の実行対象配列へ入っているか確認します。
    """
    match = re.search(
        rf"{re.escape(array_name)}\[\]\s*=\s*\{{(?P<body>.*?)\n\}};",
        source,
        re.DOTALL,
    )
    ensure(match is not None, f"LLT registry missing: {array_name}")
    entries = re.findall(
        r'\{\s*TEXT\("([^"]+)"\)\s*,\s*TEXT\("([^"]+)"\)\s*,\s*TEXT\("([^"]+)"\)\s*\}',
        match.group("body"),
    )
    result: dict[str, tuple[str, str]] = {}
    for key, scenario, assertion in entries:
        ensure(key not in result, f"duplicate LLT coverage key: {key}")
        result[key] = (scenario, assertion)
    ensure(
        re.search(rf"for\s*\([^)]*:\s*{re.escape(array_name)}\s*\)", source) is not None,
        f"LLT registry is not executed: {array_name}",
    )
    ensure("CHECK(!FString(Cell.EvalAssertionId).IsEmpty())" in source, "LLT eval assertion check missing")
    return result


def _resolve_static_evidence(
    ids: Mapping[str, frozenset[str]],
    pairs: tuple[tuple[str, str, str, str], ...],
) -> Mapping[str, Mapping[str, Mapping[str, bool]]]:
    """production・training consumer・LLT を別々に解析し5ゲート証跡を作る。

    初心者向け:
    どれか一つの許可リストから全項目を成功にせず、実装、取得、観測、訓練、評価を別々に確認します。
    """
    logic = _LOGIC_PATH.read_text(encoding="utf-8")
    types = _TYPES_PATH.read_text(encoding="utf-8")
    constants = _CONSTANTS_PATH.read_text(encoding="utf-8")
    llt = _LLT_PATH.read_text(encoding="utf-8")
    weapon_names = _enum_id_names(types, "EWeaponType")
    passive_names = _enum_id_names(types, "EPassiveItemType")
    weapon_factory = _function_body(logic, "FSurvivorsGameLogic::CreateWeaponLogic")
    choices = _function_body(logic, "FSurvivorsGameLogic::BuildLevelUpChoices")
    passive_effects = _function_body(logic, "FSurvivorsGameLogic::ComputePassiveEffects")
    observation = _function_body(logic, "FSurvivorsGameLogic::GetObservation")
    enemy_table = _function_body(logic, "FSurvivorsGameLogic::InitDefaultEnemyTable")
    gem_drop = _function_body(logic, "FSurvivorsGameLogic::DropGem")
    cells = _coverage_cells(llt, "ContentCoverageCells")
    llt_assertion_anchors = {
        "weapons": "CHECK(static_cast<int32>(Logic.GetWeaponSlot(0).Type) == WeaponId)",
        "passives": "CHECK(static_cast<int32>(Logic.GetPassiveSlot(0).Type) == PassiveId)",
        "gems": "CHECK(static_cast<int32>(Logic.GetItemGemType(0)) == GemIndex)",
        "evolutions": "CHECK(Logic.GetWeaponSlot(0).Type == Rule.EvolvedWeapon)",
        "enemies": "CHECK(Logic.GetEnemyType(0) == EnemyId)",
    }
    for collection, anchor in llt_assertion_anchors.items():
        ensure(anchor in llt, f"{collection} table-driven eval assertion missing")
    expected_keys = {
        f"{prefix}:{item_id}"
        for collection, prefix in (
            ("weapons", "weapon"), ("passives", "passive"), ("gems", "gem"),
            ("evolutions", "evolution"), ("enemies", "enemy"),
        )
        for item_id in ids[collection]
    }
    ensure(frozenset(cells) == frozenset(expected_keys), "LLT content coverage keys mismatch")
    training = build_content_training_cells(ids)
    ensure(frozenset(training) == frozenset(expected_keys), "training content keys mismatch")
    enemy_rows = re.findall(r'\{TEXT\("[^"]+"\),[^{}]+\}', enemy_table)
    evolution_names = {
        f"{base}:{evolved}": (weapon_names[base], weapon_names[evolved])
        for base, evolved, _, _ in pairs
    }
    resolved: dict[str, dict[str, dict[str, bool]]] = {}
    for collection in _COLLECTIONS:
        prefix = {"weapons": "weapon", "passives": "passive", "gems": "gem",
                  "evolutions": "evolution", "enemies": "enemy"}[collection]
        resolved[collection] = {}
        for item_id in ids[collection]:
            key = f"{prefix}:{item_id}"
            if collection == "weapons":
                name = weapon_names[item_id]
                implemented = re.search(rf"case\s+EWeaponType::{re.escape(name)}\s*:", weapon_factory) is not None
                reachable = (
                    re.search(rf"EWeaponType::{re.escape(name)}\b", choices) is not None
                    or re.search(rf"EWeaponType::{re.escape(name)}\b", constants) is not None
                )
                observed = "WeaponSlots" in observation and 'TEXT("weapon_slots")' in logic
            elif collection == "passives":
                name = passive_names[item_id]
                implemented = (
                    re.search(rf"case\s+EPassiveItemType::{re.escape(name)}\s*:", passive_effects) is not None
                    or name in {"Clover", "StoneMask"}
                )
                reachable = re.search(rf"EPassiveItemType::{re.escape(name)}\b", choices) is not None
                observed = 'TEXT("passive_slots")' in llt and "PassiveSlots" in observation
            elif collection == "gems":
                implemented = "GemXPValues" in gem_drop and "Gems.Add" in gem_drop
                reachable = "DropGem(" in logic
                observed = "Gems" in observation and f"gem_{item_id}_pickup" in llt
            elif collection == "evolutions":
                base_name, evolved_name = evolution_names[item_id]
                implemented = all(name in constants for name in (base_name, evolved_name))
                reachable = "WeaponEvolve" in choices and all(name in constants for name in (base_name, evolved_name))
                observed = "WeaponSlots" in observation
            else:
                implemented = int(item_id) < len(enemy_rows)
                reachable = implemented and ("SpawnEnemy" in logic or "SpawnOneEnemy" in logic)
                observed = 'TEXT("enemy_type")' in llt and "Enemies" in observation
            scenario, assertion = cells[key]
            resolved[collection][item_id] = {
                "implemented": implemented,
                "reachable": reachable,
                "observed": observed,
                "trained": training.get(key) == scenario,
                "evaluated": bool(assertion) and key in cells,
            }
    return resolved


def load_content_schema(value: Mapping[str, Any]) -> tuple[dict[str, frozenset[str]], dict[str, dict[str, int]], tuple[tuple[str, str, str, str], ...]]:
    """canonical C++ export を全 collection で fail-closed 検証する。

    初心者向け:
    ID、最大レベル、進化条件、XP、offer、slot、action を利用前にまとめて確認します。
    """
    top = _exact_object(value, _TOP_KEYS, "schema")
    ensure(top["schema_version"] == "survivors.content_schema.v1", "unsupported schema_version")
    content = _exact_object(top["content"], _CONTENT_KEYS, "content")
    ids: dict[str, frozenset[str]] = {}
    levels: dict[str, dict[str, int]] = {}
    for collection in ("weapons", "passives"):
        rows = content[collection]
        ensure(isinstance(rows, list) and rows, f"{collection} must be a non-empty array")
        found: set[str] = set()
        level_map: dict[str, int] = {}
        for row in rows:
            item = _exact_object(row, frozenset({"id", "max_level"}), f"{collection} row")
            item_id = _strict_id(item["id"], f"{collection}.id")
            ensure(item_id not in found, f"duplicate {collection} id: {item_id}")
            found.add(item_id)
            level_map[item_id] = _positive_int(item["max_level"], f"{collection}.max_level")
        ids[collection] = frozenset(found)
        levels[collection] = level_map
    gems = content["gems"]
    ensure(isinstance(gems, list) and len(gems) == 3, "gems must have exactly three rows")
    gem_ids: set[str] = set()
    for row in gems:
        item = _exact_object(row, frozenset({"id", "xp"}), "gems row")
        item_id = _strict_id(item["id"], "gems.id")
        ensure(item_id not in gem_ids, f"duplicate gems id: {item_id}")
        gem_ids.add(item_id)
        _finite_number(item["xp"], "gems.xp", nonnegative=True)
    ensure(gem_ids == {"blue", "green", "red"}, "gem ids mismatch")
    ids["gems"] = frozenset(gem_ids)
    ids["enemies"] = _ENEMY_IDS
    xp_curve = content["xp_curve"]
    max_level = _positive_int(content["max_level"], "content.max_level")
    ensure(isinstance(xp_curve, list) and len(xp_curve) == max_level, "xp_curve length mismatch")
    for item in xp_curve:
        _finite_number(item, "xp_curve entry", nonnegative=True)
    ensure(content["level_cadence"] == "xp_threshold", "unknown level cadence")
    offer = _exact_object(content["offer"], frozenset({"count", "fallback"}), "offer")
    _positive_int(offer["count"], "offer.count")
    ensure(offer["fallback"] == "none_when_pool_empty", "unknown offer fallback")
    slots = _exact_object(content["slots"], frozenset({"weapon", "passive"}), "slots")
    _positive_int(slots["weapon"], "slots.weapon")
    _positive_int(slots["passive"], "slots.passive")
    chest = _exact_object(content["chest"], frozenset({"boss_drop", "evolution_enabled_by_config"}), "chest")
    ensure(type(chest["boss_drop"]) is bool and type(chest["evolution_enabled_by_config"]) is bool, "chest flags must be bool")
    pairs: list[tuple[str, str, str, str]] = []
    seen_pairs: set[tuple[str, str, str, str]] = set()
    evolutions = content["evolutions"]
    ensure(isinstance(evolutions, list) and evolutions, "evolutions must be a non-empty array")
    for row in evolutions:
        item = _exact_object(
            row, frozenset({"base_id", "evolved_id", "passive_id", "union_partner_id"}), "evolution row",
        )
        pair = tuple(_strict_id(item[key], f"evolution.{key}") for key in (
            "base_id", "evolved_id", "passive_id", "union_partner_id",
        ))
        ensure(pair not in seen_pairs, "duplicate evolution row")
        seen_pairs.add(pair)
        ensure(pair[0] in ids["weapons"] and pair[1] in ids["weapons"], "unknown evolution weapon")
        ensure(pair[2] == "0" or pair[2] in ids["passives"], "unknown evolution passive")
        ensure(pair[3] == "0" or pair[3] in ids["weapons"], "unknown union partner")
        pairs.append(pair)
    ids["evolutions"] = frozenset(f"{base}:{evolved}" for base, evolved, _, _ in pairs)
    action = _exact_object(top["action_time"], _ACTION_KEYS, "action_time")
    ensure(_finite_number(action["physics_dt"], "physics_dt") > 0, "physics_dt must be positive")
    _positive_int(action["decision_steps"], "decision_steps")
    ensure(_finite_number(action["move_speed"], "move_speed") > 0, "move_speed must be positive")
    ensure(
        _finite_number(action["screen_displacement_per_step"], "screen_displacement_per_step") > 0,
        "screen_displacement_per_step must be positive",
    )
    ensure(type(action["pause_during_level_up"]) is bool, "pause_during_level_up must be bool")
    ensure(action["level_up_timing"] == "same_physics_step", "unknown level_up_timing")
    directions = action["directions"]
    ensure(isinstance(directions, list) and len(directions) == 9, "directions mismatch")
    direction_ids: set[int] = set()
    for row in directions:
        item = _exact_object(row, frozenset({"id", "x", "y"}), "direction row")
        ensure(type(item["id"]) is int and item["id"] in range(9) and item["id"] not in direction_ids, "direction ids mismatch")
        direction_ids.add(item["id"])
        x = _finite_number(item["x"], "direction.x")
        y = _finite_number(item["y"], "direction.y")
        ensure(-1 <= x <= 1 and -1 <= y <= 1, "direction component out of range")
    return ids, levels, tuple(pairs)


def load_annotations(path: Path) -> Mapping[str, Any]:
    """YAML annotation を安全に読み込む。

    初心者向け:
    YAML ライブラリがない場合や、object でない設定を明確な監査失敗にします。
    """
    ensure(yaml is not None, "PyYAML is required to read annotations")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    ensure(isinstance(loaded, Mapping), "annotations must be an object")
    return loaded


def build_manifest(schema: Mapping[str, Any], annotations: Mapping[str, Any]) -> ContentManifest:
    """schema と annotation を exact-set で結合し coverage 不変条件を検証する。

    初心者向け:
    全行に実装、取得、観測、訓練、評価の証拠が揃うまで manifest を作りません。
    """
    ids, levels, pairs = load_content_schema(schema)
    static_evidence = _resolve_static_evidence(ids, pairs)
    wire = _exact_object(annotations, _ANNOTATION_TOP_KEYS, "annotations")
    ensure(wire["schema_version"] == "survivors.content_annotations.v1", "unsupported annotation schema")
    collections = _exact_object(wire["collections"], _COLLECTIONS, "annotation collections")
    resolved_gates: dict[str, dict[str, dict[str, bool]]] = {}
    for collection in _COLLECTIONS:
        rows = collections[collection]
        ensure(isinstance(rows, Mapping), f"{collection} annotations must be an object")
        ensure(all(isinstance(key, str) for key in rows), f"{collection} annotation ids must be strings")
        ensure(frozenset(str(key) for key in rows) == ids[collection], f"{collection} annotation ids mismatch")
        resolved_gates[collection] = {}
        for item_id, raw in rows.items():
            row = _exact_object(raw, _TRACE_KEYS, f"{collection}:{item_id}")
            ensure(type(row["target_relevant"]) is bool, "target_relevant must be bool")
            for gate in _GATES:
                ensure(type(row[gate]) is bool, f"{collection}:{item_id}.{gate} must be bool")
            for key in ("scenario", "effect_handler", "obs_category"):
                ensure(isinstance(row[key], str) and bool(row[key].strip()), f"{collection}:{item_id}.{key} required")
            ensure(row["target_relevant"], f"{collection}:{item_id} cannot be excluded from full coverage")
            expected_scenarios = _expected_scenario(collection, item_id)
            ensure(row["scenario"] in expected_scenarios, f"{collection}:{item_id} unresolved scenario")
            prefix = {"weapons": "weapon", "passives": "passive", "gems": "gem",
                      "evolutions": "evolution", "enemies": "enemy"}[collection]
            registry_scenario = build_content_training_cells(ids)[f"{prefix}:{item_id}"]
            ensure(row["scenario"] == registry_scenario, f"{collection}:{item_id} training scenario mismatch")
            resolved = dict(static_evidence[collection][item_id])
            ensure(all(resolved.values()), f"{collection}:{item_id} unresolved static evidence")
            for gate in _GATES:
                ensure(row[gate] == resolved[gate], f"{collection}:{item_id}.{gate} contradicts resolved evidence")
            resolved_gates[collection][item_id] = resolved
    combinations = wire["combinations"]
    combination_cells = _coverage_cells(
        _LLT_PATH.read_text(encoding="utf-8"), "CombinationCoverageCells",
    )
    llt_source = _LLT_PATH.read_text(encoding="utf-8")
    combination_sections = {
        "pair_evolution_union": "pair evolution union",
        "weak_defensive_weapon": "weak defensive weapon",
        "instant_kill_resistance": "instant kill resistance",
        "boss_interaction": "boss interaction",
    }
    for kind, section_name in combination_sections.items():
        start = llt_source.find(f'SECTION("{section_name}")')
        ensure(start >= 0, f"combination LLT section missing: {kind}")
        next_start = llt_source.find("SECTION(", start + 1)
        section = llt_source[start:next_start if next_start >= 0 else len(llt_source)]
        ensure("CHECK(" in section, f"combination eval assertion missing: {kind}")
    ensure(
        frozenset(combination_cells)
        == frozenset(f"combination:{kind}" for kind in _REQUIRED_COMBINATIONS),
        "LLT combination coverage keys mismatch",
    )
    ensure(isinstance(combinations, list), "combinations must be an array")
    kinds: set[str] = set()
    for raw in combinations:
        row = _exact_object(raw, _COMBINATION_KEYS, "combination")
        ensure(isinstance(row["kind"], str) and row["kind"] not in kinds, "duplicate/invalid combination kind")
        ensure(row["kind"] in _REQUIRED_COMBINATIONS, "unknown combination kind")
        kinds.add(row["kind"])
        ensure(isinstance(row["members"], list) and row["members"], "combination members required")
        ensure(all(isinstance(item, str) and re.fullmatch(r"(weapon|passive|gem|evolution|enemy):.+", item)
                   for item in row["members"]), "combination member invalid")
        ensure(len(row["members"]) == len(set(row["members"])), "duplicate combination member")
        for member in row["members"]:
            member_collection, member_id = member.split(":", 1)
            collection_name = {
                "weapon": "weapons", "passive": "passives", "gem": "gems",
                "evolution": "evolutions", "enemy": "enemies",
            }[member_collection]
            ensure(collection_name in ids and member_id in ids[collection_name], "unknown combination member")
        ensure(isinstance(row["scenario"], str) and row["scenario"], "combination scenario required")
        ensure(isinstance(row["eval_assertion"], str) and row["eval_assertion"], "combination eval required")
        expected_members, expected_scenario, expected_eval = _COMBINATION_EVIDENCE[row["kind"]]
        ensure(frozenset(row["members"]) == expected_members, "combination members evidence mismatch")
        registry_scenario, registry_eval = combination_cells[f"combination:{row['kind']}"]
        ensure(row["scenario"] == expected_scenario == registry_scenario, "unresolved combination scenario")
        ensure(row["eval_assertion"] == expected_eval == registry_eval, "unresolved combination eval assertion")
    ensure(kinds == _REQUIRED_COMBINATIONS, "combination kinds mismatch")
    exclusions = wire["intentional_exclusions"]
    ensure(isinstance(exclusions, list), "intentional_exclusions must be an array")
    exclusion_ids: set[str] = set()
    for raw in exclusions:
        row = _exact_object(raw, _EXCLUSION_KEYS, "intentional exclusion")
        ensure(all(isinstance(row[key], str) for key in _EXCLUSION_KEYS),
               "intentional exclusion fields must be strings")
        ensure(row["scope"] == "starting_weapon", "only starting_weapon exclusion is supported")
        ensure(row["collection"] == "weapons" and row["id"] in ids["weapons"], "unknown exclusion target")
        ensure(row["id"] not in exclusion_ids, "duplicate intentional exclusion")
        exclusion_ids.add(row["id"])
        ensure(isinstance(row["reason"], str) and row["reason"], "exclusion reason required")
        ensure(isinstance(row["alternative_coverage"], str) and row["alternative_coverage"], "alternative coverage required")
        covered = collections["weapons"][row["id"]]
        ensure(row["alternative_coverage"] in _expected_scenario("weapons", row["id"]), "unresolved alternative coverage")
        ensure(all(resolved_gates["weapons"][row["id"]][gate] for gate in _GATES),
               "starting exclusion cannot remove coverage")
    ensure(frozenset(exclusion_ids) == _STARTING_EXCLUSION_IDS, "starting exclusion ids mismatch")
    return ContentManifest(schema, annotations, ids, levels, pairs, canonical_json_bytes(schema), resolved_gates)


def audit_manifest(manifest: ContentManifest) -> dict[str, Any]:
    """5 gates の blocking 件数と exclusion 情報を生成する。

    初心者向け:
    各ゲートで false の行を数え、ひとつでもあれば終了コードを失敗にできます。
    """
    gates = {
        gate: {
            "blocking": sum(
                1 for collection, rows in manifest.resolved_gates.items() for item_id, row in rows.items()
                if manifest.annotations["collections"][collection][item_id]["target_relevant"] and not row[gate]
            )
        }
        for gate in _GATES
    }
    return {
        "schema_version": "survivors.content_audit.v1",
        "gates": gates,
        "blocking": sum(value["blocking"] for value in gates.values()),
        "intentional_exclusions": list(manifest.annotations["intentional_exclusions"]),
    }


def compatibility_constants(manifest: ContentManifest) -> Mapping[str, Any]:
    """generated schema 由来の互換定義値を公開する。

    初心者向け:
    呼び出し側が ID、最大レベル、進化組を必要とするとき、手書き表ではなく検証済み schema から受け取ります。
    """
    content = manifest.schema["content"]
    return {
        "weapon_ids": manifest.ids["weapons"],
        "passive_ids": manifest.ids["passives"],
        "gem_ids": manifest.ids["gems"],
        "enemy_ids": manifest.ids["enemies"],
        "weapon_max_levels": dict(manifest.max_levels["weapons"]),
        "passive_max_levels": dict(manifest.max_levels["passives"]),
        "evolution_pairs": manifest.evolution_pairs,
        "gem_xp": {row["id"]: row["xp"] for row in content["gems"]},
        "max_level": content["max_level"],
        "xp_curve": tuple(content["xp_curve"]),
        "level_cadence": content["level_cadence"],
        "offer": dict(content["offer"]),
        "slots": dict(content["slots"]),
        "chest": dict(content["chest"]),
        "action_time": manifest.schema["action_time"],
    }
