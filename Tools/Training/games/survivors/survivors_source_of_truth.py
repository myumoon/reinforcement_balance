"""Source-of-truth extraction for Survivors EUREKA prompts."""

from __future__ import annotations

import datetime as _dt
import json
import re
from pathlib import Path
from typing import Any

from common.obs_schema import fetch_obs_schema


def _read(repo_root: Path, rel_path: str) -> str:
    return (repo_root / rel_path).read_text(encoding="utf-8")


def _extract_number(text: str, name: str) -> float | None:
    match = re.search(
        rf"\b(?:float|double)\s+{re.escape(name)}\s*=\s*([0-9]+(?:\.[0-9]*)?)f?\b",
        text,
    )
    return float(match.group(1)) if match else None


def _extract_const_number(text: str, name: str) -> float | int | None:
    match = re.search(
        rf"\bstatic\s+constexpr\s+(?:int32|float|double)\s+{re.escape(name)}\s*=\s*"
        rf"([0-9]+(?:\.[0-9]*)?)f?\b",
        text,
    )
    if not match:
        return None
    raw = match.group(1)
    return float(raw) if "." in raw else int(raw)


def _extract_inline_array(text: str, name: str) -> str | None:
    match = re.search(
        rf"\binline\s+constexpr\s+[^\n]+?\s+{re.escape(name)}\s*\[[^\]]*\]\s*=\s*\{{"
        rf"(?P<body>.*?)\s*\}};",
        text,
        flags=re.DOTALL,
    )
    return match.group(0).strip() if match else None


def _extract_garlic_table(text: str) -> list[dict[str, float]]:
    table = _extract_inline_array(text, "GarlicTable")
    if not table:
        return []
    rows = []
    for damage, interval, radius in re.findall(
        r"\{\s*([0-9]+(?:\.[0-9]*)?)f?\s*,\s*([0-9]+(?:\.[0-9]*)?)f?\s*,\s*"
        r"([0-9]+(?:\.[0-9]*)?)f?\s*\}",
        table,
    ):
        rows.append({
            "damage": float(damage),
            "hit_interval": float(interval),
            "area_radius": float(radius),
        })
    return rows


def _extract_float_array_values(text: str, name: str) -> list[float]:
    array = _extract_inline_array(text, name)
    if not array:
        return []
    body_match = re.search(r"\{(?P<body>.*?)\}", array, flags=re.DOTALL)
    body = body_match.group("body") if body_match else array
    return [float(v) for v in re.findall(r"([0-9]+(?:\.[0-9]*)?)f?\b", body)]


def _extract_enemy_types(text: str) -> list[dict[str, Any]]:
    table_match = re.search(
        r"static\s+const\s+FRow\s+Rows\[\]\s*=\s*\{(?P<body>.*?)\n\s*\};",
        text,
        flags=re.DOTALL,
    )
    if not table_match:
        return []
    rows = []
    row_re = re.compile(
        r"\{\s*TEXT\(\"(?P<name>[^\"]+)\"\)\s*,\s*"
        r"(?P<hp>[0-9]+(?:\.[0-9]*)?)f?\s*,\s*"
        r"(?P<speed>[0-9]+(?:\.[0-9]*)?)f?\s*,\s*"
        r"(?P<damage>[0-9]+(?:\.[0-9]*)?)f?\s*,\s*"
        r"(?P<radius>[0-9]+(?:\.[0-9]*)?)f?\s*,\s*"
        r"(?P<knockback>[0-9]+(?:\.[0-9]*)?)f?\s*,\s*"
        r"(?P<boss>true|false)\s*\}",
    )
    for idx, match in enumerate(row_re.finditer(table_match.group("body"))):
        rows.append({
            "type_id": idx,
            "name": match.group("name"),
            "base_hp": float(match.group("hp")),
            "speed": float(match.group("speed")),
            "contact_damage": float(match.group("damage")),
            "collision_radius": float(match.group("radius")),
            "knockback_resistance": float(match.group("knockback")),
            "is_boss": match.group("boss") == "true",
        })
    return rows


def _direction_bin_for(dx: float, dy: float, dir_count: int) -> int:
    import math

    angle_rad = math.atan2(dy, dx)
    angle01 = (angle_rad + math.pi) / (2.0 * math.pi)
    return max(0, min(dir_count - 1, int(math.floor(angle01 * dir_count))))


def _directional_density_semantics(dir_count: int | None) -> dict[str, Any]:
    count = int(dir_count or 16)
    return {
        "source_function": "BuildDirectionalDensityFeatures",
        "formula": "Dir = floor(((atan2(Rel.Y, Rel.X) + PI) / (2 * PI)) * DirCount)",
        "dir_count": count,
        "angle_shift": "+PI",
        "axis_mapping": {
            "+X": _direction_bin_for(1.0, 0.0, count),
            "+Y": _direction_bin_for(0.0, 1.0, count),
            "-Y": _direction_bin_for(0.0, -1.0, count),
            "-X_boundary": "Dir 0 or DirCount-1 depending on exact atan2 boundary",
        },
        "notes": [
            "Dir 0 is not +X. For 16 bins, +X maps to Dir 8.",
            "Code that defines bin angles as i * 2*pi/16 assumes Dir 0 = +X and is incorrect.",
            "Use the same atan2 + PI formula as C++ when mapping vectors to density bins.",
        ],
    }


def _snippet(text: str, rel_path: str, symbol: str, pattern: str, context: int = 2) -> dict[str, Any] | None:
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if re.search(pattern, line):
            start = max(0, idx - context)
            end = min(len(lines), idx + context + 1)
            return {
                "path": rel_path,
                "symbol": symbol,
                "line": idx + 1,
                "code": "\n".join(lines[start:end]),
            }
    return None


def _schema_payload(obs_schema: dict[str, Any] | None) -> dict[str, Any]:
    if not obs_schema:
        return {}
    segments = obs_schema.get("segments", [])
    return {
        "total_dim": obs_schema.get("total_dim"),
        "obs_schema_hash": obs_schema.get("schema_hash") or obs_schema.get("hash"),
        "segments": segments,
    }


def _fetch_schema(host: str | None, port: int | None) -> dict[str, Any] | None:
    if not host or port is None:
        return None
    try:
        return fetch_obs_schema(host, port)
    except Exception as exc:
        return {"error": str(exc)}


def build_survivors_source_of_truth(
    repo_root: Path,
    host: str | None = None,
    port: int | None = None,
    obs_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build structured facts from C++ source and, when available, UE5 obs_schema."""

    game_h_path = "ReinBalance/Source/ReinBalance/Public/Survivors/Logic/SurvivorsGame.h"
    constants_h_path = "ReinBalance/Source/ReinBalance/Public/Survivors/Logic/SurvivorsGameConstants.h"
    gem_cpp_path = "ReinBalance/Source/ReinBalance/Private/Survivors/Logic/SurvivorsGemComponent.cpp"
    enemy_cpp_path = "ReinBalance/Source/ReinBalance/Private/Survivors/Logic/SurvivorsEnemyComponent.cpp"
    obs_cpp_path = "ReinBalance/Source/ReinBalance/Private/Survivors/Logic/SurvivorsObservationComponent.cpp"
    player_cpp_path = "ReinBalance/Source/ReinBalance/Private/Survivors/Logic/SurvivorsPlayerComponent.cpp"
    spawn_cpp_path = "ReinBalance/Source/ReinBalance/Private/Survivors/Logic/SurvivorsSpawnComponent.cpp"

    game_h = _read(repo_root, game_h_path)
    constants_h = _read(repo_root, constants_h_path)
    gem_cpp = _read(repo_root, gem_cpp_path)
    enemy_cpp = _read(repo_root, enemy_cpp_path)
    obs_cpp = _read(repo_root, obs_cpp_path)
    player_cpp = _read(repo_root, player_cpp_path)
    spawn_cpp = _read(repo_root, spawn_cpp_path)

    if obs_schema is None:
        obs_schema = _fetch_schema(host, port)

    reward_constants = {
        "AliveReward": _extract_number(game_h, "AliveReward"),
        "ItemReward": _extract_number(game_h, "ItemReward"),
        "KillReward": _extract_number(game_h, "KillReward"),
    }
    player_constants = {
        "MaxPlayerHP": _extract_number(game_h, "MaxPlayerHP"),
        "MoveSpeed": _extract_number(game_h, "MoveSpeed"),
        "GemPickupRadius": _extract_number(game_h, "GemPickupRadius"),
    }
    observation_constants = {
        "NumGemObs": _extract_const_number(constants_h, "NumGemObs"),
        "MaxEnemyObs": _extract_const_number(constants_h, "MaxEnemyObs"),
        "MaxWeaponSlots": _extract_const_number(constants_h, "MaxWeaponSlots"),
        "MaxWeaponLevel": _extract_const_number(constants_h, "MaxWeaponLevel"),
        "MaxPlayerLevel": _extract_const_number(constants_h, "MaxPlayerLevel"),
        "PhysicsDt": _extract_const_number(constants_h, "PhysicsDt"),
        "MaxGameTime": _extract_const_number(constants_h, "MaxGameTime"),
        "ContactHitInterval": _extract_const_number(constants_h, "ContactHitInterval"),
        "GarlicKnockbackStrength": _extract_const_number(constants_h, "GarlicKnockbackStrength"),
        "EnemyDensityDirCount": _extract_const_number(constants_h, "EnemyDensityDirCount"),
        "EnemyNearestDistanceMax": _extract_const_number(constants_h, "EnemyNearestDistanceMax"),
        "EnemyDensityNearDistanceMax": _extract_const_number(constants_h, "EnemyDensityNearDistanceMax"),
        "EnemyDensityMidDistanceMax": _extract_const_number(constants_h, "EnemyDensityMidDistanceMax"),
        "EnemyDensityNearNormalizeFactor": _extract_const_number(
            constants_h, "EnemyDensityNearNormalizeFactor"
        ),
        "EnemyDensityMidNormalizeFactor": _extract_const_number(
            constants_h, "EnemyDensityMidNormalizeFactor"
        ),
        "GemDensityDirCount": _extract_const_number(constants_h, "GemDensityDirCount"),
        "GemNearestDistanceMax": _extract_const_number(constants_h, "GemNearestDistanceMax"),
        "GemDensityNearDistanceMax": _extract_const_number(constants_h, "GemDensityNearDistanceMax"),
        "GemDensityMidDistanceMax": _extract_const_number(constants_h, "GemDensityMidDistanceMax"),
        "GemDensityNearNormalizeFactor": _extract_const_number(
            constants_h, "GemDensityNearNormalizeFactor"
        ),
        "GemDensityMidNormalizeFactor": _extract_const_number(
            constants_h, "GemDensityMidNormalizeFactor"
        ),
    }

    item_reward = reward_constants["ItemReward"]
    kill_reward = reward_constants["KillReward"]
    alive_reward = reward_constants["AliveReward"]
    garlic_table_values = _extract_garlic_table(constants_h)
    gem_xp_values = _extract_float_array_values(constants_h, "GemXPValues")
    enemy_types = _extract_enemy_types(spawn_cpp)
    directional_density = _directional_density_semantics(
        observation_constants.get("EnemyDensityDirCount")
    )

    snippets = [
        _snippet(game_h, game_h_path, "reward constants", r"\bAliveReward\b", context=8),
        _snippet(game_h, game_h_path, "player constants", r"\bMaxPlayerHP\b", context=4),
        _snippet(gem_cpp, gem_cpp_path, "gem pickup reward", r"LastReward\s*\+=\s*Game->ItemReward"),
        _snippet(enemy_cpp, enemy_cpp_path, "enemy kill reward", r"LastReward\s*\+=\s*Game->KillReward"),
        _snippet(obs_cpp, obs_cpp_path, "observation schema", r"enemy_nearest_dist_16dir", context=8),
        _snippet(obs_cpp, obs_cpp_path, "directional density bin formula", r"AngleRad", context=10),
        _snippet(player_cpp, player_cpp_path, "XPRequiredForLevel", r"XPRequiredForLevel", context=18),
        _snippet(spawn_cpp, spawn_cpp_path, "EnemyTypeTable", r"static const FRow Rows\[\]", context=18),
    ]
    garlic_table = _extract_inline_array(constants_h, "GarlicTable")
    gem_xp_snippet = _extract_inline_array(constants_h, "GemXPValues")
    if garlic_table:
        snippets.append(
            {
                "path": constants_h_path,
                "symbol": "GarlicTable",
                "line": None,
                "code": garlic_table,
            }
        )
    if gem_xp_snippet:
        snippets.append(
            {
                "path": constants_h_path,
                "symbol": "GemXPValues",
                "line": None,
                "code": gem_xp_snippet,
            }
        )

    return {
        "game": "survivors",
        "source": {
            "mode": "cpp_static_scan_plus_ue5_obs_schema",
            "generated_at": _dt.datetime.now(_dt.timezone.utc).astimezone().isoformat(),
        },
        "reward_constants": reward_constants,
        "player_constants": player_constants,
        "observation_constants": observation_constants,
        "garlic_table": garlic_table_values,
        "gem_xp_values": gem_xp_values,
        "enemy_types": enemy_types,
        "directional_density": directional_density,
        "observation": _schema_payload(obs_schema),
        "reward_events": {
            "alive": {
                "expression": "LastReward += AliveReward",
                "value": alive_reward,
                "unit": "per physics step",
            },
            "gem_pickup": {
                "expression": "LastReward += ItemReward",
                "value": item_reward,
            },
            "enemy_kill": {
                "expression": "LastReward += KillReward",
                "value": kill_reward,
            },
        },
        "base_reward_detection_rules": {
            "gem_pickup_single": f"base_reward >= {item_reward}",
            "enemy_kill_single": f"base_reward >= {kill_reward}",
            "do_not_assume_item_reward": 3.0,
            "notes": [
                "base_reward is the sum of C++ fixed rewards for a physics step.",
                "Do not infer gem count by dividing by 3.0 when ItemReward is 1.0.",
                "Use obs_schema offsets instead of hard-coded stale indices.",
            ],
        },
        "source_snippets": [s for s in snippets if s],
    }


def render_source_of_truth_markdown(sot: dict[str, Any] | None) -> str:
    if not sot:
        return ""

    reward = sot.get("reward_constants", {})
    player = sot.get("player_constants", {})
    obs_consts = sot.get("observation_constants", {})
    observation = sot.get("observation", {})
    garlic_table = sot.get("garlic_table", [])
    gem_xp_values = sot.get("gem_xp_values", [])
    enemy_types = sot.get("enemy_types", [])
    directional = sot.get("directional_density", {})
    snippets = sot.get("source_snippets", [])

    lines = [
        "以下は C++/UE5 から自動抽出した Source of Truth です。",
        "手書き説明や過去ログと矛盾する場合は、この値を優先してください。",
        "",
        "### Reward constants",
        f"- AliveReward: {reward.get('AliveReward')} per physics step",
        f"- ItemReward: {reward.get('ItemReward')} per gem pickup",
        f"- KillReward: {reward.get('KillReward')} per enemy kill",
        "",
        "### Player / item constants",
        f"- MaxPlayerHP: {player.get('MaxPlayerHP')}",
        f"- MoveSpeed: {player.get('MoveSpeed')}",
        f"- GemPickupRadius: {player.get('GemPickupRadius')}",
        "",
        "### Observation constants",
        f"- NumGemObs: {obs_consts.get('NumGemObs')}",
        f"- MaxEnemyObs: {obs_consts.get('MaxEnemyObs')}",
        f"- EnemyDensityDirCount: {obs_consts.get('EnemyDensityDirCount')}",
        f"- EnemyNearestDistanceMax: {obs_consts.get('EnemyNearestDistanceMax')}",
        f"- EnemyDensityNearDistanceMax: {obs_consts.get('EnemyDensityNearDistanceMax')}",
        f"- EnemyDensityMidDistanceMax: {obs_consts.get('EnemyDensityMidDistanceMax')}",
        f"- GemDensityDirCount: {obs_consts.get('GemDensityDirCount')}",
        f"- GemNearestDistanceMax: {obs_consts.get('GemNearestDistanceMax')}",
        f"- GemDensityNearDistanceMax: {obs_consts.get('GemDensityNearDistanceMax')}",
        f"- GemDensityMidDistanceMax: {obs_consts.get('GemDensityMidDistanceMax')}",
        "",
        "### XP / Garlic / Enemy parameters",
        f"- GemXPValues: {gem_xp_values}",
        f"- GarlicTable levels: {len(garlic_table)}",
        f"- EnemyTypeTable rows: {len(enemy_types)}",
        "",
        "### Directional density bin semantics",
        f"- Formula: {directional.get('formula')}",
        f"- DirCount: {directional.get('dir_count')}",
        f"- Axis mapping: {directional.get('axis_mapping')}",
        "- Important: Dir 0 is not +X. For 16 bins, +X maps to Dir 8.",
        "",
        "### UE5 obs_schema",
        "```json",
        json.dumps(observation, ensure_ascii=False, indent=2),
        "```",
        "",
        "### reward_fn で守ること",
        "- Gem 単体取得は ItemReward=1.0 なので base_reward >= 1.0 で検出可能です。",
        "- `base_reward >= 2.9` を Gem 取得条件にしてはいけません。",
        "- `remaining / 3.0` で Item 数を推定してはいけません。",
        "- MaxPlayerHP は 70.0 です。`obs[12] * 100.0` のような旧値計算を使わないでください。",
        "",
        "### Source snippets",
    ]

    for snippet in snippets:
        title = snippet.get("symbol", "snippet")
        path = snippet.get("path", "")
        line = snippet.get("line")
        loc = f"{path}:{line}" if line else path
        lines.extend([
            f"#### {title} ({loc})",
            "```cpp",
            snippet.get("code", ""),
            "```",
        ])

    return "\n".join(lines)
