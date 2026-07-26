"""Survivors content manifest の一方向性と fail-closed 契約テスト。

初心者向け:
C++ schema と YAML の差、重複、欠落、手書き互換定数のずれを意図的に作って拒否を確認します。
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from games.survivors.content_manifest import (
    ContractValidationError,
    audit_manifest,
    build_manifest,
    load_annotations,
)
from games.survivors.survivors_vs_spec import EVOLUTION_TABLE, PASSIVE_MAX_LEVEL

ROOT = Path(__file__).resolve().parents[4]
ANNOTATIONS = ROOT / "Tools/Training/configs/survivors_content_annotations_v1.yaml"


def canonical_schema() -> dict:
    """現行 C++ export と同じ collection identity の fixture を返す。

    初心者向け:
    値のミラーを production へ置かず、テスト内だけで不正入力を作る土台にします。
    """
    weapon_rows = [{"id": str(item_id), "max_level": 8 if item_id < 15 else (7 if item_id == 15 else 1)}
                   for item_id in range(1, 29)]
    passive_rows = [{"id": str(item_id), "max_level": PASSIVE_MAX_LEVEL[item_id]} for item_id in range(1, 18)]
    evolution_rows = [
        {
            "base_id": str(row["base"]), "evolved_id": str(row["evolved"]),
            "passive_id": str(row["passive"]), "union_partner_id": str(row.get("union_weapon", 0)),
        }
        for row in EVOLUTION_TABLE
    ]
    directions = [
        (0.0, 1.0), (2 ** -0.5, 2 ** -0.5), (1.0, 0.0), (2 ** -0.5, -(2 ** -0.5)),
        (0.0, -1.0), (-(2 ** -0.5), -(2 ** -0.5)), (-1.0, 0.0),
        (-(2 ** -0.5), 2 ** -0.5), (0.0, 0.0),
    ]
    return {
        "schema_version": "survivors.content_schema.v1",
        "content": {
            "max_level": 100, "weapons": weapon_rows, "passives": passive_rows,
            "gems": [{"id": "blue", "xp": 2}, {"id": "green", "xp": 9}, {"id": "red", "xp": 10}],
            "xp_curve": [0] * 100, "level_cadence": "xp_threshold",
            "offer": {"count": 3, "fallback": "none_when_pool_empty"},
            "slots": {"weapon": 6, "passive": 6}, "evolutions": evolution_rows,
            "chest": {"boss_drop": True, "evolution_enabled_by_config": True},
        },
        "action_time": {
            "physics_dt": 1 / 60, "decision_steps": 1,
            "directions": [{"id": index, "x": xy[0], "y": xy[1]} for index, xy in enumerate(directions)],
            "move_speed": 80, "screen_displacement_per_step": 80 / 60,
            "pause_during_level_up": False, "level_up_timing": "same_physics_step",
        },
    }


def test_all_content_has_five_gate_trace_and_audit_is_clear() -> None:
    """全 collection の exact-set と5ゲート成功を検証する。

    初心者向け:
    追加された ID に YAML 行がなければ、このテストは必ず失敗します。
    """
    manifest = build_manifest(canonical_schema(), load_annotations(ANNOTATIONS))
    assert manifest.ids["weapons"] == frozenset(str(value) for value in range(1, 29))
    assert manifest.ids["passives"] == frozenset(str(value) for value in range(1, 18))
    assert manifest.ids["enemies"] == frozenset(str(value) for value in range(11))
    assert audit_manifest(manifest)["blocking"] == 0


@pytest.mark.parametrize("collection", ["weapons", "passives", "gems", "evolutions", "enemies"])
def test_added_content_without_annotation_is_blocked(collection: str) -> None:
    """全 collection の annotation 過不足を対称に拒否する。

    初心者向け:
    どの種類でも行を一つ消せば manifest が作れないことを確認します。
    """
    annotations = copy.deepcopy(load_annotations(ANNOTATIONS))
    annotations["collections"][collection].pop(next(iter(annotations["collections"][collection])))
    with pytest.raises(ContractValidationError, match="annotation ids mismatch"):
        build_manifest(canonical_schema(), annotations)


@pytest.mark.parametrize("collection", ["weapons", "passives", "gems"])
def test_duplicate_schema_id_is_blocked(collection: str) -> None:
    """schema collection の duplicate ID を対称に拒否する。

    初心者向け:
    同じ ID の二重定義が後勝ちで隠れることを防ぎます。
    """
    schema = canonical_schema()
    schema["content"][collection].append(copy.deepcopy(schema["content"][collection][0]))
    with pytest.raises(ContractValidationError, match="duplicate"):
        build_manifest(schema, load_annotations(ANNOTATIONS))


def test_unimplemented_level_handler_reachability_and_observation_are_blocking() -> None:
    """max level、effect、到達性、obs の欠陥を blocking にする。

    初心者向け:
    宣言だけの content を成功扱いしないことを複数の反例で確認します。
    """
    schema = canonical_schema()
    schema["content"]["weapons"][0]["max_level"] = 0
    with pytest.raises(ContractValidationError, match="positive integer"):
        build_manifest(schema, load_annotations(ANNOTATIONS))
    for key in ("effect_handler", "reachable", "obs_category"):
        annotations = copy.deepcopy(load_annotations(ANNOTATIONS))
        annotations["collections"]["weapons"]["1"][key] = "" if key != "reachable" else False
        if key == "reachable":
            assert audit_manifest(build_manifest(canonical_schema(), annotations))["blocking"] == 1
        else:
            with pytest.raises(ContractValidationError):
                build_manifest(canonical_schema(), annotations)


def test_python_compatibility_mirrors_match_generated_schema() -> None:
    """既存 Python compatibility 定数が generated schema と一致する。

    初心者向け:
    移行期間中の手書き表が C++ export とずれたら CI で発見します。
    """
    manifest = build_manifest(canonical_schema(), load_annotations(ANNOTATIONS))
    assert PASSIVE_MAX_LEVEL == {0: 0, **{int(key): value for key, value in manifest.max_levels["passives"].items()}}
    assert {
        (str(row["base"]), str(row["evolved"]), str(row["passive"]), str(row.get("union_weapon", 0)))
        for row in EVOLUTION_TABLE
    } == set(manifest.evolution_pairs)


def test_unknown_keys_and_nonfinite_values_fail_closed() -> None:
    """全入力境界で未知 key と非有限値を拒否する。

    初心者向け:
    typo や NaN を監査が読み飛ばさないことを確認します。
    """
    annotations = copy.deepcopy(load_annotations(ANNOTATIONS))
    annotations["collections"]["weapons"]["1"]["typo"] = True
    with pytest.raises(ContractValidationError, match="keys mismatch"):
        build_manifest(canonical_schema(), annotations)
    schema = canonical_schema()
    schema["content"]["gems"][0]["xp"] = float("nan")
    with pytest.raises(ContractValidationError, match="finite"):
        build_manifest(schema, load_annotations(ANNOTATIONS))
