"""Survivors content manifest の一方向性と fail-closed 契約テスト。

初心者向け:
C++ schema と YAML の差、重複、欠落、手書き互換定数のずれを意図的に作って拒否を確認します。
"""

from __future__ import annotations

import copy
import re
from pathlib import Path

import pytest

import games.survivors.content_manifest as content_manifest_module
from games.survivors.content_manifest import (
    ContractValidationError,
    audit_manifest,
    build_manifest,
    compatibility_constants,
    load_annotations,
)
from games.survivors.survivors_vs_spec import (
    EVOLVED_MAX_LEVEL, EVOLUTION_TABLE, PASSIVE_MAX_LEVEL, WEAPON_MAX_LEVEL,
    WEAPON_EXCLUDED_AS_STARTING, PassiveItemType,
)
from games.survivors.survivors_weapon_curriculum import WeaponType
from games.survivors.survivors_weapon_table import (
    WEAPON_UNLOCK_TABLES, validate_unlock_table_against_generated_ids,
)

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
    if collection == "gems":
        schema["content"][collection][1]["id"] = schema["content"][collection][0]["id"]
    else:
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
            with pytest.raises(ContractValidationError, match="contradicts resolved evidence"):
                build_manifest(canonical_schema(), annotations)
        else:
            with pytest.raises(ContractValidationError):
                build_manifest(canonical_schema(), annotations)


def test_python_compatibility_mirrors_match_generated_schema() -> None:
    """全 Python compatibility 定数と手動 mirror が generated schema と一致する。

    初心者向け:
    移行期間中の手書き表が C++ export とずれたら CI で発見します。
    """
    manifest = build_manifest(canonical_schema(), load_annotations(ANNOTATIONS))
    constants = compatibility_constants(manifest)
    weapon_enum_ids = {
        str(value) for name, value in vars(WeaponType).items()
        if name.isupper() and name != "NONE" and type(value) is int
    }
    passive_enum_ids = {
        str(value) for name, value in vars(PassiveItemType).items()
        if name.isupper() and name != "NONE" and type(value) is int
    }
    assert weapon_enum_ids == constants["weapon_ids"]
    assert passive_enum_ids == constants["passive_ids"]
    assert PASSIVE_MAX_LEVEL == {0: 0, **{int(key): value for key, value in manifest.max_levels["passives"].items()}}
    assert all(
        max_level == (WEAPON_MAX_LEVEL if int(item_id) <= 14 else
                      (7 if item_id == "15" else EVOLVED_MAX_LEVEL))
        for item_id, max_level in constants["weapon_max_levels"].items()
    )
    assert {
        (str(row["base"]), str(row["evolved"]), str(row["passive"]), str(row.get("union_weapon", 0)))
        for row in EVOLUTION_TABLE
    } == set(manifest.evolution_pairs)
    validate_unlock_table_against_generated_ids(constants["weapon_ids"])
    evolved_ids = {pair[1] for pair in constants["evolution_pairs"]}
    expected_unlock_ids = (
        constants["weapon_ids"] - evolved_ids
        - frozenset(str(value) for value in WEAPON_EXCLUDED_AS_STARTING)
    )
    for table in WEAPON_UNLOCK_TABLES.values():
        table_ids = [str(entry.weapon_id) for entry in table]
        assert len(table_ids) == len(set(table_ids))
        assert frozenset(table_ids) == expected_unlock_ids
    content = manifest.schema["content"]
    for key in ("max_level", "level_cadence", "offer", "slots", "chest"):
        assert constants[key] == content[key]
    assert constants["xp_curve"] == tuple(content["xp_curve"])
    assert constants["gem_xp"] == {row["id"]: row["xp"] for row in content["gems"]}
    assert constants["action_time"] == manifest.schema["action_time"]


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


@pytest.mark.parametrize("key", ["effect_handler", "obs_category", "scenario"])
def test_unresolved_trace_evidence_is_blocked(key: str) -> None:
    """架空の handler・観測・scenario を全て blocking にする。

    初心者向け:
    boolean が true のままでも、実在証跡へ解決できない文字列なら監査を開始しません。
    """
    annotations = copy.deepcopy(load_annotations(ANNOTATIONS))
    annotations["collections"]["weapons"]["1"][key] = "does_not_exist"
    with pytest.raises(ContractValidationError, match="unresolved"):
        build_manifest(canonical_schema(), annotations)


def test_removed_production_weapon_case_is_blocking(monkeypatch: pytest.MonkeyPatch) -> None:
    """production handler を削除した content を implemented にしない。

    初心者向け:
    Python の handler 名を残しても、ゲーム本体の武器生成 case が消えれば監査を失敗させます。
    """
    original = Path.read_text

    def mutated_read_text(path: Path, *args: object, **kwargs: object) -> str:
        source = original(path, *args, **kwargs)
        if path == content_manifest_module._LOGIC_PATH:
            source = source.replace("case EWeaponType::Garlic:\n", "", 1)
        return source

    monkeypatch.setattr(Path, "read_text", mutated_read_text)
    with pytest.raises(ContractValidationError, match="weapons:1 unresolved"):
        build_manifest(canonical_schema(), load_annotations(ANNOTATIONS))


def test_removed_llt_content_assertion_registry_row_is_blocking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """content ID ごとの LLT scenario/eval registry 欠落を拒否する。

    初心者向け:
    YAML の名前を残しても、実行対象テストから gem 行を消せば coverage は成立しません。
    """
    original = Path.read_text

    def mutated_read_text(path: Path, *args: object, **kwargs: object) -> str:
        source = original(path, *args, **kwargs)
        if path == content_manifest_module._LLT_PATH:
            source = re.sub(r'^\s*\{TEXT\("gem:blue"\).*$', "", source, count=1, flags=re.MULTILINE)
        return source

    monkeypatch.setattr(Path, "read_text", mutated_read_text)
    with pytest.raises(ContractValidationError, match="LLT content coverage keys mismatch"):
        build_manifest(canonical_schema(), load_annotations(ANNOTATIONS))


def test_removed_table_driven_eval_assertion_is_blocking(monkeypatch: pytest.MonkeyPatch) -> None:
    """registry を残したまま実評価 CHECK を消しても拒否する。

    初心者向け:
    scenario/eval の名前だけでは足りず、全 ID を回す LLT の判定本体が必要です。
    """
    original = Path.read_text

    def mutated_read_text(path: Path, *args: object, **kwargs: object) -> str:
        source = original(path, *args, **kwargs)
        if path == content_manifest_module._LLT_PATH:
            source = source.replace(
                "CHECK(static_cast<int32>(Logic.GetWeaponSlot(0).Type) == WeaponId);",
                "",
                1,
            )
        return source

    monkeypatch.setattr(Path, "read_text", mutated_read_text)
    with pytest.raises(ContractValidationError, match="table-driven eval assertion missing"):
        build_manifest(canonical_schema(), load_annotations(ANNOTATIONS))


def test_removed_training_consumer_cell_is_blocking(monkeypatch: pytest.MonkeyPatch) -> None:
    """訓練 consumer から content を除外すると trained gate を閉じる。

    初心者向け:
    scenario 名が YAML と LLT に残っていても、訓練セルに収録されていなければ拒否します。
    """
    original_builder = content_manifest_module.build_content_training_cells

    def without_weapon_one(ids: dict[str, frozenset[str]]) -> dict[str, str]:
        cells = original_builder(ids)
        cells.pop("weapon:1")
        return cells

    monkeypatch.setattr(content_manifest_module, "build_content_training_cells", without_weapon_one)
    with pytest.raises(ContractValidationError, match="training content keys mismatch"):
        build_manifest(canonical_schema(), load_annotations(ANNOTATIONS))
