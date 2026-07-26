"""Survivors scenario coverage の組合せ・除外意味論テスト。

初心者向け:
単体の行だけでなく union、防御、即死耐性、boss と starting-only 除外を検証します。
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from games.survivors.content_manifest import ContractValidationError, build_manifest, load_annotations
from games.survivors.survivors_vs_spec import WEAPON_EXCLUDED_AS_STARTING
from games.survivors.survivors_weapon_curriculum import EXCLUDED_AS_STARTING_WEAPON
from tests.survivors.test_content_manifest import canonical_schema

ROOT = Path(__file__).resolve().parents[4]
ANNOTATIONS = ROOT / "Tools/Training/configs/survivors_content_annotations_v1.yaml"


def test_required_combination_cells_have_eval_assertions() -> None:
    """4種類の combination cell と評価 assertion を固定する。

    初心者向け:
    複数 mechanics の境界シナリオが消えると manifest 作成を失敗させます。
    """
    manifest = build_manifest(canonical_schema(), load_annotations(ANNOTATIONS))
    assert {row["kind"] for row in manifest.annotations["combinations"]} == {
        "pair_evolution_union", "weak_defensive_weapon", "instant_kill_resistance", "boss_interaction",
    }


def test_starting_exclusions_do_not_remove_any_coverage_gate() -> None:
    """starting-only 定数が acquisition/effect/obs/eval を除外しない。

    初心者向け:
    Pentagram と Laurel を初期装備にしなくても、後から取得するテストは必須のままです。
    """
    manifest = build_manifest(canonical_schema(), load_annotations(ANNOTATIONS))
    assert set(WEAPON_EXCLUDED_AS_STARTING) == set(EXCLUDED_AS_STARTING_WEAPON) == {12, 15, 27}
    for item_id in ("12", "15", "27"):
        row = manifest.annotations["collections"]["weapons"][item_id]
        assert all(row[gate] for gate in ("implemented", "reachable", "observed", "trained", "evaluated"))


def test_starting_exclusion_without_alternative_coverage_is_rejected() -> None:
    """代替 coverage のない intentional exclusion を拒否する。

    初心者向け:
    「初期装備ではない」を全体除外へ広げる設定ミスを防ぎます。
    """
    annotations = copy.deepcopy(load_annotations(ANNOTATIONS))
    annotations["intentional_exclusions"][0]["alternative_coverage"] = ""
    with pytest.raises(ContractValidationError, match="alternative coverage"):
        build_manifest(canonical_schema(), annotations)


def test_starting_exclusion_exact_set_is_required() -> None:
    """両定数由来の starting exclusion 必須集合を exact-set で固定する。

    初心者向け:
    exclusion 行を削除・追加して coverage 対象をこっそり変えることを拒否します。
    """
    expected = {str(value) for value in WEAPON_EXCLUDED_AS_STARTING}
    assert expected == {str(value) for value in EXCLUDED_AS_STARTING_WEAPON}
    annotations = copy.deepcopy(load_annotations(ANNOTATIONS))
    annotations["intentional_exclusions"].pop()
    with pytest.raises(ContractValidationError, match="starting exclusion ids mismatch"):
        build_manifest(canonical_schema(), annotations)


def test_starting_exclusion_constants_have_no_eval_or_audit_consumers() -> None:
    """starting-only 定数の全参照を走査し audit/eval filter への流用を封じる。

    初心者向け:
    将来ほかの Python ファイルがこの定数で評価対象を除外すると、参照先の追加だけで失敗します。
    """
    training_root = ROOT / "Tools/Training"
    names = ("WEAPON_EXCLUDED_AS_STARTING", "EXCLUDED_AS_STARTING_WEAPON")
    consumers: set[str] = set()
    for path in training_root.rglob("*.py"):
        if "tests" in path.parts:
            continue
        source = path.read_text(encoding="utf-8")
        if any(name in source for name in names):
            consumers.add(path.relative_to(ROOT).as_posix())
    assert consumers == {
        "Tools/Training/games/survivors/survivors_vs_spec.py",
        "Tools/Training/games/survivors/survivors_weapon_curriculum.py",
    }
    manifest = build_manifest(canonical_schema(), load_annotations(ANNOTATIONS))
    for item_id in ("12", "15", "27"):
        assert all(manifest.resolved_gates["weapons"][item_id].values())


@pytest.mark.parametrize("member", ["unknown:999", "weapon:999", "weapon:13"])
def test_combination_members_are_exact_unique_and_resolved(member: str) -> None:
    """combination member の形式・ID・重複を fail-closed 検証する。

    初心者向け:
    存在しない種類や ID、同じ member の二重指定を組合せ証跡として認めません。
    """
    annotations = copy.deepcopy(load_annotations(ANNOTATIONS))
    annotations["combinations"][0]["members"].append(member)
    with pytest.raises(ContractValidationError):
        build_manifest(canonical_schema(), annotations)


@pytest.mark.parametrize("key", ["scenario", "eval_assertion"])
def test_combination_execution_evidence_must_resolve(key: str) -> None:
    """combination の scenario と eval assertion を実在 ID へ seal する。

    初心者向け:
    架空のセル名や評価名を書くだけでは組合せ coverage を通過できません。
    """
    annotations = copy.deepcopy(load_annotations(ANNOTATIONS))
    annotations["combinations"][0][key] = "does_not_exist"
    with pytest.raises(ContractValidationError, match="unresolved"):
        build_manifest(canonical_schema(), annotations)
