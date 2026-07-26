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
