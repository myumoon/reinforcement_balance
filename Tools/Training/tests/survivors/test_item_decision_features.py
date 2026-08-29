"""ItemSelector の画面観測 feature 契約と canonical wire identity を検証する。

2 種類の feature 案、候補 padding、round-trip を固定し、教師値を入力へ混ぜない境界を
golden test として維持する。
"""

from __future__ import annotations

import copy

import pytest

from reinbalance_survivors_contracts.canonical_json import canonical_hash
from reinbalance_survivors_contracts.item_decision import (
    CandidateFeatures,
    ItemDecisionFeatures,
)


def _candidate(item_id: str, *, level: int = 1) -> CandidateFeatures:
    """画面カードから取得できる候補 feature を一件作る。

    teacher score や critic latent は fixture 自体にも含めない。
    """
    return CandidateFeatures(
        kind="weapon",
        item_id=item_id,
        new_level=level,
        owned=level > 1,
        is_new=level == 1,
        is_evolve=False,
        is_union=False,
        has_prerequisite=True,
        slot_capacity=6,
    )


def _features(schema: str) -> ItemDecisionFeatures:
    """context-only または danger 付きの決定 feature を作る。

    danger 専用値は schema が要求するときだけ渡す。
    """
    danger = (
        {
            "enemy_density": 0.25,
            "gem_density": 0.5,
            "nearest_enemy_screen_dist": 0.125,
            "nearest_enemy_screen_dir": (-0.5, 0.75),
            "boss_flag": True,
            "hazard_flag": False,
            "world_validity": 1.0,
            "world_age": 0.04,
            "last_gameplay_snapshot_age": 0.08,
        }
        if schema == "context_danger_v1"
        else {}
    )
    return ItemDecisionFeatures(
        decision_id="decision-001",
        feature_schema=schema,
        elapsed_time=125.5,
        level=12,
        hp_ratio=0.75,
        xp_ratio=0.375,
        weapon_slots=(2, 1, 0, 0, 0, 0),
        passive_slots=(1, 0, 0, 0, 0, 0),
        empty_slot_count=9,
        evolution_readiness=0.5,
        choice_count=2,
        card_mask=(True, True, False),
        fallback_kind="none",
        ui_state_validity=1.0,
        ui_state_age=0.02,
        candidates=(_candidate("wand", level=2), _candidate("knife")),
        max_item_cards=3,
        **danger,
    )


@pytest.mark.parametrize("schema", ["context_only_v1", "context_danger_v1"])
def test_versioned_wire_round_trip_and_padding(schema: str) -> None:
    """両 feature schema の wire が Nmax 固定長で round-trip する。

    無効カードは sentinel 候補へ padding され、card mask と一緒に復元される。
    """
    features = _features(schema)
    wire = features.to_wire()
    assert wire["feature_schema"] == schema
    assert len(wire["candidates"]) == 3
    assert wire["candidates"][2]["kind"] == "padding"
    assert ItemDecisionFeatures.from_wire(wire) == features
    assert features.decision_hash == canonical_hash(wire)


def test_context_only_wire_canonical_hash_golden() -> None:
    """context_only_v1 の canonical hash を golden 値へ固定する。

    key 順や bool/float 表現の変更を producer/consumer の破壊的変更として検出する。
    """
    assert _features("context_only_v1").decision_hash == (
        "6546e66de9177ea48afefaab85a864ecbf1ddd3535ad37e3920f27369cbf1c73"
    )


def test_variant_fields_and_unknown_wire_fields_fail_closed() -> None:
    """danger field の片側混入と未知 wire field を拒否する。

    variant 間の暗黙 substitution と未審査 feature 追加を同じ境界で止める。
    """
    with pytest.raises(ValueError, match="danger"):
        _features("context_only_v1").with_danger(enemy_density=0.2)
    wire = copy.deepcopy(_features("context_only_v1").to_wire())
    wire["teacher_score"] = 9.0
    with pytest.raises(ValueError, match="fields"):
        ItemDecisionFeatures.from_wire(wire)


def test_occupancy_slot_binary_enforced() -> None:
    """context_danger_occupancy_v1 のスロット値は 0/1 のみ許可される。

    occupancy schema は weapon/passive slot を binary で表現するため、
    level 値 (2 以上) や bool も含め {0,1} 以外は拒否します。
    """
    danger = {
        "enemy_density": 0.25,
        "gem_density": 0.5,
        "nearest_enemy_screen_dist": 0.125,
        "nearest_enemy_screen_dir": (-0.5, 0.75),
        "boss_flag": True,
        "hazard_flag": False,
        "world_validity": 1.0,
        "world_age": 0.04,
        "last_gameplay_snapshot_age": 0.08,
    }
    with pytest.raises(ValueError, match="weapon_slots"):
        ItemDecisionFeatures(
            decision_id="decision-001",
            feature_schema="context_danger_occupancy_v1",
            elapsed_time=125.5,
            level=12,
            hp_ratio=0.75,
            xp_ratio=0.375,
            weapon_slots=(2, 1, 0, 0, 0, 0),  # 2 は binary 違反
            passive_slots=(1, 0, 0, 0, 0, 0),
            empty_slot_count=9,
            evolution_readiness=0.5,
            choice_count=2,
            card_mask=(True, True, False),
            fallback_kind="none",
            ui_state_validity=1.0,
            ui_state_age=0.02,
            candidates=(_candidate("wand", level=1), _candidate("knife")),
            max_item_cards=3,
            **danger,
        )


def test_candidate_permutation_preserves_item_identity() -> None:
    """候補順を変えても item_id が feature と一緒に移動する。

    downstream の確率復元でカード位置を item identity と誤認させない。
    """
    original = _features("context_only_v1")
    permuted = original.with_candidates(tuple(reversed(original.candidates)))
    assert [item.item_id for item in permuted.candidates] == ["knife", "wand"]
    assert permuted.card_mask == (True, True, False)
