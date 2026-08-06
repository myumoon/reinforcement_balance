"""ItemSelector 推論戦略の候補束縛と決定性を検証する。

候補の順番ではなく card identity により選択し、padding や不正な候補を選ばないことを固定する。
"""

from __future__ import annotations

import torch as th

from reinbalance_survivors_contracts.item_decision import (
    CandidateFeatures,
    ItemDecisionFeatures,
)

from games.survivors.item_selection_strategy import ItemSelectionStrategy
from games.survivors.item_selector_model import ItemSelector


def _features(*, candidates: tuple[CandidateFeatures, ...]) -> ItemDecisionFeatures:
    """選択戦略テスト用の最小有効な level-up feature を返す。

    context_only_v1 と二候補を固定して、score の順序だけをテスト対象にする。
    """
    return ItemDecisionFeatures(
        decision_id="decision-1",
        feature_schema="context_only_v1",
        elapsed_time=60.0,
        level=2,
        hp_ratio=1.0,
        xp_ratio=0.5,
        weapon_slots=(1, 0, 0, 0, 0, 0),
        passive_slots=(0, 0, 0, 0, 0, 0),
        empty_slot_count=11,
        evolution_readiness=0.0,
        choice_count=len(candidates),
        card_mask=(True,) * len(candidates),
        fallback_kind="none",
        ui_state_validity=1.0,
        ui_state_age=0.1,
        candidates=candidates,
        max_item_cards=2,
    )


def _candidate(item_id: str) -> CandidateFeatures:
    """指定した identity を持つ weapon card を返す。

    card 間の feature 差を item_id に限定する。
    """
    return CandidateFeatures(
        kind="weapon",
        item_id=item_id,
        new_level=1,
        owned=False,
        is_new=True,
        is_evolve=False,
        is_union=False,
        has_prerequisite=False,
        slot_capacity=1,
    )


def test_strategy_selects_highest_logit_and_keeps_item_id_bound_to_candidate() -> None:
    """最高 logit の candidate identity を decision と共に返す。

    model は一定の index ではなく candidate feature を採点し、返却値が live choice に渡せる。
    """
    model = ItemSelector(context_dim=24, candidate_dim=9)
    with th.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.scorer[-1].bias.fill_(1.0)
    strategy = ItemSelectionStrategy(model)

    decision = strategy.select(_features(candidates=(_candidate("wand"), _candidate("knife"))))

    assert decision.decision_id == "decision-1"
    assert decision.choice_id == "knife"
    assert decision.candidate_index == 1
    assert decision.candidate_logits == (1.0, 1.0)


def test_strategy_uses_choice_id_tie_breaker_not_live_candidate_order() -> None:
    """同点は choice ID の辞書順で解決して候補順依存を作らない。

    UI 表示順が変わっても同じ card を選び、同点を暗黙の index 選択にしない。
    """
    model = ItemSelector(context_dim=24, candidate_dim=9)
    with th.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    strategy = ItemSelectionStrategy(model)

    first = strategy.select(_features(candidates=(_candidate("wand"), _candidate("knife"))))
    second = strategy.select(_features(candidates=(_candidate("knife"), _candidate("wand"))))

    assert first.choice_id == second.choice_id == "knife"
