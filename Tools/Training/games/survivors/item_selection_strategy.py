"""Item card scorer と UI intent を結ぶ、評価専用の選択戦略。"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import torch as th

from reinbalance_survivors_contracts.item_decision import ItemDecisionFeatures
from reinbalance_survivors_contracts.ui_intent import DecisionOwner, UiIntentKind, UiIntentV1

from games.survivors.item_selector_model import ItemSelector
from games.survivors.item_selector_trainer import _candidate_vector, _flatten_feature, _wire_candidate_identity


class ItemSelectionError(ValueError):
    """カード候補と選択の binding が満たせないことを表す。"""


@dataclass(frozen=True, slots=True)
class CardTargetRef:
    """一つの feature snapshot 内の card id/index を不可分に保持する参照。"""

    decision_id: str
    card_id: str
    candidate_index: int

    @classmethod
    def from_features(cls, features: ItemDecisionFeatures, candidate_index: int) -> "CardTargetRef":
        if type(candidate_index) is not int or candidate_index < 0 or candidate_index >= features.max_item_cards:
            raise ItemSelectionError("candidate_index is outside card slots")
        if not features.card_mask[candidate_index]:
            raise ItemSelectionError("candidate_index is not a valid card")
        candidate = features.padded_candidates[candidate_index]
        return cls(features.decision_id, candidate.item_id, candidate_index)

    def validate(self, features: ItemDecisionFeatures) -> None:
        expected = CardTargetRef.from_features(features, self.candidate_index)
        if expected != self:
            raise ItemSelectionError("card id/index does not identify the same candidate")


def _features(value: ItemDecisionFeatures | Mapping[str, Any]) -> ItemDecisionFeatures:
    try:
        return value if isinstance(value, ItemDecisionFeatures) else ItemDecisionFeatures.from_wire(value)
    except (TypeError, ValueError) as exc:
        raise ItemSelectionError(f"invalid item decision features: {exc}") from exc


def card_target_from_intent(features: ItemDecisionFeatures | Mapping[str, Any], intent: UiIntentV1) -> CardTargetRef:
    """CHOOSE_CARD intent の index を live candidate へ fail-closed で再束縛する。"""
    decision = _features(features)
    if intent.kind is not UiIntentKind.CHOOSE_CARD or intent.target_index is None:
        raise ItemSelectionError("selector intent must be choose_card with target_index")
    target = CardTargetRef.from_features(decision, intent.target_index)
    if intent.candidate_set_hash != decision.decision_hash:
        raise ItemSelectionError("selector intent candidate_set_hash does not match features")
    return target


class CardSelectionStrategy(Protocol):
    """generic UI ではなく card scorer が返す唯一の契約。"""
    strategy_id: str

    def select(self, features: ItemDecisionFeatures) -> UiIntentV1: ...


class _BaseCardStrategy:
    strategy_id = "card_strategy"

    def _intent(self, features: ItemDecisionFeatures, target: CardTargetRef) -> UiIntentV1:
        target.validate(features)
        digest = features.decision_hash
        return UiIntentV1(
            kind=UiIntentKind.CHOOSE_CARD,
            semantic_action="choose_card",
            decision_owner=DecisionOwner.ITEM_SELECTOR_SESSION,
            source_snapshot_hash=digest,
            source_frame_hash=digest,
            source_content_hash=digest,
            ui_state_key="level_up",
            target_index=target.candidate_index,
            candidate_set_hash=digest,
            decision_policy_id=self.strategy_id,
            decision_rule_id="card_score",
            decision_config_hash=digest,
        )


class RandomCardStrategy(_BaseCardStrategy):
    strategy_id = "random"

    def __init__(self, *, seed: int = 0) -> None:
        self._random = random.Random(seed)

    def select(self, features: ItemDecisionFeatures | Mapping[str, Any]) -> UiIntentV1:
        decision = _features(features)
        indices = [i for i, valid in enumerate(decision.card_mask) if valid]
        return self._intent(decision, CardTargetRef.from_features(decision, self._random.choice(indices)))


class CardHeuristicV1Strategy(_BaseCardStrategy):
    strategy_id = "card_heuristic_v1"

    @staticmethod
    def _score(features: ItemDecisionFeatures, index: int) -> tuple[int, int, int, str]:
        card = features.padded_candidates[index]
        # evolutionary cards and upgrades are deliberately preferred; the ID makes ties stable.
        return (int(card.is_evolve or card.is_union), int(card.has_prerequisite), card.new_level, card.item_id)

    def select(self, features: ItemDecisionFeatures | Mapping[str, Any]) -> UiIntentV1:
        decision = _features(features)
        indices = [i for i, valid in enumerate(decision.card_mask) if valid]
        selected = min(indices, key=lambda i: tuple(-x if isinstance(x, int) else x for x in self._score(decision, i)))
        return self._intent(decision, CardTargetRef.from_features(decision, selected))


class TeacherOracleStrategy(CardHeuristicV1Strategy):
    """比較専用 oracle。正式 item_selector run からは注入しない。"""
    strategy_id = "teacher_oracle"

    def __init__(self, teacher: Callable[[ItemDecisionFeatures], int] | None = None) -> None:
        self._teacher = teacher

    def select(self, features: ItemDecisionFeatures | Mapping[str, Any]) -> UiIntentV1:
        decision = _features(features)
        if self._teacher is None:
            return super().select(decision)
        index = self._teacher(decision)
        return self._intent(decision, CardTargetRef.from_features(decision, index))


class ItemSelectionStrategy(_BaseCardStrategy):
    """checkpoint 済み ItemSelector の scorer。teacher fallback は一切持たない。"""
    strategy_id = "item_selector"

    def __init__(self, model: ItemSelector, *, device: th.device | str = "cpu") -> None:
        if not isinstance(model, ItemSelector):
            raise TypeError("model must be an ItemSelector")
        self.model, self.device = model.to(device), th.device(device)

    def select(self, features: ItemDecisionFeatures | Mapping[str, Any]) -> UiIntentV1:
        decision = _features(features)
        wire = decision.to_wire()
        raw_context, raw_candidates = wire["context_features"], wire["candidates"]
        try:
            context = _flatten_feature(raw_context, label="context_features")
            vectors = [_candidate_vector(candidate, index=i) for i, candidate in enumerate(raw_candidates)]
        except ValueError as exc:
            raise ItemSelectionError(str(exc)) from exc
        if len(context) != self.model.context_dim or any(len(v) != self.model.candidate_dim for v in vectors):
            raise ItemSelectionError("live feature dimensions do not match ItemSelector")
        mask = list(decision.card_mask)
        was_training = self.model.training
        self.model.eval()
        try:
            with th.no_grad():
                logits = self.model(th.tensor(context, dtype=th.float32, device=self.device).unsqueeze(0), th.tensor(vectors, dtype=th.float32, device=self.device).unsqueeze(0), th.tensor(mask, dtype=th.bool, device=self.device).unsqueeze(0))[0].detach().cpu()
        finally:
            self.model.train(was_training)
        valid = [i for i, ok in enumerate(mask) if ok]
        if any(not math.isfinite(float(logits[i])) for i in valid):
            raise ItemSelectionError("ItemSelector returned a non-finite valid logit")
        selected = min(valid, key=lambda i: (-float(logits[i]), decision.padded_candidates[i].item_id))
        return self._intent(decision, CardTargetRef.from_features(decision, selected))


def create_strategy(name: str, **kwargs: Any) -> CardSelectionStrategy:
    """正式な四戦略だけを名前から生成する。"""
    factories = {"random": RandomCardStrategy, "card_heuristic_v1": CardHeuristicV1Strategy, "teacher_oracle": TeacherOracleStrategy, "item_selector": ItemSelectionStrategy}
    try:
        return factories[name](**kwargs)
    except KeyError as exc:
        raise ValueError(f"unknown item selection strategy: {name}") from exc
