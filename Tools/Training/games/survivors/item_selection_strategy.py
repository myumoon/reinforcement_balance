"""Survivors ItemSelector を live card choice へ束縛する推論戦略を定義する。

学習用の teacher target や split を使わず、画面から得た ``ItemDecisionFeatures`` の候補だけを
同じ feature encoder で採点して、UE5 が受け付ける item/choice ID を返す。
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch as th

from reinbalance_survivors_contracts.item_decision import ItemDecisionFeatures

from games.survivors.item_selector_model import ItemSelector
from games.survivors.item_selector_trainer import (
    _candidate_vector,
    _flatten_feature,
    _wire_candidate_identity,
)


class ItemSelectionError(ValueError):
    """live choice feature と選択結果の束縛違反を表す。

    不完全な UI payload や model shape の不一致を fallback 選択へ変換せず、その decision を停止する。
    """


@dataclass(frozen=True, slots=True)
class ItemSelectionDecision:
    """一つの level-up decision に対する model 選択を保持する。

    ``choice_id`` は candidate wire の identity からのみ取り、index は監査用で endpoint の入力には使わない。
    """

    decision_id: str
    choice_id: str
    candidate_index: int
    candidate_logits: tuple[float, ...]


class ItemSelectionStrategy:
    """固定済み ItemSelector を一つの live card choice へ適用する。

    candidate index を tie break に使わず choice ID の辞書順で決めるため、UI の候補表示順が変わっても同じ card を選ぶ。
    """

    def __init__(self, model: ItemSelector, *, device: th.device | str = "cpu") -> None:
        """推論専用 model と device を保持する。

        学習中の任意 module や checkpoint path は受け取らず、binding 済み ``ItemSelector`` だけを採点器にする。
        """
        if not isinstance(model, ItemSelector):
            raise TypeError("model must be an ItemSelector")
        self.model = model.to(device)
        self.device = th.device(device)

    def _encode(
        self, features: ItemDecisionFeatures | Mapping[str, Any]
    ) -> tuple[str, tuple[str, ...], th.Tensor, th.Tensor, th.Tensor]:
        """共有 item decision wire を model 入力 tensor と choice identity へ変換する。

        teacher soft target や reliability を作らず、feature wire の card mask と candidate identity だけを閉ループ入力として使う。
        """
        try:
            decision = (
                features
                if isinstance(features, ItemDecisionFeatures)
                else ItemDecisionFeatures.from_wire(features)
            )
            wire = decision.to_wire()
        except (TypeError, ValueError) as exc:
            raise ItemSelectionError(f"invalid item decision features: {exc}") from exc
        raw_context = wire.get("context_features")
        raw_candidates = wire.get("candidates")
        if not isinstance(raw_context, Mapping) or not isinstance(raw_candidates, list):
            raise ItemSelectionError("item decision feature wire is malformed")
        if not raw_candidates or any(not isinstance(row, Mapping) for row in raw_candidates):
            raise ItemSelectionError("item decision candidates must be a non-empty object list")
        if len(raw_candidates) > decision.max_item_cards:
            raise ItemSelectionError("candidate count exceeds max_item_cards")
        raw_mask = raw_context.get("card_mask")
        if (
            not isinstance(raw_mask, (list, tuple))
            or len(raw_mask) != len(raw_candidates)
            or any(type(value) is not bool for value in raw_mask)
            or not any(raw_mask)
        ):
            raise ItemSelectionError("candidate card_mask is invalid")
        identities = tuple(_wire_candidate_identity(candidate)[0] for candidate in raw_candidates)
        valid_ids = [item_id for item_id, valid in zip(identities, raw_mask, strict=True) if valid]
        if len(valid_ids) != len(set(valid_ids)):
            raise ItemSelectionError("valid candidate choice IDs must be unique")
        try:
            context_vector = _flatten_feature(raw_context, label="context_features")
            candidate_vectors = [
                _candidate_vector(candidate, index=index)
                for index, candidate in enumerate(raw_candidates)
            ]
        except ValueError as exc:
            raise ItemSelectionError(str(exc)) from exc
        candidate_dim = len(candidate_vectors[0])
        if (
            not context_vector
            or len(context_vector) != self.model.context_dim
            or candidate_dim != self.model.candidate_dim
            or any(len(vector) != candidate_dim for vector in candidate_vectors)
        ):
            raise ItemSelectionError("live feature dimensions do not match ItemSelector")
        padding = decision.max_item_cards - len(candidate_vectors)
        candidate_vectors.extend([[0.0] * candidate_dim for _ in range(padding)])
        mask = list(raw_mask) + [False] * padding
        padded_ids = identities + ("__padding__",) * padding
        return (
            decision.decision_id,
            padded_ids,
            th.tensor(context_vector, dtype=th.float32, device=self.device).unsqueeze(0),
            th.tensor(candidate_vectors, dtype=th.float32, device=self.device).unsqueeze(0),
            th.tensor(mask, dtype=th.bool, device=self.device).unsqueeze(0),
        )

    def select(self, features: ItemDecisionFeatures | Mapping[str, Any]) -> ItemSelectionDecision:
        """live feature の valid card から最大 logit の choice を返す。

        評価時だけ一時的に eval mode にし、元の train/eval mode は呼出し後に戻して共有 model の状態を驚かせない。
        """
        decision_id, choice_ids, context, candidates, mask = self._encode(features)
        was_training = self.model.training
        self.model.eval()
        try:
            with th.no_grad():
                logits = self.model(context, candidates, mask)[0].detach().cpu()
        finally:
            self.model.train(was_training)
        valid_indices = [index for index, valid in enumerate(mask[0].tolist()) if valid]
        valid_logits = tuple(float(logits[index]) for index in valid_indices)
        if any(not math.isfinite(value) for value in valid_logits):
            raise ItemSelectionError("ItemSelector returned a non-finite valid logit")
        selected_index = min(
            valid_indices,
            key=lambda index: (-float(logits[index]), choice_ids[index]),
        )
        return ItemSelectionDecision(
            decision_id=decision_id,
            choice_id=choice_ids[selected_index],
            candidate_index=selected_index,
            candidate_logits=valid_logits,
        )
