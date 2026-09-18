"""``resolve_calibrated_winner`` の Protocol 準拠 artifact に対する契約テスト。

flatten_feature / encode_item_decision / calibrated_probabilities は
Deployment の ItemSession 経由で間接的に検証済みのため、ここでは
Common が唯一保証すべき orchestration 契約（schema check・predict 例外の
正規化・argmax 選定）だけを、Deployment/Training どちらの型でもない
最小 Protocol fake で検証する。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from reinbalance_survivors_contracts import (
    CandidateFeatures,
    ItemDecisionFeatures,
    ItemSelectorDecisionError,
    resolve_calibrated_winner,
)

NMAX = 3
FEATURE_SCHEMA = "context_only_v1"


@dataclass
class _FakeArtifact:
    """``CalibratedItemSelectorArtifact`` Protocol だけを満たす最小 fake。"""

    logits: tuple[float, ...]
    nmax: int = NMAX
    feature_schema: str = FEATURE_SCHEMA
    temperature: float = 1.0
    predict_error: Exception | None = None

    def predict(
        self, context_features: np.ndarray, candidate_features: np.ndarray, candidate_mask: np.ndarray
    ) -> np.ndarray:
        if self.predict_error is not None:
            raise self.predict_error
        return np.asarray([self.logits], dtype=np.float32)


def _candidate(item_id: str) -> CandidateFeatures:
    return CandidateFeatures(
        kind="item_card",
        item_id=item_id,
        new_level=1,
        owned=False,
        is_new=True,
        is_evolve=False,
        is_union=False,
        has_prerequisite=False,
        slot_capacity=1,
    )


def _item_context(*, feature_schema: str = FEATURE_SCHEMA) -> ItemDecisionFeatures:
    candidates = [_candidate("item_a"), _candidate("item_b"), _candidate("item_c")]
    return ItemDecisionFeatures(
        decision_id="s" * 64,
        feature_schema=feature_schema,
        elapsed_time=30.0,
        level=3,
        hp_ratio=0.8,
        xp_ratio=0.5,
        weapon_slots=(1, 0, 0, 0, 0, 0),
        passive_slots=(0, 0, 0, 0, 0, 0),
        empty_slot_count=11,
        evolution_readiness=0.0,
        choice_count=3,
        card_mask=(True, True, True),
        fallback_kind="chicken",
        ui_state_validity=0.9,
        ui_state_age=0.1,
        candidates=candidates,
        max_item_cards=NMAX,
    )


def test_resolves_argmax_winner_with_calibrated_confidence():
    winner = resolve_calibrated_winner(_item_context(), _FakeArtifact(logits=(0.1, 5.0, 0.2)))
    assert winner.winner_index == 1
    assert winner.confidence == pytest.approx(0.9847, abs=1e-3)
    assert winner.probabilities.shape == (NMAX,)


def test_feature_schema_mismatch_is_rejected():
    artifact = _FakeArtifact(logits=(0.0, 0.0, 0.0), feature_schema="context_danger_v1")
    with pytest.raises(ItemSelectorDecisionError, match="feature_schema mismatch"):
        resolve_calibrated_winner(_item_context(), artifact)


def test_arbitrary_predict_exception_is_normalized():
    artifact = _FakeArtifact(logits=(0.0, 0.0, 0.0), predict_error=RuntimeError("onnx session closed"))
    with pytest.raises(ItemSelectorDecisionError, match="ItemSelector inference failed"):
        resolve_calibrated_winner(_item_context(), artifact)
