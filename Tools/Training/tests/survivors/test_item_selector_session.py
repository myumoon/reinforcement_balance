"""ItemSelectorArtifactSession の numpy/torch bridge と confidence gate 契約を検証する。

resolve_calibrated_winner 自体の argmax/温度較正ロジックは
Tools/Common/tests/test_item_selector_decision.py が既に検証済みのため、ここでは
このセッション固有の責務（torch tensor bridge・feature_schema 検証・confidence
gate・UiIntentV1 組み立て）だけを、torch tensor I/O の最小 fake artifact で確認する。
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch as th

from reinbalance_survivors_contracts.item_decision import (
    CandidateFeatures,
    ItemDecisionFeatures,
)
from reinbalance_survivors_contracts.ui_intent import DecisionOwner, UiIntentKind

from games.survivors.item_selector_session import (
    ItemSelectorArtifactSession,
    ItemSelectorSessionError,
)

NMAX = 2
FEATURE_SCHEMA = "context_only_v1"


@dataclass
class _TorchFakeArtifact:
    """``ItemSelectorArtifact`` 互換の最小 fake（torch tensor I/O）。"""

    logits: tuple[float, ...]
    nmax: int = NMAX
    feature_schema: str = FEATURE_SCHEMA
    temperature: float = 1.0
    confidence_threshold: float = 0.5
    predict_error: Exception | None = None

    def predict(
        self, context_features: th.Tensor, candidate_features: th.Tensor, candidate_mask: th.Tensor
    ) -> th.Tensor:
        """torch tensor 入力であることを確認してから logits を返すか、predict_error を送出する。

        やさしい説明: このセッション固有の torch tensor bridge 契約を fake 側でも強制します。
        """
        if self.predict_error is not None:
            raise self.predict_error
        assert isinstance(context_features, th.Tensor)
        assert isinstance(candidate_features, th.Tensor)
        assert candidate_mask.dtype == th.bool
        return th.as_tensor([self.logits], dtype=th.float32)


def _candidate(item_id: str) -> CandidateFeatures:
    """item_id 以外を固定値にした最小 CandidateFeatures を返す。

    やさしい説明: winner 選定テストでは候補間の差を item_id だけに絞ります。
    """
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


def _item_context() -> ItemDecisionFeatures:
    """2 候補 (wand/knife) の ItemDecisionFeatures を返す。

    やさしい説明: confidence gate / feature_schema テストが共有する固定 context です。
    """
    candidates = [_candidate("wand"), _candidate("knife")]
    return ItemDecisionFeatures(
        decision_id="s" * 64,
        feature_schema=FEATURE_SCHEMA,
        elapsed_time=30.0,
        level=3,
        hp_ratio=0.8,
        xp_ratio=0.5,
        weapon_slots=(1, 0, 0, 0, 0, 0),
        passive_slots=(0, 0, 0, 0, 0, 0),
        empty_slot_count=11,
        evolution_readiness=0.0,
        choice_count=2,
        card_mask=(True, True),
        fallback_kind="chicken",
        ui_state_validity=0.9,
        ui_state_age=0.1,
        candidates=candidates,
        max_item_cards=NMAX,
    )


_HASH_KWARGS = dict(
    source_snapshot_hash="a" * 64,
    source_frame_hash="b" * 64,
    source_content_hash="c" * 64,
    ui_state_key="level_up",
    candidate_set_hash="d" * 64,
    inventory_hash="e" * 64,
    decision_policy_id="policy-v1",
    decision_rule_id="rule-v1",
    decision_config_hash="f" * 64,
)


def test_decide_returns_choose_card_intent_when_confidence_gate_passes():
    """test_decide_returns_choose_card_intent_when_confidence_gate_passes の契約を検証する。

    やさしい説明: confidence gate を通過した winner が CHOOSE_CARD UiIntentV1 になることを確認します。
    """
    session = ItemSelectorArtifactSession(_TorchFakeArtifact(logits=(0.1, 5.0)))
    outcome = session.decide(_item_context(), **_HASH_KWARGS)
    assert outcome.intent is not None
    assert outcome.intent.kind == UiIntentKind.CHOOSE_CARD
    assert outcome.intent.decision_owner == DecisionOwner.ITEM_SELECTOR_SESSION
    assert outcome.intent.target_index == 1
    assert outcome.intent.source_snapshot_hash == _HASH_KWARGS["source_snapshot_hash"]
    assert outcome.intent.decision_config_hash == _HASH_KWARGS["decision_config_hash"]
    assert outcome.confidence == pytest.approx(0.9926, abs=1e-3)


def test_decide_returns_none_intent_when_confidence_gate_fails():
    """test_decide_returns_none_intent_when_confidence_gate_fails の契約を検証する。

    やさしい説明: confidence が threshold 未満なら intent を出さず reason だけ返すことを確認します。
    """
    artifact = _TorchFakeArtifact(logits=(0.1, 0.2), confidence_threshold=0.99)
    session = ItemSelectorArtifactSession(artifact)
    outcome = session.decide(_item_context(), **_HASH_KWARGS)
    assert outcome.intent is None
    assert "below threshold" in outcome.reason


def test_decide_rejects_feature_schema_mismatch():
    """test_decide_rejects_feature_schema_mismatch の契約を検証する。

    やさしい説明: artifact と context の feature_schema が食い違う場合は推論前に拒否します。
    """
    artifact = _TorchFakeArtifact(logits=(0.0, 0.0), feature_schema="context_danger_v1")
    session = ItemSelectorArtifactSession(artifact)
    with pytest.raises(ItemSelectorSessionError, match="feature_schema mismatch"):
        session.decide(_item_context(), **_HASH_KWARGS)


def test_decide_normalizes_predict_exception():
    """test_decide_normalizes_predict_exception の契約を検証する。

    やさしい説明: predict が任意の例外を送出しても ItemSelectorSessionError に正規化されます。
    """
    artifact = _TorchFakeArtifact(logits=(0.0, 0.0), predict_error=RuntimeError("torchscript trap"))
    session = ItemSelectorArtifactSession(artifact)
    with pytest.raises(ItemSelectorSessionError, match="ItemSelector inference failed"):
        session.decide(_item_context(), **_HASH_KWARGS)
