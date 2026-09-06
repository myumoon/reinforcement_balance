"""ItemSession: calibrated confidence gate・target validity・一意 binding を検証する。

golden mock artifact で ItemSession の全境界テストを実行する。
confidence_threshold / temperature を無視して argmax をそのまま choose_card に
しないこと、validity=false や fallback semantic の target を選ばないことを確認する。
Training の _flatten_feature と byte-identical な変換も併せて確認する。
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from reinbalance_survivors_contracts.canonical_json import canonical_hash
from reinbalance_survivors_contracts.item_decision import (
    CandidateFeatures,
    ItemDecisionFeatures,
)
from reinbalance_survivors_contracts.ui_intent import DecisionOwner, UiIntentKind
from survivors.perception_snapshot import (
    UI_PRESENTATION_SCHEMA_HASH,
    NormalizedRoi,
    UiCandidateTargetV1,
    UiPresentationSnapshotV1,
)
from survivors.runtime.item_session import (
    ItemDecisionOutcome,
    ItemSession,
    ItemSessionError,
    _flatten_feature,
    _resolve_winner_target,
)

NMAX = 3
FEATURE_SCHEMA = "context_only_v1"

_POLICY_ARGS = dict(
    decision_policy_id="item_selector_session.v1",
    decision_rule_id="calibrated_argmax_v1",
    decision_config_hash="d" * 64,
)


@dataclass
class _FakeSelector:
    """固定 logits を返す ItemSelector adapter の代役。

    ONNX を回さずに confidence gate と target 解決だけを検証するために使う。
    """

    logits: tuple[float, ...]
    nmax: int = NMAX
    feature_schema: str = FEATURE_SCHEMA
    temperature: float = 1.0
    confidence_threshold: float = 0.0

    def predict(self, context, candidates, mask) -> np.ndarray:
        """[1, nmax] の scaled logits を返す。"""
        return np.asarray([self.logits], dtype=np.float32)


def _item_candidate(item_id: str) -> CandidateFeatures:
    """テスト用の item card candidate を返す。"""
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


def _make_item_context(
    candidates: list[CandidateFeatures], fallback_kind: str = "chicken"
) -> ItemDecisionFeatures:
    """candidates から ItemDecisionFeatures を構築する。"""
    choice_count = len(candidates)
    card_mask = [True] * choice_count + [False] * (NMAX - choice_count)
    return ItemDecisionFeatures(
        decision_id="test-decision-1",
        feature_schema=FEATURE_SCHEMA,
        elapsed_time=30.0,
        level=3,
        hp_ratio=0.8,
        xp_ratio=0.5,
        weapon_slots=(1, 0, 0, 0, 0, 0),
        passive_slots=(0, 0, 0, 0, 0, 0),
        empty_slot_count=11,
        evolution_readiness=0.0,
        choice_count=choice_count,
        card_mask=tuple(card_mask),
        fallback_kind=fallback_kind,
        ui_state_validity=0.9,
        ui_state_age=0.1,
        candidates=tuple(candidates),
        max_item_cards=NMAX,
    )


def _make_ui_presentation(
    candidates: list[CandidateFeatures],
    *,
    validity: bool = True,
    semantic_kind: str = "item_card",
    choice_id_override: dict[int, str] | None = None,
) -> UiPresentationSnapshotV1:
    """UiPresentationSnapshotV1 を candidates から構築する。

    validity / semantic_kind / choice_id を差し替えて異常系を作れる。
    """
    overrides = choice_id_override or {}
    ui_candidates = tuple(
        UiCandidateTargetV1(
            choice_id=overrides.get(index, candidate.item_id),
            choice_index=index,
            semantic_kind=semantic_kind,
            roi=NormalizedRoi(0.1, 0.1, 0.4, 0.4),
            validity=validity,
            confidence=0.95,
        )
        for index, candidate in enumerate(candidates)
    )
    source_payload = {
        "schema_hash": UI_PRESENTATION_SCHEMA_HASH,
        "snapshot_id": "s" * 64,
        "frame_id": "f" * 64,
        "parser_artifact_hash": "p" * 64,
        "timestamp_ns": 1_000_000_000,
        "screen_state": "level_up_items",
        "candidate_set_hash": "c" * 64,
        "inventory_hash": "i" * 64,
    }
    return UiPresentationSnapshotV1(
        schema_hash=UI_PRESENTATION_SCHEMA_HASH,
        snapshot_id="s" * 64,
        frame_id="f" * 64,
        parser_artifact_hash="p" * 64,
        screen_state="level_up_items",
        candidate_set_hash="c" * 64,
        inventory_hash="i" * 64,
        source_content_hash=canonical_hash(source_payload),
        ui_state_key="uk" * 32,
        candidates=ui_candidates,
        buttons=(),
    )


def _decide(selector: _FakeSelector, candidates, ui) -> ItemDecisionOutcome:
    """ItemSession.decide を既定の policy 引数で呼ぶ。"""
    return ItemSession(selector).decide(
        _make_item_context(candidates), ui, **_POLICY_ARGS
    )


class TestConfidenceGate:
    """[指摘6] calibrated confidence が閾値未満なら choose_card を作らない。"""

    def test_low_confidence_returns_no_intent(self):
        """ほぼ一様な logits で threshold 0.9 なら intent を作らない。"""
        candidates = [_item_candidate("whip"), _item_candidate("knife")]
        selector = _FakeSelector(logits=(0.01, 0.0, 0.0), confidence_threshold=0.9)
        outcome = _decide(selector, candidates, _make_ui_presentation(candidates))
        assert outcome.intent is None
        assert outcome.confidence < 0.9
        assert "below threshold" in outcome.reason

    def test_confidence_zero_point_zero_one_does_not_choose_card(self):
        """confidence が閾値を下回る限り card を選ばない (レビュー再現ケース)。"""
        candidates = [_item_candidate("whip"), _item_candidate("knife")]
        selector = _FakeSelector(logits=(0.0, 0.0, 0.0), confidence_threshold=0.51)
        outcome = _decide(selector, candidates, _make_ui_presentation(candidates))
        assert outcome.intent is None

    def test_high_confidence_returns_intent(self):
        candidates = [_item_candidate("whip"), _item_candidate("knife")]
        selector = _FakeSelector(logits=(9.0, 0.0, 0.0), confidence_threshold=0.9)
        outcome = _decide(selector, candidates, _make_ui_presentation(candidates))
        assert outcome.intent is not None
        assert outcome.intent.kind == UiIntentKind.CHOOSE_CARD
        assert outcome.confidence >= 0.9

    def test_threshold_zero_always_passes(self):
        candidates = [_item_candidate("whip"), _item_candidate("knife")]
        selector = _FakeSelector(logits=(0.1, 0.0, 0.0), confidence_threshold=0.0)
        outcome = _decide(selector, candidates, _make_ui_presentation(candidates))
        assert outcome.intent is not None

    def test_temperature_changes_confidence(self):
        """temperature が大きいほど確率分布は平坦になる。"""
        candidates = [_item_candidate("whip"), _item_candidate("knife")]
        ui = _make_ui_presentation(candidates)
        sharp = _decide(_FakeSelector(logits=(2.0, 0.0, 0.0), temperature=0.5), candidates, ui)
        flat = _decide(_FakeSelector(logits=(2.0, 0.0, 0.0), temperature=8.0), candidates, ui)
        assert sharp.confidence > flat.confidence

    def test_masked_slots_excluded_from_confidence(self):
        """padding slot の logit は確率分布へ寄与しない。"""
        candidates = [_item_candidate("whip"), _item_candidate("knife")]
        selector = _FakeSelector(logits=(5.0, 0.0, 99.0), confidence_threshold=0.0)
        outcome = _decide(selector, candidates, _make_ui_presentation(candidates))
        assert outcome.intent is not None
        assert outcome.intent.target_index == 0

    def test_invalid_threshold_rejected(self):
        with pytest.raises(ItemSessionError, match="confidence_threshold"):
            ItemSession(_FakeSelector(logits=(1.0, 0.0, 0.0), confidence_threshold=1.5))

    def test_invalid_temperature_rejected(self):
        with pytest.raises(ItemSessionError, match="temperature"):
            ItemSession(_FakeSelector(logits=(1.0, 0.0, 0.0), temperature=0.0))


class TestTargetValidity:
    """[指摘6] typed target の validity / semantic / 一意 binding を検証する。"""

    def test_invalid_target_is_rejected(self):
        """validity=false の target は選ばない (レビュー再現ケース)。"""
        candidates = [_item_candidate("whip"), _item_candidate("knife")]
        ui = _make_ui_presentation(candidates, validity=False)
        selector = _FakeSelector(logits=(9.0, 0.0, 0.0))
        with pytest.raises(ItemSessionError, match="marked invalid"):
            _decide(selector, candidates, ui)

    def test_fallback_semantic_target_is_rejected(self):
        """fallback_reward は ItemSelector の所管ではない。"""
        candidates = [_item_candidate("whip"), _item_candidate("knife")]
        ui = _make_ui_presentation(candidates, semantic_kind="fallback_reward")
        selector = _FakeSelector(logits=(9.0, 0.0, 0.0))
        with pytest.raises(ItemSessionError, match="not selectable"):
            _decide(selector, candidates, ui)

    def test_candidate_remap_is_rejected(self):
        """snapshot 側 choice_id がずれていれば stop 相当のエラーにする。"""
        candidates = [_item_candidate("whip"), _item_candidate("knife")]
        ui = _make_ui_presentation(candidates, choice_id_override={0: "unrelated"})
        selector = _FakeSelector(logits=(9.0, 0.0, 0.0))
        with pytest.raises(ItemSessionError, match="not found"):
            _decide(selector, candidates, ui)

    def test_ambiguous_binding_is_rejected(self):
        """同一 choice_id/index が複数あれば一意解決できないので拒否する。"""
        candidates = [_item_candidate("whip")]
        duplicate = UiCandidateTargetV1(
            choice_id="whip",
            choice_index=0,
            semantic_kind="item_card",
            roi=NormalizedRoi(0.1, 0.1, 0.4, 0.4),
            validity=True,
            confidence=0.9,
        )
        stub = SimpleNamespace(candidates=(duplicate, duplicate))
        with pytest.raises(ItemSessionError, match="does not bind uniquely"):
            _resolve_winner_target(0, _make_item_context(candidates), stub)

    def test_padding_slot_choice_is_rejected(self):
        candidates = [_item_candidate("whip")]
        ui = _make_ui_presentation(candidates)
        with pytest.raises(ItemSessionError, match="padding slot"):
            _resolve_winner_target(1, _make_item_context(candidates), ui)


class TestIntentBinding:
    """choose_card intent が snapshot binding を正確に引き継ぐ。"""

    def test_intent_copies_source_binding(self):
        candidates = [_item_candidate("whip"), _item_candidate("knife")]
        ui = _make_ui_presentation(candidates)
        outcome = _decide(_FakeSelector(logits=(9.0, 0.0, 0.0)), candidates, ui)
        intent = outcome.intent
        assert intent is not None
        assert intent.source_snapshot_hash == ui.snapshot_id
        assert intent.source_frame_hash == ui.frame_id
        assert intent.source_content_hash == ui.source_content_hash
        assert intent.ui_state_key == ui.ui_state_key
        assert intent.candidate_set_hash == ui.candidate_set_hash
        assert intent.inventory_hash == ui.inventory_hash

    def test_intent_owner_is_item_selector_session(self):
        candidates = [_item_candidate("whip")]
        outcome = _decide(
            _FakeSelector(logits=(9.0, 0.0, 0.0)), candidates,
            _make_ui_presentation(candidates),
        )
        assert outcome.intent is not None
        assert outcome.intent.decision_owner == DecisionOwner.ITEM_SELECTOR_SESSION

    def test_intent_target_index_matches_winner(self):
        candidates = [_item_candidate("whip"), _item_candidate("knife")]
        outcome = _decide(
            _FakeSelector(logits=(0.0, 9.0, 0.0)), candidates,
            _make_ui_presentation(candidates),
        )
        assert outcome.intent is not None
        assert outcome.intent.target_index == 1


class TestInputContracts:
    """入力型と schema の fail-closed 検証。"""

    def test_feature_schema_mismatch_rejected(self):
        candidates = [_item_candidate("whip")]
        selector = _FakeSelector(logits=(9.0, 0.0, 0.0), feature_schema="other_v1")
        with pytest.raises(ItemSessionError, match="feature_schema mismatch"):
            _decide(selector, candidates, _make_ui_presentation(candidates))

    def test_none_artifact_rejected(self):
        with pytest.raises(ItemSessionError, match="non-None artifact"):
            ItemSession(None)

    def test_wrong_context_type_rejected(self):
        candidates = [_item_candidate("whip")]
        session = ItemSession(_FakeSelector(logits=(9.0, 0.0, 0.0)))
        with pytest.raises(ItemSessionError, match="ItemDecisionFeatures"):
            session.decide(object(), _make_ui_presentation(candidates), **_POLICY_ARGS)

    def test_wrong_presentation_type_rejected(self):
        candidates = [_item_candidate("whip")]
        session = ItemSession(_FakeSelector(logits=(9.0, 0.0, 0.0)))
        with pytest.raises(ItemSessionError, match="UiPresentationSnapshotV1"):
            session.decide(_make_item_context(candidates), object(), **_POLICY_ARGS)

    def test_bad_output_shape_rejected(self):
        candidates = [_item_candidate("whip")]

        class _BadShape(_FakeSelector):
            def predict(self, context, candidates_, mask):
                return np.zeros((1, NMAX + 1), dtype=np.float32)

        with pytest.raises(ItemSessionError, match="output shape mismatch"):
            _decide(_BadShape(logits=(0.0, 0.0, 0.0)), candidates,
                    _make_ui_presentation(candidates))

    def test_non_finite_logits_rejected(self):
        candidates = [_item_candidate("whip"), _item_candidate("knife")]
        selector = _FakeSelector(logits=(float("nan"), 0.0, 0.0))
        with pytest.raises(ItemSessionError, match="non-finite"):
            _decide(selector, candidates, _make_ui_presentation(candidates))


class TestFlattenFeatureParity:
    """Training の _flatten_feature と同一の変換規則であること。"""

    def test_bool_is_one_or_zero(self):
        assert _flatten_feature(True, label="x") == [1.0]
        assert _flatten_feature(False, label="x") == [0.0]

    def test_number_passthrough(self):
        assert _flatten_feature(2.5, label="x") == [2.5]

    def test_string_is_stable_hash_in_unit_range(self):
        first = _flatten_feature("whip", label="x")
        assert first == _flatten_feature("whip", label="x")
        assert -1.0 <= first[0] <= 1.0

    def test_mapping_and_sequence_are_flattened(self):
        assert _flatten_feature({"a": 1, "b": [2, 3]}, label="x") == [1.0, 2.0, 3.0]

    def test_empty_string_rejected(self):
        with pytest.raises(ItemSessionError, match="non-empty"):
            _flatten_feature("", label="x")

    def test_non_finite_number_rejected(self):
        with pytest.raises(ItemSessionError, match="finite"):
            _flatten_feature(float("inf"), label="x")

    def test_unsupported_type_rejected(self):
        with pytest.raises(ItemSessionError, match="unsupported"):
            _flatten_feature(object(), label="x")
