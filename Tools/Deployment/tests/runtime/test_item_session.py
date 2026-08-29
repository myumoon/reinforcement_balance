"""ItemSession: 候補採点・target 解決・UiIntentV1 binding を検証する。

golden mock artifact で ItemSession の全境界テストを実行する。
Training の _flatten_feature と byte-identical な変換を確認する。
"""
from __future__ import annotations

import hashlib
from typing import Any
from dataclasses import dataclass

import numpy as np
import pytest
import torch

from reinbalance_survivors_contracts.item_decision import (
    CandidateFeatures,
    ItemDecisionFeatures,
)
from reinbalance_survivors_contracts.ui_intent import (
    DecisionOwner,
    UiIntentKind,
)
from survivors.perception_snapshot import (
    NormalizedRoi,
    UiCandidateTargetV1,
    UiButtonTargetV1,
    UiPresentationSnapshotV1,
)
from survivors.runtime.item_session import (
    ItemSession,
    ItemSessionError,
    _flatten_feature,
)
from survivors.perception_snapshot import UI_PRESENTATION_SCHEMA_HASH


# ── ヘルパー ──────────────────────────────────────────────────────────────

NMAX = 3
FEATURE_SCHEMA = "context_only_v1"


def _item_candidate(item_id: str, slot_index: int = 0) -> CandidateFeatures:
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


def _make_item_context(candidates: list[CandidateFeatures], fallback_kind: str = "chicken") -> ItemDecisionFeatures:
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
        empty_slot_count=11,  # weapon=(1,0,0,0,0,0)→5 zeros + passive=(0,0,0,0,0,0)→6 zeros
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
) -> UiPresentationSnapshotV1:
    """UiPresentationSnapshotV1 を candidates から構築する。"""
    ui_cands = tuple(
        UiCandidateTargetV1(
            choice_id=c.item_id,
            choice_index=i,
            semantic_kind="item_card",
            roi=NormalizedRoi(0.1, 0.1, 0.4, 0.4),
            validity=True,
            confidence=0.95,
        )
        for i, c in enumerate(candidates)
    )
    source_hash = "s" * 64
    frame_hash = "f" * 64
    cand_hash = "c" * 64
    inv_hash = "i" * 64
    ui_state_key = "uk" * 32

    from reinbalance_survivors_contracts.canonical_json import canonical_hash
    import json

    source_payload = {
        "schema_hash": UI_PRESENTATION_SCHEMA_HASH,
        "snapshot_id": source_hash,
        "frame_id": frame_hash,
        "parser_artifact_hash": "p" * 64,
        "timestamp_ns": 1_000_000_000,
        "screen_state": "level_up_items",
        "candidate_set_hash": cand_hash,
        "inventory_hash": inv_hash,
    }
    source_content_hash = canonical_hash(source_payload)
    return UiPresentationSnapshotV1(
        schema_hash=UI_PRESENTATION_SCHEMA_HASH,
        snapshot_id=source_hash,
        frame_id=frame_hash,
        parser_artifact_hash="p" * 64,
        screen_state="level_up_items",
        candidate_set_hash=cand_hash,
        inventory_hash=inv_hash,
        source_content_hash=source_content_hash,
        ui_state_key=ui_state_key,
        candidates=ui_cands,
        buttons=(),
    )


class _MockItemSelector:
    """ItemSelectorArtifact の interface を実装する golden mock。

    argmax が winner_index を選ぶよう、logit を制御する。
    """

    def __init__(self, nmax: int = NMAX, winner_index: int = 0, feature_schema: str = FEATURE_SCHEMA) -> None:
        self._nmax = nmax
        self._winner_index = winner_index
        self._feature_schema = feature_schema

    @property
    def nmax(self) -> int:
        return self._nmax

    @property
    def feature_schema(self) -> str:
        return self._feature_schema

    def predict(self, context: torch.Tensor, candidates: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """winner_index の slot だけ高 logit を持つ tensor を返す。"""
        batch, nmax = mask.shape
        logits = torch.full((batch, nmax), -1e9)
        for b in range(batch):
            if mask[b, self._winner_index]:
                logits[b, self._winner_index] = 100.0
        return logits.masked_fill(~mask, float("-inf"))


class _OobWinnerMockItemSelector:
    """mask を無視して winner_index を強制するスタブ。padding OOB 検出テスト専用。"""

    def __init__(self, nmax: int = NMAX, winner_index: int = 1, feature_schema: str = FEATURE_SCHEMA) -> None:
        self._nmax = nmax
        self._winner_index = winner_index
        self._feature_schema = feature_schema

    @property
    def nmax(self) -> int:
        return self._nmax

    @property
    def feature_schema(self) -> str:
        return self._feature_schema

    def predict(self, context: "torch.Tensor", candidates: "torch.Tensor", mask: "torch.Tensor") -> "torch.Tensor":
        """mask を無視して winner_index の slot に高 logit を返す。"""
        batch, nmax = mask.shape
        logits = torch.full((batch, nmax), -1e9)
        for b in range(batch):
            logits[b, self._winner_index] = 100.0  # mask check intentionally omitted
        return logits  # masked_fill を適用しない


# ── テスト ──────────────────────────────────────────────────────────────────

class TestFlattenFeature:
    """Training の _flatten_feature と byte-identical な変換を確認する。"""

    def test_bool_true(self):
        assert _flatten_feature(True, label="x") == [1.0]

    def test_bool_false(self):
        assert _flatten_feature(False, label="x") == [0.0]

    def test_int(self):
        assert _flatten_feature(3, label="x") == [3.0]

    def test_float(self):
        assert _flatten_feature(0.5, label="x") == [0.5]

    def test_string_stable_hash(self):
        """SHA-256 先頭 64 bit → [-1, 1] を Training と同じロジックで計算する。"""
        value = "chicken"
        digest = hashlib.sha256(value.encode("utf-8")).digest()
        integer = int.from_bytes(digest[:8], byteorder="big", signed=False)
        expected = [(integer / float((1 << 64) - 1)) * 2.0 - 1.0]
        assert _flatten_feature(value, label="x") == expected

    def test_list(self):
        assert _flatten_feature([1, 2], label="x") == [1.0, 2.0]

    def test_mapping(self):
        result = _flatten_feature({"a": 1.0, "b": 2.0}, label="x")
        assert result == [1.0, 2.0]

    def test_nonfinite_raises(self):
        import math
        with pytest.raises(ItemSessionError, match="finite"):
            _flatten_feature(float("nan"), label="x")

    def test_empty_string_raises(self):
        with pytest.raises(ItemSessionError, match="non-empty"):
            _flatten_feature("", label="x")

    def test_unsupported_type_raises(self):
        with pytest.raises(ItemSessionError, match="unsupported"):
            _flatten_feature(object(), label="x")


class TestItemSessionInit:
    def test_none_artifact_raises(self):
        with pytest.raises(ItemSessionError, match="non-None"):
            ItemSession(None)  # type: ignore[arg-type]

    def test_valid_artifact_accepted(self):
        session = ItemSession(_MockItemSelector())
        assert session._nmax == NMAX


class TestDecide:
    def _make_session(self, winner_index: int = 0) -> ItemSession:
        return ItemSession(_MockItemSelector(winner_index=winner_index))

    def _make_policy_args(self) -> dict:
        return {
            "decision_policy_id": "item_selector_session.v1",
            "decision_rule_id": "argmax_v1",
            "decision_config_hash": "d" * 64,
        }

    def test_returns_choose_card_intent(self):
        candidates = [_item_candidate("whip", 0), _item_candidate("axe", 1)]
        item_ctx = _make_item_context(candidates)
        ui_pres = _make_ui_presentation(candidates)
        session = self._make_session(winner_index=0)
        intent = session.decide(item_ctx, ui_pres, **self._make_policy_args())
        assert intent.kind == UiIntentKind.CHOOSE_CARD
        assert intent.decision_owner == DecisionOwner.ITEM_SELECTOR_SESSION

    def test_winner_target_id_set(self):
        candidates = [_item_candidate("whip", 0), _item_candidate("axe", 1)]
        item_ctx = _make_item_context(candidates)
        ui_pres = _make_ui_presentation(candidates)
        session = self._make_session(winner_index=1)
        intent = session.decide(item_ctx, ui_pres, **self._make_policy_args())
        assert intent.target_id is None  # CHOOSE_CARD forbids target_id; index identifies the card
        assert intent.target_index == 1

    def test_snapshot_binding_in_intent(self):
        """intent の source hash が ui_presentation から来ている。"""
        candidates = [_item_candidate("whip", 0)]
        item_ctx = _make_item_context(candidates)
        ui_pres = _make_ui_presentation(candidates)
        session = self._make_session(winner_index=0)
        intent = session.decide(item_ctx, ui_pres, **self._make_policy_args())
        assert intent.source_snapshot_hash == ui_pres.snapshot_id
        assert intent.source_frame_hash == ui_pres.frame_id
        assert intent.ui_state_key == ui_pres.ui_state_key

    def test_wrong_feature_schema_raises(self):
        candidates = [_item_candidate("whip", 0)]
        item_ctx = _make_item_context(candidates)
        ui_pres = _make_ui_presentation(candidates)
        session = ItemSession(_MockItemSelector(feature_schema="context_danger_v1"))
        with pytest.raises(ItemSessionError, match="feature_schema"):
            session.decide(item_ctx, ui_pres, **self._make_policy_args())

    def test_target_not_in_presentation_raises(self):
        """winner が ui_presentation に存在しない → ItemSessionError。"""
        candidates = [_item_candidate("whip", 0)]
        item_ctx = _make_item_context(candidates)
        # ui_pres には別の候補を渡す
        other = [_item_candidate("garlic", 0)]
        ui_pres = _make_ui_presentation(other)
        session = self._make_session(winner_index=0)
        with pytest.raises(ItemSessionError, match="not found"):
            session.decide(item_ctx, ui_pres, **self._make_policy_args())

    def test_padding_winner_raises(self):
        """argmax が padding slot を選ぶと ItemSessionError。"""
        candidates = [_item_candidate("whip", 0)]
        item_ctx = _make_item_context(candidates)
        ui_pres = _make_ui_presentation(candidates)
        # OOB mock: mask を無視して index=1 を強制し padding slot が選ばれたことを再現する
        session = ItemSession(_OobWinnerMockItemSelector(nmax=NMAX, winner_index=1))
        with pytest.raises(ItemSessionError):
            session.decide(item_ctx, ui_pres, **self._make_policy_args())
