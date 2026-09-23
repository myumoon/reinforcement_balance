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
from reinbalance_survivors_contracts.ui_policy import (
    ButtonOption,
    NonModelUiPolicyConfigV1,
    ScreenState,
    UiPolicyInputV1,
)
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
        """[1, nmax] の scaled logits を返す。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        return np.asarray([self.logits], dtype=np.float32)


def _item_candidate(item_id: str) -> CandidateFeatures:
    """テスト用の item card candidate を返す。

    やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
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


def _make_item_context(
    candidates: list[CandidateFeatures],
    fallback_kind: str = "chicken",
    *,
    decision_id: str = "s" * 64,
) -> ItemDecisionFeatures:
    """candidates から ItemDecisionFeatures を構築する。

    やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
    """
    choice_count = len(candidates)
    card_mask = [True] * choice_count + [False] * (NMAX - choice_count)
    return ItemDecisionFeatures(
        decision_id=decision_id,
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
    choice_index_override: dict[int, int] | None = None,
    validity_override: dict[int, bool] | None = None,
) -> UiPresentationSnapshotV1:
    """UiPresentationSnapshotV1 を candidates から構築する。

    validity / semantic_kind / choice_id / choice_index を差し替えて異常系を作れる。
    """
    overrides = choice_id_override or {}
    index_overrides = choice_index_override or {}
    validity_overrides = validity_override or {}
    ui_candidates = tuple(
        UiCandidateTargetV1(
            choice_id=overrides.get(index, candidate.item_id),
            choice_index=index_overrides.get(index, index),
            semantic_kind=semantic_kind,
            roi=NormalizedRoi(0.1, 0.1, 0.4, 0.4),
            validity=validity_overrides.get(index, validity),
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
    """ItemSession.decide を既定の policy 引数で呼ぶ。

    やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
    """
    return ItemSession(selector).decide(
        _make_item_context(candidates), ui, **_POLICY_ARGS
    )


class TestConfidenceGate:
    """[指摘6] calibrated confidence が閾値未満なら choose_card を作らない。

    やさしい説明: このまとまりが担当する境界を分かりやすく示します。
    """

    def test_low_confidence_returns_no_intent(self):
        """ほぼ一様な logits で threshold 0.9 なら intent を作らない。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        candidates = [_item_candidate("whip"), _item_candidate("knife")]
        selector = _FakeSelector(logits=(0.01, 0.0, 0.0), confidence_threshold=0.9)
        outcome = _decide(selector, candidates, _make_ui_presentation(candidates))
        assert outcome.intent is None
        assert outcome.confidence < 0.9
        assert "below threshold" in outcome.reason

    def test_low_confidence_transitions_to_shared_non_model_fallback(self):
        """threshold 未達時は shared meta policy の intent へ遷移する。

        やさしい説明: ItemSession が reroll 規則を再実装せず Common の結果をそのまま返します。
        """
        candidates = [_item_candidate("whip"), _item_candidate("knife")]
        selector = _FakeSelector(logits=(0.0, 0.0, 0.0), confidence_threshold=0.9)
        config = NonModelUiPolicyConfigV1(
            meta_policy_enabled=True,
            meta_priority=("reroll",),
        )
        policy_input = UiPolicyInputV1(
            source_snapshot_hash="snapshot",
            source_frame_hash="frame",
            source_content_hash="content",
            ui_state_key="state",
            screen_state=ScreenState.LEVEL_UP,
            hp_fraction=1.0,
            button=ButtonOption("reroll", True),
        )
        outcome = ItemSession(selector).decide(
            _make_item_context(candidates),
            _make_ui_presentation(candidates),
            fallback_input=policy_input,
            ui_policy_config=config,
            **_POLICY_ARGS,
        )
        assert outcome.intent is not None
        assert outcome.intent.kind == UiIntentKind.REROLL

    def test_confidence_zero_point_zero_one_does_not_choose_card(self):
        """confidence が閾値を下回る限り card を選ばない (レビュー再現ケース)。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        candidates = [_item_candidate("whip"), _item_candidate("knife")]
        selector = _FakeSelector(logits=(0.0, 0.0, 0.0), confidence_threshold=0.51)
        outcome = _decide(selector, candidates, _make_ui_presentation(candidates))
        assert outcome.intent is None

    def test_high_confidence_returns_intent(self):
        """test_high_confidence_returns_intent の契約を検証する。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        candidates = [_item_candidate("whip"), _item_candidate("knife")]
        selector = _FakeSelector(logits=(9.0, 0.0, 0.0), confidence_threshold=0.9)
        outcome = _decide(selector, candidates, _make_ui_presentation(candidates))
        assert outcome.intent is not None
        assert outcome.intent.kind == UiIntentKind.CHOOSE_CARD
        assert outcome.confidence >= 0.9

    def test_threshold_zero_always_passes(self):
        """test_threshold_zero_always_passes の契約を検証する。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        candidates = [_item_candidate("whip"), _item_candidate("knife")]
        selector = _FakeSelector(logits=(0.1, 0.0, 0.0), confidence_threshold=0.0)
        outcome = _decide(selector, candidates, _make_ui_presentation(candidates))
        assert outcome.intent is not None

    def test_temperature_changes_confidence(self):
        """temperature が大きいほど確率分布は平坦になる。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        candidates = [_item_candidate("whip"), _item_candidate("knife")]
        ui = _make_ui_presentation(candidates)
        sharp = _decide(_FakeSelector(logits=(2.0, 0.0, 0.0), temperature=0.5), candidates, ui)
        flat = _decide(_FakeSelector(logits=(2.0, 0.0, 0.0), temperature=8.0), candidates, ui)
        assert sharp.confidence > flat.confidence

    def test_masked_slots_excluded_from_confidence(self):
        """padding slot の logit は確率分布へ寄与しない。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        candidates = [_item_candidate("whip"), _item_candidate("knife")]
        selector = _FakeSelector(logits=(5.0, 0.0, 99.0), confidence_threshold=0.0)
        outcome = _decide(selector, candidates, _make_ui_presentation(candidates))
        assert outcome.intent is not None
        assert outcome.intent.target_index == 0

class TestTargetValidity:
    """[指摘6] typed target の validity / semantic / 一意 binding を検証する。

    やさしい説明: このまとまりが担当する境界を分かりやすく示します。
    """

    def test_invalid_target_is_rejected(self):
        """validity=false の target は選ばない (レビュー再現ケース)。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        candidates = [_item_candidate("whip"), _item_candidate("knife")]
        ui = _make_ui_presentation(candidates, validity=False)
        selector = _FakeSelector(logits=(9.0, 0.0, 0.0))
        with pytest.raises(ItemSessionError, match="marked invalid"):
            _decide(selector, candidates, ui)

    def test_fallback_semantic_target_is_rejected(self):
        """fallback_reward は ItemSelector の所管ではない。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        candidates = [_item_candidate("whip"), _item_candidate("knife")]
        ui = _make_ui_presentation(candidates, semantic_kind="fallback_reward")
        selector = _FakeSelector(logits=(9.0, 0.0, 0.0))
        with pytest.raises(ItemSessionError, match="not selectable"):
            _decide(selector, candidates, ui)

    def test_candidate_remap_is_rejected(self):
        """snapshot 側 choice_id がずれていれば stop 相当のエラーにする。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        candidates = [_item_candidate("whip"), _item_candidate("knife")]
        ui = _make_ui_presentation(candidates, choice_id_override={0: "unrelated"})
        selector = _FakeSelector(logits=(9.0, 0.0, 0.0))
        with pytest.raises(ItemSessionError, match="not found"):
            _decide(selector, candidates, ui)

    def test_stale_decision_id_is_rejected_even_when_candidates_match(self):
        """別 snapshot の item feature は候補が同じでも拒否する。

        やさしい説明: 古いカード特徴を現在 frame の同名カードへ再束縛する回帰を捕えます。
        """
        candidates = [_item_candidate("whip"), _item_candidate("knife")]
        ui = _make_ui_presentation(candidates)
        context = _make_item_context(candidates, decision_id="d" * 64)
        selector = _FakeSelector(logits=(9.0, 0.0, 0.0))

        with pytest.raises(ItemSessionError, match="decision_id"):
            ItemSession(selector).decide(context, ui, **_POLICY_ARGS)

    def test_ambiguous_binding_is_rejected(self):
        """同一 choice_id/index が複数あれば一意解決できないので拒否する。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
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

    def test_permuted_ui_choice_index_resolves_by_item_identity(self):
        """model slot と異なる UI choice_index でも item_id で解決する。

        やさしい説明: モデルの候補順と画面上のカード番号が違っても、同じアイテムを選べます。
        """
        candidates = [_item_candidate("whip"), _item_candidate("knife")]
        ui = _make_ui_presentation(candidates, choice_index_override={0: 1, 1: 0})
        outcome = _decide(_FakeSelector(logits=(9.0, 0.0, 0.0)), candidates, ui)
        assert outcome.intent is not None
        assert outcome.intent.target_index == 1

    def test_invalid_ui_card_excluded_from_model_slots_still_binds_valid_card(self):
        """無効 UI card を除外して詰めた model slot を valid card へ束縛する。

        やさしい説明: 先頭カードが無効でも、モデルが選んだ次の有効カードを画面の正しい番号で操作します。
        """
        model_candidates = [_item_candidate("knife")]
        ui = _make_ui_presentation(
            [_item_candidate("whip"), _item_candidate("knife")],
            validity_override={0: False},
        )
        outcome = _decide(_FakeSelector(logits=(9.0, 0.0, 0.0)), model_candidates, ui)
        assert outcome.intent is not None
        assert outcome.intent.target_index == 1

    def test_padding_slot_choice_is_rejected(self):
        """test_padding_slot_choice_is_rejected の契約を検証する。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        candidates = [_item_candidate("whip")]
        ui = _make_ui_presentation(candidates)
        with pytest.raises(ItemSessionError, match="padding slot"):
            _resolve_winner_target(1, _make_item_context(candidates), ui)


class TestIntentBinding:
    """choose_card intent が snapshot binding を正確に引き継ぐ。

    やさしい説明: このまとまりが担当する境界を分かりやすく示します。
    """

    def test_intent_copies_source_binding(self):
        """test_intent_copies_source_binding の契約を検証する。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
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
        """test_intent_owner_is_item_selector_session の契約を検証する。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        candidates = [_item_candidate("whip")]
        outcome = _decide(
            _FakeSelector(logits=(9.0, 0.0, 0.0)), candidates,
            _make_ui_presentation(candidates),
        )
        assert outcome.intent is not None
        assert outcome.intent.decision_owner == DecisionOwner.ITEM_SELECTOR_SESSION

    def test_intent_target_index_matches_winner(self):
        """test_intent_target_index_matches_winner の契約を検証する。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        candidates = [_item_candidate("whip"), _item_candidate("knife")]
        outcome = _decide(
            _FakeSelector(logits=(0.0, 9.0, 0.0)), candidates,
            _make_ui_presentation(candidates),
        )
        assert outcome.intent is not None
        assert outcome.intent.target_index == 1


class TestInputContracts:
    """入力型と schema の fail-closed 検証。

    やさしい説明: このまとまりが担当する境界を分かりやすく示します。
    """

    def test_feature_schema_mismatch_rejected(self):
        """test_feature_schema_mismatch_rejected の契約を検証する。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        candidates = [_item_candidate("whip")]
        selector = _FakeSelector(logits=(9.0, 0.0, 0.0), feature_schema="other_v1")
        with pytest.raises(ItemSessionError, match="feature_schema mismatch"):
            _decide(selector, candidates, _make_ui_presentation(candidates))

    def test_none_artifact_rejected(self):
        """test_none_artifact_rejected の契約を検証する。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        with pytest.raises(ItemSessionError, match="non-None artifact"):
            ItemSession(None)

    def test_wrong_context_type_rejected(self):
        """test_wrong_context_type_rejected の契約を検証する。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        candidates = [_item_candidate("whip")]
        session = ItemSession(_FakeSelector(logits=(9.0, 0.0, 0.0)))
        with pytest.raises(ItemSessionError, match="ItemDecisionFeatures"):
            session.decide(object(), _make_ui_presentation(candidates), **_POLICY_ARGS)

    def test_wrong_presentation_type_rejected(self):
        """test_wrong_presentation_type_rejected の契約を検証する。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        candidates = [_item_candidate("whip")]
        session = ItemSession(_FakeSelector(logits=(9.0, 0.0, 0.0)))
        with pytest.raises(ItemSessionError, match="UiPresentationSnapshotV1"):
            session.decide(_make_item_context(candidates), object(), **_POLICY_ARGS)

    def test_predict_receives_only_typed_arrays_without_presentation_roi(self):
        """selector へ mask/schema 準拠配列だけを渡す。

        やさしい説明: UiPresentationSnapshotV1 の ROI は binding にだけ使い、model tensor へ漏らしません。
        """
        seen: dict[str, np.ndarray] = {}

        class _RecordingSelector(_FakeSelector):
            """ItemSession が渡した三入力を記録する selector。

            やさしい説明: 推論値より入力 shape・dtype・mask の実値を観察します。
            """

            def predict(
                self, context: np.ndarray, candidates: np.ndarray, mask: np.ndarray
            ) -> np.ndarray:
                """三入力を保存して固定 logits を返す。

                やさしい説明: production の組み立て結果を変更せず、そのまま assertion へ渡します。
                """
                seen.update(context=context, candidates=candidates, mask=mask)
                return super().predict(context, candidates, mask)

        candidates = [_item_candidate("whip"), _item_candidate("knife")]
        _decide(
            _RecordingSelector(logits=(9.0, 0.0, 0.0)),
            candidates,
            _make_ui_presentation(candidates),
        )
        assert seen["context"].dtype == np.float32 and seen["context"].ndim == 2
        assert seen["candidates"].dtype == np.float32 and seen["candidates"].ndim == 3
        assert seen["mask"].dtype == np.bool_
        np.testing.assert_array_equal(seen["mask"], [[True, True, False]])


class TestFlattenFeatureParity:
    """Training の _flatten_feature と同一の変換規則であること。

    やさしい説明: このまとまりが担当する境界を分かりやすく示します。
    """

    def test_bool_is_one_or_zero(self):
        """test_bool_is_one_or_zero の契約を検証する。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        assert _flatten_feature(True, label="x") == [1.0]
        assert _flatten_feature(False, label="x") == [0.0]

    def test_number_passthrough(self):
        """test_number_passthrough の契約を検証する。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        assert _flatten_feature(2.5, label="x") == [2.5]

    def test_string_is_stable_hash_in_unit_range(self):
        """test_string_is_stable_hash_in_unit_range の契約を検証する。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        first = _flatten_feature("whip", label="x")
        assert first == _flatten_feature("whip", label="x")
        assert -1.0 <= first[0] <= 1.0

    def test_mapping_and_sequence_are_flattened(self):
        """test_mapping_and_sequence_are_flattened の契約を検証する。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        assert _flatten_feature({"a": 1, "b": [2, 3]}, label="x") == [1.0, 2.0, 3.0]

    def test_empty_string_rejected(self):
        """test_empty_string_rejected の契約を検証する。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        with pytest.raises(ItemSessionError, match="non-empty"):
            _flatten_feature("", label="x")

    def test_non_finite_number_rejected(self):
        """test_non_finite_number_rejected の契約を検証する。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        with pytest.raises(ItemSessionError, match="finite"):
            _flatten_feature(float("inf"), label="x")

    def test_unsupported_type_rejected(self):
        """test_unsupported_type_rejected の契約を検証する。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        with pytest.raises(ItemSessionError, match="unsupported"):
            _flatten_feature(object(), label="x")
