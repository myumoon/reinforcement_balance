"""ItemSelector session: level-up 候補から choose_card UiIntentV1 を生成する。

ONNX ItemSelector で候補を採点し、artifact 側の calibrated confidence gate を
通過した場合にだけ、UiPresentationSnapshotV1 内の typed target へ一意対応する
UiIntentV1 を返す。gate 未達は intent を作らず caller に no_op を選ばせる。
target の validity / semantic_kind / 一意 binding も併せて検証するため、
無効な card や曖昧な候補集合に対して click 指示が生成されることはない。
OS input には触れない。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from reinbalance_survivors_contracts.item_decision import ItemDecisionFeatures
from reinbalance_survivors_contracts.item_selector_decision import (
    ItemSelectorDecisionError,
)
from reinbalance_survivors_contracts.item_selector_decision import (
    flatten_feature as _shared_flatten_feature,
)
from reinbalance_survivors_contracts.item_selector_decision import (
    resolve_calibrated_winner,
)
from reinbalance_survivors_contracts.ui_intent import (
    ContractValidationError,
    DecisionOwner,
    UiIntentKind,
    UiIntentV1,
)
from reinbalance_survivors_contracts.ui_policy import (
    ButtonSemantic,
    NonModelUiPolicyConfigV1,
    UiPolicyInputV1,
    decide_non_model_ui_intent,
)
from survivors.perception_snapshot import (
    UiCandidateTargetV1,
    UiPresentationSnapshotV1,
)

# ItemSelector が選べるのは item card だけ。fallback reward は non-model policy の所管。
_SELECTABLE_SEMANTIC_KIND = "item_card"
_OWNED_META_CAPABILITIES = frozenset(value.value for value in ButtonSemantic)


class ItemSessionError(ValueError):
    """ItemSelector セッションの境界検証失敗。

    候補不一致・target 未解決・validity 違反・snapshot binding 不整合で送出する。
    caller はこれを stop decision へ変換すること。
    """


@dataclass(frozen=True)
class ItemDecisionOutcome:
    """ItemSession の 1 回分の決定結果。

    intent が None の場合は confidence gate 未達であり、caller は no_op を返す。
    intent がある場合だけ choose_card を実行してよい。
    """

    intent: UiIntentV1 | None
    confidence: float
    reason: str


def validate_ui_capability_owners(config: NonModelUiPolicyConfigV1) -> None:
    """全 reachable meta capability に共有 decision owner があることを検証する。

    やさしい説明: item/fallback/card の統合前に、未知 semantic を誰の担当か不明なまま残しません。
    """
    if not isinstance(config, NonModelUiPolicyConfigV1):
        raise ItemSessionError("ui policy config must be NonModelUiPolicyConfigV1")
    unknown = set(config.meta_priority) - _OWNED_META_CAPABILITIES
    if unknown:
        raise ItemSessionError(
            f"UI capability owner is undefined for {sorted(unknown)!r}"
        )


def _flatten_feature(value: Any, *, label: str) -> list[float]:
    """wire representation を共有 Common 実装で float vector に展開する。

    Training 側と byte-identical な変換ロジックは
    ``reinbalance_survivors_contracts.item_selector_decision`` が唯一の source of truth。
    ここでは呼び出し規約（例外型が ItemSessionError であること）だけを維持する。
    """
    try:
        return _shared_flatten_feature(value, label=label)
    except ItemSelectorDecisionError as exc:
        raise ItemSessionError(str(exc)) from exc


def _resolve_winner_target(
    winner_index: int,
    item_context: ItemDecisionFeatures,
    ui_presentation: UiPresentationSnapshotV1,
) -> UiCandidateTargetV1:
    """勝者 index を UiPresentationSnapshotV1 内の typed target に一意解決する。

    choice_id / choice_index が同一 snapshot 内の有効な item_card target へ
    ちょうど 1 件対応する場合だけ target を返す。0 件・複数件・validity=false・
    fallback semantic のいずれでも ItemSessionError にし、caller が stop を返す。
    """
    padded = item_context.padded_candidates
    if not (0 <= winner_index < len(padded)):
        raise ItemSessionError(f"winner_index {winner_index} out of range")
    winner_candidate = padded[winner_index]
    if winner_candidate.is_padding:
        raise ItemSessionError("ItemSelector chose a padding slot")

    matches = [
        ui_cand
        for ui_cand in ui_presentation.candidates
        if ui_cand.choice_id == winner_candidate.item_id
        and ui_cand.choice_index == winner_index
    ]
    if not matches:
        raise ItemSessionError(
            f"winner candidate {winner_candidate.item_id!r} not found in UiPresentationSnapshotV1"
        )
    if len(matches) > 1:
        raise ItemSessionError(
            f"winner candidate {winner_candidate.item_id!r} does not bind uniquely"
        )
    matched = matches[0]
    if matched.validity is not True:
        raise ItemSessionError(
            f"winner target {matched.choice_id!r} is marked invalid by perception"
        )
    if matched.semantic_kind != _SELECTABLE_SEMANTIC_KIND:
        raise ItemSessionError(
            f"winner target semantic_kind {matched.semantic_kind!r} is not selectable by ItemSelector"
        )
    return matched


class ItemSession:
    """ItemSelector artifact を使って level-up 候補を採点するセッション。

    artifact の temperature で較正した確率が confidence_threshold 以上で、かつ
    target が有効かつ一意に解決できる場合だけ UiIntentV1 (kind=CHOOSE_CARD) を返す。
    それ以外は intent を作らず、理由を添えて caller へ返す。
    """

    def __init__(self, artifact: Any) -> None:
        """artifact の nmax / feature_schema / 較正パラメータを記録する。

        artifact は OnnxItemSelector 互換 (nmax / feature_schema / temperature /
        confidence_threshold / predict() が必要)。
        """
        if artifact is None:
            raise ItemSessionError("ItemSession requires a non-None artifact")
        self._artifact = artifact
        self._nmax: int = artifact.nmax
        self._feature_schema: str = artifact.feature_schema
        self._temperature: float = artifact.temperature
        self._confidence_threshold: float = artifact.confidence_threshold

    @property
    def confidence_threshold(self) -> float:
        """choose_card を許可する最低確信度を返す。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        return self._confidence_threshold

    def decide(
        self,
        item_context: ItemDecisionFeatures,
        ui_presentation: UiPresentationSnapshotV1,
        *,
        decision_policy_id: str,
        decision_rule_id: str,
        decision_config_hash: str,
        fallback_input: UiPolicyInputV1 | None = None,
        ui_policy_config: NonModelUiPolicyConfigV1 | None = None,
    ) -> ItemDecisionOutcome:
        """候補を採点して CHOOSE_CARD の可否を返す。

        confidence gate 未達なら intent=None を返し、caller は no_op にする。
        feature_schema 不一致・target 解決失敗は ItemSessionError を送出し、
        caller は stop に変換すること。
        """
        if (fallback_input is None) != (ui_policy_config is None):
            raise ItemSessionError(
                "fallback_input and ui_policy_config must be supplied together"
            )
        if not isinstance(item_context, ItemDecisionFeatures):
            raise ItemSessionError("item_context must be ItemDecisionFeatures")
        if not isinstance(ui_presentation, UiPresentationSnapshotV1):
            raise ItemSessionError("ui_presentation must be UiPresentationSnapshotV1")
        if item_context.decision_id != ui_presentation.snapshot_id:
            raise ItemSessionError(
                "item_context decision_id does not match current UI snapshot"
            )
        if item_context.feature_schema != self._feature_schema:
            raise ItemSessionError(
                f"feature_schema mismatch: {item_context.feature_schema!r} != {self._feature_schema!r}"
            )

        try:
            winner = resolve_calibrated_winner(item_context, self._artifact)
        except ItemSelectorDecisionError as exc:
            raise ItemSessionError(str(exc)) from exc
        winner_index = winner.winner_index
        confidence = winner.confidence

        if confidence < self._confidence_threshold:
            # gate 未達 — card を選ばず次 tick へ委ねる。stop ではなく no_op が安全側。
            fallback_intent = (
                decide_non_model_ui_intent(fallback_input, ui_policy_config)
                if fallback_input is not None and ui_policy_config is not None
                else None
            )
            return ItemDecisionOutcome(
                intent=fallback_intent,
                confidence=confidence,
                reason=(
                    f"item confidence {confidence:.4f} below threshold "
                    f"{self._confidence_threshold:.4f}"
                    + (
                        f"; shared fallback={fallback_intent.kind.value}"
                        if fallback_intent is not None
                        else ""
                    )
                ),
            )

        target = _resolve_winner_target(winner_index, item_context, ui_presentation)

        try:
            intent = UiIntentV1(
                kind=UiIntentKind.CHOOSE_CARD,
                semantic_action="choose_card",
                decision_owner=DecisionOwner.ITEM_SELECTOR_SESSION,
                source_snapshot_hash=ui_presentation.snapshot_id,
                source_frame_hash=ui_presentation.frame_id,
                source_content_hash=ui_presentation.source_content_hash,
                ui_state_key=ui_presentation.ui_state_key,
                target_index=target.choice_index,
                candidate_set_hash=ui_presentation.candidate_set_hash,
                inventory_hash=ui_presentation.inventory_hash,
                decision_policy_id=decision_policy_id,
                decision_rule_id=decision_rule_id,
                decision_config_hash=decision_config_hash,
            )
        except ContractValidationError as exc:
            raise ItemSessionError(f"UiIntentV1 construction failed: {exc}") from exc
        return ItemDecisionOutcome(
            intent=intent, confidence=confidence, reason="item confidence gate passed"
        )
