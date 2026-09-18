"""Survivors ItemSelectorArtifact を使った Training 側 choose_card セッション。

較正済み ItemSelectorArtifact (TorchScript) で level-up 候補を採点し、Deployment の
ItemSession と同じ confidence gate 契約を Training プロセス側で再現する。UiIntentV1 の
source_snapshot_hash 等は呼び出し側 (parity test) が明示的に渡す値をそのまま使うため、
Deployment の UiPresentationSnapshotV1 binding には依存しない。Training が Deployment
module を import することは 05-01 の契約で禁止されており、この module はそれを満たす。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch as th

from games.survivors.item_selector_artifact import ItemSelectorArtifact
from reinbalance_survivors_contracts.item_decision import ItemDecisionFeatures
from reinbalance_survivors_contracts.item_selector_decision import (
    ItemSelectorDecisionError,
    resolve_calibrated_winner,
)
from reinbalance_survivors_contracts.ui_intent import (
    ContractValidationError,
    DecisionOwner,
    UiIntentKind,
    UiIntentV1,
)


class ItemSelectorSessionError(ValueError):
    """ItemSelectorArtifactSession の境界検証・推論失敗を表す。

    caller はこれを no_op/stop 相当の扱いへ変換すること。
    """


@dataclass(frozen=True)
class ItemSelectorSessionOutcome:
    """ItemSelectorArtifactSession の 1 回分の決定結果。

    intent が None の場合は confidence gate 未達であり、caller は no_op を返す。
    """

    intent: UiIntentV1 | None
    confidence: float
    reason: str


class _NumpyBridgedArtifact:
    """ItemSelectorArtifact (torch tensor I/O) を Common の numpy Protocol へ橋渡しする。

    resolve_calibrated_winner は numpy in/out の CalibratedItemSelectorArtifact
    Protocol だけを要求するため、TorchScript artifact 側の tensor 変換をここに閉じ込め、
    Deployment (ONNX/numpy) 側と同じ Common 実装を Training からも使えるようにする。
    """

    def __init__(self, artifact: ItemSelectorArtifact) -> None:
        self._artifact = artifact
        self.nmax = artifact.nmax
        self.feature_schema = artifact.feature_schema
        self.temperature = artifact.temperature

    def predict(
        self,
        context_features: np.ndarray,
        candidate_features: np.ndarray,
        candidate_mask: np.ndarray,
    ) -> np.ndarray:
        """numpy 入力を CPU float32/bool tensor へ変換して artifact 推論を呼ぶ。"""
        context_t = th.as_tensor(context_features, dtype=th.float32)
        candidate_t = th.as_tensor(candidate_features, dtype=th.float32)
        mask_t = th.as_tensor(candidate_mask, dtype=th.bool)
        with th.no_grad():
            scaled = self._artifact.predict(context_t, candidate_t, mask_t)
        return scaled.detach().cpu().numpy()


class ItemSelectorArtifactSession:
    """較正済み ItemSelectorArtifact で choose_card UiIntentV1 を生成するセッション。

    confidence_threshold 未達の場合は intent=None を返し、caller に no_op を委ねる。
    Deployment 側の non-model UI policy fallback 統合はここでは扱わない
    （test_shared_contract_smoke.py が別途保証済みの parity 範囲であり、重複実装しない）。
    """

    def __init__(self, artifact: ItemSelectorArtifact) -> None:
        """artifact を numpy bridge 越しに保持し、閾値/feature_schema を記録する。"""
        if artifact is None:
            raise ItemSelectorSessionError(
                "ItemSelectorArtifactSession requires a non-None artifact"
            )
        self._bridge = _NumpyBridgedArtifact(artifact)
        self._feature_schema: str = artifact.feature_schema
        self._confidence_threshold: float = artifact.confidence_threshold

    @property
    def confidence_threshold(self) -> float:
        """choose_card を許可する最低確信度を返す。"""
        return self._confidence_threshold

    def decide(
        self,
        item_context: ItemDecisionFeatures,
        *,
        source_snapshot_hash: str,
        source_frame_hash: str,
        source_content_hash: str,
        ui_state_key: str,
        candidate_set_hash: str | None = None,
        inventory_hash: str | None = None,
        decision_policy_id: str | None = None,
        decision_rule_id: str | None = None,
        decision_config_hash: str | None = None,
    ) -> ItemSelectorSessionOutcome:
        """候補を採点して CHOOSE_CARD の可否を返す。

        hash/id 系引数は Deployment の ItemSession.decide と byte-identical な
        UiIntentV1 を組み立てられるよう、呼び出し側 (parity test) が明示的に渡す値を
        そのまま使う。Training 内部で独自に digest を再導出することはしない。
        """
        if not isinstance(item_context, ItemDecisionFeatures):
            raise ItemSelectorSessionError("item_context must be ItemDecisionFeatures")
        if item_context.feature_schema != self._feature_schema:
            raise ItemSelectorSessionError(
                f"feature_schema mismatch: {item_context.feature_schema!r} != {self._feature_schema!r}"
            )

        try:
            winner = resolve_calibrated_winner(item_context, self._bridge)
        except ItemSelectorDecisionError as exc:
            raise ItemSelectorSessionError(str(exc)) from exc
        winner_index = winner.winner_index
        confidence = winner.confidence

        if confidence < self._confidence_threshold:
            # gate 未達 — card を選ばず次 tick へ委ねる。stop ではなく no_op が安全側。
            return ItemSelectorSessionOutcome(
                intent=None,
                confidence=confidence,
                reason=(
                    f"item confidence {confidence:.4f} below threshold "
                    f"{self._confidence_threshold:.4f}"
                ),
            )

        padded = item_context.padded_candidates
        if not (0 <= winner_index < len(padded)):
            raise ItemSelectorSessionError(f"winner_index {winner_index} out of range")
        winner_candidate = padded[winner_index]
        if winner_candidate.is_padding:
            raise ItemSelectorSessionError("ItemSelector chose a padding slot")

        try:
            intent = UiIntentV1(
                kind=UiIntentKind.CHOOSE_CARD,
                semantic_action="choose_card",
                decision_owner=DecisionOwner.ITEM_SELECTOR_SESSION,
                source_snapshot_hash=source_snapshot_hash,
                source_frame_hash=source_frame_hash,
                source_content_hash=source_content_hash,
                ui_state_key=ui_state_key,
                target_index=winner_index,
                candidate_set_hash=candidate_set_hash,
                inventory_hash=inventory_hash,
                decision_policy_id=decision_policy_id,
                decision_rule_id=decision_rule_id,
                decision_config_hash=decision_config_hash,
            )
        except ContractValidationError as exc:
            raise ItemSelectorSessionError(f"UiIntentV1 construction failed: {exc}") from exc
        return ItemSelectorSessionOutcome(
            intent=intent, confidence=confidence, reason="item confidence gate passed"
        )
