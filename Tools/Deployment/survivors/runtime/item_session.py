"""ItemSelector session: level-up 候補から choose_card UiIntentV1 を生成する。

ONNX ItemSelector で候補を採点し、artifact 側の calibrated confidence gate を
通過した場合にだけ、UiPresentationSnapshotV1 内の typed target へ一意対応する
UiIntentV1 を返す。gate 未達は intent を作らず caller に no_op を選ばせる。
target の validity / semantic_kind / 一意 binding も併せて検証するため、
無効な card や曖昧な候補集合に対して click 指示が生成されることはない。
OS input には触れない。
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from reinbalance_survivors_contracts.item_decision import (
    CandidateFeatures,
    ItemDecisionFeatures,
)
from reinbalance_survivors_contracts.ui_intent import (
    ContractValidationError,
    DecisionOwner,
    UiIntentKind,
    UiIntentV1,
)
from survivors.perception_snapshot import (
    UiCandidateTargetV1,
    UiPresentationSnapshotV1,
)

# ItemSelector が選べるのは item card だけ。fallback reward は non-model policy の所管。
_SELECTABLE_SEMANTIC_KIND = "item_card"


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


def _flatten_feature(value: Any, *, label: str) -> list[float]:
    """wire representation を Training と同じルールで float vector に展開する。

    Training 側 _flatten_feature と byte-identical な変換を行う。
    Training package を import せずに共有 wire 契約だけを使う。
    """
    if isinstance(value, bool):
        return [1.0 if value else 0.0]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if not math.isfinite(number):
            raise ItemSessionError(f"{label} must contain finite values")
        return [number]
    if isinstance(value, str):
        if not value:
            raise ItemSessionError(f"{label} strings must be non-empty")
        # Training の _stable_string_feature と同一実装: SHA-256 先頭 64 bit → [-1, 1]
        digest = hashlib.sha256(value.encode("utf-8")).digest()
        integer = int.from_bytes(digest[:8], byteorder="big", signed=False)
        return [(integer / float((1 << 64) - 1)) * 2.0 - 1.0]
    if isinstance(value, Mapping):
        result: list[float] = []
        for key in value:
            if not isinstance(key, str):
                raise ItemSessionError(f"{label} mapping keys must be strings")
            result.extend(_flatten_feature(value[key], label=f"{label}.{key}"))
        return result
    if isinstance(value, (list, tuple)):
        result = []
        for index, item in enumerate(value):
            result.extend(_flatten_feature(item, label=f"{label}[{index}]"))
        return result
    raise ItemSessionError(f"{label} contains unsupported feature type")


def _encode_item_decision(
    item_context: ItemDecisionFeatures,
    nmax: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ItemDecisionFeatures を model 入力配列に変換する。

    Training の encode_item_selector_row と同じ wire → float 変換を行う。
    context: [1, context_dim], candidates: [1, nmax, candidate_dim], mask: [1, nmax]
    """
    wire = item_context.to_wire()
    raw_context = wire["context_features"]
    raw_candidates = wire["candidates"]
    card_mask = raw_context["card_mask"]

    context_vector = _flatten_feature(raw_context, label="context_features")

    # candidate ごとに schema_version を除いた public fields を展開する
    candidate_vectors: list[list[float]] = []
    for idx, cand in enumerate(raw_candidates):
        public = {k: v for k, v in cand.items() if k != "schema_version"}
        candidate_vectors.append(_flatten_feature(public, label=f"candidate[{idx}]"))

    candidate_dim = len(candidate_vectors[0]) if candidate_vectors else 0
    # nmax へパディング
    padding_count = nmax - len(raw_candidates)
    if padding_count < 0:
        raise ItemSessionError("candidate count exceeds ItemSelector nmax")
    candidate_vectors.extend([[0.0] * candidate_dim for _ in range(padding_count)])
    mask = list(card_mask) + [False] * (nmax - len(card_mask))
    if len(mask) != nmax:
        raise ItemSessionError("card_mask width does not match ItemSelector nmax")

    context_array = np.asarray([context_vector], dtype=np.float32)
    candidate_array = np.asarray([candidate_vectors], dtype=np.float32)
    mask_array = np.asarray([mask], dtype=bool)
    return context_array, candidate_array, mask_array


def _calibrated_probabilities(
    scaled_logits: np.ndarray, mask: np.ndarray, temperature: float
) -> np.ndarray:
    """有効スロットだけで softmax を取り、較正済み確率分布を返す。

    argmax の raw logit をそのまま信用すると、全候補が等価な場合でも
    確信度 1.0 相当として扱ってしまう。masked slot を除外した確率にしてから
    confidence gate へ渡す。
    """
    logits = np.asarray(scaled_logits, dtype=np.float64).reshape(-1)
    valid = np.asarray(mask, dtype=bool).reshape(-1)
    if logits.shape != valid.shape:
        raise ItemSessionError("ItemSelector logits and mask shapes disagree")
    if not valid.any():
        raise ItemSessionError("ItemSelector received no valid candidate slot")
    if not np.all(np.isfinite(logits[valid])):
        raise ItemSessionError("ItemSelector produced non-finite valid logits")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ItemSessionError("ItemSelector temperature must be positive and finite")

    calibrated = logits / float(temperature)
    calibrated = np.where(valid, calibrated, -np.inf)
    shifted = calibrated - np.max(calibrated[valid])
    exponentials = np.where(valid, np.exp(shifted), 0.0)
    total = float(exponentials.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise ItemSessionError("ItemSelector confidence distribution is degenerate")
    return exponentials / total


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
        self._nmax: int = int(artifact.nmax)
        self._feature_schema: str = str(artifact.feature_schema)
        if self._nmax <= 0:
            raise ItemSessionError("ItemSelector nmax must be positive")
        try:
            self._temperature = float(artifact.temperature)
            self._confidence_threshold = float(artifact.confidence_threshold)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ItemSessionError(
                f"ItemSelector artifact lacks calibration parameters: {exc}"
            ) from exc
        if not math.isfinite(self._temperature) or self._temperature <= 0.0:
            raise ItemSessionError("ItemSelector temperature must be positive and finite")
        if not math.isfinite(self._confidence_threshold) or not (
            0.0 <= self._confidence_threshold <= 1.0
        ):
            raise ItemSessionError("ItemSelector confidence_threshold must be within [0, 1]")

    @property
    def confidence_threshold(self) -> float:
        """choose_card を許可する最低確信度を返す。"""
        return self._confidence_threshold

    def decide(
        self,
        item_context: ItemDecisionFeatures,
        ui_presentation: UiPresentationSnapshotV1,
        *,
        decision_policy_id: str,
        decision_rule_id: str,
        decision_config_hash: str,
    ) -> ItemDecisionOutcome:
        """候補を採点して CHOOSE_CARD の可否を返す。

        confidence gate 未達なら intent=None を返し、caller は no_op にする。
        feature_schema 不一致・target 解決失敗は ItemSessionError を送出し、
        caller は stop に変換すること。
        """
        if not isinstance(item_context, ItemDecisionFeatures):
            raise ItemSessionError("item_context must be ItemDecisionFeatures")
        if not isinstance(ui_presentation, UiPresentationSnapshotV1):
            raise ItemSessionError("ui_presentation must be UiPresentationSnapshotV1")
        if item_context.feature_schema != self._feature_schema:
            raise ItemSessionError(
                f"feature_schema mismatch: {item_context.feature_schema!r} != {self._feature_schema!r}"
            )

        context_a, cand_a, mask_a = _encode_item_decision(item_context, self._nmax)
        try:
            scaled_logits = np.asarray(
                self._artifact.predict(context_a, cand_a, mask_a), dtype=np.float64
            )
        except ItemSessionError:
            raise
        except Exception as exc:  # noqa: BLE001  # adapter ごとに例外型が異なる
            raise ItemSessionError(f"ItemSelector inference failed: {exc}") from exc
        if scaled_logits.shape != (1, self._nmax):
            raise ItemSessionError("ItemSelector output shape mismatch")

        probabilities = _calibrated_probabilities(
            scaled_logits[0], mask_a[0], self._temperature
        )
        winner_index = int(np.argmax(probabilities))
        confidence = float(probabilities[winner_index])

        if confidence < self._confidence_threshold:
            # gate 未達 — card を選ばず次 tick へ委ねる。stop ではなく no_op が安全側。
            return ItemDecisionOutcome(
                intent=None,
                confidence=confidence,
                reason=(
                    f"item confidence {confidence:.4f} below threshold "
                    f"{self._confidence_threshold:.4f}"
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
