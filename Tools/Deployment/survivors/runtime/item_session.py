"""ItemSelector session: level-up 候補から choose_card UiIntentV1 を生成する。

ItemSelectorArtifact の TorchScript model を使って候補を採点し、
UiPresentationSnapshotV1 内の typed target に一意対応する UiIntentV1 を返す。
OS input には触れない。
"""
from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping

import numpy as np
import torch as th

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


class ItemSessionError(ValueError):
    """ItemSelector セッションの境界検証失敗。

    候補不一致・target 未解決・snapshot binding 不整合時に送出する。
    """


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
) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
    """ItemDecisionFeatures を model 入力 tensor に変換する。

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
    candidate_vectors.extend([[0.0] * candidate_dim] * padding_count)
    mask = list(card_mask) + [False] * padding_count

    context_tensor = th.tensor([context_vector], dtype=th.float32)
    cand_tensor = th.tensor([candidate_vectors], dtype=th.float32)
    mask_tensor = th.tensor([mask], dtype=th.bool)
    return context_tensor, cand_tensor, mask_tensor


def _resolve_winner_target(
    winner_index: int,
    item_context: ItemDecisionFeatures,
    ui_presentation: UiPresentationSnapshotV1,
) -> UiCandidateTargetV1:
    """勝者 index を UiPresentationSnapshotV1 内の typed target に解決する。

    padded_candidates の順序で winner_index を探し、UiPresentationSnapshotV1 内の
    choice_id / choice_index が同一 snapshot の候補に一意対応しない場合は ItemSessionError。
    """
    padded = item_context.padded_candidates
    if not (0 <= winner_index < len(padded)):
        raise ItemSessionError(f"winner_index {winner_index} out of range")
    winner_candidate = padded[winner_index]
    if winner_candidate.is_padding:
        raise ItemSessionError("ItemSelector chose a padding slot")

    # UiPresentationSnapshotV1 で同じ choice_id / choice_index を持つ target を探す
    matched: UiCandidateTargetV1 | None = None
    for ui_cand in ui_presentation.candidates:
        if (
            ui_cand.choice_id == winner_candidate.item_id
            and ui_cand.choice_index == winner_index
        ):
            matched = ui_cand
            break
    if matched is None:
        raise ItemSessionError(
            f"winner candidate {winner_candidate.item_id!r} not found in UiPresentationSnapshotV1"
        )
    return matched


class ItemSession:
    """ItemSelectorArtifact を使って level-up 候補を採点するセッション。

    TorchScript model でスコアを計算し、最高スコアの有効候補に対応する
    UiIntentV1 (kind=CHOOSE_CARD) を返す。
    """

    def __init__(self, artifact: Any) -> None:
        """artifact の nmax と feature_schema を記録する。

        artifact は ItemSelectorArtifact 互換 (nmax / feature_schema / predict() が必要)。
        """
        if artifact is None:
            raise ItemSessionError("ItemSession requires a non-None artifact")
        self._artifact = artifact
        self._nmax: int = int(artifact.nmax)
        self._feature_schema: str = str(artifact.feature_schema)
        if self._nmax <= 0:
            raise ItemSessionError("ItemSelector nmax must be positive")

    def decide(
        self,
        item_context: ItemDecisionFeatures,
        ui_presentation: UiPresentationSnapshotV1,
        *,
        decision_policy_id: str,
        decision_rule_id: str,
        decision_config_hash: str,
    ) -> UiIntentV1:
        """候補を採点して CHOOSE_CARD UiIntentV1 を返す。

        feature_schema が artifact と一致しない場合や target 解決失敗の場合は
        ItemSessionError を送出する。caller は stop に変換すること。
        """
        if not isinstance(item_context, ItemDecisionFeatures):
            raise ItemSessionError("item_context must be ItemDecisionFeatures")
        if not isinstance(ui_presentation, UiPresentationSnapshotV1):
            raise ItemSessionError("ui_presentation must be UiPresentationSnapshotV1")
        if item_context.feature_schema != self._feature_schema:
            raise ItemSessionError(
                f"feature_schema mismatch: {item_context.feature_schema!r} != {self._feature_schema!r}"
            )

        context_t, cand_t, mask_t = _encode_item_decision(item_context, self._nmax)
        scaled_logits = self._artifact.predict(context_t, cand_t, mask_t)
        # masked_fill(-inf) 済みなので argmax は有効スロットを選ぶ
        winner_index = int(scaled_logits[0].argmax().item())

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
        return intent
