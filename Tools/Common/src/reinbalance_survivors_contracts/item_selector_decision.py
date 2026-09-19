"""ItemSelector artifact の calibrated 意思決定コア（Training/Deployment 共有）。

feature flatten・context/candidate encode・temperature calibration・confidence gate
winner 選定という、artifact 推論から choose_card 候補を1つに絞るまでの数値計算を
Training と Deployment の両側で同一実装として持つための唯一の source of truth。
ONNX/TorchScript どちらの backend で ``predict()`` されたかは問わない
（``CalibratedItemSelectorArtifact`` は numpy in/out の Protocol）。

target を実際の UI 要素へ bind する処理（Deployment の perception snapshot binding、
Training の ``ItemDecisionFeatures`` 上の binding）はここに含めない。呼び出し側の
責務であり、この module は ``UiIntentV1`` も ``UiPresentationSnapshotV1`` も知らない。
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

import numpy as np

from .item_decision import ItemDecisionFeatures

__all__ = [
    "ItemSelectorDecisionError",
    "CalibratedItemSelectorArtifact",
    "CalibratedWinner",
    "flatten_feature",
    "encode_item_decision",
    "calibrated_probabilities",
    "resolve_calibrated_winner",
]


class ItemSelectorDecisionError(ValueError):
    """ItemSelector calibrated decision の境界検証・推論失敗を表す。"""


class CalibratedItemSelectorArtifact(Protocol):
    """calibrated decision が要求する artifact の最小 shape。

    numpy in/out の ``predict()`` を持つならば、backend が ONNX Runtime でも
    TorchScript の numpy bridge でもよい。
    """

    nmax: int
    feature_schema: str
    temperature: float

    def predict(
        self,
        context_features: np.ndarray,
        candidate_features: np.ndarray,
        candidate_mask: np.ndarray,
    ) -> Any: ...


@dataclass(frozen=True)
class CalibratedWinner:
    """calibrated 確率分布から選ばれた候補 index と確信度。"""

    winner_index: int
    confidence: float
    probabilities: np.ndarray


def _stable_string_feature(value: str) -> float:
    """categorical wire 文字列を deterministic な有限 scalar へ写像する。

    Python の process-randomized ``hash`` を使わず SHA-256 先頭64 bit を ``[-1,1]`` にする。
    """
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    integer = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return (integer / float((1 << 64) - 1)) * 2.0 - 1.0


def flatten_feature(value: Any, *, label: str) -> list[float]:
    """共有 wire の named/list feature を schema 定義順の float vector にする。

    bool、数値、文字列、nested sequence/mapping だけを受理し、教師 object の
    自由形式入力面を encoder に作らない。
    """
    if isinstance(value, bool):
        return [1.0 if value else 0.0]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if not math.isfinite(number):
            raise ItemSelectorDecisionError(f"{label} must contain finite values")
        return [number]
    if isinstance(value, str):
        if not value:
            raise ItemSelectorDecisionError(f"{label} strings must be non-empty")
        return [_stable_string_feature(value)]
    if isinstance(value, Mapping):
        result: list[float] = []
        for key in value:
            if not isinstance(key, str):
                raise ItemSelectorDecisionError(f"{label} mapping keys must be strings")
            result.extend(flatten_feature(value[key], label=f"{label}.{key}"))
        return result
    if isinstance(value, (list, tuple)):
        result = []
        for index, item in enumerate(value):
            result.extend(flatten_feature(item, label=f"{label}[{index}]"))
        return result
    raise ItemSelectorDecisionError(f"{label} contains unsupported feature type")


def encode_item_decision(
    item_context: ItemDecisionFeatures, nmax: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``ItemDecisionFeatures`` を artifact 入力の ``[context, candidates, mask]`` へ変換する。"""
    wire = item_context.to_wire()
    raw_context = wire["context_features"]
    raw_candidates = wire["candidates"]
    card_mask = raw_context["card_mask"]
    context_vector = flatten_feature(raw_context, label="context_features")
    candidate_vectors: list[list[float]] = []
    for idx, cand in enumerate(raw_candidates):
        public = {k: v for k, v in cand.items() if k != "schema_version"}
        candidate_vectors.append(flatten_feature(public, label=f"candidate[{idx}]"))
    candidate_dim = len(candidate_vectors[0]) if candidate_vectors else 0
    padding_count = nmax - len(raw_candidates)
    if padding_count < 0:
        raise ItemSelectorDecisionError("candidate count exceeds ItemSelector nmax")
    candidate_vectors.extend([[0.0] * candidate_dim for _ in range(padding_count)])
    mask = list(card_mask) + [False] * (nmax - len(card_mask))
    if len(mask) != nmax:
        raise ItemSelectorDecisionError("card_mask width does not match ItemSelector nmax")
    context_array = np.asarray([context_vector], dtype=np.float32)
    candidate_array = np.asarray([candidate_vectors], dtype=np.float32)
    mask_array = np.asarray([mask], dtype=bool)
    return context_array, candidate_array, mask_array


def calibrated_probabilities(
    scaled_logits: np.ndarray, mask: np.ndarray, temperature: float
) -> np.ndarray:
    """student temperature-scaled logits を artifact temperature で較正した確率にする。"""
    logits = np.asarray(scaled_logits, dtype=np.float64).reshape(-1)
    valid = np.asarray(mask, dtype=bool).reshape(-1)
    calibrated = logits / float(temperature)
    calibrated = np.where(valid, calibrated, -np.inf)
    shifted = calibrated - np.max(calibrated[valid])
    exponentials = np.where(valid, np.exp(shifted), 0.0)
    return exponentials / float(exponentials.sum())


def resolve_calibrated_winner(
    item_context: ItemDecisionFeatures,
    artifact: CalibratedItemSelectorArtifact,
) -> CalibratedWinner:
    """encode → predict → calibrate → argmax までを一気通貫で行う。

    target を実際の UI/候補へ bind するのは呼び出し側の責務。
    """
    if item_context.feature_schema != artifact.feature_schema:
        raise ItemSelectorDecisionError(
            f"feature_schema mismatch: {item_context.feature_schema!r} != {artifact.feature_schema!r}"
        )
    context_a, cand_a, mask_a = encode_item_decision(item_context, artifact.nmax)
    try:
        scaled_logits = np.asarray(artifact.predict(context_a, cand_a, mask_a), dtype=np.float64)
    except ItemSelectorDecisionError:
        raise
    except Exception as exc:  # noqa: BLE001  # artifact ごとに例外型が異なる
        raise ItemSelectorDecisionError(f"ItemSelector inference failed: {exc}") from exc
    probabilities = calibrated_probabilities(scaled_logits[0], mask_a[0], artifact.temperature)
    winner_index = int(np.argmax(probabilities))
    confidence = float(probabilities[winner_index])
    return CalibratedWinner(winner_index=winner_index, confidence=confidence, probabilities=probabilities)
