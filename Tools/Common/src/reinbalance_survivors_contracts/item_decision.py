"""ItemSelector の決定／候補特徴の契約（scaffold, v1）。

``ItemDecisionFeatures`` は、1回のレベルアップ決定コンテキストと、ItemSelector
（ステップ2、プラン 02-*）が消費する候補ごとの特徴ベクトルを表す。正式な特徴集合は
02-01 で確定する。本モジュールは versioned な wire 型と canonical hashing を提供し、
producer/consumer が identity について合意できるようにする。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from .canonical_json import canonical_hash

__all__ = [
    "ITEM_DECISION_SCHEMA_VERSION",
    "CANDIDATE_FEATURES_SCHEMA_VERSION",
    "CandidateFeatures",
    "ItemDecisionFeatures",
]

ITEM_DECISION_SCHEMA_VERSION = "item_decision.v1"
CANDIDATE_FEATURES_SCHEMA_VERSION = "candidate_features.v1"


def _check_finite(values: tuple[float, ...], label: str) -> None:
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label} must be real numbers")
        if not math.isfinite(float(value)):
            raise ValueError(f"{label} must be finite")


@dataclass(frozen=True)
class CandidateFeatures:
    """ItemSelector の候補1件あたりの特徴ベクトル（versioned）。"""

    candidate_id: str
    features: tuple[float, ...]
    schema_version: str = CANDIDATE_FEATURES_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id:
            raise ValueError("candidate_id must be a non-empty string")
        _check_finite(self.features, "candidate features")

    def to_wire(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "features": [float(v) for v in self.features],
        }

    @classmethod
    def from_wire(cls, data: Mapping[str, Any]) -> "CandidateFeatures":
        if data.get("schema_version") != CANDIDATE_FEATURES_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported CandidateFeatures schema_version "
                f"{data.get('schema_version')!r}"
            )
        return cls(
            candidate_id=data["candidate_id"],
            features=tuple(data["features"]),
            schema_version=data["schema_version"],
        )


@dataclass(frozen=True)
class ItemDecisionFeatures:
    """1回のレベルアップ決定コンテキストと候補群の特徴（versioned）。"""

    decision_id: str
    context_features: tuple[float, ...]
    candidates: tuple[CandidateFeatures, ...]
    schema_version: str = ITEM_DECISION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.decision_id, str) or not self.decision_id:
            raise ValueError("decision_id must be a non-empty string")
        _check_finite(self.context_features, "context features")
        if not self.candidates:
            raise ValueError("decision requires at least one candidate")
        seen: set[str] = set()
        for candidate in self.candidates:
            if candidate.candidate_id in seen:
                raise ValueError(
                    f"duplicate candidate_id {candidate.candidate_id!r}"
                )
            seen.add(candidate.candidate_id)

    def to_wire(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "decision_id": self.decision_id,
            "context_features": [float(v) for v in self.context_features],
            "candidates": [c.to_wire() for c in self.candidates],
        }

    @property
    def decision_hash(self) -> str:
        return canonical_hash(self.to_wire())

    @classmethod
    def from_wire(cls, data: Mapping[str, Any]) -> "ItemDecisionFeatures":
        if data.get("schema_version") != ITEM_DECISION_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported ItemDecisionFeatures schema_version "
                f"{data.get('schema_version')!r}"
            )
        return cls(
            decision_id=data["decision_id"],
            context_features=tuple(data["context_features"]),
            candidates=tuple(
                CandidateFeatures.from_wire(c) for c in data["candidates"]
            ),
            schema_version=data["schema_version"],
        )
