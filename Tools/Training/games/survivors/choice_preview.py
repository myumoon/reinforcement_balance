"""Survivors level-up preview のimmutable wire契約を定義する。

初心者向け: UE5が計算したraw observationを、shape・schema・choice IDへ厳密に束縛して保持する。
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


_RESPONSE_KEYS = frozenset(
    {"decision_id", "obs_schema_hash", "base_obs", "previews"}
)
_PREVIEW_KEYS = frozenset(
    {"choice_id", "projected_obs", "changed_segments"}
)


@dataclass(frozen=True, slots=True)
class SurvivorsChoicePreview:
    """一つのchoiceをproduction適用した直後のraw observation。

    初心者向け: changed_segmentsは値そのものではなく、schema名単位の差分索引として添える。
    """

    choice_id: str
    projected_obs: tuple[float, ...]
    changed_segments: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SurvivorsLevelUpPreview:
    """pending decision全候補へ束縛された反実仮想preview。

    初心者向け: mappingもread-only化し、検証後にchoiceと観測の対応を書き換えられないようにする。
    """

    decision_id: str
    obs_schema_hash: str
    base_obs: tuple[float, ...]
    by_choice_id: Mapping[str, SurvivorsChoicePreview]


def _require_exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], label: str
) -> None:
    """wire objectの未知・不足fieldをfail-closedで拒否する。

    初心者向け: typoを黙認すると別versionのresponseを正しいものとして使うため、名前を完全一致させる。
    """

    actual = frozenset(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(
            f"{label} fields mismatch: missing={missing}, unknown={unknown}"
        )


def _require_non_empty_string(value: Any, label: str) -> str:
    """空でない文字列だけをID/hash/nameとして受理する。

    初心者向け: 数値の暗黙変換や空IDを許さず、serverとのbindingを曖昧にしない。
    """

    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _parse_observation(value: Any, obs_dim: int, label: str) -> tuple[float, ...]:
    """固定dimのfiniteな数値配列をimmutable tupleへ変換する。

    初心者向け: bool・NaN・Infinity・shape違いをscorerへ渡す前に通信境界で止める。
    """

    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or len(value) != obs_dim
    ):
        raise ValueError(f"{label} must have shape ({obs_dim},)")
    parsed: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{label}[{index}] must be a number")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{label}[{index}] must be finite")
        parsed.append(number)
    return tuple(parsed)


def _normalize_choice_ids(expected_choice_ids: Collection[str]) -> frozenset[str]:
    """callerがpending infoから得たchoice ID集合を検証する。

    初心者向け: response候補を順序で結ばず、重複のないID集合としてproduction pendingへ束縛する。
    """

    if isinstance(expected_choice_ids, (str, bytes, bytearray)):
        raise ValueError("expected choice_id set must be a collection of strings")
    normalized = frozenset(
        _require_non_empty_string(value, "expected choice_id")
        for value in expected_choice_ids
    )
    if not normalized:
        raise ValueError("expected choice_id set must not be empty")
    if len(normalized) != len(expected_choice_ids):
        raise ValueError("expected choice_id set must not contain duplicates")
    return normalized


def parse_level_up_preview(
    payload: Any,
    *,
    expected_decision_id: str,
    expected_schema_hash: str,
    expected_choice_ids: Collection[str],
    obs_dim: int,
    schema_segment_names: Collection[str],
) -> SurvivorsLevelUpPreview:
    """HTTP responseをdecision/schema/choice集合へ束縛してtyped previewへ変換する。

    初心者向け: observationの値は加工せず、UE5の結果が現在のpending契約と一致するかだけを検査する。
    """

    expected_decision_id = _require_non_empty_string(
        expected_decision_id, "expected decision_id"
    )
    expected_schema_hash = _require_non_empty_string(
        expected_schema_hash, "expected obs_schema_hash"
    )
    if isinstance(obs_dim, bool) or not isinstance(obs_dim, int) or obs_dim <= 0:
        raise ValueError("obs_dim must be a positive integer")
    expected_ids = _normalize_choice_ids(expected_choice_ids)
    if isinstance(schema_segment_names, (str, bytes, bytearray)):
        raise ValueError("schema segment names must be a collection of strings")
    segment_names = frozenset(
        _require_non_empty_string(name, "schema segment name")
        for name in schema_segment_names
    )
    if not segment_names:
        raise ValueError("schema segment names must not be empty")
    if len(segment_names) != len(schema_segment_names):
        raise ValueError("schema segment names must not contain duplicates")

    if not isinstance(payload, Mapping):
        raise ValueError("preview response must be an object")
    _require_exact_keys(payload, _RESPONSE_KEYS, "preview response")
    decision_id = _require_non_empty_string(payload["decision_id"], "decision_id")
    if decision_id != expected_decision_id:
        raise ValueError(
            "decision_id mismatch: "
            f"expected={expected_decision_id}, received={decision_id}"
        )
    schema_hash = _require_non_empty_string(
        payload["obs_schema_hash"], "obs_schema_hash"
    )
    if schema_hash != expected_schema_hash:
        raise ValueError(
            "obs_schema_hash mismatch: "
            f"expected={expected_schema_hash}, received={schema_hash}"
        )
    base_obs = _parse_observation(payload["base_obs"], obs_dim, "base_obs")

    raw_previews = payload["previews"]
    if not isinstance(raw_previews, list):
        raise ValueError("previews must be an array")
    previews: dict[str, SurvivorsChoicePreview] = {}
    for index, raw_preview in enumerate(raw_previews):
        if not isinstance(raw_preview, Mapping):
            raise ValueError(f"previews[{index}] must be an object")
        _require_exact_keys(raw_preview, _PREVIEW_KEYS, f"previews[{index}]")
        choice_id = _require_non_empty_string(
            raw_preview["choice_id"], f"previews[{index}].choice_id"
        )
        if choice_id in previews:
            raise ValueError(f"duplicate choice_id: {choice_id}")
        projected_obs = _parse_observation(
            raw_preview["projected_obs"],
            obs_dim,
            f"previews[{index}].projected_obs",
        )
        raw_segments = raw_preview["changed_segments"]
        if not isinstance(raw_segments, list):
            raise ValueError(
                f"previews[{index}].changed_segments must be an array"
            )
        changed_segments: list[str] = []
        for segment_index, raw_name in enumerate(raw_segments):
            name = _require_non_empty_string(
                raw_name,
                f"previews[{index}].changed_segments[{segment_index}]",
            )
            if name not in segment_names:
                raise ValueError(f"unknown changed segment: {name}")
            if name in changed_segments:
                raise ValueError(f"duplicate changed segment: {name}")
            changed_segments.append(name)
        previews[choice_id] = SurvivorsChoicePreview(
            choice_id=choice_id,
            projected_obs=projected_obs,
            changed_segments=tuple(changed_segments),
        )

    actual_ids = frozenset(previews)
    if actual_ids != expected_ids:
        raise ValueError(
            "choice_id set mismatch: "
            f"expected={sorted(expected_ids)}, received={sorted(actual_ids)}"
        )
    return SurvivorsLevelUpPreview(
        decision_id=decision_id,
        obs_schema_hash=schema_hash,
        base_obs=base_obs,
        by_choice_id=MappingProxyType(previews),
    )
