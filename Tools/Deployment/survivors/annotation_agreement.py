"""annotation agreement の schema と独立性条件を共有検証する。

feasibility、formal preflight、publish が同じ fail-closed 規則を使う。
"""

import math
from typing import Any, Mapping


DEFAULT_SELF_REPEAT_GAP_HOURS = 24.0
_V1_KEYS = {
    "schema_version", "pair_count", "iou_mean", "class_agreement",
    "unmatched_first", "unmatched_second", "sessions", "per_class",
}
_V2_KEYS = _V1_KEYS | {
    "independence_mode", "annotator_a", "annotator_b", "min_gap_hours",
    "required_min_gap_hours", "blind",
}


def validate_annotation_agreement(data: Any) -> str:
    """agreement JSON を検証し、報告された独立性 mode を返す。

    v1 は既存形式の independent_pair として扱い、v2 self_repeat は条件を全て確認する。
    """
    if not isinstance(data, Mapping):
        raise ValueError("annotation agreement must be an object")
    version = data.get("schema_version")
    if version == "annotation_agreement.v1":
        return "independent_pair"
    if version != "annotation_agreement.v2":
        raise ValueError("unknown annotation agreement schema")
    if set(data) != _V2_KEYS:
        raise ValueError("annotation agreement v2 fields mismatch")
    mode = data["independence_mode"]
    annotator_a, annotator_b = data["annotator_a"], data["annotator_b"]
    if not isinstance(mode, str) or mode not in {"independent_pair", "self_repeat"}:
        raise ValueError("annotation agreement independence_mode is invalid")
    if any(not isinstance(name, str) or not name.strip() for name in (annotator_a, annotator_b)):
        raise ValueError("annotation agreement annotator IDs are required")
    gap, required_gap = data["min_gap_hours"], data["required_min_gap_hours"]
    if isinstance(required_gap, bool) or not isinstance(required_gap, (int, float)) or not math.isfinite(required_gap):
        raise ValueError("annotation agreement required gap hours must be finite")
    if required_gap < DEFAULT_SELF_REPEAT_GAP_HOURS:
        raise ValueError("self_repeat required_min_gap_hours cannot be below 24 hours")
    if not isinstance(data["blind"], bool):
        raise ValueError("annotation agreement blind must be a boolean")
    if mode == "independent_pair":
        if annotator_a == annotator_b:
            raise ValueError("independent_pair requires two distinct annotators")
        if gap is not None and (
            isinstance(gap, bool) or not isinstance(gap, (int, float)) or not math.isfinite(gap)
        ):
            raise ValueError("annotation agreement gap hours must be finite or null")
        return mode
    if annotator_a != annotator_b:
        raise ValueError("self_repeat requires the same annotator")
    if (isinstance(gap, bool) or not isinstance(gap, (int, float)) or not math.isfinite(gap)
            or data["blind"] is not True or gap < required_gap):
        raise ValueError("self_repeat requires blind work-root and sufficient label update gap")
    return mode
