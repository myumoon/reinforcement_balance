"""Survivors value choice ranking v3 の構築・検証・JSONL append を定義する。

source/context identity、ordered value、margin、tie を strict schema に固定し、非有限値や
未知 field を永続化前に拒否する。
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from reinbalance_survivors_contracts.canonical_json import canonical_json_bytes

from games.survivors.value_scorer import CandidateValue, TIE_EPSILON

SCHEMA_VERSION = "survivors.value_choice_ranking.v3"
_TOP_FIELDS = frozenset(
    {
        "schema_version",
        "source",
        "context",
        "ordered_candidates",
        "margin_normalized_return",
        "tie",
        "tie_epsilon",
        "ready_for_training_label",
    }
)
_SOURCE_FIELDS = frozenset(
    {
        "identity_sha256",
        "manifest_sha256",
        "model_sha256",
        "vecnormalize_sha256",
        "observation_schema_sha256",
        "policy_state_schema_sha256",
    }
)
_CONTEXT_FIELDS = frozenset(
    {
        "sha256",
        "mode",
        "environment_step",
        "decision_id",
        "pending_obs_sha256",
    }
)
_CANDIDATE_FIELDS = frozenset(
    {
        "rank",
        "choice_id",
        "value_normalized_return",
        "value_unscaled_return",
    }
)


class ValueChoiceSchemaError(ValueError):
    """ranking row を strict schema として受理できない場合の例外。

    scorer 計算エラーと永続化エラーの間に validation boundary を置き、不正 JSONL を残さない。
    """


def _require_exact_fields(
    value: Any,
    expected: frozenset[str],
    label: str,
) -> Mapping[str, Any]:
    """mapping の未知・不足 field を完全一致で検査する。

    将来 schema や typo を黙って読み捨てず、versioned row の意味を固定する。
    """
    if not isinstance(value, Mapping):
        raise ValueChoiceSchemaError(f"{label} must be an object")
    actual = frozenset(value)
    if actual != expected:
        raise ValueChoiceSchemaError(
            f"{label} fields mismatch: "
            f"missing={sorted(expected - actual)}, unknown={sorted(actual - expected)}"
        )
    return value


def _require_sha256(value: Any, label: str) -> None:
    """小文字 64 桁の SHA-256 だけを受理する。

    source・context の全 hash sibling へ同じ形式検証を適用する。
    """
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueChoiceSchemaError(f"{label} must be lowercase sha256")


def _require_finite_number(value: Any, label: str) -> float:
    """bool を除く finite number を float として返す。

    NaN/Infinity を canonical JSON へ到達させず、append 前 gate で止める。
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueChoiceSchemaError(f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueChoiceSchemaError(f"{label} must be finite")
    return number


def build_value_choice_ranking(
    *,
    source_identity_sha256: str,
    manifest_sha256: str,
    model_sha256: str,
    vecnormalize_sha256: str,
    observation_schema_sha256: str,
    policy_state_schema_sha256: str,
    context_sha256: str,
    context_mode: str,
    environment_step: int,
    decision_id: str,
    pending_obs_sha256: str,
    values: Sequence[CandidateValue],
    zero_state_smoke: bool,
) -> dict[str, Any]:
    """candidate values から stable descending ranking row を構築する。

    value tie は 1e-5 で明示し、choice ID を二次 sort key に使わず入力順を保持する。
    """
    if len(values) < 2:
        raise ValueChoiceSchemaError("ranking requires at least two candidates")
    ordered = sorted(
        enumerate(values),
        key=lambda item: item[1].value_normalized_return,
        reverse=True,
    )
    candidates = [
        {
            "rank": rank,
            "choice_id": value.choice_id,
            "value_normalized_return": float(value.value_normalized_return),
            "value_unscaled_return": float(value.value_unscaled_return),
        }
        for rank, (_, value) in enumerate(ordered, start=1)
    ]
    margin = (
        candidates[0]["value_normalized_return"]
        - candidates[1]["value_normalized_return"]
    )
    row = {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "identity_sha256": source_identity_sha256,
            "manifest_sha256": manifest_sha256,
            "model_sha256": model_sha256,
            "vecnormalize_sha256": vecnormalize_sha256,
            "observation_schema_sha256": observation_schema_sha256,
            "policy_state_schema_sha256": policy_state_schema_sha256,
        },
        "context": {
            "sha256": context_sha256,
            "mode": context_mode,
            "environment_step": environment_step,
            "decision_id": decision_id,
            "pending_obs_sha256": pending_obs_sha256,
        },
        "ordered_candidates": candidates,
        "margin_normalized_return": margin,
        "tie": margin <= TIE_EPSILON,
        "tie_epsilon": TIE_EPSILON,
        # Ranking は release verdict ではないため、formal/zero のどちらでも label ready を主張しない。
        # zero_state_smoke は明示入力に残し、診断出力が将来 true へ変わらない境界にする。
        "ready_for_training_label": False if zero_state_smoke else False,
    }
    validate_value_choice_ranking(row)
    return row


def validate_value_choice_ranking(row: Any) -> None:
    """ranking v3 の exact schema・finite・ordering・derived field を検証する。

    JSONL writer と reader が同じ validator を共有できる純粋な Model 層として実装する。
    """
    data = _require_exact_fields(row, _TOP_FIELDS, "ranking")
    if data["schema_version"] != SCHEMA_VERSION:
        raise ValueChoiceSchemaError("unsupported ranking schema_version")
    source = _require_exact_fields(data["source"], _SOURCE_FIELDS, "source")
    for name, value in source.items():
        _require_sha256(value, f"source.{name}")
    context = _require_exact_fields(
        data["context"],
        _CONTEXT_FIELDS,
        "context",
    )
    _require_sha256(context["sha256"], "context.sha256")
    _require_sha256(
        context["pending_obs_sha256"],
        "context.pending_obs_sha256",
    )
    if context["mode"] not in {"captured", "burn_in", "zero_state_smoke"}:
        raise ValueChoiceSchemaError("context.mode is unsupported")
    if (
        type(context["environment_step"]) is not int
        or context["environment_step"] < 0
    ):
        raise ValueChoiceSchemaError(
            "context.environment_step must be a non-negative integer"
        )
    if not isinstance(context["decision_id"], str) or not context["decision_id"]:
        raise ValueChoiceSchemaError("context.decision_id must be non-empty")

    candidates = data["ordered_candidates"]
    if not isinstance(candidates, list) or len(candidates) < 2:
        raise ValueChoiceSchemaError(
            "ordered_candidates must contain at least two items"
        )
    choice_ids: set[str] = set()
    normalized_values: list[float] = []
    for index, candidate_value in enumerate(candidates, start=1):
        candidate = _require_exact_fields(
            candidate_value,
            _CANDIDATE_FIELDS,
            f"ordered_candidates[{index - 1}]",
        )
        if candidate["rank"] != index:
            raise ValueChoiceSchemaError("candidate rank must be contiguous")
        choice_id = candidate["choice_id"]
        if (
            not isinstance(choice_id, str)
            or not choice_id
            or choice_id in choice_ids
        ):
            raise ValueChoiceSchemaError(
                "candidate choice_id must be unique and non-empty"
            )
        choice_ids.add(choice_id)
        normalized_values.append(
            _require_finite_number(
                candidate["value_normalized_return"],
                f"ordered_candidates[{index - 1}].value_normalized_return",
            )
        )
        _require_finite_number(
            candidate["value_unscaled_return"],
            f"ordered_candidates[{index - 1}].value_unscaled_return",
        )
    if any(
        left < right
        for left, right in zip(normalized_values, normalized_values[1:])
    ):
        raise ValueChoiceSchemaError(
            "ordered_candidates must be sorted by normalized value descending"
        )
    margin = _require_finite_number(
        data["margin_normalized_return"],
        "margin_normalized_return",
    )
    expected_margin = normalized_values[0] - normalized_values[1]
    if abs(margin - expected_margin) > 1e-12:
        raise ValueChoiceSchemaError(
            "margin_normalized_return does not match top candidates"
        )
    if data["tie_epsilon"] != TIE_EPSILON:
        raise ValueChoiceSchemaError("tie_epsilon must be 1e-5")
    if type(data["tie"]) is not bool or data["tie"] != (
        expected_margin <= TIE_EPSILON
    ):
        raise ValueChoiceSchemaError("tie does not match margin and epsilon")
    if data["ready_for_training_label"] is not False:
        raise ValueChoiceSchemaError(
            "ranking rows cannot claim ready_for_training_label"
        )


def append_value_choice_ranking(path: Path, row: Mapping[str, Any]) -> None:
    """schema validation と finite check 後に canonical JSONL を一回 append する。

    validation 失敗時は file を開かず、成功時は O_APPEND の単一 write と fsync で永続化する。
    """
    validate_value_choice_ranking(row)
    encoded = canonical_json_bytes(dict(row)) + b"\n"
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_APPEND,
        0o644,
    )
    try:
        written = os.write(descriptor, encoded)
        if written != len(encoded):
            raise OSError("short JSONL append")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

