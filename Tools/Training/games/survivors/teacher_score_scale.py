"""development score-difference refs から teacher score scale を固定する。

raw critic/overlay ごとに identity を分離し、fit 後の改変や final-test 混入、
別 scale identity の calibration reuse を fail-closed で拒否する。
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from reinbalance_survivors_contracts.canonical_json import (
    canonical_hash,
    canonical_json_bytes,
)

SCHEMA_VERSION = "survivors.teacher_score_scale.v1"
TIE_EPSILON_Z = 0.02
_FIELDS = frozenset(
    {
        "schema_version",
        "scale_identity",
        "teacher_type",
        "teacher_identity",
        "development_fit_partition_id",
        "score_difference_refs",
        "q05",
        "q95",
        "sigma",
        "transform",
        "tie_epsilon_z",
    }
)


class ScoreScaleContractError(ValueError):
    """teacher score scale の fit・validation・binding 違反を表す。

    final ref、非有限差分、identity 改変を calibration 計算へ到達させない。
    """


def _require_sha256(value: Any, label: str) -> str:
    """小文字 64 桁 SHA-256 を検証して返す。

    teacher・split・scale の全 identity sibling に同じ形式制約を適用する。
    """
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ScoreScaleContractError(f"{label} must be lowercase sha256")
    return value


def _identity_payload(scale: Mapping[str, Any]) -> dict[str, Any]:
    """scale identity 自身を除く immutable payload を返す。

    q05/q95/sigma と全 sample refs を同じ commit identity へ束縛する。
    """
    return {key: scale[key] for key in sorted(_FIELDS - {"scale_identity"})}


def fit_teacher_score_scale(
    *,
    teacher_type: str,
    teacher_identity: str,
    development_fit_partition_id: str,
    score_difference_refs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """development score differences の q05/q95 から scale を fit する。

    final_test ref を一件でも含む場合は artifact を生成せず即時に失敗する。
    """
    if teacher_type not in {"raw_critic", "overlay", "source_critic"}:
        raise ScoreScaleContractError("unsupported teacher_type")
    _require_sha256(teacher_identity, "teacher_identity")
    _require_sha256(
        development_fit_partition_id, "development_fit_partition_id"
    )
    if (
        isinstance(score_difference_refs, (str, bytes))
        or len(score_difference_refs) < 2
    ):
        raise ScoreScaleContractError("at least two score difference refs are required")

    normalized_refs: list[dict[str, Any]] = []
    differences: list[float] = []
    seen: set[str] = set()
    for index, ref in enumerate(score_difference_refs):
        if not isinstance(ref, Mapping):
            raise ScoreScaleContractError(
                f"score_difference_refs[{index}] must be an object"
            )
        if set(ref) != {"ref_id", "partition", "difference"}:
            raise ScoreScaleContractError(
                f"score_difference_refs[{index}] fields mismatch"
            )
        ref_id = ref["ref_id"]
        partition = ref["partition"]
        difference = ref["difference"]
        if not isinstance(ref_id, str) or not ref_id or ref_id in seen:
            raise ScoreScaleContractError("score difference ref IDs must be unique")
        seen.add(ref_id)
        if partition == "final_test":
            raise ScoreScaleContractError(
                "final_test score difference cannot be used to fit scale"
            )
        if partition not in {
            "development_train",
            "development_validation",
        }:
            raise ScoreScaleContractError("score difference must be development data")
        if (
            isinstance(difference, bool)
            or not isinstance(difference, (int, float))
            or not math.isfinite(float(difference))
        ):
            raise ScoreScaleContractError("score difference must be finite")
        value = float(difference)
        normalized_refs.append(
            {
                "ref_id": ref_id,
                "partition": partition,
                "difference": value,
            }
        )
        differences.append(value)

    q05, q95 = (
        float(value)
        for value in np.quantile(
            np.asarray(differences, dtype=np.float64),
            [0.05, 0.95],
            method="linear",
        )
    )
    sigma = max((q95 - q05) / 3.29, 1e-6)
    artifact: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "scale_identity": "",
        "teacher_type": teacher_type,
        "teacher_identity": teacher_identity,
        "development_fit_partition_id": development_fit_partition_id,
        "score_difference_refs": sorted(
            normalized_refs, key=lambda item: item["ref_id"]
        ),
        "q05": q05,
        "q95": q95,
        "sigma": sigma,
        "transform": "z_i=(score_i-max_valid_score)/sigma",
        "tie_epsilon_z": TIE_EPSILON_Z,
    }
    artifact["scale_identity"] = canonical_hash(_identity_payload(artifact))
    validate_teacher_score_scale(artifact)
    return artifact


def validate_teacher_score_scale(scale: Any) -> None:
    """score scale の exact schema・derived values・identity を再検証する。

    persisted artifact の sigma や lineage ref が改変された後の再利用を拒否する。
    """
    if not isinstance(scale, Mapping):
        raise ScoreScaleContractError("score scale must be an object")
    actual = frozenset(scale)
    if actual != _FIELDS:
        raise ScoreScaleContractError(
            "score scale fields mismatch: "
            f"missing={sorted(_FIELDS - actual)}, unknown={sorted(actual - _FIELDS)}"
        )
    if scale["schema_version"] != SCHEMA_VERSION:
        raise ScoreScaleContractError("unsupported score scale schema_version")
    expected_identity = canonical_hash(_identity_payload(scale))
    if scale["scale_identity"] != expected_identity:
        raise ScoreScaleContractError("score scale identity mismatch")
    if scale["teacher_type"] not in {"raw_critic", "overlay", "source_critic"}:
        raise ScoreScaleContractError("unsupported teacher_type")
    _require_sha256(scale["teacher_identity"], "teacher_identity")
    _require_sha256(
        scale["development_fit_partition_id"],
        "development_fit_partition_id",
    )
    refs = scale["score_difference_refs"]
    if not isinstance(refs, list) or len(refs) < 2:
        raise ScoreScaleContractError("score_difference_refs are incomplete")
    differences: list[float] = []
    seen: set[str] = set()
    for ref in refs:
        if not isinstance(ref, Mapping) or set(ref) != {
            "ref_id",
            "partition",
            "difference",
        }:
            raise ScoreScaleContractError("score difference ref schema mismatch")
        if (
            not isinstance(ref["ref_id"], str)
            or not ref["ref_id"]
            or ref["ref_id"] in seen
        ):
            raise ScoreScaleContractError("score difference ref IDs must be unique")
        seen.add(ref["ref_id"])
        if ref["partition"] not in {
            "development_train",
            "development_validation",
        }:
            raise ScoreScaleContractError("final_test/non-development ref is forbidden")
        value = ref["difference"]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ScoreScaleContractError("score difference must be finite")
        differences.append(float(value))
    expected_q05, expected_q95 = (
        float(value)
        for value in np.quantile(
            np.asarray(differences, dtype=np.float64),
            [0.05, 0.95],
            method="linear",
        )
    )
    expected_sigma = max((expected_q95 - expected_q05) / 3.29, 1e-6)
    for field, expected in (
        ("q05", expected_q05),
        ("q95", expected_q95),
        ("sigma", expected_sigma),
    ):
        actual_value = scale[field]
        if (
            isinstance(actual_value, bool)
            or not isinstance(actual_value, (int, float))
            or not math.isfinite(float(actual_value))
            or abs(float(actual_value) - expected) > 1e-12
        ):
            raise ScoreScaleContractError(f"{field} does not match fit refs")
    if scale["transform"] != "z_i=(score_i-max_valid_score)/sigma":
        raise ScoreScaleContractError("score transform changed")
    if scale["tie_epsilon_z"] != TIE_EPSILON_Z:
        raise ScoreScaleContractError("tie_epsilon_z changed")


def transform_teacher_scores(
    scores: Sequence[float | None],
    scale: Mapping[str, Any],
) -> list[float | None]:
    """valid score の最大値を基準に z-score differences へ変換する。

    None は無効候補として保持し、非有限 score は暗黙欠損にせず拒否する。
    """
    validate_teacher_score_scale(scale)
    normalized: list[float | None] = []
    valid: list[float] = []
    for score in scores:
        if score is None:
            normalized.append(None)
            continue
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
        ):
            raise ScoreScaleContractError("teacher scores must be finite or None")
        value = float(score)
        normalized.append(value)
        valid.append(value)
    if not valid:
        raise ScoreScaleContractError("at least one valid teacher score is required")
    maximum = max(valid)
    sigma = float(scale["sigma"])
    return [
        None if value is None else (value - maximum) / sigma
        for value in normalized
    ]


def assert_calibration_scale_binding(
    calibration: Mapping[str, Any],
    scale: Mapping[str, Any],
) -> None:
    """calibration parent scale identity が現在の committed scale と一致するか検証する。

    teacher identity 変更や overlay/raw 間の calibration reuse を許可しない。
    """
    validate_teacher_score_scale(scale)
    if not isinstance(calibration, Mapping):
        raise ScoreScaleContractError("calibration must be an object")
    if calibration.get("teacher_score_scale_id") != scale["scale_identity"]:
        raise ScoreScaleContractError("calibration score scale binding mismatch")


def commit_teacher_score_scale(path: Path, scale: Mapping[str, Any]) -> None:
    """validated scale を create-once の canonical JSON artifact として保存する。

    既存 path が同一内容なら idempotent とし、異なる内容なら上書きせず拒否する。
    """
    validate_teacher_score_scale(scale)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json_bytes(dict(scale)) + b"\n"
    try:
        descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
    except FileExistsError:
        if destination.read_bytes() != encoded:
            raise ScoreScaleContractError(
                "committed score scale cannot be overwritten"
            )
        return
    try:
        written = os.write(descriptor, encoded)
        if written != len(encoded):
            raise OSError("short score scale write")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def load_teacher_score_scale(path: Path) -> dict[str, Any]:
    """JSON artifact を読み、全契約を再検証して返す。

    parse 成功だけで信用せず content identity と derived sigma まで照合する。
    """
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScoreScaleContractError(f"cannot load score scale: {exc}") from exc
    validate_teacher_score_scale(value)
    return dict(value)
