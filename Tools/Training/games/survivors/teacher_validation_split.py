"""teacher validation corpus の episode-group split を freeze する。

development train/validation と untouched final test を identity 付き artifact に封印し、
calibration・method selection へ final lineage が入った時点で fail-closed にする。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from reinbalance_survivors_contracts.canonical_json import canonical_hash

SCHEMA_VERSION = "survivors.teacher_validation_split.v1"
DEVELOPMENT_PARTITIONS = frozenset(
    {"development_train", "development_validation"}
)
ALL_PARTITIONS = DEVELOPMENT_PARTITIONS | {"final_test"}
_FIELDS = frozenset(
    {
        "schema_version",
        "split_identity",
        "seed",
        "ratios",
        "episode_partitions",
        "frozen",
        "final_test_open_policy",
    }
)


class SplitContractError(ValueError):
    """split freeze または final-test sealing を破る入力の例外。

    episode overlap・artifact 改変・final lineage 混入を同じ fail-closed 境界で扱う。
    """


def _identity_payload(split: Mapping[str, Any]) -> dict[str, Any]:
    """split identity を除いた immutable payload を返す。

    identity 自己参照を避け、freeze 後の全 semantic field を hash 対象にする。
    """
    return {key: split[key] for key in sorted(_FIELDS - {"split_identity"})}


def freeze_episode_split(
    episode_ids: Sequence[str],
    *,
    seed: str,
    development_train_ratio: float = 0.6,
    development_validation_ratio: float = 0.2,
) -> dict[str, Any]:
    """episode ID 集合を stable hash 順で三 partition へ割り当てる。

    decision 行ではなく episode を一単位にし、入力順が変わっても同じ artifact を返す。
    """
    if (
        isinstance(episode_ids, (str, bytes))
        or not episode_ids
        or any(not isinstance(item, str) or not item for item in episode_ids)
        or len(set(episode_ids)) != len(episode_ids)
    ):
        raise SplitContractError("episode_ids must be unique non-empty strings")
    if not isinstance(seed, str) or not seed:
        raise SplitContractError("seed must be a non-empty string")
    if (
        isinstance(development_train_ratio, bool)
        or isinstance(development_validation_ratio, bool)
        or not 0.0 < float(development_train_ratio) < 1.0
        or not 0.0 < float(development_validation_ratio) < 1.0
        or float(development_train_ratio) + float(development_validation_ratio)
        >= 1.0
    ):
        raise SplitContractError("split ratios must leave an untouched final test")

    ordered = sorted(
        set(episode_ids),
        key=lambda episode_id: (
            canonical_hash({"seed": seed, "episode_id": episode_id}),
            episode_id,
        ),
    )
    count = len(ordered)
    train_count = max(1, int(count * float(development_train_ratio)))
    validation_count = max(
        1, int(count * float(development_validation_ratio))
    )
    if train_count + validation_count >= count:
        if count < 3:
            raise SplitContractError("at least three episodes are required")
        validation_count = 1
        train_count = count - 2

    assignments: dict[str, str] = {}
    for index, episode_id in enumerate(ordered):
        if index < train_count:
            partition = "development_train"
        elif index < train_count + validation_count:
            partition = "development_validation"
        else:
            partition = "final_test"
        assignments[episode_id] = partition
    artifact: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "split_identity": "",
        "seed": seed,
        "ratios": {
            "development_train": float(development_train_ratio),
            "development_validation": float(development_validation_ratio),
            "final_test": 1.0
            - float(development_train_ratio)
            - float(development_validation_ratio),
        },
        "episode_partitions": dict(sorted(assignments.items())),
        "frozen": True,
        "final_test_open_policy": "evaluate_once_after_method_lock",
    }
    artifact["split_identity"] = canonical_hash(_identity_payload(artifact))
    validate_frozen_split(artifact)
    return artifact


def validate_frozen_split(split: Any) -> None:
    """frozen split の exact schema・partition・content identity を検証する。

    assignment の移動や ratio/policy 改変を identity mismatch として拒否する。
    """
    if not isinstance(split, Mapping):
        raise SplitContractError("split must be an object")
    actual = frozenset(split)
    if actual != _FIELDS:
        raise SplitContractError(
            "split fields mismatch: "
            f"missing={sorted(_FIELDS - actual)}, unknown={sorted(actual - _FIELDS)}"
        )
    if split["schema_version"] != SCHEMA_VERSION:
        raise SplitContractError("unsupported split schema_version")
    if split["frozen"] is not True:
        raise SplitContractError("split must be frozen")
    if split["final_test_open_policy"] != "evaluate_once_after_method_lock":
        raise SplitContractError("final test open policy changed")
    assignments = split["episode_partitions"]
    if (
        not isinstance(assignments, Mapping)
        or len(assignments) < 3
        or any(not isinstance(key, str) or not key for key in assignments)
        or any(value not in ALL_PARTITIONS for value in assignments.values())
        or set(assignments.values()) != ALL_PARTITIONS
    ):
        raise SplitContractError("episode_partitions are incomplete or invalid")
    ratios = split["ratios"]
    if (
        not isinstance(ratios, Mapping)
        or set(ratios) != ALL_PARTITIONS
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0.0 < float(value) < 1.0
            for value in ratios.values()
        )
        or abs(sum(float(value) for value in ratios.values()) - 1.0) > 1e-12
    ):
        raise SplitContractError("ratios are invalid")
    expected_identity = canonical_hash(_identity_payload(split))
    if split["split_identity"] != expected_identity:
        raise SplitContractError("split identity mismatch")


def _validate_lineage_refs(
    split: Mapping[str, Any],
    refs: Sequence[Mapping[str, Any]],
    *,
    label: str,
) -> list[str]:
    """decision/outcome refs の episode partition binding を検証する。

    final_test と assignment 詐称を同じ処理で両 lineage sibling へ適用する。
    """
    assignments = split["episode_partitions"]
    used: list[str] = []
    seen_ref_ids: set[str] = set()
    id_field = "decision_id" if label == "decision" else "outcome_ref"
    for index, ref in enumerate(refs):
        if not isinstance(ref, Mapping):
            raise SplitContractError(f"{label}_refs[{index}] must be an object")
        ref_id = ref.get(id_field)
        episode_id = ref.get("episode_id")
        partition = ref.get("partition")
        if not isinstance(ref_id, str) or not ref_id or ref_id in seen_ref_ids:
            raise SplitContractError(f"{label} ref IDs must be unique")
        seen_ref_ids.add(ref_id)
        if episode_id not in assignments:
            raise SplitContractError(f"{label} ref has an unknown episode")
        expected = assignments[episode_id]
        if partition != expected:
            raise SplitContractError(f"{label} ref partition binding mismatch")
        if expected == "final_test":
            raise SplitContractError(
                f"final_test {label} lineage cannot be used for calibration"
            )
        if expected not in DEVELOPMENT_PARTITIONS:
            raise SplitContractError(f"{label} ref is not development data")
        used.append(ref_id)
    return used


def assert_calibration_lineage(
    split: Mapping[str, Any],
    *,
    decision_refs: Sequence[Mapping[str, Any]],
    outcome_refs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """calibration parent refs が development のみであることを証明する。

    使用 ref と全 final episode exclusion を lineage object として返す。
    """
    validate_frozen_split(split)
    used_decisions = _validate_lineage_refs(
        split, decision_refs, label="decision"
    )
    used_outcomes = _validate_lineage_refs(split, outcome_refs, label="outcome")
    excluded_final = sorted(
        episode_id
        for episode_id, partition in split["episode_partitions"].items()
        if partition == "final_test"
    )
    return {
        "split_identity": split["split_identity"],
        "used_decision_refs": sorted(used_decisions),
        "used_outcome_refs": sorted(used_outcomes),
        "excluded_final_episode_ids": excluded_final,
    }

