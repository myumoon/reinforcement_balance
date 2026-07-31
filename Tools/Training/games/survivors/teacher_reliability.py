"""teacher pairwise residual の slice reliability calibration を生成する。

episode/seed cluster bootstrap の 95% CI upper bound を誤差として使い、exact support が
不足するときだけ事前固定 teacher_type×choice_kind fallback を適用する。
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from reinbalance_survivors_contracts.canonical_json import canonical_hash

SCHEMA_VERSION = "survivors.teacher_reliability_calibration.v1"
ERROR_LIMIT_Z = 1.0
ELAPSED_BANDS = (
    ("0-5m", 0.0, 300.0),
    ("5-10m", 300.0, 600.0),
    ("10-20m", 600.0, 1200.0),
    ("20-30m", 1200.0, 1800.000001),
)
SUPPORTED_CHOICE_KINDS = frozenset(
    {
        "weapon",
        "weapon_new",
        "weapon_upgrade",
        "passive",
        "passive_new",
        "passive_upgrade",
        "evolution",
        "union",
        "no_upgrade",
    }
)
_FIELDS = frozenset(
    {
        "schema_version",
        "calibration_identity",
        "teacher_identity",
        "teacher_type",
        "teacher_score_scale_id",
        "outcome_scale_ids",
        "development_split_id",
        "integration_fidelity_identity",
        "branch_outcome_refs",
        "slices",
        "bootstrap_seed",
        "bootstrap_resamples",
        "error_limit_z",
    }
)


class ReliabilityContractError(ValueError):
    """reliability fit・lineage・artifact validation 違反の例外。

    final outcome、未知 slice、scale mismatch、fabricated support/weight を拒否する。
    """


def _require_sha256(value: Any, label: str) -> str:
    """小文字 64 桁 SHA-256 を検証して返す。

    teacher/score/outcome/split/calibration identity の兄弟 field 全てへ適用する。
    """
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ReliabilityContractError(f"{label} must be lowercase sha256")
    return value


def _elapsed_band(seconds: Any) -> str:
    """elapsed seconds を事前固定 4 band の一つへ割り当てる。

    負値・30分超・非有限値は unknown band に丸めず拒否する。
    """
    if (
        isinstance(seconds, bool)
        or not isinstance(seconds, (int, float))
        or not math.isfinite(float(seconds))
    ):
        raise ReliabilityContractError("elapsed_seconds must be finite")
    value = float(seconds)
    for label, lower, upper in ELAPSED_BANDS:
        if lower <= value < upper:
            return label
    raise ReliabilityContractError("elapsed_seconds is outside registered bands")


def _cluster_key(row: Mapping[str, Any]) -> str:
    """episode と seed identity を一つの effective cluster key にする。

    同一 episode 内の複数 decisions を独立 Bernoulli として数えない。
    """
    episode_id = row.get("episode_id")
    seed_id = row.get("seed_cluster_id")
    if not isinstance(episode_id, str) or not episode_id:
        raise ReliabilityContractError("episode_id must be non-empty")
    if not isinstance(seed_id, str) or not seed_id:
        raise ReliabilityContractError("seed_cluster_id must be non-empty")
    return f"{episode_id}\x1f{seed_id}"


def _cluster_bootstrap_median(
    observations: Sequence[tuple[str, float]],
    *,
    seed: int,
    resamples: int,
) -> tuple[float, float, float, int]:
    """cluster resampling で median absolute residual と 95% CI を返す。

    cluster 内 decisions/pairs はまとめて再標本化し、cluster 数を effective support とする。
    """
    grouped: dict[str, list[float]] = defaultdict(list)
    for cluster, value in observations:
        grouped[cluster].append(float(value))
    clusters = sorted(grouped)
    if not clusters:
        raise ReliabilityContractError("residual observations are empty")
    point = float(
        np.median(
            np.asarray(
                [value for cluster in clusters for value in grouped[cluster]],
                dtype=np.float64,
            )
        )
    )
    rng = np.random.default_rng(seed)
    bootstrap_values = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sampled_indices = rng.integers(0, len(clusters), size=len(clusters))
        sample = [
            value
            for sampled_index in sampled_indices
            for value in grouped[clusters[int(sampled_index)]]
        ]
        bootstrap_values[index] = float(np.median(np.asarray(sample)))
    lower, upper = (
        float(value)
        for value in np.quantile(
            bootstrap_values,
            [0.025, 0.975],
            method="linear",
        )
    )
    return point, lower, upper, len(clusters)


def _normalize_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    str,
    str,
    dict[str, str],
    str,
]:
    """raw decision rows を residual/lineage を含む内部形式へ検証・正規化する。

    short/full outcome と score/outcome scale binding を全 row/pair で対称に確認する。
    """
    if isinstance(rows, (str, bytes)) or not rows:
        raise ReliabilityContractError("reliability rows are required")
    normalized: list[dict[str, Any]] = []
    teacher_types: set[str] = set()
    score_scale_ids: set[str] = set()
    outcome_scale_values: set[tuple[str, str]] = set()
    teacher_identities: set[str] = set()
    seen_decisions: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ReliabilityContractError(f"rows[{index}] must be an object")
        if row.get("partition") == "final_test":
            raise ReliabilityContractError("final_test outcome cannot fit calibration")
        if row.get("partition") not in {
            "development_train",
            "development_validation",
        }:
            raise ReliabilityContractError("reliability row must be development data")
        decision_id = row.get("decision_id")
        if (
            not isinstance(decision_id, str)
            or not decision_id
            or decision_id in seen_decisions
        ):
            raise ReliabilityContractError("decision IDs must be unique")
        seen_decisions.add(decision_id)
        teacher_type = row.get("teacher_type")
        if teacher_type not in {"raw_critic", "overlay", "source_critic"}:
            raise ReliabilityContractError("unsupported teacher_type")
        teacher_types.add(teacher_type)
        teacher_identity = row.get("teacher_identity")
        if teacher_identity is not None:
            teacher_identities.add(
                _require_sha256(teacher_identity, "teacher_identity")
            )
        choice_kind = row.get("choice_kind")
        if choice_kind not in SUPPORTED_CHOICE_KINDS:
            raise ReliabilityContractError("unknown choice_kind")
        band = _elapsed_band(row.get("elapsed_seconds"))
        cluster = _cluster_key(row)
        score_scale_ids.add(
            _require_sha256(
                row.get("teacher_score_scale_id"),
                "teacher_score_scale_id",
            )
        )
        outcome_scale_ids = row.get("outcome_scale_ids")
        if (
            not isinstance(outcome_scale_ids, Mapping)
            or set(outcome_scale_ids) != {"short", "full"}
        ):
            raise ReliabilityContractError("short/full outcome scale IDs are required")
        short_scale = _require_sha256(
            outcome_scale_ids["short"], "outcome_scale_ids.short"
        )
        full_scale = _require_sha256(
            outcome_scale_ids["full"], "outcome_scale_ids.full"
        )
        outcome_scale_values.add((short_scale, full_scale))
        pairs = row.get("pairs")
        if not isinstance(pairs, list) or not pairs:
            raise ReliabilityContractError("each decision requires candidate pairs")
        normalized_pairs: list[dict[str, Any]] = []
        seen_pairs: set[tuple[str, str]] = set()
        for pair_index, pair in enumerate(pairs):
            if not isinstance(pair, Mapping):
                raise ReliabilityContractError(
                    f"rows[{index}].pairs[{pair_index}] must be an object"
                )
            candidate_i = pair.get("candidate_i")
            candidate_j = pair.get("candidate_j")
            if (
                not isinstance(candidate_i, str)
                or not candidate_i
                or not isinstance(candidate_j, str)
                or not candidate_j
                or candidate_i == candidate_j
                or (candidate_i, candidate_j) in seen_pairs
            ):
                raise ReliabilityContractError("candidate pair identity is invalid")
            seen_pairs.add((candidate_i, candidate_j))
            teacher_margin = pair.get("teacher_margin_z")
            if (
                isinstance(teacher_margin, bool)
                or not isinstance(teacher_margin, (int, float))
                or not math.isfinite(float(teacher_margin))
            ):
                raise ReliabilityContractError("teacher margin must be finite")
            outcome_margins = pair.get("outcome_margin_z")
            outcome_refs = pair.get("outcome_refs")
            if (
                not isinstance(outcome_margins, Mapping)
                or set(outcome_margins) != {"short", "full"}
                or not isinstance(outcome_refs, Mapping)
                or set(outcome_refs) != {"short", "full"}
            ):
                raise ReliabilityContractError(
                    "short/full outcome margins and refs are required"
                )
            residuals: dict[str, float] = {}
            refs: dict[str, str] = {}
            for horizon in ("short", "full"):
                margin = outcome_margins[horizon]
                ref = outcome_refs[horizon]
                if (
                    isinstance(margin, bool)
                    or not isinstance(margin, (int, float))
                    or not math.isfinite(float(margin))
                ):
                    raise ReliabilityContractError(
                        f"{horizon} outcome margin must be finite"
                    )
                if not isinstance(ref, str) or not ref:
                    raise ReliabilityContractError(
                        f"{horizon} outcome lineage is missing"
                    )
                residuals[horizon] = abs(
                    float(teacher_margin) - float(margin)
                )
                refs[horizon] = ref
            normalized_pairs.append(
                {
                    "residuals": residuals,
                    "outcome_refs": refs,
                }
            )
        normalized.append(
            {
                "decision_id": decision_id,
                "cluster": cluster,
                "teacher_type": teacher_type,
                "choice_kind": choice_kind,
                "elapsed_band": band,
                "pairs": normalized_pairs,
            }
        )
    if len(teacher_types) != 1:
        raise ReliabilityContractError("teacher_type mismatch across rows")
    if len(score_scale_ids) != 1:
        raise ReliabilityContractError("teacher score scale mismatch across rows")
    if len(outcome_scale_values) != 1:
        raise ReliabilityContractError("outcome scale mismatch across rows")
    if len(teacher_identities) > 1:
        raise ReliabilityContractError("teacher identity mismatch across rows")
    teacher_type = next(iter(teacher_types))
    score_scale_id = next(iter(score_scale_ids))
    short_scale, full_scale = next(iter(outcome_scale_values))
    teacher_identity = (
        next(iter(teacher_identities))
        if teacher_identities
        else canonical_hash(
            {
                "teacher_type": teacher_type,
                "teacher_score_scale_id": score_scale_id,
            }
        )
    )
    return (
        normalized,
        teacher_type,
        score_scale_id,
        {"short": short_scale, "full": full_scale},
        teacher_identity,
    )


def _build_slice(
    *,
    requested_key: str,
    source_rows: Sequence[Mapping[str, Any]],
    fallback_level: str,
    bootstrap_seed: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    """選択済み rows から horizon CI・error UCB・weight を一 slice にまとめる。

    exact/fallback で異なる minimum support を使い、不足時は weight 0 blocker にする。
    """
    min_decisions = 30 if fallback_level == "exact" else 60
    min_clusters = 10 if fallback_level == "exact" else 15
    decision_ids = {str(row["decision_id"]) for row in source_rows}
    clusters = {str(row["cluster"]) for row in source_rows}
    horizon_estimates: dict[str, dict[str, float]] = {}
    outcome_refs: set[str] = set()
    for horizon_index, horizon in enumerate(("short", "full")):
        observations: list[tuple[str, float]] = []
        for row in source_rows:
            for pair in row["pairs"]:
                observations.append(
                    (str(row["cluster"]), float(pair["residuals"][horizon]))
                )
                outcome_refs.add(str(pair["outcome_refs"][horizon]))
        if observations:
            point, lower, upper, _ = _cluster_bootstrap_median(
                observations,
                seed=bootstrap_seed + horizon_index,
                resamples=bootstrap_resamples,
            )
        else:
            point = lower = upper = ERROR_LIMIT_Z
        horizon_estimates[horizon] = {
            "median_abs_residual_z": point,
            "ci_lower_z": lower,
            "ci_upper_z": upper,
        }
    support_ok = (
        len(decision_ids) >= min_decisions and len(clusters) >= min_clusters
    )
    point = max(
        estimate["median_abs_residual_z"]
        for estimate in horizon_estimates.values()
    )
    lower = max(
        estimate["ci_lower_z"] for estimate in horizon_estimates.values()
    )
    upper = max(
        estimate["ci_upper_z"] for estimate in horizon_estimates.values()
    )
    error_ucb = upper
    support_factor = min(len(clusters) / min_clusters, 1.0)
    weight = (
        support_factor * min(max(1.0 - error_ucb / ERROR_LIMIT_Z, 0.0), 1.0)
        if support_ok
        else 0.0
    )
    blocker = not support_ok or error_ucb >= ERROR_LIMIT_Z
    return {
        "slice_key": requested_key,
        "fallback_level": fallback_level,
        "support_decisions": len(decision_ids),
        "n_effective_clusters": len(clusters),
        "minimum_support_decisions": min_decisions,
        "minimum_effective_clusters": min_clusters,
        "median_abs_residual_z": point,
        "ci_lower_z": lower,
        "ci_upper_z": upper,
        "horizon_estimates": horizon_estimates,
        "error_ucb_z": error_ucb,
        "weight": weight,
        "release_blocker": blocker,
        "source_outcome_refs": sorted(outcome_refs),
    }


def fit_teacher_reliability(
    rows: Sequence[Mapping[str, Any]],
    *,
    development_split_id: str,
    integration_fidelity_identity: str,
    bootstrap_seed: int = 20260718,
    bootstrap_resamples: int = 2000,
) -> dict[str, Any]:
    """development pairwise rows から exact/fallback calibration を fit する。

    全 observed exact slices を評価し、support 不足時だけ同 kind 全 elapsed rows へ fallback する。
    """
    _require_sha256(development_split_id, "development_split_id")
    _require_sha256(
        integration_fidelity_identity,
        "integration_fidelity_identity",
    )
    if type(bootstrap_seed) is not int or bootstrap_seed < 0:
        raise ReliabilityContractError("bootstrap_seed must be non-negative")
    if type(bootstrap_resamples) is not int or bootstrap_resamples < 20:
        raise ReliabilityContractError("bootstrap_resamples must be at least 20")
    (
        normalized,
        teacher_type,
        score_scale_id,
        outcome_scale_ids,
        teacher_identity,
    ) = _normalize_rows(rows)

    exact_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    fallback_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in normalized:
        exact_groups[(row["choice_kind"], row["elapsed_band"])].append(row)
        fallback_groups[row["choice_kind"]].append(row)
    slices: list[dict[str, Any]] = []
    for index, ((choice_kind, band), exact_rows) in enumerate(
        sorted(exact_groups.items())
    ):
        exact = _build_slice(
            requested_key=f"{teacher_type}|{choice_kind}|{band}",
            source_rows=exact_rows,
            fallback_level="exact",
            bootstrap_seed=bootstrap_seed + index * 10,
            bootstrap_resamples=bootstrap_resamples,
        )
        if (
            exact["support_decisions"] >= exact["minimum_support_decisions"]
            and exact["n_effective_clusters"]
            >= exact["minimum_effective_clusters"]
        ):
            slices.append(exact)
            continue
        fallback = _build_slice(
            requested_key=f"{teacher_type}|{choice_kind}|{band}",
            source_rows=fallback_groups[choice_kind],
            fallback_level="choice_kind",
            bootstrap_seed=bootstrap_seed + index * 10,
            bootstrap_resamples=bootstrap_resamples,
        )
        slices.append(fallback)

    branch_refs = sorted(
        {
            str(pair["outcome_refs"][horizon])
            for row in normalized
            for pair in row["pairs"]
            for horizon in ("short", "full")
        }
    )
    artifact: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "calibration_identity": "",
        "teacher_identity": teacher_identity,
        "teacher_type": teacher_type,
        "teacher_score_scale_id": score_scale_id,
        "outcome_scale_ids": outcome_scale_ids,
        "development_split_id": development_split_id,
        "integration_fidelity_identity": integration_fidelity_identity,
        "branch_outcome_refs": branch_refs,
        "slices": slices,
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_resamples": bootstrap_resamples,
        "error_limit_z": ERROR_LIMIT_Z,
    }
    artifact["calibration_identity"] = canonical_hash(
        {
            key: artifact[key]
            for key in sorted(_FIELDS - {"calibration_identity"})
        }
    )
    validate_reliability_calibration(artifact)
    return artifact


def validate_reliability_calibration(artifact: Any) -> None:
    """calibration の schema・binding・support/weight・identity を検証する。

    weight を CI/UCB から再計算し、fabricated support や scale identity 改変を拒否する。
    """
    if not isinstance(artifact, Mapping):
        raise ReliabilityContractError("calibration must be an object")
    actual = frozenset(artifact)
    if actual != _FIELDS:
        raise ReliabilityContractError(
            "calibration fields mismatch: "
            f"missing={sorted(_FIELDS - actual)}, unknown={sorted(actual - _FIELDS)}"
        )
    if artifact["schema_version"] != SCHEMA_VERSION:
        raise ReliabilityContractError("unsupported calibration schema_version")
    for label in (
        "teacher_identity",
        "teacher_score_scale_id",
        "development_split_id",
        "integration_fidelity_identity",
    ):
        _require_sha256(artifact[label], label)
    outcome_scales = artifact["outcome_scale_ids"]
    if (
        not isinstance(outcome_scales, Mapping)
        or set(outcome_scales) != {"short", "full"}
    ):
        raise ReliabilityContractError("short/full outcome scale IDs are required")
    for horizon in ("short", "full"):
        _require_sha256(outcome_scales[horizon], f"outcome_scale_ids.{horizon}")
    refs = artifact["branch_outcome_refs"]
    if (
        not isinstance(refs, list)
        or not refs
        or any(not isinstance(ref, str) or not ref for ref in refs)
        or len(set(refs)) != len(refs)
    ):
        raise ReliabilityContractError("branch outcome refs are invalid")
    if artifact["error_limit_z"] != ERROR_LIMIT_Z:
        raise ReliabilityContractError("error limit changed")
    if (
        type(artifact["bootstrap_seed"]) is not int
        or artifact["bootstrap_seed"] < 0
        or type(artifact["bootstrap_resamples"]) is not int
        or artifact["bootstrap_resamples"] < 20
    ):
        raise ReliabilityContractError("bootstrap configuration is invalid")
    slices = artifact["slices"]
    if not isinstance(slices, list) or not slices:
        raise ReliabilityContractError("calibration slices are required")
    known_refs = set(refs)
    for item in slices:
        if not isinstance(item, Mapping):
            raise ReliabilityContractError("slice must be an object")
        required = {
            "slice_key",
            "fallback_level",
            "support_decisions",
            "n_effective_clusters",
            "minimum_support_decisions",
            "minimum_effective_clusters",
            "median_abs_residual_z",
            "ci_lower_z",
            "ci_upper_z",
            "horizon_estimates",
            "error_ucb_z",
            "weight",
            "release_blocker",
            "source_outcome_refs",
        }
        if set(item) != required:
            raise ReliabilityContractError("slice fields mismatch")
        if item["fallback_level"] not in {"exact", "choice_kind"}:
            raise ReliabilityContractError("unsupported fallback level")
        expected_min_decisions = (
            30 if item["fallback_level"] == "exact" else 60
        )
        expected_min_clusters = (
            10 if item["fallback_level"] == "exact" else 15
        )
        if (
            item["minimum_support_decisions"] != expected_min_decisions
            or item["minimum_effective_clusters"] != expected_min_clusters
        ):
            raise ReliabilityContractError("slice support thresholds changed")
        support = item["support_decisions"]
        clusters = item["n_effective_clusters"]
        if (
            type(support) is not int
            or support < 0
            or type(clusters) is not int
            or clusters < 0
            or clusters > support
        ):
            raise ReliabilityContractError("slice support is invalid")
        horizon_estimates = item["horizon_estimates"]
        if (
            not isinstance(horizon_estimates, Mapping)
            or set(horizon_estimates) != {"short", "full"}
        ):
            raise ReliabilityContractError("horizon estimates are incomplete")
        for horizon in ("short", "full"):
            estimate = horizon_estimates[horizon]
            if not isinstance(estimate, Mapping) or set(estimate) != {
                "median_abs_residual_z",
                "ci_lower_z",
                "ci_upper_z",
            }:
                raise ReliabilityContractError("horizon estimate schema mismatch")
            values = [
                estimate["median_abs_residual_z"],
                estimate["ci_lower_z"],
                estimate["ci_upper_z"],
            ]
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
                for value in values
            ):
                raise ReliabilityContractError("horizon estimate is invalid")
            if float(estimate["ci_lower_z"]) > float(estimate["ci_upper_z"]):
                raise ReliabilityContractError("horizon CI is reversed")
        expected_point = max(
            horizon_estimates[horizon]["median_abs_residual_z"]
            for horizon in ("short", "full")
        )
        expected_lower = max(
            horizon_estimates[horizon]["ci_lower_z"]
            for horizon in ("short", "full")
        )
        expected_upper = max(
            horizon_estimates[horizon]["ci_upper_z"]
            for horizon in ("short", "full")
        )
        if any(
            abs(float(item[field]) - float(expected)) > 1e-12
            for field, expected in (
                ("median_abs_residual_z", expected_point),
                ("ci_lower_z", expected_lower),
                ("ci_upper_z", expected_upper),
                ("error_ucb_z", expected_upper),
            )
        ):
            raise ReliabilityContractError("slice point/CI/UCB is inconsistent")
        support_ok = (
            support >= expected_min_decisions
            and clusters >= expected_min_clusters
        )
        support_factor = min(clusters / expected_min_clusters, 1.0)
        expected_weight = (
            support_factor
            * min(max(1.0 - expected_upper / ERROR_LIMIT_Z, 0.0), 1.0)
            if support_ok
            else 0.0
        )
        if abs(float(item["weight"]) - expected_weight) > 1e-12:
            raise ReliabilityContractError("slice weight is fabricated")
        expected_blocker = not support_ok or expected_upper >= ERROR_LIMIT_Z
        if item["release_blocker"] is not expected_blocker:
            raise ReliabilityContractError("slice blocker is inconsistent")
        source_refs = item["source_outcome_refs"]
        if (
            not isinstance(source_refs, list)
            or not source_refs
            or not set(source_refs).issubset(known_refs)
        ):
            raise ReliabilityContractError("slice outcome lineage is invalid")
    expected_identity = canonical_hash(
        {
            key: artifact[key]
            for key in sorted(_FIELDS - {"calibration_identity"})
        }
    )
    if artifact["calibration_identity"] != expected_identity:
        raise ReliabilityContractError("calibration identity mismatch")
