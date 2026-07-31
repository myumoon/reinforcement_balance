"""paired branch outcomes に対する tie-aware teacher 指標を計算する。

top-1 agreement・pairwise・NDCG@3 は primary rank、regret は development-only
normalization の diagnostic utility を使い、short/full horizon を独立評価する。
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from reinbalance_survivors_contracts.canonical_json import canonical_hash

LABEL_RELEASE_SCHEMA_VERSION = "survivors.label_release_verdict.v1"
HORIZON_SECONDS = {"short": 300.0, "full": 1800.0}
TEACHER_TIE_EPSILON_Z = 0.02
UTILITY_TIE_TOLERANCE = 1e-6
GATE_THRESHOLDS = {
    "minimum_short_decisions": 300,
    "minimum_short_episodes": 30,
    "minimum_full_decisions": 60,
    "minimum_full_episodes": 20,
    "minimum_choice_kind_decisions": 30,
    "top1_agreement": 0.65,
    "top1_cluster_ci_lower": 0.60,
    "choice_kind_pairwise": 0.55,
    "mean_ndcg_at_3": 0.80,
    "median_normalized_regret": 0.10,
    "maximum_quarantine_rate": 0.01,
    "maximum_blocking_slices": 0,
}
_OUTCOME_FIELDS = (
    "stage_cleared_or_survived_horizon",
    "survival_seconds",
    "level_gain",
    "gem_gain",
    "kill_gain",
)


class TeacherValidationError(ValueError):
    """teacher validation input・normalization・verdict binding 違反の例外。

    非有限 outcome や欠損 horizon を silent tie/zero utility に変換せず拒否する。
    """


@dataclass(frozen=True)
class OutcomeNormalizer:
    """一 horizon の development-only diagnostic utility scale。

    ground-truth ranking と分離し、level/gem/kill gains の基準と identity を固定する。
    """

    horizon: str
    partition_id: str
    level_gain_scale: float
    gem_gain_scale: float
    kill_gain_scale: float
    normalization_identity: str

    @classmethod
    def fit(
        cls,
        decisions: Sequence[Mapping[str, Any]],
        horizon: str,
        *,
        partition_id: str,
    ) -> "OutcomeNormalizer":
        """development train outcomes の positive q95 から scale を fit する。

        partition field がある行は development_train 以外を明示的に拒否する。
        """
        if horizon not in HORIZON_SECONDS:
            raise TeacherValidationError("unsupported horizon")
        if not isinstance(partition_id, str) or not partition_id:
            raise TeacherValidationError("partition_id must be non-empty")
        if isinstance(decisions, (str, bytes)) or not decisions:
            raise TeacherValidationError("normalization decisions are required")
        gains: dict[str, list[float]] = {
            "level_gain": [],
            "gem_gain": [],
            "kill_gain": [],
        }
        source_refs: list[str] = []
        for decision_index, decision in enumerate(decisions):
            if not isinstance(decision, Mapping):
                raise TeacherValidationError("decision must be an object")
            partition = decision.get("partition")
            if partition is not None and partition != "development_train":
                raise TeacherValidationError(
                    "outcome normalization may use development_train only"
                )
            candidates = decision.get("candidates")
            if not isinstance(candidates, list) or not candidates:
                raise TeacherValidationError("normalization candidates are required")
            for candidate_index, candidate in enumerate(candidates):
                outcome = _validated_outcome(candidate.get(horizon), horizon)
                for field in gains:
                    gains[field].append(max(0.0, float(outcome[field])))
                source_refs.append(
                    str(
                        outcome.get(
                            "outcome_ref",
                            f"{decision.get('decision_id', decision_index)}"
                            f":{candidate.get('choice_id', candidate_index)}:{horizon}",
                        )
                    )
                )
        scales = {
            field: max(
                float(
                    np.quantile(
                        np.asarray(values, dtype=np.float64),
                        0.95,
                        method="linear",
                    )
                ),
                1e-6,
            )
            for field, values in gains.items()
        }
        payload = {
            "schema_version": "survivors.outcome_utility_scale.v1",
            "horizon": horizon,
            "partition_id": partition_id,
            "level_gain_scale": scales["level_gain"],
            "gem_gain_scale": scales["gem_gain"],
            "kill_gain_scale": scales["kill_gain"],
            "source_outcome_refs": sorted(source_refs),
        }
        return cls(
            horizon=horizon,
            partition_id=partition_id,
            level_gain_scale=scales["level_gain"],
            gem_gain_scale=scales["gem_gain"],
            kill_gain_scale=scales["kill_gain"],
            normalization_identity=canonical_hash(payload),
        )

    def utility(self, outcome: Mapping[str, Any]) -> float:
        """仕様の diagnostic scalar utility を計算する。

        NovelD・shaped reward・HP penalty は outcome provenance に残っても参照しない。
        """
        checked = _validated_outcome(outcome, self.horizon)
        survival_fraction = min(
            max(float(checked["survival_seconds"]) / HORIZON_SECONDS[self.horizon], 0.0),
            1.0,
        )
        return (
            survival_fraction
            + 0.50
            * max(float(checked["level_gain"]), 0.0)
            / self.level_gain_scale
            + 0.25
            * max(float(checked["gem_gain"]), 0.0)
            / self.gem_gain_scale
            + 0.05
            * max(float(checked["kill_gain"]), 0.0)
            / self.kill_gain_scale
        )

    def to_wire(self) -> dict[str, Any]:
        """normalization identity と fit parameter を artifact object に変換する。

        horizon outcome scale と teacher score scale を別 schema として保持する。
        """
        return {
            "schema_version": "survivors.outcome_utility_scale.v1",
            "normalization_identity": self.normalization_identity,
            "horizon": self.horizon,
            "partition_id": self.partition_id,
            "level_gain_scale": self.level_gain_scale,
            "gem_gain_scale": self.gem_gain_scale,
            "kill_gain_scale": self.kill_gain_scale,
        }


def _validated_outcome(value: Any, horizon: str) -> Mapping[str, Any]:
    """short/full outcome の primary fields と finite 値を検証する。

    provenance の追加 field は許可するが、ground-truth 計算へは流さない。
    """
    if not isinstance(value, Mapping):
        raise TeacherValidationError(f"{horizon} outcome must be an object")
    missing = set(_OUTCOME_FIELDS) - set(value)
    if missing:
        raise TeacherValidationError(
            f"{horizon} outcome missing fields: {sorted(missing)}"
        )
    if type(value["stage_cleared_or_survived_horizon"]) is not bool:
        raise TeacherValidationError("stage/survival flag must be bool")
    for field in _OUTCOME_FIELDS[1:]:
        item = value[field]
        if (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            or float(item) < 0.0
        ):
            raise TeacherValidationError(f"{field} must be finite and non-negative")
    if float(value["survival_seconds"]) > HORIZON_SECONDS[horizon]:
        raise TeacherValidationError("survival_seconds exceeds horizon")
    return value


def primary_rank(outcome: Mapping[str, Any], horizon: str) -> tuple[Any, ...]:
    """仕様の lexicographic primary rank tuple を返す。

    training reward provenance を順位へ加えず、五つの外的 gameplay outcomes だけを使う。
    """
    checked = _validated_outcome(outcome, horizon)
    return (
        bool(checked["stage_cleared_or_survived_horizon"]),
        float(checked["survival_seconds"]),
        float(checked["level_gain"]),
        float(checked["gem_gain"]),
        float(checked["kill_gain"]),
    )


def _truth_relation(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    horizon: str,
    left_utility: float,
    right_utility: float,
) -> int:
    """二候補の truth relation を -1/0/+1 で返す。

    primary tuple 同値または utility 差が登録 tolerance 内なら tie とする。
    """
    left_rank = primary_rank(left, horizon)
    right_rank = primary_rank(right, horizon)
    if left_rank == right_rank or abs(left_utility - right_utility) <= UTILITY_TIE_TOLERANCE:
        return 0
    return 1 if left_rank > right_rank else -1


def _teacher_relation(left: float, right: float) -> int:
    """teacher score pair を tie epsilon 付き関係へ変換する。

    score scale の z 差に対して事前固定 0.02 を使う。
    """
    difference = left - right
    if abs(difference) <= TEACHER_TIE_EPSILON_Z:
        return 0
    return 1 if difference > 0.0 else -1


def _ndcg_at_3(
    candidates: Sequence[Mapping[str, Any]],
    teacher_scores: Sequence[float],
    truth_relations: Mapping[tuple[int, int], int],
) -> float:
    """truth ties を同 gain とする NDCG@3 を返す。

    teacher tie block は占有順位の discount を平均し、候補入力順を評価値へ混入させない。
    """
    count = len(candidates)
    wins = []
    for left in range(count):
        win_count = 0.0
        for right in range(count):
            if left == right:
                continue
            relation = (
                truth_relations[(left, right)]
                if left < right
                else -truth_relations[(right, left)]
            )
            if relation > 0:
                win_count += 1.0
            elif relation == 0:
                win_count += 0.5
        wins.append(win_count)
    distinct = sorted(set(wins), reverse=True)
    dense_rank = {value: index for index, value in enumerate(distinct)}
    gains = [2.0 ** (len(distinct) - dense_rank[value]) - 1.0 for value in wins]
    teacher_order = sorted(
        range(count),
        key=lambda index: (
            -teacher_scores[index],
            str(candidates[index].get("choice_id", "")),
        ),
    )
    teacher_blocks: list[list[int]] = []
    for index in teacher_order:
        if (
            not teacher_blocks
            or teacher_scores[teacher_blocks[-1][0]] - teacher_scores[index]
            > TEACHER_TIE_EPSILON_Z
        ):
            teacher_blocks.append([index])
        else:
            teacher_blocks[-1].append(index)
    ideal_order = sorted(
        range(count),
        key=lambda index: (
            -gains[index],
            str(candidates[index].get("choice_id", "")),
        ),
    )

    def dcg(order: Sequence[int]) -> float:
        """上位3候補の discounted cumulative gain を計算する。

        候補数が3未満でも存在する候補だけを使う。
        """
        return sum(
            gains[index] / math.log2(position + 2.0)
            for position, index in enumerate(order[:3])
        )

    def tie_aware_dcg(blocks: Sequence[Sequence[int]]) -> float:
        """teacher tie block の全順列に対する期待 DCG@3 を計算する。

        cutoff をまたぐ tie も block 全候補の平均 gain で評価し、列挙順依存を除く。
        """
        total = 0.0
        position = 0
        for block in blocks:
            used = min(len(block), max(0, 3 - position))
            if used:
                mean_gain = float(np.mean([gains[index] for index in block]))
                total += mean_gain * sum(
                    1.0 / math.log2(rank + 2.0)
                    for rank in range(position, position + used)
                )
            position += len(block)
            if position >= 3:
                break
        return total

    ideal = dcg(ideal_order)
    return 1.0 if ideal <= 0.0 else tie_aware_dcg(teacher_blocks) / ideal


def _decision_metrics(
    decision: Mapping[str, Any],
    horizon: str,
    normalizer: OutcomeNormalizer,
) -> dict[str, Any]:
    """一 decision の tie-aware 指標と cluster identity を計算する。

    candidate pair を一度だけ列挙し、top1/NDCG/regret と pairwise の基礎値を返す。
    """
    candidates = decision.get("candidates")
    if not isinstance(candidates, list) or len(candidates) < 2:
        raise TeacherValidationError("each decision requires at least two candidates")
    choice_ids: set[str] = set()
    scores: list[float] = []
    outcomes: list[Mapping[str, Any]] = []
    utilities: list[float] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise TeacherValidationError("candidate must be an object")
        choice_id = candidate.get("choice_id")
        if (
            not isinstance(choice_id, str)
            or not choice_id
            or choice_id in choice_ids
        ):
            raise TeacherValidationError("candidate choice IDs must be unique")
        choice_ids.add(choice_id)
        score = candidate.get("teacher_score_z", candidate.get("teacher_score"))
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
        ):
            raise TeacherValidationError("teacher score must be finite")
        outcome = _validated_outcome(candidate.get(horizon), horizon)
        scores.append(float(score))
        outcomes.append(outcome)
        utilities.append(normalizer.utility(outcome))

    truth_relations: dict[tuple[int, int], int] = {}
    pair_credits: list[float] = []
    tau_values: list[float] = []
    for left in range(len(candidates)):
        for right in range(left + 1, len(candidates)):
            truth = _truth_relation(
                outcomes[left],
                outcomes[right],
                horizon=horizon,
                left_utility=utilities[left],
                right_utility=utilities[right],
            )
            teacher = _teacher_relation(scores[left], scores[right])
            truth_relations[(left, right)] = truth
            if truth == teacher:
                credit = 1.0
            elif truth == 0 or teacher == 0:
                credit = 0.5
            else:
                credit = 0.0
            pair_credits.append(credit)
            tau_values.append(
                1.0 if truth == teacher and truth != 0
                else 0.0 if truth == 0 and teacher == 0
                else -1.0 if truth * teacher < 0
                else 0.0
            )

    teacher_max = max(scores)
    teacher_top = {
        index
        for index, score in enumerate(scores)
        if teacher_max - score <= TEACHER_TIE_EPSILON_Z
    }
    truth_top: set[int] = set()
    for candidate_index in range(len(candidates)):
        beaten = False
        for other_index in range(len(candidates)):
            if candidate_index == other_index:
                continue
            if candidate_index < other_index:
                relation = truth_relations[(candidate_index, other_index)]
            else:
                relation = -truth_relations[(other_index, candidate_index)]
            if relation < 0:
                beaten = True
                break
        if not beaten:
            truth_top.add(candidate_index)
    top1 = 1.0 if teacher_top & truth_top else 0.0
    utility_range = max(utilities) - min(utilities)
    selected_utility = float(
        np.mean([utilities[index] for index in sorted(teacher_top)])
    )
    regret = (
        0.0
        if utility_range <= UTILITY_TIE_TOLERANCE
        else (max(utilities) - selected_utility) / utility_range
    )
    episode_id = decision.get("episode_id")
    seed_id = decision.get("seed_cluster_id", episode_id)
    if not isinstance(episode_id, str) or not episode_id:
        raise TeacherValidationError("episode_id must be non-empty")
    if not isinstance(seed_id, str) or not seed_id:
        raise TeacherValidationError("seed_cluster_id must be non-empty")
    return {
        "top1": top1,
        "pairwise": float(np.mean(pair_credits)),
        "ndcg_at_3": _ndcg_at_3(candidates, scores, truth_relations),
        "normalized_regret": regret,
        "rank_correlation": float(np.mean(tau_values)),
        "cluster": f"{episode_id}\x1f{seed_id}",
        "episode_id": episode_id,
        "choice_kind": decision.get("choice_kind", "unknown"),
        "item_id": decision.get("item_id", "unknown"),
        "item_level": decision.get("item_level", "unknown"),
        "elapsed_band": elapsed_band(decision.get("elapsed_seconds", 0.0)),
    }


def elapsed_band(seconds: Any) -> str:
    """elapsed seconds を blocking report 用の固定 4 band へ変換する。

    0〜30分外や非有限値は report tuning で丸めず拒否する。
    """
    if (
        isinstance(seconds, bool)
        or not isinstance(seconds, (int, float))
        or not math.isfinite(float(seconds))
    ):
        raise TeacherValidationError("elapsed_seconds must be finite")
    value = float(seconds)
    if value < 0.0:
        raise TeacherValidationError("elapsed_seconds outside validation horizon")
    if 0.0 <= value < 300.0:
        return "0-5m"
    if value < 600.0:
        return "5-10m"
    if value < 1200.0:
        return "10-20m"
    if value <= 1800.0:
        return "20-30m"
    raise TeacherValidationError("elapsed_seconds outside validation horizon")


def _cluster_bootstrap_mean_ci(
    rows: Sequence[Mapping[str, Any]],
    metric: str,
    *,
    seed: int,
    resamples: int,
) -> tuple[float, float]:
    """episode/seed cluster bootstrap で mean metric の 95% CI を返す。

    同 cluster の decisions を一緒に再標本化し、row 独立仮定を避ける。
    """
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["cluster"])].append(float(row[metric]))
    clusters = sorted(grouped)
    rng = np.random.default_rng(seed)
    values = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sampled = rng.integers(0, len(clusters), size=len(clusters))
        observations = [
            observation
            for cluster_index in sampled
            for observation in grouped[clusters[int(cluster_index)]]
        ]
        values[index] = float(np.mean(observations))
    return tuple(
        float(value)
        for value in np.quantile(values, [0.025, 0.975], method="linear")
    )


def evaluate_teacher(
    decisions: Sequence[Mapping[str, Any]],
    normalizers: Mapping[str, OutcomeNormalizer],
    *,
    horizons: Sequence[str] = ("short", "full"),
    bootstrap_seed: int = 20260718,
    bootstrap_resamples: int = 2000,
    quarantine_count: int = 0,
) -> dict[str, Any]:
    """short/full horizon の overall・kind・elapsed metrics report を生成する。

    normalizer identity を horizon ごとに保存し、teacher score scale と混同しない。
    """
    if isinstance(decisions, (str, bytes)) or not decisions:
        raise TeacherValidationError("teacher decisions are required")
    if type(bootstrap_resamples) is not int or bootstrap_resamples < 20:
        raise TeacherValidationError("bootstrap_resamples must be at least 20")
    if type(quarantine_count) is not int or quarantine_count < 0:
        raise TeacherValidationError("quarantine_count must be non-negative")
    horizon_reports: dict[str, Any] = {}
    for horizon_index, horizon in enumerate(horizons):
        if horizon not in HORIZON_SECONDS or horizon not in normalizers:
            raise TeacherValidationError(f"missing normalizer for {horizon}")
        normalizer = normalizers[horizon]
        if normalizer.horizon != horizon:
            raise TeacherValidationError("outcome normalizer horizon mismatch")
        rows = [
            _decision_metrics(decision, horizon, normalizer)
            for decision in decisions
        ]
        top1_lower, top1_upper = _cluster_bootstrap_mean_ci(
            rows,
            "top1",
            seed=bootstrap_seed + horizon_index,
            resamples=bootstrap_resamples,
        )
        by_kind: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        by_item: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        by_level: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        by_elapsed: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            by_kind[str(row["choice_kind"])].append(row)
            by_item[str(row["item_id"])].append(row)
            by_level[str(row["item_level"])].append(row)
            by_elapsed[str(row["elapsed_band"])].append(row)
        horizon_reports[horizon] = {
            "top1_agreement": float(np.mean([row["top1"] for row in rows])),
            "top1_ci_lower": top1_lower,
            "top1_ci_upper": top1_upper,
            "pairwise_agreement": float(
                np.mean([row["pairwise"] for row in rows])
            ),
            "mean_ndcg_at_3": float(
                np.mean([row["ndcg_at_3"] for row in rows])
            ),
            "median_normalized_regret": float(
                np.median([row["normalized_regret"] for row in rows])
            ),
            "rank_correlation": float(
                np.mean([row["rank_correlation"] for row in rows])
            ),
            "support_decisions": len(rows),
            "support_episodes": len({row["episode_id"] for row in rows}),
            "n_effective_clusters": len({row["cluster"] for row in rows}),
            "choice_kind_pairwise": {
                kind: {
                    "pairwise_agreement": float(
                        np.mean([row["pairwise"] for row in kind_rows])
                    ),
                    "support_decisions": len(kind_rows),
                }
                for kind, kind_rows in sorted(by_kind.items())
            },
            "item_slices": {
                item_id: {
                    "pairwise_agreement": float(
                        np.mean([row["pairwise"] for row in item_rows])
                    ),
                    "support_decisions": len(item_rows),
                }
                for item_id, item_rows in sorted(by_item.items())
            },
            "level_slices": {
                item_level: {
                    "pairwise_agreement": float(
                        np.mean([row["pairwise"] for row in level_rows])
                    ),
                    "support_decisions": len(level_rows),
                }
                for item_level, level_rows in sorted(by_level.items())
            },
            "elapsed_slices": {
                band: {
                    "top1_agreement": float(
                        np.mean([row["top1"] for row in band_rows])
                    ),
                    "pairwise_agreement": float(
                        np.mean([row["pairwise"] for row in band_rows])
                    ),
                    "mean_ndcg_at_3": float(
                        np.mean([row["ndcg_at_3"] for row in band_rows])
                    ),
                    "median_normalized_regret": float(
                        np.median(
                            [row["normalized_regret"] for row in band_rows]
                        )
                    ),
                    "support_decisions": len(band_rows),
                }
                for band, band_rows in sorted(by_elapsed.items())
            },
            "outcome_normalization": normalizer.to_wire(),
        }
    total = len(decisions) + quarantine_count
    return {
        "schema_version": "survivors.teacher_validation_report.v1",
        "horizons": horizon_reports,
        "quarantine_count": quarantine_count,
        "quarantine_rate": 0.0 if total == 0 else quarantine_count / total,
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_resamples": bootstrap_resamples,
    }


def evaluate_release_gates(report: Mapping[str, Any]) -> dict[str, Any]:
    """pre-registered formal thresholds を short/full report へ適用する。

    全 gate を固定値で評価し、呼び出し側から override 値を受け取らない。
    """
    if (
        not isinstance(report, Mapping)
        or report.get("schema_version")
        != "survivors.teacher_validation_report.v1"
    ):
        raise TeacherValidationError("teacher validation report is invalid")
    horizons = report.get("horizons")
    if not isinstance(horizons, Mapping) or not {"short", "full"}.issubset(horizons):
        raise TeacherValidationError("short/full reports are required")
    checks: dict[str, bool] = {}
    blocking_slices: list[str] = []
    for horizon in ("short", "full"):
        metrics = horizons[horizon]
        minimum_decisions = GATE_THRESHOLDS[
            f"minimum_{horizon}_decisions"
        ]
        minimum_episodes = GATE_THRESHOLDS[f"minimum_{horizon}_episodes"]
        checks[f"{horizon}.support_decisions"] = (
            metrics["support_decisions"] >= minimum_decisions
        )
        checks[f"{horizon}.support_episodes"] = (
            metrics["support_episodes"] >= minimum_episodes
        )
        checks[f"{horizon}.top1_agreement"] = (
            metrics["top1_agreement"] >= GATE_THRESHOLDS["top1_agreement"]
        )
        checks[f"{horizon}.top1_ci_lower"] = (
            metrics["top1_ci_lower"]
            >= GATE_THRESHOLDS["top1_cluster_ci_lower"]
        )
        checks[f"{horizon}.mean_ndcg_at_3"] = (
            metrics["mean_ndcg_at_3"]
            >= GATE_THRESHOLDS["mean_ndcg_at_3"]
        )
        checks[f"{horizon}.median_normalized_regret"] = (
            metrics["median_normalized_regret"]
            <= GATE_THRESHOLDS["median_normalized_regret"]
        )
        for kind, kind_metrics in metrics["choice_kind_pairwise"].items():
            support_passed = (
                kind_metrics["support_decisions"]
                >= GATE_THRESHOLDS["minimum_choice_kind_decisions"]
            )
            pairwise_passed = (
                kind_metrics["pairwise_agreement"]
                >= GATE_THRESHOLDS["choice_kind_pairwise"]
            )
            checks[f"{horizon}.choice_kind.{kind}.support"] = support_passed
            checks[f"{horizon}.choice_kind.{kind}.pairwise"] = pairwise_passed
            if not support_passed or not pairwise_passed:
                blocking_slices.append(f"{horizon}|choice_kind|{kind}")
        for band, band_metrics in metrics["elapsed_slices"].items():
            if (
                band_metrics["top1_agreement"]
                < GATE_THRESHOLDS["top1_agreement"]
                or band_metrics["pairwise_agreement"]
                < GATE_THRESHOLDS["choice_kind_pairwise"]
                or band_metrics["mean_ndcg_at_3"]
                < GATE_THRESHOLDS["mean_ndcg_at_3"]
                or band_metrics["median_normalized_regret"]
                > GATE_THRESHOLDS["median_normalized_regret"]
            ):
                blocking_slices.append(f"{horizon}|elapsed|{band}")
    checks["quarantine_rate"] = (
        float(report.get("quarantine_rate", 1.0))
        <= GATE_THRESHOLDS["maximum_quarantine_rate"]
    )
    release_blockers = report.get("release_blockers", [])
    if not isinstance(release_blockers, list):
        raise TeacherValidationError("release_blockers must be a list")
    checks["release_blockers"] = not release_blockers
    checks["blocking_slices"] = (
        len(set(blocking_slices))
        <= GATE_THRESHOLDS["maximum_blocking_slices"]
    )
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "blocking_slices": sorted(set(blocking_slices)),
        "thresholds": dict(GATE_THRESHOLDS),
    }


def make_label_release_verdict(
    *,
    report: Mapping[str, Any],
    source_descriptor_identity: str,
    split_identity: str,
    reliability_identity: str,
    score_scale_identity: str,
    integration_fidelity_identity: str,
) -> dict[str, Any]:
    """validation report と immutable parents へ束縛した release verdict を作る。

    gate override 入力を持たず、source descriptor identity を変更せず subject に保存する。
    """
    for label, value in (
        ("source_descriptor_identity", source_descriptor_identity),
        ("split_identity", split_identity),
        ("reliability_identity", reliability_identity),
        ("score_scale_identity", score_scale_identity),
        ("integration_fidelity_identity", integration_fidelity_identity),
    ):
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise TeacherValidationError(f"{label} must be lowercase sha256")
    gates = evaluate_release_gates(report)
    subject = {
        "source_descriptor_identity": source_descriptor_identity,
        "split_identity": split_identity,
        "reliability_identity": reliability_identity,
        "score_scale_identity": score_scale_identity,
        "integration_fidelity_identity": integration_fidelity_identity,
    }
    payload = {
        "schema_version": LABEL_RELEASE_SCHEMA_VERSION,
        "subject": subject,
        "status": "PASS" if gates["passed"] else "FAIL",
        "gates": gates,
        "report_identity": canonical_hash(report),
    }
    return {
        **payload,
        "verdict_identity": canonical_hash(payload),
    }
