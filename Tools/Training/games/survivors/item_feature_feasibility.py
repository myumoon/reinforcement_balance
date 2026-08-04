"""ItemSelector の二つの画面観測 feature 案について Bayes ceiling ablation を行う。

解析対象は frozen manifest の train/validation に限定し、episode bootstrap CI、入力次元、
実 availability、latency class、事前登録 floor と収集 hard cap を JSON/Markdown data card に残す。
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from reinbalance_survivors_contracts.canonical_json import (
    canonical_hash,
    canonical_json_bytes,
)
from reinbalance_survivors_contracts.item_decision import ItemDecisionFeatures

from games.survivors.item_selector_dataset import (
    LearningCurvePolicy,
    SplitManifest,
    SplitSealedError,
    calculate_hard_cap,
)

FEATURE_VARIANTS = ("context_only_v1", "context_danger_v1")
DATA_CARD_SCHEMA_VERSION = "survivors.item_feature_data_card.v1"


class FeatureFeasibilityError(ValueError):
    """feature leakage、split binding、ceiling 入力の契約違反を表す。

    不完全な data card を出力せず method selection を fail-closed に停止する。
    """


def _label(row: Mapping[str, Any]) -> str:
    """teacher label 領域から best item identity を検証して返す。

    behavior choice や feature 内の item position を正解として代用しない。
    """
    label = row.get("teacher_label")
    if not isinstance(label, Mapping):
        raise FeatureFeasibilityError("teacher_label must be an object")
    best = label.get("best_item_id")
    if not isinstance(best, str) or not best:
        raise FeatureFeasibilityError("teacher best_item_id is required")
    return best


def _episode(row: Mapping[str, Any]) -> str:
    """bootstrap cluster に使う episode identity を返す。

    decision を独立標本として resample しないため全 metric で共通利用する。
    """
    value = row.get("episode_id")
    if not isinstance(value, str) or not value:
        raise FeatureFeasibilityError("episode_id is required")
    return value


def _feature_key(features: ItemDecisionFeatures) -> str:
    """decision ID を除いた観測可能 feature の canonical group key を返す。

    teacher label/score を payload に含めず、同一観測 context の conditional label を集計する。
    """
    return canonical_hash(
        {
            "feature_schema": features.feature_schema,
            "context_features": features.context_features,
            "candidates": [candidate.features for candidate in features.candidates],
        }
    )


def _flatten_dimension(value: Any) -> int:
    """nested feature mapping の scalar 数を model 入力次元の proxy として数える。

    categorical string も一つの契約 field と数え、label/metadata object は呼び出し側で渡さない。
    """
    if isinstance(value, Mapping):
        return sum(_flatten_dimension(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_flatten_dimension(child) for child in value)
    return 1


def _conditional_predictions(
    records: Sequence[tuple[str, str]],
) -> dict[str, tuple[str, ...]]:
    """feature group ごとの empirical label 頻度順位を返す。

    count tie は item ID 辞書順で固定し Bayes top-1/NDCG ceiling を決定的にする。
    """
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for key, label in records:
        counts[key][label] += 1
    return {
        key: tuple(
            label
            for label, _ in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
        )
        for key, counter in counts.items()
    }


def _metrics(records: Sequence[tuple[str, str]]) -> tuple[float, float, float]:
    """conditional majority の top-1、single-relevant NDCG、label entropy を返す。

    entropy は自然対数、NDCG は正解 label の empirical rank から計算する。
    """
    if not records:
        raise FeatureFeasibilityError("ceiling records are empty")
    predictions = _conditional_predictions(records)
    top1 = 0.0
    ndcg = 0.0
    labels = Counter(label for _, label in records)
    for key, label in records:
        ranking = predictions[key]
        top1 += float(ranking[0] == label)
        rank = ranking.index(label)
        ndcg += 1.0 / math.log2(rank + 2.0)
    total = len(records)
    entropy = -sum(
        (count / total) * math.log(count / total)
        for count in labels.values()
        if count
    )
    return top1 / total, ndcg / total, entropy


def _bootstrap_intervals(
    rows: Sequence[Mapping[str, Any]],
    record_by_decision: Mapping[str, tuple[str, str]],
    *,
    seed: int,
    resamples: int,
) -> dict[str, list[float]]:
    """episode cluster bootstrap で top-1/NDCG の 95% percentile CI を返す。

    episode 内 decisions はまとめて再標本化し、effective sample の過大評価を防ぐ。
    """
    if type(seed) is not int or seed < 0 or type(resamples) is not int or resamples < 20:
        raise FeatureFeasibilityError("bootstrap requires non-negative seed and at least 20 resamples")
    grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for row in rows:
        decision_id = row.get("decision_id")
        if decision_id not in record_by_decision:
            raise FeatureFeasibilityError("decision feature record is missing")
        grouped[_episode(row)].append(record_by_decision[decision_id])
    episodes = sorted(grouped)
    rng = np.random.default_rng(seed)
    top1 = np.empty(resamples, dtype=np.float64)
    ndcg = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sampled = rng.integers(0, len(episodes), size=len(episodes))
        records = [
            record
            for episode_index in sampled
            for record in grouped[episodes[int(episode_index)]]
        ]
        top1[index], ndcg[index], _ = _metrics(records)
    return {
        "top1": [float(value) for value in np.quantile(top1, [0.025, 0.975], method="linear")],
        "ndcg": [float(value) for value in np.quantile(ndcg, [0.025, 0.975], method="linear")],
    }


def _variant_ablation(
    rows: Sequence[Mapping[str, Any]],
    variant: str,
    *,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    """一 feature variant の Bayes ceiling と availability/dimension を集計する。

    全 row の versioned wire を共有契約 reader で再検証してから grouping する。
    """
    records: list[tuple[str, str]] = []
    record_by_decision: dict[str, tuple[str, str]] = {}
    dimensions: list[int] = []
    availabilities: list[float] = []
    for row in rows:
        variants = row.get("feature_variants")
        if not isinstance(variants, Mapping) or set(variants) != set(FEATURE_VARIANTS):
            raise FeatureFeasibilityError("both registered feature variants are required")
        try:
            features = ItemDecisionFeatures.from_wire(variants[variant])
        except (TypeError, ValueError) as exc:
            raise FeatureFeasibilityError(f"invalid {variant} feature wire: {exc}") from exc
        if features.feature_schema != variant or features.decision_id != row.get("decision_id"):
            raise FeatureFeasibilityError("feature variant/decision binding mismatch")
        record = (_feature_key(features), _label(row))
        records.append(record)
        record_by_decision[features.decision_id] = record
        dimensions.append(
            _flatten_dimension(features.context_features)
            + sum(_flatten_dimension(candidate.features) for candidate in features.candidates)
        )
        availability = float(features.ui_state_validity)
        if variant == "context_danger_v1":
            availability *= float(features.world_validity)
        availabilities.append(availability)
    top1, ndcg, entropy = _metrics(records)
    intervals = _bootstrap_intervals(
        rows,
        record_by_decision,
        seed=seed,
        resamples=resamples,
    )
    return {
        "decision_count": len(records),
        "episode_count": len({_episode(row) for row in rows}),
        "label_entropy": entropy,
        "bayes_top1_ceiling": top1,
        "bayes_top1_ci95": intervals["top1"],
        "bayes_ndcg_ceiling": ndcg,
        "bayes_ndcg_ci95": intervals["ndcg"],
        "dimension": max(dimensions),
        "real_availability": sum(availabilities) / len(availabilities),
        "latency_class": "level_up_ui" if variant == "context_only_v1" else "cached_gameplay_snapshot",
        "latency_measurement": "contract proxy only; runtime screen-capture measurement deferred to 04-*",
    }


def _minimum_viable(ablations: Mapping[str, Mapping[str, Any]]) -> str:
    """ceiling lower bound、dimension、availability、latency の順で最小 feature 案を選ぶ。

    context-only が danger の lower bound から 0.005 以内なら低次元側を優先する。
    """
    context = ablations["context_only_v1"]
    danger = ablations["context_danger_v1"]
    if float(context["bayes_top1_ci95"][0]) + 0.005 >= float(danger["bayes_top1_ci95"][0]):
        return "context_only_v1"
    return "context_danger_v1"


def _markdown(card: Mapping[str, Any]) -> str:
    """JSON data card の主要判断を短い Markdown 表現へ変換する。

    source of truth は JSON とし、Markdown は PR/人間レビュー向けの同時生成 view にする。
    """
    lines = [
        "# Survivors ItemSelector feature feasibility",
        "",
        f"- Split manifest: `{card['split_manifest_id']}`",
        f"- Minimum viable feature: `{card['minimum_viable_feature']}`",
        f"- Offline top-1 floor: `{card['offline_top1_floor']['value']:.6f}`",
        f"- Hard cap: `{card['hard_cap']['value']}` decisions",
        f"- Test reads during ablation: `{card['partition_audit']['test_read_count']}`",
        "",
        "## Ablation",
        "",
        "| Feature | Bayes top-1 | 95% CI | NDCG | Dimension | Availability |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant in FEATURE_VARIANTS:
        item = card["ablations"][variant]
        lines.append(
            f"| {variant} | {item['bayes_top1_ceiling']:.6f} | "
            f"[{item['bayes_top1_ci95'][0]:.6f}, {item['bayes_top1_ci95'][1]:.6f}] | "
            f"{item['bayes_ndcg_ceiling']:.6f} | {item['dimension']} | "
            f"{item['real_availability']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Collection rule",
            "",
            "Coverage 完了後、validation NDCG gain < 0.005 が 2 block 継続した時点で停止する。",
            "hard cap は throughput/time と storage budget の小さい方から算出する。",
            "",
        ]
    )
    return "\n".join(lines)


def run_feature_feasibility(
    rows: Sequence[Mapping[str, Any]],
    *,
    split_manifest: SplitManifest,
    business_top1_floor: float,
    ceiling_retention: float,
    bootstrap_resamples: int,
    throughput_decisions_per_hour: float,
    collection_hours: float,
    storage_budget_bytes: int,
    estimated_row_bytes: int,
    bootstrap_seed: int = 20260731,
    output_json: Path | None = None,
    output_markdown: Path | None = None,
) -> dict[str, Any]:
    """development-only ablation を実行し JSON/Markdown data card を任意出力する。

    offline floor は選定案 ceiling lower bound の retention と business floor の最大値として
    pre-register し、固定 0.85 を使用しない。
    """
    for value, label in (
        (business_top1_floor, "business_top1_floor"),
        (ceiling_retention, "ceiling_retention"),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 < float(value) <= 1.0
        ):
            raise FeatureFeasibilityError(f"{label} must be in (0, 1]")
    try:
        development = split_manifest.read_development_rows(
            rows, purpose="feature_ablation"
        )
    except SplitSealedError as exc:
        raise FeatureFeasibilityError(f"split access rejected: {exc}") from exc
    ablations = {
        variant: _variant_ablation(
            development,
            variant,
            seed=bootstrap_seed + index * 1000,
            resamples=bootstrap_resamples,
        )
        for index, variant in enumerate(FEATURE_VARIANTS)
    }
    selected = _minimum_viable(ablations)
    lower_bound = float(ablations[selected]["bayes_top1_ci95"][0])
    floor = max(float(business_top1_floor), lower_bound * float(ceiling_retention))
    learning_curve = LearningCurvePolicy().to_wire()
    try:
        hard_cap = calculate_hard_cap(
            throughput_decisions_per_hour=throughput_decisions_per_hour,
            collection_hours=collection_hours,
            storage_budget_bytes=storage_budget_bytes,
            estimated_row_bytes=estimated_row_bytes,
        )
    except ValueError as exc:
        raise FeatureFeasibilityError(str(exc)) from exc
    audit = split_manifest.access_audit
    card: dict[str, Any] = {
        "schema_version": DATA_CARD_SCHEMA_VERSION,
        "split_manifest_id": split_manifest.manifest_id,
        "partitions_used": ["train", "validation"],
        "ablations": ablations,
        "minimum_viable_feature": selected,
        "minimum_viable_rationale": (
            "Bayes top-1 lower bound within 0.005, then lower dimension/latency; "
            "otherwise higher lower bound"
        ),
        "offline_top1_floor": {
            "value": floor,
            "business_floor": float(business_top1_floor),
            "ceiling_lower_bound": lower_bound,
            "ceiling_retention": float(ceiling_retention),
            "formula": "max(business_floor, ceiling_lower_bound * ceiling_retention)",
            "pre_registered_before_test_open": True,
        },
        "learning_curve": learning_curve,
        "hard_cap": hard_cap,
        "partition_audit": {
            "events": [dict(item) for item in audit],
            "test_read_count": sum(
                int(item["row_count"]) for item in audit if item["split"] == "test"
            ),
            "test_partition_opened": any(item["split"] == "test" for item in audit),
        },
        "runtime_latency_note": "本家 screen capture での実測は 04-* フェーズ対象",
    }
    if output_json is not None:
        destination = Path(output_json)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(canonical_json_bytes(card) + b"\n")
    if output_markdown is not None:
        destination = Path(output_markdown)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(_markdown(card), encoding="utf-8")
    return card
