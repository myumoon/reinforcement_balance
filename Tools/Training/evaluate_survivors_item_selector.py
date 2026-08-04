"""Sealed test splitでSurvivors ItemSelector packageを評価するCLI。"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import math
import os
import sys
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch as th

from reinbalance_survivors_contracts.canonical_json import (
    canonical_hash,
    canonical_json_bytes,
)

from games.survivors.item_selector_artifact import (
    ArtifactBindingError,
    ItemSelectorArtifact,
)
from games.survivors.item_selector_dataset import SplitManifest
from games.survivors.item_selector_trainer import (
    EncodedItemSelectorRow,
    encode_item_selector_row,
)

VERDICT_SCHEMA_VERSION = "survivors.item_selector_verdict.v1"
BOOTSTRAP_SEED = 42
BOOTSTRAP_RESAMPLES = 2000


class ItemSelectorEvaluationError(ValueError):
    """dataset access、temperature binding、metric入力の違反。"""


def candidate_permutations(candidate_count: int) -> tuple[tuple[int, ...], ...]:
    """小さいUI候補集合は全順列、大きい集合は決定的な生成集合を返す。"""
    if type(candidate_count) is not int or candidate_count <= 0:
        raise ItemSelectorEvaluationError("candidate count must be a positive integer")
    identity = tuple(range(candidate_count))
    if candidate_count <= 6:
        return tuple(itertools.permutations(identity))
    generated = {identity, tuple(reversed(identity))}
    for offset in range(1, candidate_count):
        generated.add(identity[offset:] + identity[:offset])
    for index in range(candidate_count - 1):
        swapped = list(identity)
        swapped[index], swapped[index + 1] = swapped[index + 1], swapped[index]
        generated.add(tuple(swapped))
    return tuple(sorted(generated))


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ItemSelectorEvaluationError(f"cannot load {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ItemSelectorEvaluationError(f"{label} must be a JSON object")
    return value


def load_dataset_manifest(dataset_dir: Path) -> dict[str, Any]:
    """dataset identityとshard metadataを読み込む。"""
    manifest = _read_json_object(Path(dataset_dir) / "manifest.json", label="dataset manifest")
    identity = manifest.get("dataset_identity")
    if (
        not isinstance(identity, str)
        or len(identity) != 64
        or any(character not in "0123456789abcdef" for character in identity)
    ):
        raise ItemSelectorEvaluationError("dataset identity must be lowercase SHA-256")
    return manifest


def _safe_shard_paths(
    dataset_dir: Path,
    manifest: Mapping[str, Any],
    field: str,
) -> list[Path]:
    entries = manifest.get(field)
    if not isinstance(entries, list) or not entries:
        raise ItemSelectorEvaluationError(f"dataset manifest has no {field}")
    root = Path(dataset_dir).resolve()
    result: list[Path] = []
    for entry in entries:
        logical_id = entry.get("logical_id") if isinstance(entry, Mapping) else None
        if not isinstance(logical_id, str) or not logical_id:
            raise ItemSelectorEvaluationError(f"{field} entry has no logical_id")
        path = (root / logical_id).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ItemSelectorEvaluationError(f"shard escapes dataset directory: {logical_id}") from exc
        result.append(path)
    return result


def _read_jsonl(paths: Sequence[Path], *, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError) as exc:
            raise ItemSelectorEvaluationError(f"cannot read {label} shard: {exc}") from exc
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ItemSelectorEvaluationError(f"{path}:{index + 1}: {exc}") from exc
            if not isinstance(value, dict):
                raise ItemSelectorEvaluationError(f"{path}:{index + 1} must be an object")
            rows.append(value)
    if not rows:
        raise ItemSelectorEvaluationError(f"{label} rows are empty")
    return rows


def _load_split_manifest(dataset_dir: Path, dataset_manifest: Mapping[str, Any]) -> SplitManifest:
    logical_id = dataset_manifest.get("split_manifest", "split_manifest.json")
    if not isinstance(logical_id, str) or not logical_id:
        raise ItemSelectorEvaluationError("split_manifest path is invalid")
    root = Path(dataset_dir).resolve()
    path = (root / logical_id).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ItemSelectorEvaluationError("split manifest escapes dataset directory") from exc
    return SplitManifest.load(path)


def load_development_rows(
    dataset_dir: Path,
) -> tuple[dict[str, Any], tuple[Mapping[str, Any], ...], tuple[Mapping[str, Any], ...]]:
    """development shardだけをopenし、manifest binding済みtrain/validationを返す。"""
    dataset_manifest = load_dataset_manifest(dataset_dir)
    split_manifest = _load_split_manifest(dataset_dir, dataset_manifest)
    rows = _read_jsonl(
        _safe_shard_paths(dataset_dir, dataset_manifest, "shards"),
        label="development",
    )
    if any(row.get("split", row.get("partition")) == "test" for row in rows):
        raise ItemSelectorEvaluationError("development shard contains a test row")
    train = split_manifest.read_partition_rows(rows, "train", purpose="artifact_export")
    validation = split_manifest.read_partition_rows(
        rows, "validation", purpose="temperature_calibration"
    )
    if not train or not validation:
        raise ItemSelectorEvaluationError("train and validation rows are required")
    return dataset_manifest, train, validation


def load_sealed_test_rows(
    dataset_dir: Path,
    dataset_manifest: Mapping[str, Any] | None = None,
) -> tuple[Mapping[str, Any], ...]:
    """test_shardsを明示openし、唯一のsealed reader経由でtest rowsを返す。"""
    manifest = dataset_manifest or load_dataset_manifest(dataset_dir)
    split_manifest = _load_split_manifest(dataset_dir, manifest)
    rows = _read_jsonl(
        _safe_shard_paths(dataset_dir, manifest, "test_shards"),
        label="test",
    )
    selected = split_manifest.read_test_rows(rows, purpose="sealed_evaluation")
    if not selected:
        raise ItemSelectorEvaluationError("sealed test rows are empty")
    return selected


def _direct_encoded_row(row: Mapping[str, Any], *, nmax: int) -> EncodedItemSelectorRow:
    decision_id = row.get("decision_id")
    if not isinstance(decision_id, str) or not decision_id:
        raise ItemSelectorEvaluationError("direct row decision_id must be non-empty")
    context = row.get("context_features")
    candidates = row.get("candidate_features")
    mask = row.get("candidate_mask")
    target = row.get("teacher_soft_target")
    if isinstance(target, Mapping):
        target = target.get("probabilities")
    if (
        not isinstance(context, list)
        or not context
        or not isinstance(candidates, list)
        or not candidates
        or len(candidates) > nmax
        or not isinstance(mask, list)
        or not isinstance(target, list)
        or len(candidates) != len(mask)
        or len(mask) != len(target)
        or any(type(value) is not bool for value in mask)
    ):
        raise ItemSelectorEvaluationError("direct encoded row shape is invalid")
    candidate_dim = len(candidates[0]) if isinstance(candidates[0], list) else 0
    if candidate_dim <= 0 or any(
        not isinstance(vector, list) or len(vector) != candidate_dim for vector in candidates
    ):
        raise ItemSelectorEvaluationError("direct candidate feature dimensions are invalid")
    try:
        context_tensor = th.tensor(context, dtype=th.float32)
        candidate_tensor = th.tensor(candidates, dtype=th.float32)
        target_tensor = th.tensor(target, dtype=th.float32)
    except (TypeError, ValueError) as exc:
        raise ItemSelectorEvaluationError(f"direct encoded row has non-numeric data: {exc}") from exc
    if not bool(th.isfinite(context_tensor).all()) or not bool(th.isfinite(candidate_tensor).all()):
        raise ItemSelectorEvaluationError("direct feature values must be finite")
    if not bool(th.isfinite(target_tensor).all()) or bool((target_tensor < 0).any()):
        raise ItemSelectorEvaluationError("teacher target must be finite and non-negative")
    if not any(mask) or float(target_tensor[th.tensor(mask)].sum()) <= 0.0:
        raise ItemSelectorEvaluationError("direct row has no valid target mass")
    item_ids = row.get("candidate_item_ids")
    kinds = row.get("candidate_kinds")
    if (
        not isinstance(item_ids, list)
        or not isinstance(kinds, list)
        or len(item_ids) != len(candidates)
        or len(kinds) != len(candidates)
        or any(not isinstance(value, str) or not value for value in (*item_ids, *kinds))
    ):
        raise ItemSelectorEvaluationError("direct row candidate identities are invalid")
    valid_ids = [item_id for item_id, valid in zip(item_ids, mask, strict=True) if valid]
    if len(set(valid_ids)) != len(valid_ids):
        raise ItemSelectorEvaluationError("direct row valid candidate identities must be unique")
    reliability = row.get("reliability_weight", 1.0)
    if (
        isinstance(reliability, bool)
        or not isinstance(reliability, (int, float))
        or not math.isfinite(float(reliability))
        or not 0.0 <= float(reliability) <= 1.0
    ):
        raise ItemSelectorEvaluationError("direct row reliability must be in [0, 1]")
    padding = nmax - len(candidates)
    return EncodedItemSelectorRow(
        decision_id=decision_id,
        partition=str(row.get("split", row.get("partition", ""))),
        context_features=context_tensor,
        candidate_features=th.cat(
            (candidate_tensor, th.zeros((padding, candidate_dim), dtype=th.float32)), dim=0
        ),
        candidate_mask=th.tensor(mask + [False] * padding, dtype=th.bool),
        teacher_soft_target=th.cat((target_tensor, th.zeros(padding, dtype=th.float32))),
        reliability_weight=float(reliability),
        candidate_item_ids=tuple(item_ids + ["__padding__"] * padding),
        candidate_kinds=tuple(kinds + ["padding"] * padding),
    )


def encode_runtime_row(row: Mapping[str, Any], *, nmax: int) -> EncodedItemSelectorRow:
    """released formal wireまたは明示済みtensor fixtureをruntime inputへ変換する。"""
    if "context_features" in row and "candidate_features" in row:
        return _direct_encoded_row(row, nmax=nmax)
    partition = row.get("split", row.get("partition"))
    normalized = dict(row)
    normalized["split"] = "validation"
    encoded = encode_item_selector_row(normalized, nmax=nmax)
    return dataclasses.replace(encoded, partition=str(partition))


def _stack_encoded(
    rows: Sequence[Mapping[str, Any]],
    *,
    nmax: int,
    context_dim: int,
    candidate_dim: int,
) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
    encoded = [encode_runtime_row(row, nmax=nmax) for row in rows]
    if any(item.context_features.numel() != context_dim for item in encoded):
        raise ItemSelectorEvaluationError("dataset context dimension does not match artifact")
    if any(item.candidate_features.shape != (nmax, candidate_dim) for item in encoded):
        raise ItemSelectorEvaluationError("dataset candidate dimension does not match artifact")
    return (
        th.stack([item.context_features for item in encoded]),
        th.stack([item.candidate_features for item in encoded]),
        th.stack([item.candidate_mask for item in encoded]),
        th.stack([item.teacher_soft_target for item in encoded]),
    )


def fit_student_output_temperature(
    logits: th.Tensor,
    target: th.Tensor,
    mask: th.Tensor,
    *,
    initial_temperature: float = 1.0,
) -> float:
    """validation logitsへscalar NLL temperatureを決定的golden searchでfitする。"""
    if (
        logits.ndim != 2
        or target.shape != logits.shape
        or mask.shape != logits.shape
        or mask.dtype != th.bool
        or logits.shape[0] == 0
        or bool((~mask.any(dim=1)).any())
    ):
        raise ItemSelectorEvaluationError("temperature fit tensors are invalid")
    if not math.isfinite(initial_temperature) or initial_temperature <= 0.0:
        raise ItemSelectorEvaluationError("initial student temperature must be positive")
    normalized_target = target.masked_fill(~mask, 0.0)
    sums = normalized_target.sum(dim=1, keepdim=True)
    if bool((sums <= 0.0).any()):
        raise ItemSelectorEvaluationError("temperature fit target has no valid mass")
    normalized_target = normalized_target / sums
    finite_logits = logits.detach().to(dtype=th.float64, device="cpu")
    normalized_target = normalized_target.to(dtype=th.float64, device="cpu")
    cpu_mask = mask.to(device="cpu")

    def objective(log_temperature: float) -> float:
        temperature = math.exp(log_temperature)
        scaled = (finite_logits / temperature).masked_fill(~cpu_mask, -th.inf)
        log_probabilities = th.log_softmax(scaled, dim=1)
        safe_log_probabilities = log_probabilities.masked_fill(~cpu_mask, 0.0)
        return float(-(normalized_target * safe_log_probabilities).sum(dim=1).mean())

    center = math.log(float(initial_temperature))
    left = max(math.log(0.01), center - math.log(100.0))
    right = min(math.log(100.0), center + math.log(100.0))
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    x1 = right - ratio * (right - left)
    x2 = left + ratio * (right - left)
    f1, f2 = objective(x1), objective(x2)
    for _ in range(96):
        if f1 <= f2:
            right, x2, f2 = x2, x1, f1
            x1 = right - ratio * (right - left)
            f1 = objective(x1)
        else:
            left, x1, f1 = x1, x2, f2
            x2 = left + ratio * (right - left)
            f2 = objective(x2)
    return float(math.exp((left + right) * 0.5))


def fit_package_temperature(
    artifact: ItemSelectorArtifact,
    validation_rows: Sequence[Mapping[str, Any]],
) -> float:
    """packageのraw logitsをvalidationだけで再現しtemperatureをfitする。"""
    context, candidates, mask, target = _stack_encoded(
        validation_rows,
        nmax=artifact.nmax,
        context_dim=int(artifact.manifest["context_dim"]),
        candidate_dim=int(artifact.manifest["candidate_dim"]),
    )
    scaled = artifact.predict(context, candidates, mask)
    raw = scaled * float(artifact.manifest["student_output_temperature"])
    return fit_student_output_temperature(
        raw,
        target,
        mask,
        initial_temperature=1.0,
    )


def _ndcg3(target: np.ndarray, order: np.ndarray, valid: np.ndarray) -> float:
    valid_indices = np.flatnonzero(valid)
    predicted = [int(index) for index in order if valid[index]][:3]
    ideal = sorted(valid_indices.tolist(), key=lambda index: (-float(target[index]), index))[:3]

    def dcg(indices: Sequence[int]) -> float:
        return sum(
            (2.0 ** float(target[index]) - 1.0) / math.log2(position + 2.0)
            for position, index in enumerate(indices)
        )

    ideal_value = dcg(ideal)
    return 1.0 if ideal_value <= 0.0 else float(dcg(predicted) / ideal_value)


def _normalized_regret(row: Mapping[str, Any], predicted: int, target: np.ndarray, valid: np.ndarray) -> float:
    raw_scores = row.get("teacher_scores")
    if isinstance(raw_scores, Mapping):
        raw_scores = raw_scores.get("scores")
    if not isinstance(raw_scores, (list, tuple)) or len(raw_scores) > len(valid):
        scores = target.astype(np.float64)
    else:
        scores = np.full(len(valid), np.nan, dtype=np.float64)
        for index, value in enumerate(raw_scores):
            if value is not None and not isinstance(value, bool) and isinstance(value, (int, float)):
                scores[index] = float(value)
        if not np.all(np.isfinite(scores[valid])):
            scores = target.astype(np.float64)
    values = scores[valid]
    span = float(np.max(values) - np.min(values))
    if span <= 1e-12:
        return 0.0
    return float((np.max(values) - scores[predicted]) / span)


def _slice_values(row: Mapping[str, Any], encoded: EncodedItemSelectorRow) -> dict[str, str]:
    kind = row.get("kind", row.get("choice_kind"))
    if not isinstance(kind, str) or not kind:
        kinds = sorted(
            {value for value in encoded.candidate_kinds if value not in {"padding", "__padding__"}}
        )
        kind = kinds[0] if len(kinds) == 1 else "mixed"
    time_bucket = row.get("time_bucket")
    if not isinstance(time_bucket, str) or not time_bucket:
        elapsed = row.get("elapsed_time", row.get("elapsed_seconds"))
        if not isinstance(elapsed, (int, float)) or isinstance(elapsed, bool):
            feature_wire = row.get("item_decision_features")
            elapsed = feature_wire.get("elapsed_time") if isinstance(feature_wire, Mapping) else None
        if not isinstance(elapsed, (int, float)) or isinstance(elapsed, bool):
            time_bucket = "unknown"
        elif float(elapsed) < 300.0:
            time_bucket = "0-5m"
        elif float(elapsed) < 600.0:
            time_bucket = "5-10m"
        else:
            time_bucket = "10m+"
    occupancy = row.get("slot_occupancy_bucket")
    if not isinstance(occupancy, str) or not occupancy:
        feature_wire = row.get("item_decision_features")
        empty_slots = feature_wire.get("empty_slot_count") if isinstance(feature_wire, Mapping) else None
        occupancy = "full" if empty_slots == 0 else "open" if isinstance(empty_slots, int) else "unknown"
    return {"kind": kind, "time_bucket": time_bucket, "slot_occupancy_bucket": occupancy}


def _bootstrap_ci(row_metrics: Sequence[Mapping[str, Any]]) -> dict[str, list[float]]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    count = len(row_metrics)
    values: dict[str, np.ndarray] = {
        "top1_teacher_agreement": np.empty(BOOTSTRAP_RESAMPLES),
        "mean_ndcg3": np.empty(BOOTSTRAP_RESAMPLES),
        "median_normalized_regret": np.empty(BOOTSTRAP_RESAMPLES),
        "permutation_violations": np.empty(BOOTSTRAP_RESAMPLES),
        "confidence_coverage": np.empty(BOOTSTRAP_RESAMPLES),
        "accepted_accuracy": np.empty(BOOTSTRAP_RESAMPLES),
    }
    for resample in range(BOOTSTRAP_RESAMPLES):
        sample = [row_metrics[index] for index in rng.integers(0, count, size=count)]
        accepted = [row for row in sample if row["accepted"]]
        values["top1_teacher_agreement"][resample] = np.mean([row["top1"] for row in sample])
        values["mean_ndcg3"][resample] = np.mean([row["ndcg3"] for row in sample])
        values["median_normalized_regret"][resample] = np.median(
            [row["normalized_regret"] for row in sample]
        )
        values["permutation_violations"][resample] = np.sum(
            [row["permutation_violation"] for row in sample]
        )
        values["confidence_coverage"][resample] = len(accepted) / count
        values["accepted_accuracy"][resample] = (
            float(np.mean([row["top1"] for row in accepted])) if accepted else 0.0
        )
    return {
        key: [float(item) for item in np.quantile(value, [0.025, 0.975], method="linear")]
        for key, value in values.items()
    }


def _evaluate_rows(
    artifact: ItemSelectorArtifact,
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, list[float]]]:
    row_metrics: list[dict[str, Any]] = []
    slices: dict[str, dict[str, list[float]]] = {
        "kind": defaultdict(list),
        "time_bucket": defaultdict(list),
        "slot_occupancy_bucket": defaultdict(list),
    }
    max_parity_error = 0.0
    threshold = float(artifact.manifest["confidence_threshold"])
    for row in rows:
        encoded = encode_runtime_row(row, nmax=artifact.nmax)
        context = encoded.context_features.unsqueeze(0)
        candidates = encoded.candidate_features.unsqueeze(0)
        mask = encoded.candidate_mask.unsqueeze(0)
        logits = artifact.predict(context, candidates, mask)[0]
        valid = encoded.candidate_mask.numpy()
        target = encoded.teacher_soft_target.numpy()
        predicted = int(th.argmax(logits).item())
        best_teacher_value = float(np.max(target[valid]))
        order = np.argsort(-logits.numpy(), kind="stable")
        probabilities = th.softmax(logits, dim=0)
        confidence = float(probabilities[predicted])

        finite = encoded.candidate_mask
        permutation_violation = 0
        for permutation_value in candidate_permutations(artifact.nmax):
            permutation = th.tensor(permutation_value, dtype=th.long)
            permuted_logits = artifact.predict(
                context,
                candidates[:, permutation],
                mask[:, permutation],
            )[0]
            inverse = th.argsort(permutation)
            restored = permuted_logits[inverse]
            if not th.allclose(logits[finite], restored[finite], atol=1e-6, rtol=1e-6):
                permutation_violation = 1
                break

        onnx_logits = artifact.onnx_session.run(
            ["logits"],
            {
                "context_features": context.numpy(),
                "candidate_features": candidates.numpy(),
                "candidate_mask": mask.numpy(),
            },
        )[0][0]
        torch_values = logits.numpy()
        max_parity_error = max(
            max_parity_error,
            float(np.max(np.abs(torch_values[valid] - onnx_logits[valid]))),
        )
        metric = {
            "top1": float(
                valid[predicted]
                and math.isclose(
                    float(target[predicted]),
                    best_teacher_value,
                    rel_tol=0.0,
                    abs_tol=1e-8,
                )
            ),
            "ndcg3": _ndcg3(target, order, valid),
            "normalized_regret": _normalized_regret(row, predicted, target, valid),
            "permutation_violation": permutation_violation,
            "accepted": confidence >= threshold,
        }
        row_metrics.append(metric)
        for dimension, value in _slice_values(row, encoded).items():
            slices[dimension][value].append(metric["top1"])

    accepted = [row for row in row_metrics if row["accepted"]]
    overall = {
        "row_count": len(row_metrics),
        "top1_teacher_agreement": float(np.mean([row["top1"] for row in row_metrics])),
        "mean_ndcg3": float(np.mean([row["ndcg3"] for row in row_metrics])),
        "median_normalized_regret": float(
            np.median([row["normalized_regret"] for row in row_metrics])
        ),
        "permutation_violations": int(
            sum(row["permutation_violation"] for row in row_metrics)
        ),
        "onnx_pytorch_max_abs_logit_error": max_parity_error,
        "confidence_coverage": len(accepted) / len(row_metrics),
        "accepted_accuracy": (
            float(np.mean([row["top1"] for row in accepted])) if accepted else 0.0
        ),
    }
    per_slice = {
        dimension: {
            value: {
                "count": len(samples),
                "top1_teacher_agreement": float(np.mean(samples)),
            }
            for value, samples in sorted(groups.items())
        }
        for dimension, groups in slices.items()
    }
    bootstrap = _bootstrap_ci(row_metrics)
    bootstrap["onnx_pytorch_max_abs_logit_error"] = [max_parity_error, max_parity_error]
    return overall, per_slice, bootstrap


def evaluate_item_selector(
    *,
    package_dir: Path,
    dataset_dir: Path,
    output_dir: Path,
    floor_top1: float,
    device: str = "cpu",
) -> dict[str, Any]:
    """validation temperatureを確認後、sealed test metrics/gates/verdictをatomic保存する。"""
    if device != "cpu":
        raise ItemSelectorEvaluationError("artifact evaluation currently requires device='cpu'")
    if not math.isfinite(floor_top1) or not 0.0 <= floor_top1 <= 1.0:
        raise ItemSelectorEvaluationError("floor-top1 must be in [0, 1]")
    artifact = ItemSelectorArtifact.load(package_dir)
    dataset_manifest, _, validation = load_development_rows(dataset_dir)
    if dataset_manifest["dataset_identity"] != artifact.manifest["dataset_identity"]:
        raise ArtifactBindingError("evaluation dataset identity mismatch")
    fitted_temperature = fit_package_temperature(artifact, validation)
    package_temperature = float(artifact.manifest["student_output_temperature"])
    if not math.isclose(fitted_temperature, package_temperature, rel_tol=1e-7, abs_tol=1e-7):
        raise ArtifactBindingError("student output temperature validation binding mismatch")

    test_rows = load_sealed_test_rows(dataset_dir, dataset_manifest)
    overall, per_slice, bootstrap = _evaluate_rows(artifact, test_rows)
    kind_metrics = per_slice["kind"]
    checks = {
        "top1_teacher_agreement": overall["top1_teacher_agreement"] >= floor_top1,
        "mean_ndcg3": overall["mean_ndcg3"] >= 0.95,
        "median_normalized_regret": overall["median_normalized_regret"] <= 0.03,
        "each_kind_top1": bool(kind_metrics)
        and all(value["top1_teacher_agreement"] >= 0.75 for value in kind_metrics.values()),
        "permutation_violations_zero": overall["permutation_violations"] == 0,
        "onnx_pytorch_parity": overall["onnx_pytorch_max_abs_logit_error"] <= 1e-5,
        "confidence_coverage": overall["confidence_coverage"] >= 0.80
        and overall["accepted_accuracy"] >= 0.90,
    }
    passed = all(checks.values())
    payload: dict[str, Any] = {
        "schema_version": VERDICT_SCHEMA_VERSION,
        "status": "PASS" if passed else "FAIL",
        "subject": {
            "artifact_identity": artifact.manifest["artifact_identity"],
            "dataset_identity": dataset_manifest["dataset_identity"],
            "student_output_temperature": fitted_temperature,
        },
        "gates": {"passed": passed, "checks": checks},
        "metrics": {
            "overall": overall,
            "per_slice": per_slice,
            "bootstrap_ci_95": bootstrap,
        },
        "bootstrap": {"seed": BOOTSTRAP_SEED, "resamples": BOOTSTRAP_RESAMPLES},
    }
    payload["verdict_identity"] = canonical_hash(payload)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".item_selector_verdict.", dir=destination
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_bytes(canonical_json_bytes(payload))
        os.replace(temporary, destination / "item_selector_verdict.json")
    finally:
        temporary.unlink(missing_ok=True)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a packaged Survivors ItemSelector.")
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--floor-top1", type=float, required=True)
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        verdict = evaluate_item_selector(
            package_dir=args.package_dir,
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            floor_top1=args.floor_top1,
            device=args.device,
        )
    except (ArtifactBindingError, ItemSelectorEvaluationError, ValueError, RuntimeError) as exc:
        print(f"item selector evaluation failed: {exc}", file=sys.stderr)
        return 2
    sys.stdout.write(json.dumps(verdict) + "\n")
    return 0 if verdict["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
