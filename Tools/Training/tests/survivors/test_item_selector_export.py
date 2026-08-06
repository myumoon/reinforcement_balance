"""ItemSelector package export の破損拒否とCPU ONNX smokeを検証する。"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

from reinbalance_survivors_contracts.canonical_json import canonical_hash

from evaluate_survivors_item_selector import evaluate_item_selector
from games.survivors.item_selector_artifact import ArtifactBindingError, ItemSelectorArtifact
from test_item_selector_artifact import _mutate_manifest, export_fixture


def test_corrupted_manifest(tmp_path: Path) -> None:
    package = export_fixture(tmp_path)
    (package / "manifest.json").write_text("{broken", encoding="utf-8")
    with pytest.raises(ArtifactBindingError, match="manifest"):
        ItemSelectorArtifact.load(package)


def test_cpu_only_load_smoke(tmp_path: Path) -> None:
    artifact = ItemSelectorArtifact.load(export_fixture(tmp_path))
    assert artifact.nmax == 2
    assert artifact.vocabulary == frozenset({"knife", "wand"})


def test_onnx_runtime_benchmark(tmp_path: Path) -> None:
    package = export_fixture(tmp_path)
    session = ort.InferenceSession(str(package / "model.onnx"), providers=["CPUExecutionProvider"])
    inputs = {
        "context_features": np.zeros((8, 4), dtype=np.float32),
        "candidate_features": np.zeros((8, 2, 3), dtype=np.float32),
        "candidate_mask": np.ones((8, 2), dtype=np.bool_),
    }
    session.run(["logits"], inputs)
    started = time.perf_counter()
    session.run(["logits"], inputs)
    assert time.perf_counter() - started < 1.0


@pytest.mark.parametrize("field", ["policy_config_hash", "policy_impl_hash"])
def test_policy_hash_substitution_rejected(tmp_path: Path, field: str) -> None:
    package = export_fixture(tmp_path)
    _mutate_manifest(package, lambda value: value.__setitem__(field, "f" * 64))
    with pytest.raises(ArtifactBindingError, match="policy"):
        ItemSelectorArtifact.load(package)


def test_policy_impl_hash_missing(tmp_path: Path) -> None:
    package = export_fixture(tmp_path)
    _mutate_manifest(package, lambda value: value.pop("policy_impl_hash"))
    with pytest.raises(ArtifactBindingError, match="manifest fields"):
        ItemSelectorArtifact.load(package)


def test_evaluator_writes_metrics_gates_and_fixed_bootstrap(tmp_path: Path) -> None:
    package = export_fixture(tmp_path)
    output = tmp_path / "evaluation"
    verdict = evaluate_item_selector(
        package_dir=package,
        dataset_dir=tmp_path / "dataset",
        output_dir=output,
        floor_top1=0.0,
        device="cpu",
    )
    persisted = json.loads((output / "item_selector_verdict.json").read_text(encoding="utf-8"))
    assert persisted == verdict
    identity_payload = {
        key: value for key, value in verdict.items() if key != "verdict_identity"
    }
    assert verdict["verdict_identity"] == canonical_hash(identity_payload)
    assert verdict["bootstrap"] == {"seed": 42, "resamples": 2000}
    assert set(verdict["metrics"]["per_slice"]) == {
        "kind",
        "time_bucket",
        "slot_occupancy_bucket",
    }
    assert set(verdict["gates"]["checks"]) == {
        "top1_teacher_agreement",
        "mean_ndcg3",
        "median_normalized_regret",
        "each_kind_top1",
        "permutation_violations_zero",
        "onnx_pytorch_parity",
        "confidence_coverage",
    }
    assert verdict["metrics"]["overall"]["onnx_pytorch_max_abs_logit_error"] <= 1e-5
