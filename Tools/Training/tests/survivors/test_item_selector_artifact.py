"""ItemSelector runtime artifact の binding、shape、fail-closed 契約を検証する。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch as th

from reinbalance_survivors_contracts.canonical_json import (
    canonical_hash,
    canonical_json_bytes,
)

from export_survivors_item_selector import export_item_selector_artifact
from games.survivors.item_selector_artifact import (
    ArtifactBindingError,
    ItemSelectorArtifact,
)
from games.survivors.item_selector_dataset import SplitManifest
from games.survivors.item_selector_model import ItemSelector
from games.survivors.item_selector_trainer import ItemSelectorTrainer

CAPABILITY = "a" * 64
SOURCE = "b" * 64
TEACHER = "c" * 64


def _row(decision: str, split: str, episode: str, target: list[float]) -> dict:
    return {
        "decision_id": decision,
        "episode_id": episode,
        "split": split,
        "label_action_kind": "choose_card",
        "context_features": [0.2, 0.4, 0.6, 0.8],
        "candidate_features": [[1.0, 0.0, 0.2], [0.0, 1.0, 0.4]],
        "candidate_mask": [True, True],
        "candidate_item_ids": ["wand", "knife"],
        "candidate_kinds": ["weapon", "weapon"],
        "teacher_soft_target": target,
        "teacher_scores": target,
        "kind": "weapon",
        "time_bucket": "0-5m",
        "slot_occupancy_bucket": "open",
    }


def export_fixture(tmp_path: Path) -> Path:
    """4/3/2 の小さい checkpoint と sealed dataset からpackageを生成する。"""
    raw = [
        {"decision_id": f"d-{index}", "episode_id": f"e-{index}"}
        for index in range(6)
    ]
    split_manifest = SplitManifest.freeze(raw, seed="artifact-test")
    rows = [
        _row(
            item["decision_id"],
            split_manifest.split_for(item),
            item["episode_id"],
            [0.9, 0.1] if index % 2 == 0 else [0.1, 0.9],
        )
        for index, item in enumerate(raw)
    ]
    dataset = tmp_path / "dataset"
    (dataset / "shards").mkdir(parents=True)
    split_manifest.commit(dataset / "split_manifest.json")
    development = [row for row in rows if row["split"] != "test"]
    test = [row for row in rows if row["split"] == "test"]
    (dataset / "shards" / "development.jsonl").write_bytes(
        b"".join(canonical_json_bytes(row) + b"\n" for row in development)
    )
    (dataset / "shards" / "test.jsonl").write_bytes(
        b"".join(canonical_json_bytes(row) + b"\n" for row in test)
    )
    dataset_identity = canonical_hash({"rows": rows, "split": split_manifest.manifest_id})
    (dataset / "manifest.json").write_bytes(
        canonical_json_bytes(
            {
                "schema_version": "survivors.item_selector_dataset.v1",
                "dataset_identity": dataset_identity,
                "split_manifest": "split_manifest.json",
                "shards": [{"logical_id": "shards/development.jsonl"}],
                "test_shards": [{"logical_id": "shards/test.jsonl"}],
            }
        )
    )

    checkpoint = tmp_path / "checkpoint.pt"
    model = ItemSelector(context_dim=4, candidate_dim=3)
    ItemSelectorTrainer(model, target_capability_hash=CAPABILITY, nmax=2).save_checkpoint(
        checkpoint,
        optimizer=None,
        epoch=1,
        best_val_ndcg=0.5,
    )
    vocabulary = tmp_path / "vocabulary.json"
    vocabulary.write_bytes(canonical_json_bytes(["knife", "wand"]))
    package = tmp_path / "package"
    export_item_selector_artifact(
        checkpoint=checkpoint,
        dataset_dir=dataset,
        output_dir=package,
        target_capability_hash=CAPABILITY,
        nmax=2,
        context_dim=4,
        candidate_dim=3,
        feature_schema="context_only_v1",
        vocabulary_json=vocabulary,
        temperature=1.0,
        confidence_threshold=0.5,
        student_output_temperature=1.0,
        dataset_identity=dataset_identity,
        source_descriptor_identity=SOURCE,
        teacher_verdict_identity=TEACHER,
        run_id="unit-test",
        device="cpu",
    )
    return package


def _mutate_manifest(package: Path, mutator) -> None:
    path = package / "manifest.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    mutator(value)
    value["artifact_identity"] = canonical_hash(
        {key: item for key, item in value.items() if key != "artifact_identity"}
    )
    path.write_bytes(canonical_json_bytes(value))


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda value: value.__setitem__("feature_schema", "wrong_v9"), "feature schema"),
        (lambda value: value.__setitem__("dataset_identity", "d" * 64), "dataset identity"),
    ],
)
def test_load_rejects_wrong_feature_or_dataset_binding(tmp_path, mutation, match) -> None:
    package = export_fixture(tmp_path)
    _mutate_manifest(package, mutation)
    with pytest.raises(ArtifactBindingError, match=match):
        ItemSelectorArtifact.load(package)


def test_load_rejects_wrong_vocabulary_hash(tmp_path: Path) -> None:
    package = export_fixture(tmp_path)
    _mutate_manifest(package, lambda value: value["item_vocabulary"].append("unknown"))
    with pytest.raises(ArtifactBindingError, match="vocabulary"):
        ItemSelectorArtifact.load(package)


def test_load_rejects_wrong_model_state_hash(tmp_path: Path) -> None:
    package = export_fixture(tmp_path)
    with (package / "model.pt").open("ab") as file:
        file.write(b"corruption")
    with pytest.raises(ArtifactBindingError, match="model"):
        ItemSelectorArtifact.load(package)


@pytest.mark.parametrize("batch", [1, 2, 4])
def test_onnx_shape_dynamic_batch(tmp_path: Path, batch: int) -> None:
    package = export_fixture(tmp_path)
    session = ort.InferenceSession(str(package / "model.onnx"), providers=["CPUExecutionProvider"])
    outputs = session.run(
        ["logits"],
        {
            "context_features": np.zeros((batch, 4), dtype=np.float32),
            "candidate_features": np.zeros((batch, 2, 3), dtype=np.float32),
            "candidate_mask": np.ones((batch, 2), dtype=np.bool_),
        },
    )
    assert outputs[0].shape == (batch, 2)


def test_predict_rejects_all_masked_and_candidate_overflow(tmp_path: Path) -> None:
    artifact = ItemSelectorArtifact.load(export_fixture(tmp_path))
    context = th.zeros((1, 4))
    with pytest.raises(ValueError, match="all-masked"):
        artifact.predict(context, th.zeros((1, 2, 3)), th.zeros((1, 2), dtype=th.bool))
    with pytest.raises(ValueError, match="Nmax"):
        artifact.predict(context, th.zeros((1, 3, 3)), th.ones((1, 3), dtype=th.bool))


def test_load_rejects_wrong_or_missing_policy_hash(tmp_path: Path) -> None:
    package = export_fixture(tmp_path)
    _mutate_manifest(package, lambda value: value.__setitem__("policy_config_hash", "e" * 64))
    with pytest.raises(ArtifactBindingError, match="policy config"):
        ItemSelectorArtifact.load(package)

    package = export_fixture(tmp_path / "second")
    _mutate_manifest(package, lambda value: value.pop("policy_schema_hash"))
    with pytest.raises(ArtifactBindingError, match="manifest fields"):
        ItemSelectorArtifact.load(package)
