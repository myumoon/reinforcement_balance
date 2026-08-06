"""Survivors ItemSelector checkpointをTorchScript + ONNX packageへexportする。"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch as th

from reinbalance_survivors_contracts.canonical_json import (
    canonical_hash,
    canonical_json_bytes,
    sha256_hex,
)
from reinbalance_survivors_contracts.ui_policy import NonModelUiPolicyConfigV1

from evaluate_survivors_item_selector import (
    candidate_permutations,
    encode_runtime_row,
    fit_student_output_temperature,
    load_development_rows,
    load_sealed_test_rows,
)
from games.survivors.item_selector_artifact import (
    ARTIFACT_SCHEMA_VERSION,
    ItemSelectorArtifact,
    artifact_binding_payload,
    installed_policy_impl_hash,
    policy_schema_hash,
)
from games.survivors.item_selector_model import ItemSelector
from games.survivors.item_selector_trainer import (
    CHECKPOINT_SCHEMA_VERSION,
    ItemSelectorTrainer,
)

MAX_PARITY_ERROR = 1e-5


class ItemSelectorExportError(ValueError):
    """checkpoint/dataset/gate/export binding違反。"""


class _TemperatureScaledSelector(th.nn.Module):
    """ONNX出力へvalidation固定temperatureを適用する薄いwrapper。"""

    def __init__(self, model: ItemSelector, temperature: float) -> None:
        super().__init__()
        self.model = model
        self.temperature = float(temperature)

    def forward(
        self,
        context_features: th.Tensor,
        candidate_features: th.Tensor,
        candidate_mask: th.Tensor,
    ) -> th.Tensor:
        return self.model(context_features, candidate_features, candidate_mask) / self.temperature


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_export_arguments(
    *,
    target_capability_hash: str,
    nmax: int,
    context_dim: int,
    candidate_dim: int,
    feature_schema: str,
    temperature: float,
    confidence_threshold: float,
    student_output_temperature: float,
    dataset_identity: str,
    source_descriptor_identity: str,
    teacher_verdict_identity: str,
    run_id: str,
    device: str,
) -> None:
    for name, value in (
        ("target capability", target_capability_hash),
        ("dataset identity", dataset_identity),
        ("source descriptor identity", source_descriptor_identity),
        ("teacher verdict identity", teacher_verdict_identity),
    ):
        if not _is_sha256(value):
            raise ItemSelectorExportError(f"{name} must be lowercase SHA-256")
    for name, value in (
        ("nmax", nmax),
        ("context_dim", context_dim),
        ("candidate_dim", candidate_dim),
    ):
        if type(value) is not int or value <= 0:
            raise ItemSelectorExportError(f"{name} must be a positive integer")
    if not isinstance(feature_schema, str) or not feature_schema:
        raise ItemSelectorExportError("feature schema must be non-empty")
    for name, value in (
        ("temperature", temperature),
        ("student output temperature", student_output_temperature),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ItemSelectorExportError(f"{name} must be positive and finite")
    if not math.isfinite(confidence_threshold) or not 0.0 <= confidence_threshold <= 1.0:
        raise ItemSelectorExportError("confidence threshold must be in [0, 1]")
    if not isinstance(run_id, str) or not run_id:
        raise ItemSelectorExportError("run ID must be non-empty")
    if device != "cpu":
        raise ItemSelectorExportError("reproducible artifact export currently requires device='cpu'")


def _load_vocabulary(path: Path) -> list[str]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ItemSelectorExportError(f"cannot load vocabulary: {exc}") from exc
    if isinstance(value, Mapping):
        value = value.get("item_vocabulary", value.get("items"))
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
        or len(set(value)) != len(value)
    ):
        raise ItemSelectorExportError("vocabulary must be a non-empty unique string array")
    return sorted(value)


def _load_checkpoint_model(
    checkpoint: Path,
    *,
    target_capability_hash: str,
    nmax: int,
    context_dim: int,
    candidate_dim: int,
    device: str,
) -> ItemSelector:
    try:
        payload = th.load(Path(checkpoint), map_location=device, weights_only=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ItemSelectorExportError(f"cannot inspect checkpoint: {exc}") from exc
    if not isinstance(payload, Mapping) or payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ItemSelectorExportError("unsupported ItemSelector checkpoint schema")
    expected = {
        "target_capability_hash": target_capability_hash,
        "nmax": nmax,
        "context_dim": context_dim,
        "candidate_dim": candidate_dim,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ItemSelectorExportError(f"checkpoint {field} mismatch")
    model = ItemSelector(context_dim=context_dim, candidate_dim=candidate_dim)
    trainer = ItemSelectorTrainer(
        model,
        target_capability_hash=target_capability_hash,
        nmax=nmax,
        device=device,
    )
    trainer.load_checkpoint(checkpoint)
    model.eval()
    return model


def _validate_gate_verdict(path: Path | None) -> None:
    if path is None:
        return
    try:
        verdict = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ItemSelectorExportError(f"cannot load gate verdict: {exc}") from exc
    if not isinstance(verdict, Mapping) or verdict.get("status") != "PASS":
        raise ItemSelectorExportError("gate verdict is not PASS; package was not generated")
    identity = verdict.get("verdict_identity")
    if not _is_sha256(identity):
        raise ItemSelectorExportError("gate verdict identity is invalid")
    payload = {key: value for key, value in verdict.items() if key != "verdict_identity"}
    if canonical_hash(payload) != identity:
        raise ItemSelectorExportError("gate verdict identity mismatch")


def _stack_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    nmax: int,
    context_dim: int,
    candidate_dim: int,
) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
    encoded = [encode_runtime_row(row, nmax=nmax) for row in rows]
    if any(item.context_features.numel() != context_dim for item in encoded):
        raise ItemSelectorExportError("dataset context dimension mismatch")
    if any(item.candidate_features.shape != (nmax, candidate_dim) for item in encoded):
        raise ItemSelectorExportError("dataset candidate dimension mismatch")
    return (
        th.stack([item.context_features for item in encoded]),
        th.stack([item.candidate_features for item in encoded]),
        th.stack([item.candidate_mask for item in encoded]),
        th.stack([item.teacher_soft_target for item in encoded]),
    )


def _validate_row_vocabulary(
    rows: Sequence[Mapping[str, Any]],
    vocabulary: Sequence[str],
    *,
    nmax: int,
) -> None:
    """全partitionのvalid candidate identityを同じpackage vocabularyへ束縛する。"""
    allowed = frozenset(vocabulary)
    for row in rows:
        encoded = encode_runtime_row(row, nmax=nmax)
        observed = {
            item_id
            for item_id, valid in zip(
                encoded.candidate_item_ids,
                encoded.candidate_mask.tolist(),
                strict=True,
            )
            if valid
        }
        unknown = observed - allowed
        if unknown:
            raise ItemSelectorExportError(
                f"dataset contains items outside package vocabulary: {sorted(unknown)}"
            )


def _fit_validation_temperature(
    model: ItemSelector,
    validation_rows: Sequence[Mapping[str, Any]],
    *,
    nmax: int,
    context_dim: int,
    candidate_dim: int,
) -> float:
    context, candidates, mask, target = _stack_rows(
        validation_rows,
        nmax=nmax,
        context_dim=context_dim,
        candidate_dim=candidate_dim,
    )
    with th.no_grad():
        logits = model(context, candidates, mask)
    return fit_student_output_temperature(
        logits,
        target,
        mask,
        initial_temperature=1.0,
    )


def _export_models(
    model: ItemSelector,
    directory: Path,
    *,
    binding: Mapping[str, Any],
    temperature: float,
    nmax: int,
    context_dim: int,
    candidate_dim: int,
) -> th.jit.ScriptModule:
    context = th.zeros((1, context_dim), dtype=th.float32)
    candidates = th.zeros((1, nmax, candidate_dim), dtype=th.float32)
    mask = th.ones((1, nmax), dtype=th.bool)
    with th.no_grad():
        scripted = th.jit.trace(model, (context, candidates, mask), strict=True)
    th.jit.save(
        scripted,
        str(directory / "model.pt"),
        _extra_files={
            "artifact_bindings.json": canonical_json_bytes(dict(binding)).decode("utf-8")
        },
    )
    wrapper = _TemperatureScaledSelector(model, temperature).eval()
    try:
        th.onnx.export(
            wrapper,
            (context, candidates, mask),
            str(directory / "model.onnx"),
            input_names=["context_features", "candidate_features", "candidate_mask"],
            output_names=["logits"],
            dynamic_axes={
                "context_features": {0: "batch"},
                "candidate_features": {0: "batch", 1: "candidates"},
                "candidate_mask": {0: "batch", 1: "candidates"},
                "logits": {0: "batch", 1: "candidates"},
            },
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise ItemSelectorExportError(f"ONNX export failed: {exc}") from exc
    return scripted


def _parity_check(
    scripted: th.jit.ScriptModule,
    onnx_path: Path,
    test_rows: Sequence[Mapping[str, Any]],
    *,
    temperature: float,
    nmax: int,
    context_dim: int,
    candidate_dim: int,
) -> float:
    try:
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    except Exception as exc:
        raise ItemSelectorExportError(f"cannot load exported ONNX model: {exc}") from exc
    fixtures: list[tuple[th.Tensor, th.Tensor, th.Tensor]] = []
    for row in test_rows:
        encoded = encode_runtime_row(row, nmax=nmax)
        if encoded.context_features.numel() != context_dim or encoded.candidate_features.shape != (
            nmax,
            candidate_dim,
        ):
            raise ItemSelectorExportError("test row dimensions do not match package")
        context = encoded.context_features.unsqueeze(0)
        candidates = encoded.candidate_features.unsqueeze(0)
        mask = encoded.candidate_mask.unsqueeze(0)
        for permutation_value in candidate_permutations(nmax):
            permutation = th.tensor(permutation_value, dtype=th.long)
            fixtures.append((context, candidates[:, permutation], mask[:, permutation]))
        first_valid = int(th.nonzero(mask[0], as_tuple=False)[0].item())
        sparse_mask = th.zeros_like(mask)
        sparse_mask[0, first_valid] = True
        fixtures.append((context, candidates, sparse_mask))
    max_error = 0.0
    with th.no_grad():
        for context, candidates, mask in fixtures:
            torch_logits = (scripted(context, candidates, mask) / temperature).numpy()
            onnx_logits = session.run(
                ["logits"],
                {
                    "context_features": context.numpy(),
                    "candidate_features": candidates.numpy(),
                    "candidate_mask": mask.numpy(),
                },
            )[0]
            valid = mask.numpy()
            if not np.all(np.isneginf(torch_logits[~valid]) == np.isneginf(onnx_logits[~valid])):
                raise ItemSelectorExportError("ONNX/TorchScript masked-logit parity failed")
            if np.any(valid):
                max_error = max(
                    max_error,
                    float(np.max(np.abs(torch_logits[valid] - onnx_logits[valid]))),
                )
    if max_error > MAX_PARITY_ERROR:
        raise ItemSelectorExportError(
            f"ONNX/TorchScript max absolute logit error {max_error} exceeds {MAX_PARITY_ERROR}"
        )
    return max_error


def _normalized_version(value: str) -> str:
    return value.split("+", maxsplit=1)[0]


def export_item_selector_artifact(
    *,
    checkpoint: Path,
    dataset_dir: Path,
    output_dir: Path,
    target_capability_hash: str,
    nmax: int,
    context_dim: int,
    candidate_dim: int,
    feature_schema: str,
    vocabulary_json: Path,
    temperature: float,
    confidence_threshold: float,
    student_output_temperature: float,
    dataset_identity: str,
    source_descriptor_identity: str,
    teacher_verdict_identity: str,
    run_id: str,
    device: str = "cpu",
    gate_verdict: Path | None = None,
) -> dict[str, Any]:
    """checkpointとsealed datasetから検証済みpackageをdirectory単位で公開する。"""
    _validate_export_arguments(
        target_capability_hash=target_capability_hash,
        nmax=nmax,
        context_dim=context_dim,
        candidate_dim=candidate_dim,
        feature_schema=feature_schema,
        temperature=temperature,
        confidence_threshold=confidence_threshold,
        student_output_temperature=student_output_temperature,
        dataset_identity=dataset_identity,
        source_descriptor_identity=source_descriptor_identity,
        teacher_verdict_identity=teacher_verdict_identity,
        run_id=run_id,
        device=device,
    )
    _validate_gate_verdict(gate_verdict)
    destination = Path(output_dir)
    if destination.exists():
        raise ItemSelectorExportError("package output directory already exists")
    model = _load_checkpoint_model(
        checkpoint,
        target_capability_hash=target_capability_hash,
        nmax=nmax,
        context_dim=context_dim,
        candidate_dim=candidate_dim,
        device=device,
    )
    vocabulary = _load_vocabulary(vocabulary_json)
    dataset_manifest, train_rows, validation_rows = load_development_rows(dataset_dir)
    if dataset_manifest["dataset_identity"] != dataset_identity:
        raise ItemSelectorExportError("dataset identity argument does not match manifest")
    _validate_row_vocabulary((*train_rows, *validation_rows), vocabulary, nmax=nmax)
    fitted_temperature = _fit_validation_temperature(
        model,
        validation_rows,
        nmax=nmax,
        context_dim=context_dim,
        candidate_dim=candidate_dim,
    )

    manifest: dict[str, Any] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_identity": "0" * 64,
        "target_capability_hash": target_capability_hash,
        "nmax": nmax,
        "context_dim": context_dim,
        "candidate_dim": candidate_dim,
        "feature_schema": feature_schema,
        "vocabulary_hash": canonical_hash(vocabulary),
        "item_vocabulary": vocabulary,
        "temperature": float(temperature),
        "confidence_threshold": float(confidence_threshold),
        "student_output_temperature": fitted_temperature,
        "dataset_identity": dataset_identity,
        "model_state_hash": "0" * 64,
        "onnx_model_hash": "0" * 64,
        "files": {name: "0" * 64 for name in ("model.pt", "model.onnx", "ui_policy_config.json")},
        "policy_schema_hash": "0" * 64,
        "policy_config_hash": "0" * 64,
        "policy_impl_hash": "0" * 64,
        "lineage": {
            "source_descriptor_identity": source_descriptor_identity,
            "teacher_verdict_identity": teacher_verdict_identity,
            "trace_dataset_identity": dataset_identity,
            "model_training_run_id": run_id,
        },
        "dependency_versions": {
            "torch": _normalized_version(th.__version__),
            "onnx": _normalized_version(onnx.__version__),
            "onnxruntime": _normalized_version(ort.__version__),
            "numpy": _normalized_version(np.__version__),
        },
        "onnx_input_tensors": [
            {"name": "context_features", "shape": ["batch", context_dim], "dtype": "float32"},
            {
                "name": "candidate_features",
                "shape": ["batch", nmax, candidate_dim],
                "dtype": "float32",
            },
            {"name": "candidate_mask", "shape": ["batch", nmax], "dtype": "bool"},
        ],
        "onnx_output_tensors": [
            {"name": "logits", "shape": ["batch", nmax], "dtype": "float32"}
        ],
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        config = NonModelUiPolicyConfigV1.load_default()
        config_bytes = canonical_json_bytes(config.to_wire())
        (temporary / "ui_policy_config.json").write_bytes(config_bytes)
        manifest["policy_schema_hash"] = policy_schema_hash(config)
        manifest["policy_config_hash"] = config.config_hash
        manifest["policy_impl_hash"] = installed_policy_impl_hash()
        scripted = _export_models(
            model,
            temporary,
            binding=artifact_binding_payload(manifest),
            temperature=fitted_temperature,
            nmax=nmax,
            context_dim=context_dim,
            candidate_dim=candidate_dim,
        )

        # test partition is opened only after model/calibration are fixed, solely for final parity.
        test_rows = load_sealed_test_rows(dataset_dir, dataset_manifest)
        _validate_row_vocabulary(test_rows, vocabulary, nmax=nmax)
        _parity_check(
            scripted,
            temporary / "model.onnx",
            test_rows,
            temperature=fitted_temperature,
            nmax=nmax,
            context_dim=context_dim,
            candidate_dim=candidate_dim,
        )
        for name in manifest["files"]:
            manifest["files"][name] = sha256_hex((temporary / name).read_bytes())
        manifest["model_state_hash"] = manifest["files"]["model.pt"]
        manifest["onnx_model_hash"] = manifest["files"]["model.onnx"]
        manifest["artifact_identity"] = canonical_hash(
            {key: value for key, value in manifest.items() if key != "artifact_identity"}
        )
        (temporary / "manifest.json").write_bytes(canonical_json_bytes(manifest))
        ItemSelectorArtifact.load(temporary)
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Package a Survivors ItemSelector artifact.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-capability-hash", required=True)
    parser.add_argument("--nmax", type=int, required=True)
    parser.add_argument("--context-dim", type=int, required=True)
    parser.add_argument("--candidate-dim", type=int, required=True)
    parser.add_argument("--feature-schema", required=True)
    parser.add_argument("--vocabulary-json", type=Path, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--confidence-threshold", type=float, required=True)
    parser.add_argument("--student-output-temperature", type=float, required=True)
    parser.add_argument("--dataset-identity", required=True)
    parser.add_argument("--source-descriptor-identity", required=True)
    parser.add_argument("--teacher-verdict-identity", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--gate-verdict", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = export_item_selector_artifact(
            checkpoint=args.checkpoint,
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            target_capability_hash=args.target_capability_hash,
            nmax=args.nmax,
            context_dim=args.context_dim,
            candidate_dim=args.candidate_dim,
            feature_schema=args.feature_schema,
            vocabulary_json=args.vocabulary_json,
            temperature=args.temperature,
            confidence_threshold=args.confidence_threshold,
            student_output_temperature=args.student_output_temperature,
            dataset_identity=args.dataset_identity,
            source_descriptor_identity=args.source_descriptor_identity,
            teacher_verdict_identity=args.teacher_verdict_identity,
            run_id=args.run_id,
            device=args.device,
            gate_verdict=args.gate_verdict,
        )
    except (ItemSelectorExportError, ValueError, RuntimeError, OSError) as exc:
        print(f"item selector export failed: {exc}", file=sys.stderr)
        return 2
    sys.stdout.write(json.dumps(manifest) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
