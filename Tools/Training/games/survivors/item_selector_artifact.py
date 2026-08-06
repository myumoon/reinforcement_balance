"""Survivors ItemSelector の自己完結runtime artifact loader。

package manifest、TorchScript埋め込みbinding、ONNX signature、共有NonModelUiPolicyを
load時に相互照合し、別dataset/model/policyの置換を推論開始前に拒否する。
"""

from __future__ import annotations

import inspect
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import onnxruntime as ort
import torch as th

from reinbalance_survivors_contracts import ui_policy as installed_ui_policy
from reinbalance_survivors_contracts.canonical_json import (
    canonical_hash,
    canonical_json_bytes,
    sha256_hex,
)
from reinbalance_survivors_contracts.ui_policy import NonModelUiPolicyConfigV1

ARTIFACT_SCHEMA_VERSION = "survivors.item_selector_artifact.v1"
_EMBEDDED_BINDING_FILE = "artifact_bindings.json"
_PACKAGE_FILES = frozenset({"model.pt", "model.onnx", "ui_policy_config.json"})
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_identity",
        "target_capability_hash",
        "nmax",
        "context_dim",
        "candidate_dim",
        "feature_schema",
        "vocabulary_hash",
        "item_vocabulary",
        "temperature",
        "confidence_threshold",
        "student_output_temperature",
        "dataset_identity",
        "model_state_hash",
        "onnx_model_hash",
        "files",
        "policy_schema_hash",
        "policy_config_hash",
        "policy_impl_hash",
        "lineage",
        "dependency_versions",
        "onnx_input_tensors",
        "onnx_output_tensors",
    }
)
_LINEAGE_FIELDS = frozenset(
    {
        "source_descriptor_identity",
        "teacher_verdict_identity",
        "trace_dataset_identity",
        "model_training_run_id",
    }
)
_DEPENDENCY_FIELDS = frozenset({"torch", "onnx", "onnxruntime", "numpy"})


class ArtifactBindingError(ValueError):
    """artifact packageのschema/content/distribution binding違反。"""


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def policy_schema_hash(config: NonModelUiPolicyConfigV1) -> str:
    """共有wireそのものからNonModelUiPolicyConfigV1 schema bindingを返す。"""
    return canonical_hash(config.to_wire())


def installed_policy_impl_hash() -> str:
    """importされた共有ui_policy実装sourceのcontent hashを返す。"""
    source_name = inspect.getsourcefile(installed_ui_policy)
    if not source_name:
        raise ArtifactBindingError("installed policy implementation source is unavailable")
    try:
        return sha256_hex(Path(source_name).read_bytes())
    except OSError as exc:
        raise ArtifactBindingError(
            f"cannot read installed policy implementation: {exc}"
        ) from exc


def artifact_binding_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """TorchScriptへ封入するmodel/data/feature bindingの固定部分を返す。"""
    fields = (
        "target_capability_hash",
        "nmax",
        "context_dim",
        "candidate_dim",
        "feature_schema",
        "vocabulary_hash",
        "dataset_identity",
        "temperature",
        "confidence_threshold",
        "student_output_temperature",
    )
    return {field: manifest[field] for field in fields}


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactBindingError(f"cannot load {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ArtifactBindingError(f"{label} must be a JSON object")
    return value


def _validate_scalar_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ArtifactBindingError("unsupported artifact schema version")
    for field in (
        "artifact_identity",
        "target_capability_hash",
        "vocabulary_hash",
        "dataset_identity",
        "model_state_hash",
        "onnx_model_hash",
        "policy_schema_hash",
        "policy_config_hash",
        "policy_impl_hash",
    ):
        if not _is_sha256(manifest.get(field)):
            raise ArtifactBindingError(f"{field} must be lowercase SHA-256")
    for field in ("nmax", "context_dim", "candidate_dim"):
        if type(manifest.get(field)) is not int or int(manifest[field]) <= 0:
            raise ArtifactBindingError(f"{field} must be a positive integer")
    feature_schema = manifest.get("feature_schema")
    if not isinstance(feature_schema, str) or not feature_schema:
        raise ArtifactBindingError("feature schema must be non-empty")
    for field in ("temperature", "student_output_temperature"):
        value = manifest.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ArtifactBindingError(f"{field} must be positive and finite")
    threshold = manifest.get("confidence_threshold")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
        or not 0.0 <= float(threshold) <= 1.0
    ):
        raise ArtifactBindingError("confidence_threshold must be in [0, 1]")


def _validate_vocabulary(manifest: Mapping[str, Any]) -> frozenset[str]:
    vocabulary = manifest.get("item_vocabulary")
    if (
        not isinstance(vocabulary, list)
        or not vocabulary
        or any(not isinstance(item, str) or not item for item in vocabulary)
        or len(set(vocabulary)) != len(vocabulary)
        or vocabulary != sorted(vocabulary)
    ):
        raise ArtifactBindingError("item vocabulary must be a sorted unique string list")
    if canonical_hash(vocabulary) != manifest["vocabulary_hash"]:
        raise ArtifactBindingError("item vocabulary hash mismatch")
    return frozenset(vocabulary)


def _validate_tensor_manifest(manifest: Mapping[str, Any]) -> None:
    nmax = manifest["nmax"]
    context_dim = manifest["context_dim"]
    candidate_dim = manifest["candidate_dim"]
    expected_inputs = [
        {"name": "context_features", "shape": ["batch", context_dim], "dtype": "float32"},
        {
            "name": "candidate_features",
            "shape": ["batch", nmax, candidate_dim],
            "dtype": "float32",
        },
        {"name": "candidate_mask", "shape": ["batch", nmax], "dtype": "bool"},
    ]
    expected_outputs = [
        {"name": "logits", "shape": ["batch", nmax], "dtype": "float32"}
    ]
    if manifest.get("onnx_input_tensors") != expected_inputs:
        raise ArtifactBindingError("ONNX input tensor manifest mismatch")
    if manifest.get("onnx_output_tensors") != expected_outputs:
        raise ArtifactBindingError("ONNX output tensor manifest mismatch")


def _validate_metadata_objects(manifest: Mapping[str, Any]) -> None:
    files = manifest.get("files")
    if not isinstance(files, Mapping) or frozenset(files) != _PACKAGE_FILES:
        raise ArtifactBindingError("package files manifest mismatch")
    if any(not _is_sha256(value) for value in files.values()):
        raise ArtifactBindingError("package file hash must be lowercase SHA-256")
    lineage = manifest.get("lineage")
    if not isinstance(lineage, Mapping) or frozenset(lineage) != _LINEAGE_FIELDS:
        raise ArtifactBindingError("lineage fields mismatch")
    for field in _LINEAGE_FIELDS - {"model_training_run_id"}:
        if not _is_sha256(lineage.get(field)):
            raise ArtifactBindingError(f"lineage {field} must be lowercase SHA-256")
    if not isinstance(lineage.get("model_training_run_id"), str) or not lineage[
        "model_training_run_id"
    ]:
        raise ArtifactBindingError("model training run ID must be non-empty")
    dependencies = manifest.get("dependency_versions")
    if (
        not isinstance(dependencies, Mapping)
        or frozenset(dependencies) != _DEPENDENCY_FIELDS
        or any(not isinstance(value, str) or not value for value in dependencies.values())
    ):
        raise ArtifactBindingError("dependency versions manifest mismatch")


def _load_torchscript(path: Path) -> tuple[th.jit.ScriptModule, dict[str, Any]]:
    extra_files: dict[str, Any] = {_EMBEDDED_BINDING_FILE: b""}
    try:
        model = th.jit.load(str(path), map_location="cpu", _extra_files=extra_files)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ArtifactBindingError(f"cannot load TorchScript model: {exc}") from exc
    raw_binding = extra_files[_EMBEDDED_BINDING_FILE]
    try:
        encoded_binding = (
            raw_binding.encode("utf-8")
            if isinstance(raw_binding, str)
            else bytes(raw_binding)
        )
        binding = json.loads(encoded_binding.decode("utf-8"))
    except (TypeError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactBindingError("TorchScript artifact binding is missing or invalid") from exc
    if not isinstance(binding, dict):
        raise ArtifactBindingError("TorchScript artifact binding must be an object")
    model.eval()
    return model, binding


def _load_onnx_session(path: Path, manifest: Mapping[str, Any]) -> ort.InferenceSession:
    try:
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    except Exception as exc:  # onnxruntime exposes provider-specific exception subclasses
        raise ArtifactBindingError(f"cannot load ONNX model: {exc}") from exc
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if [item.name for item in inputs] != [
        "context_features",
        "candidate_features",
        "candidate_mask",
    ] or [item.name for item in outputs] != ["logits"]:
        raise ArtifactBindingError("ONNX runtime tensor names mismatch")
    if (
        inputs[0].shape != ["batch", manifest["context_dim"]]
        or inputs[1].shape
        != ["batch", "candidates", manifest["candidate_dim"]]
        or inputs[2].shape != ["batch", "candidates"]
        or outputs[0].shape != ["batch", "candidates"]
    ):
        raise ArtifactBindingError("ONNX dynamic batch/candidate shape mismatch")
    if [item.type for item in inputs] != [
        "tensor(float)",
        "tensor(float)",
        "tensor(bool)",
    ] or outputs[0].type != "tensor(float)":
        raise ArtifactBindingError("ONNX runtime tensor dtype mismatch")
    return session


class ItemSelectorArtifact:
    """package directoryからruntime inferenceに必要な全成分をloadする。"""

    def __init__(
        self,
        *,
        package_dir: Path,
        manifest: Mapping[str, Any],
        model: th.jit.ScriptModule,
        onnx_session: ort.InferenceSession,
        vocabulary: frozenset[str],
        ui_policy_config: NonModelUiPolicyConfigV1,
    ) -> None:
        self.package_dir = package_dir
        self.manifest = dict(manifest)
        self._model = model
        self._onnx_session = onnx_session
        self._vocabulary = vocabulary
        self._ui_policy_config = ui_policy_config

    @classmethod
    def load(cls, package_dir: Path) -> "ItemSelectorArtifact":
        """全manifest/file/model/policy bindingを検証後にartifactを返す。"""
        package = Path(package_dir)
        manifest = _read_json_object(package / "manifest.json", label="artifact manifest")
        if frozenset(manifest) != _MANIFEST_FIELDS:
            raise ArtifactBindingError("artifact manifest fields mismatch")
        _validate_scalar_manifest(manifest)
        _validate_metadata_objects(manifest)
        _validate_tensor_manifest(manifest)
        expected_identity = canonical_hash(
            {key: value for key, value in manifest.items() if key != "artifact_identity"}
        )
        if manifest["artifact_identity"] != expected_identity:
            raise ArtifactBindingError("artifact identity mismatch")
        vocabulary = _validate_vocabulary(manifest)

        try:
            package_entries = {path.name for path in package.iterdir()}
        except OSError as exc:
            raise ArtifactBindingError(f"cannot enumerate artifact package: {exc}") from exc
        expected_entries = set(_PACKAGE_FILES) | {"manifest.json"}
        if package_entries != expected_entries:
            raise ArtifactBindingError("artifact package contains missing or unlisted files")

        actual_hashes: dict[str, str] = {}
        for name in sorted(_PACKAGE_FILES):
            path = package / name
            if path.is_symlink() or not path.is_file():
                raise ArtifactBindingError(f"package file must be a regular file: {name}")
            try:
                actual_hashes[name] = sha256_hex(path.read_bytes())
            except OSError as exc:
                raise ArtifactBindingError(f"cannot read package file {name}: {exc}") from exc
            if actual_hashes[name] != manifest["files"][name]:
                raise ArtifactBindingError(f"package file hash mismatch: {name}")
        if actual_hashes["model.pt"] != manifest["model_state_hash"]:
            raise ArtifactBindingError("model state hash mismatch")
        if actual_hashes["model.onnx"] != manifest["onnx_model_hash"]:
            raise ArtifactBindingError("ONNX model hash mismatch")

        model, embedded_binding = _load_torchscript(package / "model.pt")
        expected_binding = artifact_binding_payload(manifest)
        for field, expected in expected_binding.items():
            if embedded_binding.get(field) != expected:
                label = field.replace("_", " ")
                raise ArtifactBindingError(f"{label} binding mismatch")
        if frozenset(embedded_binding) != frozenset(expected_binding):
            raise ArtifactBindingError("TorchScript artifact binding fields mismatch")

        config_wire = _read_json_object(
            package / "ui_policy_config.json", label="UI policy config"
        )
        try:
            config = NonModelUiPolicyConfigV1.from_wire(config_wire)
            installed = NonModelUiPolicyConfigV1.load_default()
        except ValueError as exc:
            raise ArtifactBindingError(f"UI policy config is invalid: {exc}") from exc
        if canonical_json_bytes(config_wire) != (package / "ui_policy_config.json").read_bytes():
            raise ArtifactBindingError("UI policy config must use canonical JSON bytes")
        if config.to_wire() != installed.to_wire():
            raise ArtifactBindingError("installed policy config mismatch")
        if manifest["policy_schema_hash"] != policy_schema_hash(installed):
            raise ArtifactBindingError("policy schema hash mismatch")
        if manifest["policy_config_hash"] != installed.config_hash:
            raise ArtifactBindingError("policy config hash mismatch")
        if manifest["policy_impl_hash"] != installed_policy_impl_hash():
            raise ArtifactBindingError("policy implementation hash mismatch")

        onnx_session = _load_onnx_session(package / "model.onnx", manifest)
        return cls(
            package_dir=package,
            manifest=manifest,
            model=model,
            onnx_session=onnx_session,
            vocabulary=vocabulary,
            ui_policy_config=config,
        )

    def predict(
        self,
        context_features: th.Tensor,
        candidate_features: th.Tensor,
        candidate_mask: th.Tensor,
    ) -> th.Tensor:
        """shape/taxonomy/maskをfail-closed検証してtemperature-scaled logitsを返す。"""
        if not isinstance(context_features, th.Tensor) or context_features.ndim != 2:
            raise ValueError("context_features must have shape [B, context_dim]")
        if not isinstance(candidate_features, th.Tensor) or candidate_features.ndim != 3:
            raise ValueError("candidate_features must have shape [B, N, candidate_dim]")
        if (
            not isinstance(candidate_mask, th.Tensor)
            or candidate_mask.ndim != 2
            or candidate_mask.dtype != th.bool
        ):
            raise ValueError("candidate_mask must be bool with shape [B, N]")
        if context_features.dtype != th.float32 or candidate_features.dtype != th.float32:
            raise ValueError("ItemSelector features must use float32")
        if (
            context_features.device.type != "cpu"
            or candidate_features.device.type != "cpu"
            or candidate_mask.device.type != "cpu"
        ):
            raise ValueError("packaged ItemSelector requires CPU tensors")
        if not bool(th.isfinite(context_features).all()) or not bool(
            th.isfinite(candidate_features).all()
        ):
            raise ValueError("ItemSelector features must be finite")
        batch, candidates, feature_dim = candidate_features.shape
        if batch <= 0 or candidates <= 0:
            raise ValueError("ItemSelector batch and candidate count must be positive")
        if candidates > self.nmax:
            raise ValueError("candidate count exceeds artifact Nmax")
        if context_features.shape != (batch, self.manifest["context_dim"]):
            raise ValueError("context feature shape does not match artifact")
        if feature_dim != self.manifest["candidate_dim"]:
            raise ValueError("candidate feature shape does not match artifact")
        if candidate_mask.shape != (batch, candidates):
            raise ValueError("candidate_mask shape does not match candidates")
        if bool((~candidate_mask.any(dim=1)).any().item()):
            raise ValueError("all-masked candidate row is not allowed")
        with th.no_grad():
            logits = self._model(context_features, candidate_features, candidate_mask)
        if logits.shape != (batch, candidates):
            raise ArtifactBindingError("TorchScript output shape changed")
        scaled = logits / float(self.manifest["student_output_temperature"])
        if not bool(th.isfinite(scaled[candidate_mask]).all()):
            raise ArtifactBindingError("ItemSelector produced non-finite valid logits")
        return scaled

    @property
    def nmax(self) -> int:
        return int(self.manifest["nmax"])

    @property
    def vocabulary(self) -> frozenset[str]:
        return self._vocabulary

    @property
    def feature_schema(self) -> str:
        return str(self.manifest["feature_schema"])

    @property
    def ui_policy_config(self) -> NonModelUiPolicyConfigV1:
        return self._ui_policy_config

    @property
    def temperature(self) -> float:
        return float(self.manifest["temperature"])

    @property
    def confidence_threshold(self) -> float:
        return float(self.manifest["confidence_threshold"])

    @property
    def student_output_temperature(self) -> float:
        return float(self.manifest["student_output_temperature"])

    @property
    def provenance(self) -> Mapping[str, Any]:
        return dict(self.manifest["lineage"])

    @property
    def onnx_session(self) -> ort.InferenceSession:
        """parity/evaluator向けにload済みCPU sessionを返す。"""
        return self._onnx_session
