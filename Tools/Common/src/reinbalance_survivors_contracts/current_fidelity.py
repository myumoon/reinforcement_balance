"""Resolve fidelity identity from the current checkout and fixed producers.

Caller supplied hash maps are deliberately absent from this API.  The packaged producer manifest,
strict generated-input descriptor, and fresh fixed-subject UBT attestation are resolved together.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping

from .canonical_json import canonical_hash, canonical_json_bytes, sha256_hex
from .fidelity_producer_paths import load_producer_path_manifest, resolve_gating_producer_hashes
from .ubt_action_graph import read_ubt_action_graph_attestation
from .ui_intent import ContractValidationError, ensure

GENERATED_INPUT_DESCRIPTOR_SCHEMA_VERSION = "survivors.generated_fidelity_inputs.v1"
_DESCRIPTOR_KEYS = {"schema_version", "inputs"}
_INPUT_SPEC_KEYS = {"path", "format"}
_FORMATS = {"json", "yaml"}


@dataclass(frozen=True)
class GeneratedInputSourceIdentity:
    """Generated input の source bytes と canonical object identity。"""

    source_path: str
    source_format: str
    source_sha256: str
    canonical_sha256: str


@dataclass(frozen=True)
class CurrentFidelityContext:
    """Current producer authority と監査可能な source identities。"""

    manifest_hash: str
    action_graph_identity: str
    generated_input_source_identities: Mapping[str, GeneratedInputSourceIdentity]
    current_gating_producer_hashes: Mapping[str, str]


def _repo_input_path(repo_root: Path, value: Any, label: str) -> tuple[str, Path]:
    ensure(isinstance(value, str) and value and "\\" not in value, f"{label} must be a repo-relative POSIX path")
    pure = PurePosixPath(value)
    ensure(not pure.is_absolute() and ".." not in pure.parts, f"{label} escapes repository")
    resolved = repo_root.joinpath(*pure.parts).resolve()
    try:
        resolved.relative_to(repo_root)
    except ValueError as exc:
        raise ContractValidationError(f"{label} escapes repository") from exc
    ensure(resolved.is_file(), f"missing generated input source: {value}")
    return value, resolved


def _load_yaml(raw_bytes: bytes, label: str) -> Any:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - package dependency protects this boundary
        raise ContractValidationError("PyYAML is required for generated YAML input") from exc
    try:
        return yaml.safe_load(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ContractValidationError(f"invalid generated YAML input {label}: {exc}") from exc


def _load_generated_source(path: Path, source_format: str, label: str) -> tuple[Any, bytes]:
    try:
        raw_bytes = path.read_bytes()
    except OSError as exc:
        raise ContractValidationError(f"cannot read generated input {label}: {exc}") from exc
    try:
        value = json.loads(raw_bytes) if source_format == "json" else _load_yaml(raw_bytes, label)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractValidationError(f"invalid generated JSON input {label}: {exc}") from exc
    ensure(isinstance(value, Mapping), f"generated input {label} must be an object")
    try:
        canonical = canonical_json_bytes(dict(value))
    except (TypeError, ValueError) as exc:
        raise ContractValidationError(f"generated input {label} is not canonicalizable: {exc}") from exc
    return json.loads(canonical), raw_bytes


def _read_descriptor(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(Path(path).read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractValidationError(f"cannot read generated input descriptor: {exc}") from exc
    ensure(isinstance(value, Mapping) and set(value) == _DESCRIPTOR_KEYS, "generated input descriptor fields mismatch")
    ensure(value["schema_version"] == GENERATED_INPUT_DESCRIPTOR_SCHEMA_VERSION, "unsupported generated input descriptor schema_version")
    ensure(isinstance(value["inputs"], Mapping), "generated input descriptor inputs must be an object")
    return value


def resolve_current_gating_producer_hashes(
    repo_root: Path,
    generated_input_descriptor_path: Path,
    ubt_action_graph_path: Path,
) -> CurrentFidelityContext:
    """Current checkout から13 producer hashを毎回再解決する。"""
    root = Path(repo_root).resolve()
    ensure(root.is_dir() and (root / "ReinBalance/ReinBalance.uproject").is_file(), "repo_root is not a ReinBalance checkout")
    manifest = load_producer_path_manifest()
    descriptor = _read_descriptor(generated_input_descriptor_path)
    expected_names = {
        name
        for entry in manifest.producers.values()
        for name in entry["generated_inputs"]
        if name != "target_build_attestation"
    }
    inputs = descriptor["inputs"]
    ensure(set(inputs) == expected_names, f"generated input keys mismatch: missing={sorted(expected_names-set(inputs))}, unknown={sorted(set(inputs)-expected_names)}")

    generated_values: dict[str, Any] = {}
    identities: dict[str, GeneratedInputSourceIdentity] = {}
    for name in sorted(expected_names):
        spec = inputs[name]
        if spec == "absent":
            ensure(name in {"deploy_obs_schema", "deploy_release_adapter"}, f"{name} cannot be absent")
            generated_values[name] = "absent"
            absent_hash = canonical_hash("absent")
            identities[name] = GeneratedInputSourceIdentity("absent", "absent", absent_hash, absent_hash)
            continue
        ensure(isinstance(spec, Mapping) and set(spec) == _INPUT_SPEC_KEYS, f"generated input spec keys mismatch: {name}")
        source_format = spec["format"]
        ensure(source_format in _FORMATS, f"generated input format invalid: {name}")
        relative, source_path = _repo_input_path(root, spec["path"], f"generated input path {name}")
        value, raw_bytes = _load_generated_source(source_path, source_format, name)
        generated_values[name] = value
        identities[name] = GeneratedInputSourceIdentity(
            relative,
            source_format,
            sha256_hex(raw_bytes),
            canonical_hash(value),
        )

    attestation_path = Path(ubt_action_graph_path)
    attestation = read_ubt_action_graph_attestation(attestation_path, repo_root=root)
    attestation_bytes = canonical_json_bytes(attestation.to_wire())
    generated_values["target_build_attestation"] = attestation.to_wire()
    identities["target_build_attestation"] = GeneratedInputSourceIdentity(
        str(attestation_path),
        "json",
        sha256_hex(attestation_path.read_bytes()),
        sha256_hex(attestation_bytes),
    )
    current = resolve_gating_producer_hashes(root, manifest, generated_values, attestation)
    return CurrentFidelityContext(
        manifest.manifest_hash,
        attestation.identity_sha256,
        MappingProxyType(identities),
        MappingProxyType(current),
    )


__all__ = [
    "GENERATED_INPUT_DESCRIPTOR_SCHEMA_VERSION",
    "GeneratedInputSourceIdentity",
    "CurrentFidelityContext",
    "resolve_current_gating_producer_hashes",
]
