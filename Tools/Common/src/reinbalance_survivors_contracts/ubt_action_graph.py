"""Fixed-subject UBT action graph attestation contract.

`ReinBalanceEditor/Win64/Development` の GenerateClangDatabase 結果を、current
build inputs と UBT binary identity に結び付ける。別 subject、stale input、欠落 source、
非 canonical identity は current fidelity の計算前に拒否する。
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .canonical_json import canonical_hash, canonical_json_bytes, sha256_hex
from .ui_intent import ContractValidationError, ensure

UBT_ACTION_GRAPH_SCHEMA_VERSION = "survivors.ubt_action_graph.v1"
UBT_TARGET = "ReinBalanceEditor"
UBT_PLATFORM = "Win64"
UBT_CONFIGURATION = "Development"
UBT_PROJECT_FILE = "ReinBalance/ReinBalance.uproject"

_WIRE_KEYS = {
    "schema_version",
    "target",
    "platform",
    "configuration",
    "project_file",
    "build_input_hashes",
    "compiled_sources",
    "ubt_identity",
    "identity_sha256",
}
_BUILD_INPUT_SUFFIXES = (".Build.cs", ".Target.cs", ".uproject", ".uplugin")


def _sha256(value: Any, label: str) -> str:
    ensure(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label} must be lowercase sha256",
    )
    return value


def _repo_path(value: Any, label: str, *, suffix: str | None = None) -> str:
    ensure(isinstance(value, str) and value and "\\" not in value, f"{label} must be repo-relative POSIX path")
    path = PurePosixPath(value)
    ensure(not path.is_absolute() and ".." not in path.parts, f"{label} escapes repository")
    if suffix is not None:
        ensure(value.endswith(suffix), f"{label} must end with {suffix}")
    return value


def current_build_input_hashes(repo_root: Path) -> dict[str, str]:
    """current project 内の UBT rule/project/plugin inputs を exact map としてhashする。"""
    root = Path(repo_root).resolve()
    project_root = root / "ReinBalance"
    ensure(project_root.is_dir(), "missing ReinBalance project directory")
    candidates: set[Path] = set(project_root.glob("*.uproject"))
    source_root = project_root / "Source"
    if source_root.is_dir():
        candidates.update(source_root.rglob("*.Build.cs"))
        candidates.update(source_root.glob("*.Target.cs"))
    plugins_root = project_root / "Plugins"
    if plugins_root.is_dir():
        for plugin_root in (path for path in plugins_root.iterdir() if path.is_dir()):
            candidates.update(plugin_root.glob("*.uplugin"))
            plugin_source_root = plugin_root / "Source"
            if plugin_source_root.is_dir():
                candidates.update(plugin_source_root.rglob("*.Build.cs"))
    inputs: dict[str, str] = {}
    for path in sorted(candidate for candidate in candidates if candidate.is_file()):
        relative = path.relative_to(root).as_posix()
        inputs[relative] = sha256_hex(path.read_bytes())
    ensure(UBT_PROJECT_FILE in inputs, f"missing build input: {UBT_PROJECT_FILE}")
    ensure(any(path.endswith(".Target.cs") for path in inputs), "missing .Target.cs build input")
    ensure(any(path.endswith(".Build.cs") for path in inputs), "missing .Build.cs build input")
    return inputs


@dataclass(frozen=True)
class UbtActionGraphAttestation:
    """Canonical fixed-subject action graph attestation."""

    target: str
    platform: str
    configuration: str
    project_file: str
    build_input_hashes: Mapping[str, str]
    compiled_sources: tuple[str, ...]
    ubt_identity: str
    identity_sha256: str

    @property
    def subject(self) -> tuple[str, str, str]:
        return (self.target, self.platform, self.configuration)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": UBT_ACTION_GRAPH_SCHEMA_VERSION,
            "target": self.target,
            "platform": self.platform,
            "configuration": self.configuration,
            "project_file": self.project_file,
            "build_input_hashes": dict(self.build_input_hashes),
            "compiled_sources": list(self.compiled_sources),
            "ubt_identity": self.ubt_identity,
        }

    def to_wire(self) -> dict[str, Any]:
        return self._identity_payload() | {"identity_sha256": self.identity_sha256}

    @classmethod
    def from_wire(cls, value: Any) -> "UbtActionGraphAttestation":
        ensure(isinstance(value, Mapping) and set(value) == _WIRE_KEYS, "UBT action graph fields mismatch")
        ensure(value["schema_version"] == UBT_ACTION_GRAPH_SCHEMA_VERSION, "unsupported UBT action graph schema_version")
        ensure(value["target"] == UBT_TARGET, "target must be ReinBalanceEditor")
        ensure(value["platform"] == UBT_PLATFORM, "platform must be Win64")
        ensure(value["configuration"] == UBT_CONFIGURATION, "configuration must be Development")
        ensure(value["project_file"] == UBT_PROJECT_FILE, "project_file must be ReinBalance/ReinBalance.uproject")

        build_inputs = value["build_input_hashes"]
        ensure(isinstance(build_inputs, Mapping) and build_inputs, "build_input_hashes must be a non-empty object")
        normalized_inputs: dict[str, str] = {}
        for relative, digest in build_inputs.items():
            path = _repo_path(relative, "build input path")
            ensure(path.endswith(_BUILD_INPUT_SUFFIXES), "unsupported build input path")
            normalized_inputs[path] = _sha256(digest, f"build input hash for {path}")
        sources = value["compiled_sources"]
        ensure(isinstance(sources, (list, tuple)) and sources, "compiled_sources must be a non-empty array")
        normalized_sources = tuple(_repo_path(source, "compiled source", suffix=".cpp") for source in sources)
        ensure(normalized_sources == tuple(sorted(set(normalized_sources))), "compiled_sources must be sorted unique")
        ubt_identity = _sha256(value["ubt_identity"], "ubt_identity")
        identity = _sha256(value["identity_sha256"], "identity_sha256")
        attestation = cls(
            UBT_TARGET,
            UBT_PLATFORM,
            UBT_CONFIGURATION,
            UBT_PROJECT_FILE,
            MappingProxyType(normalized_inputs),
            normalized_sources,
            ubt_identity,
            identity,
        )
        ensure(canonical_hash(attestation._identity_payload()) == identity, "UBT action graph identity mismatch")
        return attestation

    def validate_current(self, repo_root: Path) -> None:
        """build inputs と action sources が current checkout と一致することを検証する。"""
        root = Path(repo_root).resolve()
        current_inputs = current_build_input_hashes(root)
        attested_inputs = dict(self.build_input_hashes)
        ensure(
            attested_inputs == current_inputs,
            "UBT build input mismatch: "
            f"missing={sorted(set(current_inputs)-set(attested_inputs))}, "
            f"extra={sorted(set(attested_inputs)-set(current_inputs))}, "
            f"stale={sorted(path for path in set(current_inputs)&set(attested_inputs) if current_inputs[path] != attested_inputs[path])}",
        )
        for relative in self.compiled_sources:
            ensure((root / relative).is_file(), f"missing compiled source: {relative}")


def make_ubt_action_graph_attestation(
    repo_root: Path,
    compiled_sources: Sequence[str],
    *,
    ubt_identity: str,
) -> UbtActionGraphAttestation:
    """current build inputs と normalized source set から attestation を生成する。"""
    root = Path(repo_root).resolve()
    normalized_sources = sorted({_repo_path(source, "compiled source", suffix=".cpp") for source in compiled_sources})
    ensure(normalized_sources, "compiled_sources must not be empty")
    for relative in normalized_sources:
        ensure((root / relative).is_file(), f"missing compiled source: {relative}")
    payload = {
        "schema_version": UBT_ACTION_GRAPH_SCHEMA_VERSION,
        "target": UBT_TARGET,
        "platform": UBT_PLATFORM,
        "configuration": UBT_CONFIGURATION,
        "project_file": UBT_PROJECT_FILE,
        "build_input_hashes": current_build_input_hashes(root),
        "compiled_sources": normalized_sources,
        "ubt_identity": _sha256(ubt_identity, "ubt_identity"),
    }
    return UbtActionGraphAttestation.from_wire(payload | {"identity_sha256": canonical_hash(payload)})


def read_ubt_action_graph_attestation(path: Path, *, repo_root: Path | None = None) -> UbtActionGraphAttestation:
    """JSON attestation を読み、必要なら current checkout と照合する。"""
    try:
        value = json.loads(Path(path).read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractValidationError(f"cannot read UBT action graph attestation: {exc}") from exc
    attestation = UbtActionGraphAttestation.from_wire(value)
    if repo_root is not None:
        attestation.validate_current(repo_root)
    return attestation


def write_ubt_action_graph_attestation(path: Path, attestation: UbtActionGraphAttestation) -> None:
    """canonical attestation bytes を同一 directory の一時fileからatomic replaceする。"""
    ensure(isinstance(attestation, UbtActionGraphAttestation), "attestation type mismatch")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(attestation.to_wire()) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            Path(temporary_name).unlink(missing_ok=True)
        except OSError:
            pass
        raise


__all__ = [
    "UBT_ACTION_GRAPH_SCHEMA_VERSION",
    "UBT_TARGET",
    "UBT_PLATFORM",
    "UBT_CONFIGURATION",
    "UBT_PROJECT_FILE",
    "UbtActionGraphAttestation",
    "current_build_input_hashes",
    "make_ubt_action_graph_attestation",
    "read_ubt_action_graph_attestation",
    "write_ubt_action_graph_attestation",
]
