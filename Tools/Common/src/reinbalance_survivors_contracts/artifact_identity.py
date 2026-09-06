"""Survivors artifact identity contract.

Descriptor identity is computed only from immutable content references,
ordered parent refs, and stable metadata. Timestamps, operators, local paths,
and validation results belong to new child nodes or non-identity metadata.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .canonical_json import canonical_hash, canonical_json_bytes
from .ui_intent import (
    ContractValidationError,
    ensure,
    is_strict_int,
    require_schema_version,
)

OBJECT_URI_PREFIX = "artifact://sha256/"
ARTIFACT_REF_SCHEMA_VERSION = "artifact_ref.v1"
ARTIFACT_NODE_REF_SCHEMA_VERSION = "artifact_node_ref.v1"
ARTIFACT_DESCRIPTOR_SCHEMA_VERSION = "artifact_descriptor.v1"
VALIDATION_VERDICT_SCHEMA_VERSION = "validation_verdict.v1"
RESTORE_TEST_VERDICT_SCHEMA_VERSION = "restore_test_verdict.v1"

ARTIFACT_NODE_KINDS = frozenset(
    {
        "source_descriptor",
        "teacher_validation_verdict",
        "choice_dataset_release",
        "item_selector_release",
        "combat_student_release",
        "perception_calibration_profile",
        "perception_final_verdict",
        "runtime_bundle",
        "replay_shadow_verdict",
        "canary_campaign",
        "goal_evidence",
        "restore_test_verdict",
    }
)

_VERDICT_NODE_KINDS = frozenset(
    {"teacher_validation_verdict", "replay_shadow_verdict"}
)
_RESTORE_VERIFY_MODES = frozenset({"full", "sample"})
_PRIVATE_PATH_MARKERS = ("..", "\\")
_VOLATILE_IDENTITY_KEYS = frozenset(
    {
        "created_at_utc",
        "updated_at_utc",
        "validated_at_utc",
        "verified_at_utc",
        "operator",
        "operator_id",
        "hostname",
        "host",
        "path",
        "local_path",
        "absolute_path",
        "source_path",
        "workspace_path",
        "validation_result",
        "validation_errors",
    }
)
_HEX_DIGITS = frozenset("0123456789abcdef")
_ARTIFACT_REF_WIRE_KEYS = frozenset(
    {
        "schema_version",
        "logical_id",
        "sha256",
        "size_bytes",
        "media_type",
        "store_uri",
    }
)
_ARTIFACT_NODE_REF_WIRE_KEYS = frozenset(
    {"schema_version", "logical_id", "node_kind", "identity_hash"}
)
_ARTIFACT_DESCRIPTOR_WIRE_KEYS = frozenset(
    {
        "schema_version",
        "identity_hash",
        "identity",
        "non_identity_metadata",
    }
)
_ARTIFACT_DESCRIPTOR_IDENTITY_WIRE_KEYS = frozenset(
    {
        "schema_version",
        "logical_id",
        "node_kind",
        "producer_id",
        "producer_version",
        "identity_metadata",
        "parents",
        "files",
    }
)

__all__ = [
    "OBJECT_URI_PREFIX",
    "ARTIFACT_REF_SCHEMA_VERSION",
    "ARTIFACT_NODE_REF_SCHEMA_VERSION",
    "ARTIFACT_DESCRIPTOR_SCHEMA_VERSION",
    "VALIDATION_VERDICT_SCHEMA_VERSION",
    "RESTORE_TEST_VERDICT_SCHEMA_VERSION",
    "ARTIFACT_NODE_KINDS",
    "ArtifactRef",
    "ArtifactNodeRef",
    "ArtifactDescriptor",
    "ValidationVerdict",
    "RestoreTestVerdict",
    "artifact_uri",
    "parse_artifact_uri",
    "is_sha256_hex",
]


def is_sha256_hex(value: Any) -> bool:
    """Return True only for lowercase 64-character SHA-256 hex strings."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in _HEX_DIGITS for ch in value)
    )


def _require_sha256(value: Any, label: str) -> str:
    ensure(is_sha256_hex(value), f"{label} must be lowercase 64-hex sha256")
    return value


def artifact_uri(sha256: str) -> str:
    """Build the canonical content-addressed object URI."""
    return f"{OBJECT_URI_PREFIX}{_require_sha256(sha256, 'sha256')}"


def parse_artifact_uri(uri: str) -> str:
    """Return the SHA-256 hash from an ``artifact://sha256/<hash>`` URI."""
    ensure(isinstance(uri, str), "artifact uri must be a string")
    ensure(uri.startswith(OBJECT_URI_PREFIX), "artifact uri must use artifact://sha256/")
    return _require_sha256(uri[len(OBJECT_URI_PREFIX) :], "artifact uri sha256")


def _require_nonempty_str(value: Any, label: str) -> str:
    ensure(isinstance(value, str) and value != "", f"{label} must be a non-empty string")
    return value


def _ensure_wire_keys(data: Any, allowed: frozenset[str], label: str) -> None:
    ensure(isinstance(data, Mapping), f"{label} must be an object")
    keys = set(data)
    unknown = keys - allowed
    if unknown:
        raise ContractValidationError(f"unknown {label} fields: {sorted(unknown)}")
    missing = allowed - keys
    if missing:
        raise ContractValidationError(f"missing {label} fields: {sorted(missing)}")


def _validate_logical_id(value: str, label: str = "logical_id") -> None:
    _require_nonempty_str(value, label)
    ensure(not value.startswith("/"), f"{label} must not be an absolute path")
    ensure(":" not in value, f"{label} must not contain a drive or URI separator")
    ensure(
        not any(part in ("", ".") for part in value.split("/")),
        f"{label} must be a slash-separated logical id",
    )
    ensure(".." not in value.split("/"), f"{label} must not escape its root")
    ensure(
        not any(marker in value for marker in _PRIVATE_PATH_MARKERS),
        f"{label} must not contain local path markers",
    )


def _canonical_copy(value: Any, label: str) -> Any:
    try:
        return json.loads(canonical_json_bytes(value).decode("utf-8"))
    except Exception as exc:
        if isinstance(exc, ContractValidationError):
            raise
        raise ContractValidationError(f"{label} must be canonical JSON compatible") from exc


def _reject_volatile_identity_keys(value: Any, path: str = "identity_metadata") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = key.lower()
            ensure(
                lowered not in _VOLATILE_IDENTITY_KEYS
                and not lowered.endswith("_path")
                and not lowered.endswith("_at_utc"),
                f"{path}.{key} is volatile and must not be part of identity",
            )
            _reject_volatile_identity_keys(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_volatile_identity_keys(child, f"{path}[{index}]")


@dataclass(frozen=True)
class ArtifactRef:
    """Content-addressed file reference stored outside Git."""

    logical_id: str
    sha256: str
    size_bytes: int
    media_type: str
    store_uri: str
    schema_version: str = ARTIFACT_REF_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ARTIFACT_REF_SCHEMA_VERSION:
            raise ContractValidationError(
                f"unsupported artifact ref schema_version {self.schema_version!r}"
            )
        _validate_logical_id(self.logical_id)
        _require_sha256(self.sha256, "artifact ref sha256")
        ensure(
            is_strict_int(self.size_bytes) and self.size_bytes >= 0,
            "artifact ref size_bytes must be a non-negative int",
        )
        _require_nonempty_str(self.media_type, "artifact ref media_type")
        parsed = parse_artifact_uri(self.store_uri)
        ensure(
            parsed == self.sha256,
            "artifact ref store_uri sha256 must match content sha256",
        )

    def to_wire(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "logical_id": self.logical_id,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "media_type": self.media_type,
            "store_uri": self.store_uri,
        }

    @classmethod
    def from_wire(cls, data: Mapping[str, Any]) -> "ArtifactRef":
        _ensure_wire_keys(data, _ARTIFACT_REF_WIRE_KEYS, "artifact ref")
        require_schema_version(data, ARTIFACT_REF_SCHEMA_VERSION, "artifact ref")
        return cls(
            logical_id=data["logical_id"],
            sha256=data["sha256"],
            size_bytes=data["size_bytes"],
            media_type=data["media_type"],
            store_uri=data["store_uri"],
            schema_version=data["schema_version"],
        )


@dataclass(frozen=True)
class ArtifactNodeRef:
    """Immutable reference from a child descriptor to one parent descriptor."""

    logical_id: str
    node_kind: str
    identity_hash: str
    schema_version: str = ARTIFACT_NODE_REF_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ARTIFACT_NODE_REF_SCHEMA_VERSION:
            raise ContractValidationError(
                f"unsupported artifact node ref schema_version {self.schema_version!r}"
            )
        _validate_logical_id(self.logical_id)
        _require_nonempty_str(self.node_kind, "artifact node ref node_kind")
        ensure(self.node_kind in ARTIFACT_NODE_KINDS, f"unknown node_kind {self.node_kind!r}")
        _require_sha256(self.identity_hash, "artifact node ref identity_hash")

    def to_wire(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "logical_id": self.logical_id,
            "node_kind": self.node_kind,
            "identity_hash": self.identity_hash,
        }

    @classmethod
    def from_wire(cls, data: Mapping[str, Any]) -> "ArtifactNodeRef":
        _ensure_wire_keys(data, _ARTIFACT_NODE_REF_WIRE_KEYS, "artifact node ref")
        require_schema_version(data, ARTIFACT_NODE_REF_SCHEMA_VERSION, "artifact node ref")
        return cls(
            logical_id=data["logical_id"],
            node_kind=data["node_kind"],
            identity_hash=data["identity_hash"],
            schema_version=data["schema_version"],
        )


def _coerce_parent_refs(values: Sequence[ArtifactNodeRef | Mapping[str, Any]]) -> tuple[ArtifactNodeRef, ...]:
    return tuple(
        value if isinstance(value, ArtifactNodeRef) else ArtifactNodeRef.from_wire(value)
        for value in values
    )


def _coerce_file_refs(values: Sequence[ArtifactRef | Mapping[str, Any]]) -> tuple[ArtifactRef, ...]:
    return tuple(
        value if isinstance(value, ArtifactRef) else ArtifactRef.from_wire(value)
        for value in values
    )


@dataclass(frozen=True)
class ArtifactDescriptor:
    """Immutable artifact node descriptor.

    ``identity_hash`` excludes ``non_identity_metadata`` by construction.
    """

    logical_id: str
    node_kind: str
    producer_id: str
    producer_version: str
    identity_metadata: Mapping[str, Any] = field(default_factory=dict)
    parents: Sequence[ArtifactNodeRef | Mapping[str, Any]] = field(default_factory=tuple)
    files: Sequence[ArtifactRef | Mapping[str, Any]] = field(default_factory=tuple)
    non_identity_metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = ARTIFACT_DESCRIPTOR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ARTIFACT_DESCRIPTOR_SCHEMA_VERSION:
            raise ContractValidationError(
                f"unsupported artifact descriptor schema_version {self.schema_version!r}"
            )
        _validate_logical_id(self.logical_id)
        _require_nonempty_str(self.node_kind, "artifact descriptor node_kind")
        ensure(self.node_kind in ARTIFACT_NODE_KINDS, f"unknown node_kind {self.node_kind!r}")
        _require_nonempty_str(self.producer_id, "artifact descriptor producer_id")
        _require_nonempty_str(
            self.producer_version, "artifact descriptor producer_version"
        )
        metadata = _canonical_copy(dict(self.identity_metadata), "identity_metadata")
        _reject_volatile_identity_keys(metadata)
        non_identity = _canonical_copy(
            dict(self.non_identity_metadata), "non_identity_metadata"
        )
        parents = _coerce_parent_refs(tuple(self.parents))
        files = _coerce_file_refs(tuple(self.files))
        object.__setattr__(self, "identity_metadata", metadata)
        object.__setattr__(self, "non_identity_metadata", non_identity)
        object.__setattr__(self, "parents", parents)
        object.__setattr__(self, "files", files)

    def identity_payload(self) -> dict[str, Any]:
        """Return the canonical identity payload for hash calculation."""
        return {
            "schema_version": self.schema_version,
            "logical_id": self.logical_id,
            "node_kind": self.node_kind,
            "producer_id": self.producer_id,
            "producer_version": self.producer_version,
            "identity_metadata": self.identity_metadata,
            "parents": [parent.to_wire() for parent in self.parents],
            "files": [file_ref.to_wire() for file_ref in self.files],
        }

    @property
    def identity_hash(self) -> str:
        return canonical_hash(self.identity_payload())

    def node_ref(self) -> ArtifactNodeRef:
        return ArtifactNodeRef(
            logical_id=self.logical_id,
            node_kind=self.node_kind,
            identity_hash=self.identity_hash,
        )

    def to_wire(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "identity_hash": self.identity_hash,
            "identity": self.identity_payload(),
            "non_identity_metadata": self.non_identity_metadata,
        }

    @classmethod
    def from_wire(cls, data: Mapping[str, Any]) -> "ArtifactDescriptor":
        _ensure_wire_keys(data, _ARTIFACT_DESCRIPTOR_WIRE_KEYS, "artifact descriptor")
        require_schema_version(data, ARTIFACT_DESCRIPTOR_SCHEMA_VERSION, "artifact descriptor")
        _require_sha256(data["identity_hash"], "artifact descriptor identity_hash")
        identity = data["identity"]
        _ensure_wire_keys(
            identity,
            _ARTIFACT_DESCRIPTOR_IDENTITY_WIRE_KEYS,
            "artifact descriptor identity",
        )
        require_schema_version(
            identity, ARTIFACT_DESCRIPTOR_SCHEMA_VERSION, "artifact descriptor identity"
        )
        descriptor = cls(
            logical_id=identity["logical_id"],
            node_kind=identity["node_kind"],
            producer_id=identity["producer_id"],
            producer_version=identity["producer_version"],
            identity_metadata=identity["identity_metadata"],
            parents=identity["parents"],
            files=identity["files"],
            non_identity_metadata=data["non_identity_metadata"],
            schema_version=data["schema_version"],
        )
        ensure(
            descriptor.identity_hash == data["identity_hash"],
            "artifact descriptor identity_hash does not match identity fields",
        )
        return descriptor


@dataclass(frozen=True)
class ValidationVerdict:
    """Validation result as a new immutable child artifact node."""

    logical_id: str
    verdict_kind: str
    subject: ArtifactNodeRef | Mapping[str, Any]
    gate_version: str
    metrics: Mapping[str, Any]
    split_ids: Sequence[str]
    session_ids: Sequence[str]
    passed: bool
    blocking_reasons: Sequence[str]
    producer_id: str = "artifact-validation"
    producer_version: str = VALIDATION_VERDICT_SCHEMA_VERSION
    files: Sequence[ArtifactRef | Mapping[str, Any]] = field(default_factory=tuple)
    non_identity_metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = VALIDATION_VERDICT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != VALIDATION_VERDICT_SCHEMA_VERSION:
            raise ContractValidationError(
                f"unsupported validation verdict schema_version {self.schema_version!r}"
            )
        _validate_logical_id(self.logical_id)
        ensure(
            self.verdict_kind in _VERDICT_NODE_KINDS,
            f"unsupported validation verdict kind {self.verdict_kind!r}",
        )
        subject = (
            self.subject
            if isinstance(self.subject, ArtifactNodeRef)
            else ArtifactNodeRef.from_wire(self.subject)
        )
        _require_nonempty_str(self.gate_version, "validation verdict gate_version")
        metrics = _canonical_copy(dict(self.metrics), "validation verdict metrics")
        split_ids = tuple(_require_nonempty_str(value, "split_id") for value in self.split_ids)
        session_ids = tuple(
            _require_nonempty_str(value, "session_id") for value in self.session_ids
        )
        ensure(isinstance(self.passed, bool), "validation verdict passed must be bool")
        blocking = tuple(
            _require_nonempty_str(value, "blocking_reason")
            for value in self.blocking_reasons
        )
        files = _coerce_file_refs(tuple(self.files))
        non_identity = _canonical_copy(
            dict(self.non_identity_metadata), "validation verdict non_identity_metadata"
        )
        object.__setattr__(self, "subject", subject)
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "split_ids", split_ids)
        object.__setattr__(self, "session_ids", session_ids)
        object.__setattr__(self, "blocking_reasons", blocking)
        object.__setattr__(self, "files", files)
        object.__setattr__(self, "non_identity_metadata", non_identity)

    def to_descriptor(self) -> ArtifactDescriptor:
        subject = self.subject
        return ArtifactDescriptor(
            logical_id=self.logical_id,
            node_kind=self.verdict_kind,
            producer_id=self.producer_id,
            producer_version=self.producer_version,
            identity_metadata={
                "verdict_schema_version": self.schema_version,
                "verdict_kind": self.verdict_kind,
                "subject_logical_id": subject.logical_id,
                "subject_node_kind": subject.node_kind,
                "subject_identity_hash": subject.identity_hash,
                "gate_version": self.gate_version,
                "metrics": self.metrics,
                "split_ids": list(self.split_ids),
                "session_ids": list(self.session_ids),
                "passed": self.passed,
                "blocking_reasons": list(self.blocking_reasons),
            },
            parents=(subject,),
            files=self.files,
            non_identity_metadata=self.non_identity_metadata,
        )


@dataclass(frozen=True)
class RestoreTestVerdict:
    """Restore-test verdict required before final/canary release."""

    logical_id: str
    subject: ArtifactNodeRef | Mapping[str, Any]
    manifest_hash: str
    primary_root: str
    backup_root: str
    verify_mode: str
    checked_object_count: int
    passed: bool
    blocking_reasons: Sequence[str]
    producer_id: str = "artifact-restore-test"
    producer_version: str = RESTORE_TEST_VERDICT_SCHEMA_VERSION
    non_identity_metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = RESTORE_TEST_VERDICT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RESTORE_TEST_VERDICT_SCHEMA_VERSION:
            raise ContractValidationError(
                f"unsupported restore test verdict schema_version {self.schema_version!r}"
            )
        _validate_logical_id(self.logical_id)
        subject = (
            self.subject
            if isinstance(self.subject, ArtifactNodeRef)
            else ArtifactNodeRef.from_wire(self.subject)
        )
        _require_sha256(self.manifest_hash, "restore test manifest_hash")
        _require_nonempty_str(self.primary_root, "restore test primary_root")
        _require_nonempty_str(self.backup_root, "restore test backup_root")
        ensure(self.verify_mode in _RESTORE_VERIFY_MODES, "restore test verify_mode invalid")
        ensure(
            is_strict_int(self.checked_object_count) and self.checked_object_count >= 0,
            "restore test checked_object_count must be a non-negative int",
        )
        ensure(isinstance(self.passed, bool), "restore test passed must be bool")
        blocking = tuple(
            _require_nonempty_str(value, "blocking_reason")
            for value in self.blocking_reasons
        )
        non_identity = _canonical_copy(
            dict(self.non_identity_metadata), "restore test non_identity_metadata"
        )
        object.__setattr__(self, "subject", subject)
        object.__setattr__(self, "blocking_reasons", blocking)
        object.__setattr__(self, "non_identity_metadata", non_identity)

    def to_descriptor(self) -> ArtifactDescriptor:
        subject = self.subject
        non_identity = dict(self.non_identity_metadata)
        non_identity.update(
            {
                "primary_root": self.primary_root,
                "backup_root": self.backup_root,
            }
        )
        return ArtifactDescriptor(
            logical_id=self.logical_id,
            node_kind="restore_test_verdict",
            producer_id=self.producer_id,
            producer_version=self.producer_version,
            identity_metadata={
                "restore_test_schema_version": self.schema_version,
                "subject_logical_id": subject.logical_id,
                "subject_node_kind": subject.node_kind,
                "subject_identity_hash": subject.identity_hash,
                "manifest_hash": self.manifest_hash,
                "verify_mode": self.verify_mode,
                "checked_object_count": self.checked_object_count,
                "passed": self.passed,
                "blocking_reasons": list(self.blocking_reasons),
            },
            parents=(subject,),
            files=(),
            non_identity_metadata=non_identity,
        )
