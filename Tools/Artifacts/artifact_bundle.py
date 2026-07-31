"""Deterministic artifact export/import and backup verification helpers."""

from __future__ import annotations

import json
import os
import random
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from reinbalance_survivors_contracts import (
    canonical_hash,
    canonical_json_bytes,
    sha256_hex,
    validate_artifact_dag,
)
from reinbalance_survivors_contracts.artifact_identity import (
    ArtifactDescriptor,
    ArtifactRef,
)

if __package__:
    from .artifact_store import (
        ArtifactStore,
        ArtifactStoreError,
        iter_manifest_object_entries,
        object_ref_from_manifest_entry,
        validate_manifest_descriptor_object_refs,
        verify_manifest_hash,
    )
else:
    from artifact_store import (
        ArtifactStore,
        ArtifactStoreError,
        iter_manifest_object_entries,
        object_ref_from_manifest_entry,
        validate_manifest_descriptor_object_refs,
        verify_manifest_hash,
    )

BUNDLE_MANIFEST_SCHEMA_VERSION = "artifact_bundle_manifest.v1"
_PRIVATE_CLASSES = frozenset({"private", "restricted", "secret"})
_PRIVACY_CLASSES = frozenset({"public", "internal", *_PRIVATE_CLASSES})
_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)
_BUNDLE_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "export_id",
        "include_private",
        "descriptors",
        "objects",
        "manifest_hash",
    }
)
_BUNDLE_OBJECT_KEYS = frozenset(
    {
        "schema_version",
        "logical_id",
        "sha256",
        "size_bytes",
        "media_type",
        "store_uri",
        "license",
        "privacy_classification",
        "retention_until_utc",
        "export_included",
    }
)


@dataclass(frozen=True)
class VolumeIdentity:
    resolved_path: Path
    device_id: object


@dataclass(frozen=True)
class BundleVerifyReport:
    checked_count: int
    missing_objects: tuple[str, ...]
    corrupt_objects: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.missing_objects and not self.corrupt_objects


@dataclass(frozen=True)
class BundleImportResult:
    manifest: dict[str, Any]
    manifest_hash: str
    verification_report: BundleVerifyReport


@dataclass(frozen=True)
class _PendingImportObject:
    ref: ArtifactRef
    data: bytes


def resolve_volume_identity(path: str | Path) -> VolumeIdentity:
    resolved = Path(path).expanduser().resolve()
    stat_target = resolved if resolved.exists() else _nearest_existing_parent(resolved)
    return VolumeIdentity(resolved_path=resolved, device_id=os.stat(stat_target).st_dev)


def _nearest_existing_parent(path: Path) -> Path:
    current = path
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def assert_distinct_store_roots(
    primary_root: str | Path,
    backup_root: str | Path,
    *,
    resolver: Callable[[Path], VolumeIdentity] = resolve_volume_identity,
) -> None:
    primary = resolver(Path(primary_root))
    backup = resolver(Path(backup_root))
    if primary.resolved_path == backup.resolved_path:
        raise ValueError("primary and backup artifact roots resolve to the same path")
    if primary.device_id == backup.device_id:
        raise ValueError("primary and backup artifact roots must be on different volumes/devices")


def _coerce_descriptor(value: ArtifactDescriptor | Mapping[str, Any]) -> ArtifactDescriptor:
    return value if isinstance(value, ArtifactDescriptor) else ArtifactDescriptor.from_wire(value)


def _default_privacy(ref: ArtifactRef) -> str:
    if ref.media_type.startswith("video/") or ref.media_type.startswith("image/"):
        return "private"
    return "internal"


def _sanitize_descriptor_wire(descriptor: ArtifactDescriptor) -> dict[str, Any]:
    wire = descriptor.to_wire()
    # Exported manifests must not leak local paths, operators, hostnames, or timestamps.
    wire["non_identity_metadata"] = {}
    return wire


def _parse_retention_timestamp(value: str, *, label: str) -> None:
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ArtifactStoreError(f"{label} must be an ISO-8601 timestamp") from exc


def _validate_bundle_manifest_shape(manifest: Mapping[str, Any]) -> None:
    if not isinstance(manifest, Mapping):
        raise ArtifactStoreError("bundle manifest must be an object")
    keys = set(manifest)
    unknown = keys - _BUNDLE_MANIFEST_KEYS
    if unknown:
        raise ArtifactStoreError(f"unknown bundle manifest fields: {sorted(unknown)}")
    missing = _BUNDLE_MANIFEST_KEYS - keys
    if missing:
        raise ArtifactStoreError(f"missing bundle manifest fields: {sorted(missing)}")
    if manifest["schema_version"] != BUNDLE_MANIFEST_SCHEMA_VERSION:
        raise ArtifactStoreError("unsupported bundle manifest schema_version")
    if not isinstance(manifest["export_id"], str) or not manifest["export_id"]:
        raise ArtifactStoreError("bundle manifest export_id must be a non-empty string")
    if not isinstance(manifest["include_private"], bool):
        raise ArtifactStoreError("bundle manifest include_private must be bool")


def _validate_bundle_object_entry(
    entry: Mapping[str, Any],
    *,
    include_private: bool,
    index: int,
) -> None:
    keys = set(entry)
    unknown = keys - _BUNDLE_OBJECT_KEYS
    if unknown:
        raise ArtifactStoreError(
            f"unknown bundle object fields at objects[{index}]: {sorted(unknown)}"
        )
    missing = _BUNDLE_OBJECT_KEYS - keys
    if missing:
        raise ArtifactStoreError(
            f"missing bundle object fields at objects[{index}]: {sorted(missing)}"
        )
    if not isinstance(entry["license"], str) or not entry["license"]:
        raise ArtifactStoreError(
            f"bundle objects[{index}] license must be a non-empty string"
        )
    privacy = entry["privacy_classification"]
    if not isinstance(privacy, str) or privacy not in _PRIVACY_CLASSES:
        raise ArtifactStoreError(
            f"bundle objects[{index}] privacy_classification is unsupported"
        )
    retention = entry["retention_until_utc"]
    if not isinstance(retention, str) or not retention:
        raise ArtifactStoreError(
            f"bundle objects[{index}] retention_until_utc must be a non-empty string"
        )
    _parse_retention_timestamp(
        retention,
        label=f"bundle objects[{index}] retention_until_utc",
    )
    export_included = entry["export_included"]
    if not isinstance(export_included, bool):
        raise ArtifactStoreError(f"bundle objects[{index}] export_included must be bool")
    expected_export_included = include_private or privacy not in _PRIVATE_CLASSES
    if export_included == expected_export_included:
        return
    if not include_private and privacy in _PRIVATE_CLASSES:
        raise ArtifactStoreError(
            f"public bundle manifest includes private object {entry['logical_id']!r}"
        )
    raise ArtifactStoreError(
        f"bundle objects[{index}] export_included does not match privacy_classification"
    )


def _validate_bundle_manifest(manifest: Mapping[str, Any]) -> None:
    _validate_bundle_manifest_shape(manifest)
    verify_manifest_hash(manifest)
    validate_manifest_descriptor_object_refs(manifest)
    validate_artifact_dag(manifest["descriptors"])
    include_private = manifest["include_private"]
    for index, entry in enumerate(iter_manifest_object_entries(manifest)):
        _validate_bundle_object_entry(entry, include_private=include_private, index=index)


def create_bundle_manifest(
    *,
    descriptors: Sequence[ArtifactDescriptor | Mapping[str, Any]],
    export_id: str,
    license_by_logical_id: Mapping[str, str] | None = None,
    privacy_by_logical_id: Mapping[str, str] | None = None,
    retention_until_utc: str,
    include_private: bool = False,
) -> dict[str, Any]:
    license_by_logical_id = dict(license_by_logical_id or {})
    privacy_by_logical_id = dict(privacy_by_logical_id or {})
    coerced = tuple(_coerce_descriptor(value) for value in descriptors)
    validate_artifact_dag(coerced)

    object_entries: list[dict[str, Any]] = []
    logical_to_sha: dict[str, str] = {}
    for descriptor in coerced:
        for ref in descriptor.files:
            existing = logical_to_sha.get(ref.logical_id)
            if existing is not None and existing != ref.sha256:
                raise ArtifactStoreError(
                    f"logical id {ref.logical_id!r} has multiple object hashes"
                )
            logical_to_sha[ref.logical_id] = ref.sha256
            privacy = privacy_by_logical_id.get(ref.logical_id, _default_privacy(ref))
            export_included = include_private or privacy not in _PRIVATE_CLASSES
            object_entries.append(
                {
                    "schema_version": ref.schema_version,
                    "logical_id": ref.logical_id,
                    "sha256": ref.sha256,
                    "size_bytes": ref.size_bytes,
                    "media_type": ref.media_type,
                    "store_uri": ref.store_uri,
                    "license": license_by_logical_id.get(ref.logical_id, "unspecified"),
                    "privacy_classification": privacy,
                    "retention_until_utc": retention_until_utc,
                    "export_included": export_included,
                }
            )

    core = {
        "schema_version": BUNDLE_MANIFEST_SCHEMA_VERSION,
        "export_id": export_id,
        "include_private": include_private,
        "descriptors": [
            _sanitize_descriptor_wire(descriptor)
            for descriptor in sorted(coerced, key=lambda value: value.identity_hash)
        ],
        "objects": sorted(
            object_entries,
            key=lambda entry: (entry["sha256"], entry["logical_id"]),
        ),
    }
    manifest = dict(core)
    manifest["manifest_hash"] = canonical_hash(core)
    _validate_bundle_manifest(manifest)
    return manifest


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(filename=name, date_time=_ZIP_EPOCH)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    return info


def export_bundle(store: ArtifactStore, manifest: Mapping[str, Any], zip_path: str | Path) -> Path:
    _validate_bundle_manifest(manifest)
    path = Path(zip_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    seen_objects: set[str] = set()
    with zipfile.ZipFile(path, "w", allowZip64=True) as zf:
        zf.writestr(_zip_info("manifest.json"), canonical_json_bytes(manifest))
        for entry in sorted(
            manifest.get("objects", []), key=lambda item: (item["sha256"], item["logical_id"])
        ):
            if entry.get("export_included") is False or entry["sha256"] in seen_objects:
                continue
            ref = object_ref_from_manifest_entry(entry, label="manifest object entry")
            verification = store.verify(ref)
            if not verification.ok:
                raise ArtifactStoreError(
                    f"cannot export {ref.store_uri}: {verification.reason}"
                )
            zf.writestr(
                _zip_info(f"objects/sha256/{ref.sha256}"),
                store.object_path(ref.store_uri).read_bytes(),
            )
            seen_objects.add(ref.sha256)
    return path


def import_bundle(
    zip_path: str | Path,
    target_store: ArtifactStore,
    *,
    verify_mode: str = "full",
    sample_size: int | None = None,
    random_seed: int = 0,
) -> BundleImportResult:
    path = Path(zip_path).expanduser().resolve()
    with zipfile.ZipFile(path) as zf:
        manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
        _validate_bundle_manifest(manifest)
        manifest_hash = manifest["manifest_hash"]
        pending_objects = _read_verified_bundle_objects(zf, manifest)
    _preflight_target_store_import(target_store, pending_objects)
    for pending in pending_objects:
        target_store.put_bytes(
            logical_id=pending.ref.logical_id,
            data=pending.data,
            media_type=pending.ref.media_type,
        )
    report = verify_bundle_objects(
        target_store,
        manifest,
        verify_mode=verify_mode,
        sample_size=sample_size,
        random_seed=random_seed,
    )
    return BundleImportResult(
        manifest=manifest,
        manifest_hash=manifest_hash,
        verification_report=report,
    )


def _read_verified_bundle_objects(
    zf: zipfile.ZipFile,
    manifest: Mapping[str, Any],
) -> list[_PendingImportObject]:
    pending: list[_PendingImportObject] = []
    logical_to_sha: dict[str, str] = {}
    for entry in iter_manifest_object_entries(manifest):
        if entry.get("export_included") is False:
            continue
        ref = object_ref_from_manifest_entry(entry, label="manifest object entry")
        existing_sha = logical_to_sha.get(ref.logical_id)
        if existing_sha is not None and existing_sha != ref.sha256:
            raise ArtifactStoreError(
                f"bundle logical id {ref.logical_id!r} points to multiple hashes"
            )
        logical_to_sha[ref.logical_id] = ref.sha256
        object_name = f"objects/sha256/{ref.sha256}"
        try:
            data = zf.read(object_name)
        except KeyError as exc:
            raise ArtifactStoreError(f"bundle object is missing: {object_name}") from exc
        actual_sha = sha256_hex(data)
        if actual_sha != ref.sha256:
            raise ArtifactStoreError(
                f"bundle object sha256 mismatch: {ref.store_uri}"
            )
        if len(data) != ref.size_bytes:
            raise ArtifactStoreError(
                f"bundle object size mismatch: {ref.store_uri}"
            )
        pending.append(_PendingImportObject(ref=ref, data=data))
    return pending


def _preflight_target_store_import(
    target_store: ArtifactStore,
    pending_objects: Sequence[_PendingImportObject],
) -> None:
    for pending in pending_objects:
        target_store._ensure_logical_id_available(pending.ref)
        verification = target_store.verify(pending.ref)
        if verification.reason == "missing":
            continue
        if not verification.ok:
            raise ArtifactStoreError(
                f"existing target object {pending.ref.store_uri} is corrupt: {verification.reason}"
            )


def _entries_to_check(
    manifest: Mapping[str, Any],
    *,
    verify_mode: str,
    sample_size: int | None,
    random_seed: int,
) -> list[Mapping[str, Any]]:
    entries = [
        entry
        for entry in iter_manifest_object_entries(manifest)
        if entry.get("export_included") is not False
    ]
    if verify_mode == "full":
        return entries
    if verify_mode != "sample":
        raise ArtifactStoreError("verify_mode must be 'full' or 'sample'")
    if not entries:
        return []
    count = sample_size if sample_size is not None else 1
    count = max(1, min(count, len(entries)))
    return random.Random(random_seed).sample(entries, count)


def verify_bundle_objects(
    store: ArtifactStore,
    manifest: Mapping[str, Any],
    *,
    verify_mode: str = "full",
    sample_size: int | None = None,
    random_seed: int = 0,
) -> BundleVerifyReport:
    _validate_bundle_manifest(manifest)
    missing: list[str] = []
    corrupt: list[str] = []
    entries = _entries_to_check(
        manifest,
        verify_mode=verify_mode,
        sample_size=sample_size,
        random_seed=random_seed,
    )
    for entry in entries:
        ref = object_ref_from_manifest_entry(entry, label="manifest object entry")
        verification = store.verify(ref)
        if verification.reason == "missing":
            missing.append(ref.store_uri)
        elif not verification.ok:
            corrupt.append(ref.store_uri)
    return BundleVerifyReport(
        checked_count=len(entries),
        missing_objects=tuple(sorted(set(missing))),
        corrupt_objects=tuple(sorted(set(corrupt))),
    )
