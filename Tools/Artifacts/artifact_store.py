"""Content-addressed artifact object store for Survivors release assets."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from reinbalance_survivors_contracts import canonical_json_bytes, sha256_hex
from reinbalance_survivors_contracts.artifact_identity import (
    ArtifactRef,
    artifact_uri,
    is_sha256_hex,
    parse_artifact_uri,
)


class ArtifactStoreError(RuntimeError):
    """Raised when the artifact store cannot safely complete an operation."""


@dataclass(frozen=True)
class ObjectRecord:
    sha256: str
    size_bytes: int
    store_uri: str
    path: Path


@dataclass(frozen=True)
class ObjectVerification:
    store_uri: str
    ok: bool
    expected_sha256: str
    actual_sha256: str | None
    expected_size_bytes: int | None
    actual_size_bytes: int | None
    reason: str | None = None


@dataclass(frozen=True)
class StoreAuditReport:
    missing_objects: tuple[str, ...]
    corrupt_objects: tuple[str, ...]
    retention_due_objects: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not (
            self.missing_objects
            or self.corrupt_objects
            or self.retention_due_objects
        )


def _resolve_existing_or_parent(path: Path) -> Path:
    current = path
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def _resolve_under(root: Path, relative: str | Path) -> Path:
    rel = Path(relative)
    if rel.is_absolute():
        raise ArtifactStoreError("materialize destination must be relative")
    if any(part in ("", ".", "..") for part in rel.parts):
        raise ArtifactStoreError("materialize destination must not escape root")
    resolved = (root / rel).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ArtifactStoreError("path escapes artifact store root") from exc
    return resolved


def _parse_utc(value: str) -> datetime:
    text = value.replace("Z", "+00:00")
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class ArtifactStore:
    """Filesystem-backed content-addressed store.

    The root is always explicit. Objects are written temp-file first and then
    atomically renamed into ``objects/sha256/<prefix>/<hash>``.
    """

    def __init__(self, root: str | Path):
        if root is None:
            raise ArtifactStoreError("store root must be explicit")
        self.root = Path(root).expanduser().resolve()
        self.objects_root = self.root / "objects" / "sha256"
        self.logical_root = self.root / "index" / "logical"
        self.materialized_root = self.root / "materialized"
        self.objects_root.mkdir(parents=True, exist_ok=True)
        self.logical_root.mkdir(parents=True, exist_ok=True)
        self.materialized_root.mkdir(parents=True, exist_ok=True)

    def object_path(self, uri_or_hash: str) -> Path:
        sha256 = (
            parse_artifact_uri(uri_or_hash)
            if uri_or_hash.startswith("artifact://")
            else uri_or_hash
        )
        if not is_sha256_hex(sha256):
            raise ArtifactStoreError("object path requires a 64-hex sha256")
        path = (self.objects_root / sha256[:2] / sha256).resolve()
        try:
            path.relative_to(self.objects_root.resolve())
        except ValueError as exc:
            raise ArtifactStoreError("object path escapes store root") from exc
        return path

    def _logical_index_path(self, logical_id: str) -> Path:
        key = sha256_hex(logical_id.encode("utf-8"))
        return self.logical_root / f"{key}.json"

    def put(
        self,
        *,
        logical_id: str,
        source_path: str | Path,
        media_type: str,
    ) -> ArtifactRef:
        path = Path(source_path).expanduser().resolve()
        if not path.is_file():
            raise ArtifactStoreError(f"source artifact is not a file: {path}")
        return self.put_bytes(
            logical_id=logical_id,
            data=path.read_bytes(),
            media_type=media_type,
        )

    def put_bytes(self, *, logical_id: str, data: bytes, media_type: str) -> ArtifactRef:
        if not isinstance(data, bytes):
            raise ArtifactStoreError("put_bytes data must be bytes")
        sha256 = sha256_hex(data)
        ref = ArtifactRef(
            logical_id=logical_id,
            sha256=sha256,
            size_bytes=len(data),
            media_type=media_type,
            store_uri=artifact_uri(sha256),
        )
        self._ensure_logical_id_available(ref)
        destination = self.object_path(ref.store_uri)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            existing = self.verify(ref)
            if not existing.ok:
                raise ArtifactStoreError(
                    f"existing object {ref.store_uri} is corrupt: {existing.reason}"
                )
        else:
            fd, temp_name = tempfile.mkstemp(
                prefix=f".{sha256}.", suffix=".tmp", dir=str(destination.parent)
            )
            temp_path = Path(temp_name)
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
                temp_bytes = temp_path.read_bytes()
                if sha256_hex(temp_bytes) != sha256 or len(temp_bytes) != len(data):
                    raise ArtifactStoreError("temporary object failed hash/size verification")
                os.replace(temp_path, destination)
            finally:
                if temp_path.exists():
                    temp_path.unlink()

        self._record_logical_id(ref)
        return ref

    def _ensure_logical_id_available(self, ref: ArtifactRef) -> None:
        index_path = self._logical_index_path(ref.logical_id)
        if not index_path.exists():
            return
        existing = json.loads(index_path.read_text(encoding="utf-8"))
        if existing["sha256"] != ref.sha256:
            raise ArtifactStoreError(
                f"logical id {ref.logical_id!r} already points to a different hash"
            )

    def _record_logical_id(self, ref: ArtifactRef) -> None:
        index_path = self._logical_index_path(ref.logical_id)
        if index_path.exists():
            return
        fd, temp_name = tempfile.mkstemp(
            prefix=".logical.", suffix=".tmp", dir=str(index_path.parent)
        )
        temp_path = Path(temp_name)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(canonical_json_bytes(ref.to_wire()))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, index_path)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def verify(
        self,
        ref_or_uri: ArtifactRef | str,
        *,
        expected_size_bytes: int | None = None,
    ) -> ObjectVerification:
        if isinstance(ref_or_uri, ArtifactRef):
            uri = ref_or_uri.store_uri
            expected_sha = ref_or_uri.sha256
            expected_size = ref_or_uri.size_bytes
        else:
            uri = ref_or_uri
            expected_sha = parse_artifact_uri(uri)
            expected_size = expected_size_bytes
        path = self.object_path(uri)
        if not path.exists():
            return ObjectVerification(
                store_uri=uri,
                ok=False,
                expected_sha256=expected_sha,
                actual_sha256=None,
                expected_size_bytes=expected_size,
                actual_size_bytes=None,
                reason="missing",
            )
        data = path.read_bytes()
        actual_sha = sha256_hex(data)
        actual_size = len(data)
        if actual_sha != expected_sha:
            return ObjectVerification(
                store_uri=uri,
                ok=False,
                expected_sha256=expected_sha,
                actual_sha256=actual_sha,
                expected_size_bytes=expected_size,
                actual_size_bytes=actual_size,
                reason="sha256 mismatch",
            )
        if expected_size is not None and actual_size != expected_size:
            return ObjectVerification(
                store_uri=uri,
                ok=False,
                expected_sha256=expected_sha,
                actual_sha256=actual_sha,
                expected_size_bytes=expected_size,
                actual_size_bytes=actual_size,
                reason="size mismatch",
            )
        return ObjectVerification(
            store_uri=uri,
            ok=True,
            expected_sha256=expected_sha,
            actual_sha256=actual_sha,
            expected_size_bytes=expected_size,
            actual_size_bytes=actual_size,
        )

    def materialize(self, uri: str, relative_destination: str | Path) -> Path:
        source = self.object_path(uri)
        if not source.exists():
            raise ArtifactStoreError(f"object is missing: {uri}")
        destination = _resolve_under(self.materialized_root, relative_destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
        )
        temp_path = Path(temp_name)
        try:
            with os.fdopen(fd, "wb") as handle:
                shutil.copyfileobj(source.open("rb"), handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, destination)
        finally:
            if temp_path.exists():
                temp_path.unlink()
        verification = self.verify(uri)
        copied = sha256_hex(destination.read_bytes())
        if not verification.ok or copied != verification.expected_sha256:
            raise ArtifactStoreError("materialized object failed verification")
        return destination

    def list_objects(self) -> list[ObjectRecord]:
        records: list[ObjectRecord] = []
        if not self.objects_root.exists():
            return records
        for path in self.objects_root.glob("*/*"):
            if not path.is_file() or not is_sha256_hex(path.name):
                continue
            records.append(
                ObjectRecord(
                    sha256=path.name,
                    size_bytes=path.stat().st_size,
                    store_uri=artifact_uri(path.name),
                    path=path,
                )
            )
        return sorted(records, key=lambda record: record.sha256)

    def audit_manifest(
        self,
        manifest: Mapping[str, Any],
        *,
        now_utc: str | None = None,
    ) -> StoreAuditReport:
        now = _parse_utc(now_utc) if now_utc is not None else datetime.now(timezone.utc)
        missing: list[str] = []
        corrupt: list[str] = []
        retention_due: list[str] = []

        for entry in iter_manifest_object_entries(manifest):
            if entry.get("export_included") is False:
                continue
            ref = ArtifactRef(
                logical_id=entry["logical_id"],
                sha256=entry["sha256"],
                size_bytes=entry["size_bytes"],
                media_type=entry["media_type"],
                store_uri=entry["store_uri"],
            )
            verification = self.verify(ref)
            if verification.reason == "missing":
                missing.append(ref.store_uri)
            elif not verification.ok:
                corrupt.append(ref.store_uri)
            retention_until = entry.get("retention_until_utc")
            if isinstance(retention_until, str) and _parse_utc(retention_until) <= now:
                retention_due.append(ref.store_uri)

        return StoreAuditReport(
            missing_objects=tuple(sorted(set(missing))),
            corrupt_objects=tuple(sorted(set(corrupt))),
            retention_due_objects=tuple(sorted(set(retention_due))),
        )


def iter_manifest_object_entries(manifest: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    objects = manifest.get("objects", ())
    if not isinstance(objects, list):
        raise ArtifactStoreError("manifest objects must be a list")
    for entry in objects:
        if not isinstance(entry, Mapping):
            raise ArtifactStoreError("manifest object entry must be an object")
        yield entry
