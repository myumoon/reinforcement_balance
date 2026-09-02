"""Content-addressed artifact object store for Survivors release assets."""

from __future__ import annotations

import ctypes
import json
import os
import shutil
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from reinbalance_survivors_contracts import (
    ContractValidationError,
    canonical_hash,
    canonical_json_bytes,
    sha256_hex,
)
from reinbalance_survivors_contracts.artifact_identity import (
    ArtifactDescriptor,
    ArtifactRef,
    artifact_uri,
    is_sha256_hex,
    parse_artifact_uri,
)

try:
    import fcntl
except ImportError:  # pragma: no cover - exercised on Windows
    fcntl = None
    import msvcrt


_THREAD_LOCKS_GUARD = threading.Lock()


class _ThreadLockEntry:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.users = 0


_THREAD_LOCKS: dict[str, _ThreadLockEntry] = {}
_MOVEFILE_REPLACE_EXISTING = 0x1
_MOVEFILE_WRITE_THROUGH = 0x8


class ArtifactStoreError(RuntimeError):
    """Raised when the artifact store cannot safely complete an operation."""


@contextmanager
def _thread_lock(path: Path):
    lock_key = str(path.resolve())
    with _THREAD_LOCKS_GUARD:
        entry = _THREAD_LOCKS.get(lock_key)
        if entry is None:
            entry = _ThreadLockEntry()
            _THREAD_LOCKS[lock_key] = entry
        entry.users += 1

    try:
        with entry.lock:
            yield
    finally:
        with _THREAD_LOCKS_GUARD:
            entry.users -= 1
            if entry.users == 0 and _THREAD_LOCKS.get(lock_key) is entry:
                del _THREAD_LOCKS[lock_key]


@contextmanager
def _exclusive_file_lock(path: Path):
    with _thread_lock(path):
        with path.open("a+b") as stream:
            if fcntl is not None:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            else:
                if os.fstat(stream.fileno()).st_size == 0:
                    stream.write(b"\0")
                    stream.flush()
                    os.fsync(stream.fileno())
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
                else:
                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def _durable_replace(
    source: Path,
    destination: Path,
    *,
    platform: str | None = None,
) -> None:
    """Atomically publish one fsynced file and durably commit its directory entry."""
    platform = os.name if platform is None else platform
    if platform == "nt":
        move_file_ex = ctypes.WinDLL("kernel32", use_last_error=True).MoveFileExW
        move_file_ex.argtypes = [
            ctypes.c_wchar_p,
            ctypes.c_wchar_p,
            ctypes.c_uint32,
        ]
        move_file_ex.restype = ctypes.c_int
        flags = _MOVEFILE_REPLACE_EXISTING | _MOVEFILE_WRITE_THROUGH
        if not move_file_ex(str(source), str(destination), flags):
            raise ctypes.WinError(ctypes.get_last_error())
        return
    if platform == "posix":
        os.replace(source, destination)
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        descriptor = os.open(destination.parent, flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return
    raise OSError(f"unsupported durability platform: {platform}")


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

    @staticmethod
    def _strip_win_prefix(path: Path) -> Path:
        """Windows 拡張長パスの \\?\ プレフィックスを正規化して比較できるようにする。"""
        s = str(path)
        return Path(s[4:]) if s.startswith("\\\\?\\") else path

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
            # \\?\ プレフィックスを両辺で除去してから比較する（Windows 拡張長パス対策）
            self._strip_win_prefix(path).relative_to(self._strip_win_prefix(self.objects_root.resolve()))
        except ValueError as exc:
            raise ArtifactStoreError("object path escapes store root") from exc
        return path

    def _logical_index_path(self, logical_id: str) -> Path:
        key = sha256_hex(logical_id.encode("utf-8"))
        return self.logical_root / f"{key}.json"

    def _logical_lock_path(self, logical_id: str) -> Path:
        key = sha256_hex(logical_id.encode("utf-8"))
        return self.logical_root / f"{key}.lock"

    def _read_logical_index(self, index_path: Path) -> dict[str, Any]:
        def reject_duplicate_keys(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate key: {key}")
                result[key] = value
            return result

        try:
            existing = json.loads(
                index_path.read_text(encoding="utf-8"),
                object_pairs_hook=reject_duplicate_keys,
            )
            ArtifactRef.from_wire(existing)
        except (
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            ValueError,
            TypeError,
        ) as exc:
            raise ArtifactStoreError(
                f"invalid logical id index: {index_path}"
            ) from exc
        return existing

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
        self._publish_object_bytes(ref, data)
        self._record_logical_id(ref)
        return ref

    def put_bytes_create_once(
        self, *, logical_id: str, data: bytes, media_type: str
    ) -> ArtifactRef:
        """logical id を lock 下で一度だけ atomic publish する。

        通常の ``put_bytes`` は同一内容の idempotent 再 publish を許す。lineage
        open marker のように二回目そのものを拒否する用途だけ、この境界を使う。
        object publish と logical index commit の間は同じ logical-id lock を保持する。
        """
        if not isinstance(data, bytes):
            raise ArtifactStoreError("put_bytes_create_once data must be bytes")
        sha256 = sha256_hex(data)
        ref = ArtifactRef(
            logical_id=logical_id,
            sha256=sha256,
            size_bytes=len(data),
            media_type=media_type,
            store_uri=artifact_uri(sha256),
        )
        index_path = self._logical_index_path(logical_id)
        with _exclusive_file_lock(self._logical_lock_path(logical_id)):
            if index_path.exists():
                # 破損 index も再利用せず fail-closed にする。
                self._read_logical_index(index_path)
                raise ArtifactStoreError(f"logical id {logical_id!r} already exists")
            self._publish_object_bytes(ref, data)
            self._record_logical_id_unlocked(ref, index_path)
        return ref

    def _publish_object_bytes(self, ref: ArtifactRef, data: bytes) -> None:
        """content object を temp+fsync+replace で公開し、公開後 hash を再検証する。"""
        sha256 = ref.sha256
        destination = self.object_path(ref.store_uri)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with _thread_lock(destination):  # 同一 SHA への並行書込みを直列化する
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
                    _durable_replace(temp_path, destination)
                finally:
                    if temp_path.exists():
                        temp_path.unlink()
        verification = self.verify(ref)
        if not verification.ok:
            raise ArtifactStoreError(
                f"published object {ref.store_uri} failed revalidation: {verification.reason}"
            )

    def _ensure_logical_id_available(self, ref: ArtifactRef) -> None:
        index_path = self._logical_index_path(ref.logical_id)
        with _exclusive_file_lock(self._logical_lock_path(ref.logical_id)):
            if not index_path.exists():
                return
            existing = self._read_logical_index(index_path)
            if existing != ref.to_wire():
                raise ArtifactStoreError(
                    f"logical id {ref.logical_id!r} already points to different metadata"
                )

    def _record_logical_id(self, ref: ArtifactRef) -> None:
        index_path = self._logical_index_path(ref.logical_id)
        expected = ref.to_wire()
        with _exclusive_file_lock(self._logical_lock_path(ref.logical_id)):
            if index_path.exists():
                existing = self._read_logical_index(index_path)
                if existing == expected:
                    return
                raise ArtifactStoreError(
                    f"logical id {ref.logical_id!r} already points to different metadata"
                )

            self._record_logical_id_unlocked(ref, index_path)

    def _record_logical_id_unlocked(
        self, ref: ArtifactRef, index_path: Path
    ) -> None:
        """呼出側が logical-id lock を保持した状態で index を durable commit する。"""
        expected = ref.to_wire()

        fd, temp_name = tempfile.mkstemp(
            prefix=".logical.", suffix=".tmp", dir=str(index_path.parent)
        )
        temp_path = Path(temp_name)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(canonical_json_bytes(expected))
                handle.flush()
                os.fsync(handle.fileno())
            _durable_replace(temp_path, index_path)
            if self._read_logical_index(index_path) != expected:
                raise ArtifactStoreError(
                    f"logical id {ref.logical_id!r} publish revalidation failed"
                )
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
        verify_manifest_hash(manifest)
        validate_manifest_descriptor_object_refs(manifest)
        now = _parse_utc(now_utc) if now_utc is not None else datetime.now(timezone.utc)
        missing: list[str] = []
        corrupt: list[str] = []
        retention_due: list[str] = []

        for entry in iter_manifest_object_entries(manifest):
            ref = object_ref_from_manifest_entry(
                entry,
                label="manifest object entry",
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


def manifest_core(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in manifest.items() if key != "manifest_hash"}


def verify_manifest_hash(manifest: Mapping[str, Any]) -> str:
    expected = manifest.get("manifest_hash")
    if not isinstance(expected, str):
        raise ArtifactStoreError("manifest_hash is required")
    actual = canonical_hash(manifest_core(manifest))
    if actual != expected:
        raise ArtifactStoreError("manifest_hash does not match manifest content")
    return expected


def iter_manifest_object_entries(manifest: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    objects = manifest.get("objects", ())
    if not isinstance(objects, list):
        raise ArtifactStoreError("manifest objects must be a list")
    for entry in objects:
        if not isinstance(entry, Mapping):
            raise ArtifactStoreError("manifest object entry must be an object")
        yield entry


def object_ref_from_manifest_entry(entry: Mapping[str, Any], *, label: str) -> ArtifactRef:
    try:
        return ArtifactRef(
            logical_id=entry["logical_id"],
            sha256=entry["sha256"],
            size_bytes=entry["size_bytes"],
            media_type=entry["media_type"],
            store_uri=entry["store_uri"],
            schema_version=entry["schema_version"],
        )
    except (KeyError, ContractValidationError) as exc:
        raise ArtifactStoreError(f"{label} has invalid artifact ref metadata") from exc


def manifest_object_refs_by_logical_id(
    manifest: Mapping[str, Any],
) -> dict[str, ArtifactRef]:
    refs: dict[str, ArtifactRef] = {}
    for index, entry in enumerate(iter_manifest_object_entries(manifest)):
        ref = object_ref_from_manifest_entry(entry, label=f"manifest objects[{index}]")
        existing = refs.get(ref.logical_id)
        if existing is not None and existing.to_wire() != ref.to_wire():
            raise ArtifactStoreError(
                f"bundle logical id {ref.logical_id!r} has conflicting object metadata"
            )
        refs[ref.logical_id] = ref
    return refs


def manifest_descriptor_entries(manifest: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    descriptors = manifest.get("descriptors", ())
    if not isinstance(descriptors, list):
        raise ArtifactStoreError("manifest descriptors must be a list")
    for descriptor in descriptors:
        if not isinstance(descriptor, Mapping):
            raise ArtifactStoreError("manifest descriptor entry must be an object")
    return descriptors


def validate_manifest_descriptor_object_refs(manifest: Mapping[str, Any]) -> None:
    object_refs = manifest_object_refs_by_logical_id(manifest)
    descriptor_file_refs: dict[str, ArtifactRef] = {}
    for index, descriptor_entry in enumerate(manifest_descriptor_entries(manifest)):
        try:
            descriptor = ArtifactDescriptor.from_wire(descriptor_entry)
        except ContractValidationError as exc:
            raise ArtifactStoreError(
                f"manifest descriptors[{index}] has invalid descriptor metadata"
            ) from exc
        for file_ref in descriptor.files:
            existing_file_ref = descriptor_file_refs.get(file_ref.logical_id)
            if (
                existing_file_ref is not None
                and existing_file_ref.to_wire() != file_ref.to_wire()
            ):
                raise ArtifactStoreError(
                    f"descriptor file {file_ref.logical_id!r} has conflicting metadata"
                )
            descriptor_file_refs[file_ref.logical_id] = file_ref
            object_ref = object_refs.get(file_ref.logical_id)
            if object_ref is None:
                raise ArtifactStoreError(
                    f"descriptor file {file_ref.logical_id!r} is missing from bundle objects"
                )
            if object_ref.to_wire() != file_ref.to_wire():
                raise ArtifactStoreError(
                    f"descriptor file {file_ref.logical_id!r} metadata mismatch with bundle objects"
                )
    for logical_id in sorted(set(object_refs) - set(descriptor_file_refs)):
        raise ArtifactStoreError(
            f"manifest object {logical_id!r} is not referenced by descriptor files"
        )
