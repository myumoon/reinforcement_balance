"""descriptor の object closure を別 store へ復元して検証する。

共有 ArtifactStore と RestoreTestVerdict の wire 契約をそのまま使います。
"""

from __future__ import annotations

from pathlib import Path

from reinbalance_survivors_contracts.artifact_identity import (
    ArtifactDescriptor,
    RestoreTestVerdict,
)
from reinbalance_survivors_contracts.artifact_store import ArtifactStore, ArtifactStoreError
from reinbalance_survivors_contracts.canonical_json import canonical_json_bytes


def _empty_root(value: str | Path, label: str) -> Path:
    """新しい restore store に使える空 directory を検証する。

既存データを削除せず、再利用 root の混入を fail-closed で拒否します。
    """
    root = Path(value).expanduser().resolve()
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise ValueError(f"{label} must be empty")
    return root


def _overlaps(left: Path, right: Path) -> bool:
    """二つの store root が同一または親子関係かを返す。

復元先が元 store 内にできて自己コピーや既存 object の誤認を起こす構成を避けます。
    """
    return left == right or left in right.parents or right in left.parents


def _manifest_ref(descriptor: ArtifactDescriptor):
    """descriptor の manifest object を identity_metadata から特定する。

logical id が無い互換 descriptor では JSON file がただ一つの場合だけ推定します。
    """
    manifest_id = descriptor.identity_metadata.get("manifest_logical_id")
    if manifest_id is not None:
        matches = [ref for ref in descriptor.files if ref.logical_id == manifest_id]
        if len(matches) != 1:
            raise ValueError("descriptor manifest_logical_id does not identify one file")
        return matches[0]
    json_files = [ref for ref in descriptor.files if ref.media_type == "application/json"]
    if len(json_files) != 1:
        raise ValueError("descriptor must identify exactly one JSON manifest")
    return json_files[0]


def run_restore_test(
    primary: ArtifactStore,
    descriptor: ArtifactDescriptor,
    *,
    backup_root: str | Path,
    restore_root: str | Path,
    verdict_logical_id: str,
) -> tuple[RestoreTestVerdict, ArtifactDescriptor]:
    """descriptor の全 file を backup 経由で空 store に復元して verdict を発行する。

復元 object を一件ずつ hash 検証し、結果 descriptor JSON を primary store に保存します。
    """
    if not isinstance(primary, ArtifactStore) or not isinstance(descriptor, ArtifactDescriptor):
        raise TypeError("primary and descriptor must use the shared artifact contracts")
    if not descriptor.files:
        raise ValueError("restore test requires descriptor files")
    manifest = _manifest_ref(descriptor)
    backup_path = _empty_root(backup_root, "backup_root")
    restore_path = _empty_root(restore_root, "restore_root")
    roots = (primary.root.resolve(), backup_path, restore_path)
    if any(_overlaps(roots[i], roots[j]) for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("primary, backup, and restore roots must be disjoint")

    backup = ArtifactStore(backup_path)
    restored = ArtifactStore(restore_path)
    failures: list[str] = []
    for ref in descriptor.files:
        source_check = primary.verify(ref)
        if not source_check.ok:
            failures.append(f"primary:{ref.logical_id}:{source_check.reason}")
            continue
        try:
            content = primary.object_path(ref.store_uri).read_bytes()
            backup_ref = backup.put_bytes(
                logical_id=ref.logical_id, data=content, media_type=ref.media_type
            )
            backup_check = backup.verify(backup_ref)
            if not backup_check.ok:
                failures.append(f"backup:{ref.logical_id}:{backup_check.reason}")
                continue
            restored_ref = restored.put_bytes(
                logical_id=ref.logical_id,
                data=backup.object_path(backup_ref.store_uri).read_bytes(),
                media_type=ref.media_type,
            )
            restored_check = restored.verify(restored_ref)
            if not restored_check.ok or restored_ref != ref:
                reason = restored_check.reason or "artifact reference mismatch"
                failures.append(f"restore:{ref.logical_id}:{reason}")
        except (OSError, ArtifactStoreError, ValueError) as exc:
            failures.append(f"restore:{ref.logical_id}:{type(exc).__name__}")

    verdict = RestoreTestVerdict(
        logical_id=verdict_logical_id,
        subject=descriptor.node_ref(),
        manifest_hash=manifest.sha256,
        primary_root=str(primary.root),
        backup_root=str(backup_path),
        verify_mode="full",
        checked_object_count=len(descriptor.files),
        passed=not failures,
        blocking_reasons=tuple(failures),
    )
    verdict_descriptor = verdict.to_descriptor()
    primary.put_bytes(
        logical_id=verdict_logical_id,
        data=canonical_json_bytes(verdict_descriptor.to_wire()),
        media_type="application/json",
    )
    return verdict, verdict_descriptor
