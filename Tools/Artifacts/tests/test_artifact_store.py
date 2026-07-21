from pathlib import Path

import pytest

from reinbalance_survivors_contracts import canonical_hash
from reinbalance_survivors_contracts.artifact_identity import (
    ArtifactDescriptor,
    ArtifactRef,
    artifact_uri,
)
from Tools.Artifacts.artifact_bundle import create_bundle_manifest
from Tools.Artifacts.artifact_store import ArtifactStore, ArtifactStoreError


def _object_ref(logical_id, sha256, size_bytes, media_type):
    return ArtifactRef(
        logical_id=logical_id,
        sha256=sha256,
        size_bytes=size_bytes,
        media_type=media_type,
        store_uri=artifact_uri(sha256),
    )


def _descriptor(*refs):
    return ArtifactDescriptor(
        logical_id="audit-source",
        node_kind="source_descriptor",
        producer_id="audit-test",
        producer_version="v1",
        identity_metadata={"stable_config_hash": "1" * 64},
        parents=(),
        files=refs,
    )


def _with_recomputed_manifest_hash(manifest):
    manifest["manifest_hash"] = canonical_hash(
        {key: value for key, value in manifest.items() if key != "manifest_hash"}
    )
    return manifest


def test_put_verify_list_and_duplicate_logical_id_guard(tmp_path):
    store = ArtifactStore(tmp_path / "store")
    source = tmp_path / "model.onnx"
    source.write_bytes(b"model-bytes")

    ref = store.put(
        logical_id="models/item-selector.onnx",
        source_path=source,
        media_type="application/onnx",
    )
    duplicate = store.put(
        logical_id="models/item-selector.onnx",
        source_path=source,
        media_type="application/onnx",
    )

    assert duplicate == ref
    assert store.verify(ref).ok
    assert [record.sha256 for record in store.list_objects()] == [ref.sha256]

    changed = tmp_path / "changed.onnx"
    changed.write_bytes(b"different-model-bytes")
    with pytest.raises(ArtifactStoreError):
        store.put(
            logical_id="models/item-selector.onnx",
            source_path=changed,
            media_type="application/onnx",
        )


def test_materialize_rejects_root_escape(tmp_path):
    store = ArtifactStore(tmp_path / "store")
    ref = store.put_bytes(
        logical_id="datasets/choices.jsonl",
        data=b"choice-data",
        media_type="application/jsonl",
    )

    restored = store.materialize(ref.store_uri, "restored/choices.jsonl")

    assert restored == store.root / "materialized" / "restored" / "choices.jsonl"
    assert restored.read_bytes() == b"choice-data"
    with pytest.raises(ArtifactStoreError):
        store.materialize(ref.store_uri, "../escape.jsonl")


def test_manifest_audit_reports_missing_corrupt_and_retention_due(tmp_path):
    store = ArtifactStore(tmp_path / "store")
    ref = store.put_bytes(
        logical_id="bundles/runtime.zip",
        data=b"bundle-data",
        media_type="application/zip",
    )
    missing_hash = "9" * 64
    missing_ref = _object_ref(
        "missing.bin",
        missing_hash,
        10,
        "application/octet-stream",
    )
    manifest = create_bundle_manifest(
        descriptors=[_descriptor(ref, missing_ref)],
        export_id="audit-missing-corrupt-retention",
        retention_until_utc="2026-01-01T00:00:00Z",
        include_private=True,
    )

    store.object_path(ref.store_uri).write_bytes(b"corrupt")
    report = store.audit_manifest(manifest, now_utc="2026-07-21T00:00:00Z")

    assert ref.store_uri in report.corrupt_objects
    assert artifact_uri(missing_hash) in report.missing_objects
    assert ref.store_uri in report.retention_due_objects


def test_manifest_audit_includes_private_objects_excluded_from_public_export(tmp_path):
    store = ArtifactStore(tmp_path / "store")
    private_ref = store.put_bytes(
        logical_id="evidence/private-frame.png",
        data=b"private-frame-data",
        media_type="image/png",
    )
    missing_hash = "8" * 64
    missing_ref = _object_ref(
        "evidence/missing-private-video.mp4",
        missing_hash,
        10,
        "video/mp4",
    )
    manifest = create_bundle_manifest(
        descriptors=[_descriptor(private_ref, missing_ref)],
        export_id="audit-private-excluded",
        privacy_by_logical_id={
            private_ref.logical_id: "private",
            missing_ref.logical_id: "private",
        },
        retention_until_utc="2026-01-01T00:00:00Z",
    )

    store.object_path(private_ref.store_uri).write_bytes(b"corrupt-private-frame")
    report = store.audit_manifest(manifest, now_utc="2026-07-21T00:00:00Z")

    assert private_ref.store_uri in report.corrupt_objects
    assert artifact_uri(missing_hash) in report.missing_objects
    assert private_ref.store_uri in report.retention_due_objects
    assert artifact_uri(missing_hash) in report.retention_due_objects


def test_manifest_audit_rejects_manifest_hash_mismatch(tmp_path):
    store = ArtifactStore(tmp_path / "store")
    ref = store.put_bytes(
        logical_id="bundles/runtime.zip",
        data=b"bundle-data",
        media_type="application/zip",
    )
    manifest = create_bundle_manifest(
        descriptors=[_descriptor(ref)],
        export_id="audit-hash-mismatch",
        retention_until_utc="2027-07-21T00:00:00Z",
        include_private=True,
    )
    manifest["export_id"] = "tampered-export-id"

    with pytest.raises(ArtifactStoreError, match="manifest_hash"):
        store.audit_manifest(manifest, now_utc="2026-07-21T00:00:00Z")


def test_manifest_audit_rejects_descriptor_object_removed_from_manifest(tmp_path):
    store = ArtifactStore(tmp_path / "store")
    ref = store.put_bytes(
        logical_id="bundles/runtime.zip",
        data=b"bundle-data",
        media_type="application/zip",
    )
    manifest = create_bundle_manifest(
        descriptors=[_descriptor(ref)],
        export_id="audit-missing-descriptor-object",
        retention_until_utc="2027-07-21T00:00:00Z",
        include_private=True,
    )
    manifest["objects"] = []
    manifest = _with_recomputed_manifest_hash(manifest)

    with pytest.raises(ArtifactStoreError, match="missing from bundle objects"):
        store.audit_manifest(manifest, now_utc="2026-07-21T00:00:00Z")
