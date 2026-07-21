from pathlib import Path

import pytest

from reinbalance_survivors_contracts.artifact_identity import artifact_uri
from Tools.Artifacts.artifact_store import ArtifactStore, ArtifactStoreError


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
    manifest = {
        "schema_version": "artifact_bundle_manifest.v1",
        "objects": [
            {
                "logical_id": ref.logical_id,
                "sha256": ref.sha256,
                "size_bytes": ref.size_bytes,
                "media_type": ref.media_type,
                "store_uri": ref.store_uri,
                "retention_until_utc": "2026-01-01T00:00:00Z",
                "export_included": True,
            },
            {
                "logical_id": "missing.bin",
                "sha256": missing_hash,
                "size_bytes": 10,
                "media_type": "application/octet-stream",
                "store_uri": artifact_uri(missing_hash),
                "export_included": True,
            },
        ],
    }

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
    manifest = {
        "schema_version": "artifact_bundle_manifest.v1",
        "objects": [
            {
                "logical_id": private_ref.logical_id,
                "sha256": private_ref.sha256,
                "size_bytes": private_ref.size_bytes,
                "media_type": private_ref.media_type,
                "store_uri": private_ref.store_uri,
                "privacy_classification": "private",
                "retention_until_utc": "2026-01-01T00:00:00Z",
                "export_included": False,
            },
            {
                "logical_id": "evidence/missing-private-video.mp4",
                "sha256": missing_hash,
                "size_bytes": 10,
                "media_type": "video/mp4",
                "store_uri": artifact_uri(missing_hash),
                "privacy_classification": "private",
                "retention_until_utc": "2026-01-01T00:00:00Z",
                "export_included": False,
            },
        ],
    }

    store.object_path(private_ref.store_uri).write_bytes(b"corrupt-private-frame")
    report = store.audit_manifest(manifest, now_utc="2026-07-21T00:00:00Z")

    assert private_ref.store_uri in report.corrupt_objects
    assert artifact_uri(missing_hash) in report.missing_objects
    assert private_ref.store_uri in report.retention_due_objects
    assert artifact_uri(missing_hash) in report.retention_due_objects
