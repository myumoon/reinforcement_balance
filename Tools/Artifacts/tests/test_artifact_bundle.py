import zipfile
from copy import deepcopy
from pathlib import Path

import pytest

from reinbalance_survivors_contracts import canonical_hash, canonical_json_bytes
from reinbalance_survivors_contracts.artifact_identity import (
    ArtifactDescriptor,
    RestoreTestVerdict,
)
from reinbalance_survivors_contracts.artifact_dag import ArtifactDagValidationError
from Tools.Artifacts.artifact_bundle import (
    VolumeIdentity,
    assert_distinct_store_roots,
    create_bundle_manifest,
    export_bundle,
    import_bundle,
    verify_bundle_objects,
)
from Tools.Artifacts.artifact_store import ArtifactStore
from Tools.Artifacts.artifact_store import ArtifactStoreError


def _store_snapshot(store: ArtifactStore):
    object_hashes = tuple(record.sha256 for record in store.list_objects())
    logical_index = {
        path.name: path.read_bytes()
        for path in sorted(store.logical_root.glob("*.json"))
    }
    return object_hashes, logical_index


def _descriptor(ref):
    return ArtifactDescriptor(
        logical_id="runtime-bundle",
        node_kind="source_descriptor",
        producer_id="bundle-builder",
        producer_version="v1",
        identity_metadata={"bundle_config_hash": "1" * 64},
        parents=(),
        files=(ref,),
        non_identity_metadata={
            "created_at_utc": "2026-07-21T00:00:00Z",
            "local_path": "/private/absolute/path/runtime.zip",
        },
    )


def _source_descriptor(ref):
    return ArtifactDescriptor(
        logical_id="source/model",
        node_kind="source_descriptor",
        producer_id="source-builder",
        producer_version="v1",
        identity_metadata={"config_hash": "0" * 64},
        parents=(),
        files=(ref,),
    )


def _verdict_descriptor(source):
    return ArtifactDescriptor(
        logical_id="source/model.validation",
        node_kind="teacher_validation_verdict",
        producer_id="validator",
        producer_version="v1",
        identity_metadata={"gate_version": "v1", "passed": True},
        parents=(source.node_ref(),),
        files=(),
    )


def _release_evidence_lineage(*, include_restore_parent=True):
    source = ArtifactDescriptor(
        logical_id="phase5-source",
        node_kind="source_descriptor",
        producer_id="source-builder",
        producer_version="v1",
        identity_metadata={"source_config_hash": "0" * 64},
        parents=(),
        files=(),
    )
    verdict = ArtifactDescriptor(
        logical_id="teacher-verdict",
        node_kind="teacher_validation_verdict",
        producer_id="validator",
        producer_version="v1",
        identity_metadata={"gate_version": "teacher-gate.v1", "passed": True},
        parents=(source.node_ref(),),
        files=(),
    )
    dataset = ArtifactDescriptor(
        logical_id="choice-dataset",
        node_kind="choice_dataset_release",
        producer_id="dataset-builder",
        producer_version="v1",
        identity_metadata={"dataset_config_hash": "1" * 64},
        parents=(verdict.node_ref(),),
        files=(),
    )
    model = ArtifactDescriptor(
        logical_id="item-selector",
        node_kind="item_selector_release",
        producer_id="model-builder",
        producer_version="v1",
        identity_metadata={"model_config_hash": "2" * 64},
        parents=(dataset.node_ref(),),
        files=(),
    )
    bundle = ArtifactDescriptor(
        logical_id="runtime-bundle",
        node_kind="runtime_bundle",
        producer_id="bundle-builder",
        producer_version="v1",
        identity_metadata={"bundle_config_hash": "3" * 64},
        parents=(model.node_ref(),),
        files=(),
    )
    shadow = ArtifactDescriptor(
        logical_id="shadow-verdict",
        node_kind="replay_shadow_verdict",
        producer_id="shadow-runner",
        producer_version="v1",
        identity_metadata={"gate_version": "shadow-gate.v1", "passed": True},
        parents=(bundle.node_ref(),),
        files=(),
    )
    campaign = ArtifactDescriptor(
        logical_id="canary-campaign",
        node_kind="canary_campaign",
        producer_id="campaign-runner",
        producer_version="v1",
        identity_metadata={"campaign_id": "c0", "passed": True},
        parents=(shadow.node_ref(),),
        files=(),
    )
    restore = ArtifactDescriptor(
        logical_id="restore-test",
        node_kind="restore_test_verdict",
        producer_id="artifact-restore-test",
        producer_version="restore_test_verdict.v1",
        identity_metadata={"manifest_hash": "4" * 64, "passed": True},
        parents=(bundle.node_ref(),),
        files=(),
    )
    evidence_parents = (
        (campaign.node_ref(), restore.node_ref())
        if include_restore_parent
        else (campaign.node_ref(),)
    )
    evidence = ArtifactDescriptor(
        logical_id="goal-evidence",
        node_kind="goal_evidence",
        producer_id="goal-evidence-builder",
        producer_version="v1",
        identity_metadata={"goal_id": "survivors-phase5", "passed": True},
        parents=evidence_parents,
        files=(),
    )
    return (source, verdict, dataset, model, bundle, shadow, campaign, restore, evidence)


def _with_recomputed_manifest_hash(manifest):
    updated = deepcopy(manifest)
    updated["manifest_hash"] = canonical_hash(
        {key: value for key, value in updated.items() if key != "manifest_hash"}
    )
    return updated


def _write_bundle(zip_path, manifest, objects):
    with zipfile.ZipFile(zip_path, "w", allowZip64=True) as zf:
        zf.writestr("manifest.json", canonical_json_bytes(manifest))
        for ref, data in objects:
            zf.writestr(f"objects/sha256/{ref.sha256}", data)


def test_deterministic_export_import_and_full_verify(tmp_path):
    primary = ArtifactStore(tmp_path / "primary")
    backup = ArtifactStore(tmp_path / "backup")
    ref = primary.put_bytes(
        logical_id="bundles/runtime.zip",
        data=b"runtime-bundle",
        media_type="application/zip",
    )
    manifest = create_bundle_manifest(
        descriptors=[_descriptor(ref)],
        export_id="survivors-runtime-bundle",
        license_by_logical_id={ref.logical_id: "project-private"},
        privacy_by_logical_id={ref.logical_id: "internal"},
        retention_until_utc="2027-07-21T00:00:00Z",
        include_private=True,
    )

    zip_a = tmp_path / "bundle-a.zip"
    zip_b = tmp_path / "bundle-b.zip"
    export_bundle(primary, manifest, zip_a)
    export_bundle(primary, manifest, zip_b)

    assert zip_a.read_bytes() == zip_b.read_bytes()
    imported = import_bundle(zip_a, backup, verify_mode="full")
    report = verify_bundle_objects(backup, imported.manifest, verify_mode="full")
    assert imported.manifest_hash == manifest["manifest_hash"]
    assert report.ok


def test_import_corrupt_bundle_does_not_mutate_target_store(tmp_path):
    primary = ArtifactStore(tmp_path / "primary")
    target = ArtifactStore(tmp_path / "target")
    existing = target.put_bytes(
        logical_id="models/existing.onnx",
        data=b"existing-model",
        media_type="application/onnx",
    )
    ref = primary.put_bytes(
        logical_id="bundles/runtime.zip",
        data=b"runtime-bundle",
        media_type="application/zip",
    )
    manifest = create_bundle_manifest(
        descriptors=[_descriptor(ref)],
        export_id="corrupt-runtime-bundle",
        license_by_logical_id={ref.logical_id: "project-private"},
        privacy_by_logical_id={ref.logical_id: "internal"},
        retention_until_utc="2027-07-21T00:00:00Z",
        include_private=True,
    )
    zip_path = tmp_path / "corrupt.zip"
    with zipfile.ZipFile(zip_path, "w", allowZip64=True) as zf:
        zf.writestr("manifest.json", canonical_json_bytes(manifest))
        zf.writestr(f"objects/sha256/{ref.sha256}", b"corrupt-runtime-bundle")
    before = _store_snapshot(target)

    with pytest.raises(ArtifactStoreError, match="sha256 mismatch"):
        import_bundle(zip_path, target, verify_mode="full")

    assert _store_snapshot(target) == before
    assert target.verify(existing).ok


def test_import_rejects_self_consistent_manifest_missing_descriptor_object(tmp_path):
    primary = ArtifactStore(tmp_path / "primary")
    target = ArtifactStore(tmp_path / "target")
    ref = primary.put_bytes(
        logical_id="bundles/runtime.zip",
        data=b"runtime-bundle",
        media_type="application/zip",
    )
    manifest = create_bundle_manifest(
        descriptors=[_descriptor(ref)],
        export_id="missing-descriptor-object",
        license_by_logical_id={ref.logical_id: "project-private"},
        privacy_by_logical_id={ref.logical_id: "internal"},
        retention_until_utc="2027-07-21T00:00:00Z",
        include_private=True,
    )
    manifest["objects"] = []
    manifest = _with_recomputed_manifest_hash(manifest)
    zip_path = tmp_path / "missing-descriptor-object.zip"
    _write_bundle(zip_path, manifest, ())
    before = _store_snapshot(target)

    with pytest.raises(ArtifactStoreError, match="missing from bundle objects"):
        import_bundle(zip_path, target, verify_mode="full")

    assert _store_snapshot(target) == before


def test_verify_rejects_self_consistent_manifest_missing_descriptor_object(tmp_path):
    store = ArtifactStore(tmp_path / "store")
    ref = store.put_bytes(
        logical_id="bundles/runtime.zip",
        data=b"runtime-bundle",
        media_type="application/zip",
    )
    manifest = create_bundle_manifest(
        descriptors=[_descriptor(ref)],
        export_id="verify-missing-descriptor-object",
        license_by_logical_id={ref.logical_id: "project-private"},
        privacy_by_logical_id={ref.logical_id: "internal"},
        retention_until_utc="2027-07-21T00:00:00Z",
        include_private=True,
    )
    manifest["objects"] = []
    manifest = _with_recomputed_manifest_hash(manifest)

    with pytest.raises(ArtifactStoreError, match="missing from bundle objects"):
        verify_bundle_objects(store, manifest, verify_mode="full")


def test_import_rejects_descriptor_object_metadata_mismatch(tmp_path):
    descriptor_store = ArtifactStore(tmp_path / "descriptor")
    alternate_store = ArtifactStore(tmp_path / "alternate")
    target = ArtifactStore(tmp_path / "target")
    ref = descriptor_store.put_bytes(
        logical_id="bundles/runtime.zip",
        data=b"runtime-bundle",
        media_type="application/zip",
    )
    alternate_ref = alternate_store.put_bytes(
        logical_id=ref.logical_id,
        data=b"alternate-runtime-bundle",
        media_type=ref.media_type,
    )
    manifest = create_bundle_manifest(
        descriptors=[_descriptor(ref)],
        export_id="descriptor-object-metadata-mismatch",
        license_by_logical_id={ref.logical_id: "project-private"},
        privacy_by_logical_id={ref.logical_id: "internal"},
        retention_until_utc="2027-07-21T00:00:00Z",
        include_private=True,
    )
    manifest["objects"][0].update(alternate_ref.to_wire())
    manifest = _with_recomputed_manifest_hash(manifest)
    zip_path = tmp_path / "metadata-mismatch.zip"
    _write_bundle(zip_path, manifest, ((alternate_ref, b"alternate-runtime-bundle"),))
    before = _store_snapshot(target)

    with pytest.raises(ArtifactStoreError, match="metadata mismatch"):
        import_bundle(zip_path, target, verify_mode="full")

    assert _store_snapshot(target) == before


def test_verify_rejects_descriptor_object_metadata_mismatch(tmp_path):
    descriptor_store = ArtifactStore(tmp_path / "descriptor")
    store = ArtifactStore(tmp_path / "store")
    ref = descriptor_store.put_bytes(
        logical_id="bundles/runtime.zip",
        data=b"runtime-bundle",
        media_type="application/zip",
    )
    alternate_ref = store.put_bytes(
        logical_id=ref.logical_id,
        data=b"alternate-runtime-bundle",
        media_type=ref.media_type,
    )
    manifest = create_bundle_manifest(
        descriptors=[_descriptor(ref)],
        export_id="verify-descriptor-object-metadata-mismatch",
        license_by_logical_id={ref.logical_id: "project-private"},
        privacy_by_logical_id={ref.logical_id: "internal"},
        retention_until_utc="2027-07-21T00:00:00Z",
        include_private=True,
    )
    manifest["objects"][0].update(alternate_ref.to_wire())
    manifest = _with_recomputed_manifest_hash(manifest)

    with pytest.raises(ArtifactStoreError, match="metadata mismatch"):
        verify_bundle_objects(store, manifest, verify_mode="full")


def test_public_export_excludes_private_video_objects_by_default(tmp_path):
    store = ArtifactStore(tmp_path / "store")
    video = store.put_bytes(
        logical_id="evidence/c0-run.mp4",
        data=b"video-bytes",
        media_type="video/mp4",
    )
    manifest = create_bundle_manifest(
        descriptors=[_descriptor(video)],
        export_id="public-evidence",
        license_by_logical_id={video.logical_id: "not-redistributable"},
        privacy_by_logical_id={video.logical_id: "private"},
        retention_until_utc="2027-07-21T00:00:00Z",
    )
    zip_path = tmp_path / "public.zip"

    export_bundle(store, manifest, zip_path)

    object_entries = [entry for entry in manifest["objects"] if entry["logical_id"] == video.logical_id]
    assert object_entries[0]["export_included"] is False
    with zipfile.ZipFile(zip_path) as zf:
        assert f"objects/sha256/{video.sha256}" not in zf.namelist()
        exported_manifest = zf.read("manifest.json")
    assert b"/private/absolute/path" not in exported_manifest


def test_public_export_rejects_private_object_even_if_export_included_is_tampered(tmp_path):
    store = ArtifactStore(tmp_path / "store")
    video = store.put_bytes(
        logical_id="evidence/c0-run.mp4",
        data=b"video-bytes",
        media_type="video/mp4",
    )
    manifest = create_bundle_manifest(
        descriptors=[_descriptor(video)],
        export_id="tampered-public-evidence",
        license_by_logical_id={video.logical_id: "not-redistributable"},
        privacy_by_logical_id={video.logical_id: "private"},
        retention_until_utc="2027-07-21T00:00:00Z",
    )
    manifest["objects"][0]["export_included"] = True
    manifest = _with_recomputed_manifest_hash(manifest)

    with pytest.raises(ArtifactStoreError, match="private object"):
        export_bundle(store, manifest, tmp_path / "tampered-public.zip")


def test_create_bundle_manifest_rejects_lineage_violation_for_non_root_descriptor(tmp_path):
    store = ArtifactStore(tmp_path / "store")
    ref = store.put_bytes(
        logical_id="datasets/choice.jsonl",
        data=b"dataset",
        media_type="application/jsonl",
    )
    dataset_without_parent = ArtifactDescriptor(
        logical_id="datasets/choice",
        node_kind="choice_dataset_release",
        producer_id="dataset-builder",
        producer_version="v1",
        identity_metadata={"dataset_config_hash": "3" * 64},
        parents=(),
        files=(ref,),
    )

    with pytest.raises(ArtifactDagValidationError, match="must declare at least one parent"):
        create_bundle_manifest(
            descriptors=[dataset_without_parent],
            export_id="bad-lineage",
            retention_until_utc="2027-07-21T00:00:00Z",
            include_private=True,
        )


def test_create_bundle_manifest_rejects_goal_evidence_without_restore_test_verdict():
    descriptors = _release_evidence_lineage(include_restore_parent=False)

    with pytest.raises(ArtifactDagValidationError, match="restore_test_verdict"):
        create_bundle_manifest(
            descriptors=descriptors,
            export_id="goal-evidence-without-restore-test",
            retention_until_utc="2027-07-21T00:00:00Z",
            include_private=True,
        )


def test_bundle_paths_reject_goal_evidence_without_restore_test_verdict(tmp_path):
    primary = ArtifactStore(tmp_path / "primary")
    target = ArtifactStore(tmp_path / "target")
    manifest = create_bundle_manifest(
        descriptors=_release_evidence_lineage(include_restore_parent=True),
        export_id="goal-evidence-with-restore-test",
        retention_until_utc="2027-07-21T00:00:00Z",
        include_private=True,
    )
    manifest["descriptors"] = [
        descriptor.to_wire()
        for descriptor in sorted(
            _release_evidence_lineage(include_restore_parent=False),
            key=lambda value: value.identity_hash,
        )
    ]
    manifest = _with_recomputed_manifest_hash(manifest)

    with pytest.raises(ArtifactDagValidationError, match="restore_test_verdict"):
        export_bundle(primary, manifest, tmp_path / "bad-goal-evidence-export.zip")

    with pytest.raises(ArtifactDagValidationError, match="restore_test_verdict"):
        verify_bundle_objects(primary, manifest, verify_mode="full")

    zip_path = tmp_path / "bad-goal-evidence-import.zip"
    _write_bundle(zip_path, manifest, ())
    with pytest.raises(ArtifactDagValidationError, match="restore_test_verdict"):
        import_bundle(zip_path, target, verify_mode="full")


def test_export_and_import_reject_tampered_manifest_with_lineage_violation(tmp_path):
    primary = ArtifactStore(tmp_path / "primary")
    target = ArtifactStore(tmp_path / "target")
    ref = primary.put_bytes(
        logical_id="models/source.onnx",
        data=b"source-model",
        media_type="application/onnx",
    )
    source = _source_descriptor(ref)
    verdict = _verdict_descriptor(source)
    manifest = create_bundle_manifest(
        descriptors=[source, verdict],
        export_id="tampered-lineage",
        privacy_by_logical_id={ref.logical_id: "internal"},
        retention_until_utc="2027-07-21T00:00:00Z",
        include_private=True,
    )
    tampered_verdict = ArtifactDescriptor(
        logical_id=verdict.logical_id,
        node_kind=verdict.node_kind,
        producer_id=verdict.producer_id,
        producer_version=verdict.producer_version,
        identity_metadata=verdict.identity_metadata,
        parents=(),
        files=verdict.files,
    )
    for index, entry in enumerate(manifest["descriptors"]):
        if entry["identity"]["logical_id"] == verdict.logical_id:
            manifest["descriptors"][index] = tampered_verdict.to_wire()
            break
    manifest = _with_recomputed_manifest_hash(manifest)

    with pytest.raises(ArtifactDagValidationError, match="must declare at least one parent"):
        export_bundle(primary, manifest, tmp_path / "tampered-lineage-export.zip")

    zip_path = tmp_path / "tampered-lineage-import.zip"
    _write_bundle(zip_path, manifest, ((ref, b"source-model"),))
    with pytest.raises(ArtifactDagValidationError, match="must declare at least one parent"):
        import_bundle(zip_path, target, verify_mode="full")


def test_sample_verify_and_distinct_backup_volume_check(tmp_path):
    primary = ArtifactStore(tmp_path / "primary")
    backup = ArtifactStore(tmp_path / "backup")
    ref = primary.put_bytes(
        logical_id="models/combat-student.onnx",
        data=b"student-model",
        media_type="application/onnx",
    )
    manifest = create_bundle_manifest(
        descriptors=[_descriptor(ref)],
        export_id="student-model",
        license_by_logical_id={ref.logical_id: "project-private"},
        privacy_by_logical_id={ref.logical_id: "internal"},
        retention_until_utc="2027-07-21T00:00:00Z",
        include_private=True,
    )
    zip_path = tmp_path / "student.zip"
    export_bundle(primary, manifest, zip_path)
    imported = import_bundle(zip_path, backup, verify_mode="sample", sample_size=1, random_seed=7)

    assert verify_bundle_objects(
        backup,
        imported.manifest,
        verify_mode="sample",
        sample_size=1,
        random_seed=7,
    ).ok

    def resolver(path: Path):
        return VolumeIdentity(resolved_path=path.resolve(), device_id=("backup" if path == backup.root else "primary"))

    assert_distinct_store_roots(primary.root, backup.root, resolver=resolver)
    with pytest.raises(ValueError):
        assert_distinct_store_roots(primary.root, primary.root, resolver=resolver)


def test_restore_test_verdict_descriptor_schema():
    primary = Path("/mnt/store-primary")
    backup = Path("/mnt/store-backup")
    bundle = ArtifactDescriptor(
        logical_id="runtime-bundle",
        node_kind="runtime_bundle",
        producer_id="bundle-builder",
        producer_version="v1",
        identity_metadata={"bundle_config_hash": "1" * 64},
        parents=(),
        files=(),
    )
    verdict = RestoreTestVerdict(
        logical_id="runtime-bundle.restore-test",
        subject=bundle.node_ref(),
        manifest_hash="2" * 64,
        primary_root=str(primary),
        backup_root=str(backup),
        verify_mode="full",
        checked_object_count=12,
        passed=True,
        blocking_reasons=(),
    ).to_descriptor()

    assert verdict.node_kind == "restore_test_verdict"
    assert verdict.parents == (bundle.node_ref(),)
    assert verdict.identity_metadata["manifest_hash"] == "2" * 64
