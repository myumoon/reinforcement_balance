import json
import multiprocessing
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from reinbalance_survivors_contracts import canonical_hash, canonical_json_bytes
from reinbalance_survivors_contracts.artifact_identity import (
    ArtifactDescriptor,
    ArtifactRef,
    artifact_uri,
)
from Tools.Artifacts import artifact_store
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


def _race_put_bytes(store, monkeypatch, requests):
    barrier = threading.Barrier(len(requests))
    original_record = store._record_logical_id

    def synchronized_record(ref):
        barrier.wait()
        return original_record(ref)

    monkeypatch.setattr(store, "_record_logical_id", synchronized_record)
    refs = []
    errors = []

    def put(request):
        try:
            refs.append(store.put_bytes(**request))
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=put, args=(request,)) for request in requests]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert all(not thread.is_alive() for thread in threads)
    return refs, errors


def _race_store_instances_put_bytes(stores, monkeypatch, requests):
    barrier = threading.Barrier(len(requests))
    refs = []
    errors = []

    for store in stores:
        original_record = store._record_logical_id

        def synchronized_record(ref, *, original=original_record):
            barrier.wait()
            return original(ref)

        monkeypatch.setattr(store, "_record_logical_id", synchronized_record)

    def put(store, request):
        try:
            refs.append(store.put_bytes(**request))
        except Exception as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=put, args=(store, request))
        for store, request in zip(stores, requests)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert all(not thread.is_alive() for thread in threads)
    return refs, errors


def _process_put_bytes(root, request, barrier, result_queue):
    store = ArtifactStore(root)
    original_record = store._record_logical_id

    def synchronized_record(ref):
        barrier.wait(timeout=15)
        return original_record(ref)

    store._record_logical_id = synchronized_record
    try:
        result_queue.put(("success", store.put_bytes(**request).to_wire()))
    except ArtifactStoreError as exc:
        result_queue.put(("artifact_error", str(exc)))
    except Exception as exc:
        result_queue.put(("unexpected_error", repr(exc)))


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


def test_concurrent_different_bytes_publish_only_one_logical_ref(
    tmp_path, monkeypatch
):
    store = ArtifactStore(tmp_path / "store")
    logical_id = "models/concurrent.onnx"

    refs, errors = _race_put_bytes(
        store,
        monkeypatch,
        (
            {
                "logical_id": logical_id,
                "data": b"model-a",
                "media_type": "application/onnx",
            },
            {
                "logical_id": logical_id,
                "data": b"model-b",
                "media_type": "application/onnx",
            },
        ),
    )

    assert len(refs) == 1
    assert len(errors) == 1
    assert isinstance(errors[0], ArtifactStoreError)
    index = json.loads(
        store._logical_index_path(logical_id).read_text(encoding="utf-8")
    )
    assert index == refs[0].to_wire()


def test_concurrent_identical_publish_is_idempotent(tmp_path, monkeypatch):
    store = ArtifactStore(tmp_path / "store")
    request = {
        "logical_id": "models/idempotent.onnx",
        "data": b"same-model",
        "media_type": "application/onnx",
    }

    refs, errors = _race_put_bytes(store, monkeypatch, (request, request))

    assert errors == []
    assert len(refs) == 2
    assert refs[0] == refs[1]


def test_concurrent_same_hash_with_different_media_type_is_rejected(
    tmp_path, monkeypatch
):
    store = ArtifactStore(tmp_path / "store")
    logical_id = "models/media-type"

    refs, errors = _race_put_bytes(
        store,
        monkeypatch,
        (
            {
                "logical_id": logical_id,
                "data": b"same-bytes",
                "media_type": "application/onnx",
            },
            {
                "logical_id": logical_id,
                "data": b"same-bytes",
                "media_type": "application/octet-stream",
            },
        ),
    )

    assert len(refs) == 1
    assert len(errors) == 1
    assert isinstance(errors[0], ArtifactStoreError)
    index = json.loads(
        store._logical_index_path(logical_id).read_text(encoding="utf-8")
    )
    assert index == refs[0].to_wire()


def test_concurrent_different_store_instances_serialize_same_logical_id(
    tmp_path, monkeypatch
):
    root = tmp_path / "store"
    stores = (ArtifactStore(root), ArtifactStore(root))
    logical_id = "models/cross-instance.onnx"
    requests = (
        {
            "logical_id": logical_id,
            "data": b"instance-a",
            "media_type": "application/onnx",
        },
        {
            "logical_id": logical_id,
            "data": b"instance-b",
            "media_type": "application/onnx",
        },
    )

    refs, errors = _race_store_instances_put_bytes(
        stores,
        monkeypatch,
        requests,
    )

    assert len(refs) == 1
    assert len(errors) == 1
    assert isinstance(errors[0], ArtifactStoreError)
    index = json.loads(
        stores[0]._logical_index_path(logical_id).read_text(encoding="utf-8")
    )
    assert index == refs[0].to_wire()


def test_concurrent_processes_serialize_same_logical_id(tmp_path):
    root = tmp_path / "store"
    logical_id = "models/cross-process.onnx"
    requests = (
        {
            "logical_id": logical_id,
            "data": b"process-a",
            "media_type": "application/onnx",
        },
        {
            "logical_id": logical_id,
            "data": b"process-b",
            "media_type": "application/onnx",
        },
    )
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)
    result_queue = context.Queue()
    processes = [
        context.Process(
            target=_process_put_bytes,
            args=(root, request, barrier, result_queue),
        )
        for request in requests
    ]

    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=20)

        assert all(not process.is_alive() for process in processes)
        assert [process.exitcode for process in processes] == [0, 0]
        results = [result_queue.get(timeout=5) for _ in processes]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        result_queue.close()
        result_queue.join_thread()

    successes = [payload for status, payload in results if status == "success"]
    errors = [payload for status, payload in results if status == "artifact_error"]
    unexpected = [payload for status, payload in results if status == "unexpected_error"]
    assert unexpected == []
    assert len(successes) == 1
    assert len(errors) == 1
    index = json.loads(
        ArtifactStore(root)._logical_index_path(logical_id).read_text(encoding="utf-8")
    )
    assert index == successes[0]


def test_thread_lock_registry_releases_unique_logical_ids(tmp_path):
    store = ArtifactStore(tmp_path / "store")
    lock_keys = set()

    for index in range(32):
        logical_id = f"datasets/unique-{index}.jsonl"
        lock_keys.add(str(store._logical_lock_path(logical_id).resolve()))
        store.put_bytes(
            logical_id=logical_id,
            data=f"row-{index}".encode("utf-8"),
            media_type="application/jsonl",
        )

    with artifact_store._THREAD_LOCKS_GUARD:
        retained_keys = lock_keys.intersection(artifact_store._THREAD_LOCKS)
    assert retained_keys == set()


def test_corrupt_logical_index_fails_closed_and_releases_lock(tmp_path):
    store = ArtifactStore(tmp_path / "store")
    logical_id = "models/corrupt-index.onnx"
    ref = store.put_bytes(
        logical_id=logical_id,
        data=b"model-bytes",
        media_type="application/onnx",
    )
    index_path = store._logical_index_path(logical_id)
    index_path.write_text('{"schema_version":', encoding="utf-8")

    with pytest.raises(ArtifactStoreError, match="invalid logical id index"):
        store.put_bytes(
            logical_id=logical_id,
            data=b"model-bytes",
            media_type="application/onnx",
        )

    index_path.write_text(json.dumps(ref.to_wire()), encoding="utf-8")
    assert (
        store.put_bytes(
            logical_id=logical_id,
            data=b"model-bytes",
            media_type="application/onnx",
        )
        == ref
    )


def test_logical_index_publish_uses_durable_replace_before_revalidation(
    tmp_path, monkeypatch
):
    store = ArtifactStore(tmp_path / "store")
    ref = _object_ref(
        "models/durable.onnx",
        "a" * 64,
        1,
        "application/onnx",
    )
    events = []
    original_read = store._read_logical_index

    def tracked_replace(source, destination):
        artifact_store.os.replace(source, destination)
        events.append(("durable_replace", Path(destination)))

    def tracked_read(path):
        events.append(("read", path))
        return original_read(path)

    monkeypatch.setattr(
        artifact_store,
        "_durable_replace",
        tracked_replace,
        raising=False,
    )
    monkeypatch.setattr(store, "_read_logical_index", tracked_read)

    store._record_logical_id(ref)

    index_path = store._logical_index_path(ref.logical_id)
    assert events == [
        ("durable_replace", index_path),
        ("read", index_path),
    ]


def test_object_durability_completes_before_logical_index_publish(
    tmp_path, monkeypatch
):
    """object directory entry is durable before its logical index can appear."""
    store = ArtifactStore(tmp_path / "store")
    events = []

    def tracked_replace(source, destination):
        artifact_store.os.replace(source, destination)
        events.append(Path(destination))

    monkeypatch.setattr(artifact_store, "_durable_replace", tracked_replace)

    ref = store.put_bytes(
        logical_id="models/object-before-index.onnx",
        data=b"durable-object-bytes",
        media_type="application/onnx",
    )

    assert events == [
        store.object_path(ref.store_uri),
        store._logical_index_path(ref.logical_id),
    ]


def test_durable_replace_fsyncs_object_parent_directory_on_posix(monkeypatch):
    source_path = Path("store/objects/sha256/aa/pending.tmp")
    object_path = Path("store/objects/sha256/aa/aa-object")
    calls = []
    flags = artifact_store.os.O_RDONLY | getattr(
        artifact_store.os, "O_DIRECTORY", 0
    )

    monkeypatch.setattr(
        artifact_store.os,
        "replace",
        lambda source, destination: calls.append(
            ("replace", source, destination)
        ),
    )
    monkeypatch.setattr(
        artifact_store.os,
        "open",
        lambda path, actual_flags: calls.append(
            ("open", path, actual_flags)
        )
        or 71,
    )
    monkeypatch.setattr(
        artifact_store.os,
        "fsync",
        lambda descriptor: calls.append(("fsync", descriptor)),
    )
    monkeypatch.setattr(
        artifact_store.os,
        "close",
        lambda descriptor: calls.append(("close", descriptor)),
    )

    artifact_store._durable_replace(source_path, object_path, platform="posix")

    assert calls == [
        ("replace", source_path, object_path),
        ("open", object_path.parent, flags),
        ("fsync", 71),
        ("close", 71),
    ]


def test_durable_replace_uses_windows_write_through_flags(monkeypatch):
    source_path = Path("not-a-real-windows-directory/pending.tmp")
    index_path = Path("not-a-real-windows-directory/id.json")
    calls = []

    class FakeMoveFileEx:
        def __call__(self, source, destination, flags):
            calls.append((source, destination, flags))
            return 1

    fake_move_file_ex = FakeMoveFileEx()
    fake_ctypes = SimpleNamespace(
        WinDLL=lambda name, use_last_error: SimpleNamespace(
            MoveFileExW=fake_move_file_ex
        ),
        WinError=lambda code: AssertionError(f"unexpected WinError: {code}"),
        get_last_error=lambda: 0,
        c_wchar_p=object(),
        c_uint32=object(),
        c_int=object(),
    )
    monkeypatch.setattr(
        artifact_store,
        "ctypes",
        fake_ctypes,
        raising=False,
    )

    artifact_store._durable_replace(source_path, index_path, platform="nt")

    assert calls == [(str(source_path), str(index_path), 0x1 | 0x8)]
    assert fake_move_file_ex.argtypes == [
        fake_ctypes.c_wchar_p,
        fake_ctypes.c_wchar_p,
        fake_ctypes.c_uint32,
    ]
    assert fake_move_file_ex.restype is fake_ctypes.c_int


def test_durable_replace_propagates_windows_error(monkeypatch):
    source_path = Path("not-a-real-windows-directory/pending.tmp")
    index_path = Path("not-a-real-windows-directory/id.json")
    expected_error = OSError("MoveFileExW failed")
    calls = []

    class FakeMoveFileEx:
        def __call__(self, source, destination, flags):
            calls.append((source, destination, flags))
            return 0

    fake_move_file_ex = FakeMoveFileEx()

    def win_error(code):
        calls.append(("WinError", code))
        return expected_error

    fake_ctypes = SimpleNamespace(
        WinDLL=lambda name, use_last_error: SimpleNamespace(
            MoveFileExW=fake_move_file_ex
        ),
        WinError=win_error,
        get_last_error=lambda: 1234,
        c_wchar_p=object(),
        c_uint32=object(),
        c_int=object(),
    )
    monkeypatch.setattr(
        artifact_store,
        "ctypes",
        fake_ctypes,
        raising=False,
    )

    with pytest.raises(OSError) as caught:
        artifact_store._durable_replace(source_path, index_path, platform="nt")

    assert caught.value is expected_error
    assert calls == [
        (str(source_path), str(index_path), 0x1 | 0x8),
        ("WinError", 1234),
    ]


@pytest.mark.skipif(artifact_store.os.name != "nt", reason="Windows contract")
def test_durable_replace_moves_exact_full_bytes_on_windows(tmp_path):
    ref = _object_ref(
        "models/windows-durable.onnx",
        "b" * 64,
        17,
        "application/onnx",
    )
    expected = canonical_json_bytes(ref.to_wire())
    source_path = tmp_path / "pending.tmp"
    index_path = tmp_path / "index.json"
    source_path.write_bytes(expected)
    index_path.write_bytes(b"old-index")

    artifact_store._durable_replace(source_path, index_path, platform="nt")

    assert not source_path.exists()
    assert index_path.read_bytes() == expected
    assert json.loads(index_path.read_text(encoding="utf-8")) == ref.to_wire()


@pytest.mark.skipif(artifact_store.os.name != "nt", reason="Windows contract")
def test_put_bytes_uses_real_windows_write_through_for_object_then_index(
    tmp_path, monkeypatch
):
    """Both published paths pass through the real MoveFileExW helper in order."""
    store = ArtifactStore(tmp_path / "store")
    destinations = []
    real_replace = artifact_store._durable_replace

    def tracked_replace(source, destination):
        destinations.append(Path(destination))
        real_replace(source, destination)

    monkeypatch.setattr(artifact_store, "_durable_replace", tracked_replace)

    ref = store.put_bytes(
        logical_id="models/windows-object-before-index.onnx",
        data=b"windows-durable-object",
        media_type="application/onnx",
    )

    assert destinations == [
        store.object_path(ref.store_uri),
        store._logical_index_path(ref.logical_id),
    ]


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
