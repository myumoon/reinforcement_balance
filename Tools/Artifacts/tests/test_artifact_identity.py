import pytest

from reinbalance_survivors_contracts import ContractValidationError
from reinbalance_survivors_contracts.artifact_identity import (
    ARTIFACT_DESCRIPTOR_SCHEMA_VERSION,
    ARTIFACT_REF_SCHEMA_VERSION,
    OBJECT_URI_PREFIX,
    ArtifactDescriptor,
    ArtifactRef,
    ValidationVerdict,
    artifact_uri,
)


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64
HASH_E = "e" * 64


def _ref(sha256=HASH_A, logical_id="phase5/model.onnx"):
    return ArtifactRef(
        logical_id=logical_id,
        sha256=sha256,
        size_bytes=123,
        media_type="application/octet-stream",
        store_uri=artifact_uri(sha256),
    )


def _source_descriptor(
    *,
    content_hash=HASH_A,
    config_hash=HASH_C,
    non_identity_metadata=None,
    logical_id="phase5-source",
):
    return ArtifactDescriptor(
        logical_id=logical_id,
        node_kind="source_descriptor",
        producer_id="phase5-trainer",
        producer_version="trainer.v1",
        identity_metadata={
            "config_hash": config_hash,
            "target_profile_hash": HASH_D,
        },
        parents=(),
        files=(_ref(content_hash),),
        non_identity_metadata=non_identity_metadata or {},
    )


def test_descriptor_identity_excludes_timestamp_operator_and_paths():
    left = _source_descriptor(
        non_identity_metadata={
            "created_at_utc": "2026-07-20T01:02:03Z",
            "operator": "alice",
            "local_path": "/mnt/private/source-a.onnx",
        }
    )
    right = _source_descriptor(
        non_identity_metadata={
            "created_at_utc": "2026-07-21T04:05:06Z",
            "operator": "bob",
            "local_path": "D:/private/source-b.onnx",
        }
    )

    assert left.schema_version == ARTIFACT_DESCRIPTOR_SCHEMA_VERSION
    assert left.identity_hash == right.identity_hash
    assert left.to_wire()["identity_hash"] == right.to_wire()["identity_hash"]


def test_descriptor_identity_changes_for_content_and_config_changes():
    baseline = _source_descriptor()

    assert _source_descriptor(content_hash=HASH_B).identity_hash != baseline.identity_hash
    assert _source_descriptor(config_hash=HASH_E).identity_hash != baseline.identity_hash


def test_artifact_ref_uses_content_addressed_uri():
    ref = _ref()

    assert ref.schema_version == ARTIFACT_REF_SCHEMA_VERSION
    assert ref.store_uri == f"{OBJECT_URI_PREFIX}{HASH_A}"
    assert ref.to_wire()["store_uri"] == ref.store_uri


def test_validation_verdict_stores_subject_hash_without_mutating_subject():
    source = _source_descriptor()
    source_hash_before = source.identity_hash
    verdict = ValidationVerdict(
        logical_id="phase5-source.teacher-gate",
        verdict_kind="teacher_validation_verdict",
        subject=source.node_ref(),
        gate_version="teacher-gate.v1",
        metrics={"win_rate": 1.0, "paired_score_delta": 0.25},
        split_ids=("development", "holdout"),
        session_ids=("session-001",),
        passed=True,
        blocking_reasons=(),
    )

    verdict_descriptor = verdict.to_descriptor()

    assert source.identity_hash == source_hash_before
    assert verdict_descriptor.node_kind == "teacher_validation_verdict"
    assert verdict_descriptor.parents == (source.node_ref(),)
    assert verdict_descriptor.identity_metadata["subject_identity_hash"] == source.identity_hash
    assert verdict_descriptor.identity_hash != source.identity_hash


def test_identity_metadata_rejects_volatile_fields():
    with pytest.raises(ContractValidationError):
        ArtifactDescriptor(
            logical_id="bad-source",
            node_kind="source_descriptor",
            producer_id="phase5-trainer",
            producer_version="trainer.v1",
            identity_metadata={"created_at_utc": "2026-07-21T00:00:00Z"},
            parents=(),
            files=(_ref(),),
        )


def test_wire_identity_hash_mismatch_rejects_in_place_mutation():
    wire = _source_descriptor().to_wire()
    wire["identity"]["identity_metadata"]["config_hash"] = HASH_E

    with pytest.raises(ContractValidationError):
        ArtifactDescriptor.from_wire(wire)
