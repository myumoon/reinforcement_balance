import pytest

from reinbalance_survivors_contracts.artifact_dag import (
    ArtifactDagValidationError,
    validate_artifact_dag,
    validate_formal_runtime_dag,
)
from reinbalance_survivors_contracts.artifact_identity import (
    ArtifactDescriptor,
    ArtifactRef,
    ValidationVerdict,
    artifact_uri,
)


def _hash(ch):
    return ch * 64


def _file(logical_id, ch):
    return ArtifactRef(
        logical_id=logical_id,
        sha256=_hash(ch),
        size_bytes=16,
        media_type="application/octet-stream",
        store_uri=artifact_uri(_hash(ch)),
    )


def _node(logical_id, node_kind, parents=(), ch="a"):
    return ArtifactDescriptor(
        logical_id=logical_id,
        node_kind=node_kind,
        producer_id="test-producer",
        producer_version="v1",
        identity_metadata={"stable_config_hash": _hash("f")},
        parents=parents,
        files=(_file(f"{logical_id}.bin", ch),),
    )


def _formal_verdict(profile):
    return ArtifactDescriptor(
        logical_id="perception-verdict",
        node_kind="perception_final_verdict",
        producer_id="test-producer",
        producer_version="v1",
        identity_metadata={
            "passed": True,
            "development_only": False,
            "subject_hashes": {"model_hash": _hash("9")},
        },
        parents=(profile.node_ref(),),
        files=(_file("perception-verdict.json", "c"),),
    )


def _campaign_lineage():
    source = _node("phase5-source", "source_descriptor", ch="a")
    verdict = _node(
        "teacher-verdict",
        "teacher_validation_verdict",
        parents=(source.node_ref(),),
        ch="b",
    )
    dataset = _node(
        "choice-dataset",
        "choice_dataset_release",
        parents=(verdict.node_ref(),),
        ch="c",
    )
    model = _node(
        "item-selector",
        "item_selector_release",
        parents=(dataset.node_ref(),),
        ch="d",
    )
    bundle = _node(
        "runtime-bundle",
        "runtime_bundle",
        parents=(model.node_ref(),),
        ch="e",
    )
    shadow = _node(
        "shadow-verdict",
        "replay_shadow_verdict",
        parents=(bundle.node_ref(),),
        ch="1",
    )
    campaign = _node(
        "canary-campaign",
        "canary_campaign",
        parents=(shadow.node_ref(),),
        ch="2",
    )
    return source, verdict, dataset, model, bundle, shadow, campaign


def test_perception_profile_and_final_verdict_real_descriptors_pass_consumer_dag():
    """producer と同じ ArtifactDescriptor 列を consumer validator へ通す。"""
    source = _node("perception-source", "source_descriptor", ch="a")
    profile = _node(
        "perception-profile",
        "perception_calibration_profile",
        parents=(source.node_ref(),),
        ch="b",
    )
    verdict = _node(
        "perception-verdict",
        "perception_final_verdict",
        parents=(profile.node_ref(),),
        ch="c",
    )

    report = validate_artifact_dag([verdict, source, profile])

    assert report.node_count == 3
    assert report.topological_identity_hashes == (
        source.identity_hash,
        profile.identity_hash,
        verdict.identity_hash,
    )


def test_formal_runtime_requires_passed_production_perception_verdict():
    source = _node("source", "source_descriptor", ch="a")
    teacher = _node("teacher", "teacher_validation_verdict", (source.node_ref(),), "b")
    dataset = _node("dataset", "choice_dataset_release", (teacher.node_ref(),), "c")
    item = _node("item", "item_selector_release", (dataset.node_ref(),), "d")
    combat = _node("combat", "combat_student_release", (dataset.node_ref(),), "e")
    profile = _node("profile", "perception_calibration_profile", (source.node_ref(),), "1")
    verdict = _formal_verdict(profile)
    runtime = ArtifactDescriptor(
        logical_id="runtime",
        node_kind="runtime_bundle",
        producer_id="test-producer",
        producer_version="v1",
        identity_metadata={"perception_subject_hashes": {"model_hash": _hash("9")}},
        parents=(item.node_ref(), combat.node_ref(), verdict.node_ref()),
        files=(_file("runtime.bin", "2"),),
    )
    assert validate_formal_runtime_dag(
        [source, teacher, dataset, item, combat, profile, verdict, runtime]
    ).node_count == 8

    missing = ArtifactDescriptor(
        logical_id="runtime-no-perception",
        node_kind="runtime_bundle",
        producer_id="test-producer",
        producer_version="v1",
        identity_metadata={"perception_subject_hashes": {"model_hash": _hash("9")}},
        parents=(item.node_ref(), combat.node_ref()),
        files=(_file("runtime-no-perception.bin", "3"),),
    )
    with pytest.raises(ArtifactDagValidationError, match="perception_final_verdict"):
        validate_formal_runtime_dag(
            [source, teacher, dataset, item, combat, profile, verdict, missing]
        )


def test_child_verdict_parent_reference_is_valid_storage_direction():
    source = _node("phase5-source", "source_descriptor", ch="a")
    verdict = ValidationVerdict(
        logical_id="phase5-source.teacher-verdict",
        verdict_kind="teacher_validation_verdict",
        subject=source.node_ref(),
        gate_version="teacher-gate.v1",
        metrics={"pass_rate": 1.0},
        split_ids=("dev",),
        session_ids=("s1",),
        passed=True,
        blocking_reasons=(),
    ).to_descriptor()

    report = validate_artifact_dag([source, verdict])

    assert report.node_count == 2
    assert report.topological_identity_hashes == (
        source.identity_hash,
        verdict.identity_hash,
    )


def test_source_to_verdict_to_dataset_to_model_to_bundle_to_campaign_dag_passes():
    source, verdict, dataset, model, bundle, shadow, campaign = _campaign_lineage()
    restore = _node(
        "restore-test",
        "restore_test_verdict",
        parents=(bundle.node_ref(),),
        ch="4",
    )
    evidence = _node(
        "goal-evidence",
        "goal_evidence",
        parents=(campaign.node_ref(), restore.node_ref()),
        ch="3",
    )

    report = validate_artifact_dag(
        [evidence, restore, campaign, shadow, bundle, model, dataset, verdict, source]
    )

    assert report.node_count == 9
    assert report.topological_identity_hashes[0] == source.identity_hash
    assert report.topological_identity_hashes[-1] == evidence.identity_hash


def test_goal_evidence_requires_restore_test_verdict_parent():
    source, verdict, dataset, model, bundle, shadow, campaign = _campaign_lineage()
    evidence = _node(
        "goal-evidence",
        "goal_evidence",
        parents=(campaign.node_ref(),),
        ch="d",
    )

    with pytest.raises(ArtifactDagValidationError, match="restore_test_verdict"):
        validate_artifact_dag(
            [source, verdict, dataset, model, bundle, shadow, campaign, evidence]
        )


def test_goal_evidence_requires_canary_campaign_parent():
    source, verdict, dataset, model, bundle, shadow, campaign = _campaign_lineage()
    restore = _node(
        "restore-test",
        "restore_test_verdict",
        parents=(bundle.node_ref(),),
        ch="d",
    )
    evidence = _node(
        "goal-evidence",
        "goal_evidence",
        parents=(restore.node_ref(),),
        ch="e",
    )

    with pytest.raises(ArtifactDagValidationError, match="canary_campaign"):
        validate_artifact_dag(
            [source, verdict, dataset, model, bundle, shadow, campaign, restore, evidence]
        )


def test_goal_evidence_with_campaign_and_restore_test_verdict_parent_passes():
    source, verdict, dataset, model, bundle, shadow, campaign = _campaign_lineage()
    restore = _node(
        "restore-test",
        "restore_test_verdict",
        parents=(bundle.node_ref(),),
        ch="d",
    )
    evidence = _node(
        "goal-evidence",
        "goal_evidence",
        parents=(campaign.node_ref(), restore.node_ref()),
        ch="e",
    )

    report = validate_artifact_dag(
        [source, verdict, dataset, model, bundle, shadow, campaign, restore, evidence]
    )

    assert report.node_count == 9
    assert report.topological_identity_hashes[-1] == evidence.identity_hash


def test_non_root_node_without_parent_is_rejected():
    dataset = _node(
        "choice-dataset",
        "choice_dataset_release",
        parents=(),
        ch="c",
    )

    with pytest.raises(ArtifactDagValidationError, match="must declare at least one parent"):
        validate_artifact_dag([dataset])


def test_source_descriptor_must_not_reference_descendant_verdict():
    source = _node("reference-source", "source_descriptor", ch="a")
    verdict_like_parent = _node(
        "teacher-verdict",
        "teacher_validation_verdict",
        parents=(source.node_ref(),),
        ch="b",
    )
    bad_source = _node(
        "phase5-source",
        "source_descriptor",
        parents=(verdict_like_parent.node_ref(),),
        ch="c",
    )

    with pytest.raises(ArtifactDagValidationError):
        validate_artifact_dag([source, verdict_like_parent, bad_source])


def test_missing_parent_rejected():
    source = _node("phase5-source", "source_descriptor", ch="a")
    dataset = _node(
        "choice-dataset",
        "choice_dataset_release",
        parents=(source.node_ref(),),
        ch="c",
    )

    with pytest.raises(ArtifactDagValidationError):
        validate_artifact_dag([dataset])


def test_same_logical_id_with_different_identity_hash_rejected():
    first = _node("phase5-source", "source_descriptor", ch="a")
    second = _node("phase5-source", "source_descriptor", ch="b")

    with pytest.raises(ArtifactDagValidationError):
        validate_artifact_dag([first, second])


def test_in_place_identity_mutation_rejected():
    wire = _node("phase5-source", "source_descriptor").to_wire()
    wire["identity"]["identity_metadata"]["stable_config_hash"] = _hash("e")

    with pytest.raises(ArtifactDagValidationError):
        validate_artifact_dag([wire])


def test_self_reference_or_mutual_reference_wire_is_rejected():
    source = _node("phase5-source", "source_descriptor", ch="a")
    wire = source.to_wire()
    wire["identity"]["parents"] = [source.node_ref().to_wire()]

    with pytest.raises(ArtifactDagValidationError):
        validate_artifact_dag([wire])
