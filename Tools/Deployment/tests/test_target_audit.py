import copy
import pytest

from survivors.target_audit import AuditError, SaveLifecycle, audit_target
from survivors.target_profile import load_target_profile


@pytest.mark.parametrize("path", [
    ("build", "build_id"), ("progression", "save_artifact_hash"),
    ("progression", "purchased_power_ups"), ("display", "ui_scale"),
    ("display", "windows_dpi_scale"), ("input", "key_bindings"),
    ("hardware", "gpu_name"), ("choice_taxonomy", "candidate_vocabulary_hash"),
])
def test_exact_audit_rejects_missing_or_different_fields(path):
    expected = load_target_profile().to_wire(); actual = copy.deepcopy(expected)
    del actual[path[0]][path[1]]
    with pytest.raises(AuditError): audit_target(expected, actual)


def test_manual_build_attestation_is_complete():
    expected = load_target_profile().to_wire(); actual = copy.deepcopy(expected)
    actual["build"]["executable_hash"] = None
    with pytest.raises(AuditError): audit_target(expected, actual)
    actual["build"]["manual_attestation"] = {"operator": "op", "date": "2026-07-22", "evidence_hash": "a" * 64}
    audit_target(expected, actual)


@pytest.mark.parametrize("override", [
    {"game_stopped": False}, {"launcher_stopped": False}, {"cloud_sync_disabled": None},
    {"restore_conflict": True}, {"canonical_hash_matches": False},
])
def test_save_preflight_failure_has_attempt_only(override):
    lifecycle = SaveLifecycle.valid(); values = lifecycle.to_wire(); values.update(override)
    result = SaveLifecycle.from_wire(values).preflight("attempt")
    assert not result.can_reserve and result.attempt_id == "attempt"
    assert result.reserved_run_id is None and result.process_ref is None


def test_save_lifecycle_tracks_backup_and_hashes():
    lifecycle = SaveLifecycle.valid()
    assert lifecycle.original_backup_hash and lifecycle.canonical_save_hash
    result = lifecycle.preflight("attempt")
    assert result.can_reserve
    evidence = lifecycle.record_post_run("b" * 64)
    assert evidence["pre_run_hash"] == lifecycle.canonical_save_hash
