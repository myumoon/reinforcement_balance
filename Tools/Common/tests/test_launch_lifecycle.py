import pytest

from reinbalance_survivors_contracts.launch_lifecycle import (
    LaunchLifecycle,
    LaunchState,
    PlatformGate,
    LaunchAuthorization,
    AuditVerdict, SaveVerdict,
    _verify_audit_evidence, _verify_save_lifecycle,
)
from reinbalance_survivors_contracts.canonical_json import canonical_hash, canonical_json_bytes

def auth(tmp_path,attempt="attempt"):
    target="a"*64; evidence=b"audited"; eh=canonical_hash({"bytes_hex":evidence.hex()})
    audit=_verify_audit_evidence(attempt_id=attempt,target_identity_hash=target,evidence_bytes=evidence,expected_evidence_hash=eh)
    records=[{"attempt_id":attempt,"stage":s,"verdict":"PASS","hash":"b"*64 if s=="PRE_RUN_AUDIT" else None} for s in ("PREFLIGHT","ORIGINAL_BACKUP","CANONICAL_RESTORE","PRE_RUN_AUDIT")]
    path=tmp_path/(attempt+".jsonl"); path.write_bytes(b"".join(canonical_json_bytes(r)+b"\n" for r in records))
    save=_verify_save_lifecycle(record_path=path,attempt_id=attempt,target_identity_hash=target,expected_pre_run_hash="b"*64)
    platform=PlatformGate(True,True,True,True,True,True,True)
    return LaunchAuthorization.issue(audit,platform.verified(attempt,target),save)

def test_authorization_cannot_be_self_attested():
    with pytest.raises((TypeError,ValueError)): LaunchAuthorization("a"*64,"b"*64,"c"*64)
    assert not hasattr(AuditVerdict,"_verified") and not hasattr(SaveVerdict,"_verified")
    with pytest.raises((TypeError,ValueError)): AuditVerdict("a","b","c","d")

def test_forged_pass_mappings_and_durable_bool_cannot_authorize():
    fake={"schema_version":"target_audit.v1","target_hash":"a"*64,"status":"PASS"}
    with pytest.raises((TypeError,ValueError)):
        LaunchAuthorization.issue(fake,PlatformGate(True,True,True,True,True,True,True),{"status":"PASS"},durable_commit=True)

def test_evidence_and_identity_binding_are_fail_closed(tmp_path):
    with pytest.raises(ValueError): _verify_audit_evidence(attempt_id="a",target_identity_hash="1"*64,evidence_bytes=b"actual",expected_evidence_hash="2"*64)
    a=auth(tmp_path,"a")
    b=auth(tmp_path,"b")
    with pytest.raises(ValueError):
        LaunchAuthorization.issue(
            _verify_audit_evidence(attempt_id="a",target_identity_hash=a.target_identity_hash,evidence_bytes=b"x",expected_evidence_hash=canonical_hash({"bytes_hex":b"x".hex()})),
            PlatformGate(True,True,True,True,True,True,True).verified("b",b.target_identity_hash),
            _verify_save_lifecycle(record_path=tmp_path/"b.jsonl",attempt_id="b",target_identity_hash=b.target_identity_hash,expected_pre_run_hash="b"*64))


def test_launch_intent_and_activation_cardinality(tmp_path):
    lifecycle = LaunchLifecycle.begin("attempt-1")
    with pytest.raises(ValueError):
        lifecycle.activate("proc-1")
    lifecycle = lifecycle.reserve("run-1", "gameplay-1", "nonce-1", authorization=auth(tmp_path,"attempt-1"),intent_log=tmp_path/"intent.jsonl")
    with pytest.raises(ValueError):
        lifecycle.reserve("run-2", "gameplay-2", "nonce-2", authorization=auth(tmp_path,"attempt-1"),intent_log=tmp_path/"intent.jsonl")
    lifecycle = lifecycle.activate("proc-1")
    assert lifecycle.state is LaunchState.FORMAL_RUN_ACTIVATED
    assert lifecycle.counts_toward_outcome_denominator
    with pytest.raises(ValueError):
        lifecycle.activate("proc-2")


def test_prelaunch_and_uncertain_failures_do_not_mix_with_outcomes(tmp_path):
    failed = LaunchLifecycle.begin("a").preflight_failure("cloud_sync_unknown")
    assert failed.reserved_run_id is None and failed.process_ref is None
    gate = LaunchLifecycle.begin("b").reserve("r", "g", "n", authorization=auth(tmp_path,"b"),intent_log=tmp_path/"intent.jsonl")
    gate = gate.create_process_failed()
    assert gate.state is LaunchState.LAUNCH_GATE_FAILED
    assert not gate.counts_toward_outcome_denominator
    uncertain = LaunchLifecycle.begin("c").reserve("r2", "g2", "n2", authorization=auth(tmp_path,"c"),intent_log=tmp_path/"intent.jsonl")
    uncertain = uncertain.launch_uncertain()
    assert uncertain.campaign_blocked and not uncertain.replacement_allowed


@pytest.mark.parametrize("field", ["local_fixed_volume", "ntfs", "wal", "synchronous_full", "integrity_ok", "flush_ok", "broker_attested"])
def test_platform_gate_is_fail_closed(field):
    values = dict(local_fixed_volume=True, ntfs=True, wal=True, synchronous_full=True,
                  integrity_ok=True, flush_ok=True, broker_attested=True)
    values[field] = False
    with pytest.raises(ValueError):
        PlatformGate(**values).validate()


def test_unknown_wire_fields_are_rejected():
    data = LaunchLifecycle.begin("a").to_wire()
    data["future"] = True
    with pytest.raises(ValueError):
        LaunchLifecycle.from_wire(data)

def test_malformed_activation_is_rejected():
    data=LaunchLifecycle.begin("a").to_wire(); data["state"]="FORMAL_RUN_ACTIVATED"
    with pytest.raises(ValueError): LaunchLifecycle.from_wire(data)

def test_launch_intent_requires_successful_durable_commit(tmp_path):
    with pytest.raises(ValueError):
        LaunchLifecycle.begin("a").reserve("r","g","n",authorization=auth(tmp_path,"a"),intent_log=None)
    directory=tmp_path/"directory"; directory.mkdir()
    with pytest.raises(ValueError):
        LaunchLifecycle.begin("a").reserve("r","g","n",authorization=auth(tmp_path,"a"),intent_log=directory)
