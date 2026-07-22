import json
import os
from types import SimpleNamespace
from pathlib import Path
import pytest
import reinbalance_survivors_contracts.launch_lifecycle as launch_lifecycle

from reinbalance_survivors_contracts.launch_lifecycle import (
    LaunchLifecycle,
    LaunchState,
    PlatformGate,
    LaunchAuthorization,
    AuditVerdict, SaveVerdict, LaunchIntentStore,
    _verify_audit_evidence, _finalize_save_execution, _sync_launch_intent,
)
from reinbalance_survivors_contracts.canonical_json import canonical_hash, canonical_json_bytes

def audit_verdict(attempt,target,save_hash):
    evidence=b"audited"; eh=canonical_hash({"bytes_hex":evidence.hex()})
    return _verify_audit_evidence(attempt_id=attempt,target_identity_hash=target,canonical_save_hash=save_hash,save_semantics_hash="c"*64,evidence_bytes=evidence,expected_evidence_hash=eh)

def save_verdict(tmp_path,attempt,target):
    records=[]; previous=None
    for stage in ("PREFLIGHT","ORIGINAL_BACKUP","CANONICAL_RESTORE","PRE_RUN_AUDIT"):
        body={"attempt_id":attempt,"stage":stage,"content_hash":"b"*64 if stage=="PRE_RUN_AUDIT" else None,"previous_step_hash":previous}
        record={**body,"step_hash":canonical_hash(body)}; records.append(record); previous=record["step_hash"]
    evidence=tmp_path/"fixed-evidence"; evidence.mkdir(parents=True,exist_ok=True)
    lifecycle_root=evidence/"lifecycles"; lifecycle_root.mkdir(exist_ok=True)
    lifecycle=lifecycle_root/(attempt+".jsonl"); lifecycle.write_bytes(b"".join(canonical_json_bytes(r)+b"\n" for r in records))
    canonical=evidence/"canonical.save"; canonical.write_bytes(b"")
    backup=evidence/"backup.save"; backup.write_bytes(b"backup")
    pre=canonical_hash({"bytes_hex":""})
    backup_hash=canonical_hash({"bytes_hex":b"backup".hex()})
    # Bind the chain to the actual files.
    records=[]; previous=None
    for stage in ("PREFLIGHT","ORIGINAL_BACKUP","CANONICAL_RESTORE","PRE_RUN_AUDIT"):
        content=pre if stage in {"CANONICAL_RESTORE","PRE_RUN_AUDIT"} else backup_hash if stage=="ORIGINAL_BACKUP" else None
        body={"attempt_id":attempt,"stage":stage,"content_hash":content,"previous_step_hash":previous}
        record={**body,"step_hash":canonical_hash(body)}; records.append(record); previous=record["step_hash"]
    lifecycle.write_bytes(b"".join(canonical_json_bytes(r)+b"\n" for r in records))
    return _finalize_save_execution(records=tuple(records),audit=audit_verdict(attempt,target,pre),expected_pre_run_hash=pre,save_semantics_hash="c"*64,lifecycle_record_path=lifecycle,canonical_save_path=canonical,original_backup_path=backup)

def auth(tmp_path,attempt="attempt"):
    target="a"*64; evidence=b"audited"; eh=canonical_hash({"bytes_hex":evidence.hex()})
    save=save_verdict(tmp_path,attempt,target)
    audit=audit_verdict(attempt,target,save.pre_run_hash)
    config=tmp_path/"launch_gate.json"
    config.write_text('{"schema_version":"launch_gate.v1","lifecycle_root":"'+str(Path(save.lifecycle_record_path).parent)+'","canonical_save_path":"'+save.canonical_save_path+'","original_backup_path":"'+save.original_backup_path+'"}')
    launch_lifecycle.LAUNCH_GATE_CONFIG_PATH=config
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
    with pytest.raises(ValueError): _verify_audit_evidence(attempt_id="a",target_identity_hash="1"*64,canonical_save_hash="b"*64,save_semantics_hash="c"*64,evidence_bytes=b"actual",expected_evidence_hash="2"*64)
    a=auth(tmp_path,"a")
    b=auth(tmp_path,"b")
    with pytest.raises(ValueError):
        LaunchAuthorization.issue(
            _verify_audit_evidence(attempt_id="a",target_identity_hash=a.target_identity_hash,canonical_save_hash="b"*64,save_semantics_hash="c"*64,evidence_bytes=b"x",expected_evidence_hash=canonical_hash({"bytes_hex":b"x".hex()})),
            PlatformGate(True,True,True,True,True,True,True).verified("b",b.target_identity_hash),
            save_verdict(tmp_path,"b",b.target_identity_hash))

def test_handwritten_save_jsonl_has_no_verdict_path(tmp_path):
    path=tmp_path/"forged.jsonl"; path.write_text('{"attempt_id":"a","stage":"PRE_RUN_AUDIT","result":"PASS"}\n')
    with pytest.raises((TypeError,ValueError)):
        SaveVerdict("a","f"*64,"a","c"*64,"b"*64,"d"*64)
    assert not hasattr(launch_lifecycle, "_SAVE_EXECUTION_SEAL")
    assert not hasattr(launch_lifecycle, "_mint_executed_save_verdict")

def store(tmp_path, campaign):
    config=tmp_path/"launch_store.json"
    config.write_text('{"schema_version":"launch_store.v1","root":"'+str(tmp_path/'fixed')+'","local_fixed_volume":true,"ntfs":true}')
    launch_lifecycle.LAUNCH_STORE_CONFIG_PATH=config
    return LaunchIntentStore.for_campaign(campaign)


@pytest.mark.parametrize("campaign_id", ["..", "../escape", "/absolute", r"C:\\escape", "has/slash", r"has\\slash"])
def test_campaign_identity_cannot_escape_fixed_store_root(tmp_path, campaign_id):
    config=tmp_path/"launch_store.json"
    config.write_text('{"schema_version":"launch_store.v1","root":"'+str(tmp_path/'fixed')+'","local_fixed_volume":true,"ntfs":true}')
    launch_lifecycle.LAUNCH_STORE_CONFIG_PATH=config
    with pytest.raises(ValueError, match="campaign identity"):
        LaunchIntentStore.for_campaign(campaign_id)


def test_failed_commit_after_marker_creation_rolls_back_for_retry(tmp_path, monkeypatch):
    ledger=store(tmp_path,"campaign-rollback")
    authorization=auth(tmp_path,"attempt-rollback")
    original_sync=launch_lifecycle._sync_launch_intent
    calls=0

    def fail_first_sync(path, *, platform=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("injected post-replace failure")
        return original_sync(path, platform=platform)

    monkeypatch.setattr(launch_lifecycle,"_sync_launch_intent",fail_first_sync)
    lifecycle=LaunchLifecycle.begin("attempt-rollback")
    with pytest.raises(ValueError, match="commit failed"):
        lifecycle.reserve("run-rollback","gameplay-rollback","nonce-rollback",authorization=authorization,store=ledger)

    assert not ledger.intent_log.exists()
    assert not tuple((ledger.root/"reservations").glob("*.lock"))
    reserved=lifecycle.reserve("run-rollback","gameplay-rollback","nonce-rollback",authorization=authorization,store=ledger)
    assert reserved.state is LaunchState.LAUNCH_INTENT


def test_orphan_markers_without_committed_intent_are_reconciled(tmp_path):
    ledger=store(tmp_path,"campaign-orphan")
    marker_dir=ledger.root/"reservations"
    marker_dir.mkdir(parents=True)
    identities=(("attempt","attempt-orphan"),("run","run-orphan"))
    markers=[]
    for kind, identity in identities:
        marker=marker_dir/(canonical_hash({"kind":kind,"identity":identity})+".lock")
        marker.touch(exist_ok=False)
        markers.append(marker)

    reserved=LaunchLifecycle.begin("attempt-orphan").reserve(
        "run-orphan","gameplay-orphan","nonce-orphan",
        authorization=auth(tmp_path,"attempt-orphan"),store=ledger)

    assert reserved.state is LaunchState.LAUNCH_INTENT
    assert all(marker.exists() for marker in markers)
    committed=[LaunchLifecycle.from_wire(json.loads(line))
               for line in ledger.intent_log.read_bytes().splitlines()]
    assert committed == [reserved]


def test_launch_intent_and_activation_cardinality(tmp_path):
    lifecycle = LaunchLifecycle.begin("attempt-1")
    with pytest.raises(ValueError):
        lifecycle.activate("proc-1")
    ledger=store(tmp_path,"campaign-cardinality")
    lifecycle = lifecycle.reserve("run-1", "gameplay-1", "nonce-1", authorization=auth(tmp_path,"attempt-1"),store=ledger)
    with pytest.raises(ValueError):
        lifecycle.reserve("run-2", "gameplay-2", "nonce-2", authorization=auth(tmp_path,"attempt-1"),store=ledger)
    with pytest.raises(ValueError):
        LaunchLifecycle.begin("attempt-1").reserve("run-other", "gameplay-other", "nonce-other", authorization=auth(tmp_path,"attempt-1"),store=ledger)
    lifecycle = lifecycle.activate("proc-1",store=ledger)
    assert lifecycle.state is LaunchState.FORMAL_RUN_ACTIVATED
    assert lifecycle.counts_toward_outcome_denominator
    with pytest.raises(ValueError):
        lifecycle.activate("proc-2",store=ledger)

def test_activation_and_terminal_are_recoverable_from_durable_ledger(tmp_path):
    ledger=store(tmp_path,"campaign-outcomes")
    lifecycle=LaunchLifecycle.begin("attempt-1").reserve("run-1","gameplay-1","nonce-1",authorization=auth(tmp_path,"attempt-1"),store=ledger)
    lifecycle=lifecycle.activate("proc-1",store=ledger)
    lifecycle=lifecycle.terminal("SUCCESS",store=ledger)
    restarted=LaunchIntentStore.for_campaign("campaign-outcomes")
    assert restarted.outcome_summary()=={"activated_runs":1,"terminal_outcomes":{"run-1":"SUCCESS"}}

def test_durable_ledger_rejects_reserved_run_id_shared_by_distinct_attempts(tmp_path):
    ledger=store(tmp_path,"campaign-duplicate-run")
    first=LaunchLifecycle.begin("attempt-1").reserve(
        "run-shared","gameplay-1","nonce-1",authorization=auth(tmp_path,"attempt-1"),store=ledger)
    duplicate=launch_lifecycle.replace(
        first,attempt_id="attempt-2",gameplay_attempt_id="gameplay-2",launch_nonce="nonce-2")
    with ledger.intent_log.open("ab") as stream:
        stream.write(canonical_json_bytes(duplicate.to_wire())+b"\n")

    with pytest.raises(ValueError,match="reserved_run_id"):
        ledger.records()
    with pytest.raises(ValueError,match="reserved_run_id"):
        ledger.outcome_summary()

@pytest.mark.parametrize(("field","value"), [
    ("attempt_id","different-attempt"),
    ("reserved_run_id","different-run"),
    ("target_identity_hash","b"*64),
])
def test_activation_rejects_identity_different_from_durable_reservation(tmp_path,field,value):
    ledger=store(tmp_path,"campaign-activation-identity")
    intent=LaunchLifecycle.begin("attempt-1").reserve(
        "run-1","gameplay-1","nonce-1",authorization=auth(tmp_path,"attempt-1"),store=ledger)
    forged=launch_lifecycle.replace(intent,**{field:value})
    with pytest.raises(ValueError,match="identity|predecessor"):
        forged.activate("proc-1",store=ledger)

def test_terminal_rejects_identity_different_from_reserved_activation(tmp_path):
    ledger=store(tmp_path,"campaign-terminal-identity")
    activated=LaunchLifecycle.begin("attempt-1").reserve(
        "run-1","gameplay-1","nonce-1",authorization=auth(tmp_path,"attempt-1"),store=ledger).activate("proc-1",store=ledger)
    forged=launch_lifecycle.replace(activated,target_identity_hash="b"*64)
    with pytest.raises(ValueError,match="identity|predecessor"):
        forged.terminal("SUCCESS",store=ledger)

def test_in_memory_activation_cannot_affect_durable_denominator(tmp_path):
    ledger=store(tmp_path,"campaign-memory")
    intent=LaunchLifecycle.begin("attempt-1").reserve("run-1","gameplay-1","nonce-1",authorization=auth(tmp_path,"attempt-1"),store=ledger)
    with pytest.raises(ValueError,match="store"):
        intent.activate("proc-1")
    assert LaunchIntentStore.for_campaign("campaign-memory").outcome_summary()["activated_runs"]==0

def test_campaign_store_cannot_be_rebound_to_bypass_uniqueness(tmp_path):
    ledger=store(tmp_path,"campaign-fixed")
    LaunchLifecycle.begin("same").reserve("run","gameplay","nonce",authorization=auth(tmp_path,"same"),store=ledger)
    second=LaunchIntentStore.for_campaign("campaign-fixed")
    with pytest.raises(ValueError, match="commit failed"):
        LaunchLifecycle.begin("same").reserve("other","gameplay","nonce",authorization=auth(tmp_path,"same"),store=second)

def test_forged_chain_without_real_save_lifecycle_cannot_authorize(tmp_path):
    valid=auth(tmp_path,"forged")
    save=save_verdict(tmp_path,"forged","a"*64)
    Path(save.canonical_save_path).write_bytes(b"tampered")
    audit_bytes=b"audited"; evidence_hash=canonical_hash({"bytes_hex":audit_bytes.hex()})
    audit=_verify_audit_evidence(attempt_id="forged",target_identity_hash="a"*64,canonical_save_hash=save.pre_run_hash,save_semantics_hash=save.save_semantics_hash,evidence_bytes=audit_bytes,expected_evidence_hash=evidence_hash)
    platform=PlatformGate(True,True,True,True,True,True,True).verified("forged","a"*64)
    with pytest.raises(ValueError, match="evidence mismatch"):
        LaunchAuthorization.issue(audit,platform,save)

@pytest.mark.parametrize("unsupported_index", range(3))
def test_authorization_rejects_save_evidence_on_unsupported_volume(tmp_path,monkeypatch,unsupported_index):
    attempt="unsupported-save-volume"
    target="a"*64
    save=save_verdict(tmp_path,attempt,target)
    paths=(Path(save.lifecycle_record_path),Path(save.canonical_save_path),Path(save.original_backup_path))
    audit_bytes=b"audited"
    audit=_verify_audit_evidence(
        attempt_id=attempt,target_identity_hash=target,canonical_save_hash=save.pre_run_hash,save_semantics_hash=save.save_semantics_hash,evidence_bytes=audit_bytes,
        expected_evidence_hash=canonical_hash({"bytes_hex":audit_bytes.hex()}))
    platform=PlatformGate(True,True,True,True,True,True,True).verified(attempt,target)
    config=tmp_path/"launch_gate.json"
    config.write_text(json.dumps({
        "schema_version":"launch_gate.v1","lifecycle_root":str(paths[0].parent),
        "canonical_save_path":str(paths[1]),"original_backup_path":str(paths[2])}))
    monkeypatch.setattr(launch_lifecycle,"LAUNCH_GATE_CONFIG_PATH",config)
    checked=[]
    def reject_unsupported(path):
        checked.append(path)
        if path==paths[unsupported_index]:
            raise ValueError("save evidence must be on local fixed NTFS volume")
    monkeypatch.setattr(launch_lifecycle,"_validate_store_volume",reject_unsupported)

    with pytest.raises(ValueError,match="local fixed NTFS"):
        LaunchAuthorization.issue(audit,platform,save)
    assert paths[unsupported_index] in checked

def test_lock_competes_for_a_fixed_region(tmp_path):
    lock_path=tmp_path/"fixed.lock"
    with lock_path.open("a+b") as first, lock_path.open("a+b") as second:
        launch_lifecycle._lock(first)
        with pytest.raises((BlockingIOError,OSError)):
            launch_lifecycle._lock(second)

def test_win64_lock_uses_nonblocking_byte_zero_region(tmp_path,monkeypatch):
    calls=[]
    fake=SimpleNamespace(LK_NBLCK=7,locking=lambda fd,mode,count:calls.append((fd,mode,count)))
    monkeypatch.setattr(launch_lifecycle,"fcntl",None)
    monkeypatch.setattr(launch_lifecycle,"msvcrt",fake,raising=False)
    lock_path=tmp_path/"win64-fixed.lock"
    with lock_path.open("a+b") as stream:
        launch_lifecycle._lock(stream)
        assert stream.tell()==0
        assert calls==[(stream.fileno(),fake.LK_NBLCK,1)]

def test_non_supported_fixed_store_rejects_commit(tmp_path):
    config=tmp_path/"launch_store.json"
    config.write_text('{"schema_version":"launch_store.v1","root":"'+str(tmp_path/'fixed')+'","local_fixed_volume":true,"ntfs":false}')
    launch_lifecycle.LAUNCH_STORE_CONFIG_PATH=config
    ledger=LaunchIntentStore.for_campaign("unsupported")
    with pytest.raises(ValueError, match="support envelope"):
        LaunchLifecycle.begin("a").reserve("r","g","n",authorization=auth(tmp_path,"a"),store=ledger)


def test_prelaunch_and_uncertain_failures_do_not_mix_with_outcomes(tmp_path):
    failed = LaunchLifecycle.begin("a").preflight_failure("cloud_sync_unknown")
    assert failed.reserved_run_id is None and failed.process_ref is None
    ledger=store(tmp_path,"campaign-failures")
    gate = LaunchLifecycle.begin("b").reserve("r", "g", "n", authorization=auth(tmp_path,"b"),store=ledger)
    gate = gate.create_process_failed(store=ledger)
    assert gate.state is LaunchState.LAUNCH_GATE_FAILED
    assert not gate.counts_toward_outcome_denominator
    uncertain = LaunchLifecycle.begin("c").reserve("r2", "g2", "n2", authorization=auth(tmp_path,"c"),store=ledger)
    uncertain = uncertain.launch_uncertain(store=ledger)
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
        LaunchLifecycle.begin("a").reserve("r","g","n",authorization=auth(tmp_path,"a"),store=None)

@pytest.mark.parametrize(("platform","expected"), [("nt", True), ("posix", True), ("unsupported", False)])
def test_launch_intent_durability_platform_contract(tmp_path, monkeypatch, platform, expected):
    path=tmp_path/f"{platform}.jsonl"; path.write_bytes(b"intent")
    calls=[]
    monkeypatch.setattr(os,"fsync",lambda fd:calls.append(fd))
    if expected:
        _sync_launch_intent(path, platform=platform)
        assert calls
    else:
        with pytest.raises(OSError, match="unsupported"):
            _sync_launch_intent(path, platform=platform)
