import copy
import yaml
import pytest
from reinbalance_survivors_contracts.canonical_json import canonical_hash
from reinbalance_survivors_contracts.launch_lifecycle import AuditVerdict
from survivors.target_audit import AuditError, AuditEvidence, SaveLifecycle, audit_target
from survivors.target_profile import load_target_profile

ATTEST_BASE={"operator":"op","date":"2026-07-22","fields":["build_id","executable_version","os_build","gpu_name","vram_mb","driver_version","cuda_version","pytorch_version"],"unavailable_reason":"store and driver APIs unavailable"}

def file_hash(data:bytes): return canonical_hash({"bytes_hex":data.hex()})

def resolved(tmp_path):
    exe=tmp_path/"game.exe"; exe.write_bytes(b"executable")
    save=tmp_path/"save.dat"; save.write_bytes(b"canonical-save")
    data=load_target_profile().to_wire()
    data["build"].update(build_id="steam-1794680-measured",executable_version="measured-v1",executable_hash=file_hash(exe.read_bytes()))
    data["progression"].update(save_artifact_hash=file_hash(save.read_bytes()),save_format_version="measured-v1")
    data["hardware"].update(profile_id="reference-1",os_build="26100",gpu_name="reference-gpu",vram_mb=12288,driver_version="1",cuda_version="12.4",pytorch_version="2.5")
    measurements={"build_id":data["build"]["build_id"],"executable_version":data["build"]["executable_version"],**{k:data["hardware"][k] for k in ("os_build","gpu_name","vram_mb","driver_version","cuda_version","pytorch_version")}}
    manual=tmp_path/"manual-evidence.yaml"; manual.write_text(yaml.safe_dump({"schema_version":"survivors_manual_attestation.v1","measurements":measurements},sort_keys=True))
    attest={**ATTEST_BASE,"evidence_hash":file_hash(manual.read_bytes())}; data["build"]["manual_attestation"]=attest
    return data,AuditEvidence(exe,save,attest,manual)

@pytest.mark.parametrize("path", [("build","build_id"),("progression","save_artifact_hash"),("progression","purchased_power_ups"),("display","ui_scale"),("display","windows_dpi_scale"),("input","key_bindings"),("hardware","gpu_name"),("choice_taxonomy","candidate_vocabulary_hash")])
def test_exact_audit_rejects_missing_or_different_fields(tmp_path,path):
    expected,evidence=resolved(tmp_path); actual=copy.deepcopy(expected); del actual[path[0]][path[1]]
    with pytest.raises(AuditError): audit_target(expected,actual,evidence,attempt_id="attempt-1")

def test_audit_is_typed_and_bound_to_real_file_bytes(tmp_path):
    expected,evidence=resolved(tmp_path)
    verdict=audit_target(expected,copy.deepcopy(expected),evidence,attempt_id="attempt-1")
    assert isinstance(verdict,AuditVerdict) and verdict.status=="PASS"
    evidence.executable_path.write_bytes(b"tampered")
    with pytest.raises(AuditError): audit_target(expected,copy.deepcopy(expected),evidence,attempt_id="attempt-1")

def test_placeholder_or_missing_attestation_never_passes(tmp_path):
    template=load_target_profile().to_wire(); _,evidence=resolved(tmp_path)
    with pytest.raises(AuditError): audit_target(template,template,evidence,attempt_id="attempt-1")
    expected,evidence=resolved(tmp_path)
    with pytest.raises(AuditError): audit_target(expected,copy.deepcopy(expected),AuditEvidence(evidence.executable_path,evidence.save_path,{},evidence.manual_evidence_path),attempt_id="attempt-1")
    evidence.manual_evidence_path.write_bytes(b"tampered")
    with pytest.raises(AuditError): audit_target(expected,copy.deepcopy(expected),evidence,attempt_id="attempt-1")

def test_manual_evidence_must_contain_claimed_values(tmp_path):
    expected,evidence=resolved(tmp_path)
    evidence.manual_evidence_path.write_text(yaml.safe_dump({"schema_version":"survivors_manual_attestation.v1","measurements":{"gpu_name":"different"}}))
    attest={**evidence.manual_attestation,"evidence_hash":file_hash(evidence.manual_evidence_path.read_bytes())}
    expected["build"]["manual_attestation"]=attest
    bad=AuditEvidence(evidence.executable_path,evidence.save_path,attest,evidence.manual_evidence_path)
    with pytest.raises(AuditError): audit_target(expected,copy.deepcopy(expected),bad,attempt_id="attempt-1")

@pytest.mark.parametrize("override", [{"game_stopped":False},{"launcher_stopped":False},{"cloud_sync_disabled":None},{"restore_conflict":True},{"canonical_hash_matches":False}])
def test_save_preflight_failure_has_attempt_only(override):
    life=SaveLifecycle.valid(); values=life.to_wire(); values.update(override)
    result=SaveLifecycle.from_wire(values).preflight("attempt")
    assert not result.can_reserve and result.reserved_run_id is None and result.process_ref is None

def test_durable_backup_restore_hashes_and_verdict(tmp_path):
    target=tmp_path/"save"; target.write_bytes(b"original")
    canonical=tmp_path/"canonical"; canonical.write_bytes(b"canonical")
    oh=file_hash(b"original"); ch=file_hash(b"canonical")
    life=SaveLifecycle(True,True,True,False,True,True,oh,ch,record_path=tmp_path/"lifecycle.jsonl"); backup=tmp_path/"backup"
    life.create_original_backup(target,backup,"a1"); life.atomic_restore(canonical,target,"a1")
    assert life.verified_verdict("a1","f"*64).pre_run_hash==ch
    target.write_bytes(b"post-run"); life.record_post_run(target,"a1")
    life.restore_original(backup,target,"a1")
    assert target.read_bytes()==b"original" and {r["stage"] for r in life.records}>={"ORIGINAL_BACKUP","CANONICAL_RESTORE","PRE_RUN_AUDIT","POST_RUN","ORIGINAL_RESTORE"}

def test_interruption_conflict_and_restore_gate_fail_closed(tmp_path,monkeypatch):
    target=tmp_path/"save"; target.write_bytes(b"original"); backup=tmp_path/"backup"; backup.write_bytes(b"exists")
    life=SaveLifecycle(True,True,True,False,True,True,"a"*64,"b"*64)
    with pytest.raises(AuditError): life.create_original_backup(target,backup,"a")
    with pytest.raises(AuditError): life.atomic_restore(target,tmp_path/"restored","a")
    blocked=SaveLifecycle(False,True,True,False,True,True,file_hash(b"original"),"b"*64)
    with pytest.raises(AuditError): blocked.restore_original(target,tmp_path/"restored","a")
    with pytest.raises(AuditError): life.verified_verdict("interrupted","f"*64)

def test_in_memory_or_tampered_lifecycle_cannot_issue_pass(tmp_path):
    life=SaveLifecycle.valid(); life.records=[{"attempt_id":"a","stage":s,"verdict":"PASS","hash":"c"*64} for s in ("PREFLIGHT","ORIGINAL_BACKUP","CANONICAL_RESTORE","PRE_RUN_AUDIT")]
    with pytest.raises(AuditError): life.verified_verdict("a","f"*64)
    path=tmp_path/"records.jsonl"; path.write_text('{"attempt_id":"a","stage":"PREFLIGHT","verdict":"PASS"}\n')
    life.record_path=path
    with pytest.raises(AuditError): life.verified_verdict("a","f"*64)
