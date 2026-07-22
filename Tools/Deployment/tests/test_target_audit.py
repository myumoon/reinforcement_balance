import copy
import yaml
import pytest
from reinbalance_survivors_contracts.canonical_json import canonical_hash
from reinbalance_survivors_contracts.launch_lifecycle import AuditVerdict, _verify_audit_evidence
from survivors.target_audit import AuditError, AuditEvidence, SaveLifecycle, audit_target
from survivors.target_profile import load_target_profile

ATTEST_BASE={"operator":"op","date":"2026-07-22","fields":["build_id","executable_version","os_build","gpu_name","vram_mb","driver_version","cuda_version","pytorch_version"],"unavailable_reason":"store and driver APIs unavailable"}

def file_hash(data:bytes): return canonical_hash({"bytes_hex":data.hex()})

def lifecycle_audit(attempt,life):
    evidence=b"audit"
    semantics={"canonical_save_hash":life.canonical_save_hash,"save_format_version":life.expected_save_format_version,"progression":dict(life.expected_progression)}
    return _verify_audit_evidence(attempt_id=attempt,target_identity_hash="f"*64,canonical_save_hash=life.canonical_save_hash,save_semantics_hash=canonical_hash(semantics),evidence_bytes=evidence,expected_evidence_hash=file_hash(evidence))

def resolved(tmp_path):
    exe=tmp_path/"game.exe"; exe.write_bytes(b"executable")
    save=tmp_path/"save.dat"; save.write_bytes(b"canonical-save")
    data=load_target_profile().to_wire()
    data["provenance"]="operator-attested"
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

def test_audit_rejects_noncanonical_expected_even_when_actual_matches(tmp_path):
    expected,evidence=resolved(tmp_path)
    expected["base"]["stage"]="different_stage"
    with pytest.raises(AuditError,match="canonical target profile"):
        audit_target(expected,copy.deepcopy(expected),evidence,attempt_id="attempt-1")

@pytest.mark.parametrize(("section","field","value"), [
    ("build","build_id","different-build"),
    ("progression","save_artifact_hash","f"*64),
    ("hardware","profile_id","different-hardware"),
])
def test_measured_identity_is_bound_to_canonical_profile(tmp_path,section,field,value):
    expected,evidence=resolved(tmp_path)
    expected[section][field]=value
    actual=copy.deepcopy(expected)
    with pytest.raises(AuditError):
        audit_target(expected,actual,evidence,attempt_id="attempt-1")

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
    semantic=tmp_path/"semantic.yaml"
    semantic.write_text(yaml.safe_dump({"schema_version":"survivors_save_semantics.v1","canonical_save_hash":ch,"save_format_version":"measured-v1","progression":{"unlocked_characters":["antonio"],"unlocked_items":["whip"],"unlocked_stages":["mad_forest"],"collection_pool":["whip"],"purchased_power_ups":[],"reroll_count":0,"skip_count":0,"banish_count":0}}))
    life=SaveLifecycle(True,True,True,False,True,oh,ch,semantic_attestation_path=semantic,expected_save_format_version="measured-v1",expected_progression=yaml.safe_load(semantic.read_text())["progression"],record_path=tmp_path/"lifecycle.jsonl"); backup=tmp_path/"backup"
    life.create_original_backup(target,backup,"a1"); life.atomic_restore(canonical,target,"a1")
    assert life.verified_verdict(lifecycle_audit("a1",life)).pre_run_hash==ch
    target.write_bytes(b"post-run"); life.record_post_run(target,"a1")
    life.restore_original(backup,target,"a1")
    assert target.read_bytes()==b"original" and {r["stage"] for r in life.records}>={"ORIGINAL_BACKUP","CANONICAL_RESTORE","PRE_RUN_AUDIT","POST_RUN","ORIGINAL_RESTORE"}

def test_interruption_conflict_and_restore_gate_fail_closed(tmp_path,monkeypatch):
    target=tmp_path/"save"; target.write_bytes(b"original"); backup=tmp_path/"backup"; backup.write_bytes(b"exists")
    life=SaveLifecycle.valid()
    with pytest.raises(AuditError): life.create_original_backup(target,backup,"a")
    with pytest.raises(AuditError): life.atomic_restore(target,tmp_path/"restored","a")
    blocked=SaveLifecycle.from_wire({**life.to_wire(),"game_stopped":False})
    with pytest.raises(AuditError): blocked.restore_original(target,tmp_path/"restored","a")
    with pytest.raises(AuditError): life.verified_verdict(lifecycle_audit("interrupted",life))

def test_in_memory_or_tampered_lifecycle_cannot_issue_pass(tmp_path):
    life=SaveLifecycle.valid(); life.records=[{"attempt_id":"a","stage":s,"verdict":"PASS","hash":"c"*64} for s in ("PREFLIGHT","ORIGINAL_BACKUP","CANONICAL_RESTORE","PRE_RUN_AUDIT")]
    with pytest.raises(AuditError): life.verified_verdict(lifecycle_audit("a",life))

def test_semantic_bool_shortcut_or_missing_evidence_cannot_pass():
    wire=SaveLifecycle.valid().to_wire()
    assert "semantic_attestation_valid" not in wire
    with pytest.raises(ValueError): SaveLifecycle.from_wire({**wire,"semantic_attestation_valid":True})
    assert not SaveLifecycle.valid().preflight("a").can_reserve

def test_semantic_attestation_must_match_canonical_hash_and_progression(tmp_path):
    record={"schema_version":"survivors_save_semantics.v1","canonical_save_hash":"b"*64,"save_format_version":"v1","progression":{"unlocked_items":["whip"]}}
    path=tmp_path/"semantic.yaml"; path.write_text(yaml.safe_dump(record))
    life=SaveLifecycle(True,True,True,False,True,"a"*64,"c"*64,semantic_attestation_path=path,expected_save_format_version="v1",expected_progression=record["progression"])
    result=life.preflight("a")
    assert not result.can_reserve and result.failure_reason=="semantic_attestation_invalid"

@pytest.mark.parametrize("kind",["unc","removable","non_ntfs"])
def test_every_save_copy_path_rejects_unsupported_volume(tmp_path,monkeypatch,kind):
    life=SaveLifecycle.valid(); source=tmp_path/"source"; source.write_bytes(b"x")
    monkeypatch.setattr("survivors.target_audit._validate_save_volume",lambda path: (_ for _ in ()).throw(AuditError(kind)))
    for operation,args in ((life.create_original_backup,(source,tmp_path/"backup")),(life.atomic_restore,(source,tmp_path/"target")),(life.restore_original,(source,tmp_path/"target"))):
        with pytest.raises(AuditError,match=kind): operation(*args)
    with pytest.raises(AuditError,match=kind): life.record_post_run(source)
    path=tmp_path/"records.jsonl"; path.write_text('{"attempt_id":"a","stage":"PREFLIGHT","verdict":"PASS"}\n')
    life.record_path=path
    with pytest.raises(AuditError): life.verified_verdict(lifecycle_audit("a",life))

def test_save_verdict_rejects_audited_profile_with_different_canonical_save(tmp_path):
    target=tmp_path/"save"; target.write_bytes(b"original")
    canonical=tmp_path/"canonical"; canonical.write_bytes(b"canonical")
    semantic=tmp_path/"semantic.yaml"; ch=file_hash(b"canonical")
    progression={"unlocked_items":["whip"]}
    semantic.write_text(yaml.safe_dump({"schema_version":"survivors_save_semantics.v1","canonical_save_hash":ch,"save_format_version":"v1","progression":progression}))
    life=SaveLifecycle(True,True,True,False,True,file_hash(b"original"),ch,semantic_attestation_path=semantic,expected_save_format_version="v1",expected_progression=progression,record_path=tmp_path/"lifecycle.jsonl")
    life.create_original_backup(target,tmp_path/"backup","a"); life.atomic_restore(canonical,target,"a")
    wrong=_verify_audit_evidence(attempt_id="a",target_identity_hash="f"*64,canonical_save_hash="e"*64,save_semantics_hash="d"*64,evidence_bytes=b"audit",expected_evidence_hash=file_hash(b"audit"))
    with pytest.raises(AuditError,match="audited canonical profile"):
        life.verified_verdict(wrong)

def test_audit_target_rejects_unsupported_save_volume(tmp_path,monkeypatch):
    expected,evidence=resolved(tmp_path)
    monkeypatch.setattr("survivors.target_audit._validate_save_volume",lambda path: (_ for _ in ()).throw(AuditError("non_ntfs")))
    with pytest.raises(AuditError,match="non_ntfs"):
        audit_target(expected,copy.deepcopy(expected),evidence,attempt_id="a")
