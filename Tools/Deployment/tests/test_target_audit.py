import copy
from pathlib import Path
import pytest
from reinbalance_survivors_contracts.canonical_json import canonical_hash
from survivors.target_audit import AuditError, SaveLifecycle, audit_target
from survivors.target_profile import load_target_profile

def resolved():
    data=load_target_profile().to_wire()
    data["build"].update(build_id="steam-1794680-measured",executable_version="measured-v1",executable_hash="1"*64,manual_attestation={"operator":"op","date":"2026-07-22","evidence_hash":"2"*64,"fields":["build_id"],"unavailable_reason":"store API unavailable"})
    data["progression"].update(save_artifact_hash="3"*64,save_format_version="measured-v1")
    data["hardware"].update(profile_id="reference-1",os_build="26100",gpu_name="reference-gpu",vram_mb=12288,driver_version="1",cuda_version="12.4",pytorch_version="2.5")
    return data

@pytest.mark.parametrize("path", [("build","build_id"),("progression","save_artifact_hash"),("progression","purchased_power_ups"),("display","ui_scale"),("display","windows_dpi_scale"),("input","key_bindings"),("hardware","gpu_name"),("choice_taxonomy","candidate_vocabulary_hash")])
def test_exact_audit_rejects_missing_or_different_fields(path):
    expected=resolved(); actual=copy.deepcopy(expected); del actual[path[0]][path[1]]
    with pytest.raises(AuditError):audit_target(expected,actual)

def test_placeholder_and_missing_measured_build_never_pass():
    template=load_target_profile().to_wire()
    with pytest.raises(AuditError):audit_target(template,template)
    expected=resolved(); actual=copy.deepcopy(expected); actual["build"]["executable_hash"]=None
    with pytest.raises(AuditError):audit_target(expected,actual)

@pytest.mark.parametrize("override", [{"game_stopped":False},{"launcher_stopped":False},{"cloud_sync_disabled":None},{"restore_conflict":True},{"canonical_hash_matches":False}])
def test_save_preflight_failure_has_attempt_only(override):
    life=SaveLifecycle.valid(); values=life.to_wire(); values.update(override)
    result=SaveLifecycle.from_wire(values).preflight("attempt")
    assert not result.can_reserve and result.reserved_run_id is None and result.process_ref is None

def test_real_atomic_backup_restore_and_hashes(tmp_path):
    target=tmp_path/"save"; target.write_bytes(b"original")
    canonical=tmp_path/"canonical"; canonical.write_bytes(b"canonical")
    oh=canonical_hash({"bytes_hex":b"original".hex()}); ch=canonical_hash({"bytes_hex":b"canonical".hex()})
    life=SaveLifecycle(True,True,True,False,True,True,oh,ch); backup=tmp_path/"backup"
    life.create_original_backup(target,backup); life.atomic_restore(canonical,target)
    assert target.read_bytes()==b"canonical" and life.record_post_run(target)["post_run_hash"]==ch
    life.restore_original(backup,target); assert target.read_bytes()==b"original"

def test_backup_conflict_and_hash_mismatch_fail_closed(tmp_path):
    target=tmp_path/"save"; target.write_bytes(b"original"); backup=tmp_path/"backup"; backup.write_bytes(b"exists")
    life=SaveLifecycle(True,True,True,False,True,True,"a"*64,"b"*64)
    with pytest.raises(AuditError):life.create_original_backup(target,backup)
    with pytest.raises(AuditError):life.atomic_restore(target,tmp_path/"restored")
