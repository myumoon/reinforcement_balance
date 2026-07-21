from __future__ import annotations
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any,Mapping
import os, shutil, tempfile
from reinbalance_survivors_contracts.canonical_json import canonical_hash
from .target_profile import TargetProfile

class AuditError(ValueError):pass

def _valid_manual(value:Any)->bool:
    return isinstance(value,dict) and set(value)=={"operator","date","evidence_hash"} and all(isinstance(value[k],str) and value[k] for k in value) and len(value["evidence_hash"])==64

def audit_target(expected:Mapping[str,Any],actual:Mapping[str,Any]):
    # Both sides must first satisfy the closed schema/taxonomy contract.
    try: TargetProfile.from_wire(expected); TargetProfile.from_wire(actual)
    except (ValueError,KeyError,TypeError) as exc: raise AuditError(str(exc)) from exc
    differences=[]
    for section in expected:
        if section=="schema_version":continue
        for field,value in expected[section].items():
            observed=actual[section][field]
            if observed!=value:
                if section=="build" and field=="manual_attestation" and _valid_manual(observed):continue
                if section=="build" and field in {"build_id","executable_version","executable_hash"} and observed is None and _valid_manual(actual["build"]["manual_attestation"]):continue
                differences.append(f"{section}.{field}")
    if differences:raise AuditError("target mismatch: "+", ".join(differences))
    report={"schema_version":"target_audit.v1","target_hash":canonical_hash(expected),"status":"PASS"}
    return {**report,"audit_hash":canonical_hash(report)}

@dataclass(frozen=True)
class PreflightResult:
    attempt_id:str; can_reserve:bool; failure_reason:str|None=None; reserved_run_id:str|None=None; process_ref:str|None=None

@dataclass(frozen=True)
class SaveLifecycle:
    game_stopped:bool; launcher_stopped:bool; cloud_sync_disabled:bool|None; restore_conflict:bool
    canonical_hash_matches:bool; semantic_attestation_valid:bool; original_backup_hash:str; canonical_save_hash:str
    _KEYS=frozenset({"game_stopped","launcher_stopped","cloud_sync_disabled","restore_conflict","canonical_hash_matches","semantic_attestation_valid","original_backup_hash","canonical_save_hash"})
    @classmethod
    def valid(cls):return cls(True,True,True,False,True,True,"a"*64,"c"*64)
    def to_wire(self):return dict(self.__dict__)
    @classmethod
    def from_wire(cls,data:Mapping[str,Any]):
        if set(data)!=cls._KEYS:raise ValueError("unknown/missing save lifecycle field")
        return cls(**data)
    def preflight(self,attempt_id:str)->PreflightResult:
        checks={"game_process_running":self.game_stopped is not True,"launcher_running":self.launcher_stopped is not True,"cloud_sync_unknown_or_enabled":self.cloud_sync_disabled is not True,"restore_conflict":self.restore_conflict is not False,"canonical_hash_mismatch":self.canonical_hash_matches is not True,"semantic_attestation_invalid":self.semantic_attestation_valid is not True}
        failed=next((k for k,v in checks.items() if v),None)
        return PreflightResult(attempt_id,failed is None,failed)
    def record_post_run(self,post_hash:str):return {"pre_run_hash":self.canonical_save_hash,"post_run_hash":post_hash,"rng_control":"uncontrolled"}
    def atomic_restore(self,canonical:Path,target:Path)->None:
        if not self.preflight("restore").can_reserve:raise AuditError("restore gate failed")
        expected=canonical_hash({"bytes_hex":canonical.read_bytes().hex()})
        if expected!=self.canonical_save_hash:raise AuditError("canonical bytes hash mismatch")
        fd,name=tempfile.mkstemp(dir=target.parent,prefix=target.name+".",suffix=".tmp");os.close(fd)
        temp=Path(name)
        try:
            shutil.copyfile(canonical,temp)
            if canonical_hash({"bytes_hex":temp.read_bytes().hex()})!=expected:raise AuditError("temporary copy hash mismatch")
            os.replace(temp,target)
        finally:
            if temp.exists():temp.unlink()
