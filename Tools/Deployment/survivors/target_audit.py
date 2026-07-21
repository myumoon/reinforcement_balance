from __future__ import annotations
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any,Mapping
import os, re, shutil, tempfile
from reinbalance_survivors_contracts.canonical_json import canonical_hash
from .target_profile import TargetProfile

class AuditError(ValueError):pass
_SHA256=re.compile(r"[0-9a-f]{64}")
_PLACEHOLDER=re.compile(r"(MANUAL|REQUIRED|PLACEHOLDER)")

def _file_hash(path:Path)->str:return canonical_hash({"bytes_hex":path.read_bytes().hex()})

def _valid_manual(value:Any)->bool:
    return isinstance(value,dict) and set(value)=={"operator","date","evidence_hash","fields","unavailable_reason"} and isinstance(value["operator"],str) and bool(value["operator"]) and isinstance(value["date"],str) and bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}",value["date"])) and isinstance(value["evidence_hash"],str) and bool(_SHA256.fullmatch(value["evidence_hash"])) and isinstance(value["fields"],list) and bool(value["fields"]) and all(isinstance(x,str) and x for x in value["fields"]) and isinstance(value["unavailable_reason"],str) and bool(value["unavailable_reason"])

def _assert_resolved(profile:Mapping[str,Any])->None:
    def values(x):
        if isinstance(x,Mapping):
            for v in x.values(): yield from values(v)
        elif isinstance(x,list):
            for v in x: yield from values(v)
        else: yield x
    if any(v is None or isinstance(v,str) and _PLACEHOLDER.search(v) for v in values(profile)): raise AuditError("target profile is not materialized from measured evidence")
    for section,field in (("build","executable_hash"),("progression","save_artifact_hash")):
        if not isinstance(profile[section][field],str) or not _SHA256.fullmatch(profile[section][field]): raise AuditError(f"invalid measured {section}.{field}")

def audit_target(expected:Mapping[str,Any],actual:Mapping[str,Any]):
    # Both sides must first satisfy the closed schema/taxonomy contract.
    try: TargetProfile.from_wire(expected); TargetProfile.from_wire(actual)
    except (ValueError,KeyError,TypeError) as exc: raise AuditError(str(exc)) from exc
    _assert_resolved(expected); _assert_resolved(actual)
    if not _valid_manual(expected["build"]["manual_attestation"]): raise AuditError("complete build attestation required")
    differences=[]
    for section in expected:
        if section=="schema_version":continue
        for field,value in expected[section].items():
            observed=actual[section][field]
            if observed!=value:differences.append(f"{section}.{field}")
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
        if not isinstance(data,Mapping) or set(data)!=cls._KEYS:raise ValueError("unknown/missing save lifecycle field")
        obj=cls(**data)
        if any(type(getattr(obj,k)) is not bool for k in ("game_stopped","launcher_stopped","restore_conflict","canonical_hash_matches","semantic_attestation_valid")) or obj.cloud_sync_disabled is not None and type(obj.cloud_sync_disabled) is not bool: raise ValueError("invalid save gate type")
        if not all(isinstance(getattr(obj,k),str) and _SHA256.fullmatch(getattr(obj,k)) for k in ("original_backup_hash","canonical_save_hash")): raise ValueError("invalid save hash")
        return obj
    def preflight(self,attempt_id:str)->PreflightResult:
        checks={"game_process_running":self.game_stopped is not True,"launcher_running":self.launcher_stopped is not True,"cloud_sync_unknown_or_enabled":self.cloud_sync_disabled is not True,"restore_conflict":self.restore_conflict is not False,"canonical_hash_mismatch":self.canonical_hash_matches is not True,"semantic_attestation_invalid":self.semantic_attestation_valid is not True}
        failed=next((k for k,v in checks.items() if v),None)
        return PreflightResult(attempt_id,failed is None,failed)
    def record_post_run(self,target:Path):return {"pre_run_hash":self.canonical_save_hash,"post_run_hash":_file_hash(target),"rng_control":"uncontrolled"}
    def create_original_backup(self,target:Path,backup:Path)->str:
        if backup.exists(): raise AuditError("original backup already exists")
        self._atomic_copy(target,backup,_file_hash(target))
        actual=_file_hash(backup)
        if actual!=self.original_backup_hash: raise AuditError("original backup identity mismatch")
        return actual
    def restore_original(self,backup:Path,target:Path)->None:
        self._atomic_copy(backup,target,self.original_backup_hash)
    def atomic_restore(self,canonical:Path,target:Path)->None:
        if not self.preflight("restore").can_reserve:raise AuditError("restore gate failed")
        expected=_file_hash(canonical)
        if expected!=self.canonical_save_hash:raise AuditError("canonical bytes hash mismatch")
        self._atomic_copy(canonical,target,expected)
    @staticmethod
    def _atomic_copy(source:Path,target:Path,expected:str)->None:
        target.parent.mkdir(parents=True,exist_ok=True)
        fd,name=tempfile.mkstemp(dir=target.parent,prefix=target.name+".",suffix=".tmp");os.close(fd)
        temp=Path(name)
        try:
            shutil.copyfile(source,temp)
            with temp.open("rb+") as stream: os.fsync(stream.fileno())
            if _file_hash(temp)!=expected:raise AuditError("temporary copy hash mismatch")
            os.replace(temp,target)
            with target.open("rb") as stream: os.fsync(stream.fileno())
            if _file_hash(target)!=expected:raise AuditError("atomic replacement hash mismatch")
            if os.name!="nt":
                dfd=os.open(target.parent,os.O_RDONLY)
                try:os.fsync(dfd)
                finally:os.close(dfd)
        finally:
            if temp.exists():temp.unlink()
