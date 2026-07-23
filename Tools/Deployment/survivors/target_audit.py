from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any,Mapping
import os, re, shutil, tempfile, yaml
from datetime import datetime, timezone
from reinbalance_survivors_contracts.canonical_json import canonical_hash, canonical_json_bytes
from reinbalance_survivors_contracts.launch_lifecycle import AuditVerdict, SaveVerdict, _verify_audit_evidence, _finalize_save_execution, _validate_store_volume
from .target_profile import TargetProfile
from .target_profile import load_target_profile

class AuditError(ValueError):pass
_SHA256=re.compile(r"[0-9a-f]{64}")
_PLACEHOLDER=re.compile(r"(TEST_FIXTURE|FIXTURE|MANUAL|REQUIRED|PLACEHOLDER)",re.IGNORECASE)
_ATTESTED_PROFILE_FIELDS=frozenset({
    ("build","build_id"), ("build","executable_version"),
    ("progression","save_format_version"),
    *(("hardware",key) for key in ("os_build","gpu_name","vram_mb","driver_version","cuda_version","pytorch_version")),
})
_FILE_DERIVED_PROFILE_FIELDS=frozenset({("build","executable_hash"),("progression","save_artifact_hash")})
_CANONICAL_EXEMPT_FIELDS=_FILE_DERIVED_PROFILE_FIELDS|frozenset({("build","manual_attestation")})

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

def _assert_canonical_identity_finalized(canonical:Mapping[str,Any])->None:
    if canonical.get("provenance")!="operator-attested":
        raise AuditError("canonical target identity not finalized; cannot issue operator-attested audit verdict")
    for section,field in _ATTESTED_PROFILE_FIELDS:
        value=canonical[section][field]
        if value is None or isinstance(value,str) and (not value.strip() or _PLACEHOLDER.search(value)):
            raise AuditError("canonical target identity not finalized; cannot issue operator-attested audit verdict")

@dataclass(frozen=True)
class AuditEvidence:
    executable_path:Path; save_path:Path; manual_attestation:Mapping[str,Any]; manual_evidence_path:Path

def _file_evidence(path:Path)->Mapping[str,Any]:
    if not path.is_file(): raise AuditError(f"evidence file does not exist: {path}")
    stat=path.stat()
    return {"path":str(path.resolve()),"size":stat.st_size,"mtime_ns":stat.st_mtime_ns,"content_hash":_file_hash(path)}

def audit_target(expected:Mapping[str,Any],actual:Mapping[str,Any],evidence:AuditEvidence,*,attempt_id:str)->AuditVerdict:
    # Both sides must first satisfy the closed schema/taxonomy contract.
    try: TargetProfile.from_wire(expected); TargetProfile.from_wire(actual)
    except (ValueError,KeyError,TypeError) as exc: raise AuditError(str(exc)) from exc
    canonical=load_target_profile().to_wire()
    _assert_canonical_identity_finalized(canonical)
    if expected["provenance"]!="operator-attested" or actual["provenance"]!="operator-attested": raise AuditError("real target requires an operator-attested canonical profile")
    for section in canonical:
        if section in {"schema_version","provenance"}:
            if section=="schema_version" and expected[section]!=canonical[section]: raise AuditError("expected is not the canonical target profile")
            continue
        for field,value in canonical[section].items():
            # Operator/date/evidence identify only the measurement event. Attested
            # values are pinned here and cannot be selected by the audit caller.
            if (section,field) not in _CANONICAL_EXEMPT_FIELDS and expected[section][field]!=value:
                raise AuditError("expected is not the canonical target profile")
    _assert_resolved(expected); _assert_resolved(actual)
    if not isinstance(evidence,AuditEvidence) or not _valid_manual(evidence.manual_attestation): raise AuditError("operator/date/evidence-hash attestation required")
    required_attested={field for _,field in _ATTESTED_PROFILE_FIELDS}
    if not required_attested<=set(evidence.manual_attestation["fields"]): raise AuditError("build/hardware/save format attestation coverage incomplete")
    if expected["build"]["manual_attestation"]!=evidence.manual_attestation or actual["build"]["manual_attestation"]!=evidence.manual_attestation: raise AuditError("attestation is not bound to profile")
    if not evidence.manual_evidence_path.is_file() or canonical_hash({"bytes_hex":evidence.manual_evidence_path.read_bytes().hex()})!=evidence.manual_attestation["evidence_hash"]: raise AuditError("manual attestation evidence bytes mismatch")
    try: manual_content=yaml.safe_load(evidence.manual_evidence_path.read_text(encoding="utf-8"))
    except (OSError,UnicodeDecodeError,yaml.YAMLError) as exc: raise AuditError("manual evidence is not parseable") from exc
    claimed={
        "build_id":actual["build"]["build_id"], "executable_version":actual["build"]["executable_version"],
        "save_format_version":actual["progression"]["save_format_version"],
        **{key:actual["hardware"][key] for key in ("os_build","gpu_name","vram_mb","driver_version","cuda_version","pytorch_version")},
    }
    if not isinstance(manual_content,Mapping) or set(manual_content)!={"schema_version","measurements"} or manual_content.get("schema_version")!="survivors_manual_attestation.v1" or manual_content.get("measurements")!=claimed:
        raise AuditError("manual evidence content does not match attested build/hardware/save format")
    _validate_save_volume(evidence.save_path)
    executable=_file_evidence(evidence.executable_path); save=_file_evidence(evidence.save_path)
    if executable["content_hash"]!=actual["build"]["executable_hash"] or save["content_hash"]!=actual["progression"]["save_artifact_hash"]: raise AuditError("observed file identity mismatch")
    differences=[]
    for section in expected:
        if section in {"schema_version","provenance"}:continue
        for field,value in expected[section].items():
            observed=actual[section][field]
            if observed!=value:differences.append(f"{section}.{field}")
    if differences:raise AuditError("target mismatch: "+", ".join(differences))
    observed_evidence={"executable":executable,"save":save,"attestation":dict(evidence.manual_attestation),"observed_profile_hash":canonical_hash(actual)}
    evidence_bytes=canonical_json_bytes(observed_evidence)
    evidence_hash=canonical_hash({"bytes_hex":evidence_bytes.hex()})
    semantics={"canonical_save_hash":expected["progression"]["save_artifact_hash"],"save_format_version":expected["progression"]["save_format_version"],"progression":{k:v for k,v in expected["progression"].items() if k not in {"save_artifact_hash","save_format_version"}}}
    return _verify_audit_evidence(attempt_id=attempt_id,target_identity_hash=canonical_hash(expected),canonical_save_hash=expected["progression"]["save_artifact_hash"],save_semantics_hash=canonical_hash(semantics),evidence_bytes=evidence_bytes,expected_evidence_hash=evidence_hash)

@dataclass(frozen=True)
class PreflightResult:
    attempt_id:str; can_reserve:bool; failure_reason:str|None=None; reserved_run_id:str|None=None; process_ref:str|None=None

@dataclass
class SaveLifecycle:
    game_stopped:bool; launcher_stopped:bool; cloud_sync_disabled:bool|None; restore_conflict:bool
    canonical_hash_matches:bool; original_backup_hash:str; canonical_save_hash:str
    semantic_attestation_path:Path|None=None; expected_save_format_version:str=""; expected_progression:Mapping[str,Any]=field(default_factory=dict)
    records:list[Mapping[str,Any]]=field(default_factory=list,repr=False)
    record_path:Path|None=field(default=None,repr=False)
    canonical_path:Path|None=field(default=None,repr=False)
    backup_path:Path|None=field(default=None,repr=False)
    _KEYS=frozenset({"game_stopped","launcher_stopped","cloud_sync_disabled","restore_conflict","canonical_hash_matches","original_backup_hash","canonical_save_hash","semantic_attestation_path","expected_save_format_version","expected_progression"})
    @classmethod
    def valid(cls):return cls(True,True,True,False,True,"a"*64,"c"*64)
    def to_wire(self):return {k:(str(getattr(self,k)) if k=="semantic_attestation_path" and getattr(self,k) is not None else getattr(self,k)) for k in self._KEYS}
    @classmethod
    def from_wire(cls,data:Mapping[str,Any]):
        if not isinstance(data,Mapping) or set(data)!=cls._KEYS:raise ValueError("unknown/missing save lifecycle field")
        values=dict(data); values["semantic_attestation_path"]=Path(values["semantic_attestation_path"]) if values["semantic_attestation_path"] is not None else None
        obj=cls(**values)
        if any(type(getattr(obj,k)) is not bool for k in ("game_stopped","launcher_stopped","restore_conflict","canonical_hash_matches")) or obj.cloud_sync_disabled is not None and type(obj.cloud_sync_disabled) is not bool: raise ValueError("invalid save gate type")
        if not all(isinstance(getattr(obj,k),str) and _SHA256.fullmatch(getattr(obj,k)) for k in ("original_backup_hash","canonical_save_hash")): raise ValueError("invalid save hash")
        return obj
    def preflight(self,attempt_id:str)->PreflightResult:
        checks={"game_process_running":self.game_stopped is not True,"launcher_running":self.launcher_stopped is not True,"cloud_sync_unknown_or_enabled":self.cloud_sync_disabled is not True,"restore_conflict":self.restore_conflict is not False,"canonical_hash_mismatch":self.canonical_hash_matches is not True,"semantic_attestation_invalid":not self._semantic_evidence_valid()}
        failed=next((k for k,v in checks.items() if v),None)
        self._record(attempt_id,"PREFLIGHT",None,failed or "PASS")
        return PreflightResult(attempt_id,failed is None,failed)
    def _semantic_evidence_valid(self)->bool:
        try:
            if self.semantic_attestation_path is None or not self.semantic_attestation_path.is_file(): return False
            record=yaml.safe_load(self.semantic_attestation_path.read_text(encoding="utf-8"))
            return isinstance(record,Mapping) and set(record)=={"schema_version","canonical_save_hash","save_format_version","progression"} and record["schema_version"]=="survivors_save_semantics.v1" and record["canonical_save_hash"]==self.canonical_save_hash and record["save_format_version"]==self.expected_save_format_version and record["progression"]==self.expected_progression
        except (OSError,UnicodeDecodeError,yaml.YAMLError): return False
    def _record(self,attempt_id,stage,content_hash,verdict):
        previous=self.records[-1]["step_hash"] if self.records else None
        body={"attempt_id":attempt_id,"timestamp":datetime.now(timezone.utc).isoformat(),"stage":stage,"content_hash":content_hash,"result":verdict,"previous_step_hash":previous}
        record={**body,"step_hash":canonical_hash(body)}
        self.records.append(record)
        if self.record_path is not None:
            self.record_path.parent.mkdir(parents=True,exist_ok=True)
            with self.record_path.open("ab") as stream:
                stream.write(canonical_json_bytes(record)+b"\n"); stream.flush(); os.fsync(stream.fileno())
    @property
    def lifecycle_hash(self): return canonical_hash(self.records)
    def verified_verdict(self,audit:AuditVerdict)->SaveVerdict:
        if self.record_path is None or not self.record_path.is_file(): raise AuditError("durable lifecycle evidence required")
        expected=b"".join(canonical_json_bytes(r)+b"\n" for r in self.records)
        if self.record_path.read_bytes()!=expected: raise AuditError("durable lifecycle chain mismatch")
        if self.canonical_path is None or self.backup_path is None: raise AuditError("canonical save and original backup evidence required")
        semantics={"canonical_save_hash":self.canonical_save_hash,"save_format_version":self.expected_save_format_version,"progression":dict(self.expected_progression)}
        try:return _finalize_save_execution(records=tuple(self.records),audit=audit,expected_pre_run_hash=self.canonical_save_hash,save_semantics_hash=canonical_hash(semantics),lifecycle_record_path=self.record_path,canonical_save_path=self.canonical_path,original_backup_path=self.backup_path)
        except (TypeError,ValueError) as exc: raise AuditError(str(exc)) from exc
    def record_post_run(self,target:Path,attempt_id="run"):
        _validate_save_volume(target)
        post=_file_hash(target); self._record(attempt_id,"POST_RUN",post,"PASS")
        return {"pre_run_hash":self.canonical_save_hash,"post_run_hash":post,"rng_control":"uncontrolled","lifecycle_hash":self.lifecycle_hash}
    def create_original_backup(self,target:Path,backup:Path,attempt_id="run")->str:
        _validate_save_volume(target); _validate_save_volume(backup)
        if not self.preflight(attempt_id).can_reserve: raise AuditError("backup gate failed")
        if backup.exists(): raise AuditError("original backup already exists")
        self._atomic_copy(target,backup,_file_hash(target))
        actual=_file_hash(backup)
        if actual!=self.original_backup_hash: raise AuditError("original backup identity mismatch")
        self._record(attempt_id,"ORIGINAL_BACKUP",actual,"PASS")
        self.backup_path=backup.resolve()
        return actual
    def restore_original(self,backup:Path,target:Path,attempt_id="run")->None:
        _validate_save_volume(backup); _validate_save_volume(target)
        if not self.preflight(attempt_id).can_reserve: raise AuditError("original restore gate failed")
        self._atomic_copy(backup,target,self.original_backup_hash)
        if _file_hash(target)!=self.original_backup_hash: raise AuditError("original restore verification failed")
        self._record(attempt_id,"ORIGINAL_RESTORE",self.original_backup_hash,"PASS")
    def atomic_restore(self,canonical:Path,target:Path,attempt_id="run")->None:
        _validate_save_volume(canonical); _validate_save_volume(target)
        if not self.preflight(attempt_id).can_reserve:raise AuditError("restore gate failed")
        expected=_file_hash(canonical)
        if expected!=self.canonical_save_hash:raise AuditError("canonical bytes hash mismatch")
        self._atomic_copy(canonical,target,expected)
        self._record(attempt_id,"CANONICAL_RESTORE",expected,"PASS")
        self._record(attempt_id,"PRE_RUN_AUDIT",_file_hash(target),"PASS")
        # Authorization re-reads the live canonical save location, not merely the source artifact.
        self.canonical_path=target.resolve()
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

def _validate_save_volume(path:Path)->None:
    try:_validate_store_volume(path.resolve() if not str(path).startswith(("//","\\\\")) else path)
    except ValueError as exc: raise AuditError(str(exc)) from exc
