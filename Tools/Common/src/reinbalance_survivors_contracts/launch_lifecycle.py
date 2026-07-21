"""Fail-closed formal-run identity and durable launch transitions."""
from __future__ import annotations
import enum, os, re, json
try:
    import fcntl
except ImportError:  # Win64 support envelope
    fcntl = None
    import msvcrt
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping
from .canonical_json import canonical_hash, canonical_json_bytes

SCHEMA_VERSION = "launch_lifecycle.v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_AUTH_SEAL = object()
_VERDICT_SEAL = object()

def _lock(stream):
    if fcntl is not None: fcntl.flock(stream.fileno(),fcntl.LOCK_EX)
    else:
        stream.seek(0); stream.write(b"\0"); stream.flush(); msvcrt.locking(stream.fileno(),msvcrt.LK_LOCK,1)

class LaunchState(str, enum.Enum):
    PREFLIGHT="PREFLIGHT"; PREFLIGHT_FAILED="PREFLIGHT_FAILED"; LAUNCH_INTENT="LAUNCH_INTENT"
    FORMAL_RUN_ACTIVATED="FORMAL_RUN_ACTIVATED"; LAUNCH_GATE_FAILED="LAUNCH_GATE_FAILED"
    LAUNCH_UNCERTAIN="LAUNCH_UNCERTAIN"; TERMINAL="TERMINAL"

@dataclass(frozen=True)
class PlatformGate:
    local_fixed_volume:bool; ntfs:bool; wal:bool; synchronous_full:bool; integrity_ok:bool; flush_ok:bool; broker_attested:bool
    def validate(self):
        failed=[n for n,v in self.__dict__.items() if v is not True]
        if failed: raise ValueError("formal launch platform gate failed: "+", ".join(failed))
    def verified(self, attempt_id:str, target_identity_hash:str) -> "PlatformVerdict":
        self.validate()
        evidence=dict(self.__dict__)
        return PlatformVerdict(attempt_id,target_identity_hash,canonical_hash(evidence),_VERDICT_SEAL)

@dataclass(frozen=True)
class AuditVerdict:
    attempt_id:str; target_identity_hash:str; evidence_hash:str; canonical_hash:str; status:str="PASS"
    _seal:object=None
    def __post_init__(self):
        payload={"schema_version":"target_audit.v1","attempt_id":self.attempt_id,"target_identity_hash":self.target_identity_hash,"evidence_hash":self.evidence_hash,"status":self.status}
        if self._seal is not _VERDICT_SEAL or not self.attempt_id or self.status!="PASS" or not all(isinstance(v,str) and _SHA256.fullmatch(v) for v in (self.target_identity_hash,self.evidence_hash,self.canonical_hash)) or self.canonical_hash!=canonical_hash(payload):
            raise ValueError("unverified audit verdict")
    def to_wire(self):
        return {"schema_version":"target_audit.v1","attempt_id":self.attempt_id,"target_identity_hash":self.target_identity_hash,"evidence_hash":self.evidence_hash,"status":self.status,"canonical_hash":self.canonical_hash}

def _verify_audit_evidence(*,attempt_id:str,target_identity_hash:str,evidence_bytes:bytes,expected_evidence_hash:str)->AuditVerdict:
    if not isinstance(evidence_bytes,bytes): raise ValueError("audit evidence bytes required")
    actual=canonical_hash({"bytes_hex":evidence_bytes.hex()})
    if actual!=expected_evidence_hash: raise ValueError("audit evidence hash mismatch")
    payload={"schema_version":"target_audit.v1","attempt_id":attempt_id,"target_identity_hash":target_identity_hash,"evidence_hash":actual,"status":"PASS"}
    return AuditVerdict(attempt_id,target_identity_hash,actual,canonical_hash(payload),"PASS",_VERDICT_SEAL)

@dataclass(frozen=True)
class SaveVerdict:
    attempt_id:str; target_identity_hash:str; lifecycle_attempt_id:str; lifecycle_hash:str; pre_run_hash:str; canonical_hash:str
    _seal:object=None
    def __post_init__(self):
        payload={"schema_version":"save_lifecycle.v1","attempt_id":self.attempt_id,"target_identity_hash":self.target_identity_hash,"lifecycle_attempt_id":self.lifecycle_attempt_id,"lifecycle_hash":self.lifecycle_hash,"pre_run_hash":self.pre_run_hash,"status":"PASS"}
        if self._seal is not _VERDICT_SEAL or not self.attempt_id or self.lifecycle_attempt_id!=self.attempt_id or not all(isinstance(v,str) and _SHA256.fullmatch(v) for v in (self.target_identity_hash,self.lifecycle_hash,self.pre_run_hash,self.canonical_hash)) or self.canonical_hash!=canonical_hash(payload): raise ValueError("unverified save verdict")

def _mint_save_verdict(*,records:tuple[Mapping[str,Any],...],attempt_id:str,target_identity_hash:str,expected_pre_run_hash:str,seal:object)->SaveVerdict:
    """Mint only from the live save executor; persisted JSON is evidence, not authority."""
    if seal is not _SAVE_EXECUTION_SEAL: raise ValueError("save lifecycle execution seal required")
    required=("PREFLIGHT","ORIGINAL_BACKUP","CANONICAL_RESTORE","PRE_RUN_AUDIT")
    stages=tuple(r.get("stage") for r in records)
    cursor=0
    for stage in stages:
        if cursor<len(required) and stage==required[cursor]: cursor+=1
    if cursor!=len(required): raise ValueError("incomplete save lifecycle")
    previous=None
    for record in records:
        body={k:v for k,v in record.items() if k!="step_hash"}
        if body.get("attempt_id")!=attempt_id or body.get("previous_step_hash")!=previous:
            raise ValueError("save lifecycle chain identity mismatch")
        if record.get("step_hash")!=canonical_hash(body): raise ValueError("save lifecycle step hash mismatch")
        previous=record["step_hash"]
    if records[-1].get("content_hash")!=expected_pre_run_hash: raise ValueError("pre-run hash mismatch")
    lifecycle_hash=canonical_hash(list(records))
    payload={"schema_version":"save_lifecycle.v1","attempt_id":attempt_id,"target_identity_hash":target_identity_hash,"lifecycle_attempt_id":attempt_id,"lifecycle_hash":lifecycle_hash,"pre_run_hash":expected_pre_run_hash,"status":"PASS"}
    return SaveVerdict(attempt_id,target_identity_hash,attempt_id,lifecycle_hash,expected_pre_run_hash,canonical_hash(payload),_VERDICT_SEAL)

_SAVE_EXECUTION_SEAL = object()

def _mint_executed_save_verdict(*,records:tuple[Mapping[str,Any],...],attempt_id:str,target_identity_hash:str,expected_pre_run_hash:str,execution_seal:object)->SaveVerdict:
    return _mint_save_verdict(records=records,attempt_id=attempt_id,target_identity_hash=target_identity_hash,expected_pre_run_hash=expected_pre_run_hash,seal=execution_seal)


@dataclass(frozen=True)
class PlatformVerdict:
    attempt_id:str; target_identity_hash:str; canonical_hash:str; _seal:object=None
    def __post_init__(self):
        if self._seal is not _VERDICT_SEAL or not self.attempt_id or not all(isinstance(x,str) and _SHA256.fullmatch(x) for x in (self.target_identity_hash,self.canonical_hash)): raise ValueError("unverified platform verdict")

@dataclass(frozen=True)
class LaunchAuthorization:
    attempt_id:str; target_identity_hash:str; save_attempt_id:str; lifecycle_attempt_id:str; audit_hash:str; save_gate_hash:str; platform_gate_hash:str
    _seal:object=None
    def __post_init__(self):
        if self._seal is not _AUTH_SEAL or self.save_attempt_id!=self.attempt_id or self.lifecycle_attempt_id!=self.attempt_id or not all(isinstance(v,str) and _SHA256.fullmatch(v) for v in (self.target_identity_hash,self.audit_hash,self.save_gate_hash,self.platform_gate_hash)):
            raise ValueError("launch authorization requires audited durable gates")
    @classmethod
    def issue(cls, audit:AuditVerdict, platform:PlatformVerdict, save:SaveVerdict):
        if not isinstance(audit,AuditVerdict) or not isinstance(platform,PlatformVerdict) or not isinstance(save,SaveVerdict): raise ValueError("verified typed verdicts required")
        if len({audit.attempt_id,platform.attempt_id,save.attempt_id,save.lifecycle_attempt_id})!=1 or len({audit.target_identity_hash,platform.target_identity_hash,save.target_identity_hash})!=1: raise ValueError("authorization identity binding mismatch")
        return cls(audit.attempt_id,audit.target_identity_hash,save.attempt_id,save.lifecycle_attempt_id,audit.canonical_hash,save.canonical_hash,platform.canonical_hash,_AUTH_SEAL)

@dataclass(frozen=True)
class LaunchLifecycle:
    attempt_id:str; state:LaunchState=LaunchState.PREFLIGHT; reserved_run_id:str|None=None
    gameplay_attempt_id:str|None=None; launch_nonce:str|None=None; process_ref:str|None=None
    activation_source:str|None=None; failure_reason:str|None=None; authorization_hash:str|None=None
    schema_version:str=SCHEMA_VERSION
    _KEYS=frozenset({"schema_version","attempt_id","state","reserved_run_id","gameplay_attempt_id","launch_nonce","process_ref","activation_source","failure_reason","authorization_hash"})
    def __post_init__(self):
        if self.schema_version!=SCHEMA_VERSION or not isinstance(self.attempt_id,str) or not self.attempt_id: raise ValueError("invalid identity/version")
        present=lambda x:isinstance(x,str) and bool(x)
        reserved=all(present(x) for x in (self.reserved_run_id,self.gameplay_attempt_id,self.launch_nonce,self.authorization_hash))
        none=all(x is None for x in (self.reserved_run_id,self.gameplay_attempt_id,self.launch_nonce,self.authorization_hash))
        if not (reserved or none): raise ValueError("reserved fields must be all present or absent")
        rules={
          LaunchState.PREFLIGHT:(False,False,False), LaunchState.PREFLIGHT_FAILED:(False,False,True),
          LaunchState.LAUNCH_INTENT:(True,False,False), LaunchState.FORMAL_RUN_ACTIVATED:(True,True,False),
          LaunchState.LAUNCH_GATE_FAILED:(True,False,True), LaunchState.LAUNCH_UNCERTAIN:(True,False,True),
          LaunchState.TERMINAL:(True,True,True)}
        want_reserved,want_process,want_failure=rules[self.state]
        if reserved!=want_reserved or present(self.process_ref)!=want_process or present(self.failure_reason)!=want_failure: raise ValueError("fields invalid for launch state")
        if (self.activation_source is not None) != want_process or (want_process and self.activation_source not in {"observer","reconciliation"}): raise ValueError("invalid activation source")
    @classmethod
    def begin(cls,attempt_id): return cls(attempt_id)
    def preflight_failure(self,reason):
        if self.state is not LaunchState.PREFLIGHT or not reason: raise ValueError("invalid preflight failure")
        return replace(self,state=LaunchState.PREFLIGHT_FAILED,failure_reason=reason)
    def reserve(self,run_id,gameplay_attempt_id,nonce,*,authorization:LaunchAuthorization,intent_log:Path):
        if self.state is not LaunchState.PREFLIGHT or not isinstance(authorization,LaunchAuthorization): raise ValueError("validated authorization required")
        if not isinstance(intent_log,Path): raise ValueError("durable launch intent log required")
        if authorization.attempt_id!=self.attempt_id: raise ValueError("authorization attempt mismatch")
        auth_wire={k:getattr(authorization,k) for k in ("attempt_id","target_identity_hash","save_attempt_id","lifecycle_attempt_id","audit_hash","save_gate_hash","platform_gate_hash")}
        reserved=replace(self,state=LaunchState.LAUNCH_INTENT,reserved_run_id=run_id,gameplay_attempt_id=gameplay_attempt_id,launch_nonce=nonce,authorization_hash=canonical_hash(auth_wire))
        intent_log.parent.mkdir(parents=True,exist_ok=True)
        encoded=canonical_json_bytes(reserved.to_wire())+b"\n"
        temp=intent_log.with_name(intent_log.name+".tmp")
        lock_path=intent_log.with_name(intent_log.name+".lock")
        try:
            with lock_path.open("a+b") as lock:
                _lock(lock)
                previous=intent_log.read_bytes() if intent_log.exists() else b""
                for line in previous.splitlines():
                    prior=LaunchLifecycle.from_wire(json.loads(line))
                    if prior.attempt_id==self.attempt_id or prior.reserved_run_id==run_id:
                        raise ValueError("attempt/target identity already reserved")
                with temp.open("wb") as stream:
                    stream.write(previous+encoded); stream.flush(); os.fsync(stream.fileno())
                os.replace(temp,intent_log)
                dfd=os.open(intent_log.parent,os.O_RDONLY)
                try: os.fsync(dfd)
                finally: os.close(dfd)
                lines=intent_log.read_bytes().splitlines()
                if not lines or json.loads(lines[-1])!=reserved.to_wire(): raise ValueError("durable LAUNCH_INTENT revalidation failed")
        except OSError as exc:
            raise ValueError("durable LAUNCH_INTENT commit failed") from exc
        finally:
            if temp.exists(): temp.unlink()
        return reserved
    def activate(self,process_ref,*,source="observer"):
        if self.state is not LaunchState.LAUNCH_INTENT or source not in {"observer","reconciliation"} or not process_ref: raise ValueError("activation requires launch intent and process")
        return replace(self,state=LaunchState.FORMAL_RUN_ACTIVATED,process_ref=process_ref,activation_source=source)
    def create_process_failed(self):
        if self.state is not LaunchState.LAUNCH_INTENT: raise ValueError("no launch intent")
        return replace(self,state=LaunchState.LAUNCH_GATE_FAILED,failure_reason="CREATE_PROCESS_FAILED")
    def launch_uncertain(self):
        if self.state is not LaunchState.LAUNCH_INTENT: raise ValueError("no launch intent")
        return replace(self,state=LaunchState.LAUNCH_UNCERTAIN,failure_reason="PROCESS_IDENTITY_UNCERTAIN")
    def terminal(self,reason):
        if self.state is not LaunchState.FORMAL_RUN_ACTIVATED or not reason: raise ValueError("terminal requires activation")
        return replace(self,state=LaunchState.TERMINAL,failure_reason=reason)
    @property
    def counts_toward_outcome_denominator(self): return self.state in {LaunchState.FORMAL_RUN_ACTIVATED,LaunchState.TERMINAL}
    @property
    def campaign_blocked(self): return self.state is LaunchState.LAUNCH_UNCERTAIN
    @property
    def replacement_allowed(self): return False
    def to_wire(self):
        d=dict(self.__dict__); d["state"]=self.state.value; return d
    @property
    def record_hash(self): return canonical_hash(self.to_wire())
    @classmethod
    def from_wire(cls,data:Mapping[str,Any]):
        if not isinstance(data,Mapping) or set(data)!=cls._KEYS: raise ValueError("unknown/missing fields")
        try: return cls(**{**data,"state":LaunchState(data["state"])})
        except (TypeError,ValueError) as e: raise ValueError("invalid launch record") from e
