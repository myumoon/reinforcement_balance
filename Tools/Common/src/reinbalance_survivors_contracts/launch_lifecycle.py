"""Fail-closed formal-run identity and durable launch transitions."""
from __future__ import annotations
import enum, re
from dataclasses import dataclass, replace
from typing import Any, Mapping
from .canonical_json import canonical_hash

SCHEMA_VERSION = "launch_lifecycle.v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_AUTH_SEAL = object()

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

@dataclass(frozen=True)
class LaunchAuthorization:
    audit_hash:str; save_gate_hash:str; platform_gate_hash:str; durable_commit:bool
    _seal:object=None
    def __post_init__(self):
        if self._seal is not _AUTH_SEAL or not all(isinstance(v,str) and _SHA256.fullmatch(v) for v in (self.audit_hash,self.save_gate_hash,self.platform_gate_hash)) or self.durable_commit is not True:
            raise ValueError("launch authorization requires audited durable gates")
    @classmethod
    def issue(cls, audit_report:Mapping[str,Any], platform:PlatformGate, save_gate:Mapping[str,Any], *, durable_commit:bool):
        if not isinstance(audit_report,Mapping) or set(audit_report)!={"schema_version","target_hash","status"} or audit_report.get("schema_version")!="target_audit.v1" or audit_report.get("status")!="PASS": raise ValueError("passing target audit required")
        platform.validate()
        if not isinstance(save_gate,Mapping) or set(save_gate)!={"schema_version","status","pre_run_hash"} or save_gate.get("schema_version")!="save_gate.v1" or save_gate.get("status")!="PASS" or not isinstance(save_gate.get("pre_run_hash"),str) or not _SHA256.fullmatch(save_gate["pre_run_hash"]): raise ValueError("passing save gate required")
        return cls(canonical_hash(audit_report),canonical_hash(save_gate),canonical_hash(platform.__dict__),durable_commit,_AUTH_SEAL)

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
    def reserve(self,run_id,gameplay_attempt_id,nonce,*,authorization:LaunchAuthorization):
        if self.state is not LaunchState.PREFLIGHT or not isinstance(authorization,LaunchAuthorization): raise ValueError("validated authorization required")
        auth_wire={"audit_hash":authorization.audit_hash,"save_gate_hash":authorization.save_gate_hash,"platform_gate_hash":authorization.platform_gate_hash,"durable_commit":authorization.durable_commit}
        return replace(self,state=LaunchState.LAUNCH_INTENT,reserved_run_id=run_id,gameplay_attempt_id=gameplay_attempt_id,launch_nonce=nonce,authorization_hash=canonical_hash(auth_wire))
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
