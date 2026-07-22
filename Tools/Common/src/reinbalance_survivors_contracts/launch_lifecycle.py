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
_CAMPAIGN_ID = re.compile(r"[A-Za-z0-9_-]+")
_AUTH_SEAL = object()
_VERDICT_SEAL = object()
LAUNCH_STORE_CONFIG_PATH = Path(r"C:\ProgramData\ReinBalance\launch_store.json")
LAUNCH_GATE_CONFIG_PATH = Path(r"C:\ProgramData\ReinBalance\launch_gate.json")

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
    lifecycle_record_path:str=""; canonical_save_path:str=""; original_backup_path:str=""
    _seal:object=None
    def __post_init__(self):
        payload={"schema_version":"save_lifecycle.v1","attempt_id":self.attempt_id,"target_identity_hash":self.target_identity_hash,"lifecycle_attempt_id":self.lifecycle_attempt_id,"lifecycle_hash":self.lifecycle_hash,"pre_run_hash":self.pre_run_hash,"status":"PASS"}
        if self._seal is not _VERDICT_SEAL or not self.attempt_id or self.lifecycle_attempt_id!=self.attempt_id or not all(isinstance(v,str) and _SHA256.fullmatch(v) for v in (self.target_identity_hash,self.lifecycle_hash,self.pre_run_hash,self.canonical_hash)) or self.canonical_hash!=canonical_hash(payload): raise ValueError("unverified save verdict")

def _read_and_verify_save_evidence(save:"SaveVerdict", *, require_fixed_paths:bool=False)->None:
    paths=tuple(Path(value) for value in (save.lifecycle_record_path,save.canonical_save_path,save.original_backup_path))
    if any(not value or not path.is_absolute() or not path.is_file() for value,path in zip((save.lifecycle_record_path,save.canonical_save_path,save.original_backup_path),paths)):
        raise ValueError("complete on-disk save evidence required")
    if require_fixed_paths:
        try: config=json.loads(LAUNCH_GATE_CONFIG_PATH.read_text(encoding="utf-8"))
        except (OSError,UnicodeDecodeError,json.JSONDecodeError) as exc: raise ValueError("fixed launch gate config unavailable") from exc
        if not isinstance(config,dict) or set(config)!={"schema_version","lifecycle_root","canonical_save_path","original_backup_path"} or config["schema_version"]!="launch_gate.v1": raise ValueError("invalid fixed launch gate config")
        expected=(Path(config["lifecycle_root"]).resolve()/(save.attempt_id+".jsonl"),Path(config["canonical_save_path"]).resolve(),Path(config["original_backup_path"]).resolve())
        if paths!=expected: raise ValueError("save evidence is not at fixed gate paths")
    try: records=tuple(json.loads(line) for line in paths[0].read_bytes().splitlines())
    except (OSError,UnicodeDecodeError,json.JSONDecodeError) as exc: raise ValueError("invalid lifecycle evidence") from exc
    previous=None
    for record in records:
        body={k:v for k,v in record.items() if k!="step_hash"}
        if body.get("attempt_id")!=save.attempt_id or body.get("previous_step_hash")!=previous or record.get("step_hash")!=canonical_hash(body):
            raise ValueError("on-disk lifecycle chain mismatch")
        previous=record["step_hash"]
    if canonical_hash(list(records))!=save.lifecycle_hash: raise ValueError("on-disk lifecycle hash mismatch")
    if not records or records[-1].get("content_hash")!=save.pre_run_hash: raise ValueError("on-disk pre-run audit mismatch")
    if canonical_hash({"bytes_hex":paths[1].read_bytes().hex()})!=save.pre_run_hash: raise ValueError("canonical save evidence mismatch")
    backup_hash=next((r.get("content_hash") for r in records if r.get("stage")=="ORIGINAL_BACKUP"),None)
    if backup_hash is None or canonical_hash({"bytes_hex":paths[2].read_bytes().hex()})!=backup_hash: raise ValueError("original backup evidence mismatch")

def _finalize_save_execution(*,records:tuple[Mapping[str,Any],...],attempt_id:str,target_identity_hash:str,expected_pre_run_hash:str,lifecycle_record_path:Path,canonical_save_path:Path,original_backup_path:Path)->SaveVerdict:
    """Mint only from the live save executor; persisted JSON is evidence, not authority."""
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
    verdict=SaveVerdict(attempt_id,target_identity_hash,attempt_id,lifecycle_hash,expected_pre_run_hash,canonical_hash(payload),str(lifecycle_record_path.resolve()),str(canonical_save_path.resolve()),str(original_backup_path.resolve()),_VERDICT_SEAL)
    _read_and_verify_save_evidence(verdict)
    return verdict


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
        _read_and_verify_save_evidence(save,require_fixed_paths=True)
        return cls(audit.attempt_id,audit.target_identity_hash,save.attempt_id,save.lifecycle_attempt_id,audit.canonical_hash,save.canonical_hash,platform.canonical_hash,_AUTH_SEAL)

def _sync_launch_intent(path:Path, *, platform:str|None=None)->None:
    """Flush a replaced intent; Win64 directory-entry durability relies on NTFS journaling."""
    platform=os.name if platform is None else platform
    if platform=="nt":
        with path.open("rb") as stream: os.fsync(stream.fileno())
        return
    if platform=="posix":
        descriptor=os.open(path.parent,os.O_RDONLY)
        try: os.fsync(descriptor)
        finally: os.close(descriptor)
        return
    raise OSError(f"unsupported durability platform: {platform}")

def _validate_store_volume(path:Path)->None:
    text=str(path)
    if text.startswith(("//", "\\\\")): raise ValueError("UNC launch store is unsupported")
    if os.name!="nt": return  # Host-independent contract tests use the fixed config attestations.
    import ctypes
    root=Path(path.anchor)
    drive_type=ctypes.windll.kernel32.GetDriveTypeW(str(root))
    if drive_type!=3: raise ValueError("launch store must be on a local fixed volume")
    fs_name=ctypes.create_unicode_buffer(32)
    if not ctypes.windll.kernel32.GetVolumeInformationW(str(root),None,0,None,None,None,fs_name,len(fs_name)) or fs_name.value.upper()!="NTFS":
        raise ValueError("launch store must be on NTFS")

@dataclass(frozen=True)
class LaunchIntentStore:
    """Canonical ledger resolved only from the deployment's fixed local-NTFS config."""
    campaign_id:str
    root:Path
    support_envelope:Mapping[str,Any]
    def __post_init__(self):
        if not self.campaign_id or not self.root.is_absolute() or set(self.support_envelope)!={"local_fixed_volume","ntfs"}: raise ValueError("invalid canonical store config")
    @classmethod
    def for_campaign(cls,campaign_id:str)->"LaunchIntentStore":
        if not isinstance(campaign_id,str) or _CAMPAIGN_ID.fullmatch(campaign_id) is None:
            raise ValueError("campaign identity must use only ASCII letters, digits, '_' or '-'")
        try: config=json.loads(LAUNCH_STORE_CONFIG_PATH.read_text(encoding="utf-8"))
        except (OSError,UnicodeDecodeError,json.JSONDecodeError) as exc: raise ValueError("fixed launch store config unavailable") from exc
        if not isinstance(config,dict) or set(config)!={"schema_version","root","local_fixed_volume","ntfs"} or config["schema_version"]!="launch_store.v1": raise ValueError("invalid fixed launch store config")
        root=Path(config["root"])
        if not root.is_absolute(): raise ValueError("canonical store root must be absolute")
        canonical_root=Path(os.path.realpath(root))
        campaign_root=Path(os.path.realpath(root/campaign_id))
        try: contained=os.path.commonpath((canonical_root,campaign_root))==str(canonical_root)
        except ValueError: contained=False  # Different Win32 drives, for example.
        if not contained or campaign_root==canonical_root:
            raise ValueError("campaign identity escapes canonical store root")
        return cls(campaign_id,campaign_root,{"local_fixed_volume":config["local_fixed_volume"],"ntfs":config["ntfs"]})
    @property
    def intent_log(self)->Path: return self.root/"launch_intents.jsonl"
    def records(self)->tuple["LaunchLifecycle",...]:
        if not self.intent_log.exists(): return ()
        try: records=tuple(LaunchLifecycle.from_wire(json.loads(line)) for line in self.intent_log.read_bytes().splitlines())
        except (OSError,UnicodeDecodeError,json.JSONDecodeError,ValueError) as exc: raise ValueError("invalid durable launch ledger") from exc
        latest={}
        allowed={
            LaunchState.LAUNCH_INTENT:{LaunchState.FORMAL_RUN_ACTIVATED,LaunchState.LAUNCH_GATE_FAILED,LaunchState.LAUNCH_UNCERTAIN},
            LaunchState.FORMAL_RUN_ACTIVATED:{LaunchState.TERMINAL},
        }
        for record in records:
            prior=latest.get(record.attempt_id)
            if prior is None and record.state is not LaunchState.LAUNCH_INTENT: raise ValueError("durable lifecycle must begin with launch intent")
            if prior is not None and record.state not in allowed.get(prior.state,set()): raise ValueError("invalid durable launch transition")
            latest[record.attempt_id]=record
        return records
    def append_transition(self,previous:"LaunchLifecycle",current:"LaunchLifecycle")->None:
        configured=LaunchIntentStore.for_campaign(self.campaign_id)
        if self.root!=configured.root or dict(self.support_envelope)!=dict(configured.support_envelope): raise ValueError("store does not match fixed config")
        if any(value is not True for value in self.support_envelope.values()): raise ValueError("launch store is outside local fixed NTFS support envelope")
        _validate_store_volume(self.root)
        intent_log=self.intent_log; temp=intent_log.with_name(intent_log.name+".tmp"); lock_path=intent_log.with_name(intent_log.name+".lock")
        with lock_path.open("a+b") as lock:
            _lock(lock)
            before=intent_log.read_bytes() if intent_log.exists() else b""
            records=tuple(LaunchLifecycle.from_wire(json.loads(line)) for line in before.splitlines())
            prior=next((record for record in reversed(records) if record.attempt_id==previous.attempt_id),None)
            if prior!=previous: raise ValueError("durable ledger predecessor mismatch")
            try:
                with temp.open("wb") as stream:
                    stream.write(before+canonical_json_bytes(current.to_wire())+b"\n"); stream.flush(); os.fsync(stream.fileno())
                os.replace(temp,intent_log); _sync_launch_intent(intent_log)
                if self.records()[-1]!=current: raise ValueError("durable transition revalidation failed")
            finally:
                if temp.exists(): temp.unlink()
    def outcome_summary(self)->Mapping[str,Any]:
        records=self.records()
        activated={record.reserved_run_id for record in records if record.state is LaunchState.FORMAL_RUN_ACTIVATED}
        terminal={record.reserved_run_id:record.failure_reason for record in records if record.state is LaunchState.TERMINAL}
        return {"activated_runs":len(activated),"terminal_outcomes":terminal}

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
    def reserve(self,run_id,gameplay_attempt_id,nonce,*,authorization:LaunchAuthorization,store:LaunchIntentStore):
        if self.state is not LaunchState.PREFLIGHT or not isinstance(authorization,LaunchAuthorization): raise ValueError("validated authorization required")
        if not isinstance(store,LaunchIntentStore): raise ValueError("canonical launch intent store required")
        configured=LaunchIntentStore.for_campaign(store.campaign_id)
        if store.root!=configured.root or dict(store.support_envelope)!=dict(configured.support_envelope): raise ValueError("store does not match fixed config")
        if any(value is not True for value in store.support_envelope.values()): raise ValueError("launch store is outside local fixed NTFS support envelope")
        _validate_store_volume(store.root)
        intent_log=store.intent_log
        if authorization.attempt_id!=self.attempt_id: raise ValueError("authorization attempt mismatch")
        auth_wire={k:getattr(authorization,k) for k in ("attempt_id","target_identity_hash","save_attempt_id","lifecycle_attempt_id","audit_hash","save_gate_hash","platform_gate_hash")}
        reserved=replace(self,state=LaunchState.LAUNCH_INTENT,reserved_run_id=run_id,gameplay_attempt_id=gameplay_attempt_id,launch_nonce=nonce,authorization_hash=canonical_hash(auth_wire))
        intent_log.parent.mkdir(parents=True,exist_ok=True)
        encoded=canonical_json_bytes(reserved.to_wire())+b"\n"
        temp=intent_log.with_name(intent_log.name+".tmp")
        lock_path=intent_log.with_name(intent_log.name+".lock")
        marker_dir=store.root/"reservations"; marker_dir.mkdir(parents=True,exist_ok=True)
        marker_paths=(marker_dir/(canonical_hash({"kind":"attempt","identity":self.attempt_id})+".lock"),marker_dir/(canonical_hash({"kind":"run","identity":run_id})+".lock"))
        created=[]
        previous=b""
        ledger_replaced=False
        with lock_path.open("a+b") as lock:
            _lock(lock)
            try:
                previous=intent_log.read_bytes() if intent_log.exists() else b""
                for line in previous.splitlines():
                    prior=LaunchLifecycle.from_wire(json.loads(line))
                    if prior.attempt_id==self.attempt_id or prior.reserved_run_id==run_id:
                        raise ValueError("attempt/target identity already reserved")
                # The fsync'd ledger is authoritative.  A marker with no matching
                # committed entry is residue from a process that stopped before
                # its LAUNCH_INTENT commit and may be reclaimed while serialized
                # by the ledger lock.
                for marker in marker_paths:
                    try:
                        fd=os.open(marker,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
                    except FileExistsError:
                        marker.unlink()
                        fd=os.open(marker,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
                    os.close(fd); created.append(marker)
                with temp.open("wb") as stream:
                    stream.write(previous+encoded); stream.flush(); os.fsync(stream.fileno())
                os.replace(temp,intent_log)
                ledger_replaced=True
                _sync_launch_intent(intent_log)
                lines=intent_log.read_bytes().splitlines()
                if not lines or json.loads(lines[-1])!=reserved.to_wire(): raise ValueError("durable LAUNCH_INTENT revalidation failed")
            except Exception as exc:
                rollback_ok=True
                try:
                    if ledger_replaced:
                        if previous:
                            with temp.open("wb") as stream:
                                stream.write(previous); stream.flush(); os.fsync(stream.fileno())
                            os.replace(temp,intent_log)
                            _sync_launch_intent(intent_log)
                        else:
                            intent_log.unlink(missing_ok=True)
                            if os.name=="posix":
                                descriptor=os.open(intent_log.parent,os.O_RDONLY)
                                try: os.fsync(descriptor)
                                finally: os.close(descriptor)
                except Exception:
                    rollback_ok=False
                if rollback_ok:
                    for marker in created:
                        try: marker.unlink()
                        except OSError: rollback_ok=False
                message="durable LAUNCH_INTENT commit failed"
                if not rollback_ok: message += "; rollback failed and reservation remains blocked"
                raise ValueError(message) from exc
            finally:
                if temp.exists(): temp.unlink()
        return reserved
    def activate(self,process_ref,*,source="observer",store:LaunchIntentStore|None=None):
        if self.state is not LaunchState.LAUNCH_INTENT or source not in {"observer","reconciliation"} or not process_ref: raise ValueError("activation requires launch intent and process")
        if not isinstance(store,LaunchIntentStore): raise ValueError("canonical store required for durable activation")
        activated=replace(self,state=LaunchState.FORMAL_RUN_ACTIVATED,process_ref=process_ref,activation_source=source)
        store.append_transition(self,activated); return activated
    def create_process_failed(self,*,store:LaunchIntentStore|None=None):
        if self.state is not LaunchState.LAUNCH_INTENT: raise ValueError("no launch intent")
        if not isinstance(store,LaunchIntentStore): raise ValueError("canonical store required for durable failure")
        failed=replace(self,state=LaunchState.LAUNCH_GATE_FAILED,failure_reason="CREATE_PROCESS_FAILED")
        store.append_transition(self,failed); return failed
    def launch_uncertain(self,*,store:LaunchIntentStore|None=None):
        if self.state is not LaunchState.LAUNCH_INTENT: raise ValueError("no launch intent")
        if not isinstance(store,LaunchIntentStore): raise ValueError("canonical store required for durable failure")
        failed=replace(self,state=LaunchState.LAUNCH_UNCERTAIN,failure_reason="PROCESS_IDENTITY_UNCERTAIN")
        store.append_transition(self,failed); return failed
    def terminal(self,reason,*,store:LaunchIntentStore|None=None):
        if self.state is not LaunchState.FORMAL_RUN_ACTIVATED or not reason: raise ValueError("terminal requires activation")
        if not isinstance(store,LaunchIntentStore): raise ValueError("canonical store required for durable terminal")
        terminal=replace(self,state=LaunchState.TERMINAL,failure_reason=reason)
        store.append_transition(self,terminal); return terminal
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
