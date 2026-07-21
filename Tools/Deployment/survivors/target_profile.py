from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import copy, re, yaml
from reinbalance_survivors_contracts.canonical_json import canonical_hash

VERSION="survivors_target.v1"
_SHA256=re.compile(r"[0-9a-f]{64}")
CONFIG=Path(__file__).parents[1]/"configs"/"mad_forest_standard_v1.yaml"
SECTIONS={
 "base":frozenset({"platform","window_mode","client_resolution","ui_language","stage","character","modifiers","success_timer_seconds","post_timer_event_required"}),
 "build":frozenset({"build_id","executable_version","executable_hash","distribution","app_id","content_revision","ui_revision","manual_attestation"}),
 "progression":frozenset({"save_artifact_hash","save_format_version","unlocked_characters","unlocked_items","unlocked_stages","collection_pool","purchased_power_ups","reroll_count","skip_count","banish_count"}),
 "display":frozenset({"ui_scale","windows_dpi_scale","monitor_output"}),
 "hardware":frozenset({"profile_id","os_build","gpu_name","vram_mb","driver_version","cuda_version","pytorch_version","capture_backend","cpu_live_fallback"}),
 "input":frozenset({"key_bindings","action_semantics_version","frame_skip","physics_hz","decision_hz","key_hold_ms","lease_ms"}),
 "choice_taxonomy":frozenset({"level_up_card_counts","fallbacks","capabilities","states","candidate_vocabulary","candidate_vocabulary_hash"}),
}
CLOSED_TAXONOMY={
 "unlocked_characters":frozenset({"antonio"}), "unlocked_items":frozenset({"whip"}),
 "unlocked_stages":frozenset({"mad_forest"}), "collection_pool":frozenset({"whip"}),
 "purchased_power_ups":frozenset(), "candidate_vocabulary":frozenset({"whip","gold","chicken"}),
}

@dataclass(frozen=True)
class SuccessObservation:
    run_id:str; event:str; observed_at_seconds:float; evidence_hash:str
    @classmethod
    def from_telemetry(cls,run_id:str,path:Path):
        if not isinstance(run_id,str) or not run_id or not path.is_file(): raise ValueError("run-bound telemetry required")
        data=yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data,Mapping) or set(data)!={"run_id","event","observed_at_seconds","evidence_hash"} or data["run_id"]!=run_id: raise ValueError("telemetry run identity mismatch")
        if data["event"] not in {"result_screen","death_transition"} or type(data["observed_at_seconds"]) not in (int,float): raise ValueError("invalid post-30 telemetry")
        evidence={k:data[k] for k in ("run_id","event","observed_at_seconds")}
        actual=canonical_hash(evidence)
        if data["evidence_hash"]!=actual: raise ValueError("post-30 event evidence hash mismatch")
        return cls(run_id,data["event"],float(data["observed_at_seconds"]),actual)

@dataclass(frozen=True)
class TargetProfile:
    sections: Mapping[str,Mapping[str,Any]]; schema_version:str=VERSION
    def __post_init__(self):
        if self.schema_version!=VERSION or set(self.sections)!=set(SECTIONS): raise ValueError("unknown/missing target profile section")
        for name,keys in SECTIONS.items():
            if set(self.sections[name])!=keys: raise ValueError(f"unknown/missing {name} fields")
        base=self.sections["base"]
        if base["success_timer_seconds"]!=1800 or base["post_timer_event_required"] is not True: raise ValueError("invalid success semantics")
        if base["modifiers"]!={"hyper":False,"hurry":False,"inverse":False,"arcana":False,"limit_break":False,"golden_eggs":False,"dlc":False}: raise ValueError("invalid modifiers")
        tax=self.sections["choice_taxonomy"]; vocab=tax["candidate_vocabulary"]
        if not isinstance(vocab,list) or not vocab or len(vocab)!=len(set(vocab)) or not all(isinstance(x,str) and x for x in vocab): raise ValueError("invalid candidate vocabulary")
        if tax["level_up_card_counts"] != [3,4] or tax["fallbacks"] != ["gold","chicken"] or tax["capabilities"] != ["reroll","skip","banish"] or tax["states"] != ["gameplay","level_up","fallback","chest","result","post_30_transition"]: raise ValueError("invalid closed choice taxonomy")
        p=self.sections["progression"]
        for k,allowed in CLOSED_TAXONOMY.items():
            values=vocab if k=="candidate_vocabulary" else p[k]
            if not isinstance(values,list) or len(values)!=len(set(values)) or frozenset(values)!=allowed or not all(type(x) is str for x in values): raise ValueError(f"invalid closed taxonomy {k}")
        if not all(type(p[k]) is int and p[k]>=0 for k in ("reroll_count","skip_count","banish_count")): raise ValueError("invalid capability count")
        if not all(type(x) is int and x>0 for x in tax["level_up_card_counts"]): raise ValueError("invalid card count")
        if self.sections["choice_taxonomy"]["candidate_vocabulary_hash"]!=canonical_hash(vocab): raise ValueError("candidate vocabulary hash mismatch")
        inp=self.sections["input"]
        if inp["key_bindings"]!={"up":"W","down":"S","left":"A","right":"D"} or (inp["physics_hz"],inp["frame_skip"],inp["decision_hz"])!=(60,4,15): raise ValueError("invalid input contract")
    def to_wire(self): return {"schema_version":self.schema_version,**copy.deepcopy(dict(self.sections))}
    @property
    def target_hash(self): return canonical_hash(self.to_wire())
    def success_state(self,timer_seconds:int,post_timer_event:SuccessObservation|None,run_id:str|None=None)->str:
        if timer_seconds<1800:return "RUNNING"
        confirmed=isinstance(post_timer_event,SuccessObservation) and bool(run_id) and post_timer_event.run_id==run_id and post_timer_event.observed_at_seconds>=1800 and bool(_SHA256.fullmatch(post_timer_event.evidence_hash))
        return "TARGET_REACHED_CONFIRMED" if confirmed else "TARGET_REACHED_PENDING_TRANSITION"
    @classmethod
    def from_wire(cls,data:Mapping[str,Any]):
        if set(data)!={"schema_version",*SECTIONS}: raise ValueError("unknown/missing target fields")
        return cls({k:copy.deepcopy(data[k]) for k in SECTIONS},data["schema_version"])

def load_target_profile(path:Path=CONFIG):
    with path.open(encoding="utf-8") as f:return TargetProfile.from_wire(yaml.safe_load(f))
