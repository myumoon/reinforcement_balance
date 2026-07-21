from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import json, yaml
from reinbalance_survivors_contracts.canonical_json import canonical_hash

VERSION = "survivors_action.v1"
CONFIG = Path(__file__).parents[1] / "configs" / "action_semantics_v1.yaml"

@dataclass(frozen=True)
class ActionRow:
    index: int; name: str; sim_vector: tuple[int, int]; screen_direction: str; wasd_chord: tuple[str, ...]

@dataclass(frozen=True)
class ActionContract:
    rows: tuple[ActionRow, ...]; physics_hz: int; frame_skip: int; decision_hz: int
    schema_version: str = VERSION
    _KEYS = frozenset({"schema_version", "physics_hz", "frame_skip", "decision_hz", "rows", "golden_fixture", "source_descriptor"})
    def __post_init__(self):
        canonical = [(0,"N",(0,1),"up",("W",)),(1,"NE",(1,1),"up_right",("W","D")),(2,"E",(1,0),"right",("D",)),(3,"SE",(1,-1),"down_right",("S","D")),(4,"S",(0,-1),"down",("S",)),(5,"SW",(-1,-1),"down_left",("S","A")),(6,"W",(-1,0),"left",("A",)),(7,"NW",(-1,1),"up_left",("W","A")),(8,"idle",(0,0),"idle",())]
        actual=[(r.index,r.name,r.sim_vector,r.screen_direction,r.wasd_chord) for r in self.rows]
        if self.schema_version != VERSION or actual != canonical: raise ValueError("action mapping/golden east-west parity mismatch")
        if (self.physics_hz,self.frame_skip,self.decision_hz)!=(60,4,15): raise ValueError("unsupported action cadence")
        if any(type(r.index) is not int or not all(type(v) is int for v in r.sim_vector) for r in self.rows): raise ValueError("invalid action row type")
    def to_wire(self):
        return {"schema_version":self.schema_version,"physics_hz":self.physics_hz,"frame_skip":self.frame_skip,"decision_hz":self.decision_hz,"golden_fixture":"simulator_displacement_and_key_dry_run.v1","source_descriptor":{"path":"ReinBalance/Source/ReinBalanceLogic/Private/Survivors/SurvivorsGameLogic.cpp","function":"FSurvivorsGameLogic::ApplyAction","physics_dt":"1/60","telemetry":"golden/action_displacement_wasd_v1.json"},"rows":[{"index":r.index,"name":r.name,"sim_vector":list(r.sim_vector),"screen_direction":r.screen_direction,"wasd_chord":list(r.wasd_chord)} for r in self.rows]}
    @property
    def contract_hash(self): return canonical_hash(self.to_wire())
    @classmethod
    def from_wire(cls,data:Mapping[str,Any]):
        source=data.get("source_descriptor") if isinstance(data,Mapping) else None
        expected_source={"path":"ReinBalance/Source/ReinBalanceLogic/Private/Survivors/SurvivorsGameLogic.cpp","function":"FSurvivorsGameLogic::ApplyAction","physics_dt":"1/60","telemetry":"golden/action_displacement_wasd_v1.json"}
        if not isinstance(data,Mapping) or set(data)!=cls._KEYS or data.get("golden_fixture")!="simulator_displacement_and_key_dry_run.v1" or source!=expected_source: raise ValueError("unknown/missing action fields or fixture")
        if not isinstance(data["rows"],list): raise ValueError("rows must be list")
        rows=[]
        for d in data["rows"]:
            if set(d)!={"index","name","sim_vector","screen_direction","wasd_chord"}: raise ValueError("unknown action row field")
            rows.append(ActionRow(d["index"],d["name"],tuple(d["sim_vector"]),d["screen_direction"],tuple(d["wasd_chord"])))
        return cls(tuple(rows),data["physics_hz"],data["frame_skip"],data["decision_hz"],data["schema_version"])

def load_action_contract(path: Path=CONFIG):
    with path.open(encoding="utf-8") as f: contract=ActionContract.from_wire(yaml.safe_load(f))
    validate_golden_fixture(contract, path.parent/"golden"/"action_displacement_wasd_v1.json")
    return contract

def validate_golden_fixture(contract:ActionContract,path:Path)->None:
    with path.open(encoding="utf-8") as f:data=json.load(f)
    if set(data)!={"schema_version","measurement","samples"} or data["schema_version"]!="survivors_action_telemetry.v1" or not isinstance(data["measurement"],str): raise ValueError("invalid golden telemetry fixture")
    expected=[]
    for row in contract.rows:
        x,y=row.sim_vector
        expected.append({"index":row.index,"sim_delta_sign":[x,y],"screen_delta_sign":[x,-y],"keys":list(row.wasd_chord)})
    if data["samples"]!=expected: raise ValueError("golden telemetry displacement/WASD mismatch")
