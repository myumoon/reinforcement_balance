from pathlib import Path
from typing import Mapping,Any
import yaml
from reinbalance_survivors_contracts.canonical_json import canonical_hash

CONFIG=Path(__file__).parents[1]/"configs"/"artifact_compatibility_v1.yaml"
TOP=frozenset({"schema_version","required_parent","classes","artifacts"})

def load_matrix(path:Path=CONFIG):
    with path.open(encoding="utf-8") as f:data=yaml.safe_load(f)
    if set(data)!=TOP or data["schema_version"]!="artifact_compatibility.v1" or data["required_parent"]!="target_audit_hash":raise ValueError("invalid compatibility matrix")
    return data

def invalidation_for(changed_fields:set[str],matrix:Mapping[str,Any]|None=None):
    matrix=matrix or load_matrix(); scopes=set(); full=False; classified=set()
    for rule in matrix["classes"].values():
        overlap=changed_fields&set(rule["changed_fields"])
        if overlap: classified|=overlap; scopes|=set(rule["invalidate"]); full|=rule["full_retrain"] is True
    if classified!=changed_fields:raise ValueError("unclassified target difference")
    result={"invalidate":sorted(scopes),"full_retrain":full,"matrix_hash":canonical_hash(matrix)}
    return result

def validate_manifest(manifest:Mapping[str,Any]):
    if not isinstance(manifest.get("target_audit_hash"),str) or len(manifest["target_audit_hash"])!=64:raise ValueError("target audit parent required")
