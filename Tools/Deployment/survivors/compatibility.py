"""artifact 互換性／無効化マトリクスを検証し、manifest の親 lineage(target_audit_hash 等)を確認する契約。

生成物(モデルやデータ)が『どの条件が変わると作り直しになるか』の一覧を管理します。
各成果物が正しい前提(＝正解ターゲットの監査結果)から作られたかも確認します。
"""

from pathlib import Path
from typing import Mapping,Any
import yaml
from reinbalance_survivors_contracts.canonical_json import canonical_hash
from reinbalance_survivors_contracts.launch_lifecycle import AuditVerdict

CONFIG=Path(__file__).parents[1]/"configs"/"artifact_compatibility_v1.yaml"
TOP=frozenset({"schema_version","required_parent","classes","artifacts"})
RULE_KEYS=frozenset({"changed_fields","invalidate","full_retrain"})
ARTIFACT_KEYS=frozenset({"compatible_fields","retest_scope"})
MANIFEST_KEYS=frozenset({"schema_version","artifact_kind","target_audit_hash","target_audit_verdict"})
REQUIRED_CLASSES={
 "ui_only":{"changed_fields":["window_mode","client_resolution","ui_language","ui_revision","ui_scale","windows_dpi_scale","monitor_output","capture_backend"],"invalidate":["parser","replay"],"full_retrain":False},
 "content_unlock":{"changed_fields":["stage","character","modifiers","success_timer_seconds","post_timer_event_required","distribution","app_id","content_revision","manual_attestation","save_artifact_hash","save_format_version","unlocked_characters","unlocked_items","unlocked_stages","collection_pool","purchased_power_ups","reroll_count","skip_count","banish_count","level_up_card_counts","fallbacks","capabilities","states","candidate_vocabulary","candidate_vocabulary_hash"],"invalidate":["taxonomy","selector","fidelity"],"full_retrain":False},
 "physics_input":{"changed_fields":["schema_version","platform","build_id","executable_version","executable_hash","profile_id","os_build","gpu_name","vram_mb","driver_version","cuda_version","pytorch_version","cpu_live_fallback","key_bindings","action_semantics_version","frame_skip","physics_hz","decision_hz","key_hold_ms","lease_ms"],"invalidate":["fidelity","policy","canary"],"full_retrain":True},
}
REQUIRED_ARTIFACTS={
 "capture":{"compatible_fields":["content_revision"],"retest_scope":["parser","replay"]},
 "model":{"compatible_fields":["ui_revision"],"retest_scope":["policy","canary"]},
 "runtime":{"compatible_fields":[],"retest_scope":["fidelity","canary"]},
 "campaign":{"compatible_fields":[],"retest_scope":["full_preflight"]},
}

def load_matrix(path:Path=CONFIG):
    with path.open(encoding="utf-8") as f:data=yaml.safe_load(f)
    if not isinstance(data,dict) or set(data)!=TOP or data["schema_version"]!="artifact_compatibility.v1" or data["required_parent"]!="target_audit_hash":raise ValueError("invalid compatibility matrix")
    if set(data["classes"])!={"ui_only","content_unlock","physics_input"} or set(data["artifacts"])!={"capture","model","runtime","campaign"}: raise ValueError("invalid compatibility taxonomy")
    if any(set(v)!=RULE_KEYS or type(v["full_retrain"]) is not bool or not all(isinstance(x,list) and all(isinstance(i,str) for i in x) for x in (v["changed_fields"],v["invalidate"])) for v in data["classes"].values()): raise ValueError("invalid compatibility rule")
    if any(set(v)!=ARTIFACT_KEYS or not all(isinstance(x,list) and all(isinstance(i,str) for i in x) for x in (v["compatible_fields"],v["retest_scope"])) for v in data["artifacts"].values()): raise ValueError("invalid artifact rule")
    from .target_profile import SECTIONS
    classified=[field for rule in data["classes"].values() for field in rule["changed_fields"]]
    expected={"schema_version",*(field for fields in SECTIONS.values() for field in fields)}
    if set(classified)!=expected or len(classified)!=len(set(classified)): raise ValueError("target profile matrix coverage mismatch")
    if data["classes"]!=REQUIRED_CLASSES or data["artifacts"]!=REQUIRED_ARTIFACTS: raise ValueError("compatibility semantics changed")
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
    if not isinstance(manifest,Mapping) or set(manifest)!=MANIFEST_KEYS or manifest.get("schema_version")!="artifact_manifest.v1" or manifest.get("artifact_kind") not in {"capture","model","runtime","campaign"}: raise ValueError("invalid manifest")
    verdict=manifest.get("target_audit_verdict")
    if not isinstance(verdict,AuditVerdict): raise ValueError("verified audit verdict parent required")
    if manifest.get("target_audit_hash")!=verdict.canonical_hash: raise ValueError("target audit parent identity mismatch")
