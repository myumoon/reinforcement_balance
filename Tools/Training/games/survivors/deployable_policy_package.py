"""正式 Deployable Student Policy checkpoint の Python runtime package builder。
development/非eligible metadata を出力作成前に拒否し、optimizer や training RNG を除いた
model state と hash-bound manifest だけを directory package として公開します。
"""
from __future__ import annotations
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping
import torch as th
from reinbalance_survivors_contracts.canonical_json import canonical_json_bytes, sha256_hex
from games.survivors.deployable_policy_trainer import CHECKPOINT_SCHEMA_VERSION
PACKAGE_SCHEMA_VERSION = "survivors.deployable_policy_package.v1"
_MANIFEST_KEYS = frozenset({
    "schema_version", "checkpoint_sha256", "model_sha256", "deploy_schema_hash", "model_config",
    "development_only", "formal_student_eligible", "formal_dependency_identities", "files",
})
def _sha256(value: Any, label: str) -> str:
    """lowercase SHA-256 identity だけを受理する。
    package の lineage と content hash を空値や短縮値で代用させません。
    """
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{label} must be lowercase SHA-256")
    return value
def _formal_metadata(value: Mapping[str, Any]) -> Mapping[str, str]:
    """development/eligible/dependency identity siblings を対称検証する。
    checkpoint reader と package reader の双方で同じ正式昇格条件を再利用します。
    """
    if value.get("development_only") is not False:
        raise ValueError("development_only artifact cannot be packaged or loaded")
    if value.get("formal_student_eligible") is not True:
        raise ValueError("formal_student_eligible=false artifact cannot be packaged or loaded")
    identities = value.get("formal_dependency_identities")
    if not isinstance(identities, Mapping) or set(identities) != {"fidelity_verdict", "perception_profile"}:
        raise ValueError("formal dependency identities are missing")
    for name, identity in identities.items():
        _sha256(identity, name)
    return identities
def _checkpoint(path: Path) -> Mapping[str, Any]:
    """package mutation 前に checkpoint schema と eligibility siblings を検証する。
    development_only と formal_student_eligible の片方だけを確認する抜け道を作りません。
    """
    try:
        payload = th.load(Path(path), map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"cannot inspect deployable checkpoint: {exc}") from exc
    if not isinstance(payload, Mapping) or payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported deployable checkpoint")
    _formal_metadata(payload)
    _sha256(payload.get("deploy_schema_hash"), "deploy_schema_hash")
    if not isinstance(payload.get("model_config"), Mapping) or not isinstance(payload.get("model_state_dict"), Mapping):
        raise ValueError("checkpoint model payload is missing")
    return payload
def package_deployable_policy(checkpoint_path: Path, output_dir: Path) -> Path:
    """eligible checkpoint から model-only runtime package を原子的に作る。
    gate 完了後に sibling directory で全 file を構築し、最後の rename だけを公開点にします。
    """
    checkpoint_path, output = Path(checkpoint_path), Path(output_dir)
    payload = _checkpoint(checkpoint_path)
    if output.exists():
        raise ValueError("package output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        model_payload = {"model_config": payload["model_config"], "model_state_dict": payload["model_state_dict"]}
        th.save(model_payload, staging / "model.pt")
        manifest = {
            "schema_version": PACKAGE_SCHEMA_VERSION, "checkpoint_sha256": sha256_hex(checkpoint_path.read_bytes()),
            "model_sha256": sha256_hex((staging / "model.pt").read_bytes()), "deploy_schema_hash": payload["deploy_schema_hash"],
            "model_config": payload["model_config"], "development_only": False, "formal_student_eligible": True,
            "formal_dependency_identities": dict(payload["formal_dependency_identities"]), "files": ["manifest.json", "model.pt"],
        }
        (staging / "manifest.json").write_bytes(canonical_json_bytes(manifest))
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output
def load_deployable_policy_package(package_dir: Path) -> tuple[dict[str, Any], Mapping[str, Any]]:
    """runtime package の exact files、metadata、model hash を再検証して読む。
    load 側でも development/eligibility gate を繰り返し、改変 manifest を正式 policy として扱いません。
    """
    root = Path(package_dir)
    try:
        manifest = json.loads((root / "manifest.json").read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read deployable package: {exc}") from exc
    if not isinstance(manifest, dict) or set(manifest) != _MANIFEST_KEYS or manifest["schema_version"] != PACKAGE_SCHEMA_VERSION:
        raise ValueError("deployable package manifest mismatch")
    _formal_metadata(manifest)
    for field in ("checkpoint_sha256", "model_sha256", "deploy_schema_hash"):
        _sha256(manifest[field], field)
    if manifest["files"] != ["manifest.json", "model.pt"] or {path.name for path in root.iterdir()} != set(manifest["files"]):
        raise ValueError("deployable package files mismatch")
    if sha256_hex((root / "model.pt").read_bytes()) != manifest["model_sha256"]:
        raise ValueError("deployable package model hash mismatch")
    model = th.load(root / "model.pt", map_location="cpu", weights_only=False)
    if not isinstance(model, dict) or set(model) != {"model_config", "model_state_dict"} or model["model_config"] != manifest["model_config"]:
        raise ValueError("deployable package model binding mismatch")
    return manifest, model
