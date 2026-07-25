"""Versioned fidelity producer path manifest の厳格 loader と hash 解決。

各 gating key の exact path、recursive root、generated input、dependency mode を検証し、
manifest bytes 自体も producer identity に含めます。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .canonical_json import canonical_hash, sha256_hex
from .fidelity_verdict import GATING_KEYS, PRODUCER_ALLOWLIST_VERSION
from .ui_intent import ContractValidationError, ensure

_ENTRY_KEYS = {"ordered_exact_paths", "recursive_roots", "explicit_excludes", "generated_inputs", "transitive_dependency_mode"}


@dataclass(frozen=True)
class ProducerPathManifest:
    """検証済み producer manifest と canonical identity。

    raw object は canonical JSON compatible として固定的に扱い、manifest hash を verdict subject に保存できます。
    """
    schema_version: str
    producers: Mapping[str, Mapping[str, Any]]
    manifest_hash: str


def _repo_path(value: Any, label: str) -> str:
    """repo-relative POSIX path を traversal なしで検証する。

    absolute path、backslash、空 path を拒否して platform 差と repository escape を防ぎます。
    """
    ensure(isinstance(value, str) and value and "\\" not in value, f"{label} must be a repo-relative POSIX path")
    path = PurePosixPath(value)
    ensure(not path.is_absolute() and ".." not in path.parts, f"{label} escapes repository")
    return value


def load_producer_path_manifest(path: Path | None = None) -> ProducerPathManifest:
    """JSON manifest を未知・欠落 key なしで読み込む。

    全13 producer と全 nested entry を対称に検証し、C++ entry には compiled closure を要求します。
    """
    manifest_path = path or Path(str(files("reinbalance_survivors_contracts").joinpath("schemas/fidelity_producer_paths_v1.json")))
    raw_bytes = manifest_path.read_bytes()
    try:
        data = json.loads(raw_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractValidationError(f"invalid producer manifest JSON: {exc}") from exc
    ensure(isinstance(data, Mapping) and set(data) == {"schema_version", "producers"}, "producer manifest top-level keys mismatch")
    ensure(data["schema_version"] == PRODUCER_ALLOWLIST_VERSION, "unsupported producer manifest schema_version")
    producers = data["producers"]
    ensure(isinstance(producers, Mapping) and set(producers) == set(GATING_KEYS), "producer manifest gating keys mismatch")
    for key, entry in producers.items():
        ensure(isinstance(entry, Mapping), f"producer {key} must be an object")
        expected = set(_ENTRY_KEYS)
        if key in {"logic_public", "logic_private", "game_facade", "http_service"}:
            expected.add("compiled_module_closure")
        ensure(set(entry) == expected, f"producer {key} keys mismatch")
        for array_key in ("ordered_exact_paths", "recursive_roots", "generated_inputs", "explicit_excludes"):
            ensure(isinstance(entry[array_key], list), f"{key}.{array_key} must be an array")
        for item in entry["ordered_exact_paths"]:
            _repo_path(item, f"{key}.ordered_exact_paths")
        for root in entry["recursive_roots"]:
            ensure(isinstance(root, Mapping) and set(root) == {"path", "include_globs"}, f"{key}.recursive_root keys mismatch")
            _repo_path(root["path"], f"{key}.recursive_root.path")
            ensure(isinstance(root["include_globs"], list) and all(isinstance(x, str) and x for x in root["include_globs"]), f"{key}.include_globs invalid")
        for excluded in entry["explicit_excludes"]:
            ensure(isinstance(excluded, Mapping) and set(excluded) == {"path", "reason"}, f"{key}.exclude keys mismatch")
            _repo_path(excluded["path"], f"{key}.exclude.path")
            ensure(isinstance(excluded["reason"], str) and excluded["reason"], f"{key}.exclude reason required")
        ensure(entry["transitive_dependency_mode"] in {"none", "generated_schema", "python_import_closure", "compiled_module_closure"}, f"{key}.transitive_dependency_mode invalid")
        if "compiled_module_closure" in entry:
            closure = entry["compiled_module_closure"]
            closure_keys = {"module_name", "build_cs", "private_source_roots", "compiled_tu_include_glob", "repo_local_module_dependency_edges", "allowed_non_behavior_excludes"}
            ensure(isinstance(closure, Mapping) and set(closure) == closure_keys, f"{key}.compiled_module_closure keys mismatch")
            ensure(isinstance(closure["module_name"], str) and closure["module_name"], f"{key}.module_name required")
            _repo_path(closure["build_cs"], f"{key}.build_cs")
            ensure(isinstance(closure["private_source_roots"], list) and closure["private_source_roots"], f"{key}.private_source_roots required")
            for root in closure["private_source_roots"]:
                _repo_path(root, f"{key}.private_source_root")
            ensure(isinstance(closure["compiled_tu_include_glob"], str) and closure["compiled_tu_include_glob"].endswith(".cpp"), f"{key}.compiled_tu_include_glob invalid")
            ensure(isinstance(closure["repo_local_module_dependency_edges"], list), f"{key}.dependency_edges invalid")
            for edge in closure["repo_local_module_dependency_edges"]:
                ensure(isinstance(edge, Mapping) and set(edge) == {"module_name", "build_cs", "private_source_roots"}, f"{key}.dependency edge keys mismatch")
                _repo_path(edge["build_cs"], f"{key}.dependency build_cs")
                ensure(isinstance(edge["private_source_roots"], list), f"{key}.dependency roots invalid")
                for root in edge["private_source_roots"]:
                    _repo_path(root, f"{key}.dependency root")
            ensure(isinstance(closure["allowed_non_behavior_excludes"], list), f"{key}.closure excludes invalid")
            for excluded in closure["allowed_non_behavior_excludes"]:
                ensure(isinstance(excluded, Mapping) and set(excluded) == {"path", "reason"}, f"{key}.closure exclude keys mismatch")
                _repo_path(excluded["path"], f"{key}.closure exclude path")
                ensure(not excluded["path"].endswith(".cpp"), f"{key} cannot exclude behavior TU")
                ensure(isinstance(excluded["reason"], str) and excluded["reason"], f"{key}.closure exclude reason required")
    return ProducerPathManifest(data["schema_version"], dict(producers), sha256_hex(raw_bytes))


def resolve_gating_producer_hashes(repo_root: Path, manifest: ProducerPathManifest, generated_inputs: Mapping[str, Any]) -> dict[str, str]:
    """manifest の exact file bytes と generated canonical bytes から gating map を作る。

    absent は明示入力に限って認め、欠落 file や未指定 generated input を推測で補いません。
    """
    ensure(isinstance(generated_inputs, Mapping), "generated_inputs must be an object")
    result: dict[str, str] = {}
    for key in GATING_KEYS:
        entry = manifest.producers[key]
        if generated_inputs.get(key) == "absent":
            ensure(key in {"deploy_obs_schema", "deploy_release_adapter"}, f"{key} cannot be absent")
            result[key] = "absent"
            continue
        records: list[dict[str, str]] = []
        for relative in entry["ordered_exact_paths"]:
            path = repo_root / relative
            ensure(path.is_file(), f"missing producer path: {relative}")
            records.append({"path": relative, "sha256": sha256_hex(path.read_bytes())})
        generated = {}
        for name in entry["generated_inputs"]:
            ensure(name in generated_inputs, f"missing generated input: {name}")
            generated[name] = generated_inputs[name]
        result[key] = canonical_hash({"manifest_hash": manifest.manifest_hash, "key": key, "files": records, "generated_inputs": generated})
    return result
