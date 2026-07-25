"""Versioned fidelity producer path manifest の厳格 loader と hash 解決。

各 gating key の exact path、recursive root、generated input、dependency mode を検証し、
manifest bytes 自体も producer identity に含めます。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping

from .canonical_json import canonical_hash, sha256_hex
from .cpp_producer_closure import resolve_cpp_producer_closure
from .fidelity_verdict import GATING_KEYS, PRODUCER_ALLOWLIST_VERSION
from .ui_intent import ContractValidationError, ensure

_ENTRY_KEYS = {"ordered_exact_paths", "recursive_roots", "explicit_excludes", "generated_inputs", "transitive_dependency_mode"}
_CPP_PRODUCER_KEYS = {"logic_public", "logic_private", "game_facade", "http_service"}
_DEPENDENCY_MODES = {"none", "generated_schema", "python_import_closure", "compiled_module_closure"}


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


def _validate_producers(producers: Any) -> None:
    """全 producer entry を利用時と読込時に同じ規則で再検証する。

    検証済み object の差し替えや欠落も gating 計算直前に検知し、fail-closed にします。
    """
    ensure(isinstance(producers, Mapping) and set(producers) == set(GATING_KEYS), "producer manifest gating keys mismatch")
    for key, entry in producers.items():
        ensure(isinstance(entry, Mapping), f"producer {key} must be an object")
        expected = set(_ENTRY_KEYS)
        if key in _CPP_PRODUCER_KEYS:
            expected.add("compiled_module_closure")
        ensure(set(entry) == expected, f"producer {key} keys mismatch")
        for array_key in ("ordered_exact_paths", "recursive_roots", "generated_inputs", "explicit_excludes"):
            ensure(isinstance(entry[array_key], (list, tuple)), f"{key}.{array_key} must be an array")
        for item in entry["ordered_exact_paths"]:
            _repo_path(item, f"{key}.ordered_exact_paths")
        for root in entry["recursive_roots"]:
            ensure(isinstance(root, Mapping) and set(root) == {"path", "include_globs"}, f"{key}.recursive_root keys mismatch")
            _repo_path(root["path"], f"{key}.recursive_root.path")
            ensure(isinstance(root["include_globs"], (list, tuple)) and all(isinstance(x, str) and x for x in root["include_globs"]), f"{key}.include_globs invalid")
        for excluded in entry["explicit_excludes"]:
            ensure(isinstance(excluded, Mapping) and set(excluded) == {"path", "reason"}, f"{key}.exclude keys mismatch")
            _repo_path(excluded["path"], f"{key}.exclude.path")
            ensure(isinstance(excluded["reason"], str) and excluded["reason"], f"{key}.exclude reason required")
        ensure(entry["transitive_dependency_mode"] in _DEPENDENCY_MODES, f"{key}.transitive_dependency_mode invalid")
        if key in _CPP_PRODUCER_KEYS:
            ensure(
                entry["transitive_dependency_mode"] == "compiled_module_closure",
                f"{key}.transitive_dependency_mode must be compiled_module_closure",
            )
        if "compiled_module_closure" in entry:
            closure = entry["compiled_module_closure"]
            closure_keys = {"module_name", "build_cs", "private_source_roots", "compiled_tu_include_glob", "repo_local_module_dependency_edges", "allowed_non_behavior_excludes"}
            ensure(isinstance(closure, Mapping) and set(closure) == closure_keys, f"{key}.compiled_module_closure keys mismatch")
            ensure(isinstance(closure["module_name"], str) and closure["module_name"], f"{key}.module_name required")
            _repo_path(closure["build_cs"], f"{key}.build_cs")
            ensure(isinstance(closure["private_source_roots"], (list, tuple)) and closure["private_source_roots"], f"{key}.private_source_roots required")
            for root in closure["private_source_roots"]:
                _repo_path(root, f"{key}.private_source_root")
            ensure(isinstance(closure["compiled_tu_include_glob"], str) and closure["compiled_tu_include_glob"].endswith(".cpp"), f"{key}.compiled_tu_include_glob invalid")
            ensure(isinstance(closure["repo_local_module_dependency_edges"], (list, tuple)), f"{key}.dependency_edges invalid")
            for edge in closure["repo_local_module_dependency_edges"]:
                ensure(isinstance(edge, Mapping) and set(edge) == {"module_name", "build_cs", "private_source_roots"}, f"{key}.dependency edge keys mismatch")
                ensure(isinstance(edge["module_name"], str) and edge["module_name"], f"{key}.dependency module_name required")
                _repo_path(edge["build_cs"], f"{key}.dependency build_cs")
                ensure(isinstance(edge["private_source_roots"], (list, tuple)), f"{key}.dependency roots invalid")
                for root in edge["private_source_roots"]:
                    _repo_path(root, f"{key}.dependency root")
            ensure(isinstance(closure["allowed_non_behavior_excludes"], (list, tuple)), f"{key}.closure excludes invalid")
            for excluded in closure["allowed_non_behavior_excludes"]:
                ensure(isinstance(excluded, Mapping) and set(excluded) == {"path", "reason"}, f"{key}.closure exclude keys mismatch")
                _repo_path(excluded["path"], f"{key}.closure exclude path")
                ensure(not excluded["path"].endswith(".cpp"), f"{key} cannot exclude behavior TU")
                ensure(isinstance(excluded["reason"], str) and excluded["reason"], f"{key}.closure exclude reason required")


def _deep_freeze(value: Any) -> Any:
    """JSON tree を再帰的な immutable representation へ変換する。

    nested dict/list も MappingProxyType/tuple に変え、検証後の identity 改変を防ぎます。
    """
    if isinstance(value, Mapping):
        return MappingProxyType({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    return value


def load_producer_path_manifest(path: Path | None = None) -> ProducerPathManifest:
    """JSON manifest を検証し、全 nested 階層を immutable 化して読み込む。

    全13 producer と C++ compiled closure を対称に固定し、読込後の改変を拒否します。
    """
    manifest_path = path or Path(str(files("reinbalance_survivors_contracts").joinpath("schemas/fidelity_producer_paths_v1.json")))
    raw_bytes = manifest_path.read_bytes()
    try:
        data = json.loads(raw_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractValidationError(f"invalid producer manifest JSON: {exc}") from exc
    ensure(isinstance(data, Mapping) and set(data) == {"schema_version", "producers"}, "producer manifest top-level keys mismatch")
    ensure(data["schema_version"] == PRODUCER_ALLOWLIST_VERSION, "unsupported producer manifest schema_version")
    _validate_producers(data["producers"])
    return ProducerPathManifest(data["schema_version"], _deep_freeze(data["producers"]), sha256_hex(raw_bytes))


def resolve_gating_producer_hashes(repo_root: Path, manifest: ProducerPathManifest, generated_inputs: Mapping[str, Any]) -> dict[str, str]:
    """manifest の exact file bytes と generated canonical bytes から gating map を作る。

    absent は明示入力に限って認め、欠落 file や未指定 generated input を推測で補いません。
    """
    ensure(isinstance(generated_inputs, Mapping), "generated_inputs must be an object")
    ensure(isinstance(manifest, ProducerPathManifest), "manifest must be ProducerPathManifest")
    ensure(isinstance(manifest.schema_version, str) and manifest.schema_version, "manifest schema_version required")
    ensure(isinstance(manifest.manifest_hash, str) and len(manifest.manifest_hash) == 64, "manifest_hash invalid")
    _validate_producers(manifest.producers)
    result: dict[str, str] = {}
    for key in GATING_KEYS:
        entry = manifest.producers[key]
        if generated_inputs.get(key) == "absent":
            ensure(key in {"deploy_obs_schema", "deploy_release_adapter"}, f"{key} cannot be absent")
            result[key] = "absent"
            continue
        records: list[dict[str, str]] = []
        known_paths: set[str] = set()
        for relative in entry["ordered_exact_paths"]:
            path = repo_root / relative
            ensure(path.is_file(), f"missing producer path: {relative}")
            records.append({"path": relative, "sha256": sha256_hex(path.read_bytes())})
            known_paths.add(relative)
        excluded_paths = {item["path"] for item in entry["explicit_excludes"]}
        for root_spec in entry["recursive_roots"]:
            root_relative = root_spec["path"]
            root = repo_root / root_relative
            ensure(root.is_dir(), f"missing producer recursive root: {root_relative}")
            matched: set[Path] = set()
            for include_glob in root_spec["include_globs"]:
                matched.update(path for path in root.glob(include_glob) if path.is_file())
            for path in sorted(matched):
                relative = path.relative_to(repo_root).as_posix()
                if relative in excluded_paths or relative in known_paths:
                    continue
                records.append({"path": relative, "sha256": sha256_hex(path.read_bytes())})
                known_paths.add(relative)
        generated = {}
        for name in entry["generated_inputs"]:
            ensure(name in generated_inputs, f"missing generated input: {name}")
            generated[name] = generated_inputs[name]
        closure_identity = None
        if entry["transitive_dependency_mode"] == "compiled_module_closure":
            closure = resolve_cpp_producer_closure(repo_root, entry["compiled_module_closure"])
            closure_identity = closure.identity_hash
            for relative in (*closure.build_files, *(source.path for source in closure.sources)):
                if relative not in known_paths:
                    records.append({"path": relative, "sha256": sha256_hex((repo_root / relative).read_bytes())})
                    known_paths.add(relative)
        result[key] = canonical_hash({
            "manifest_hash": manifest.manifest_hash,
            "key": key,
            "files": records,
            "generated_inputs": generated,
            "compiled_closure_hash": closure_identity,
        })
    return result
