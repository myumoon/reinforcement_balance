"""C++ producer の compiled module implementation closure を解決する。

Build.cs、Private 配下の translation unit、依存 module の実装を exact set として扱い、
新規・削除・未分類・片側だけの behavior source を fail-closed にします。
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from .canonical_json import canonical_hash, sha256_hex
from .ui_intent import ContractValidationError, ensure


@dataclass(frozen=True)
class ResolvedCppSource:
    """分類済み compiled translation unit。

    path と classification を検証後に固定し、hash 集計中の書き換えを防ぎます。
    """
    path: str
    classification: str

    def __post_init__(self) -> None:
        """repo-relative path と既知 classification を検証する。

        絶対 path や traversal を拒否して repository 外の bytes を identity に混ぜません。
        """
        pure = PurePosixPath(self.path)
        ensure(not pure.is_absolute() and ".." not in pure.parts, "C++ source path must be repo-relative")
        ensure(self.path.endswith(".cpp"), "compiled source must be .cpp")
        ensure(self.classification in {"survivors_logic", "weapon_logic", "collision", "content", "helper", "module_dependency"}, "unknown C++ source classification")


@dataclass(frozen=True)
class CppProducerClosure:
    """Build.cs と分類済み TU の不変 closure。

    canonical identity は path、classification、各 file bytes の共有 SHA-256 から生成します。
    """
    build_files: tuple[str, ...]
    sources: tuple[ResolvedCppSource, ...]
    identity_hash: str


def _classify(path: str) -> str:
    """ReinBalanceLogic の TU を責務別に分類する。

    weapon、collision、content/helper、中心 Logic の順で判定し、監査表示を安定させます。
    """
    name = PurePosixPath(path).name
    if "/Weapons/" in path:
        return "weapon_logic"
    if "Collision" in name:
        return "collision"
    if "GameLogic" in name:
        return "survivors_logic"
    if any(token in name for token in ("Content", "Wiki", "Table")):
        return "content"
    return "helper"


def _matches_compiled_glob(path: str, pattern: str) -> bool:
    """`**/` のゼロ階層を含めて compiled TU glob を評価する。

    Python fnmatch の実装差で Private 直下の TU が未分類扱いにならないよう、再帰 glob の
    ゼロ directory 形も同じ規則として照合します。
    """
    candidates = {pattern}
    while "/**/" in pattern:
        pattern = pattern.replace("/**/", "/", 1)
        candidates.add(pattern)
    return any(fnmatch.fnmatch(path, candidate) for candidate in candidates)


def resolve_cpp_producer_closure(
    repo_root: Path,
    spec: Mapping[str, Any],
    *,
    action_graph_sources: Sequence[str] | None = None,
) -> CppProducerClosure:
    """manifest spec と実 filesystem から compiled C++ closure を exact 解決する。

    behavior TU の exclude、manifest/action graph の片側存在、未分類 source を拒否します。
    """
    required = {"module_name", "build_cs", "private_source_roots", "compiled_tu_include_glob", "repo_local_module_dependency_edges", "allowed_non_behavior_excludes"}
    ensure(isinstance(spec, Mapping) and set(spec) == required, "compiled_module_closure keys mismatch")
    ensure(isinstance(spec["module_name"], str) and spec["module_name"], "module_name must be non-empty")
    roots_value = spec["private_source_roots"]
    roots = list(roots_value) if isinstance(roots_value, list) else roots_value
    ensure(isinstance(roots, list) and roots and all(isinstance(x, str) for x in roots), "private_source_roots must be strings")
    glob_pattern = spec["compiled_tu_include_glob"]
    ensure(isinstance(glob_pattern, str) and glob_pattern.endswith(".cpp"), "compiled_tu_include_glob must select .cpp")
    excludes = spec["allowed_non_behavior_excludes"]
    ensure(isinstance(excludes, list), "allowed_non_behavior_excludes must be an array")
    for entry in excludes:
        ensure(isinstance(entry, Mapping) and set(entry) == {"path", "reason"}, "exclude entry keys mismatch")
        ensure(not str(entry["path"]).endswith(".cpp"), "behavior .cpp cannot be excluded")
        ensure(isinstance(entry["reason"], str) and entry["reason"], "exclude reason required")
    build_files = [spec["build_cs"]]
    edges = spec["repo_local_module_dependency_edges"]
    ensure(isinstance(edges, list), "repo_local_module_dependency_edges must be an array")
    for edge in edges:
        ensure(isinstance(edge, Mapping) and set(edge) == {"module_name", "build_cs", "private_source_roots"}, "module dependency edge keys mismatch")
        build_files.append(edge["build_cs"])
        roots.extend(x for x in edge["private_source_roots"] if x not in roots)
    resolved: list[ResolvedCppSource] = []
    for root in roots:
        root_path = repo_root / root
        ensure(root_path.is_dir(), f"missing private source root: {root}")
        for path in sorted(root_path.rglob("*.cpp")):
            relative = path.relative_to(repo_root).as_posix()
            ensure(_matches_compiled_glob(relative, glob_pattern) or root != roots[0], f"unclassified behavior TU: {relative}")
            classification = _classify(relative) if root == roots[0] else "module_dependency"
            resolved.append(ResolvedCppSource(relative, classification))
    actual = tuple(x.path for x in resolved)
    if action_graph_sources is not None:
        graph = tuple(sorted(action_graph_sources))
        ensure(tuple(sorted(actual)) == graph, f"compiled TU set mismatch: manifest_only={sorted(set(actual)-set(graph))}, action_only={sorted(set(graph)-set(actual))}")
    payload_files = []
    for relative in sorted(set(build_files + list(actual))):
        path = repo_root / relative
        ensure(path.is_file(), f"missing producer file: {relative}")
        payload_files.append({"path": relative, "sha256": sha256_hex(path.read_bytes())})
    identity = canonical_hash({"module": spec["module_name"], "files": payload_files, "sources": [{"path": x.path, "classification": x.classification} for x in resolved]})
    return CppProducerClosure(tuple(build_files), tuple(resolved), identity)
