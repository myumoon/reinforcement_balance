"""C++ producer の compiled module implementation closure を解決する。

Build.cs、Private 配下の translation unit、依存 module の実装を exact set として扱い、
新規・削除・未分類・片側だけの behavior source を fail-closed にします。
"""

from __future__ import annotations

import fnmatch
import re
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
    headers: tuple[str, ...]
    action_graph_subject_identity: str
    identity_hash: str


_INCLUDE_DIRECTIVE = re.compile(
    r"^\s*#\s*include(?P<body>[^\r\n]*)$", re.MULTILINE
)
_LITERAL_INCLUDE = re.compile(
    r'^\s*(?:"(?P<quote>[^"\r\n]+)"|<(?P<angle>[^>\r\n]+)>)'
    r"\s*(?://[^\r\n]*)?$"
)


def _repo_relative_path(value: Any, label: str) -> str:
    """POSIX repo-relative path を traversal なしで検証する。"""
    ensure(isinstance(value, str) and value and "\\" not in value, f"{label} must be a repo-relative POSIX path")
    path = PurePosixPath(value)
    ensure(not path.is_absolute() and ".." not in path.parts, f"{label} escapes repository")
    return value


def _is_below(relative: str, root: str) -> bool:
    """relative path が root 自身または配下かを path component 単位で判定する。"""
    path_parts = PurePosixPath(relative).parts
    root_parts = PurePosixPath(root).parts
    return path_parts[: len(root_parts)] == root_parts


def _read_includes(path: Path) -> tuple[tuple[str, str], ...]:
    """Literal quote/angle includesを返し、macro等はfail-closedにする。"""
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ContractValidationError(f"cannot read C++ include source: {path}") from exc
    includes: list[tuple[str, str]] = []
    for directive in _INCLUDE_DIRECTIVE.finditer(source):
        match = _LITERAL_INCLUDE.fullmatch(directive.group("body"))
        ensure(match is not None, f"unsupported C++ include directive: {directive.group(0).strip()}")
        quote = match.group("quote")
        includes.append(("quote", quote) if quote is not None else ("angle", match.group("angle")))
    return tuple(includes)


def _resolve_header_closure(
    repo_root: Path,
    source_paths: Sequence[str],
    header_roots: Sequence[str],
    allowed_external_quote_includes: Sequence[str],
) -> tuple[str, ...]:
    """compiled TU を起点に repo-local literal include を再帰解決する。"""
    root_paths = [repo_root / relative for relative in header_roots]
    for relative, path in zip(header_roots, root_paths):
        ensure(path.is_dir(), f"missing header root: {relative}")

    project_header_index: dict[str, list[Path]] = {}
    project_source_roots = [repo_root / "ReinBalance/Source"]
    plugins_root = repo_root / "ReinBalance/Plugins"
    if plugins_root.is_dir():
        project_source_roots.extend(
            plugin_root / "Source"
            for plugin_root in plugins_root.iterdir()
            if plugin_root.is_dir() and (plugin_root / "Source").is_dir()
        )
    for project_source_root in project_source_roots:
        if not project_source_root.is_dir():
            continue
        for header in project_source_root.rglob("*"):
            if header.is_file() and header.suffix.lower() in {".h", ".hpp", ".inl"}:
                project_header_index.setdefault(header.name, []).append(header)

    discovered: set[str] = set()
    pending = list(source_paths)
    processed: set[str] = set()
    while pending:
        including_relative = pending.pop()
        if including_relative in processed:
            continue
        processed.add(including_relative)
        including_path = repo_root / including_relative
        for include_kind, token in _read_includes(including_path):
            ensure("\\" not in token, f"{include_kind} include must use POSIX separators: {token}")
            if token.endswith(".generated.h"):
                continue
            token_path = PurePosixPath(token)
            ensure(not token_path.is_absolute() and ".." not in token_path.parts, f"{include_kind} include escapes repository: {token}")
            candidates: set[Path] = set()
            adjacent = including_path.parent.joinpath(*token_path.parts)
            if adjacent.is_file():
                candidates.add(adjacent.resolve())
            for header_root in root_paths:
                candidate = header_root.joinpath(*token_path.parts)
                if candidate.is_file():
                    candidates.add(candidate.resolve())
            repo_candidates = {
                candidate.resolve()
                for candidate in project_header_index.get(token_path.name, ())
                if candidate.is_file()
                and (
                    candidate.relative_to(repo_root).as_posix() == token
                    or candidate.relative_to(repo_root).as_posix().endswith(f"/{token}")
                )
            }
            candidates.update(repo_candidates)
            ensure(len(candidates) <= 1, f"ambiguous repo-local {include_kind} include {token}: {sorted(str(path) for path in candidates)}")
            if not candidates:
                ensure(
                    token in allowed_external_quote_includes,
                    f"missing repo-local {include_kind} include: {token}",
                )
                continue
            resolved = next(iter(candidates))
            try:
                relative = resolved.relative_to(repo_root).as_posix()
            except ValueError as exc:
                raise ContractValidationError(f"{include_kind} include escapes repository: {token}") from exc
            ensure(any(_is_below(relative, root) for root in header_roots), f"repo-local {include_kind} include is outside declared header roots: {relative}")
            ensure(relative.endswith((".h", ".hpp", ".inl")), f"repo-local {include_kind} include is not a header: {relative}")
            if relative not in discovered:
                discovered.add(relative)
                pending.append(relative)
    return tuple(sorted(discovered))


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


def _declared_module_dependencies(build_cs: Path) -> set[str]:
    """Build.cs の Public/Private dependency 宣言から module 名を抽出する。

    AddRange と Add の文字列 literal を読み、実宣言された repo-local edge の照合に使います。
    """
    try:
        source = build_cs.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ContractValidationError(f"cannot read Build.cs: {build_cs}") from exc
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    source = re.sub(r"//[^\r\n]*", "", source)
    declarations: set[str] = set()
    pattern = re.compile(
        r"(?:Public|Private)DependencyModuleNames\s*\.\s*(?:AddRange|Add)\s*\((.*?)\)\s*;",
        re.DOTALL,
    )
    for match in pattern.finditer(source):
        declarations.update(re.findall(r'"([^"\\]+)"', match.group(1)))
    return declarations


def _repo_local_build_modules(repo_root: Path) -> dict[str, str]:
    """repository 内の .Build.cs を module 名から exact path へ索引化する。

    同名 module が複数あれば曖昧な dependency identity になるため拒否します。
    """
    modules: dict[str, str] = {}
    project_root = repo_root / "ReinBalance"
    if project_root.is_dir():
        candidates: set[Path] = set()
        source_root = project_root / "Source"
        if source_root.is_dir():
            candidates.update(source_root.rglob("*.Build.cs"))
        plugins_root = project_root / "Plugins"
        if plugins_root.is_dir():
            for plugin_root in (entry for entry in plugins_root.iterdir() if entry.is_dir()):
                plugin_source_root = plugin_root / "Source"
                if plugin_source_root.is_dir():
                    candidates.update(plugin_source_root.rglob("*.Build.cs"))
    else:
        candidates = {
            path
            for path in repo_root.rglob("*.Build.cs")
            if not any(part.startswith(".") for part in path.relative_to(repo_root).parts)
        }
    for path in sorted(candidates):
        relative = path.relative_to(repo_root).as_posix()
        module_name = path.name.removesuffix(".Build.cs")
        ensure(module_name not in modules, f"duplicate repo-local module: {module_name}")
        modules[module_name] = relative
    return modules


def resolve_cpp_producer_closure(
    repo_root: Path,
    spec: Mapping[str, Any],
    *,
    action_graph_sources: Sequence[str] | None = None,
    action_graph_subject_identity: str | None = None,
    allowed_external_quote_includes: Sequence[str] = (),
) -> CppProducerClosure:
    """manifest spec と実 filesystem から compiled C++ closure を exact 解決する。

    behavior TU の exclude、manifest/action graph の片側存在、未分類 source を拒否します。
    """
    required = {"module_name", "build_cs", "private_source_roots", "header_roots", "compiled_tu_include_glob", "repo_local_module_dependency_edges", "allowed_non_behavior_excludes"}
    ensure(isinstance(spec, Mapping) and set(spec) == required, "compiled_module_closure keys mismatch")
    ensure(isinstance(spec["module_name"], str) and spec["module_name"], "module_name must be non-empty")
    ensure(action_graph_sources is not None, "action graph sources are required")
    ensure(
        isinstance(action_graph_subject_identity, str)
        and len(action_graph_subject_identity) == 64
        and all(character in "0123456789abcdef" for character in action_graph_subject_identity),
        "action graph subject identity must be lowercase sha256",
    )
    ensure(
        isinstance(allowed_external_quote_includes, (list, tuple))
        and all(isinstance(token, str) and token for token in allowed_external_quote_includes),
        "allowed external quote includes must be strings",
    )
    normalized_external_includes = tuple(
        sorted(_repo_relative_path(token, "allowed external quote include") for token in allowed_external_quote_includes)
    )
    ensure(
        len(normalized_external_includes) == len(set(normalized_external_includes)),
        "duplicate allowed external quote include",
    )
    ensure(
        all(not token.endswith(".generated.h") for token in normalized_external_includes),
        "generated headers must not be external quote includes",
    )
    roots_value = spec["private_source_roots"]
    roots = list(roots_value) if isinstance(roots_value, (list, tuple)) else roots_value
    ensure(isinstance(roots, list) and roots and all(isinstance(x, str) for x in roots), "private_source_roots must be strings")
    roots = [_repo_relative_path(root, "private_source_root") for root in roots]
    header_roots_value = spec["header_roots"]
    header_roots = list(header_roots_value) if isinstance(header_roots_value, (list, tuple)) else header_roots_value
    ensure(isinstance(header_roots, list) and header_roots and all(isinstance(x, str) for x in header_roots), "header_roots must be strings")
    header_roots = [_repo_relative_path(root, "header_root") for root in header_roots]
    glob_pattern = spec["compiled_tu_include_glob"]
    ensure(isinstance(glob_pattern, str) and glob_pattern.endswith(".cpp"), "compiled_tu_include_glob must select .cpp")
    excludes = spec["allowed_non_behavior_excludes"]
    ensure(isinstance(excludes, (list, tuple)), "allowed_non_behavior_excludes must be an array")
    for entry in excludes:
        ensure(isinstance(entry, Mapping) and set(entry) == {"path", "reason"}, "exclude entry keys mismatch")
        ensure(not str(entry["path"]).endswith(".cpp"), "behavior .cpp cannot be excluded")
        ensure(isinstance(entry["reason"], str) and entry["reason"], "exclude reason required")
    build_files = [_repo_relative_path(spec["build_cs"], "build_cs")]
    edges = spec["repo_local_module_dependency_edges"]
    ensure(isinstance(edges, (list, tuple)), "repo_local_module_dependency_edges must be an array")
    declared_edge_names: set[str] = set()
    for edge in edges:
        ensure(isinstance(edge, Mapping) and set(edge) == {"module_name", "build_cs", "private_source_roots", "header_roots"}, "module dependency edge keys mismatch")
        ensure(isinstance(edge["module_name"], str) and edge["module_name"], "dependency module_name required")
        ensure(edge["module_name"] not in declared_edge_names, "duplicate module dependency edge")
        declared_edge_names.add(edge["module_name"])
        build_files.append(_repo_relative_path(edge["build_cs"], "dependency build_cs"))
        ensure(isinstance(edge["private_source_roots"], (list, tuple)) and edge["private_source_roots"], "dependency private_source_roots required")
        for root in edge["private_source_roots"]:
            normalized_root = _repo_relative_path(root, "dependency private_source_root")
            if normalized_root not in roots:
                roots.append(normalized_root)
        ensure(isinstance(edge["header_roots"], (list, tuple)) and edge["header_roots"], "dependency header_roots required")
        for root in edge["header_roots"]:
            normalized_root = _repo_relative_path(root, "dependency header_root")
            if normalized_root not in header_roots:
                header_roots.append(normalized_root)
    repo_modules = _repo_local_build_modules(repo_root)
    discovered_local_dependencies: set[str] = set()
    for build_file in build_files:
        for dependency in _declared_module_dependencies(repo_root / build_file):
            if dependency in repo_modules and dependency != spec["module_name"]:
                discovered_local_dependencies.add(dependency)
    ensure(
        discovered_local_dependencies == declared_edge_names,
        f"repo-local module dependency edges mismatch: missing={sorted(discovered_local_dependencies-declared_edge_names)}, "
        f"undeclared={sorted(declared_edge_names-discovered_local_dependencies)}",
    )
    for edge in edges:
        ensure(repo_modules.get(edge["module_name"]) == edge["build_cs"], f"dependency Build.cs mismatch: {edge['module_name']}")
    resolved: list[ResolvedCppSource] = []
    for root in roots:
        root_path = repo_root / root
        ensure(root_path.is_dir(), f"missing private source root: {root}")
        for path in sorted(root_path.rglob("*.cpp")):
            relative = path.relative_to(repo_root).as_posix()
            ensure(_matches_compiled_glob(relative, glob_pattern) or root != roots[0], f"unclassified behavior TU: {relative}")
            classification = _classify(relative) if root == roots[0] else "module_dependency"
            resolved.append(ResolvedCppSource(relative, classification))
    resolved.sort(key=lambda source: source.path)
    actual = tuple(x.path for x in resolved)
    normalized_graph = tuple(sorted({_repo_relative_path(source, "action graph source") for source in action_graph_sources}))
    ensure(all(source.endswith(".cpp") for source in normalized_graph), "action graph source must be .cpp")
    graph = tuple(source for source in normalized_graph if any(_is_below(source, root) for root in roots))
    ensure(tuple(sorted(actual)) == graph, f"compiled TU set mismatch: manifest_only={sorted(set(actual)-set(graph))}, action_only={sorted(set(graph)-set(actual))}")
    headers = _resolve_header_closure(
        repo_root,
        actual,
        header_roots,
        normalized_external_includes,
    )
    payload_files = []
    for relative in sorted(set(build_files + list(actual) + list(headers))):
        path = repo_root / relative
        ensure(path.is_file(), f"missing producer file: {relative}")
        payload_files.append({"path": relative, "sha256": sha256_hex(path.read_bytes())})
    identity = canonical_hash({
        "module": spec["module_name"],
        "files": payload_files,
        "sources": [{"path": x.path, "classification": x.classification} for x in resolved],
        "headers": list(headers),
        "allowed_external_quote_includes": list(normalized_external_includes),
        "action_graph_subject_identity": action_graph_subject_identity,
    })
    return CppProducerClosure(tuple(build_files), tuple(resolved), headers, action_graph_subject_identity, identity)
