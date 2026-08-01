"""Compiled C++ producer closure の exact-set と byte sensitivity を検証する。

header を変えない TU の変更・追加・削除と action graph の片側存在を fixture で扱います。
"""

from pathlib import Path

import pytest

from reinbalance_survivors_contracts.cpp_producer_closure import resolve_cpp_producer_closure
from reinbalance_survivors_contracts.ui_intent import ContractValidationError


def _fixture(root: Path) -> dict:
    """最小 module fixture と厳格 compiled closure spec を作る。

    weapon と helper の両 TU を含め、byte 差分検証に利用します。
    """
    private = root / "Module/Private/Survivors/Weapons"
    public = root / "Module/Public/Survivors/Weapons"
    private.mkdir(parents=True)
    public.mkdir(parents=True)
    (root / "Module/Module.Build.cs").write_text("module", encoding="utf-8")
    (private / "Weapon.cpp").write_text('#include "Survivors/Weapons/Weapon.h"\none', encoding="utf-8")
    (public / "Weapon.h").write_text("header", encoding="utf-8")
    return {"module_name": "Module", "build_cs": "Module/Module.Build.cs", "private_source_roots": ["Module/Private"], "header_roots": ["Module/Public", "Module/Private"], "compiled_tu_include_glob": "Module/Private/**/*.cpp", "repo_local_module_dependency_edges": [], "allowed_non_behavior_excludes": []}


def _resolve(root: Path, spec: dict, *, allowed_external_quote_includes=()):
    sources = sorted(path.relative_to(root).as_posix() for path in root.rglob("*.cpp"))
    return resolve_cpp_producer_closure(
        root,
        spec,
        action_graph_sources=sources,
        action_graph_subject_identity="a" * 64,
        allowed_external_quote_includes=allowed_external_quote_includes,
    )


def test_cpp_byte_change_and_new_tu_change_identity(tmp_path) -> None:
    """既存 weapon byte 変更と新 TU 追加で identity が変わる。

    header-only identity では見逃す implementation 差を検出します。
    """
    spec = _fixture(tmp_path)
    first = _resolve(tmp_path, spec)
    source = tmp_path / "Module/Private/Survivors/Weapons/Weapon.cpp"
    source.write_text("two", encoding="utf-8")
    second = _resolve(tmp_path, spec)
    assert first.identity_hash != second.identity_hash
    (source.parent / "NewWeapon.cpp").write_text("new", encoding="utf-8")
    third = _resolve(tmp_path, spec)
    assert second.identity_hash != third.identity_hash


def test_action_graph_mismatch_and_behavior_exclude_are_rejected(tmp_path) -> None:
    """action graph/manifest の片側 TU と behavior exclude を拒否する。

    compiled source の未分類や隠蔽を fail-closed にします。
    """
    spec = _fixture(tmp_path)
    with pytest.raises(ContractValidationError):
        resolve_cpp_producer_closure(
            tmp_path,
            spec,
            action_graph_sources=["only/action.cpp"],
            action_graph_subject_identity="a" * 64,
        )
    with pytest.raises(ContractValidationError, match="action graph"):
        resolve_cpp_producer_closure(tmp_path, spec)
    spec["allowed_non_behavior_excludes"] = [{"path": "Module/Private/Survivors/Weapons/Weapon.cpp", "reason": "bad"}]
    with pytest.raises(ContractValidationError):
        _resolve(tmp_path, spec)


def test_untracked_repo_local_build_dependency_is_rejected(tmp_path) -> None:
    """Build.cs にだけ追加された repo-local module dependency を拒否する。

    manifest edge に無い helper module の TU が gating identity から漏れるのを防ぎます。
    """
    spec = _fixture(tmp_path)
    helper_private = tmp_path / "Helper/Private"
    helper_private.mkdir(parents=True)
    (tmp_path / "Helper/Helper.Build.cs").write_text("using UnrealBuildTool;", encoding="utf-8")
    (helper_private / "Behavior.cpp").write_text("behavior", encoding="utf-8")
    (tmp_path / "Module/Module.Build.cs").write_text(
        'PublicDependencyModuleNames.AddRange(new string[] { "Core", "Helper" });',
        encoding="utf-8",
    )
    with pytest.raises(ContractValidationError, match="missing=.*Helper"):
        _resolve(tmp_path, spec)

    spec["repo_local_module_dependency_edges"] = [{
        "module_name": "Helper",
        "build_cs": "Helper/Helper.Build.cs",
        "private_source_roots": ["Helper/Private"],
        "header_roots": ["Helper/Public", "Helper/Private"],
    }]
    (tmp_path / "Helper/Public").mkdir()
    closure = _resolve(tmp_path, spec)
    assert any(source.path == "Helper/Private/Behavior.cpp" for source in closure.sources)


def test_dependency_public_header_transitive_leaf_changes_identity(tmp_path: Path) -> None:
    """dependency Public header の再帰 leaf bytesをclosure identityへ含める。"""
    spec = _fixture(tmp_path)
    helper_public = tmp_path / "Helper/Public/Net"
    helper_private = tmp_path / "Helper/Private"
    helper_public.mkdir(parents=True)
    helper_private.mkdir(parents=True)
    (tmp_path / "Helper/Helper.Build.cs").write_text("module", encoding="utf-8")
    (helper_private / "Http.cpp").write_text('#include "Net/HttpEnvServerBase.h"', encoding="utf-8")
    (helper_public / "HttpEnvServerBase.h").write_text('#include "Net/Leaf.h"', encoding="utf-8")
    leaf = helper_public / "Leaf.h"
    leaf.write_text("one", encoding="utf-8")
    (tmp_path / "Module/Module.Build.cs").write_text(
        'PublicDependencyModuleNames.Add("Helper");', encoding="utf-8"
    )
    spec["repo_local_module_dependency_edges"] = [{
        "module_name": "Helper",
        "build_cs": "Helper/Helper.Build.cs",
        "private_source_roots": ["Helper/Private"],
        "header_roots": ["Helper/Public", "Helper/Private"],
    }]

    first = _resolve(tmp_path, spec)
    assert "Helper/Public/Net/HttpEnvServerBase.h" in first.headers
    assert "Helper/Public/Net/Leaf.h" in first.headers
    leaf.write_text("two", encoding="utf-8")
    assert first.identity_hash != _resolve(tmp_path, spec).identity_hash


def test_repo_local_angle_header_leaf_changes_identity(tmp_path: Path) -> None:
    """Include-root angle dependencies and their leaves participate in identity."""
    spec = _fixture(tmp_path)
    source = tmp_path / "Module/Private/Survivors/Weapons/Weapon.cpp"
    header = tmp_path / "Module/Public/Survivors/Weapons/Weapon.h"
    leaf = tmp_path / "Module/Public/Survivors/Weapons/Leaf.h"
    source.write_text(
        "#include <Survivors/Weapons/Weapon.h>", encoding="utf-8"
    )
    header.write_text(
        "#include <Survivors/Weapons/Leaf.h>", encoding="utf-8"
    )
    leaf.write_text("one", encoding="utf-8")

    first = _resolve(tmp_path, spec)
    assert first.headers == (
        "Module/Public/Survivors/Weapons/Leaf.h",
        "Module/Public/Survivors/Weapons/Weapon.h",
    )
    leaf.write_text("two", encoding="utf-8")
    assert first.identity_hash != _resolve(tmp_path, spec).identity_hash


@pytest.mark.parametrize(
    ("directive", "message"),
    [
        ("#include <MissingLocal.h>", "missing repo-local angle include"),
        ("#include SURVIVORS_HEADER", r"unsupported C\+\+ include directive"),
        (
            "#if WITH_EDITOR\n#include CONDITIONAL_HEADER\n#endif",
            r"unsupported C\+\+ include directive",
        ),
    ],
)
def test_unresolved_angle_macro_and_conditional_include_fail_closed(
    tmp_path: Path, directive: str, message: str
) -> None:
    """Unresolved non-literal include syntax cannot silently leave the closure."""
    spec = _fixture(tmp_path)
    source = tmp_path / "Module/Private/Survivors/Weapons/Weapon.cpp"
    source.write_text(directive, encoding="utf-8")

    with pytest.raises(ContractValidationError, match=message):
        _resolve(tmp_path, spec)


def test_external_angle_include_requires_exact_manifest_allowlist(
    tmp_path: Path,
) -> None:
    """External angle headers use the same exact reasoned allowlist as quotes."""
    spec = _fixture(tmp_path)
    source = tmp_path / "Module/Private/Survivors/Weapons/Weapon.cpp"
    source.write_text("#include <algorithm>", encoding="utf-8")

    with pytest.raises(
        ContractValidationError, match="missing repo-local angle include"
    ):
        _resolve(tmp_path, spec)

    closure = _resolve(
        tmp_path,
        spec,
        allowed_external_quote_includes=("algorithm",),
    )
    assert closure.headers == ()


def test_missing_ambiguous_include_fails_and_generated_header_is_excluded(tmp_path: Path) -> None:
    """repo-local include のmissing/ambiguousを拒否し、generated.hだけ除外する。"""
    spec = _fixture(tmp_path)
    source = tmp_path / "Module/Private/Survivors/Weapons/Weapon.cpp"
    source.write_text('#include "Survivors/Missing.h"', encoding="utf-8")
    with pytest.raises(ContractValidationError, match="missing repo-local quote include"):
        _resolve(tmp_path, spec)

    source.write_text('#include "MissingLocal.h"', encoding="utf-8")
    with pytest.raises(ContractValidationError, match="missing repo-local quote include"):
        _resolve(tmp_path, spec)

    source.write_text('#include "Shared.h"', encoding="utf-8")
    (tmp_path / "Module/Public/Shared.h").write_text("public", encoding="utf-8")
    (tmp_path / "Module/Private/Shared.h").write_text("private", encoding="utf-8")
    with pytest.raises(ContractValidationError, match="ambiguous"):
        _resolve(tmp_path, spec)

    source.write_text(
        '#include "Survivors/Weapons/Weapon.h"\n#include "Weapon.generated.h"',
        encoding="utf-8",
    )
    closure = _resolve(tmp_path, spec)
    assert all(not path.endswith(".generated.h") for path in closure.headers)


def test_unresolved_external_quote_include_requires_exact_allowlist(tmp_path: Path) -> None:
    """外部 quote include は manifest 由来の完全一致 allowlist だけで許可する。"""
    spec = _fixture(tmp_path)
    source = tmp_path / "Module/Private/Survivors/Weapons/Weapon.cpp"
    source.write_text('#include "CoreMinimal.h"', encoding="utf-8")

    with pytest.raises(ContractValidationError, match="missing repo-local quote include"):
        _resolve(tmp_path, spec)

    closure = _resolve(
        tmp_path,
        spec,
        allowed_external_quote_includes=("CoreMinimal.h",),
    )
    assert closure.headers == ()


def test_non_project_cache_header_does_not_create_false_ambiguity(tmp_path: Path) -> None:
    """project source外のcache copyをrepo-local include候補として扱わない。

    pytest/build cache や退避copyがcurrent C++ closureを偶発的に失効させるのを防ぐ。
    """
    spec = _fixture(tmp_path)
    cached = tmp_path / ".cache/Module/Public/Survivors/Weapons/Weapon.h"
    cached.parent.mkdir(parents=True)
    cached.write_text("stale cache", encoding="utf-8")
    (tmp_path / ".cache/Module/Module.Build.cs").write_text("stale module", encoding="utf-8")
    closure = _resolve(tmp_path, spec)
    assert closure.headers == ("Module/Public/Survivors/Weapons/Weapon.h",)
