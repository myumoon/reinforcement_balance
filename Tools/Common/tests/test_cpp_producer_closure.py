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
    private.mkdir(parents=True)
    (root / "Module/Module.Build.cs").write_text("module", encoding="utf-8")
    (private / "Weapon.cpp").write_text("one", encoding="utf-8")
    return {"module_name": "Module", "build_cs": "Module/Module.Build.cs", "private_source_roots": ["Module/Private"], "compiled_tu_include_glob": "Module/Private/**/*.cpp", "repo_local_module_dependency_edges": [], "allowed_non_behavior_excludes": []}


def test_cpp_byte_change_and_new_tu_change_identity(tmp_path) -> None:
    """既存 weapon byte 変更と新 TU 追加で identity が変わる。

    header-only identity では見逃す implementation 差を検出します。
    """
    spec = _fixture(tmp_path)
    first = resolve_cpp_producer_closure(tmp_path, spec)
    source = tmp_path / "Module/Private/Survivors/Weapons/Weapon.cpp"
    source.write_text("two", encoding="utf-8")
    second = resolve_cpp_producer_closure(tmp_path, spec)
    assert first.identity_hash != second.identity_hash
    (source.parent / "NewWeapon.cpp").write_text("new", encoding="utf-8")
    third = resolve_cpp_producer_closure(tmp_path, spec)
    assert second.identity_hash != third.identity_hash


def test_action_graph_mismatch_and_behavior_exclude_are_rejected(tmp_path) -> None:
    """action graph/manifest の片側 TU と behavior exclude を拒否する。

    compiled source の未分類や隠蔽を fail-closed にします。
    """
    spec = _fixture(tmp_path)
    with pytest.raises(ContractValidationError):
        resolve_cpp_producer_closure(tmp_path, spec, action_graph_sources=["only/action.cpp"])
    spec["allowed_non_behavior_excludes"] = [{"path": "Module/Private/Survivors/Weapons/Weapon.cpp", "reason": "bad"}]
    with pytest.raises(ContractValidationError):
        resolve_cpp_producer_closure(tmp_path, spec)


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
        resolve_cpp_producer_closure(tmp_path, spec)

    spec["repo_local_module_dependency_edges"] = [{
        "module_name": "Helper",
        "build_cs": "Helper/Helper.Build.cs",
        "private_source_roots": ["Helper/Private"],
    }]
    closure = resolve_cpp_producer_closure(tmp_path, spec)
    assert any(source.path == "Helper/Private/Behavior.cpp" for source in closure.sources)
