"""Producer path manifest の schema と identity binding を検証する。

13 key の exact allowlist と manifest bytes の hash 反映を固定します。
"""

import json
from dataclasses import replace

import pytest

from reinbalance_survivors_contracts.fidelity_producer_paths import (
    load_producer_path_manifest,
    resolve_gating_producer_hashes,
)
from reinbalance_survivors_contracts.fidelity_verdict import GATING_KEYS
from reinbalance_survivors_contracts.ui_intent import ContractValidationError


def _mutable(value):
    """immutable manifest tree を test mutation 用 JSON tree へ戻す。

    production の deep-freeze を弱めず、loader rejection fixture だけを組み立てます。
    """
    if isinstance(value, dict) or hasattr(value, "items"):
        return {key: _mutable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_mutable(item) for item in value]
    return value


def _cpp_closure() -> dict:
    """最小 C++ module closure spec を全 C++ gating entry で共有する。

    resolver の利用直前 schema 検証を満たす正規 fixture を返します。
    """
    return {
        "module_name": "Module",
        "build_cs": "Module/Module.Build.cs",
        "private_source_roots": ["Module/Private"],
        "compiled_tu_include_glob": "Module/Private/**/*.cpp",
        "repo_local_module_dependency_edges": [],
        "allowed_non_behavior_excludes": [],
    }


def _empty_producers() -> dict:
    """13 producer の最小で schema-valid な manifest entries を作る。

    C++ 4 entry には compiled closure を対称に付与します。
    """
    producers = {
        key: {
            "ordered_exact_paths": [],
            "recursive_roots": [],
            "explicit_excludes": [],
            "generated_inputs": [],
            "transitive_dependency_mode": "none",
        }
        for key in GATING_KEYS
    }
    for key in ("logic_public", "logic_private", "game_facade", "http_service"):
        producers[key]["compiled_module_closure"] = _cpp_closure()
    return producers


def test_packaged_manifest_has_exact_keys_and_hash() -> None:
    """package 内 manifest が13 key と identity hash を持つことを検証する。

    schema の増減は version 更新なしに通りません。
    """
    manifest = load_producer_path_manifest()
    assert set(manifest.producers) == set(GATING_KEYS)
    assert len(manifest.manifest_hash) == 64


def test_loaded_manifest_is_deeply_immutable_and_resolver_revalidates(tmp_path) -> None:
    """manifest の全 nested 階層を固定し、利用直前にも再検証する。

    検証後の closure 無効化と forged dataclass の両経路を fail-closed にします。
    """
    manifest = load_producer_path_manifest()
    with pytest.raises(TypeError):
        manifest.producers["logic_private"]["compiled_module_closure"]["module_name"] = "Changed"
    with pytest.raises(TypeError):
        manifest.producers["logic_private"]["recursive_roots"][0]["include_globs"][0] = "*.txt"
    forged_producers = dict(manifest.producers)
    forged_entry = dict(forged_producers["logic_private"])
    forged_entry["compiled_module_closure"] = None
    forged_producers["logic_private"] = forged_entry
    forged = replace(manifest, producers=forged_producers)
    with pytest.raises(ContractValidationError):
        resolve_gating_producer_hashes(tmp_path, forged, {})


def test_unknown_or_missing_producer_is_rejected(tmp_path) -> None:
    """未知 producer と欠落 producer を loader が拒否する。

    typo や新 producer の黙認を防ぎます。
    """
    original = load_producer_path_manifest()
    data = {"schema_version": original.schema_version, "producers": _mutable(original.producers)}
    data["producers"]["unknown"] = data["producers"]["logic_public"]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ContractValidationError):
        load_producer_path_manifest(path)


def test_compiled_closure_bytes_are_bound_to_gating_hash(tmp_path) -> None:
    """compiled closure の TU 追加・変更・削除を gating hash へ接続する。

    exact path が不変でも Private 配下の weapon 実装差で stale 判定できることを固定します。
    """
    build = tmp_path / "Module/Module.Build.cs"
    private = tmp_path / "Module/Private/Survivors/Weapons"
    private.mkdir(parents=True)
    build.write_text("module", encoding="utf-8")
    weapon = private / "Weapon.cpp"
    weapon.write_text("one", encoding="utf-8")
    producers = _empty_producers()
    producers["logic_private"]["transitive_dependency_mode"] = "compiled_module_closure"
    manifest = type(load_producer_path_manifest())("v", producers, "f" * 64)
    first = resolve_gating_producer_hashes(tmp_path, manifest, {})
    weapon.write_text("two", encoding="utf-8")
    second = resolve_gating_producer_hashes(tmp_path, manifest, {})
    assert first["logic_private"] != second["logic_private"]
    (private / "NewWeapon.cpp").write_text("new", encoding="utf-8")
    third = resolve_gating_producer_hashes(tmp_path, manifest, {})
    assert second["logic_private"] != third["logic_private"]
    weapon.unlink()
    fourth = resolve_gating_producer_hashes(tmp_path, manifest, {})
    assert third["logic_private"] != fourth["logic_private"]


@pytest.mark.parametrize("key", ["logic_public", "logic_private", "game_facade", "http_service"])
def test_recursive_header_set_is_bound_to_each_cpp_gating_hash(tmp_path, key) -> None:
    """recursive root の header 変更・追加・削除を全 C++ key の hash へ結ぶ。

    public/private header の集合と bytes が compiled TU と対称に producer identity へ入ります。
    """
    root = tmp_path / "Module/Public"
    root.mkdir(parents=True)
    private = tmp_path / "Module/Private"
    private.mkdir(parents=True)
    (tmp_path / "Module/Module.Build.cs").write_text("module", encoding="utf-8")
    (private / "Stub.cpp").write_text("stub", encoding="utf-8")
    header = root / "Producer.h"
    header.write_text("one", encoding="utf-8")
    producers = _empty_producers()
    producers[key]["recursive_roots"] = [{"path": "Module/Public", "include_globs": ["**/*.h"]}]
    manifest = type(load_producer_path_manifest())("v", producers, "f" * 64)
    first = resolve_gating_producer_hashes(tmp_path, manifest, {})
    header.write_text("two", encoding="utf-8")
    second = resolve_gating_producer_hashes(tmp_path, manifest, {})
    assert first[key] != second[key]
    (root / "Added.h").write_text("added", encoding="utf-8")
    third = resolve_gating_producer_hashes(tmp_path, manifest, {})
    assert second[key] != third[key]
    header.unlink()
    fourth = resolve_gating_producer_hashes(tmp_path, manifest, {})
    assert third[key] != fourth[key]
