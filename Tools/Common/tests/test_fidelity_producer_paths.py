"""Producer path manifest の schema と identity binding を検証する。

13 key の exact allowlist と manifest bytes の hash 反映を固定します。
"""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from reinbalance_survivors_contracts.fidelity_producer_paths import (
    load_producer_path_manifest,
    resolve_gating_producer_hashes,
)
from reinbalance_survivors_contracts.fidelity_verdict import GATING_KEYS
from reinbalance_survivors_contracts.ubt_action_graph import make_ubt_action_graph_attestation
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
        "header_roots": ["Module/Public", "Module/Private"],
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
        producers[key]["transitive_dependency_mode"] = "compiled_module_closure"
        producers[key]["compiled_module_closure"] = _cpp_closure()
    return producers


def _external_quote_entries(manifest) -> list[dict[str, str]]:
    """検証済み token 列を loader 用の理由付き JSON entry に戻す。"""
    return [
        {"include": token, "reason": "test fixture external header"}
        for token in manifest.external_quote_includes
    ]


def _attestation(root):
    """fixture repo の current build inputs と全 .cpp から fresh attestation を作る。"""
    (root / "ReinBalance").mkdir(exist_ok=True)
    (root / "ReinBalance/ReinBalance.uproject").write_text("{}", encoding="utf-8")
    (root / "ReinBalance/Source").mkdir(exist_ok=True)
    (root / "ReinBalance/Source/ReinBalanceEditor.Target.cs").write_text("target", encoding="utf-8")
    (root / "ReinBalance/Source/Fixture").mkdir(exist_ok=True)
    (root / "ReinBalance/Source/Fixture/Fixture.Build.cs").write_text("module", encoding="utf-8")
    sources = sorted(path.relative_to(root).as_posix() for path in root.rglob("*.cpp"))
    return make_ubt_action_graph_attestation(root, sources, ubt_identity="a" * 64)


def test_packaged_manifest_has_exact_keys_and_hash() -> None:
    """package 内 manifest が13 key と identity hash を持つことを検証する。

    schema の増減は version 更新なしに通りません。
    """
    manifest = load_producer_path_manifest()
    assert set(manifest.producers) == set(GATING_KEYS)
    assert len(manifest.manifest_hash) == 64
    assert "CoreMinimal.h" in manifest.external_quote_includes


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
    data = {
        "schema_version": original.schema_version,
        "external_quote_includes": _external_quote_entries(original),
        "producers": _mutable(original.producers),
    }
    data["producers"]["unknown"] = data["producers"]["logic_public"]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ContractValidationError):
        load_producer_path_manifest(path)


@pytest.mark.parametrize(
    "entry",
    [
        {"include": "CoreMinimal.h"},
        {"include": "../Engine/CoreMinimal.h", "reason": "escape"},
        {"include": "Generated.generated.h", "reason": "generated headers are implicit"},
    ],
)
def test_external_quote_include_allowlist_is_strict(tmp_path, entry) -> None:
    """外部 include allowlist の未知形・traversal・generated header を拒否する。"""
    original = load_producer_path_manifest()
    data = {
        "schema_version": original.schema_version,
        "external_quote_includes": [entry],
        "producers": _mutable(original.producers),
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ContractValidationError):
        load_producer_path_manifest(path)


@pytest.mark.parametrize("key", ["logic_public", "logic_private", "game_facade", "http_service"])
@pytest.mark.parametrize("mode", ["none", "generated_schema", "python_import_closure"])
def test_cpp_dependency_mode_is_fixed_at_load_and_use(tmp_path, key, mode) -> None:
    """C++ producer の dependency mode を load／利用直前の両方で固定する。

    compiled TU closure を外す既知の全 mode を4つの C++ identity 経路で拒否します。
    """
    original = load_producer_path_manifest()
    producers = _mutable(original.producers)
    producers[key]["transitive_dependency_mode"] = mode
    path = tmp_path / f"{key}-{mode}.json"
    path.write_text(
        json.dumps({
            "schema_version": original.schema_version,
            "external_quote_includes": _external_quote_entries(original),
            "producers": producers,
        }),
        encoding="utf-8",
    )
    with pytest.raises(ContractValidationError):
        load_producer_path_manifest(path)

    forged = replace(original, producers=producers)
    with pytest.raises(ContractValidationError):
        resolve_gating_producer_hashes(tmp_path, forged, {})


def test_compiled_closure_bytes_are_bound_to_gating_hash(tmp_path) -> None:
    """compiled closure の TU 追加・変更・削除を gating hash へ接続する。

    exact path が不変でも Private 配下の weapon 実装差で stale 判定できることを固定します。
    """
    build = tmp_path / "Module/Module.Build.cs"
    private = tmp_path / "Module/Private/Survivors/Weapons"
    private.mkdir(parents=True)
    (tmp_path / "Module/Public").mkdir()
    build.write_text("module", encoding="utf-8")
    weapon = private / "Weapon.cpp"
    weapon.write_text("one", encoding="utf-8")
    producers = _empty_producers()
    manifest = type(load_producer_path_manifest())("v", producers, "f" * 64)
    with pytest.raises(ContractValidationError, match="action graph"):
        resolve_gating_producer_hashes(tmp_path, manifest, {})
    first = resolve_gating_producer_hashes(tmp_path, manifest, {}, _attestation(tmp_path))
    weapon.write_text("two", encoding="utf-8")
    second = resolve_gating_producer_hashes(tmp_path, manifest, {}, _attestation(tmp_path))
    assert first["logic_private"] != second["logic_private"]
    (private / "NewWeapon.cpp").write_text("new", encoding="utf-8")
    third = resolve_gating_producer_hashes(tmp_path, manifest, {}, _attestation(tmp_path))
    assert second["logic_private"] != third["logic_private"]
    weapon.unlink()
    fourth = resolve_gating_producer_hashes(tmp_path, manifest, {}, _attestation(tmp_path))
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
    first = resolve_gating_producer_hashes(tmp_path, manifest, {}, _attestation(tmp_path))
    header.write_text("two", encoding="utf-8")
    second = resolve_gating_producer_hashes(tmp_path, manifest, {}, _attestation(tmp_path))
    assert first[key] != second[key]
    (root / "Added.h").write_text("added", encoding="utf-8")
    third = resolve_gating_producer_hashes(tmp_path, manifest, {}, _attestation(tmp_path))
    assert second[key] != third[key]
    header.unlink()
    fourth = resolve_gating_producer_hashes(tmp_path, manifest, {}, _attestation(tmp_path))
    assert third[key] != fourth[key]


def test_deploy_obs_v2_sources_are_bound_to_deploy_gating_hashes(tmp_path) -> None:
    """DeployObs v2 の schema・ビルダー・yaml・sim adapter を1 byte 変えると deploy 系 gating hash が変わる。

    packaged manifest の deploy_obs_schema / deploy_release_adapter entry をそのまま使い、
    実ファイルの内容を一時 repo へ写して stale verdict を検出できることを固定します。
    Training の deploy_raw_env.py / deploy_obs_wrapper.py（03-07）は release adapter 側だけに入ります。
    """
    repo = Path(__file__).resolve().parents[3]
    packaged = load_producer_path_manifest()
    common = "Tools/Common/src/reinbalance_survivors_contracts/"
    schema_paths = [f"{common}deploy_obs.py", f"{common}schemas/deploy_obs_v2.yaml", f"{common}deploy_obs_v2_features.py", f"{common}schemas/deploy_obs_v2_features.yaml"]
    adapter_paths = schema_paths[2:] + ["Tools/Training/games/survivors/deploy_raw_env.py", "Tools/Training/games/survivors/deploy_obs_wrapper.py"]
    assert list(packaged.producers["deploy_obs_schema"]["ordered_exact_paths"]) == schema_paths
    assert list(packaged.producers["deploy_release_adapter"]["ordered_exact_paths"]) == adapter_paths
    (tmp_path / "Module/Private").mkdir(parents=True)
    (tmp_path / "Module/Public").mkdir()
    (tmp_path / "Module/Module.Build.cs").write_text("module", encoding="utf-8")
    (tmp_path / "Module/Private/Weapon.cpp").write_text("one", encoding="utf-8")
    all_paths = schema_paths + adapter_paths[2:]
    for relative in all_paths:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((repo / relative).read_bytes())
    producers = _empty_producers()
    for key in ("deploy_obs_schema", "deploy_release_adapter"):
        producers[key] = _mutable(packaged.producers[key])
    manifest = type(packaged)("v", producers, "f" * 64)
    generated = {"deploy_obs_schema": {"v": 1}, "deploy_release_adapter": {"v": 1}}
    previous = resolve_gating_producer_hashes(tmp_path, manifest, generated, _attestation(tmp_path))
    for relative in all_paths:
        target = tmp_path / relative
        target.write_bytes(target.read_bytes() + b" ")
        current = resolve_gating_producer_hashes(tmp_path, manifest, generated, _attestation(tmp_path))
        schema_changed = current["deploy_obs_schema"] != previous["deploy_obs_schema"]
        assert schema_changed == (relative in schema_paths), relative
        adapter_changed = current["deploy_release_adapter"] != previous["deploy_release_adapter"]
        assert adapter_changed == (relative in adapter_paths), relative
        previous = current
