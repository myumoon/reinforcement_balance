"""Current fidelity resolver が current checkout のproducerだけをauthorityにすることを検証する。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reinbalance_survivors_contracts.current_fidelity import (
    GENERATED_INPUT_DESCRIPTOR_SCHEMA_VERSION,
    resolve_current_gating_producer_hashes,
)
from reinbalance_survivors_contracts.fidelity_producer_paths import load_producer_path_manifest
from reinbalance_survivors_contracts.ubt_action_graph import (
    make_ubt_action_graph_attestation,
    write_ubt_action_graph_attestation,
)
from reinbalance_survivors_contracts.ui_intent import ContractValidationError


_LOCAL_DEPENDENCIES = {
    "ReinBalanceLogic": [],
    "ReinBalance": ["ReinBalanceLogic"],
    "ReinBalanceEditor": ["ReinBalance", "ReinBalanceLogic", "PythonTrainingComm"],
    "PythonTrainingComm": [],
}


def _write(path: Path, text: str = "fixture") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(text, encoding="utf-8")


def _materialize_packaged_manifest_repo(root: Path) -> tuple[Path, Path]:
    """packaged manifestを変更せず解決できる最小current checkoutを構築する。"""
    manifest = load_producer_path_manifest()
    _write(root / "ReinBalance/ReinBalance.uproject", "{}")
    _write(root / "ReinBalance/Source/ReinBalanceEditor.Target.cs", "target")
    _write(root / "ReinBalance/Plugins/Fixture/Fixture.uplugin", "{}")

    closure_specs: dict[str, dict] = {}
    for entry in manifest.producers.values():
        closure = entry.get("compiled_module_closure")
        if closure is None:
            continue
        closure_specs[closure["module_name"]] = closure
        for edge in closure["repo_local_module_dependency_edges"]:
            closure_specs[edge["module_name"]] = edge
    for module_name, spec in closure_specs.items():
        dependencies = _LOCAL_DEPENDENCIES[module_name]
        declaration = (
            "PublicDependencyModuleNames.AddRange(new string[] { "
            + ", ".join(f'\"{name}\"' for name in dependencies)
            + " });"
        )
        _write(root / spec["build_cs"], declaration)
        for private_root in spec["private_source_roots"]:
            _write(root / private_root / f"{module_name}Fixture.cpp")
        for header_root in spec["header_roots"]:
            (root / header_root).mkdir(parents=True, exist_ok=True)

    for entry in manifest.producers.values():
        for relative in entry["ordered_exact_paths"]:
            _write(root / relative)
        for recursive in entry["recursive_roots"]:
            (root / recursive["path"]).mkdir(parents=True, exist_ok=True)

    generated_names = {
        name
        for entry in manifest.producers.values()
        for name in entry["generated_inputs"]
        if name != "target_build_attestation"
    }
    descriptor_inputs = {}
    for name in sorted(generated_names):
        source = root / "FidelityInputs" / f"{name}.json"
        _write(source, json.dumps({"name": name, "value": 1}))
        descriptor_inputs[name] = {
            "path": source.relative_to(root).as_posix(),
            "format": "json",
        }
    descriptor = root / "FidelityInputs/generated_inputs.json"
    descriptor.write_text(
        json.dumps(
            {
                "schema_version": GENERATED_INPUT_DESCRIPTOR_SCHEMA_VERSION,
                "inputs": descriptor_inputs,
            }
        ),
        encoding="utf-8",
    )

    compiled_sources = sorted(path.relative_to(root).as_posix() for path in root.rglob("*.cpp"))
    attestation = make_ubt_action_graph_attestation(
        root,
        compiled_sources,
        ubt_identity="a" * 64,
    )
    attestation_path = root / "FidelityInputs/ubt_action_graph.json"
    write_ubt_action_graph_attestation(attestation_path, attestation)
    return descriptor, attestation_path


def test_current_context_uses_packaged_manifest_descriptor_and_fresh_attestation(tmp_path: Path) -> None:
    """current repo bytes・source identities・action graph identityを一体で返す。"""
    descriptor, attestation = _materialize_packaged_manifest_repo(tmp_path)
    context = resolve_current_gating_producer_hashes(tmp_path, descriptor, attestation)

    assert context.manifest_hash == load_producer_path_manifest().manifest_hash
    assert len(context.action_graph_identity) == 64
    assert set(context.generated_input_source_identities) == {
        "content_schema",
        "action_time_schema",
        "external_decision_schema",
        "preview_schema",
        "deploy_obs_schema",
        "deploy_release_adapter",
        "target_profile",
        "target_build_attestation",
    }
    assert len(context.current_gating_producer_hashes) == 13


def test_generated_input_change_requires_fresh_current_context(tmp_path: Path) -> None:
    """descriptorが指すschema bytesの変更をcurrent gating mapへ反映する。"""
    descriptor, attestation = _materialize_packaged_manifest_repo(tmp_path)
    first = resolve_current_gating_producer_hashes(tmp_path, descriptor, attestation)
    (tmp_path / "FidelityInputs/content_schema.json").write_text(
        json.dumps({"name": "content_schema", "value": 2}), encoding="utf-8"
    )
    second = resolve_current_gating_producer_hashes(tmp_path, descriptor, attestation)
    assert first.current_gating_producer_hashes["content_schema"] != second.current_gating_producer_hashes["content_schema"]


@pytest.mark.parametrize("mutation", ["missing", "unknown", "escape"])
def test_descriptor_missing_unknown_and_repo_escape_are_rejected(tmp_path: Path, mutation: str) -> None:
    """strict descriptorの欠落・未知key・repo外pathをfail closedにする。"""
    descriptor, attestation = _materialize_packaged_manifest_repo(tmp_path)
    wire = json.loads(descriptor.read_text(encoding="utf-8"))
    if mutation == "missing":
        wire["inputs"].pop("content_schema")
    elif mutation == "unknown":
        wire["inputs"]["unknown"] = {"path": "FidelityInputs/content_schema.json", "format": "json"}
    else:
        wire["inputs"]["content_schema"]["path"] = "../outside.json"
    descriptor.write_text(json.dumps(wire), encoding="utf-8")
    with pytest.raises(ContractValidationError):
        resolve_current_gating_producer_hashes(tmp_path, descriptor, attestation)


def test_stale_action_graph_is_rejected_before_gating_hashes_are_returned(tmp_path: Path) -> None:
    """attestation後のBuild.cs変更をcurrent producer mapとして返さない。"""
    descriptor, attestation = _materialize_packaged_manifest_repo(tmp_path)
    build_file = tmp_path / "ReinBalance/Source/ReinBalanceLogic/ReinBalanceLogic.Build.cs"
    build_file.write_text(build_file.read_text(encoding="utf-8") + "\n// stale", encoding="utf-8")
    with pytest.raises(ContractValidationError, match="build input"):
        resolve_current_gating_producer_hashes(tmp_path, descriptor, attestation)
