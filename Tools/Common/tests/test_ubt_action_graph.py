"""Fixed-subject UBT action graph attestation の厳格性を検証する。"""

from __future__ import annotations

from pathlib import Path

import pytest

from reinbalance_survivors_contracts.canonical_json import canonical_hash
from reinbalance_survivors_contracts.ubt_action_graph import (
    UbtActionGraphAttestation,
    make_ubt_action_graph_attestation,
    read_ubt_action_graph_attestation,
    write_ubt_action_graph_attestation,
)
from reinbalance_survivors_contracts.ui_intent import ContractValidationError


def _repo(root: Path) -> Path:
    """最小の project/build inputs/compiled source を作る。"""
    (root / "ReinBalance/Source/Game/Private").mkdir(parents=True)
    (root / "ReinBalance/Plugins/Fixture").mkdir(parents=True)
    (root / "ReinBalance/ReinBalance.uproject").write_text("{}", encoding="utf-8")
    (root / "ReinBalance/Source/Game/Game.Build.cs").write_text("module", encoding="utf-8")
    (root / "ReinBalance/Source/ReinBalanceEditor.Target.cs").write_text("target", encoding="utf-8")
    (root / "ReinBalance/Plugins/Fixture/Fixture.uplugin").write_text("{}", encoding="utf-8")
    (root / "ReinBalance/Source/Game/Private/Game.cpp").write_text("source", encoding="utf-8")
    return root


def _rehash(wire: dict) -> dict:
    payload = {key: value for key, value in wire.items() if key != "identity_sha256"}
    wire["identity_sha256"] = canonical_hash(payload)
    return wire


def test_attestation_round_trip_binds_fixed_subject_build_inputs_and_identity(tmp_path: Path) -> None:
    """固定 target と current build inputs を canonical identity へ結ぶ。"""
    repo = _repo(tmp_path)
    attestation = make_ubt_action_graph_attestation(
        repo,
        ["ReinBalance/Source/Game/Private/Game.cpp"],
        ubt_identity="a" * 64,
    )
    output = tmp_path / "attestation.json"
    write_ubt_action_graph_attestation(output, attestation)

    loaded = read_ubt_action_graph_attestation(output, repo_root=repo)
    assert loaded.subject == ("ReinBalanceEditor", "Win64", "Development")
    assert loaded.project_file == "ReinBalance/ReinBalance.uproject"
    assert set(loaded.build_input_hashes) == {
        "ReinBalance/ReinBalance.uproject",
        "ReinBalance/Plugins/Fixture/Fixture.uplugin",
        "ReinBalance/Source/Game/Game.Build.cs",
        "ReinBalance/Source/ReinBalanceEditor.Target.cs",
    }
    assert loaded.compiled_sources == ("ReinBalance/Source/Game/Private/Game.cpp",)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("target", "OtherEditor"),
        ("platform", "Linux"),
        ("configuration", "Shipping"),
        ("project_file", "Other/Other.uproject"),
    ],
)
def test_noncanonical_subject_is_rejected(tmp_path: Path, field: str, value: str) -> None:
    """別 target/platform/config/project を current fidelity に流用させない。"""
    repo = _repo(tmp_path)
    wire = make_ubt_action_graph_attestation(
        repo,
        ["ReinBalance/Source/Game/Private/Game.cpp"],
        ubt_identity="a" * 64,
    ).to_wire()
    wire[field] = value
    _rehash(wire)
    with pytest.raises(ContractValidationError, match=field):
        UbtActionGraphAttestation.from_wire(wire)


def test_stale_missing_and_extra_build_inputs_fail_current_validation(tmp_path: Path) -> None:
    """attestation 後の build input 差と missing/extra key を全て拒否する。"""
    repo = _repo(tmp_path)
    original = make_ubt_action_graph_attestation(
        repo,
        ["ReinBalance/Source/Game/Private/Game.cpp"],
        ubt_identity="a" * 64,
    )
    (repo / "ReinBalance/Source/Game/Game.Build.cs").write_text("changed", encoding="utf-8")
    with pytest.raises(ContractValidationError, match="build input"):
        original.validate_current(repo)
    (repo / "ReinBalance/Source/Game/Game.Build.cs").write_text("module", encoding="utf-8")
    original.validate_current(repo)

    for mutation in ("missing", "extra"):
        wire = original.to_wire()
        if mutation == "missing":
            wire["build_input_hashes"].pop("ReinBalance/Source/ReinBalanceEditor.Target.cs")
        else:
            wire["build_input_hashes"]["ReinBalance/Source/Extra.Build.cs"] = "b" * 64
        candidate = UbtActionGraphAttestation.from_wire(_rehash(wire))
        with pytest.raises(ContractValidationError, match="build input"):
            candidate.validate_current(repo)


def test_identity_tamper_or_missing_compiled_source_is_rejected(tmp_path: Path) -> None:
    """canonical identity 改変と存在しない action source をfail closedにする。"""
    repo = _repo(tmp_path)
    attestation = make_ubt_action_graph_attestation(
        repo,
        ["ReinBalance/Source/Game/Private/Game.cpp"],
        ubt_identity="a" * 64,
    )
    wire = attestation.to_wire()
    wire["identity_sha256"] = "f" * 64
    with pytest.raises(ContractValidationError, match="identity"):
        UbtActionGraphAttestation.from_wire(wire)

    (repo / "ReinBalance/Source/Game/Private/Game.cpp").unlink()
    with pytest.raises(ContractValidationError, match="compiled source"):
        attestation.validate_current(repo)


def test_non_project_cache_build_files_do_not_change_current_identity(tmp_path: Path) -> None:
    """固定project境界外のBuild/Target/project/plugin copyをbuild inputに数えない。

    cacheや退避copyの存在でfresh UBT attestationが偶発的にstale化するのを防ぐ。
    """
    repo = _repo(tmp_path)
    attestation = make_ubt_action_graph_attestation(
        repo,
        ["ReinBalance/Source/Game/Private/Game.cpp"],
        ubt_identity="a" * 64,
    )
    cache = repo / "ReinBalance/.pytest-cache"
    (cache / "Source/Game").mkdir(parents=True)
    (cache / "Plugins/Fixture").mkdir(parents=True)
    (cache / "Source/Game/Game.Build.cs").write_text("cached", encoding="utf-8")
    (cache / "Source/Cached.Target.cs").write_text("cached", encoding="utf-8")
    (cache / "Cached.uproject").write_text("{}", encoding="utf-8")
    (cache / "Plugins/Fixture/Fixture.uplugin").write_text("{}", encoding="utf-8")
    attestation.validate_current(repo)
