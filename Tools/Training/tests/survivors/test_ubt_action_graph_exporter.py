"""UBT exporter の command/output failure boundary を挙動で検証する。"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from reinbalance_survivors_contracts.ubt_action_graph import read_ubt_action_graph_attestation
from export_survivors_ubt_action_graph import ExportError, export_ubt_action_graph


def _repo(root: Path) -> tuple[Path, Path]:
    repo = root / "repo"
    engine = root / "engine"
    (repo / "ReinBalance/Source/Game/Private").mkdir(parents=True)
    (repo / "ReinBalance/ReinBalance.uproject").write_text("{}", encoding="utf-8")
    (repo / "ReinBalance/Source/Game/Game.Build.cs").write_text("module", encoding="utf-8")
    (repo / "ReinBalance/Source/ReinBalanceEditor.Target.cs").write_text("target", encoding="utf-8")
    (repo / "ReinBalance/Source/Game/Private/Game.cpp").write_text("source", encoding="utf-8")
    (engine / "Engine/Build/BatchFiles").mkdir(parents=True)
    (engine / "Engine/Binaries/DotNET/UnrealBuildTool").mkdir(parents=True)
    (engine / "Engine/Build/BatchFiles/Build.bat").write_text("fake", encoding="utf-8")
    (engine / "Engine/Binaries/DotNET/UnrealBuildTool/UnrealBuildTool.dll").write_text("ubt", encoding="utf-8")
    return repo, engine


def test_exporter_runs_fixed_generate_clang_database_and_publishes_repo_sources(tmp_path: Path) -> None:
    """実UBT boundaryの出力をrepo-local `.cpp` attestationへ正規化する。"""
    repo, engine = _repo(tmp_path)
    observed: list[str] = []

    def fake_run(command, **kwargs):
        observed.extend(str(part) for part in command)
        output_arg = next(part for part in command if str(part).startswith("-OutputDir="))
        output_dir = Path(str(output_arg).split("=", 1)[1])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "compile_commands.json").write_text(
            json.dumps(
                [
                    {"file": str(repo / "ReinBalance/Source/Game/Private/Game.cpp"), "command": "clang", "directory": str(repo), "output": "Game.obj"},
                    {"file": str(engine / "Engine/Source/External.cpp"), "command": "clang", "directory": str(engine), "output": "External.obj"},
                ]
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, "ok", "")

    output = tmp_path / "action-graph.json"
    export_ubt_action_graph(repo, engine, output, command_runner=fake_run)
    attestation = read_ubt_action_graph_attestation(output, repo_root=repo)

    assert attestation.compiled_sources == ("ReinBalance/Source/Game/Private/Game.cpp",)
    assert "-Mode=GenerateClangDatabase" in observed
    assert "ReinBalanceEditor" in observed
    assert "Win64" in observed
    assert "Development" in observed
    assert "-NoExecCodeGenActions" in observed


@pytest.mark.parametrize("mode", ["command_failure", "empty_sources"])
def test_exporter_failure_never_publishes_attestation(tmp_path: Path, mode: str) -> None:
    """UBT failure/empty graphではpartialまたはstale outputを発行しない。"""
    repo, engine = _repo(tmp_path)
    output = tmp_path / "action-graph.json"

    def fake_run(command, **kwargs):
        if mode == "command_failure":
            return subprocess.CompletedProcess(command, 6, "shared log permission denied", "")
        output_arg = next(part for part in command if str(part).startswith("-OutputDir="))
        output_dir = Path(str(output_arg).split("=", 1)[1])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "compile_commands.json").write_text("[]", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "ok", "")

    expected = "shared log permission denied" if mode == "command_failure" else "no repo-local"
    with pytest.raises(ExportError, match=expected):
        export_ubt_action_graph(repo, engine, output, command_runner=fake_run)
    assert not output.exists()
