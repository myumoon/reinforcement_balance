"""実 UBT GenerateClangDatabase から canonical Survivors action graph を発行する。"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from reinbalance_survivors_contracts.canonical_json import sha256_hex
from reinbalance_survivors_contracts.ubt_action_graph import (
    make_ubt_action_graph_attestation,
    write_ubt_action_graph_attestation,
)


class ExportError(RuntimeError):
    """UBT command、compile database、output contract の失敗。"""


def _repository_root() -> Path:
    """script の checkout だけをcurrent repositoryとして返す。"""
    return Path(__file__).resolve().parents[2]


def _engine_paths(engine_root: Path) -> tuple[Path, Path]:
    root = Path(engine_root).resolve()
    build = root / "Engine/Build/BatchFiles/Build.bat"
    ubt = root / "Engine/Binaries/DotNET/UnrealBuildTool/UnrealBuildTool.dll"
    if not build.is_file() or not ubt.is_file():
        raise ExportError(f"UE5.4 engine root is incomplete: {root}")
    return build, ubt


def _resolve_engine_root(explicit: Path | None) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    configured = os.environ.get("UE_ROOT")
    if configured:
        candidates.append(Path(configured))
    candidates.extend(
        [
            Path(r"C:\UnrealEngine\UE_5.4"),
            Path(r"C:\Program Files\Epic Games\UE_5.4"),
        ]
    )
    for candidate in candidates:
        try:
            _engine_paths(candidate)
        except ExportError:
            continue
        return candidate.resolve()
    raise ExportError("UE5.4 engine root was not found; pass --engine-root or set UE_ROOT")


def _compiled_sources(repo_root: Path, database_path: Path) -> list[str]:
    try:
        database = json.loads(database_path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExportError(f"cannot read compile_commands.json: {exc}") from exc
    if not isinstance(database, list):
        raise ExportError("compile_commands.json must be an array")
    sources: set[str] = set()
    for index, entry in enumerate(database):
        if not isinstance(entry, dict) or not isinstance(entry.get("file"), str):
            raise ExportError(f"compile_commands entry {index} is invalid")
        source = Path(entry["file"])
        if not source.is_absolute():
            directory = entry.get("directory")
            if not isinstance(directory, str) or not directory:
                raise ExportError(f"compile_commands entry {index} has no directory")
            source = Path(directory) / source
        resolved = source.resolve()
        try:
            relative = resolved.relative_to(repo_root).as_posix()
        except ValueError:
            continue
        if resolved.suffix.lower() != ".cpp":
            continue
        if not resolved.is_file():
            raise ExportError(f"compiled source is missing: {relative}")
        sources.add(relative)
    if not sources:
        raise ExportError("UBT compile database contains no repo-local .cpp sources")
    return sorted(sources)


def export_ubt_action_graph(
    repo_root: Path,
    engine_root: Path,
    output_path: Path,
    *,
    command_runner: Callable[..., Any] = subprocess.run,
) -> None:
    """fixed target の実 UBT database を検証し、成功時だけatomic publishする。"""
    repo = Path(repo_root).resolve()
    engine = Path(engine_root).resolve()
    project = repo / "ReinBalance/ReinBalance.uproject"
    if not project.is_file():
        raise ExportError(f"not a ReinBalance checkout: {repo}")
    build, ubt = _engine_paths(engine)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="reinbalance-ubt-", dir=destination.parent) as temporary:
        output_dir = Path(temporary)
        command = [
            str(build),
            "-Mode=GenerateClangDatabase",
            "ReinBalanceEditor",
            "Win64",
            "Development",
            f"-Project={project}",
            "-NoExecCodeGenActions",
            f"-OutputDir={output_dir}",
        ]
        try:
            result = command_runner(
                command,
                cwd=repo,
                text=True,
                capture_output=True,
                check=False,
            )
        except OSError as exc:
            raise ExportError(f"could not launch UBT GenerateClangDatabase: {exc}") from exc
        if not isinstance(getattr(result, "returncode", None), int) or result.returncode != 0:
            stdout = str(getattr(result, "stdout", "")).strip()
            stderr = str(getattr(result, "stderr", "")).strip()
            diagnostic = "\n".join(part for part in (stdout, stderr) if part)
            raise ExportError(f"UBT GenerateClangDatabase failed with exit {getattr(result, 'returncode', '?')}: {diagnostic}")
        sources = _compiled_sources(repo, output_dir / "compile_commands.json")
        attestation = make_ubt_action_graph_attestation(
            repo,
            sources,
            ubt_identity=sha256_hex(ubt.read_bytes()),
        )
        write_ubt_action_graph_attestation(destination, attestation)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export the fixed Survivors UBT action graph attestation.")
    parser.add_argument("--engine-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        export_ubt_action_graph(
            _repository_root(),
            _resolve_engine_root(args.engine_root),
            args.output,
        )
    except ExportError as exc:
        print(f"UBT action graph export failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
