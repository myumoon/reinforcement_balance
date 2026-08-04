"""The formal commands published in the training guide remain executable argv."""

from __future__ import annotations

import re
from pathlib import Path

from collect_survivors_value_choices import _parser as collection_parser
from validate_survivors_value_teacher import _build_parser as teacher_parser


_OVERVIEW = Path(__file__).resolve().parents[4] / "docs/train/overview.md"


def _documented_argv(script_name: str) -> list[str]:
    """Extract one continued Python command and return the argv after its script."""
    markdown = _OVERVIEW.read_text(encoding="utf-8")
    blocks = re.findall(r"```(?:bash|powershell)\s*\n(.*?)```", markdown, re.DOTALL)
    block = next(value for value in blocks if f"python {script_name}" in value)
    command_lines: list[str] = []
    collecting = False
    for raw_line in block.splitlines():
        line = raw_line.strip()
        if line.startswith(f"python {script_name}"):
            collecting = True
        if not collecting:
            continue
        continued = line.endswith(("\\", "`"))
        command_lines.append(line[:-1].rstrip() if continued else line)
        if not continued:
            break
    words = " ".join(command_lines).split()
    assert words[:2] == ["python", script_name]
    return words[2:]


def test_documented_collector_command_parses() -> None:
    """Renaming a required collector option must invalidate this operational guide."""
    args = collection_parser().parse_args(
        _documented_argv("collect_survivors_value_choices.py")
    )
    assert args.generated_input_descriptor.name == "generated-inputs.json"
    assert args.ubt_action_graph.name == "ubt-action-graph.json"


def test_documented_teacher_validator_command_parses() -> None:
    """Teacher validation docs must carry both current-authority inputs."""
    args = teacher_parser().parse_args(
        _documented_argv("validate_survivors_value_teacher.py")
    )
    assert args.generated_input_descriptor.name == "generated-inputs.json"
    assert args.ubt_action_graph.name == "ubt-action-graph.json"
