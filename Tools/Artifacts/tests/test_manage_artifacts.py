import json
import subprocess
import sys
from pathlib import Path


def test_manage_artifacts_direct_script_invocation_is_not_cwd_dependent(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "Tools" / "Artifacts" / "manage_artifacts.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--store-root",
            str(tmp_path / "store"),
            "list",
        ],
        cwd=tmp_path,
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == []


def test_manage_artifacts_reports_common_install_requirement_when_missing(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "Tools" / "Artifacts" / "manage_artifacts.py"

    result = subprocess.run(
        [
            sys.executable,
            "-S",
            str(script),
            "--help",
        ],
        cwd=tmp_path,
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode != 0
    assert "pip install -e Tools/Common" in result.stderr
    assert "reinbalance_survivors_contracts" in result.stderr
