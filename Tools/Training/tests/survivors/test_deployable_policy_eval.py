"""Deployable policy package の eligibility gate と評価 CLI を検証する。
development checkpoint が正式 package へ昇格できないことと、CLI discovery が軽量なことを確認する。
"""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path
import pytest
import torch as th
from games.survivors.deployable_policy_package import package_deployable_policy
@pytest.mark.parametrize(
    "development_only,eligible,match",
    [(True, False, "development_only"), (False, False, "formal_student_eligible")],
)
def test_release_packager_fails_closed_before_creating_output(
    tmp_path: Path, development_only: bool, eligible: bool, match: str,
) -> None:
    """二つの checkpoint eligibility sibling を package 前に別々に拒否する。
    例外後に output directory が存在しないことで部分 package も公開されないことを確かめる。
    """
    checkpoint = tmp_path / "student.pt"
    th.save(
        {
            "schema_version": "survivors.deployable_policy_checkpoint.v1",
            "development_only": development_only,
            "formal_student_eligible": eligible,
            "deploy_schema_hash": "a" * 64,
            "model_config": {"observation_dim": 3, "action_dim": 2, "hidden_dim": 4},
            "model_state_dict": {},
            "formal_dependency_identities": {},
        },
        checkpoint,
    )
    output = tmp_path / "package"
    with pytest.raises(ValueError, match=match):
        package_deployable_policy(checkpoint, output)
    assert not output.exists()
def test_eval_cli_help_exits_zero_without_loading_training_artifacts() -> None:
    """評価 CLI の --help が model/dataset 引数なしで成功する。
    package や UE5 を用意しない discovery 経路でも import error を起こさないことを subprocess で確認する。
    """
    script = Path(__file__).parents[2] / "eval_survivors_deployable_policy.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"], capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "deployable" in result.stdout.lower()
