"""Survivors value source audit CLI の終了コード契約を検証する。

初心者向け:
自動化から ready・not ready・入力不正を区別できるよう、3種類の終了コードを固定します。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_TRAINING_ROOT = Path(__file__).resolve().parents[2]
if str(_TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAINING_ROOT))

_TEST_SURVIVORS_DIR = Path(__file__).resolve().parent
if str(_TEST_SURVIVORS_DIR) not in sys.path:
    sys.path.insert(0, str(_TEST_SURVIVORS_DIR))

from survivors_value_source_audit import main

from test_value_source_descriptor import (
    _obs_schema,
    _write_inputs,
)


def _write_audit_metadata(
    run_dir: Path,
    completion: dict,
    provenance: dict,
) -> Path:
    """CLI が読む log metadata と observation schema を保存する。

    初心者向け:
    audit は訓練プロセスに依存せず、run 内の固定済み入力だけを再検査します。
    """
    (run_dir / "log" / "value_source_completion.json").write_text(
        json.dumps(completion), encoding="utf-8"
    )
    (run_dir / "log" / "value_source_provenance.json").write_text(
        json.dumps(provenance), encoding="utf-8"
    )
    (run_dir / "log" / "run_meta.json").write_text(
        json.dumps({"git_commit": "a" * 40}), encoding="utf-8"
    )
    obs_path = run_dir / "result" / "obs_schema.json"
    obs_path.write_text(json.dumps(_obs_schema()), encoding="utf-8")
    return obs_path


def test_audit_returns_zero_and_publishes_only_for_ready_source(tmp_path: Path) -> None:
    """ready run を audit すると descriptor を atomic publish して 0 を返す。

    初心者向け:
    後工程は終了コード 0 と result の immutable descriptor の両方を確認できます。
    """
    run_dir, completion, provenance = _write_inputs(tmp_path)
    obs_path = _write_audit_metadata(run_dir, completion, provenance)
    result = main(
        [
            "--run-dir",
            str(run_dir),
            "--obs-schema-json",
            str(obs_path),
            "--created-at-utc",
            "2026-07-29T00:00:00Z",
        ]
    )
    assert result == 0
    assert (run_dir / "result" / "value_source_descriptor.json").is_file()


def test_audit_returns_two_for_not_ready_and_three_for_invalid(tmp_path: Path) -> None:
    """not ready と invalid を別終了コードに分離する。

    初心者向け:
    artifact 欠落は再実行可能な gate 不通過、JSON 不正は入力修復が必要なエラーです。
    """
    run_dir, completion, provenance = _write_inputs(tmp_path / "not-ready")
    run_dir.joinpath("result/vecnormalize.pkl").unlink()
    obs_path = _write_audit_metadata(run_dir, completion, provenance)
    assert main(
        [
            "--run-dir",
            str(run_dir),
            "--obs-schema-json",
            str(obs_path),
            "--created-at-utc",
            "2026-07-29T00:00:00Z",
        ]
    ) == 2

    invalid_run, invalid_completion, invalid_provenance = _write_inputs(
        tmp_path / "invalid"
    )
    invalid_obs = _write_audit_metadata(
        invalid_run, invalid_completion, invalid_provenance
    )
    invalid_obs.write_text("{", encoding="utf-8")
    assert main(
        [
            "--run-dir",
            str(invalid_run),
            "--obs-schema-json",
            str(invalid_obs),
            "--created-at-utc",
            "2026-07-29T00:00:00Z",
        ]
    ) == 3
