"""Value source loader の artifact と policy binding を検証する。

正式 scorer が model・VecNormalize・descriptor のどれか一つでも別内容ならロードを
中止し、保存 policy class から algorithm を決定することを確認する。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_TRAINING_ROOT = Path(__file__).resolve().parents[2]
if str(_TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAINING_ROOT))

from games.survivors.value_source_loader import (
    ValueSourceLoadError,
    load_value_source,
)
from value_scorer_fixtures import build_saved_value_source


def test_loader_binds_saved_policy_and_eval_vecnormalize(tmp_path: Path) -> None:
    """保存 policy class と manifest 設定が一致する source をロードする。

    VecNormalize は評価専用に固定し、reward 統計を更新しない設定へ必ず上書きする。
    """
    manifest_path, _, _ = build_saved_value_source(tmp_path, recurrent=True)
    source = load_value_source(manifest_path)

    assert source.algorithm == "RecurrentPPO"
    assert source.vecnormalize.training is False
    assert source.vecnormalize.norm_reward is False
    assert source.observation_dim == 3
    assert source.policy_state_schema["shared_lstm"] is False
    assert source.policy_state_schema["enable_critic_lstm"] is True


@pytest.mark.parametrize(
    "artifact_name",
    ["model", "vecnormalize", "package_freeze"],
)
def test_loader_recomputes_all_runtime_artifact_hashes(
    tmp_path: Path,
    artifact_name: str,
) -> None:
    """model と VecNormalize の両 artifact substitution を拒否する。

    descriptor に保存された hash を信用せず、ロード直前の bytes から再計算する。
    """
    manifest_path, _, _ = build_saved_value_source(tmp_path, recurrent=False)
    descriptor = json.loads(manifest_path.read_text(encoding="utf-8"))
    relative = descriptor["artifacts"][artifact_name]["path_relative"]
    manifest_path.parents[1].joinpath(relative).write_bytes(b"substituted")

    with pytest.raises(ValueSourceLoadError, match=f"{artifact_name}.*hash"):
        load_value_source(manifest_path)


def test_loader_rejects_missing_vecnormalize_and_manifest_hash_mismatch(
    tmp_path: Path,
) -> None:
    """VecNormalize 欠落と immutable manifest 改変を両方 fail-closed にする。

    formal ranking が raw observation を未正規化で評価したり、改変 descriptor を使わない。
    """
    missing_path, _, _ = build_saved_value_source(
        tmp_path / "missing",
        recurrent=False,
    )
    descriptor = json.loads(missing_path.read_text(encoding="utf-8"))
    missing_path.parents[1].joinpath(
        descriptor["artifacts"]["vecnormalize"]["path_relative"]
    ).unlink()
    with pytest.raises(ValueSourceLoadError, match="vecnormalize"):
        load_value_source(missing_path)

    mismatch_path, _, _ = build_saved_value_source(
        tmp_path / "manifest",
        recurrent=False,
    )
    mismatch = json.loads(mismatch_path.read_text(encoding="utf-8"))
    mismatch["model_spec"]["algorithm"] = "RecurrentPPO"
    mismatch_path.write_text(json.dumps(mismatch), encoding="utf-8")
    with pytest.raises(ValueSourceLoadError, match="descriptor|identity"):
        load_value_source(mismatch_path)
