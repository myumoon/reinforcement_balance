"""episode 単位の freeze split と final-test sealing を検証する。

development train/validation と untouched final test の lineage を交差させず、
calibration fit・method selection から final episode を fail-closed で隔離する。
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import validate_survivors_value_teacher as validation_cli
from reinbalance_survivors_contracts.fidelity_verdict import FidelityMetric, FidelityVerdict, GATING_KEYS
from games.survivors.teacher_validation_split import (
    SplitContractError,
    assert_calibration_lineage,
    commit_frozen_episode_split,
    freeze_episode_split,
    validate_frozen_split,
)
from value_scorer_fixtures import build_saved_value_source


def test_episode_split_is_deterministic_disjoint_and_frozen() -> None:
    """入力順を変えても episode assignment と split identity を固定する。

    同じ episode の decision は必ず同じ partition に所属する。
    """
    episodes = [f"episode-{index:03d}" for index in range(60)]
    first = freeze_episode_split(episodes, seed="phase6")
    second = freeze_episode_split(list(reversed(episodes)), seed="phase6")
    assert first == second
    validate_frozen_split(first)
    partitions = first["episode_partitions"]
    assert set(partitions) == set(episodes)
    assert set(partitions.values()) == {
        "development_train",
        "development_validation",
        "final_test",
    }


def test_frozen_split_commit_is_create_once(tmp_path: Path) -> None:
    """byte-identical split の再利用だけを許し、別 seed の上書きを拒否する。

    拒否後も初回 artifact bytes と final_test assignment が変わらないことを確認する。
    """
    episodes = [f"episode-{index:03d}" for index in range(60)]
    destination = tmp_path / "teacher_validation_split.json"
    first = freeze_episode_split(episodes, seed="phase6-a")
    changed = freeze_episode_split(episodes, seed="phase6-b")
    commit_frozen_episode_split(destination, first)
    initial_bytes = destination.read_bytes()
    commit_frozen_episode_split(destination, copy.deepcopy(first))
    assert destination.read_bytes() == initial_bytes
    with pytest.raises(SplitContractError, match="cannot be overwritten"):
        commit_frozen_episode_split(destination, changed)
    assert destination.read_bytes() == initial_bytes
    stored = json.loads(destination.read_text(encoding="utf-8"))
    assert {
        episode
        for episode, partition in stored["episode_partitions"].items()
        if partition == "final_test"
    } == {
        episode
        for episode, partition in first["episode_partitions"].items()
        if partition == "final_test"
    }


def test_cli_split_rewrite_fails_before_child_artifact_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """同じ output directory への別 seed CLI 再実行を split commit で停止する。

    初回の split・scale・reliability・report・verdict bytes を一件も変更させない。
    """
    episodes = [f"episode-{index:03d}" for index in range(60)]
    output_dir = tmp_path / "artifacts"

    def fake_pipeline(**kwargs) -> dict:
        """CLI persistence 順序だけを検証する最小 artifacts を返す。

        split seed 以外の計算を省き、create-once failure が子 artifact 書込み前かを観測する。
        """
        split = freeze_episode_split(episodes, seed=kwargs["split_seed"])
        return {
            "split": split,
            "score_scale": {"identity": kwargs["split_seed"]},
            "reliability": {"identity": kwargs["split_seed"]},
            "report": {"identity": kwargs["split_seed"]},
            "verdict": {
                "identity": kwargs["split_seed"],
                "status": "FAIL",
            },
        }

    monkeypatch.setattr(validation_cli, "_load_json", lambda *_args: {})
    monkeypatch.setattr(
        validation_cli,
        "resolve_current_gating_producer_hashes",
        lambda *_args: type("Context", (), {"current_gating_producer_hashes": {}})(),
    )
    monkeypatch.setattr(
        validation_cli,
        "run_validation_pipeline",
        fake_pipeline,
    )
    monkeypatch.setattr(
        validation_cli,
        "commit_teacher_score_scale",
        validation_cli._write_canonical,
    )
    common_args = [
        "--corpus",
        str(tmp_path / "corpus.json"),
        "--source-descriptor",
        str(tmp_path / "source.json"),
        "--integration-fidelity-verdict",
        str(tmp_path / "fidelity.json"),
        "--generated-input-descriptor",
        str(tmp_path / "generated-inputs.json"),
        "--ubt-action-graph",
        str(tmp_path / "ubt-action-graph.json"),
        "--output-dir",
        str(output_dir),
    ]
    assert validation_cli.main(common_args + ["--split-seed", "seed-a"]) == 2
    initial = {
        path.name: path.read_bytes()
        for path in sorted(output_dir.iterdir())
    }
    with pytest.raises(SplitContractError, match="cannot be overwritten"):
        validation_cli.main(common_args + ["--split-seed", "seed-b"])
    assert {
        path.name: path.read_bytes()
        for path in sorted(output_dir.iterdir())
    } == initial


def test_pipeline_rejects_old_verdict_and_corpus_map_when_current_resolver_differs(tmp_path: Path) -> None:
    """corpus自己申告とverdictが一致しても明示current差をrelease前に拒否する。

    current checkout から解決した map だけを authority とし、旧 map 二点セットの再利用を防ぐ。
    """
    manifest_path, _, _ = build_saved_value_source(tmp_path / "source", recurrent=True)
    source_descriptor = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_identity = source_descriptor["identity_sha256"]
    old = {key: "a" * 64 for key in GATING_KEYS}
    current = dict(old)
    current["logic_private"] = "f" * 64
    verdict = FidelityVerdict(
        "integration",
        {
            "target_profile_hash": "1" * 64,
            "target_build_attestation_hash": "2" * 64,
            "report_scope": "exact_target",
            "producer_allowlist_version": "fidelity_producer_paths.v1",
            "producer_manifest_hash": "3" * 64,
            "resolved_producers": {key: [] for key in GATING_KEYS},
        },
        (FidelityMetric("deploy_obs_visibility", 0.01, "normalized_error", True, None, True),),
        (),
        {
            "git_commit": "fixture",
            "workspace_dirty_summary": "",
            "audit_tool_version": "fixture",
            "dependency_versions": {},
            "operator": "pytest",
            "timestamp": "2026-08-01T00:00:00Z",
        },
        old,
    )
    corpus = {
        "schema_version": "survivors.paired_rollout_corpus.v1",
        "source_identity": source_identity,
        "teacher_type": "raw_critic",
        "teacher_identity": "b" * 64,
        "current_gating_producer_hashes": old,
    }

    with pytest.raises(validation_cli.TeacherPipelineError, match="current"):
        validation_cli.run_validation_pipeline(
            corpus=corpus,
            source_descriptor=source_descriptor,
            integration_fidelity_verdict=verdict.to_wire(),
            current_gating_producer_hashes=current,
        )


def test_tampered_assignment_or_identity_is_rejected() -> None:
    """freeze 後の episode 移動と artifact identity 改変を拒否する。

    final 開封後の report tuning による再分割を防止する。
    """
    split = freeze_episode_split(
        [f"episode-{index:03d}" for index in range(30)], seed="phase6"
    )
    tampered = copy.deepcopy(split)
    episode = next(iter(tampered["episode_partitions"]))
    tampered["episode_partitions"][episode] = "final_test"
    with pytest.raises(SplitContractError, match="identity"):
        validate_frozen_split(tampered)


def test_final_decision_or_outcome_in_calibration_fails() -> None:
    """final episode に属する decision/outcome ref の fit 使用を即時 FAIL にする。

    decision lineage と outcome lineage の両経路を対称に検査する。
    """
    split = freeze_episode_split(
        [f"episode-{index:03d}" for index in range(30)], seed="phase6"
    )
    final_episode = next(
        episode
        for episode, partition in split["episode_partitions"].items()
        if partition == "final_test"
    )
    with pytest.raises(SplitContractError, match="final_test"):
        assert_calibration_lineage(
            split,
            decision_refs=[
                {
                    "decision_id": "d-final",
                    "episode_id": final_episode,
                    "partition": "final_test",
                }
            ],
            outcome_refs=[],
        )
    with pytest.raises(SplitContractError, match="final_test"):
        assert_calibration_lineage(
            split,
            decision_refs=[],
            outcome_refs=[
                {
                    "outcome_ref": "o-final",
                    "episode_id": final_episode,
                    "partition": "final_test",
                }
            ],
        )


def test_development_lineage_passes_and_records_final_exclusions() -> None:
    """development train/validation refs だけの calibration lineage を許可する。

    final episode 一覧を明示 exclusion として返し、親 artifact へ保存できるようにする。
    """
    split = freeze_episode_split(
        [f"episode-{index:03d}" for index in range(30)], seed="phase6"
    )
    development = [
        (episode, partition)
        for episode, partition in split["episode_partitions"].items()
        if partition != "final_test"
    ][:2]
    lineage = assert_calibration_lineage(
        split,
        decision_refs=[
            {
                "decision_id": f"d-{index}",
                "episode_id": episode,
                "partition": partition,
            }
            for index, (episode, partition) in enumerate(development)
        ],
        outcome_refs=[
            {
                "outcome_ref": f"o-{index}",
                "episode_id": episode,
                "partition": partition,
            }
            for index, (episode, partition) in enumerate(development)
        ],
    )
    assert lineage["split_identity"] == split["split_identity"]
    assert lineage["excluded_final_episode_ids"]
