"""episode 単位の freeze split と final-test sealing を検証する。

development train/validation と untouched final test の lineage を交差させず、
calibration fit・method selection から final episode を fail-closed で隔離する。
"""

from __future__ import annotations

import copy

import pytest

from games.survivors.teacher_validation_split import (
    SplitContractError,
    assert_calibration_lineage,
    freeze_episode_split,
    validate_frozen_split,
)


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

