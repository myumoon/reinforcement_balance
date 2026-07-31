"""paired rollout の semantic replay と RNG seam 契約を検証する。

候補の実行順や worker 数に依存しない stream 割当と、異常 branch の quarantine を
決定的な fake record だけで確認する。
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from games.survivors.choice_branch_rollout import (
    BRANCH_RNG_SCHEMA_VERSION,
    BranchRolloutError,
    build_horizon_outcome,
    build_branch_assignments,
    derive_replication_key,
    effective_replication_count,
    quarantine_branch_records,
    validate_complete_semantic_trace,
    validate_replay_observation,
    validate_rng_seam,
)


def _trace() -> dict:
    """完全 semantic trace の最小 fixture を返す。

    replay に必要な reset・task・移動・choice acknowledgement を全て明示する。
    """
    return {
        "source_identity": "a" * 64,
        "episode_logical_id": "episode-1",
        "decision_id": "decision-4",
        "reset_options": {"seed": 7},
        "task": {"cell": "weapon"},
        "curriculum": {"phase": 2},
        "movement_actions": [0, 2, 8],
        "prior_choices": [{"decision_id": "decision-1", "choice_id": "choice-0"}],
        "prior_acks": [{"decision_id": "decision-1", "choice_id": "choice-0"}],
        "base_obs_sha256": "b" * 64,
        "choice_ids": ["choice-a", "choice-b", "choice-c"],
    }


def _record(
    *,
    candidate: str,
    replication_index: int,
    stream_sha256: str,
    outcome: float,
    status: str = "ok",
) -> dict:
    """RNG seam 判定用の branch record を組み立てる。

    replication key は candidate を含めず、同じ r で common random numbers を共有する。
    """
    trace = _trace()
    return {
        "decision_id": trace["decision_id"],
        "candidate_id": candidate,
        "replication_index": replication_index,
        "replication_key": derive_replication_key(
            trace["source_identity"],
            trace["episode_logical_id"],
            trace["decision_id"],
            replication_index,
        ),
        "stream_sha256": stream_sha256,
        "draw_trace_sha256": f"{replication_index + 1:064x}",
        "outcome_utility": outcome,
        "status": status,
        "expected_branch_count": 2,
        "policy_state_before_sha256": "c" * 64,
    }


def test_replication_mapping_is_candidate_and_worker_order_independent() -> None:
    """candidate permutation と worker 数を変えても割当を同一に保つ。

    replication key と stream identity の mapping を集合比較し、execution index を除外する。
    """
    first = build_branch_assignments(
        _trace(),
        ["choice-a", "choice-b", "choice-c"],
        3,
        worker_count=1,
    )
    second = build_branch_assignments(
        _trace(),
        ["choice-c", "choice-a", "choice-b"],
        3,
        worker_count=7,
    )
    project = lambda rows: {
        (
            row.candidate_id,
            row.replication_index,
            row.replication_key,
            row.stream_sha256,
        )
        for row in rows
    }
    assert project(first) == project(second)
    for replication_index in range(3):
        same_r = [
            row for row in first if row.replication_index == replication_index
        ]
        assert len({row.replication_key for row in same_r}) == 1
        assert len({row.stream_sha256 for row in same_r}) == 1
    assert len({row.replication_key for row in first}) == 3
    assert BRANCH_RNG_SCHEMA_VERSION == "survivors.branch_rng.v1"


@pytest.mark.parametrize(
    "missing",
    [
        "source_identity",
        "episode_logical_id",
        "decision_id",
        "reset_options",
        "task",
        "curriculum",
        "movement_actions",
        "prior_choices",
        "prior_acks",
        "base_obs_sha256",
        "choice_ids",
    ],
)
def test_incomplete_semantic_trace_is_rejected(missing: str) -> None:
    """complete replay の兄弟 field 全てを fail-closed で検査する。

    一つでも欠落した prefix は同じ decision を再構築できないため score 対象外とする。
    """
    trace = _trace()
    del trace[missing]
    with pytest.raises(BranchRolloutError, match=missing):
        validate_complete_semantic_trace(trace)


def test_duplicate_choice_ack_is_rejected() -> None:
    """同一 decision の choice 重複と acknowledgement 不一致を拒否する。

    exactly-once choice の破壊を replay mismatch として早期に検出する。
    """
    trace = _trace()
    trace["prior_choices"].append(copy.deepcopy(trace["prior_choices"][0]))
    with pytest.raises(BranchRolloutError, match="duplicate-choice"):
        validate_complete_semantic_trace(trace)
    trace = _trace()
    trace["prior_acks"][0]["choice_id"] = "choice-other"
    with pytest.raises(BranchRolloutError, match="replay-mismatch"):
        validate_complete_semantic_trace(trace)


def test_replayed_decision_identity_mismatch_is_rejected() -> None:
    """complete trace でも再構築された base observation 不一致を拒否する。

    semantic field の存在確認だけで replay 成功と誤認しない。
    """
    trace = _trace()
    replayed = {
        field: trace[field]
        for field in (
            "source_identity",
            "episode_logical_id",
            "decision_id",
            "base_obs_sha256",
            "choice_ids",
        )
    }
    replayed["base_obs_sha256"] = "f" * 64
    with pytest.raises(BranchRolloutError, match="replay-mismatch"):
        validate_replay_observation(trace, replayed)

@pytest.mark.parametrize(
    ("status", "reason"),
    [
        ("replay_mismatch", "replay-mismatch"),
        ("early_death_before_decision", "early-death-before-decision"),
        ("duplicate_choice", "duplicate-choice"),
        ("timeout", "timeout"),
        ("branch_count_missing", "branch-count-missing"),
        ("recurrent_state_leakage", "recurrent-state-leakage"),
    ],
)
def test_invalid_branch_statuses_are_quarantined(status: str, reason: str) -> None:
    """score 禁止の全 branch status を quarantine へ送る。

    異常 outcome を欠損補完せず、理由付きで有効標本から除外する。
    """
    records = [
        _record(
            candidate="a",
            replication_index=0,
            stream_sha256="1" * 64,
            outcome=0.4,
            status=status,
        )
    ]
    accepted, quarantined = quarantine_branch_records(records)
    assert accepted == []
    assert quarantined[0]["quarantine_reason"] == reason


def test_same_stream_duplicate_is_not_effective_replication() -> None:
    """same stream の duplicate replay を effective sample に数えない。

    candidate ごとの行数ではなく unique replication key/stream pair を数える。
    """
    records = [
        _record(
            candidate=candidate,
            replication_index=0,
            stream_sha256="1" * 64,
            outcome=0.2,
        )
        for candidate in ("a", "b")
    ]
    duplicate = copy.deepcopy(records)
    assert effective_replication_count(records + duplicate) == 1


def test_missing_candidate_branch_is_detected_from_expected_count() -> None:
    """status 自己申告が ok でも candidate branch 欠落を quarantine する。

    worker timeout で一行消えた反例を expected branch count から検出する。
    """
    accepted, quarantined = quarantine_branch_records(
        [
            _record(
                candidate="a",
                replication_index=0,
                stream_sha256="1" * 64,
                outcome=0.2,
            )
        ]
    )
    assert accepted == []
    assert quarantined[0]["quarantine_reason"] == "branch-count-missing"


def test_zero_outcome_variance_with_distinct_streams_passes_rng_seam() -> None:
    """distinct stream で outcomes 全同一でも RNG seam を PASS にする。

    outcome variance は統計診断へ残し、stream identity の成立条件には使わない。
    """
    records = []
    for replication_index, stream in enumerate(("1" * 64, "2" * 64)):
        for candidate in ("a", "b"):
            records.append(
                _record(
                    candidate=candidate,
                    replication_index=replication_index,
                    stream_sha256=stream,
                    outcome=0.5,
                )
            )
    verdict = validate_rng_seam(records)
    assert verdict["passed"] is True
    assert verdict["zero_outcome_variance"] is True


def test_same_stream_with_thread_noise_fails_rng_seam() -> None:
    """異なる r が same stream を共有し outcome だけ揺れる反例を拒否する。

    thread noise を独立 replication と誤認しないため stream collision を blocking にする。
    """
    records = []
    for replication_index, outcome in enumerate((0.2, 0.8)):
        for candidate in ("a", "b"):
            records.append(
                _record(
                    candidate=candidate,
                    replication_index=replication_index,
                    stream_sha256="1" * 64,
                    outcome=outcome,
                )
            )
    verdict = validate_rng_seam(records)
    assert verdict["passed"] is False
    assert "stream-collision" in verdict["blocking_reasons"]


def test_recurrent_state_leakage_fails_candidate_crn_group() -> None:
    """同じ r の candidates が別 recurrent input state を使う漏洩を拒否する。

    candidate 実行順で hidden state が進む実装を effective branch として数えない。
    """
    records = [
        _record(
            candidate=candidate,
            replication_index=0,
            stream_sha256="1" * 64,
            outcome=0.5,
        )
        for candidate in ("a", "b")
    ]
    records[1]["policy_state_before_sha256"] = "d" * 64
    verdict = validate_rng_seam(records)
    assert verdict["passed"] is False
    assert "recurrent-state-leakage" in verdict["blocking_reasons"]


def test_failed_horizon_outcome_cannot_carry_scoreable_utility() -> None:
    """failed branch と observed/censored outcome を schema status で区別する。

    failed row に gameplay gains を残して utility 計算へ混入させる反例を拒否する。
    """
    with pytest.raises(BranchRolloutError, match="scoreable utility"):
        build_horizon_outcome(
            decision_id="decision",
            candidate_id="choice-a",
            horizon="short",
            replication_key="a" * 64,
            stream_sha256="b" * 64,
            policy_rng_mode="deterministic",
            status="failed",
            failure_reason="timeout",
            censored=False,
            terminal=False,
            stage_cleared_or_survived_horizon=False,
            survival_seconds=1.0,
            level_gain=0.0,
            gem_gain=0.0,
            kill_gain=0.0,
            provenance={"noveld": 10.0},
        )


def test_post_decision_early_death_is_an_observed_low_outcome() -> None:
    """candidate 適用後の early death を quarantine せず observed outcome にする。

    replay prefix 中の死亡と区別し、短い survival を teacher の反例として score する。
    """
    outcome = build_horizon_outcome(
        decision_id="decision",
        candidate_id="choice-a",
        horizon="short",
        replication_key="a" * 64,
        stream_sha256="b" * 64,
        policy_rng_mode="deterministic",
        status="observed",
        failure_reason=None,
        censored=False,
        terminal=True,
        stage_cleared_or_survived_horizon=False,
        survival_seconds=3.0,
        level_gain=0.0,
        gem_gain=0.0,
        kill_gain=0.0,
        provenance={"termination": "early_death"},
    )
    assert outcome["status"] == "observed"
    assert outcome["terminal"] is True


def test_survivors_step_info_exposes_read_only_outcome_metrics() -> None:
    """step info の六 metrics と既存 reward sum を同時に固定する。

    metrics 追加が base/shaped/HP penalty の reward semantics を書き換えないことを確認する。
    """
    source = (
        Path(__file__).resolve().parents[2]
        / "games"
        / "survivors"
        / "survivors_env.py"
    ).read_text(encoding="utf-8")
    for field in (
        '"elapsed"',
        '"level"',
        '"gems"',
        '"kills"',
        '"alive"',
        '"stage_clear"',
    ):
        assert field in source
    assert "base_reward + shaped + hp_penalty" in source
