"""teacher pairwise residual の cluster CI・fallback・weight を検証する。

精密に誤る teacher を block し、誤差上限が許容内の teacher だけへ保守的な正 weight を
付与する counterexample を含む。
"""

from __future__ import annotations

import copy

import pytest

from games.survivors.teacher_reliability import (
    ReliabilityContractError,
    fit_teacher_reliability,
    validate_reliability_calibration,
)


def _rows(*, residual: float, count: int, kind: str = "weapon") -> list[dict]:
    """指定 residual を持つ development outcome rows を作る。

    各 decision を episode/seed cluster へ束ね、short/full lineage を両方持たせる。
    """
    rows = []
    for index in range(count):
        teacher_margin = 1.0
        outcome_margin = teacher_margin - residual
        rows.append(
            {
                "decision_id": f"d-{index}",
                "episode_id": f"episode-{index // 2}",
                "seed_cluster_id": f"seed-{index // 2}",
                "partition": "development_train",
                "teacher_type": "raw_critic",
                "choice_kind": kind,
                "elapsed_seconds": 60.0,
                "teacher_score_scale_id": "a" * 64,
                "outcome_scale_ids": {
                    "short": "b" * 64,
                    "full": "c" * 64,
                },
                "pairs": [
                    {
                        "candidate_i": "best",
                        "candidate_j": "other",
                        "teacher_margin_z": teacher_margin,
                        "outcome_margin_z": {
                            "short": outcome_margin,
                            "full": outcome_margin,
                        },
                        "outcome_refs": {
                            "short": f"short-{index}",
                            "full": f"full-{index}",
                        },
                    }
                ],
            }
        )
    return rows


def test_consistently_wrong_narrow_ci_teacher_is_blocked() -> None:
    """大きな一定 residual の narrow CI teacher を weight 0 で block する。

    CI が狭いことを信頼性と誤認せず、error UCB 自体で gate する。
    """
    artifact = fit_teacher_reliability(
        _rows(residual=1.2, count=30),
        development_split_id="d" * 64,
        integration_fidelity_identity="e" * 64,
        bootstrap_seed=7,
        bootstrap_resamples=100,
    )
    exact = artifact["slices"][0]
    assert exact["fallback_level"] == "exact"
    assert exact["error_ucb_z"] >= 1.0
    assert exact["weight"] == 0.0
    assert exact["release_blocker"] is True


def test_accurate_teacher_with_wider_ci_gets_positive_ucb_weight() -> None:
    """小さく変動する residual の wider CI teacher に正 weight を与える。

    point estimate ではなく 95% upper bound 相当から weight を計算する。
    """
    rows = _rows(residual=0.2, count=30)
    for index, row in enumerate(rows):
        margin = 0.08 + 0.01 * (index % 20)
        for horizon in ("short", "full"):
            row["pairs"][0]["outcome_margin_z"][horizon] = 1.0 - margin
    artifact = fit_teacher_reliability(
        rows,
        development_split_id="d" * 64,
        integration_fidelity_identity="e" * 64,
        bootstrap_seed=9,
        bootstrap_resamples=200,
    )
    exact = artifact["slices"][0]
    assert 0.0 < exact["weight"] < 1.0
    assert exact["weight"] == pytest.approx(1.0 - exact["error_ucb_z"])
    assert exact["ci_upper_z"] >= exact["median_abs_residual_z"]
    assert exact["release_blocker"] is False


def test_underpowered_exact_uses_prefixed_fallback() -> None:
    """exact elapsed slice が不足しても kind fallback の support が十分なら採用する。

    fallback 集約は事前固定 teacher_type × choice_kind だけを使う。
    """
    rows = _rows(residual=0.2, count=60)
    for index, row in enumerate(rows):
        row["elapsed_seconds"] = (index % 4) * 360.0 + 30.0
    artifact = fit_teacher_reliability(
        rows,
        development_split_id="d" * 64,
        integration_fidelity_identity="e" * 64,
        bootstrap_resamples=100,
    )
    fallback_rows = [
        item for item in artifact["slices"] if item["fallback_level"] == "choice_kind"
    ]
    assert fallback_rows
    assert fallback_rows[0]["support_decisions"] == 60
    assert fallback_rows[0]["n_effective_clusters"] == 30


@pytest.mark.parametrize("mutation", ["final", "unknown_kind", "missing_outcome"])
def test_invalid_lineage_or_slice_is_release_blocker(mutation: str) -> None:
    """final 混入・unknown kind・outcome lineage 欠落の全経路を拒否する。

    fabricated support や global 既定値へフォールバックしない。
    """
    rows = _rows(residual=0.2, count=30)
    if mutation == "final":
        rows[0]["partition"] = "final_test"
    elif mutation == "unknown_kind":
        rows[0]["choice_kind"] = "unknown"
    else:
        rows[0]["pairs"][0]["outcome_refs"]["short"] = ""
    with pytest.raises(ReliabilityContractError):
        fit_teacher_reliability(
            rows,
            development_split_id="d" * 64,
            integration_fidelity_identity="e" * 64,
            bootstrap_resamples=20,
        )


def test_scale_identity_tampering_rejects_calibration_reuse() -> None:
    """teacher score scale identity 変更後の calibration reuse を拒否する。

    artifact 全 slice と親 scale binding を再検証する。
    """
    artifact = fit_teacher_reliability(
        _rows(residual=0.2, count=30),
        development_split_id="d" * 64,
        integration_fidelity_identity="e" * 64,
        bootstrap_resamples=20,
    )
    tampered = copy.deepcopy(artifact)
    tampered["teacher_score_scale_id"] = "f" * 64
    with pytest.raises(ReliabilityContractError, match="identity"):
        validate_reliability_calibration(tampered)
