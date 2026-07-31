"""short/full horizon の tie-aware teacher 指標と release gate を検証する。

primary rank と diagnostic utility を分離し、top-1・pairwise・NDCG@3・regret を
episode cluster 単位で評価する。
"""

from __future__ import annotations

from games.survivors.teacher_validation import (
    GATE_THRESHOLDS,
    OutcomeNormalizer,
    evaluate_teacher,
    make_label_release_verdict,
)


def _candidate(
    choice_id: str,
    teacher_score: float,
    survival: float,
    level: float,
    gems: float,
    kills: float,
) -> dict:
    """候補の teacher score と外的 outcome を作る。

    NovelD・shaped・HP penalty は provenance にだけ残して utility 入力から分離する。
    """
    outcome = {
        "stage_cleared_or_survived_horizon": survival >= 300.0,
        "survival_seconds": survival,
        "level_gain": level,
        "gem_gain": gems,
        "kill_gain": kills,
        "provenance": {
            "noveld": 99.0,
            "shaped_reward": 88.0,
            "hp_penalty": -77.0,
        },
    }
    return {
        "choice_id": choice_id,
        "teacher_score": teacher_score,
        "short": dict(outcome),
        "full": dict(outcome),
    }


def _decision(index: int, reversed_teacher: bool = False) -> dict:
    """明確な正解順位を持つ decision fixture を返す。

    teacher の向きだけを切替え、outcome と cluster identity は固定する。
    """
    candidates = [
        _candidate("best", 3.0, 300.0, 3.0, 30.0, 20.0),
        _candidate("mid", 2.0, 250.0, 2.0, 20.0, 10.0),
        _candidate("low", 1.0, 200.0, 1.0, 10.0, 5.0),
    ]
    if reversed_teacher:
        for candidate in candidates:
            candidate["teacher_score"] *= -1.0
    return {
        "decision_id": f"d-{index}",
        "episode_id": f"episode-{index // 2}",
        "seed_cluster_id": f"seed-{index // 2}",
        "choice_kind": "weapon",
        "elapsed_seconds": 60.0,
        "candidates": candidates,
    }


def test_short_and_full_metrics_are_tie_aware() -> None:
    """short/full の全必須指標を完全一致 teacher で確認する。

    episode cluster bootstrap を使っても点推定が理想値になることを検証する。
    """
    decisions = [_decision(index) for index in range(12)]
    normalizers = {
        horizon: OutcomeNormalizer.fit(decisions, horizon, partition_id="dev-train")
        for horizon in ("short", "full")
    }
    report = evaluate_teacher(decisions, normalizers, bootstrap_resamples=100)
    for horizon in ("short", "full"):
        metrics = report["horizons"][horizon]
        assert metrics["top1_agreement"] == 1.0
        assert metrics["pairwise_agreement"] == 1.0
        assert metrics["mean_ndcg_at_3"] == 1.0
        assert metrics["median_normalized_regret"] == 0.0
        assert metrics["n_effective_clusters"] == 6


def test_primary_ties_receive_tie_aware_credit() -> None:
    """primary tuple 同値の候補を teacher が tie にしても完全 credit とする。

    continuous utility の微小差も登録 tolerance 内なら順位を捏造しない。
    """
    decision = _decision(0)
    decision["candidates"][1]["teacher_score"] = 3.0
    decision["candidates"][1]["short"] = dict(decision["candidates"][0]["short"])
    decision["candidates"][1]["full"] = dict(decision["candidates"][0]["full"])
    normalizers = {
        horizon: OutcomeNormalizer.fit([decision], horizon, partition_id="dev-train")
        for horizon in ("short", "full")
    }
    report = evaluate_teacher([decision], normalizers, bootstrap_resamples=20)
    assert report["horizons"]["short"]["top1_agreement"] == 1.0
    assert report["horizons"]["short"]["pairwise_agreement"] == 1.0


def test_ground_truth_utility_excludes_training_reward_terms() -> None:
    """NovelD/shaped/HP penalty の変更で ground-truth utility が変わらない。

    training reward provenance と teacher validation target の混線を防ぐ。
    """
    first = _decision(0)
    second = _decision(0)
    for candidate in second["candidates"]:
        candidate["short"]["provenance"] = {
            "noveld": -1e9,
            "shaped_reward": 1e9,
            "hp_penalty": -1e9,
        }
    normalizer = OutcomeNormalizer.fit([first], "short", partition_id="dev-train")
    report_a = evaluate_teacher(
        [first], {"short": normalizer}, horizons=("short",), bootstrap_resamples=20
    )
    report_b = evaluate_teacher(
        [second], {"short": normalizer}, horizons=("short",), bootstrap_resamples=20
    )
    assert report_a["horizons"]["short"] == report_b["horizons"]["short"]


def test_gate_cannot_be_manually_overridden_and_binds_subject() -> None:
    """gate 未達 report の verdict を override 引数なしで FAIL に固定する。

    source descriptor hash は変更せず、verdict identity の subject として束縛する。
    """
    decisions = [_decision(index, reversed_teacher=True) for index in range(40)]
    normalizers = {
        horizon: OutcomeNormalizer.fit(decisions, horizon, partition_id="dev-train")
        for horizon in ("short", "full")
    }
    report = evaluate_teacher(decisions, normalizers, bootstrap_resamples=100)
    source_identity = "a" * 64
    verdict = make_label_release_verdict(
        report=report,
        source_descriptor_identity=source_identity,
        split_identity="b" * 64,
        reliability_identity="c" * 64,
        score_scale_identity="d" * 64,
        integration_fidelity_identity="e" * 64,
    )
    assert verdict["schema_version"] == "survivors.label_release_verdict.v1"
    assert verdict["status"] == "FAIL"
    assert verdict["subject"]["source_descriptor_identity"] == source_identity
    assert source_identity == "a" * 64
    assert GATE_THRESHOLDS["top1_agreement"] == 0.65

