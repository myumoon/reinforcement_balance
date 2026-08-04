"""ItemSelector dataset の split sealing、soft target、release gate を検証する。

episode/prefix 単位の固定 split と既存 teacher artifact への束縛を counterexample 付きで
確認し、test partition を開かずに derived dataset を構築する。
"""

from __future__ import annotations

import copy
import json

import pytest

from games.survivors.item_selector_dataset import (
    DatasetReleaseError,
    ItemSelectorDatasetBuilder,
    LearningCurvePolicy,
    SplitManifest,
    SplitSealedError,
    TeacherArtifactError,
    TeacherSoftTargetBuilder,
)
from games.survivors.teacher_reliability import fit_teacher_reliability
from games.survivors.teacher_score_scale import fit_teacher_score_scale


def _split_rows(count: int = 20) -> list[dict]:
    """20 episode group と各 group の複数 decision を作る。

    同一 episode の行は必ず同じ prefix group を共有する。
    """
    return [
        {
            "decision_id": f"d-{episode}-{decision}",
            "episode_id": f"episode-{episode}",
            "prefix_group_id": f"prefix-{episode}",
            "near_duplicate_key": f"trace-{episode}",
        }
        for episode in range(count)
        for decision in range(2)
    ]


def _score_scale() -> dict:
    """01-06 形式の検証済み score scale artifact を作る。

    sigma は dataset 側では fit せず既存 helper の出力を親とする。
    """
    return fit_teacher_score_scale(
        teacher_type="raw_critic",
        teacher_identity="a" * 64,
        development_fit_partition_id="b" * 64,
        score_difference_refs=[
            {
                "ref_id": f"diff-{index}",
                "partition": "development_train",
                "difference": float(index),
            }
            for index in range(21)
        ],
    )


def _calibration(scale: dict, *, residual: float = 0.2) -> dict:
    """01-07 形式の十分な support を持つ reliability artifact を作る。

    outcome refs と episode/seed cluster は各 decision から追跡できる値にする。
    """
    rows = []
    for index in range(30):
        rows.append(
            {
                "decision_id": f"cal-{index}",
                "episode_id": f"cal-episode-{index // 2}",
                "seed_cluster_id": f"seed-{index // 2}",
                "partition": "development_train",
                "teacher_type": "raw_critic",
                "teacher_identity": "a" * 64,
                "choice_kind": "weapon",
                "elapsed_seconds": 60.0,
                "teacher_score_scale_id": scale["scale_identity"],
                "outcome_scale_ids": {"short": "c" * 64, "full": "d" * 64},
                "pairs": [
                    {
                        "candidate_i": "wand",
                        "candidate_j": "knife",
                        "teacher_margin_z": 1.0,
                        "outcome_margin_z": {
                            "short": 1.0 - residual,
                            "full": 1.0 - residual,
                        },
                        "outcome_refs": {
                            "short": f"short-{index}",
                            "full": f"full-{index}",
                        },
                    }
                ],
            }
        )
    return fit_teacher_reliability(
        rows,
        development_split_id="e" * 64,
        integration_fidelity_identity="f" * 64,
        bootstrap_seed=7,
        bootstrap_resamples=20,
    )


def test_split_freeze_is_grouped_70_15_15_and_create_once(tmp_path) -> None:
    """episode prefix group を 70/15/15 へ一度だけ割り当てる。

    decision 行を増減しても同じ group の partition は変わらず、異なる manifest 上書きは拒否する。
    """
    rows = _split_rows()
    manifest = SplitManifest.freeze(rows, seed="selector-v1")
    counts = manifest.partition_group_counts
    assert counts == {"train": 14, "validation": 3, "test": 3}
    assert all(manifest.split_for(row) == manifest.split_for({**row, "decision_id": "retry"}) for row in rows)
    path = tmp_path / "split.json"
    manifest.commit(path)
    manifest.commit(path)
    changed = SplitManifest.freeze(rows, seed="other-seed")
    with pytest.raises(SplitSealedError, match="overwrite"):
        changed.commit(path)


def test_episode_prefix_binding_is_enforced_across_all_split_paths() -> None:
    """frozen prefix へ別 episode を差し替える再 grouping を全経路で拒否する。

    split_for、development/partition/test reader は同じ manifest membership を共有する。
    """
    rows = _split_rows()
    manifest = SplitManifest.freeze(rows, seed="selector-v1")
    original = rows[0]
    substituted = {**original, "episode_id": rows[-1]["episode_id"]}
    with pytest.raises(SplitSealedError, match="membership"):
        manifest.split_for(substituted)
    for reader in (
        lambda: manifest.read_partition_rows(
            [substituted], "train", purpose="training"
        ),
        lambda: manifest.read_development_rows(
            [substituted], purpose="feature_ablation"
        ),
        lambda: manifest.read_test_rows(
            [substituted], purpose="sealed_evaluation"
        ),
    ):
        with pytest.raises(SplitSealedError, match="membership"):
            reader()

    regrouped = copy.deepcopy(rows)
    regrouped[1]["prefix_group_id"] = "different-prefix"
    regrouped[1]["near_duplicate_key"] = "different-trace"
    with pytest.raises(SplitSealedError, match="episode overlap"):
        SplitManifest.freeze(regrouped, seed="selector-v1")


def test_test_partition_read_is_sealed_from_development_callers() -> None:
    """test 行を汎用 partition reader と ablation purpose から読めなくする。

    test 専用 API でも feature selection 系 purpose は拒否し、監査記録を残さない。
    """
    rows = _split_rows()
    manifest = SplitManifest.freeze(rows, seed="selector-v1")
    with pytest.raises(SplitSealedError, match="test"):
        manifest.read_partition_rows(rows, "test", purpose="ablation")
    with pytest.raises(SplitSealedError, match="ablation"):
        manifest.read_test_rows(rows, purpose="ablation")
    development = manifest.read_development_rows(rows, purpose="learning_curve")
    assert development
    assert all(manifest.split_for(row) != "test" for row in development)


def test_select_temperature_requires_manifest_and_rejects_test_rows() -> None:
    """select_temperature が frozen manifest 経由で development 行のみを受理する。

    caller-declared split='validation' で偽装した test 行を渡しても拒否し、
    ablation purpose の test read も allowlist にない purpose は全て拒否する。
    """
    scale = _score_scale()
    calibration = _calibration(scale)
    builder = TeacherSoftTargetBuilder(scale, calibration)
    rows = _split_rows()
    manifest = SplitManifest.freeze(rows, seed="selector-v1")

    test_row = next(r for r in rows if manifest.split_for(r) == "test")
    forged = {
        **test_row,
        "split": "validation",
        "candidate_item_ids": ("wand", "knife"),
        "teacher_scores": (2.0, 0.0),
        "card_mask": (True, True),
        "observed_best_item_id": "wand",
    }
    with pytest.raises((TeacherArtifactError, SplitSealedError)):
        builder.select_temperature(
            [forged], slice_key="raw_critic|weapon|0-5m", split_manifest=manifest
        )

    for bad_purpose in ("ablation", "temperature_selection", "feature_selection", "arbitrary"):
        with pytest.raises(SplitSealedError, match="allowlist"):
            manifest.read_test_rows(rows, purpose=bad_purpose)

    test_rows = manifest.read_test_rows(rows, purpose="sealed_evaluation")
    assert test_rows
    assert all(manifest.split_for(r) == "test" for r in test_rows)


def test_overlap_and_near_duplicate_cross_partition_are_rejected() -> None:
    """同一 trace fingerprint と near duplicate の split 越境を検出する。

    episode ID が異なっていても同じ prefix 内容なら独立 sample とみなさない。
    """
    rows = _split_rows()
    rows[-1]["near_duplicate_key"] = rows[0]["near_duplicate_key"]
    with pytest.raises(SplitSealedError, match="near-duplicate"):
        SplitManifest.freeze(rows, seed="selector-v1")


def test_soft_target_ties_masking_probability_and_permutation() -> None:
    """tie epsilon、masked softmax、item ID 復元を exact に検証する。

    near tie は同一 group とし、mask 候補は常に確率 0、valid 確率和は 1 にする。
    """
    scale = _score_scale()
    calibration = _calibration(scale)
    builder = TeacherSoftTargetBuilder(scale, calibration)
    sigma = scale["sigma"]
    result = builder.build(
        candidate_item_ids=("wand", "knife", "axe", "masked"),
        teacher_scores=(2.0, 2.0 - sigma * 0.019, 1.0, None),
        card_mask=(True, True, True, False),
        slice_key="raw_critic|weapon|0-5m",
        temperature=1.0,
    )
    by_id = result.probability_by_item_id
    assert by_id["wand"] == by_id["knife"]
    assert by_id["masked"] == 0.0
    assert sum(result.probabilities) == 1.0
    assert result.tie_groups[0] == ("wand", "knife")
    permuted = builder.build(
        candidate_item_ids=("axe", "wand", "masked", "knife"),
        teacher_scores=(1.0, 2.0, None, 2.0 - sigma * 0.019),
        card_mask=(True, True, False, True),
        slice_key="raw_critic|weapon|0-5m",
        temperature=1.0,
    )
    assert permuted.probability_by_item_id == pytest.approx(by_id)


def test_all_tie_and_temperature_grid_selection_are_deterministic() -> None:
    """all-tie は valid 候補へ等確率を与え、development NLL で温度を選ぶ。

    candidate count 不整合は softmax 前に拒否し、欠損 score を有効候補へ補完しない。
    """
    scale = _score_scale()
    builder = TeacherSoftTargetBuilder(scale, _calibration(scale))
    tied = builder.build(
        candidate_item_ids=("wand", "knife", "axe", "masked"),
        teacher_scores=(2.0, 2.0, 2.0, None),
        card_mask=(True, True, True, False),
        slice_key="raw_critic|weapon|0-5m",
        temperature=1.0,
    )
    assert tied.probabilities[:3] == pytest.approx((1.0 / 3.0,) * 3)
    assert tied.tie_groups == (("axe", "knife", "wand"),)
    _rows = _split_rows()
    _manifest = SplitManifest.freeze(_rows, seed="selector-v1")
    _dev_row = next(r for r in _rows if _manifest.split_for(r) in ("train", "validation"))
    examples = [
        {
            **_dev_row,
            "candidate_item_ids": ("wand", "knife"),
            "teacher_scores": (2.0, 0.0),
            "card_mask": (True, True),
            "observed_best_item_id": "wand",
        }
    ]
    assert builder.select_temperature(
        examples, slice_key="raw_critic|weapon|0-5m", split_manifest=_manifest
    ) == 0.25
    with pytest.raises(TeacherArtifactError, match="invalid"):
        builder.build(
            candidate_item_ids=("wand", "knife"),
            teacher_scores=(2.0,),
            card_mask=(True, True),
            slice_key="raw_critic|weapon|0-5m",
            temperature=1.0,
        )


def test_scale_substitution_refit_and_blocked_reliability_are_rejected() -> None:
    """scale identity substitution、02-01 再 fit、error UCB blocker を拒否する。

    teacher type/identity/sigma/tie unit の全 binding sibling をロード時に検証する。
    """
    scale = _score_scale()
    calibration = _calibration(scale)
    other = _score_scale()
    other["teacher_identity"] = "9" * 64
    with pytest.raises(TeacherArtifactError):
        TeacherSoftTargetBuilder(other, calibration)
    builder = TeacherSoftTargetBuilder(scale, calibration)
    with pytest.raises(TeacherArtifactError, match="refit"):
        builder.refit_scale([])
    blocked = _calibration(scale, residual=1.2)
    with pytest.raises(TeacherArtifactError, match="blocker"):
        TeacherSoftTargetBuilder(scale, blocked)


def test_release_builder_rejects_identity_verdict_cycle_split_and_test_write(tmp_path) -> None:
    """source identity、approval、DAG、split binding、test write を fail-closed にする。

    各 counterexample は出力 directory を公開する前に失敗する。
    """
    source = "1" * 64
    manifest = SplitManifest.freeze(_split_rows(), seed="selector-v1")
    base = _split_rows()[0]
    split = manifest.split_for(base)
    row = {
        **base,
        "split": split,
        "source_identity": source,
        "source_trace_ref": "trace/source-1",
        "ancestor_refs": ["trace/root"],
        "label_verdict": {"status": "approved", "teacher_identity": "a" * 64, "action_kind": "choose_card"},
    }
    builder = ItemSelectorDatasetBuilder(
        split_manifest=manifest,
        source_identity=source,
        teacher_identity="a" * 64,
        derived_logical_id="derived/item-selector-v1",
    )
    output = tmp_path / "release"
    released = builder.release([row], output)
    assert released["row_count"] == 1
    assert json.loads((output / "manifest.json").read_text(encoding="utf-8"))["test_row_count"] == 0

    mutations = []
    wrong_source = copy.deepcopy(row)
    wrong_source["source_identity"] = "2" * 64
    mutations.append(wrong_source)
    unapproved = copy.deepcopy(row)
    unapproved["label_verdict"]["status"] = "pending"
    mutations.append(unapproved)
    cyclic = copy.deepcopy(row)
    cyclic["ancestor_refs"].append("derived/item-selector-v1")
    mutations.append(cyclic)
    mismatched = copy.deepcopy(row)
    mismatched["split"] = "test" if split != "test" else "train"
    mutations.append(mismatched)
    regrouped = copy.deepcopy(row)
    regrouped["episode_id"] = _split_rows()[-1]["episode_id"]
    mutations.append(regrouped)
    for index, mutation in enumerate(mutations):
        with pytest.raises(DatasetReleaseError):
            builder.release([mutation], tmp_path / f"bad-{index}")

    test_row = next(row for row in _split_rows() if manifest.split_for(row) == "test")
    test_payload = {**row, **test_row, "split": "test"}
    with pytest.raises(DatasetReleaseError, match="test"):
        builder.release([test_payload], tmp_path / "test-write")


def test_learning_curve_policy_enforces_cadence_and_rejects_non_boundary_counts() -> None:
    """LearningCurvePolicy.should_stop が block cadence を強制し境界外は拒否する。

    initial_decisions=2000、block_size=2000 の場合、count が 1/2/3 では
    DatasetReleaseError を送出し、2000/4000/6000 の正規 cadence のみを受理する。
    """
    policy = LearningCurvePolicy(
        initial_decisions=2000,
        block_size=2000,
        ndcg_gain_threshold=0.005,
        consecutive_blocks=2,
    )

    # 境界外の point (1, 2, 3) は拒否される
    for bad_counts in ([1], [2], [3], [1, 2, 3]):
        bad_points = [{"decision_count": c, "validation_ndcg": 0.5 + i * 0.01}
                      for i, c in enumerate(bad_counts)]
        with pytest.raises(DatasetReleaseError, match="initial_decisions|cadence"):
            policy.should_stop(bad_points, coverage_complete=True, hard_cap=100_000)

    # 正規 cadence: 2000, 4000 → 2 点でまだ停止しない
    two_points = [
        {"decision_count": 2000, "validation_ndcg": 0.700},
        {"decision_count": 4000, "validation_ndcg": 0.701},
    ]
    assert policy.should_stop(two_points, coverage_complete=True, hard_cap=100_000) is False

    # 正規 cadence: 2000, 4000, 6000 で NDCG gain < 0.005 が 2 block 継続 → 停止
    plateau_points = [
        {"decision_count": 2000, "validation_ndcg": 0.700},
        {"decision_count": 4000, "validation_ndcg": 0.7003},
        {"decision_count": 6000, "validation_ndcg": 0.7005},
    ]
    assert policy.should_stop(plateau_points, coverage_complete=True, hard_cap=100_000) is True

    # hard_cap 超過は cadence 問わず停止
    assert policy.should_stop(two_points[:1], coverage_complete=False, hard_cap=2000) is True

    # cadence は正しいが gap が合わない (2000, 5000) は拒否
    wrong_gap = [
        {"decision_count": 2000, "validation_ndcg": 0.700},
        {"decision_count": 5000, "validation_ndcg": 0.701},
    ]
    with pytest.raises(DatasetReleaseError, match="cadence"):
        policy.should_stop(wrong_gap, coverage_complete=True, hard_cap=100_000)


def test_release_builder_requires_explicit_choose_card_action_kind(tmp_path) -> None:
    """ItemSelectorDatasetBuilder.release が action_kind='choose_card' を明示的に要求する。

    action_kind が None / fallback / missing の場合は DatasetReleaseError を送出し、
    デフォルト値への暗黙 fallback を許さない。
    """
    source = "1" * 64
    teacher = "a" * 64
    manifest = SplitManifest.freeze(_split_rows(), seed="selector-v1")
    base = _split_rows()[0]
    split = manifest.split_for(base)
    builder = ItemSelectorDatasetBuilder(
        split_manifest=manifest,
        source_identity=source,
        teacher_identity=teacher,
        derived_logical_id="derived/item-selector-action-test",
    )

    def _row(action_kind_override=None, explicit_none=False):
        verdict: dict = {"status": "approved", "teacher_identity": teacher}
        if explicit_none:
            verdict["action_kind"] = None
        elif action_kind_override is not None:
            verdict["action_kind"] = action_kind_override
        return {
            **base,
            "split": split,
            "source_identity": source,
            "source_trace_ref": "trace/source-1",
            "ancestor_refs": ["trace/root"],
            "label_verdict": verdict,
        }

    # 正常: action_kind='choose_card' 明示
    explicit_row = _row("choose_card")
    result = builder.release([explicit_row], tmp_path / "valid")
    assert result["row_count"] == 1

    # 異常: action_kind なし (key 自体がない)
    no_kind_row = _row()
    with pytest.raises(DatasetReleaseError, match="choose_card"):
        builder.release([no_kind_row], tmp_path / "no-kind")

    # 異常: action_kind=None 明示
    none_row = _row(explicit_none=True)
    with pytest.raises(DatasetReleaseError, match="choose_card"):
        builder.release([none_row], tmp_path / "none-kind")

    # 異常: fallback action_kind
    for bad_kind in ("choose_fallback", "reroll", "skip", "banish", "ack_chest", "confirm"):
        bad_row = _row(bad_kind)
        with pytest.raises(DatasetReleaseError, match="choose_card"):
            builder.release([bad_row], tmp_path / f"bad-{bad_kind}")
