"""ItemSelector feature leakage と development-only ceiling ablation を検証する。

test split を封印したまま 2 feature 案を比較し、data card の floor・収集停止・hard cap
根拠が再現可能な形で出力されることを確認する。
"""

from __future__ import annotations

import json

import pytest

from games.survivors.item_feature_feasibility import (
    FeatureFeasibilityError,
    run_feature_feasibility,
)
from games.survivors.item_selector_dataset import LearningCurvePolicy, SplitManifest
from test_item_decision_features import _features


def _rows() -> tuple[SplitManifest, list[dict]]:
    """両 feature variant を持つ development/test 混在 rows を作る。

    label は item ID と順位を含み、入力 feature から独立した教師領域へ置く。
    """
    split_basis = [
        {
            "decision_id": f"decision-{index}",
            "episode_id": f"episode-{index}",
            "prefix_group_id": f"prefix-{index}",
            "near_duplicate_key": f"trace-{index}",
        }
        for index in range(20)
    ]
    manifest = SplitManifest.freeze(split_basis, seed="selector-v1")
    rows = []
    for index, basis in enumerate(split_basis):
        context_only = _features("context_only_v1").to_wire()
        danger = _features("context_danger_v1").to_wire()
        context_only["decision_id"] = basis["decision_id"]
        danger["decision_id"] = basis["decision_id"]
        rows.append(
            {
                **basis,
                "split": manifest.split_for(basis),
                "feature_variants": {
                    "context_only_v1": context_only,
                    "context_danger_v1": danger,
                },
                "teacher_label": {
                    "best_item_id": "wand" if index % 3 else "knife",
                    "ranked_item_ids": ["wand", "knife"],
                },
            }
        )
    return manifest, rows


def test_feature_objects_do_not_accept_privileged_or_teacher_values() -> None:
    """teacher value、raw world、off-screen count を feature wire へ混入できない。

    exact schema reader で全 variant/candidate sibling を対称に拒否する。
    """
    for variant in ("context_only_v1", "context_danger_v1"):
        wire = _features(variant).to_wire()
        for leaked in ("teacher_score", "critic_latent", "raw_world_state", "offscreen_enemy_count"):
            mutated = json.loads(json.dumps(wire))
            mutated["context_features"][leaked] = 1.0
            with pytest.raises(ValueError, match="context"):
                type(_features(variant)).from_wire(mutated)
            candidate_mutated = json.loads(json.dumps(wire))
            candidate_mutated["candidates"][0][leaked] = 1.0
            with pytest.raises(ValueError, match="candidate"):
                type(_features(variant)).from_wire(candidate_mutated)


def test_ablation_reads_development_only_and_writes_data_card(tmp_path) -> None:
    """train/validation のみで ceiling ablation と bootstrap CI を出力する。

    data card は test access なし、pre-registered floor、learning curve、budget hard cap を記録する。
    """
    manifest, rows = _rows()
    output = tmp_path / "item-data-card.json"
    markdown = tmp_path / "item-data-card.md"
    card = run_feature_feasibility(
        rows,
        split_manifest=manifest,
        output_json=output,
        output_markdown=markdown,
        business_top1_floor=0.62,
        ceiling_retention=0.9,
        bootstrap_resamples=20,
        bootstrap_seed=11,
        throughput_decisions_per_hour=1000,
        collection_hours=12,
        storage_budget_bytes=1_000_000,
        estimated_row_bytes=200,
    )
    assert set(card["ablations"]) == {"context_only_v1", "context_danger_v1"}
    assert card["partition_audit"]["test_read_count"] == 0
    assert card["offline_top1_floor"]["value"] != 0.85
    assert "ceiling_lower_bound" in card["offline_top1_floor"]["formula"]
    assert card["learning_curve"]["initial_decisions"] == 2000
    assert card["learning_curve"]["block_size"] == 2000
    assert card["learning_curve"]["ndcg_gain_threshold"] == 0.005
    assert card["learning_curve"]["consecutive_blocks"] == 2
    assert card["hard_cap"]["value"] == 5000
    assert output.exists() and markdown.exists()


def test_ablation_fails_if_rows_bypass_manifest_split_binding() -> None:
    """row の declared split が manifest assignment と違えば解析を拒否する。

    test を validation と偽装して ceiling に混ぜる substitution を防ぐ。
    """
    manifest, rows = _rows()
    test_row = next(row for row in rows if row["split"] == "test")
    test_row["split"] = "validation"
    with pytest.raises(FeatureFeasibilityError, match="split"):
        run_feature_feasibility(
            rows,
            split_manifest=manifest,
            business_top1_floor=0.6,
            ceiling_retention=0.9,
            bootstrap_resamples=20,
            throughput_decisions_per_hour=100,
            collection_hours=2,
            storage_budget_bytes=10_000,
            estimated_row_bytes=100,
        )


def test_learning_curve_requires_coverage_and_two_small_gain_blocks() -> None:
    """coverage 完了と二つの小 gain block が揃うまで収集を継続する。

    hard cap 到達は plateau と独立した安全上限として停止する。
    """
    policy = LearningCurvePolicy()
    points = [
        {"decision_count": 2000, "validation_ndcg": 0.70},
        {"decision_count": 4000, "validation_ndcg": 0.703},
        {"decision_count": 6000, "validation_ndcg": 0.707},
    ]
    assert not policy.should_stop(points, coverage_complete=False, hard_cap=10000)
    assert policy.should_stop(points, coverage_complete=True, hard_cap=10000)
    assert policy.should_stop(points[:1], coverage_complete=False, hard_cap=2000)
