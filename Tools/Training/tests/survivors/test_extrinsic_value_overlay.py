"""Survivors extrinsic value overlay の freeze・artifact binding を検証する。

source recurrent policy を更新せず、小さな utility head だけが学習・保存・ロードされる
契約を実モデル fixture と改変 manifest で確認する。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from reinbalance_survivors_contracts.canonical_json import canonical_json_bytes

from games.survivors.extrinsic_value_overlay import (
    MANIFEST_FILENAME,
    STATE_FILENAME,
    ExtrinsicValueOverlay,
    OverlayExample,
    fit_overlay,
    load_extrinsic_value_overlay,
    overlay_examples_from_paired_decisions,
    policy_invariance_hash,
    prepare_overlay_partitions,
    save_extrinsic_value_overlay,
)
from games.survivors.teacher_validation import OutcomeNormalizer
from games.survivors.value_scorer import ValueScorer
from value_scorer_fixtures import build_saved_value_source


def _examples(input_dim: int) -> list[OverlayExample]:
    """明確な candidate 順位を持つ三 partition 以上の fixture を作る。

    episode ごとの decision seed を固定し、各 decision に二 branch を置く。
    """
    rows: list[OverlayExample] = []
    for episode in range(10):
        decision_id = f"decision-{episode}"
        for branch, target in (("low", float(episode)), ("high", float(episode + 2))):
            feature = np.linspace(
                -1.0,
                1.0,
                input_dim,
                dtype=np.float32,
            )
            if branch == "high":
                feature = feature + np.float32(0.5)
            rows.append(
                OverlayExample(
                    example_id=f"{decision_id}-{branch}",
                    episode_id=f"episode-{episode}",
                    decision_seed=f"seed-{episode}",
                    decision_id=decision_id,
                    choice_id=branch,
                    features=feature,
                    target=target,
                    primary_rank=(target,),
                )
            )
    return rows


def test_optimizer_and_training_update_overlay_head_only(tmp_path: Path) -> None:
    """optimizer parameter IDs と source policy の bitwise freeze を確認する。

    actor action、actor weights、feature extractor、critic LSTM を一回の学習前後で比較する。
    """
    manifest_path, _, _ = build_saved_value_source(tmp_path, recurrent=True)
    scorer = ValueScorer.load(manifest_path)
    overlay = ExtrinsicValueOverlay.create(scorer.source)
    partitions = prepare_overlay_partitions(
        _examples(overlay.input_dim),
        split_seed="overlay-test",
    )
    optimizer = overlay.make_optimizer()
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert optimizer_ids == {id(parameter) for parameter in overlay.head.parameters()}

    raw_obs = np.asarray([[0.25, -0.5, 1.0]], dtype=np.float32)
    normalized = scorer.source.normalize_raw_obs(raw_obs)
    action_before, state_before = scorer.source.policy.predict(
        normalized,
        deterministic=True,
    )
    hash_before = policy_invariance_hash(scorer.source.policy)

    summary = fit_overlay(
        overlay,
        partitions,
        max_epochs=4,
        patience=2,
    )
    scorer_with_overlay = ValueScorer(scorer.source, overlay)

    action_after, state_after = scorer_with_overlay.source.policy.predict(
        normalized,
        deterministic=True,
    )
    assert action_after.tobytes() == action_before.tobytes()
    assert all(
        after.tobytes() == before.tobytes()
        for after, before in zip(state_after, state_before)
    )
    assert policy_invariance_hash(scorer.source.policy) == hash_before
    assert summary.actor_invariance_hash == hash_before


def test_manifest_and_state_are_saved_with_all_source_bindings(
    tmp_path: Path,
) -> None:
    """manifest と state dict を同時保存し必須 binding を検査する。

    target scale、validation NDCG、actor invariance も release artifact に残す。
    """
    source_path, _, _ = build_saved_value_source(
        tmp_path / "source",
        recurrent=True,
    )
    scorer = ValueScorer.load(source_path)
    overlay = ExtrinsicValueOverlay.create(scorer.source)
    partitions = prepare_overlay_partitions(
        _examples(overlay.input_dim),
        split_seed="overlay-save",
    )
    summary = fit_overlay(overlay, partitions, max_epochs=3, patience=2)
    artifact_dir = tmp_path / "overlay"

    manifest_path = save_extrinsic_value_overlay(
        overlay,
        artifact_dir,
        summary=summary,
        paired_dataset_hash="d" * 64,
        partitions=partitions,
    )

    assert manifest_path == artifact_dir / MANIFEST_FILENAME
    assert (artifact_dir / STATE_FILENAME).is_file()
    state_bytes = (artifact_dir / STATE_FILENAME).read_bytes()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for field in (
        "source_model_hash",
        "schema_hash",
        "vecnormalize_hash",
        "actor_invariance_hash",
        "best_val_ndcg",
        "target_mean",
        "target_std",
    ):
        assert field in manifest
    loaded = load_extrinsic_value_overlay(manifest_path, scorer.source)
    assert loaded.input_dim == overlay.input_dim
    overlay_scorer = ValueScorer(scorer.source, loaded)
    movement = np.asarray([0.0, 0.5, 1.0], dtype=np.float32)
    session = overlay_scorer.new_session()
    session.advance_movement(movement, episode_start=True)
    context = session.begin_level_up(
        environment_step=2,
        decision_id="decision-overlay",
        pending_obs=movement,
        episode_start=False,
    )
    overlay_value = overlay_scorer.score(
        np.asarray([[0.25, 0.5, 1.0]], dtype=np.float32),
        context,
    )[0]
    assert np.isfinite(overlay_value.value_normalized_return)
    assert np.isfinite(overlay_value.value_unscaled_return)
    with pytest.raises(ValueError, match="already exists"):
        save_extrinsic_value_overlay(
            overlay,
            artifact_dir,
            summary=summary,
            paired_dataset_hash="d" * 64,
            partitions=partitions,
        )
    assert (artifact_dir / STATE_FILENAME).read_bytes() == state_bytes


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source_model_hash", "0" * 64, "source model"),
        ("schema_hash", "1" * 64, "schema"),
        ("context_shape", [99, 1, 99], "context shape"),
    ],
)
def test_load_rejects_wrong_source_schema_and_context_shape(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    """全 source/context binding sibling の一フィールド改変を拒否する。

    state dict が正しくても manifest の model・schema・shape が違えば load しない。
    """
    source_path, _, _ = build_saved_value_source(
        tmp_path / field,
        recurrent=True,
    )
    scorer = ValueScorer.load(source_path)
    overlay = ExtrinsicValueOverlay.create(scorer.source)
    partitions = prepare_overlay_partitions(
        _examples(overlay.input_dim),
        split_seed=f"binding-{field}",
    )
    summary = fit_overlay(overlay, partitions, max_epochs=2, patience=1)
    manifest_path = save_extrinsic_value_overlay(
        overlay,
        tmp_path / f"artifact-{field}",
        summary=summary,
        paired_dataset_hash="e" * 64,
        partitions=partitions,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = value
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")

    with pytest.raises(ValueError, match=message):
        load_extrinsic_value_overlay(manifest_path, scorer.source)


def test_tie_censored_early_failure_and_quarantine_are_masked() -> None:
    """score不能 branch と primary tie を loss pair から除外する。

    scalar mask は観測済み非隔離 branch のみ、pair mask はさらに tie を除く。
    """
    rows = [
        OverlayExample(
            example_id=f"row-{index}",
            episode_id=f"episode-{index // 2}",
            decision_seed=f"seed-{index // 2}",
            decision_id=f"decision-{index // 2}",
            choice_id=f"choice-{index}",
            features=np.asarray([float(index), 1.0], dtype=np.float32),
            target=float(index),
            primary_rank=(1.0 if index < 2 else float(index),),
            status=status,
            quarantined=quarantined,
        )
        for index, (status, quarantined) in enumerate(
            (
                ("observed", False),
                ("observed", False),
                ("censored", False),
                ("early_failure", False),
                ("observed", True),
                ("observed", False),
            )
        )
    ]

    partitions = prepare_overlay_partitions(
        rows,
        split_seed="mask-test",
        require_all_partitions=False,
    )

    assert partitions.scalar_example_ids == {
        "row-0",
        "row-1",
        "row-5",
    }
    assert partitions.pair_example_ids == set()


def test_paired_outcomes_build_external_utility_and_mask_censored() -> None:
    """paired candidate outcome から utility と primary rank を生成する。

    censored branch は不完全 outcome でも target を捏造せず null/masked row として保持する。
    """
    normalizer = OutcomeNormalizer(
        horizon="short",
        partition_id="development-train",
        level_gain_scale=10.0,
        gem_gain_scale=100.0,
        kill_gain_scale=50.0,
        normalization_identity="a" * 64,
    )
    observed = {
        "status": "observed",
        "stage_cleared_or_survived_horizon": True,
        "survival_seconds": 300.0,
        "level_gain": 5.0,
        "gem_gain": 50.0,
        "kill_gain": 25.0,
        "provenance": {
            "noveld": 999.0,
            "shaped_reward": 999.0,
            "hp_penalty": -999.0,
        },
    }
    decisions = [
        {
            "episode_id": "episode-1",
            "decision_seed": "seed-1",
            "decision_id": "decision-1",
            "candidates": [
                {
                    "choice_id": "observed",
                    "critic_features": [1.0, 2.0],
                    "short": observed,
                },
                {
                    "choice_id": "censored",
                    "critic_features": [2.0, 3.0],
                    "short": {"status": "censored"},
                },
            ],
        }
    ]

    examples = overlay_examples_from_paired_decisions(
        decisions,
        normalizer,
        horizon="short",
    )

    assert examples[0].target == pytest.approx(1.4)
    assert examples[0].primary_rank[0] is True
    assert examples[1].target is None
    assert examples[1].loss_eligible is False


def test_save_rejects_when_final_test_has_been_opened(tmp_path: Path) -> None:
    """final_test 開封後の manifest 保存を拒否する。

    fit → open_final_test の順に呼んでも sealed_unopened を書けないことで
    final_test_status 偽装を防ぐ。
    """
    source_path, _, _ = build_saved_value_source(
        tmp_path / "source",
        recurrent=True,
    )
    scorer = ValueScorer.load(source_path)
    overlay = ExtrinsicValueOverlay.create(scorer.source)
    partitions = prepare_overlay_partitions(
        _examples(overlay.input_dim),
        split_seed="seal-test",
    )
    summary = fit_overlay(overlay, partitions, max_epochs=2, patience=1)

    partitions.open_final_test(method_locked=True)
    assert partitions.final_test_opened

    with pytest.raises(Exception, match="final test"):
        save_extrinsic_value_overlay(
            overlay,
            tmp_path / "artifact-seal",
            summary=summary,
            paired_dataset_hash="f" * 64,
            partitions=partitions,
        )


def test_pair_example_ids_excludes_final_test_rows() -> None:
    """pair_example_ids が final_test 行を含まないことを確認する。

    development partition の pair IDs だけを返し、final_test data を leakage しない。
    """
    rows = [
        OverlayExample(
            example_id=f"ep{ep}-b{branch}",
            episode_id=f"episode-{ep}",
            decision_seed=f"seed-{ep}",
            decision_id=f"decision-{ep}",
            choice_id=f"choice-{branch}",
            features=np.array([float(ep), float(branch)], dtype=np.float32),
            target=float(ep + branch),
            primary_rank=(float(ep + branch),),
        )
        for ep in range(12)
        for branch in range(2)
    ]
    partitions = prepare_overlay_partitions(rows, split_seed="pair-test")

    # pair_example_ids は development のみ
    for example_id in partitions.pair_example_ids:
        row = next(r for r in rows if r.example_id == example_id)
        assert partitions.assignment_for(row) in (
            "development_train",
            "development_validation",
        ), f"{example_id} should not be in final_test"
