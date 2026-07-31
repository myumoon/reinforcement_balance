"""Extrinsic value overlay の split・CLI・scorer 後方互換を検証する。

development だけで fit/model selection を行い、untouched final test と raw critic path を
学習・overlay 導入から分離する。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import train_survivors_value_overlay as training_cli
from games.survivors.extrinsic_value_overlay import (
    ExtrinsicValueOverlay,
    OverlayExample,
    prepare_overlay_partitions,
)
from games.survivors.recurrent_policy_session import CriticContext
from games.survivors.value_scorer import ValueScorer
from value_scorer_fixtures import build_saved_value_source


def _row(
    episode: int,
    branch: int,
    *,
    input_dim: int = 4,
    target: float | None = None,
) -> OverlayExample:
    """episode/seed と candidate feature を固定した一行を返す。

    target override は final leakage を検出する極端値の注入に使う。
    """
    value = float(episode * 2 + branch) if target is None else target
    return OverlayExample(
        example_id=f"example-{episode}-{branch}",
        episode_id=f"episode-{episode}",
        decision_seed=f"seed-{episode}",
        decision_id=f"decision-{episode}",
        choice_id=f"choice-{branch}",
        features=np.full(input_dim, value, dtype=np.float32),
        target=value,
        primary_rank=(value,),
    )


def _context(scorer: ValueScorer) -> CriticContext:
    """fixture policy に合う zero-state smoke context を作る。

    raw critic と overlay scorer の同一入力比較にだけ利用する。
    """
    return CriticContext.zero_state(
        environment_step=1,
        decision_id="decision-1",
        pending_obs=np.zeros(scorer.source.observation_dim, dtype=np.float32),
        episode_start=True,
        policy_state_schema=dict(scorer.source.policy_state_schema),
    )


def test_split_is_grouped_and_target_stats_use_train_only() -> None:
    """同じ episode branches を一 partition に固定し train 統計だけを fit する。

    validation/final target を置換しても target mean/std が変わらないことを確認する。
    """
    rows = [_row(episode, branch) for episode in range(15) for branch in range(2)]
    first = prepare_overlay_partitions(rows, split_seed="fixed-split")
    assignment_by_episode = {
        row.episode_id: first.assignment_for(row)
        for row in rows
    }
    for episode in range(15):
        assert len(
            {
                first.assignment_for(_row(episode, branch))
                for branch in range(2)
            }
        ) == 1

    mutated = [
        _row(
            episode,
            branch,
            target=(
                1_000_000.0
                if assignment_by_episode[f"episode-{episode}"] != "development_train"
                else None
            ),
        )
        for episode in range(15)
        for branch in range(2)
    ]
    second = prepare_overlay_partitions(mutated, split_seed="fixed-split")

    assert second.target_mean == first.target_mean
    assert second.target_std == first.target_std
    assert first.final_test_opened is False
    with pytest.raises(ValueError, match="method lock"):
        first.open_final_test(method_locked=False)


def test_episode_with_multiple_decision_seeds_remains_one_partition() -> None:
    """一 episode の複数 decision-seed group を同じ partition に固定する。

    seed 集合を episode group identity に含め、decision ごとに split して leakage させない。
    """
    rows = [_row(episode, branch) for episode in range(6) for branch in range(2)]
    changed = rows[1]
    rows[1] = OverlayExample(
        example_id=changed.example_id,
        episode_id=changed.episode_id,
        decision_seed="different-seed",
        decision_id=changed.decision_id,
        choice_id=changed.choice_id,
        features=changed.features,
        target=changed.target,
        primary_rank=changed.primary_rank,
    )

    partitions = prepare_overlay_partitions(rows, split_seed="fixed-split")

    assert partitions.assignment_for(rows[0]) == partitions.assignment_for(
        rows[1]
    )


def test_value_scorer_keeps_raw_critic_path_without_overlay(
    tmp_path: Path,
) -> None:
    """--value-overlay 未指定時に direct raw critic 値を維持する。

    optional overlay 引数の追加で既存 caller の数値経路を変えない。
    """
    manifest_path, model, vecnormalize = build_saved_value_source(
        tmp_path,
        recurrent=True,
    )
    scorer = ValueScorer.load(manifest_path)
    raw = np.asarray([[0.25, -0.5, 1.0]], dtype=np.float32)
    context = _context(scorer)
    normalized = vecnormalize.normalize_obs(raw.copy())
    obs_tensor, _ = model.policy.obs_to_tensor(normalized)
    import torch as th

    state = tuple(th.as_tensor(item) for item in context.vf)
    starts = th.ones(1, dtype=th.float32)
    with th.no_grad():
        expected = float(
            model.policy.predict_values(obs_tensor, state, starts).reshape(-1)[0]
        )

    actual = scorer.score(
        raw,
        context,
        allow_zero_state_smoke=True,
    )[0]

    assert actual.value_normalized_return == expected


def test_train_cli_parser_and_dataset_loader_bind_source_hashes(
    tmp_path: Path,
) -> None:
    """training CLI の固定既定値と dataset source binding を検証する。

    model/schema/VecNormalize の別 source 混入を学習開始前に拒否する。
    """
    source_path, _, _ = build_saved_value_source(tmp_path, recurrent=True)
    scorer = ValueScorer.load(source_path)
    overlay = ExtrinsicValueOverlay.create(scorer.source)
    dataset_path = tmp_path / "dataset.json"
    payload = training_cli.build_training_dataset_payload(
        [_row(episode, branch, input_dim=overlay.input_dim) for episode in range(6) for branch in range(2)],
        scorer.source,
    )
    dataset_path.write_text(json.dumps(payload), encoding="utf-8")
    args = training_cli.build_parser().parse_args(
        [
            "--source-manifest",
            str(source_path),
            "--dataset",
            str(dataset_path),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    assert args.learning_rate == 1e-3
    assert args.weight_decay == 1e-4
    assert args.max_epochs == 100
    assert args.patience == 10

    payload["source_model_hash"] = "0" * 64
    dataset_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="source model"):
        training_cli.load_training_dataset(dataset_path, scorer.source)
