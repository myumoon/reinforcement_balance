"""Combat distillation dataset の sequence 境界と release sealing を検証する。
小さな NumPy fixture で episode reset・burn-in・padding・教師 logits・schema hash の
保存再読込と、release 学習へ持ち込めない三種類の leakage を確認する。
"""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
from reinbalance_survivors_contracts.deploy_obs import DeployObsSchema
from games.survivors.combat_distillation_dataset import CombatDistillationDataset
SCHEMA = DeployObsSchema.default_v1()
def _dataset() -> CombatDistillationDataset:
    """padding を含む二 episode の release dataset を作る。
    全 field を canonical missing にした観測を使い、dataset 固有の mask 契約だけを分離する。
    """
    observations = np.zeros((2, 4, SCHEMA.dim * 3), dtype=np.float32)
    neutral = np.concatenate([
        np.full(field.size, field.neutral, dtype=np.float32) for field in SCHEMA.fields
    ])
    observations[:, :, :SCHEMA.dim] = neutral
    observations[:, :, SCHEMA.dim * 2 :] = 1.0
    valid = np.array([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=np.bool_)
    burn_in = np.array([[1, 0, 0, 0], [1, 0, 0, 0]], dtype=np.bool_)
    resets = np.array([[1, 0, 0, 0], [1, 0, 0, 0]], dtype=np.bool_)
    logits = np.arange(24, dtype=np.float32).reshape(2, 4, 3)
    values = np.arange(8, dtype=np.float32).reshape(2, 4)
    logits[~valid] = 0.0
    values[~valid] = 0.0
    return CombatDistillationDataset(
        observations=observations,
        action_logits=logits,
        teacher_values=values,
        valid_mask=valid,
        burn_in_mask=burn_in,
        episode_reset_mask=resets,
        episode_ids=("episode-a", "episode-b"),
        splits=("train", "train"),
        deploy_schema_hash=SCHEMA.schema_hash,
        burn_in_steps=1,
        observation_source_classes=(
            "hud_inventory", "screen_world_observed", "temporal_inferred", "constant",
        ),
    )
def test_dataset_round_trip_preserves_sequence_contract(tmp_path: Path) -> None:
    """manifest と NPZ の全 sequence field が byte-equivalent に戻る。
    episode identity と schema hash も配列外 manifest から欠落せず復元されることを確かめる。
    """
    source = _dataset()
    source.save(tmp_path / "dataset")
    restored = CombatDistillationDataset.load(tmp_path / "dataset")
    assert restored.episode_ids == source.episode_ids
    assert restored.splits == source.splits
    assert restored.deploy_schema_hash == SCHEMA.schema_hash
    for name in (
        "observations", "action_logits", "teacher_values", "valid_mask",
        "burn_in_mask", "episode_reset_mask",
    ):
        assert np.array_equal(getattr(restored, name), getattr(source, name))
    restored.assert_release_training_ready(SCHEMA)
@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda data: replace(data, observation_source_classes=("unobservable",)), "unobservable"),
        (lambda data: replace(data, teacher_actions=np.zeros((2, 4), dtype=np.int64)), "teacher actions"),
        (lambda data: replace(data, splits=("train", "validation")), "split leakage"),
    ],
)
def test_release_training_rejects_privileged_actions_and_split_leakage(
    mutation, match: str,
) -> None:
    """release step-0 gate が全 leakage sibling を対称に拒否する。
    dataset 自体は development 解析用に保持できても、optimizer へ渡す境界では必ず停止する。
    """
    with pytest.raises(ValueError, match=match):
        mutation(_dataset()).assert_release_training_ready(SCHEMA)
