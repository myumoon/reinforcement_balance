"""Survivors combat distillation dataset 収集 CLI。

UE5 simulator または synthetic fixture からの teacher trajectory を、
DeployObs schema と episode boundary を保持した dataset として保存します。
正式収集は teacher source descriptor と fidelity verdict が揃った後に実行します。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from games.survivors.combat_distillation_dataset import CombatDistillationDataset
from reinbalance_survivors_contracts.deploy_obs import DeployObsSchema


def _parser() -> argparse.ArgumentParser:
    """dataset 保存先と収集設定だけを要求する収集 CLI parser を作る。

    --help は実環境へ接続しないため、引数なし discovery でも exit 0 になります。
    """
    parser = argparse.ArgumentParser(
        description=(
            "Collect Survivors combat distillation sequences from teacher trajectories. "
            "Formal collection requires a validated teacher source descriptor and "
            "current-hash fidelity verdict (04-07 D04-PERCEPTION-CALIBRATION)."
        )
    )
    parser.add_argument("--output", type=Path, required=True, help="Output directory for dataset.")
    parser.add_argument(
        "--source-descriptor", type=Path, default=None,
        help="Path to teacher ValueSourceDescriptor JSON (required for formal collection).",
    )
    parser.add_argument(
        "--episodes", type=int, default=64,
        help="Number of episodes to collect (synthetic fixture default: 64).",
    )
    parser.add_argument(
        "--sequence-length", type=int, default=64, help="Padded sequence length.",
    )
    parser.add_argument(
        "--burn-in", type=int, default=16, help="Burn-in steps per episode boundary.",
    )
    parser.add_argument(
        "--action-dim", type=int, default=9, help="Combat action dimension.",
    )
    return parser


def _collect_synthetic(
    output: Path,
    *,
    episodes: int,
    sequence_length: int,
    burn_in: int,
    action_dim: int,
) -> None:
    """development-only synthetic dataset を生成して保存する。

    UE5 接続なしで trainer/resume/eval のパイプラインを検証するためだけに使い、
    正式 student 訓練や release packager への入力にはなりません。
    """
    import numpy as np

    schema = DeployObsSchema.default_v1()
    obs_dim = schema.dim * 3
    rng = np.random.default_rng(seed=0)

    observations = rng.uniform(-0.5, 0.5, size=(episodes, sequence_length, obs_dim)).astype(np.float32)
    # validity と age を [0,1] に収める
    observations[:, :, schema.dim : schema.dim * 2] = np.clip(
        np.abs(observations[:, :, schema.dim : schema.dim * 2]), 0.0, 1.0
    )
    observations[:, :, schema.dim * 2 :] = np.clip(
        np.abs(observations[:, :, schema.dim * 2 :]), 0.0, 1.0
    )

    action_logits = rng.standard_normal(size=(episodes, sequence_length, action_dim)).astype(np.float32)
    teacher_values = rng.standard_normal(size=(episodes, sequence_length)).astype(np.float32)

    valid_mask = np.ones((episodes, sequence_length), dtype=np.bool_)
    episode_reset_mask = np.zeros((episodes, sequence_length), dtype=np.bool_)
    episode_reset_mask[:, 0] = True

    burn_in_mask = np.zeros((episodes, sequence_length), dtype=np.bool_)
    burn_in_mask[:, : burn_in] = True

    episode_ids = tuple(f"synthetic-{i:04d}" for i in range(episodes))
    splits = tuple(
        "train" if i < int(episodes * 0.8) else "validation"
        for i in range(episodes)
    )

    # release training gate は train-only を要求するので、train split だけを保存する
    train_idx = [i for i, s in enumerate(splits) if s == "train"]
    n_train = len(train_idx)

    dataset = CombatDistillationDataset(
        observations=observations[train_idx],
        action_logits=action_logits[train_idx],
        teacher_values=teacher_values[train_idx],
        valid_mask=valid_mask[train_idx],
        burn_in_mask=burn_in_mask[train_idx],
        episode_reset_mask=episode_reset_mask[train_idx],
        episode_ids=tuple(episode_ids[i] for i in train_idx),
        splits=tuple("train" for _ in range(n_train)),
        deploy_schema_hash=schema.schema_hash,
        burn_in_steps=burn_in,
        observation_source_classes=("hud_inventory", "screen_world_observed", "temporal_inferred", "constant"),
    )
    dataset.save(output)


def main(argv: list[str] | None = None) -> int:
    """CLI 引数を解析して synthetic dataset を development 用途で保存する。

    正式 teacher source descriptor が未指定の場合は synthetic fixture を生成し、
    収集結果に development_only ラベルが付与されることをログへ記録します。
    """
    args = _parser().parse_args(argv)

    if args.source_descriptor is not None and not args.source_descriptor.exists():
        print(f"source descriptor not found: {args.source_descriptor}", file=sys.stderr)
        return 2

    if args.source_descriptor is None:
        print(
            "WARNING: --source-descriptor not provided. "
            "Generating development-only synthetic dataset. "
            "This dataset cannot be used for formal student training.",
            file=sys.stderr,
        )
        try:
            _collect_synthetic(
                args.output,
                episodes=args.episodes,
                sequence_length=args.sequence_length,
                burn_in=args.burn_in,
                action_dim=args.action_dim,
            )
        except (OSError, ValueError) as exc:
            print(f"synthetic collection failed: {exc}", file=sys.stderr)
            return 2
        print(f"synthetic dataset saved to {args.output} (development_only=true)")
        return 0

    print(
        "Formal collection requires teacher source descriptor validation and "
        "current-hash fidelity verdict (04-07 D04-PERCEPTION-CALIBRATION). "
        "This path is not yet implemented.",
        file=sys.stderr,
    )
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
