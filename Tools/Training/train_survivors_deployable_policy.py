"""Survivors Deploy可能 combat student policy 訓練 CLI。

distillation → PPO → error curriculum → DAgger の全段階を一つのエントリポイントで
制御し、checkpoint resume と formal dependency gate を訓練ループへ接続します。
正式訓練は 04-07 D04-PERCEPTION-CALIBRATION と 03-04 post-curriculum fidelity verdict
が揃った後にのみ実行します。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch as th

from games.survivors.combat_distillation_dataset import CombatDistillationDataset
from games.survivors.deployable_policy_trainer import (
    DeployableCombatPolicy,
    DeployablePolicyTrainer,
    FormalDependencies,
    validate_formal_dependencies,
)


def _parser() -> argparse.ArgumentParser:
    """訓練 CLI の全引数を定義する。

    --help はモデル・dataset を開かないため、引数なし discovery でも exit 0 になります。
    """
    parser = argparse.ArgumentParser(
        description=(
            "Train a deployable Survivors combat student policy via behavior distillation. "
            "Formal training requires 04-07 D04-PERCEPTION-CALIBRATION measured perception "
            "profile and current-hash post-curriculum fidelity verdict."
        )
    )
    parser.add_argument("--dataset", type=Path, required=True, help="Path to combat distillation dataset directory.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for checkpoints.")
    parser.add_argument(
        "--resume", type=Path, default=None,
        help="Resume from existing checkpoint path.",
    )
    parser.add_argument(
        "--formal-deps", type=Path, default=None,
        help="Path to formal_dependencies JSON (required for formal training).",
    )
    parser.add_argument(
        "--updates", type=int, default=10,
        help="Number of optimizer update steps (development default: 10).",
    )
    parser.add_argument(
        "--action-dim", type=int, default=9, help="Combat action dimension."
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=256, help="GRU hidden dimension."
    )
    parser.add_argument(
        "--lr", type=float, default=3e-4, help="AdamW learning rate."
    )
    return parser


def _load_formal_deps(path: Path | None) -> FormalDependencies | None:
    """formal_dependencies JSON を FormalDependencies へ変換する。

    path が None のときは development mode として None を返し、
    存在しないパスや不正 JSON は fail closed にします。
    """
    if path is None:
        return None
    try:
        data: Any = json.loads(path.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read formal dependencies: {exc}") from exc
    from reinbalance_survivors_contracts.fidelity_verdict import FidelityVerdict
    from reinbalance_survivors_contracts.perception_error import PerceptionErrorProfile
    if not isinstance(data, dict):
        raise ValueError("formal dependencies must be a JSON object")
    _REQUIRED_KEYS = frozenset(
        {"fidelity_verdict", "perception_profile", "required_perception_profile_hash",
         "current_gating_producer_hashes", "profile_source"}
    )
    missing = _REQUIRED_KEYS - set(data)
    if missing:
        raise ValueError(f"formal dependencies missing keys: {sorted(missing)}")
    try:
        verdict = FidelityVerdict.from_wire(data["fidelity_verdict"])
        profile = PerceptionErrorProfile.from_wire(data["perception_profile"])
        required_hash = data["required_perception_profile_hash"]
        current_hashes = data["current_gating_producer_hashes"]
        profile_source = data["profile_source"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"formal dependencies schema error: {exc}") from exc
    if not isinstance(current_hashes, dict):
        raise ValueError("current_gating_producer_hashes must be a JSON object")
    return FormalDependencies(
        fidelity_verdict=verdict,
        current_gating_producer_hashes=current_hashes,
        perception_profile=profile,
        required_perception_profile_hash=required_hash,
        profile_source=profile_source,
    )


def _run_training(
    dataset: CombatDistillationDataset,
    output_dir: Path,
    *,
    resume: Path | None,
    formal_deps: FormalDependencies | None,
    updates: int,
    action_dim: int,
    hidden_dim: int,
    lr: float,
) -> None:
    """dataset を読んで指定 update 数の distillation を実行し、最終 checkpoint を保存する。

    formal_deps が None の場合は development mode (formal_mode=False) で動作し、
    生成 checkpoint は development_only=true・formal_student_eligible=false になります。
    """
    formal_mode = formal_deps is not None
    obs_dim = dataset.observations.shape[2]
    model = DeployableCombatPolicy(
        observation_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
    )
    trainer = DeployablePolicyTrainer(
        model,
        formal_mode=formal_mode,
        formal_dependencies=formal_deps,
        learning_rate=lr,
    )
    if resume is not None:
        trainer.load_checkpoint(resume)

    output_dir.mkdir(parents=True, exist_ok=True)

    for step in range(updates):
        losses = trainer.train_step(dataset)
        if step % max(1, updates // 10) == 0 or step == updates - 1:
            print(
                f"step {trainer.training_steps}: "
                f"actor_kl={losses['actor_kl']:.4f} "
                f"value_huber={losses['value_huber']:.4f} "
                f"total={losses['total']:.4f}"
            )
        checkpoint_path = output_dir / f"checkpoint_{trainer.training_steps:06d}.pt"
        trainer.save_checkpoint(checkpoint_path)

    if not formal_mode:
        print(
            "Training complete (development_only=true, formal_student_eligible=false). "
            "This checkpoint cannot be packaged as a formal runtime artifact.",
            file=sys.stderr,
        )


def main(argv: list[str] | None = None) -> int:
    """CLI 引数を解析して訓練ループを実行する。

    dataset・model・checkpoint の error は stderr と exit 2 へまとめます。
    """
    args = _parser().parse_args(argv)

    try:
        dataset = CombatDistillationDataset.load(args.dataset)
        formal_deps = _load_formal_deps(args.formal_deps)
    except (OSError, ValueError) as exc:
        print(f"input validation failed: {exc}", file=sys.stderr)
        return 2

    try:
        _run_training(
            dataset,
            args.output_dir,
            resume=args.resume,
            formal_deps=formal_deps,
            updates=args.updates,
            action_dim=args.action_dim,
            hidden_dim=args.hidden_dim,
            lr=args.lr,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"training failed: {exc}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
