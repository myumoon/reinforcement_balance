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
        "--required-calibration-descriptor-hash", type=str, default=None,
        help=(
            "Expected producer calibration descriptor identity hash. This is the trust root "
            "for store-format formal dependencies and MUST be supplied out of band (launch "
            "script / CI config), never read from the formal_dependencies JSON that names the "
            "artifact store root and calibration commit."
        ),
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
    parser.add_argument(
        "--dagger-shard-dirs", type=Path, nargs="*", default=[],
        help="Paths to DAgger shard dataset directories to mix into training.",
    )
    return parser


def _require_trusted_sha256(value: object) -> str:
    """別チャネル由来の信頼済み SHA-256 期待値だけを受け付ける。

    formal_dependencies.json の中身と違い、この値は store locator を書ける主体が
    差し替えられない場所（CLI 引数・起動スクリプト・CI 設定）から来る前提です。
    形式が SHA-256 でなければ、その場で fail closed にします。
    """
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(
            "required calibration descriptor hash must be a 64-character lowercase SHA-256"
        )
    return value


def _load_formal_deps(
    path: Path | None,
    *,
    required_calibration_descriptor_hash: str | None = None,
) -> FormalDependencies | None:
    """formal_dependencies JSON を FormalDependencies へ変換する。

    path が None のときは development mode として None を返し、
    存在しないパスや不正 JSON は fail closed にします。

    formal profile のロード形式は2種類を受け付けます:
    - 開発用: "perception_profile" に artifact wire を直接埋め込む（development_only=True のみ）。
    - 正式用: "perception_profile_store_root" + "perception_calibration_commit_logical_id" で
              producer の calibration commit を開き、descriptor identity が
              required_calibration_descriptor_hash と一致した場合だけロードする。

    重要: 期待 descriptor identity は **この JSON からは読みません**。JSON は store の所在
    （root と commit logical ID）しか指定できず、期待値は呼び出し元（CLI 引数
    --required-calibration-descriptor-hash）という別チャネルから受け取ります。
    両方を同じ入力から取ると、攻撃者が store・commit・期待値を自己整合的に用意でき、
    照合がトートロジーになるためです。正式形式で期待値が無い場合は必ず失敗します。
    """
    if path is None:
        return None
    try:
        data: Any = json.loads(path.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read formal dependencies: {exc}") from exc
    from reinbalance_survivors_contracts.fidelity_verdict import FidelityVerdict
    from reinbalance_survivors_contracts.perception_profile import FittedPerceptionErrorProfile
    if not isinstance(data, dict):
        raise ValueError("formal dependencies must be a JSON object")
    _STORE_KEYS = frozenset(
        {"fidelity_verdict", "perception_profile_store_root",
         "perception_calibration_commit_logical_id", "required_perception_profile_hash",
         "current_gating_producer_hashes", "profile_source"}
    )
    _WIRE_KEYS = frozenset(
        {"fidelity_verdict", "perception_profile", "required_perception_profile_hash",
         "current_gating_producer_hashes", "profile_source"}
    )
    if set(data) == _STORE_KEYS:
        use_store = True
    elif set(data) == _WIRE_KEYS:
        use_store = False
    else:
        extra = set(data) - (_STORE_KEYS | _WIRE_KEYS)
        missing = (_WIRE_KEYS - set(data)) & (_STORE_KEYS - set(data))
        raise ValueError(
            f"formal dependencies unknown or missing keys "
            f"(unknown/extra: {sorted(extra)}, missing from both formats: {sorted(missing)})"
        )
    # 期待値と store locator を同じ入力主体から取らないための境界。
    if use_store and required_calibration_descriptor_hash is None:
        raise ValueError(
            "store-format formal dependencies require an out-of-band expected calibration "
            "descriptor hash (--required-calibration-descriptor-hash); it must never be read "
            "from the same JSON that names the artifact store and calibration commit"
        )
    if not use_store and required_calibration_descriptor_hash is not None:
        raise ValueError(
            "wire-format formal dependencies cannot be bound to a calibration descriptor hash"
        )
    required_descriptor_hash: str | None = None
    try:
        verdict = FidelityVerdict.from_wire(data["fidelity_verdict"])
        if use_store:
            from reinbalance_survivors_contracts.artifact_store import (
                ArtifactStore,
                ArtifactStoreError,
            )
            required_descriptor_hash = _require_trusted_sha256(
                required_calibration_descriptor_hash
            )
            try:
                store = ArtifactStore(data["perception_profile_store_root"])
            except (ArtifactStoreError, OSError) as exc:
                raise ValueError(f"cannot open calibration profile artifact store: {exc}") from exc
            profile = FittedPerceptionErrorProfile.from_calibration_commit(
                store,
                commit_logical_id=data["perception_calibration_commit_logical_id"],
                expected_calibration_identity_hash=required_descriptor_hash,
            )
        else:
            # 開発用: wire 直接埋め込み（development_only=True のみ受け付ける）。
            profile = FittedPerceptionErrorProfile.from_artifact_wire(data["perception_profile"])
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
        required_calibration_descriptor_hash=required_descriptor_hash,
    )


def _make_corrupt_fn(
    seed: int = 0,
) -> Any:
    """curriculum stage の corruption_scale に比例した Gaussian noise を観測へ加える。

    perception profile が到着するまでの development 訓練に使うシンプルな noise wrapper です。
    正式訓練では PerceptionErrorWrapper を使うため、このままでは formal eligibility にならない。
    """
    import numpy as _np

    rng = _np.random.default_rng(seed=seed)

    def corrupt(obs: _np.ndarray, scale: float) -> _np.ndarray:
        return obs + rng.standard_normal(obs.shape).astype(_np.float32) * scale * 0.1

    return corrupt


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
    dagger_shard_dirs: list[Path] | None = None,
) -> None:
    """dataset を読んで指定 update 数の distillation を実行し、最終 checkpoint を保存する。

    corrupt_fn を trainer へ渡して curriculum stage に応じた corruption を観測へ適用する。
    DAgger shard は dagger_shard_dirs から読み込んで各 update で train_step へ渡す。
    formal_deps が None の場合は development mode で動作し、
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
        corrupt_fn=_make_corrupt_fn(),
    )
    if resume is not None:
        trainer.load_checkpoint(resume)

    dagger_datasets = (
        [CombatDistillationDataset.load(p) for p in dagger_shard_dirs]
        if dagger_shard_dirs
        else []
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    for step in range(updates):
        losses = trainer.train_step(dataset, dagger_datasets=dagger_datasets)
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
        # 期待 calibration descriptor identity は CLI（別チャネル）からのみ渡す。
        formal_deps = _load_formal_deps(
            args.formal_deps,
            required_calibration_descriptor_hash=args.required_calibration_descriptor_hash,
        )
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
            dagger_shard_dirs=args.dagger_shard_dirs or [],
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"training failed: {exc}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
