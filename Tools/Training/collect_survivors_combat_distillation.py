"""Survivors combat distillation dataset 収集 CLI。

UE5 で教師（privileged policy）を動かし、同じ step の deploy_raw から作った DeployObs v2 と
教師の logits・value を、episode 境界付きの dataset として artifact store へ保存します。
正式収集は教師 source descriptor と current-hash の integration fidelity verdict が揃わない限り終了コード 3 で止まります。
--source-descriptor を省略した場合だけ、UE5 に接続しない development 用 synthetic dataset を作ります。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from games.survivors import distillation_collector as collector
from games.survivors.combat_distillation_dataset import CombatDistillationDataset
from reinbalance_survivors_contracts.deploy_obs import DeployObsSchema

FORMAL_EXIT_CODE = 3


def _parser() -> argparse.ArgumentParser:
    """dataset 保存先・収集設定・正式収集の前提 artifact を受け取る CLI parser を作る。

    --help は実環境へ接続しないため、引数なし discovery でも exit 0 になります。
    """
    parser = argparse.ArgumentParser(
        description=(
            "Collect Survivors combat distillation sequences (DeployObs v2) from teacher trajectories. "
            "Formal collection requires a validated teacher source descriptor and a "
            "current-hash integration fidelity verdict; otherwise it exits with code 3."
        )
    )
    parser.add_argument("--output", type=Path, required=True, help="Output directory for dataset.")
    parser.add_argument(
        "--source-descriptor", type=Path, default=None,
        help="Path to teacher ValueSourceDescriptor JSON (required for formal collection).",
    )
    parser.add_argument("--fidelity-verdict", type=Path, default=None, help="Integration fidelity verdict JSON (formal).")
    parser.add_argument("--artifact-store", type=Path, default=None, help="Artifact store root (formal).")
    parser.add_argument(
        "--generated-input-descriptor", type=Path, default=None,
        help="Generated input descriptor used to resolve current producer hashes (formal).",
    )
    parser.add_argument("--ubt-action-graph", type=Path, default=None, help="UBT action graph JSON (formal).")
    parser.add_argument("--ue5-host", default="127.0.0.1", help="UE5 HTTP host (formal).")
    parser.add_argument("--ue5-port", type=int, default=8767, help="UE5 HTTP port (formal).")
    parser.add_argument("--connect-timeout", type=float, default=60.0, help="UE5 connect timeout seconds (formal).")
    parser.add_argument("--seed", type=int, default=0, help="Reset seed of the first episode (formal).")
    parser.add_argument(
        "--validation-every", type=int, default=5,
        help="Every N-th episode goes to the validation split (0 = train only, formal).",
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
        "--action-dim", type=int, default=9, help="Combat action dimension (synthetic only).",
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

    schema = DeployObsSchema.default_v2()
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
        observation_source_classes=collector.OBSERVATION_SOURCE_CLASSES,
    )
    dataset.save(output)


def _make_env(args: argparse.Namespace):
    """正式収集で使う UE5 SurvivorsEnv を作る（開始条件を通った後にだけ呼ばれる）。

    import を遅らせ、--help や synthetic 経路で gym / HTTP 依存を読み込まないようにします。
    """
    from games.survivors.survivors_env import SurvivorsEnv

    return SurvivorsEnv(host=args.ue5_host, port=args.ue5_port, connect_timeout=args.connect_timeout)


def _formal(args: argparse.Namespace) -> int:
    """正式収集を fail-closed で実行し、終了コード（成功 0、開始条件・収集失敗 3）を返す。

    前提 artifact の指定漏れ → current producer hash の解決 → 教師 descriptor → verdict → blocking → 収集 の順で検査します。
    どの段階の失敗も stderr に 1 行出して 3 を返し、UE5 へは開始条件を全て通るまで接続しません。
    """
    missing = [
        flag for flag, value in (
            ("--fidelity-verdict", args.fidelity_verdict), ("--artifact-store", args.artifact_store),
            ("--generated-input-descriptor", args.generated_input_descriptor), ("--ubt-action-graph", args.ubt_action_graph),
        ) if value is None
    ]
    if missing:
        print(
            "Formal collection requires teacher source descriptor validation and "
            f"current-hash integration fidelity verdict; missing {', '.join(missing)}.",
            file=sys.stderr,
        )
        return FORMAL_EXIT_CODE
    try:
        from reinbalance_survivors_contracts.current_fidelity import resolve_current_gating_producer_hashes

        if not args.source_descriptor.is_file():
            raise collector.CollectionError(f"source descriptor not found: {args.source_descriptor}")
        current = resolve_current_gating_producer_hashes(
            Path(__file__).resolve().parents[2], args.generated_input_descriptor, args.ubt_action_graph,
        ).current_gating_producer_hashes
        descriptor = collector.run_formal_collection(
            source_descriptor=args.source_descriptor, fidelity_verdict=args.fidelity_verdict,
            current_gating_producer_hashes=current, env_factory=lambda: _make_env(args),
            artifact_store=args.artifact_store, output=args.output, episodes=args.episodes,
            sequence_length=args.sequence_length, burn_in=args.burn_in,
            validation_every=args.validation_every, seed=args.seed, teacher_factory=collector.load_teacher,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"formal collection failed closed: {exc}", file=sys.stderr)
        return FORMAL_EXIT_CODE
    print(f"formal dataset saved to {args.output} (artifact {descriptor.logical_id}, identity {descriptor.identity_hash})")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI 引数を解析し、正式収集または development 用 synthetic 収集を実行する。

    --source-descriptor があれば正式経路（失敗は終了コード 3）、無ければ synthetic dataset を保存します。
    synthetic dataset は development_only で、正式 student 訓練には使えません。
    """
    args = _parser().parse_args(argv)

    if args.source_descriptor is not None:
        return _formal(args)

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


if __name__ == "__main__":
    raise SystemExit(main())
