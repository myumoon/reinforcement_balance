"""Survivors value choice trace dataset を UE5 episode から収集する CLI。

seed range、episode count、shard size、source manifest、current fidelity parent、UE5 ports、
epsilon を明示入力にし、Git 外 artifact store へ resume 可能な dataset を保存する。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from reinbalance_survivors_contracts.canonical_json import canonical_json_bytes
from reinbalance_survivors_contracts.fidelity_verdict import FidelityVerdict

from games.survivors.value_choice_collector import (
    DEFAULT_EPSILON,
    ChoiceTraceCollector,
    CollectorError,
)
from games.survivors.value_choice_dataset import DatasetError, DatasetWriter
from games.survivors.value_scorer import ValueScorer


class CollectionCliError(ValueError):
    """CLI 引数または formal artifact 入力が不正な場合の例外。

    argparse/runtime の失敗を終了コード 3 へ統一し、部分 shard は writer の recovery/
    quarantine 契約へ委譲する。
    """


class _CollectionArgumentParser(argparse.ArgumentParser):
    """構文エラーを process exit せず collection error に変換する parser。

    テストと orchestration が missing manifest や不正 seed range を同じ失敗経路で扱える
    ようにする。
    """

    def error(self, message: str) -> None:
        """argparse のエラー内容を保持して例外化する。

        parser 自身が ``SystemExit`` する前に捕捉し、main が一貫した JSON 非依存の stderr
        message と終了コードを返す。
        """

        raise CollectionCliError(f"invalid collection arguments: {message}")


def _parser() -> argparse.ArgumentParser:
    """formal Survivors choice collection の versioned CLI 引数を定義する。

    source/fidelity/current hash/artifact store を必須にし、collection budget のうち pilot 待ち
    の上限は固定値として追加しない。
    """

    parser = _CollectionArgumentParser(
        description="Collect Survivors recurrent value-choice traces.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="immutable value_source_descriptor.json",
    )
    parser.add_argument("--fidelity-verdict", type=Path, required=True)
    parser.add_argument("--current-producer-hashes", type=Path, required=True)
    parser.add_argument("--artifact-store", type=Path, required=True)
    parser.add_argument("--dataset-id")
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seed-end", type=int, required=True)
    parser.add_argument("--episode-count", type=int, required=True)
    parser.add_argument("--shard-size", type=int, default=100)
    parser.add_argument("--ue5-ports", type=int, nargs="+", default=[8767])
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--connect-timeout", type=int, default=120)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-environment-steps", type=int)
    return parser


def _read_json(path: Path, label: str) -> Any:
    """finite canonical JSON file を読み detached object として返す。

    decoder が許す NaN/Infinity も共有 canonical validation で拒否し、formal parent の
    provenance tree 全体を有限 JSON に限定する。
    """

    try:
        value = json.loads(Path(path).read_bytes())
        return json.loads(canonical_json_bytes(value))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise CollectionCliError(f"{label} could not be read: {exc}") from exc


def _current_hashes(value: Any) -> Mapping[str, str]:
    """bare map または audit wrapper から current gating map を取り出す。

    wrapper を許す場合も未知 top-level 形式から key を推測せず、明示
    ``gating_producer_hashes`` field だけを参照する。
    """

    if not isinstance(value, Mapping):
        raise CollectionCliError("current producer hashes must be an object")
    if "gating_producer_hashes" in value:
        if set(value) != {"gating_producer_hashes"}:
            raise CollectionCliError(
                "current producer hash wrapper fields mismatch"
            )
        value = value["gating_producer_hashes"]
    if not isinstance(value, Mapping):
        raise CollectionCliError(
            "gating_producer_hashes must be an object"
        )
    return dict(value)


def _validate_args(args: argparse.Namespace) -> list[int]:
    """seed/episode/shard/ports/epsilon の横断整合性を検証する。

    episode logical ID を seed へ一対一対応させるため range 幅を超える episode count を
    循環利用せず拒否する。
    """

    if args.seed_end < args.seed_start:
        raise CollectionCliError("seed-end must be greater than or equal to seed-start")
    if args.episode_count <= 0:
        raise CollectionCliError("episode-count must be positive")
    seeds = list(range(args.seed_start, args.seed_end + 1))
    if args.episode_count > len(seeds):
        raise CollectionCliError("episode-count exceeds inclusive seed range")
    if args.shard_size <= 0:
        raise CollectionCliError("shard-size must be positive")
    if (
        not args.ue5_ports
        or len(set(args.ue5_ports)) != len(args.ue5_ports)
        or any(port <= 0 or port > 65535 for port in args.ue5_ports)
    ):
        raise CollectionCliError("UE5 ports must be unique values in 1..65535")
    if not 0.0 <= args.epsilon <= 1.0:
        raise CollectionCliError("epsilon must be in [0, 1]")
    if args.connect_timeout <= 0:
        raise CollectionCliError("connect-timeout must be positive")
    if (
        args.max_environment_steps is not None
        and args.max_environment_steps <= 0
    ):
        raise CollectionCliError("max-environment-steps must be positive")
    return seeds[: args.episode_count]


def _external_store(path: Path) -> Path:
    """artifact store が current Git worktree 外にあることを検証する。

    dataset NPZ/JSONL を誤って repository へ ``git add`` できる配置にせず、明示した外部
    directory だけを作成する。
    """

    store = Path(path).resolve()
    repository_root = Path(__file__).resolve().parents[2]
    try:
        store.relative_to(repository_root)
    except ValueError:
        store.mkdir(parents=True, exist_ok=True)
        return store
    raise CollectionCliError("artifact-store must be outside the Git worktree")


def _dataset_id(
    configured: str | None,
    source_identity_sha256: str,
    seeds: list[int],
) -> str:
    """caller 指定名または provenance を含む deterministic dataset ID を返す。

    default 名は source hash prefix と inclusive seed range を含め、host/port/time のような
    collection placement を dataset identity に混ぜない。
    """

    if configured is not None:
        if not configured or "/" in configured or "\\" in configured:
            raise CollectionCliError("dataset-id must be a safe non-empty name")
        return configured
    return (
        "survivors-value-choices-"
        f"{source_identity_sha256[:12]}-seeds-{seeds[0]}-{seeds[-1]}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """UE5 ports を episode 単位で round-robin し shard dataset を収集する。

    各 episode は独立 recurrent session/journal を使い、既存 dataset への再実行は canonical
    record dedup と completed journal により同じ logical decision を増やさない。
    """

    try:
        args = _parser().parse_args(argv)
        seeds = _validate_args(args)
        artifact_store = _external_store(args.artifact_store)
        scorer = ValueScorer.load(args.manifest, device=args.device)
        source_identity = scorer.source.descriptor["identity_sha256"]
        dataset_id = _dataset_id(args.dataset_id, source_identity, seeds)
        dataset_root = artifact_store / dataset_id
        verdict = FidelityVerdict.from_wire(
            _read_json(args.fidelity_verdict, "fidelity verdict")
        )
        current_hashes = _current_hashes(
            _read_json(
                args.current_producer_hashes,
                "current producer hashes",
            )
        )
        writer = DatasetWriter(
            dataset_root,
            dataset_id=dataset_id,
            source_identity_sha256=source_identity,
        )
        from games.survivors.survivors_env import SurvivorsEnv

        for episode_index, seed in enumerate(seeds):
            port = args.ue5_ports[episode_index % len(args.ue5_ports)]
            env = SurvivorsEnv(
                host=args.host,
                port=port,
                connect_timeout=args.connect_timeout,
            )
            try:
                if not env.set_params(item_selection_mode="external"):
                    raise CollectionCliError(
                        f"UE5 port {port} rejected external item selection"
                    )
                collector = ChoiceTraceCollector(
                    env=env,
                    scorer=scorer,
                    session=scorer.new_session(),
                    writer=writer,
                    source_identity_sha256=source_identity,
                    fidelity_verdict=verdict,
                    current_gating_producer_hashes=current_hashes,
                    epsilon=args.epsilon,
                    seed=seed,
                    shard_size=args.shard_size,
                    journal_path=(
                        dataset_root
                        / "journals"
                        / f"episode-seed-{seed}.json"
                    ),
                )
                collector.collect_episode(
                    seed=seed,
                    episode_logical_id=f"seed-{seed}",
                    max_environment_steps=args.max_environment_steps,
                )
                if writer.active_row_count >= args.shard_size:
                    writer.commit_shard()
            finally:
                close = getattr(env, "close", None)
                if callable(close):
                    close()
        if writer.active_row_count:
            writer.commit_shard()
        print(
            json.dumps(
                {
                    "status": "DONE",
                    "dataset_id": dataset_id,
                    "manifest": str(writer.manifest_path),
                    "record_count": writer.manifest["record_count"],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    except (
        CollectionCliError,
        CollectorError,
        DatasetError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"[ERROR] choice collection failed: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
