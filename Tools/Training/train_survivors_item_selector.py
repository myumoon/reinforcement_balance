"""ItemSelector の教師あり学習 CLI。

development dataset を読み込み、AdamW で ItemSelector を学習して best/last checkpoint と
metrics JSONL を artifact store へ保存する。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch as th

from games.survivors.item_selector_model import ItemSelector
from games.survivors.item_selector_trainer import (
    ItemSelectorTrainer,
    TrainerConfig,
)


class ItemSelectorCliError(ValueError):
    """CLI argument、dataset 読み込み、trainer 設定の違反を表す。

    caller が部分成功と誤認しないよう終了コード 2 へ集約する。
    """


def _parser() -> argparse.ArgumentParser:
    """ItemSelector 学習 CLI の引数を定義する。

    dataset path、model 設定、artifact store を CLI で固定し設定ファイルから推測しない。
    """
    parser = argparse.ArgumentParser(description="Train a permutation-invariant ItemSelector.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--context-dim", type=int, required=True)
    parser.add_argument("--candidate-dim", type=int, required=True)
    parser.add_argument("--nmax", type=int, required=True)
    parser.add_argument("--target-capability-hash", required=True)
    parser.add_argument("--artifact-store", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--resume-from", type=Path, default=None)
    return parser


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    """JSONL shard から dataset rows を読み込む。

    blank line や非 object row を training input として受理しない。
    """
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ItemSelectorCliError(f"cannot read dataset: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ItemSelectorCliError(f"row {index}: {exc}") from exc
        if not isinstance(row, dict):
            raise ItemSelectorCliError(f"row {index} must be an object")
        rows.append(row)
    if not rows:
        raise ItemSelectorCliError("dataset JSONL has no rows")
    return rows


def _load_development_rows(
    dataset_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """dataset_dir の shards から train/validation rows を分けて返す。

    split field で振り分け、test rows を development loader へ渡さない。
    """
    shard_dir = dataset_dir / "shards"
    if not shard_dir.is_dir():
        raise ItemSelectorCliError(f"shards directory not found: {shard_dir}")
    all_rows: list[dict[str, Any]] = []
    for shard in sorted(shard_dir.glob("*.jsonl")):
        all_rows.extend(_load_jsonl_rows(shard))
    if not all_rows:
        raise ItemSelectorCliError("no rows found in dataset shards")
    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    for row in all_rows:
        split = row.get("split", row.get("partition"))
        if split == "train":
            train.append(row)
        elif split == "validation":
            validation.append(row)
    if not train or not validation:
        raise ItemSelectorCliError("dataset must contain train and validation splits")
    return train, validation


def run_training(
    *,
    dataset_dir: Path,
    context_dim: int,
    candidate_dim: int,
    nmax: int,
    target_capability_hash: str,
    artifact_store: Path,
    run_id: str,
    device: str = "cpu",
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    batch_size: int = 512,
    max_epochs: int = 100,
    patience: int = 12,
    resume_from: Path | None = None,
) -> dict[str, Any]:
    """dataset を読み込み ItemSelector を学習して metrics を返す。

    artifact_store/run_id に best/last checkpoint と metrics JSONL を保存する。
    """
    train, validation = _load_development_rows(dataset_dir)
    model = ItemSelector(context_dim=context_dim, candidate_dim=candidate_dim)
    config = TrainerConfig(
        learning_rate=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
    )
    trainer = ItemSelectorTrainer(
        model,
        target_capability_hash=target_capability_hash,
        nmax=nmax,
        config=config,
        device=device,
    )
    run_dir = artifact_store / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "best_checkpoint.pt"
    summary = trainer.fit(
        train,
        validation,
        checkpoint_path=checkpoint_path,
        resume_from=resume_from,
    )
    trainer.save_checkpoint(
        run_dir / "last_checkpoint.pt",
        optimizer=None,
        epoch=summary.epochs_run,
        best_val_ndcg=summary.best_val_ndcg,
    )
    metrics = {
        "run_id": run_id,
        "best_val_ndcg": summary.best_val_ndcg,
        "best_epoch": summary.best_epoch,
        "epochs_run": summary.epochs_run,
        "stopped_early": summary.stopped_early,
        "target_capability_hash": target_capability_hash,
        "nmax": nmax,
        "context_dim": context_dim,
        "candidate_dim": candidate_dim,
    }
    metrics_path = run_dir / "metrics.jsonl"
    with metrics_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(metrics) + "\n")
    return metrics


def main(argv: list[str] | None = None) -> int:
    """CLI 引数から ItemSelector 学習を実行し metrics を stdout へ出す。

    error 時は stderr と終了コード 2 を返し、artifact を部分公開しない。
    """
    args = _parser().parse_args(argv)
    try:
        metrics = run_training(
            dataset_dir=args.dataset_dir,
            context_dim=args.context_dim,
            candidate_dim=args.candidate_dim,
            nmax=args.nmax,
            target_capability_hash=args.target_capability_hash,
            artifact_store=args.artifact_store,
            run_id=args.run_id,
            device=args.device,
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            patience=args.patience,
            resume_from=args.resume_from,
        )
    except (ItemSelectorCliError, ValueError, RuntimeError) as exc:
        print(f"item selector training failed: {exc}", file=sys.stderr)
        return 2
    sys.stdout.write(json.dumps(metrics) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
