"""Survivors extrinsic utility overlay を development paired targets で学習する CLI。

source-bound precomputed critic latent dataset を読み、episode group split・train-only target
normalization・best validation NDCG checkpoint を固定 artifact として保存する。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from reinbalance_survivors_contracts.canonical_json import (
    canonical_hash,
)

from games.survivors.extrinsic_value_overlay import (
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_EPOCHS,
    DEFAULT_PATIENCE,
    DEFAULT_WEIGHT_DECAY,
    ExtrinsicValueOverlay,
    OverlayExample,
    fit_overlay,
    prepare_overlay_partitions,
    save_extrinsic_value_overlay,
)
from games.survivors.value_source_loader import (
    LoadedValueSource,
    load_value_source,
)

DATASET_SCHEMA_VERSION = "survivors.extrinsic_value_training_set.v1"
_DATASET_FIELDS = frozenset(
    {
        "schema_version",
        "dataset_hash",
        "source_model_hash",
        "schema_hash",
        "vecnormalize_hash",
        "examples",
    }
)


class OverlayTrainingCliError(ValueError):
    """training CLI 構文または dataset binding 違反の例外。

    source mismatch と invalid example を学習・artifact write 前の同じ失敗経路へ集約する。
    """


class _OverlayArgumentParser(argparse.ArgumentParser):
    """argparse の process exit を CLI 契約エラーへ変換する parser。

    test と orchestration が構文不正を main の終了コード 3 として扱えるようにする。
    """

    def error(self, message: str) -> None:
        """argparse error を OverlayTrainingCliError として送出する。

        部分 overlay artifact を作らず caller へ元の構文内容を残す。
        """
        raise OverlayTrainingCliError(
            f"invalid overlay training arguments: {message}"
        )


def _source_bindings(source: LoadedValueSource) -> dict[str, str]:
    """training dataset に必要な source identity sibling を返す。

    loader が再検証済みの model・policy schema・VecNormalize hash だけを使用する。
    """
    return {
        "source_model_hash": source.descriptor["artifacts"]["model"][
            "sha256"
        ],
        "schema_hash": source.policy_state_schema[
            "policy_state_schema_hash"
        ],
        "vecnormalize_hash": source.descriptor["artifacts"][
            "vecnormalize"
        ]["sha256"],
    }


def _dataset_identity_payload(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """自己参照 dataset_hash を除いた canonical payload を返す。

    source bindings と全 example rows を同じ dataset identity へ含める。
    """
    return {
        key: value[key]
        for key in sorted(_DATASET_FIELDS - {"dataset_hash"})
    }


def build_training_dataset_payload(
    examples: Sequence[OverlayExample],
    source: LoadedValueSource,
) -> dict[str, Any]:
    """source-bound overlay training dataset object を構築する。

    examples を ID 順へ canonicalize し、独自 hash ではなく Common canonical JSON を使う。
    """
    if (
        isinstance(examples, (str, bytes))
        or not examples
        or any(not isinstance(item, OverlayExample) for item in examples)
    ):
        raise OverlayTrainingCliError(
            "overlay examples are required"
        )
    payload: dict[str, Any] = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "dataset_hash": "",
        **_source_bindings(source),
        "examples": [
            example.to_wire()
            for example in sorted(
                examples,
                key=lambda item: item.example_id,
            )
        ],
    }
    payload["dataset_hash"] = canonical_hash(
        _dataset_identity_payload(payload)
    )
    return payload


def load_training_dataset(
    path: Path,
    source: LoadedValueSource,
) -> tuple[list[OverlayExample], str]:
    """exact-schema JSON dataset を読み source と content identity を検証する。

    model・schema・VecNormalize の全 binding sibling を学習開始前に対称検証する。
    """
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OverlayTrainingCliError(
            "overlay training dataset could not be read"
        ) from exc
    if not isinstance(value, Mapping) or set(value) != _DATASET_FIELDS:
        raise OverlayTrainingCliError(
            "overlay training dataset fields mismatch"
        )
    if value["schema_version"] != DATASET_SCHEMA_VERSION:
        raise OverlayTrainingCliError(
            "unsupported overlay training dataset schema"
        )
    binding_messages = {
        "source_model_hash": "source model hash mismatch",
        "schema_hash": "schema hash mismatch",
        "vecnormalize_hash": "vecnormalize hash mismatch",
    }
    for field, expected in _source_bindings(source).items():
        if value[field] != expected:
            raise OverlayTrainingCliError(
                binding_messages[field]
            )
    expected_hash = canonical_hash(_dataset_identity_payload(value))
    if value["dataset_hash"] != expected_hash:
        raise OverlayTrainingCliError(
            "overlay training dataset hash mismatch"
        )
    rows = value["examples"]
    if not isinstance(rows, list) or not rows:
        raise OverlayTrainingCliError(
            "overlay training examples are required"
        )
    try:
        examples = [OverlayExample.from_wire(row) for row in rows]
    except (TypeError, ValueError) as exc:
        raise OverlayTrainingCliError(
            "overlay training example is invalid"
        ) from exc
    return examples, expected_hash


def build_parser() -> argparse.ArgumentParser:
    """overlay training の versioned CLI 引数と固定 default を定義する。

    loss weight/head architecture は override 不可とし、plan の Adam 設定だけを公開する。
    """
    parser = _OverlayArgumentParser(
        description="Train a source-bound Survivors extrinsic value overlay."
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-seed", default="phase6-extrinsic-overlay-v1")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=DEFAULT_MAX_EPOCHS,
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_PATIENCE,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """dataset を split/fit し source-bound overlay artifact を保存する。

    final test は sealed のままにし、success=0 / invalid input or training failure=3 を返す。
    """
    try:
        args = build_parser().parse_args(argv)
        source = load_value_source(
            args.source_manifest,
            device=args.device,
        )
        examples, dataset_hash = load_training_dataset(
            args.dataset,
            source,
        )
        overlay = ExtrinsicValueOverlay.create(source)
        partitions = prepare_overlay_partitions(
            examples,
            split_seed=args.split_seed,
        )
        summary = fit_overlay(
            overlay,
            partitions,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            max_epochs=args.max_epochs,
            patience=args.patience,
        )
        manifest_path = save_extrinsic_value_overlay(
            overlay,
            args.output_dir,
            summary=summary,
            paired_dataset_hash=dataset_hash,
        )
        print(
            json.dumps(
                {
                    "status": "DONE",
                    "manifest": str(manifest_path),
                    "best_val_ndcg": summary.best_val_ndcg,
                    "best_epoch": summary.best_epoch,
                    "final_test_status": "sealed_unopened",
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    except (OSError, TypeError, ValueError) as exc:
        print(
            f"[ERROR] extrinsic value overlay training failed: {exc}",
            file=sys.stderr,
        )
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
