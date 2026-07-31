"""承認済み Survivors teacher rows を ItemSelector development dataset へ release する CLI。

source/teacher identity、frozen split、label verdict、acyclic lineage を検証し、test 行を含まない
derived shard/manifest を明示 artifact store 配下へ transactionally 保存する。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from reinbalance_survivors_contracts.canonical_json import canonical_json_bytes

from games.survivors.item_selector_dataset import (
    DatasetReleaseError,
    ItemSelectorDatasetBuilder,
    SplitManifest,
    SplitSealedError,
)


class ItemDatasetCliError(ValueError):
    """CLI argument、入力 JSON、release contract の違反を表す。

    orchestration が部分成功と誤認しないよう main の終了コード 2 へ集約する。
    """


def _parser() -> argparse.ArgumentParser:
    """ItemSelector dataset release の全 explicit parent/output 引数を定義する。

    split と identities を row から推測せず command line で固定する。
    """
    parser = argparse.ArgumentParser(
        description="Release approved Survivors ItemSelector teacher rows.",
    )
    parser.add_argument("--rows-jsonl", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--source-identity", required=True)
    parser.add_argument("--teacher-identity", required=True)
    parser.add_argument("--derived-logical-id", required=True)
    parser.add_argument("--artifact-store", type=Path, required=True)
    return parser


def _read_rows(path: Path) -> list[dict[str, Any]]:
    """finite canonical JSONL から detached row objects を読み取る。

    blank line、非 object、NaN/Infinity を正式 source rows として受理しない。
    """
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ItemDatasetCliError(f"cannot read rows JSONL: {exc}") from exc
    if not lines or any(not line.strip() for line in lines):
        raise ItemDatasetCliError("rows JSONL must contain non-empty lines")
    result: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        try:
            value = json.loads(line)
            normalized = json.loads(canonical_json_bytes(value))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ItemDatasetCliError(f"invalid row {index}: {exc}") from exc
        if not isinstance(normalized, dict):
            raise ItemDatasetCliError(f"row {index} must be an object")
        result.append(normalized)
    return result


def _output_path(store: Path, logical_id: str) -> Path:
    """artifact store root と safe slash-separated logical ID から output path を作る。

    absolute path、空 segment、dot traversal を拒否し store 外への書込みを防ぐ。
    """
    if (
        not isinstance(logical_id, str)
        or not logical_id
        or logical_id.startswith("/")
        or "\\" in logical_id
        or any(part in {"", ".", ".."} for part in logical_id.split("/"))
    ):
        raise ItemDatasetCliError("derived logical ID must be a safe relative path")
    root = Path(store).resolve()
    destination = (root / logical_id).resolve()
    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise ItemDatasetCliError("derived output escapes artifact store") from exc
    return destination


def build_dataset(
    *,
    rows_jsonl: Path,
    split_manifest_path: Path,
    source_identity: str,
    teacher_identity: str,
    derived_logical_id: str,
    artifact_store: Path,
) -> dict[str, Any]:
    """入力 artifact をロードし検証済み development release manifest を返す。

    output は ``artifact_store/derived_logical_id`` に保存し、任意 row の source trace ref を
    manifest index から逆参照可能にする。
    """
    try:
        split = SplitManifest.load(split_manifest_path)
        rows = _read_rows(rows_jsonl)
        builder = ItemSelectorDatasetBuilder(
            split_manifest=split,
            source_identity=source_identity,
            teacher_identity=teacher_identity,
            derived_logical_id=derived_logical_id,
        )
        destination = _output_path(artifact_store, derived_logical_id)
        return builder.release(rows, destination)
    except (SplitSealedError, DatasetReleaseError, OSError) as exc:
        raise ItemDatasetCliError(str(exc)) from exc


def main(argv: Sequence[str] | None = None) -> int:
    """CLI 引数から dataset release を実行し canonical manifest を stdout へ出す。

    error 時は stderr と終了コード 2 を返し、成功時だけ artifact path を公開済みとする。
    """
    args = _parser().parse_args(argv)
    try:
        manifest = build_dataset(
            rows_jsonl=args.rows_jsonl,
            split_manifest_path=args.split_manifest,
            source_identity=args.source_identity,
            teacher_identity=args.teacher_identity,
            derived_logical_id=args.derived_logical_id,
            artifact_store=args.artifact_store,
        )
    except ItemDatasetCliError as exc:
        print(f"item dataset release failed: {exc}", file=sys.stderr)
        return 2
    sys.stdout.buffer.write(canonical_json_bytes(manifest) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
