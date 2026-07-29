"""保存済み Survivors run から Value Source descriptor を監査・公開する CLI。

初心者向け:
訓練プロセスとは別に run の固定済み artifact を再検査し、probe ready・not ready・入力不正を
終了コード 0 / 2 / 3 で自動化へ返します。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from games.survivors.value_source_descriptor import (
    ValueSourceDescriptorError,
    build_value_source_descriptor,
    write_value_source_descriptor,
)


def _read_json_object(path: Path, label: str) -> Mapping[str, Any]:
    """UTF-8 JSON object を読み、欠落・構文・root 型を一つの契約エラーにする。

    初心者向け:
    audit が空 object で処理を続けず、入力修復が必要な場合は終了コード 3 にできます。
    """
    if not path.is_file():
        raise ValueSourceDescriptorError(f"{label} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueSourceDescriptorError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(value, Mapping):
        raise ValueSourceDescriptorError(f"{label} must contain a JSON object")
    return value


def _parser() -> argparse.ArgumentParser:
    """Value Source audit の versioned CLI parser を構築する。

    初心者向け:
    run・schema・時刻を全て明示入力にして、cwd や wall clock による結果の揺れを防ぎます。
    """
    parser = argparse.ArgumentParser(
        description="Audit and publish a Survivors immutable value source descriptor."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--obs-schema-json", type=Path, required=True)
    parser.add_argument("--created-at-utc", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """audit を実行し ready=0 / not-ready=2 / invalid=3 を返す。

    初心者向け:
    not-ready descriptor も不足理由の診断用に atomic publish し、schema 不正だけは公開しません。
    """
    try:
        args = _parser().parse_args(argv)
        run_dir = args.run_dir.resolve()
        completion = _read_json_object(
            run_dir / "log" / "value_source_completion.json",
            "value source completion",
        )
        provenance = _read_json_object(
            run_dir / "log" / "value_source_provenance.json",
            "value source provenance",
        )
        run_meta = _read_json_object(
            run_dir / "log" / "run_meta.json",
            "run metadata",
        )
        obs_schema = _read_json_object(
            args.obs_schema_json.resolve(),
            "observation schema",
        )
        descriptor = build_value_source_descriptor(
            run_dir=run_dir,
            completion=completion,
            obs_schema=obs_schema,
            git_commit=run_meta.get("git_commit"),
            created_at_utc=args.created_at_utc,
            source_provenance=provenance,
        )
        destination = write_value_source_descriptor(run_dir, descriptor)
        print(
            json.dumps(
                {
                    "ready_for_probe": descriptor["ready_for_probe"],
                    "blocking_reasons": descriptor["blocking_reasons"],
                    "identity_sha256": descriptor["identity_sha256"],
                    "descriptor_path": str(destination),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0 if descriptor["ready_for_probe"] else 2
    except (ValueSourceDescriptorError, OSError, TypeError, ValueError) as exc:
        print(f"[ERROR] invalid value source input: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
