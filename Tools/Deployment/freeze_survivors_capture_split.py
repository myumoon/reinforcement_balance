#!/usr/bin/env python3
"""昇格済み capture session の割当を読み込み split manifest を freeze する。

assign と freeze を一つの process で行い、昇格後の session hash を固定する。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from survivors.capture_formal import assign_from_manifests


def _parser() -> argparse.ArgumentParser:
    """split freeze CLI の引数 parser を作る。

    世代別 manifest と対象 session の絞り込みを指定できる。
    """
    parser = argparse.ArgumentParser(description="Freeze a Survivors capture split")
    parser.add_argument("--store-root", required=True, type=Path)
    parser.add_argument("--manifest-path", type=Path)
    parser.add_argument("--session-id", action="append", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    """昇格済み session を割り当てて固定 manifest を出力する。

    hash と件数を JSON で返し、既存 frozen path の再利用は失敗として返す。
    """
    args = _parser().parse_args(argv)
    try:
        freezer = assign_from_manifests(
            args.store_root, session_ids=args.session_id, manifest_path=args.manifest_path
        )
        manifest = freezer.freeze()
        print(json.dumps({
            "manifest_path": str(manifest.manifest_path),
            "manifest_sha256": manifest.manifest_sha256,
            "session_count_by_split": {name: len(values) for name, values in manifest.splits.items()},
        }))
        return 0
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
