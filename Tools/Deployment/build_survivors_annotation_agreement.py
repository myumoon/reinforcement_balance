#!/usr/bin/env python3
"""二つの X-AnyLabeling work-root から annotation agreement を生成する。

確認済み frame の box を共有 library の IoU matcher で集計する。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from survivors.capture_formal import build_annotation_agreement


def _parser() -> argparse.ArgumentParser:
    """二人の annotation root、対象 session、出力 JSON を受け取る。

    session ID は一つ以上必要で、出力先は明示して上書きを避ける。
    """
    parser = argparse.ArgumentParser(description="Build Survivors annotation agreement")
    parser.add_argument("--work-root-a", required=True, type=Path)
    parser.add_argument("--work-root-b", required=True, type=Path)
    parser.add_argument("--session-id", required=True, action="append")
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    """agreement report を JSON として保存する。

    schema version を含む集計を UTF-8 で書き、入力不備は終了コード1にする。
    """
    args = _parser().parse_args(argv)
    try:
        report = build_annotation_agreement(args.work_root_a, args.work_root_b, args.session_id)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"output": str(args.output), "pair_count": report["pair_count"]}))
        return 0
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
