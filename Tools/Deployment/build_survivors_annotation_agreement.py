#!/usr/bin/env python3
"""二つの X-AnyLabeling work-root から annotation agreement を生成する。

確認済み frame の box を共有 library の IoU matcher で集計する。
"""

from __future__ import annotations

import argparse
import getpass
import json
from pathlib import Path
import sys

from survivors.capture_formal import build_annotation_agreement


def _parser() -> argparse.ArgumentParser:
    """二つの annotation root、annotator identity、対象 session を受け取る。

    annotator ID 省略時は OS user を使い、self_repeat は既定 24 時間を要求する。
    """
    parser = argparse.ArgumentParser(description="Build Survivors annotation agreement")
    parser.add_argument("--work-root-a", required=True, type=Path)
    parser.add_argument("--work-root-b", required=True, type=Path)
    parser.add_argument("--session-id", required=True, action="append")
    parser.add_argument("--annotator-a", default=getpass.getuser())
    parser.add_argument("--annotator-b", default=getpass.getuser())
    parser.add_argument("--minimum-gap-hours", type=float, default=24.0)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    """agreement report を JSON として保存する。

    schema version を含む集計を UTF-8 で書き、入力不備は終了コード1にする。
    """
    args = _parser().parse_args(argv)
    try:
        report = build_annotation_agreement(
            args.work_root_a, args.work_root_b, args.session_id,
            annotator_a=args.annotator_a, annotator_b=args.annotator_b,
            minimum_gap_hours=args.minimum_gap_hours,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({
            "output": str(args.output), "pair_count": report["pair_count"],
            "independence_mode": report["independence_mode"],
        }))
        return 0
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
