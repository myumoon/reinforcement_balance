"""Survivors perception benchmark CLI エントリポイント。

formal 入力（D04-CAPTURE-DATASET、04-05 parser、04-08 detector）がすべて揃うまで
calibration/final セッションを開封せず、BLOCKED ステータスで終了します。
--dry-run で synthetic fixture だけを使った development-only モードで実行できます。
"""

from __future__ import annotations

import argparse
import sys


def _build_parser() -> argparse.ArgumentParser:
    """CLI 引数パーサーを構築する。

    必須の formal 入力パスと dry-run フラグを受け取ります。
    """
    p = argparse.ArgumentParser(
        description="Survivors perception pipeline のベンチマークを実行します"
    )
    p.add_argument(
        "--capture-dataset",
        help="D04-CAPTURE-DATASET マニフェストのパス（formal 実行に必須）",
    )
    p.add_argument(
        "--parser-package",
        help="04-05 formal HUD parser package のパス（formal 実行に必須）",
    )
    p.add_argument(
        "--detector-package",
        help="04-08 formal world detector package のパス（formal 実行に必須）",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="synthetic fixture を使った development-only モードで実行する",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """エントリポイント。formal 入力欠落時は BLOCKED で終了する（exit code 2）。

    calibration/final セッションは formal 入力がすべて揃うまで開封しません。
    """
    args = _build_parser().parse_args(argv)

    missing = []
    if not args.capture_dataset:
        missing.append("--capture-dataset (D04-CAPTURE-DATASET)")
    if not args.parser_package:
        missing.append("--parser-package (04-05 formal HUD parser)")
    if not args.detector_package:
        missing.append("--detector-package (04-08 formal world detector)")

    if missing and not args.dry_run:
        print(
            "BLOCKED: formal inputs required for calibration/final sessions:",
            file=sys.stderr,
        )
        for item in missing:
            print(f"  missing: {item}", file=sys.stderr)
        print(
            "Calibration and final sessions NOT opened.",
            file=sys.stderr,
        )
        print(
            "Rerun with --dry-run for synthetic development-only fixture.",
            file=sys.stderr,
        )
        return 2

    if args.dry_run or missing:
        print("development_only=true: running with synthetic fixtures")
        print("formal_perception_verdict_eligible=false")
        print("Calibration/final sessions: NOT opened (synthetic only)")
        return 0

    # formal 実行経路（D04-CAPTURE-DATASET・04-05・04-08 が揃ってから実装）
    print(
        "ERROR: formal benchmark path is not implemented in code-only PR.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
