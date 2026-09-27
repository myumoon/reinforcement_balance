"""controller telemetry JSONL から M11 shadow gate の判定 JSON/Markdown を発行する CLI。

判定ロジックは ``survivors.controller.gate`` に任せる薄いドライバです。
memory sample は telemetry に含まれないため、別途 ``--memory-samples`` で渡します
(省略時は memory gate が測定不能として FAIL になります)。
判定が FAIL でも入力が有効なら終了コード0、入力が欠損・破損しているときだけ1を返します。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Sequence

from survivors.controller.gate import issue_gate_verdict, read_telemetry, write_gate_verdict


def _sha256(path: Path) -> str:
    """ファイルを1MiBずつ読んで SHA-256 を返す。

    verdict がどの telemetry から作られたかを後から照合できるようにします。
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    """CLI の引数解析器を組み立てる。

    telemetry JSONL・memory sample JSON(任意)・JSON/Markdown の出力先を受け取ります。
    """
    parser = argparse.ArgumentParser(description="controller telemetry から M11 shadow gate の判定を発行する。")
    parser.add_argument("--telemetry", required=True, type=Path, help="controller が書いた telemetry JSONL のパス")
    parser.add_argument("--memory-samples", type=Path, default=None,
                        help="[[timestamp_ns, rss_bytes], ...] 形式の JSON(省略時は memory gate FAIL)")
    parser.add_argument("--output-json", required=True, type=Path, help="判定 JSON の出力先")
    parser.add_argument("--output-markdown", required=True, type=Path, help="判定 Markdown の出力先")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI エントリポイント。

    判定を発行できたら0を返します(PASS/FAIL は問いません)。
    入力ファイルの欠損・JSON 破損・schema 不一致は標準エラーへ理由を出して1を返します。
    """
    args = build_parser().parse_args(argv)
    try:
        samples = (json.loads(args.memory_samples.read_text(encoding="utf-8"))
                   if args.memory_samples else None)
        verdict = issue_gate_verdict(read_telemetry(args.telemetry), samples)
        verdict["telemetry_sha256"] = _sha256(args.telemetry)
        write_gate_verdict(verdict, args.output_json, args.output_markdown)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        print(f"compute-controller-gate: rejected input: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
