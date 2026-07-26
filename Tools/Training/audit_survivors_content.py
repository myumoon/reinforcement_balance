"""Survivors content coverage を検査する CLI。

初心者向け:
C++ から保存した schema JSON と YAML の証跡を読み、5ゲートの不足を終了コードで通知します。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from games.survivors.content_manifest import (
    ContractValidationError,
    audit_manifest,
    build_manifest,
    load_annotations,
)


def main(argv: list[str] | None = None) -> int:
    """監査を実行し blocking がなければ 0 を返す。

    初心者向け:
    CI はこの終了コードを使って、coverage 不足の変更をマージ前に止められます。
    """
    parser = argparse.ArgumentParser(description="Audit Survivors simulator content coverage")
    parser.add_argument("--schema", type=Path, required=True, help="canonical /content_schema JSON")
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path(__file__).parent / "configs" / "survivors_content_annotations_v1.yaml",
    )
    args = parser.parse_args(argv)
    try:
        schema = json.loads(args.schema.read_text(encoding="utf-8"))
        report = audit_manifest(build_manifest(schema, load_annotations(args.annotations)))
    except (OSError, json.JSONDecodeError, ContractValidationError) as exc:
        print(json.dumps({"blocking": 1, "error": str(exc)}, ensure_ascii=False, sort_keys=True))
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["blocking"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
