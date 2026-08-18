"""Survivors HUD parser の model_validation gate 評価 CLI。

JSON 注釈から固定済み11 gate と canonical report hash を計算し、結果を stdout または
指定 JSON へ出力します。validation 以外の split は model 層が拒否します。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from reinbalance_survivors_contracts.canonical_json import canonical_json_bytes

from survivors.vision.hud_calibration import validate

def _build_arg_parser() -> argparse.ArgumentParser:
    """HUD parser eval CLI の引数 parser を構築する。

必須の validation 注釈と任意の report 出力先、formal dataset 証明を受け取ります。
    """
    parser = argparse.ArgumentParser(description="Survivors HUD parser の11 gate を評価します")
    parser.add_argument("--annotations", required=True, type=Path, help="model_validation 注釈 JSON")
    parser.add_argument("--output", type=Path, help="validation report JSON 出力先")
    parser.add_argument("--formal", action="store_true", help="formal dataset gate を有効化")
    parser.add_argument("--formal-dataset-eligible", action="store_true", help="正式 dataset の operator 証明")
    return parser

def _load_annotations(path: Path) -> list[dict[str, object]]:
    """eval 用 JSON 配列を annotation mapping として読み込む。

壊れた JSON や object 以外の要素を拒否し、評価失敗を空 report と取り違えないようにします。
    """
    try:
        wire = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"注釈 JSON を読めません: {path}") from exc
    if not isinstance(wire, list) or not all(isinstance(item, dict) for item in wire):
        raise ValueError("注釈 JSON は object の配列である必要があります")
    return wire

def main(argv: list[str] | None = None) -> int:
    """validation report を計算して表示または保存する。

report の canonical wire をそのまま使い、出力先の有無で metric 内容が変わらないようにします。
    """
    args = _build_arg_parser().parse_args(argv)
    try:
        report = validate(
            _load_annotations(args.annotations), formal_mode=args.formal,
            formal_dataset_eligible=args.formal_dataset_eligible,
        )
        encoded = canonical_json_bytes(report.to_wire())
        if args.output is None:
            print(encoded.decode("utf-8"))
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes(encoded)
    except (OSError, TypeError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    return 0 if report.all_passed else 2

if __name__ == "__main__":
    raise SystemExit(main())
