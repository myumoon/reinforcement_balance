"""Survivors HUD parser の development calibration package 発行 CLI。

JSON の train/validation 注釈を読み、split 分離された fit と11 gate report を作って
内容 hash 付き package を atomic publish します。formal 発行は04-05まで停止します。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from survivors.vision.hud_calibration import PackageWriter, fit, validate

def _build_arg_parser() -> argparse.ArgumentParser:
    """calibration CLI の引数 parser を構築する。

train と validation の入力を別 option にして、operator が同じファイルを誤用しにくくします。
    """
    parser = argparse.ArgumentParser(description="Survivors HUD parser calibration package を発行します")
    parser.add_argument("--train-annotations", required=True, type=Path, help="model_train 注釈 JSON")
    parser.add_argument("--validation-annotations", required=True, type=Path, help="model_validation 注釈 JSON")
    parser.add_argument("--output-package", required=True, type=Path, help="development package 出力先")
    parser.add_argument("--formal", action="store_true", help="formal 発行（04-05まで未実装）")
    parser.add_argument("--formal-dataset-eligible", action="store_true", help="正式 dataset の operator 証明")
    return parser

def _load_annotations(path: Path) -> list[dict[str, object]]:
    """JSON 配列から annotation mapping を読み込む。

top-level と各要素の型を検査し、不完全な入力を model 層へ渡す前に明示的に拒否します。
    """
    try:
        wire = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"注釈 JSON を読めません: {path}") from exc
    if not isinstance(wire, list) or not all(isinstance(item, dict) for item in wire):
        raise ValueError("注釈 JSON は object の配列である必要があります")
    return wire

def main(argv: list[str] | None = None) -> int:
    """注釈を calibration して parser package を発行する。

development は atomic publish し、formal 指定時は未実装 publisher で明示的に停止します。
    """
    args = _build_arg_parser().parse_args(argv)
    try:
        train_result = fit(
            _load_annotations(args.train_annotations), formal_mode=args.formal,
            formal_dataset_eligible=args.formal_dataset_eligible,
        )
        report = validate(
            _load_annotations(args.validation_annotations), formal_mode=args.formal,
            formal_dataset_eligible=args.formal_dataset_eligible,
        )
        if args.formal:
            PackageWriter.publish_formal(args.output_package, train_result, report)
        meta = PackageWriter.publish_development(args.output_package, train_result, report)
    except (OSError, TypeError, ValueError, NotImplementedError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"package_hash": meta.package_hash, "all_gates_passed": report.all_passed}))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
