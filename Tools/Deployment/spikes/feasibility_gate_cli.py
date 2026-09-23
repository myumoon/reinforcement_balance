"""実測pilot証拠からPerception実現可能性ゲートの判定を発行するCLIドライバ。

pilotセッションmetadata・probe結果・annotation結果・TargetProfileのJSONファイルを
読み込み、`pilot_evidence.build_gate_evidence()`で`GateEvidence`を組み立てたうえで、
既存の`write_verdict()`をそのまま呼び出して判定JSON/Markdownを出力します。
判定ロジック自体には一切手を加えない薄いドライバです。
判定が`FAIL`でも入力さえ有効であれば正常終了（終了コード0）します。入力そのものが
欠損・型不正・schema不一致・annotator不足の場合だけ非ゼロ終了します。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import yaml

from reinbalance_survivors_contracts.ui_intent import ContractValidationError
from spikes.pilot_evidence import build_gate_evidence
from spikes.survivors_vertical_feasibility import write_verdict
from survivors.target_profile import load_runtime_profile


def _load_json(path: Path, label: str) -> Any:
    """指定パスのJSONファイルを読み込む。

    ファイルが無い場合やJSON構文が壊れている場合に、原因のわかる
    `ContractValidationError`へ変換します（生の`FileNotFoundError`等を伝播させない）。
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ContractValidationError(f"{label} could not be read: {path} ({exc})") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ContractValidationError(f"{label} is not valid JSON: {path} ({exc})") from exc


def build_parser() -> argparse.ArgumentParser:
    """CLIの引数解析器を組み立てる。

    実測pilot入力7種（セッション・probe結果・annotation結果・annotator一覧・
    代表frame数・target profile・出力先）を受け取る構成にします。
    """
    parser = argparse.ArgumentParser(
        description="実測pilot証拠からPerception実現可能性ゲートの判定を発行する。")
    parser.add_argument("--sessions", required=True, type=Path,
                        help="pilotセッションmetadata JSON配列のパス")
    parser.add_argument("--probe-results", required=True, type=Path, nargs="+",
                        help="evaluate_probe()の戻り値をJSON化したファイル（複数指定可）")
    parser.add_argument("--annotation-results", required=True, type=Path,
                        help="summarize_annotation()の戻り値をJSON化したファイル")
    parser.add_argument("--annotators", required=True, nargs="+",
                        help="独立annotation実施者の識別子（2名以上必須）")
    parser.add_argument("--representative-frames", required=True, type=int,
                        help="二重annotationした代表frame数")
    parser.add_argument("--target-profile", type=Path, default=None,
                        help="TargetProfile YAMLのパス（既定: .env/target_profile.resolved.yaml）")
    parser.add_argument("--output-json", required=True, type=Path,
                        help="判定JSONの出力先パス")
    parser.add_argument("--output-markdown", required=True, type=Path,
                        help="判定Markdownの出力先パス")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLIエントリポイント。

    引数を解析して`GateEvidence`を組み立て、判定を発行して終了コードを返します。
    戻り値0は「判定を発行できた」ことを意味し、判定内容がPASSかFAILかは問いません。
    """
    args = build_parser().parse_args(argv)
    try:
        sessions = _load_json(args.sessions, "--sessions")
        probe_evaluations = [_load_json(path, "--probe-results") for path in args.probe_results]
        annotation_summary = _load_json(args.annotation_results, "--annotation-results")
        target_profile = (load_runtime_profile(args.target_profile) if args.target_profile
                          else load_runtime_profile())
        evidence = build_gate_evidence(
            sessions=sessions,
            probe_evaluations=probe_evaluations,
            annotation_summary=annotation_summary,
            annotators=args.annotators,
            representative_frames=args.representative_frames,
            target_profile=target_profile,
        )
        write_verdict(evidence, args.output_json, args.output_markdown)
    except (ContractValidationError, ValueError, TypeError, KeyError,
            FileNotFoundError, OSError, yaml.YAMLError) as exc:
        print(f"feasibility-gate-cli: rejected input: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
