#!/usr/bin/env python3
"""事前登録ルールから sealed Survivors session の評価窓 plan を作る。"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

import yaml

from survivors.capture_dataset import DatasetWriter
from survivors.evaluation_windows import EvaluationWindowPlan, generate_windows, validate_rule
from survivors.vision import hud_parser as hud_parser_module
from survivors.vision.hud_parser import HudParser
from survivors.vision.hud_truth import load_frame_pixels
from survivors.vision.world_dataset import load_class_map
from survivors.vision.world_detector import CheckpointManifest, WorldDetector, load_detector_config


def _parser() -> argparse.ArgumentParser:
    """CLI が必要とする capture・rule・development model の入力を定義する。

    detector path は checkpoint hash を実データで検証するために受け取る。
    """
    parser = argparse.ArgumentParser(description="Plan Survivors annotated evaluation windows.")
    parser.add_argument("--store-root", required=True)
    parser.add_argument("--session-id", action="append", required=True)
    parser.add_argument("--rule", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--detector-manifest", required=True)
    parser.add_argument("--detector-weights", required=True)
    parser.add_argument("--detector-config", required=True)
    parser.add_argument("--class-map", required=True)
    return parser


def _load_development_detector(args: argparse.Namespace, rule: dict[str, Any]):
    """rule に登録された hash と照合して development detector を復元する。

    weight・config・class map は CheckpointManifest の値と一致する場合のみ推論へ渡す。
    """
    import torch

    manifest = CheckpointManifest.load(args.detector_manifest)
    expected = rule["boss_windows"]["checkpoint_hash"]
    if expected != manifest.model_hash:
        raise ValueError("boss checkpoint_hash does not match detector model_hash")
    for path, field in (
        (args.detector_weights, "model_hash"),
        (args.detector_config, "config_hash"),
        (args.class_map, "class_map_hash"),
    ):
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != getattr(manifest, field):
            raise ValueError(f"detector {field} does not match checkpoint manifest")
    detector = WorldDetector.from_config(
        load_detector_config(args.detector_config), args.class_map
    )
    detector._model.load_state_dict(
        torch.load(args.detector_weights, map_location="cpu", weights_only=True)
    )
    return detector, load_class_map(args.class_map)


def _frame_loader(manifest, parser: HudParser, detector, class_map, score_threshold: float):
    """各フレームを hash 検証付きで読み、parser state と boss score を返す。

    戻り値は窓生成 helper の入力だけにし、全 PNG を memory へ保持しない。
    """
    def load(record):
        """1 frame の screen state と boss 検出有無を計算する。
        frame pixels は hash を照合してから development model に渡す。
        """
        pixels = load_frame_pixels(manifest.session_path, record)
        hud = parser.parse(
            pixels,
            session_id=manifest.session_id,
            frame_index=record.frame_id,
            captured_monotonic_ns=record.captured_monotonic_ns,
        )
        result = detector.infer(pixels[..., :3], score_threshold=score_threshold)
        boss_score = max(
            (
                float(score)
                for class_id, score in zip(result.class_ids, result.scores)
                if class_map.id_to_name(int(class_id)) == "enemy_boss"
            ),
            default=0.0,
        )
        return {"screen_state": hud.screen_state, "boss_score": boss_score}

    return load


def main(argv: list[str] | None = None) -> int:
    """hash 検証済み rule・parser・detector から plan JSON を保存する。

    出力 plan は annotation より前に外部へ登録するため atomic replace で確定する。
    """
    args = _parser().parse_args(argv)
    try:
        if len(args.session_id) != len(set(args.session_id)):
            raise ValueError("--session-id values must be unique")
        rule_wire = yaml.safe_load(Path(args.rule).read_text(encoding="utf-8"))
        rule = validate_rule(rule_wire)
        parser_hash = hashlib.sha256(Path(hud_parser_module.__file__).read_bytes()).hexdigest()
        if rule["event_windows"]["parser_artifact_hash"] != parser_hash:
            raise ValueError("parser_artifact_hash does not match the development HudParser source")
        detector, class_map = _load_development_detector(args, rule)
        sessions = {}
        for session_id in args.session_id:
            manifest = DatasetWriter.restore(args.store_root, session_id, metadata_only=True)
            parser = HudParser(parser_artifact_hash=parser_hash)
            sessions[manifest.session_id] = generate_windows(
                manifest,
                rule,
                frame_loader=_frame_loader(
                    manifest, parser, detector, class_map,
                    rule["boss_windows"]["score_threshold"],
                ),
            )
        plan = EvaluationWindowPlan.create(
            sessions,
            rule=rule,
            stride=3,
            generated_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f"{output.name}.tmp")
        temporary.write_text(
            json.dumps(plan.to_wire(), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for session_id, windows in sessions.items():
        print(f"session: {session_id}")
        print(f"windows: {len(windows)}")
    print(f"plan_canonical_hash: {plan.plan_canonical_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
