#!/usr/bin/env python3
"""事前登録窓の annotation coverage と formal slice floor を照合する。

窓内 frame だけを集計し、HUD truth・COCO と必要件数の不足を確認する。
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any

from survivors.capture_dataset import DatasetWriter
from survivors.evaluation_windows import validate_plan_wire, validate_windows
from survivors.formal_ground_truth import coco_image_entries_from_wire, ground_truth_snapshot
from survivors.perception_benchmark import (
    _FORMAL_SLICE_COUNT_FLOORS,
    _FORMAL_SLICE_SESSION_FLOORS,
    slice_counts_for_snapshot,
)
from survivors.vision.hud_truth import HudTruthRecord, validate_record
from survivors.vision.world_dataset import load_class_map


def _parser() -> argparse.ArgumentParser:
    """coverage check に必要な capture・annotation・class map の path を定義する。
    contract にない model 入力を CLI に追加しない。
    """
    parser = argparse.ArgumentParser(description="Check annotated evaluation-window coverage.")
    parser.add_argument("--store-root", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--coco-dir", required=True)
    parser.add_argument("--hud-truth", required=True)
    parser.add_argument("--class-map", required=True)
    return parser


def _truth_file(root: Path, session_id: str, session_count: int) -> Path | None:
    """指定形式の hud_truth path を選び、存在しない場合は None を返す。
    複数 file の一致は session binding の曖昧さとして拒否する。
    """
    if root.is_file():
        return root if session_count == 1 else None
    candidates = (
        root / "capture_sessions" / session_id / "hud_truth.jsonl",
        root / session_id / "hud_truth.jsonl",
        root / f"{session_id}.jsonl",
    )
    found = [candidate for candidate in candidates if candidate.is_file()]
    if len(found) > 1:
        raise ValueError(f"multiple HUD truth files found for {session_id!r}")
    return found[0] if found else None


def _read_truth(path: Path | None, session_id: str) -> dict[int, HudTruthRecord]:
    """HUD truth JSONL を一意な frame mapping にし、confirmed record を検証する。
    frame/session の重複または不一致は集計に使わず停止する。
    """
    if path is None:
        return {}
    records: dict[int, HudTruthRecord] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = HudTruthRecord.from_wire(json.loads(line))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError(f"{path}: invalid HUD truth line {line_number}: {exc}") from exc
        if record.session_id != session_id or record.frame_id in records:
            raise ValueError(f"{path}: duplicate or wrong-session HUD truth record")
        if record.confirmed is True and validate_record(record):
            raise ValueError(f"{path}: invalid confirmed HUD truth frame {record.frame_id}")
        records[record.frame_id] = record
    return records


def _window_counts(args: argparse.Namespace) -> dict[str, Any]:
    """stride 適用後の GT snapshot を同じ slice helper で floor と比較する。
    window の先頭ごとに event rising-edge state を消去する。
    """
    plan = validate_plan_wire(json.loads(Path(args.plan).read_text(encoding="utf-8")))
    class_map = load_class_map(args.class_map)
    root, coco_root, truth_root = Path(args.store_root), Path(args.coco_dir), Path(args.hud_truth)
    counts: dict[str, int] = defaultdict(int)
    sessions_by_slice: dict[str, set[str]] = defaultdict(set)
    missing_truth = missing_coco = expected_count = checked_truth = 0
    all_ground_truth = True
    for session_id, raw_windows in plan.sessions.items():
        manifest = DatasetWriter.restore(root, session_id, metadata_only=True)
        windows = validate_windows(raw_windows, frame_count=manifest.frame_count)
        truth = _read_truth(_truth_file(truth_root, session_id, len(plan.sessions)), session_id)
        coco_path = coco_root / session_id / "world_coco.json"
        coco = (
            coco_image_entries_from_wire(json.loads(coco_path.read_text(encoding="utf-8")), session_id=session_id)
            if coco_path.is_file() else {}
        )
        previous_events: set[str] = set()
        for window_id, (start, end) in enumerate(windows):
            previous_events.clear()
            for frame in manifest.frame_records:
                if frame.frame_id < start or frame.frame_id > end or (frame.frame_id - start) % plan.stride:
                    continue
                expected_count += 1
                row = truth.get(frame.frame_id)
                entry = coco.get(frame.frame_id)
                checked_truth += int(row is not None and row.confirmed is True)
                missing_truth += int(row is None or row.confirmed is not True)
                missing_coco += int(entry is None)
                snapshot = ground_truth_snapshot(
                    row, entry, frame, parser_artifact_hash="coverage-check",
                    session_id=session_id, class_map=class_map,
                )
                if snapshot is None:
                    all_ground_truth = False
                    previous_events.clear()
                    continue
                frame_counts = slice_counts_for_snapshot(snapshot)
                current_events = {name for name in frame_counts if name.startswith("event:")}
                for name, amount in frame_counts.items():
                    if name not in current_events or name not in previous_events:
                        counts[name] += amount
                        if amount:
                            sessions_by_slice[name].add(session_id)
                previous_events = current_events
    shortages = {
        name: {
            "count": counts.get(name, 0), "count_floor": count_floor,
            "sessions": len(sessions_by_slice.get(name, set())),
            "session_floor": _FORMAL_SLICE_SESSION_FLOORS[name],
        }
        for name, count_floor in sorted(_FORMAL_SLICE_COUNT_FLOORS.items())
        if counts.get(name, 0) < count_floor
        or len(sessions_by_slice.get(name, set())) < _FORMAL_SLICE_SESSION_FLOORS[name]
    }
    return {
        "schema_version": "evaluation_window_coverage.v1",
        "plan_canonical_hash": plan.plan_canonical_hash,
        "stride": plan.stride,
        "expected_window_frames": expected_count,
        "confirmed_hud_truth_frames": checked_truth,
        "missing_or_unconfirmed_hud_truth_frames": missing_truth,
        "frames_not_exported_to_coco": missing_coco,
        "slice_counts": dict(sorted(counts.items())),
        "slice_sessions": {name: len(ids) for name, ids in sorted(sessions_by_slice.items())},
        "shortages": shortages,
        "provisional": not all_ground_truth,
        "provisional_reason": (
            "Development parser/detector estimates require model inputs not present in the coverage CLI contract."
            if not all_ground_truth else None
        ),
    }


def main(argv: list[str] | None = None) -> int:
    """coverage 結果を JSON で標準出力し、不足時は成功終了する。
    入力契約違反と破損データは stderr に理由を出して code 2 にする。
    """
    args = _parser().parse_args(argv)
    try:
        result = _window_counts(args)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        print(f"coverage check failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
