#!/usr/bin/env python3
"""Minimal stdin annotation state machine for published capture frames."""

from __future__ import annotations

import argparse
import sys

from survivors.capture_dataset import AnnotationWriter, SEMANTIC_CLASSES


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        usage=(
            "annotate_survivors_frames.py --store-root STORE_ROOT --session-id SESSION_ID "
            "[--annotator-id ANNOTATOR_ID] [--resume] [--second-review]"
        ),
        epilog=(
            "stdin: FRAME_ID CLASS LEFT TOP RIGHT BOTTOM; commands: undo, skip, done"
        ),
    )
    parser.add_argument("--store-root", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--annotator-id", default="operator")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--second-review", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        writer = AnnotationWriter(args.store_root, args.session_id, resume=args.resume)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if sys.stdin.isatty():
        print("classes: " + ", ".join(SEMANTIC_CLASSES))
        print("enter FRAME_ID CLASS LEFT TOP RIGHT BOTTOM, undo, skip, or done")
    for raw_line in sys.stdin:
        parts = raw_line.strip().split()
        if not parts:
            continue
        command = parts[0].lower()
        if command in {"done", "quit", "exit"}:
            break
        if command == "skip":
            if len(parts) >= 2:
                try:
                    writer.write_skip(int(parts[1]))
                except (ValueError, TypeError) as exc:
                    print(f"error: {exc}", file=sys.stderr)
            continue
        if command == "undo":
            removed = writer.undo()
            print("undo: none" if removed is None else f"undo: {removed.frame_id}/{removed.class_name}")
            continue
        try:
            if len(parts) != 6:
                raise ValueError("expected FRAME_ID CLASS LEFT TOP RIGHT BOTTOM")
            record = writer.write_annotation(
                int(parts[0]),
                tuple(float(value) for value in parts[2:6]),
                parts[1],
                args.annotator_id,
                second_review=args.second_review,
            )
            print(f"saved: {record.frame_id}/{record.class_name}")
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
