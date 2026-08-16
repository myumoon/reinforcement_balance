#!/usr/bin/env python3
"""Capture versioned Survivors sessions into an explicit external store."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

from survivors.capture import (
    CaptureSession,
    CapturedFrame,
    CtypesWin32Api,
    DxcamCaptureBackend,
    TargetWindowPolicy,
    WindowLocator,
)
from survivors.capture_dataset import DatasetWriter
from survivors.target_profile import load_target_profile


SYNTHETIC_PROFILE_HASH = "0" * 64
SYNTHETIC_BUILD_ID = "synthetic-disposable-v1"


class FakeCaptureBackend:
    """Disposable synthetic CapturedFrame source used only by smoke tests."""

    def frames(self, duration_sec: float):
        count = max(1, int(duration_sec * 2))
        started = time.monotonic_ns()
        for frame_id in range(count):
            pixels = np.zeros((1080, 1920, 4), dtype=np.uint8)
            pixels[..., 3] = 255
            pixels[0, 0, :3] = frame_id % 256
            yield CapturedFrame(
                pixels,
                started + frame_id + 1,
                frame_id,
                (0, 0, 1920, 1080),
                True,
                SYNTHETIC_PROFILE_HASH,
                SYNTHETIC_BUILD_ID,
            )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        usage=(
            "capture_survivors.py [--store-root STORE_ROOT] [--session-id SESSION_ID] "
            "[--duration-sec N] [--synthetic] [--dry-run]"
        )
    )
    parser.add_argument("--store-root")
    parser.add_argument(
        "--session-id",
        default=datetime.now(timezone.utc).strftime("survivors-%Y%m%dT%H%M%SZ"),
    )
    parser.add_argument("--duration-sec", type=float, default=1.0, metavar="N")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _start_live_session() -> CaptureSession:
    profile = load_target_profile()
    policy = TargetWindowPolicy(
        process_executable="VampireSurvivors.exe",
        window_class="YYGameMakerYY",
        window_title="Vampire Survivors",
    )
    locator = WindowLocator(CtypesWin32Api(), profile, policy)
    target = locator.locate()
    backend = DxcamCaptureBackend.create(
        output_idx=target.monitor.dxgi_output_idx,
        device_idx=target.monitor.dxgi_device_idx,
        expected_client_rect=target.client_rect_screen_px,
    )
    return CaptureSession(locator, target, backend)


def _capture_live(session: CaptureSession, duration_sec: float):
    deadline = time.monotonic() + duration_sec
    session.start()
    try:
        while time.monotonic() < deadline:
            frame = session.capture_next()
            if frame is not None:
                yield frame
    finally:
        session.close()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.store_root is None or not args.store_root.strip():
        print("error: --store-root is required; workspace fallback is forbidden", file=sys.stderr)
        return 1
    if args.duration_sec <= 0:
        print("error: --duration-sec must be positive", file=sys.stderr)
        return 1
    if not args.session_id.strip():
        print("error: --session-id must be non-empty", file=sys.stderr)
        return 1
    try:
        if args.synthetic:
            frames = FakeCaptureBackend().frames(args.duration_sec)
            identity = SYNTHETIC_PROFILE_HASH, SYNTHETIC_BUILD_ID
            live_session = None
        else:
            live_session = _start_live_session()
            frames = _capture_live(live_session, args.duration_sec)
            identity = (
                live_session.target.target_profile_hash,
                live_session.target.game_build_id,
            )
        if args.dry_run:
            first = next(iter(frames), None)
            if first is None:
                raise ValueError("capture produced no frame")
            print(json.dumps({"status": "DRY_RUN_OK", "session_id": args.session_id}))
            return 0
        with DatasetWriter(Path(args.store_root), args.session_id, *identity) as writer:
            for frame in frames:
                writer.write_frame(frame)
            manifest = writer.publish(operator_checkpoint="synthetic" if args.synthetic else "live-pilot")
        print(
            json.dumps(
                {
                    "status": "PUBLISHED",
                    "session_id": manifest.session_id,
                    "frame_count": manifest.frame_count,
                    "formal_dataset_eligible": manifest.formal_dataset_eligible,
                }
            )
        )
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
