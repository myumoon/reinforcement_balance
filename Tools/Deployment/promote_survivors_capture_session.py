#!/usr/bin/env python3
"""収録済み session を operator attestation 付き formal session へ昇格する。

frame と annotation を保ったまま session manifest の eligibility を更新する。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from survivors.capture_formal import PromotionError, _planned_prefix, promote_session
from survivors.capture_dataset import SPLIT_NAMES
from survivors.target_profile import load_runtime_profile


def _parser() -> argparse.ArgumentParser:
    """個別または planned session 一括昇格の CLI 引数を作る。

    profile は明示指定または session log の退避先から取得する。
    build/hash と operator attestation の検証はどちらの場合も必須である。
    """
    parser = argparse.ArgumentParser(description="Promote captured Survivors sessions")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--session-id")
    group.add_argument("--all-planned", action="store_true")
    parser.add_argument("--store-root", required=True, type=Path)
    parser.add_argument("--operator-id", required=True)
    parser.add_argument("--note", required=True)
    parser.add_argument("--target-profile", type=Path)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--planned-split", choices=SPLIT_NAMES)
    return parser


def main(argv: list[str] | None = None) -> int:
    """resolved profile に一致する session を昇格し hash を JSON で返す。

    --all-planned は checkpoint prefix のある全 session に operator gate を適用する。
    profile の明示指定を省いた場合は、それぞれの session log の copy を照合に使う。
    """
    args = _parser().parse_args(argv)
    try:
        explicit_profile = (
            load_runtime_profile(args.target_profile)
            if args.target_profile is not None
            else None
        )
        if args.all_planned:
            sessions = []
            for directory in sorted((args.store_root / "capture_sessions").iterdir()):
                if not directory.is_dir():
                    continue
                manifest = json.loads((directory / "session_manifest.json").read_text(encoding="utf-8"))
                checkpoint = manifest.get("operator_checkpoint")
                if not isinstance(checkpoint, str) or not checkpoint.startswith("planned_split:"):
                    continue
                try:
                    planned, _ = _planned_prefix(checkpoint)
                except ValueError as exc:
                    raise PromotionError(f"invalid planned split in {directory.name}: {exc}") from exc
                if planned is not None:
                    sessions.append(directory.name)
        else:
            sessions = [args.session_id]
        results = []
        for session_id in sessions:
            profile = explicit_profile
            if profile is None:
                profile_path = (
                    args.store_root
                    / "session_logs"
                    / session_id
                    / "target_profile.resolved.yaml"
                )
                if not profile_path.is_file():
                    raise PromotionError(
                        f"missing target profile for session {session_id}: {profile_path}"
                    )
                profile = load_runtime_profile(profile_path)
            manifest = promote_session(
                args.store_root,
                session_id,
                operator_id=args.operator_id,
                attestation_note=args.note,
                expected_target_profile_hash=profile.target_hash,
                expected_game_build_id=profile.sections["build"]["build_id"],
                split_manifest_path=args.split_manifest,
                planned_split=args.planned_split,
            )
            results.append({"session_id": session_id, "manifest_sha256": manifest.manifest_sha256})
        if not results:
            raise PromotionError("no planned sessions found")
        print(json.dumps({"promoted": results}))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
