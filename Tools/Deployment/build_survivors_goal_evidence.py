"""Build a sanitized Survivors goal-evidence bundle from one campaign artifact root.

The CLI writes only to the requested temporary primary and backup stores. It
never edits docs/goal.md or a Git-managed release manifest.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _add_common_source() -> None:
    common = Path(__file__).resolve().parents[1] / "Common" / "src"
    sys.path.insert(0, str(common))


_add_common_source()

from survivors.campaign.campaign_runner import ArtifactStore  # noqa: E402
from survivors.campaign.durable_launch_store import DurableLaunchStore  # noqa: E402
from survivors.release.evidence_builder import (  # noqa: E402
    EvidenceError,
    build_goal_evidence,
    write_evidence_bundle,
)


def main(argv: list[str] | None = None) -> int:
    """Parse the campaign, ledger, backup, and output locations and build evidence.

    A formal run needs an explicit --formal-c4-root; absent that option, only a
    synthetic development fixture is accepted.
    """
    parser = argparse.ArgumentParser(description="Build a sanitized Survivors C4 goal-evidence bundle.")
    parser.add_argument("--artifacts-root", required=True, type=Path)
    parser.add_argument("--ledger-directory", required=True, type=Path)
    parser.add_argument("--backup-directory", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--formal-c4-root", type=Path)
    args = parser.parse_args(argv)
    if not args.artifacts_root.is_dir():
        parser.error("--artifacts-root must name an existing campaign artifact directory")
    try:
        artifacts = ArtifactStore(args.artifacts_root)
        with DurableLaunchStore(args.ledger_directory) as ledger:
            evidence = build_goal_evidence(
                artifacts, ledger, formal_c4_root=args.formal_c4_root
            )
            result = write_evidence_bundle(
                evidence, artifacts, ledger, args.output_directory, args.backup_directory,
                formal_c4_root=args.formal_c4_root,
            )
        print(json.dumps({"status": "DONE", **result}, sort_keys=True, separators=(",", ":")))
        return 0
    except (EvidenceError, OSError, ValueError, RuntimeError) as exc:
        print(json.dumps({"status": "ERROR", "error": str(exc)}, sort_keys=True, separators=(",", ":")))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
