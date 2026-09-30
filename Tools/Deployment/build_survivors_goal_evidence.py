"""campaign artifact rootからsanitizedなSurvivors goal evidenceを生成します。

指定されたtemporary primary/backup storeへだけ保存し、正式goal文書は変更しません。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _add_common_source() -> None:
    """共有Python contract packageのsource rootをimport pathへ加えます。

Deployment script単体の実行でも同じcontract moduleを読み込めるようにします。
"""
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
    """artifact、ledger、backup、outputの引数からevidenceを作ります。

    formal campaignには明示的なformal C4 rootを要求し、通常はdevelopment fixtureだけを扱います。
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
