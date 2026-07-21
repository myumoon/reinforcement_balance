"""Command line entrypoint for the Survivors artifact store."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

if __package__:
    from .artifact_bundle import (
        assert_distinct_store_roots,
        export_bundle,
        import_bundle,
    )
    from .artifact_store import ArtifactStore
else:
    from artifact_bundle import (
        assert_distinct_store_roots,
        export_bundle,
        import_bundle,
    )
    from artifact_store import ArtifactStore


def _print_json(value: Any) -> None:
    print(json.dumps(value, sort_keys=True, separators=(",", ":")))


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage Survivors artifact objects.")
    parser.add_argument("--store-root", required=True, help="Explicit primary store root.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    put = subparsers.add_parser("put")
    put.add_argument("--logical-id", required=True)
    put.add_argument("--source", required=True)
    put.add_argument("--media-type", required=True)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--uri", required=True)
    verify.add_argument("--size-bytes", type=int)

    subparsers.add_parser("list")

    audit = subparsers.add_parser("audit")
    audit.add_argument("--manifest", required=True)
    audit.add_argument("--now-utc")

    export = subparsers.add_parser("export")
    export.add_argument("--manifest", required=True)
    export.add_argument("--output", required=True)

    imp = subparsers.add_parser("import")
    imp.add_argument("--bundle", required=True)
    imp.add_argument("--verify-mode", choices=("full", "sample"), default="full")
    imp.add_argument("--sample-size", type=int)
    imp.add_argument("--random-seed", type=int, default=0)

    backup = subparsers.add_parser("check-backup-root")
    backup.add_argument("--backup-root", required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    store = ArtifactStore(args.store_root)

    if args.command == "put":
        ref = store.put(
            logical_id=args.logical_id,
            source_path=args.source,
            media_type=args.media_type,
        )
        _print_json(ref.to_wire())
        return 0
    if args.command == "verify":
        result = store.verify(args.uri, expected_size_bytes=args.size_bytes)
        _print_json(result.__dict__)
        return 0 if result.ok else 2
    if args.command == "list":
        _print_json([record.__dict__ | {"path": str(record.path)} for record in store.list_objects()])
        return 0
    if args.command == "audit":
        report = store.audit_manifest(_load_json(args.manifest), now_utc=args.now_utc)
        _print_json(report.__dict__ | {"ok": report.ok})
        return 0 if report.ok else 2
    if args.command == "export":
        export_bundle(store, _load_json(args.manifest), args.output)
        _print_json({"output": str(Path(args.output).resolve())})
        return 0
    if args.command == "import":
        result = import_bundle(
            args.bundle,
            store,
            verify_mode=args.verify_mode,
            sample_size=args.sample_size,
            random_seed=args.random_seed,
        )
        _print_json(
            {
                "manifest_hash": result.manifest_hash,
                "verification": result.verification_report.__dict__,
            }
        )
        return 0 if result.verification_report.ok else 2
    if args.command == "check-backup-root":
        assert_distinct_store_roots(store.root, args.backup_root)
        _print_json({"ok": True})
        return 0
    raise AssertionError(f"unhandled command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
