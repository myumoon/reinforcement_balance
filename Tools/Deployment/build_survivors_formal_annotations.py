"""session ground truth から perception_formal_annotation.v1 を作成する。

dataset publish 前に使い、loader と同じ snapshot builder の semantic hash を保存する。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Sequence

from reinbalance_survivors_contracts.canonical_json import canonical_json_bytes
from survivors.formal_annotation_loader import _index_session
from survivors.formal_ground_truth import (
    CaptureGroundTruthFiles,
    formal_annotation_row,
    ground_truth_snapshot,
)
from survivors.vision.entity_tracker import default_class_map
from survivors.vision.world_dataset import load_class_map

_SESSION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _existing_annotations_are_current(path: Path, session_id: str) -> None:
    """既存 annotations が現行 schema 以外なら上書きを止める。

    旧 wire 行を新 schema と誤認して publish へ流さない。
    """
    if not path.is_file():
        return
    try:
        lines = path.read_bytes().decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError("existing annotations.jsonl must be UTF-8") from exc
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError("existing annotations.jsonl contains invalid JSON") from exc
        if (
            not isinstance(row, dict)
            or row.get("schema_version") != "perception_formal_annotation.v1"
            or row.get("session_id") != session_id
        ):
            raise ValueError("existing annotations.jsonl contains an unsupported schema or session")


def _atomic_write(path: Path, content: bytes) -> None:
    """same-directory temp file を fsync 後に置き換える。

    publish が中断しても従来 annotations.jsonl を部分内容で壊さない。
    """
    descriptor, temp_name = tempfile.mkstemp(prefix=".annotations-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def build_session_annotations(
    store_root: Path,
    session_id: str,
    coco_dir: Path,
    parser_artifact_hash: str,
    class_map_path: Path,
) -> Path:
    """一 session の annotation JSONL を GT snapshot builder から生成する。

    confirmed truth と COCO image が両方ある frame だけを annotation 対象にする。
    """
    if not _SESSION_ID.fullmatch(session_id):
        raise ValueError("session_id has an invalid path-safe format")
    if not _SHA256.fullmatch(parser_artifact_hash):
        raise ValueError("parser_artifact_hash must be a lowercase SHA-256")
    session_dir = store_root / "capture_sessions" / session_id
    files = CaptureGroundTruthFiles(
        (session_dir / "hud_truth.jsonl").read_bytes(),
        (coco_dir / session_id / "world_coco.json").read_bytes(),
        (session_dir / "frames.jsonl").read_bytes(),
    )
    indexed = _index_session(session_id, files)
    class_map = load_class_map(class_map_path)
    if class_map != default_class_map():
        raise ValueError("class map differs from the formal ground-truth loader class map")
    rows = []
    for frame_index in sorted(indexed.frame_offsets):
        truth = indexed.truth(frame_index)
        if truth is None or truth.confirmed is not True:
            continue
        frame_record = indexed.frame_record(frame_index)
        coco_image = indexed.coco_images.get(frame_record.frame_id)
        if coco_image is None:
            continue
        snapshot = ground_truth_snapshot(
            truth, coco_image, frame_record,
            parser_artifact_hash=parser_artifact_hash,
            session_id=session_id, class_map=class_map,
        )
        if snapshot is not None:
            rows.append(formal_annotation_row(snapshot, truth))
    target = session_dir / "annotations.jsonl"
    _existing_annotations_are_current(target, session_id)
    _atomic_write(target, b"".join(canonical_json_bytes(row) + b"\n" for row in rows))
    return target


def build_parser() -> argparse.ArgumentParser:
    """formal annotation CLI の必須引数を定義する。

    storage・session・COCO export・parser identity・class map を曖昧なく指定させる。
    """
    parser = argparse.ArgumentParser(description="Build Survivors formal perception annotations")
    parser.add_argument("--store-root", required=True, type=Path)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--coco-dir", required=True, type=Path)
    parser.add_argument("--parser-artifact-hash", required=True)
    parser.add_argument("--class-map", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI 引数を検証して一 session の annotation を作成する。

    path や schema の異常は例外として返し、成功時だけ zero status を返す。
    """
    args = build_parser().parse_args(argv)
    build_session_annotations(
        args.store_root, args.session_id, args.coco_dir,
        args.parser_artifact_hash, args.class_map,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
