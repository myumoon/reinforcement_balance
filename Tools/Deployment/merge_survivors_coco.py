#!/usr/bin/env python3
"""model train/validation COCO export を一つのモデル世代へ結合する。

image と annotation ID を再採番し、session/frame identity と categories は保持する。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any


def _parser() -> argparse.ArgumentParser:
    """COCO merge CLI の4入力と出力先を定義する。

    world と UI の train/validation ペアを個別に受け取る。
    """
    parser = argparse.ArgumentParser(description="Merge model split COCO exports")
    for role in ("world-train", "world-validation", "ui-train", "ui-validation"):
        parser.add_argument(f"--{role}", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def _merge(inputs: tuple[Path, Path]) -> dict[str, Any]:
    """二つの COCO document を参照整合性を保って一つにする。

    category schema の違いや欠落 image reference は誤った学習 data なので拒否する。
    """
    documents = [json.loads(path.read_text(encoding="utf-8")) for path in inputs]
    if any(not isinstance(doc, dict) or not all(isinstance(doc.get(key), list) for key in ("images", "annotations", "categories")) for doc in documents):
        raise ValueError("COCO document must contain images, annotations, and categories arrays")
    categories = documents[0]["categories"]
    if documents[1]["categories"] != categories:
        raise ValueError("model split COCO categories differ")
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    seen_frames: set[tuple[str, int]] = set()
    for document in documents:
        image_ids: dict[int, int] = {}
        for image in document["images"]:
            if not isinstance(image, dict):
                raise ValueError("COCO image entry must be an object")
            old_id = image.get("id")
            session_id, frame_id = image.get("session_id"), image.get("frame_id")
            if type(old_id) is not int or not isinstance(session_id, str) or type(frame_id) is not int:
                raise ValueError("COCO image lacks id, session_id, or frame_id")
            key = (session_id, frame_id)
            if key in seen_frames:
                raise ValueError(f"duplicate model frame: {session_id}/{frame_id}")
            seen_frames.add(key)
            new_id = len(images) + 1
            image_ids[old_id] = new_id
            images.append({**image, "id": new_id})
        for annotation in document["annotations"]:
            if not isinstance(annotation, dict):
                raise ValueError("COCO annotation entry must be an object")
            old_image_id = annotation.get("image_id")
            if type(old_image_id) is not int or old_image_id not in image_ids:
                raise ValueError("COCO annotation references a missing image")
            annotations.append({
                **annotation,
                "id": len(annotations) + 1,
                "image_id": image_ids[old_image_id],
            })
    return {"images": images, "annotations": annotations, "categories": categories}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """出力 JSON を同一 directory の temp file から atomic replace する。

    中断時に半分だけ書かれた COCO file が正式名で残るのを防ぐ。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def main(argv: list[str] | None = None) -> int:
    """world/UI の model split COCO を所定名で出力する。

    全入力の検査後にだけ2ファイルを置き、エラーを stderr に返す。
    """
    args = _parser().parse_args(argv)
    try:
        world = _merge((args.world_train, args.world_validation))
        ui = _merge((args.ui_train, args.ui_validation))
        world_path, ui_path = args.output_dir / "world_coco_model.json", args.output_dir / "ui_coco_model.json"
        if world_path.exists() or ui_path.exists():
            raise ValueError("model export output already exists")
        _write_json(world_path, world)
        _write_json(ui_path, ui)
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
