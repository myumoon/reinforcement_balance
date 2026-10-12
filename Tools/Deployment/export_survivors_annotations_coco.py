"""確認済み Survivors ラベルを world/ui COCO dataset に出力する。

work-root 内の八桁 frame ファイルを読み、WorldDataset と互換な JSON を作る。
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import getpass
import json
import os
import sys
from pathlib import Path

import cv2

from survivors.annotation_labels import (
    ANNOTATION_ONLY_CLASSES,
    DEFAULT_CLASS_MAP_PATH,
    REGION_LABEL,
    UI_CLASSES,
    WORLD_CLASSES,
    clip_box,
    iter_frame_files,
    read_label_file,
)
from survivors.vision.world_dataset import load_class_map
from survivors.capture_dataset import SPLIT_NAMES, DatasetWriter, _split_manifest_from_bytes


def _parser() -> argparse.ArgumentParser:
    """COCO 出力 CLI の引数 parser を返す。

    work-root と output-dir は必須、session-id は複数回指定できる。
    """
    parser = argparse.ArgumentParser(description="Export checked Survivors labels as COCO datasets.")
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--session-id", action="append", default=None)
    parser.add_argument("--class-map", type=Path, default=DEFAULT_CLASS_MAP_PATH)
    parser.add_argument("--store-root", type=Path)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--split", choices=SPLIT_NAMES)
    return parser


def _image_file_name(image_path: Path, output_dir: Path) -> str:
    """COCO JSON 親から画像への相対パスを作る。

    別ドライブなどで相対化できない場合は絶対パスを保持する。
    """
    try:
        return os.path.relpath(image_path.resolve(), output_dir.resolve())
    except ValueError:
        return str(image_path.resolve())


def _box_signature(boxes) -> list[tuple[str, float, float, float, float]]:
    """label と矩形座標だけを比較用の安定した列にする。

    検出スコアや JSON 内の shape 順序は annotation の修正判定に含めない。
    """
    return sorted((box.label, box.left, box.top, box.right, box.bottom) for box in boxes)


def _update_session_log_annotation_hours(store_root: Path, work_root: Path, session_id: str) -> None:
    """checked label の工数を session log の work-root 行へ反映する。

    log の欠落や工数情報の読み取り失敗は警告に留め、COCO export を成功させる。
    """
    session_log = store_root / "session_logs" / session_id / "session_log.md"
    if not session_log.is_file():
        print(f"warning: missing session log: {session_log}", file=sys.stderr)
        return
    try:
        session_dir = work_root / session_id
        checked_labels = []
        for _, _, label_path in iter_frame_files(session_dir):
            if label_path is None:
                continue
            boxes, checked = read_label_file(label_path)
            if checked:
                checked_labels.append((label_path, boxes))

        modified = no_draft = 0
        for label_path, boxes in checked_labels:
            draft_path = session_dir / ".prelabel_drafts" / label_path.name
            if not draft_path.is_file():
                no_draft += 1
            elif _box_signature(boxes) != _box_signature(read_label_file(draft_path)[0]):
                modified += 1
        if checked_labels:
            start = min(path.stat().st_mtime for path, _ in checked_labels)
            end = max(path.stat().st_mtime for path, _ in checked_labels)
            started_at = datetime.fromtimestamp(start, timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
            ended_at = datetime.fromtimestamp(end, timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        else:
            started_at = ended_at = "unknown"
        try:
            annotator = getpass.getuser() or "unknown"
        except Exception:
            annotator = "unknown"

        root_path = str(work_root.resolve())
        row = (
            f"| {root_path} | {annotator} | {started_at} | {ended_at} | {len(checked_labels)} | "
            f"{modified} | {no_draft} |"
        )
        lines = session_log.read_text(encoding="utf-8").splitlines()
        heading = next(
            index for index, line in enumerate(lines)
            if line.strip() in {"## annotation 工数", "### annotation 工数"}
        )
        table_start = next(
            (index for index in range(heading + 1, len(lines)) if lines[index].startswith("| ")),
            heading + 1,
        )
        table_end = table_start
        while table_end < len(lines) and lines[table_end].startswith("| "):
            table_end += 1
        normalized_root = os.path.normcase(os.path.abspath(root_path))
        existing_rows = []
        for existing in lines[table_start + 2:table_end]:
            cells = [cell.strip() for cell in existing.strip("|").split("|")]
            if len(cells) == 7 and os.path.normcase(os.path.abspath(cells[0])) != normalized_root:
                existing_rows.append(existing)
        table = [
            "| work-root | annotator | 開始時刻 | 終了時刻 | checked frame 数 | 下書きから修正した frame 数 | 下書きなし frame 数 |",
            "|---|---|---|---|---:|---:|---:|",
            *existing_rows,
            row,
        ]
        lines[table_start:table_end] = table
        session_log.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    except Exception as exc:
        print(f"warning: failed to update annotation hours in {session_log}: {exc}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    """main(argv) -> int として確認済みラベルを world/ui COCO へ出力する。

    採用画像と矩形数を表示し、入力が不正な場合は stderr と終了コード 1 を返す。
    """
    args = _parser().parse_args(argv)
    try:
        class_map = load_class_map(args.class_map)
        if (args.split_manifest is None) != (args.split is None):
            raise ValueError("--split-manifest and --split must be specified together")
        split_sessions = None
        if args.split_manifest is not None:
            split_bytes = args.split_manifest.read_bytes()
            split = _split_manifest_from_bytes(args.split_manifest, split_bytes)
            split_sessions = {unit.session_id for unit in split.splits[args.split]}
        if args.session_id is None:
            session_ids = sorted(
                path.name for path in args.work_root.iterdir() if path.is_dir()
            )
        else:
            session_ids = sorted(set(args.session_id))
        if split_sessions is not None:
            if args.session_id is not None and not set(session_ids).issubset(split_sessions):
                raise ValueError("--session-id includes a session outside the requested split")
            session_ids = sorted(split_sessions if args.session_id is None else set(session_ids))

        frame_hashes: dict[str, dict[int, str]] = {}
        if args.store_root is not None:
            for session_id in session_ids:
                manifest = DatasetWriter.restore(args.store_root, session_id, metadata_only=True)
                frame_hashes[session_id] = {
                    record.frame_id: record.object_sha256 for record in manifest.frame_records
                }

        world_categories = [
            {
                "id": item["id"],
                "name": item["name"],
                "supercategory": item["coarse_category"],
            }
            for item in sorted(class_map.foreground_classes, key=lambda item: item["id"])
        ]
        ui_categories = [
            {"id": index + 1, "name": name, "supercategory": "ui"}
            for index, name in enumerate(UI_CLASSES)
        ]
        world_images: list[dict] = []
        ui_images: list[dict] = []
        world_annotations: list[dict] = []
        ui_annotations: list[dict] = []
        skipped_unconfirmed = 0
        skipped_region = 0
        skipped_annotation_only = 0

        for session_id in session_ids:
            session_dir = args.work_root / session_id
            for frame_id, image_path, label_path in iter_frame_files(session_dir):
                if label_path is None:
                    continue
                boxes, checked = read_label_file(label_path)
                if not checked:
                    skipped_unconfirmed += 1
                    continue
                # 範囲限定ラベルのフレームは範囲外に未ラベルの物体が残るので全面 dataset に入れない。
                if any(box.label == REGION_LABEL for box in boxes):
                    skipped_region += 1
                    continue
                if image_path is None:
                    expected_image = session_dir / f"{frame_id:08d}.png"
                    raise ValueError(f"{label_path}: checked label is missing image: {expected_image}")

                image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
                if image is None:
                    raise ValueError(f"{image_path}: cannot read image")
                image_height, image_width = image.shape[:2]
                image_id = len(world_images) + 1
                image_entry = {
                    "id": image_id,
                    "file_name": _image_file_name(image_path, args.output_dir),
                    "width": image_width,
                    "height": image_height,
                    "session_id": session_id,
                    "frame_id": frame_id,
                }
                if args.store_root is not None:
                    try:
                        image_entry["object_sha256"] = frame_hashes[session_id][frame_id]
                    except KeyError as exc:
                        raise ValueError(
                            f"label references a frame absent from capture store: {session_id}/{frame_id}"
                        ) from exc
                world_images.append(image_entry)
                ui_images.append(image_entry.copy())

                for box in boxes:
                    # アノテーション専用クラス（照準など）は world/ui どちらの COCO にも入れない。
                    if box.label in ANNOTATION_ONLY_CLASSES:
                        skipped_annotation_only += 1
                        continue
                    clipped = clip_box(box, image_width=image_width, image_height=image_height)
                    if clipped is None:
                        print(
                            f"warning: {label_path}: {box.label} 矩形を除外 (clip 後の面積0)",
                            file=sys.stderr,
                        )
                        continue
                    width = clipped.right - clipped.left
                    height = clipped.bottom - clipped.top
                    if clipped.label in WORLD_CLASSES:
                        annotations = world_annotations
                        category_id = class_map.name_to_id(clipped.label)
                    else:
                        annotations = ui_annotations
                        category_id = UI_CLASSES.index(clipped.label) + 1
                    annotations.append(
                        {
                            "id": len(annotations) + 1,
                            "image_id": image_id,
                            "category_id": category_id,
                            "bbox": [clipped.left, clipped.top, width, height],
                            "area": width * height,
                            "iscrowd": 0,
                            "session_id": session_id,
                        }
                    )

        args.output_dir.mkdir(parents=True, exist_ok=True)
        for filename, categories, images, annotations in (
            ("world_coco.json", world_categories, world_images, world_annotations),
            ("ui_coco.json", ui_categories, ui_images, ui_annotations),
        ):
            document = {
                "images": images,
                "annotations": annotations,
                "categories": categories,
            }
            (args.output_dir / filename).write_text(
                json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )

        if args.store_root is not None:
            for session_id in session_ids:
                _update_session_log_annotation_hours(args.store_root, args.work_root, session_id)

        print(f"採用画像数: {len(world_images)}")
        print(f"未確認でスキップした数: {skipped_unconfirmed}")
        print(f"範囲限定でスキップした数: {skipped_region}")
        print(f"アノテーション専用で除外した矩形数: {skipped_annotation_only}")
        print(f"world 矩形数: {len(world_annotations)}")
        print(f"ui 矩形数: {len(ui_annotations)}")
        return 0
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
