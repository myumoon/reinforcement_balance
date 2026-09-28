"""確認済み Survivors ラベルを world/ui COCO dataset に出力する。

work-root 内の八桁 frame ファイルを読み、WorldDataset と互換な JSON を作る。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2

from survivors.annotation_labels import (
    DEFAULT_CLASS_MAP_PATH,
    UI_CLASSES,
    WORLD_CLASSES,
    clip_box,
    iter_frame_files,
    read_label_file,
)
from survivors.vision.world_dataset import load_class_map


def _parser() -> argparse.ArgumentParser:
    """COCO 出力 CLI の引数 parser を返す。

    work-root と output-dir は必須、session-id は複数回指定できる。
    """
    parser = argparse.ArgumentParser(description="Export checked Survivors labels as COCO datasets.")
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--session-id", action="append", default=None)
    parser.add_argument("--class-map", type=Path, default=DEFAULT_CLASS_MAP_PATH)
    return parser


def _image_file_name(image_path: Path, output_dir: Path) -> str:
    """COCO JSON 親から画像への相対パスを作る。

    別ドライブなどで相対化できない場合は絶対パスを保持する。
    """
    try:
        return os.path.relpath(image_path.resolve(), output_dir.resolve())
    except ValueError:
        return str(image_path.resolve())


def main(argv: list[str] | None = None) -> int:
    """main(argv) -> int として確認済みラベルを world/ui COCO へ出力する。

    採用画像と矩形数を表示し、入力が不正な場合は stderr と終了コード 1 を返す。
    """
    args = _parser().parse_args(argv)
    try:
        class_map = load_class_map(args.class_map)
        if args.session_id is None:
            session_ids = sorted(
                path.name for path in args.work_root.iterdir() if path.is_dir()
            )
        else:
            session_ids = sorted(set(args.session_id))

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

        for session_id in session_ids:
            session_dir = args.work_root / session_id
            for frame_id, image_path, label_path in iter_frame_files(session_dir):
                if label_path is None:
                    continue
                boxes, checked = read_label_file(label_path)
                if not checked:
                    skipped_unconfirmed += 1
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
                world_images.append(image_entry)
                ui_images.append(image_entry.copy())

                for box in boxes:
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

        print(f"採用画像数: {len(world_images)}")
        print(f"未確認でスキップした数: {skipped_unconfirmed}")
        print(f"world 矩形数: {len(world_annotations)}")
        print(f"ui 矩形数: {len(ui_annotations)}")
        return 0
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
