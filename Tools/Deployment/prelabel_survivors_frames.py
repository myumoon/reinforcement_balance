#!/usr/bin/env python3
"""候補フレームへ固定矩形と見本照合の下書きを付ける。

確認済みラベルから見本を集め、未作成のラベルファイルだけを生成する。
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

from survivors.annotation_labels import (
    LabelBox,
    clip_box,
    iter_frame_files,
    read_label_file,
    validate_label,
    write_label_file,
)


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "annotation_prelabel_v1.yaml"
_IMAGE_WIDTH = 1920
_IMAGE_HEIGHT = 1080


def load_config(path: Path | str) -> dict:
    """下書き設定を読み込み、キー・ラベル・値の範囲を検証する。

    使えるラベルは共有一覧を参照し、不正な設定では処理を始めない。
    """
    path = Path(path)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"{path}: cannot read config: {exc}") from exc
    if not isinstance(data, dict) or set(data) != {"schema_version", "fixed_boxes", "template_matching"}:
        raise ValueError(f"{path}: expected schema_version, fixed_boxes, and template_matching")
    if data["schema_version"] != "annotation_prelabel.v1":
        raise ValueError(f"{path}: unsupported schema_version: {data['schema_version']!r}")

    if not isinstance(data["fixed_boxes"], list):
        raise ValueError(f"{path}: fixed_boxes must be a list")
    fixed_boxes = []
    for index, item in enumerate(data["fixed_boxes"]):
        if not isinstance(item, dict) or set(item) != {"label", "bbox"}:
            raise ValueError(f"{path}: fixed_boxes[{index}] must contain only label and bbox")
        label, bbox = item["label"], item["bbox"]
        if not isinstance(label, str):
            raise ValueError(f"{path}: fixed_boxes[{index}].label must be a string")
        try:
            validate_label(label)
        except ValueError as exc:
            raise ValueError(f"{path}: fixed_boxes[{index}]: {exc}") from exc
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in bbox)
            or not all(math.isfinite(float(value)) for value in bbox)
        ):
            raise ValueError(f"{path}: fixed_boxes[{index}].bbox must contain four finite numbers")
        left, top, right, bottom = (float(value) for value in bbox)
        if left < 0 or top < 0 or right <= left or bottom <= top or right > _IMAGE_WIDTH or bottom > _IMAGE_HEIGHT:
            raise ValueError(f"{path}: fixed_boxes[{index}].bbox is outside 1920x1080 image bounds")
        fixed_boxes.append(LabelBox(label, left, top, right, bottom))

    matching = data["template_matching"]
    if not isinstance(matching, dict) or set(matching) != {
        "labels", "threshold", "nms_iou", "max_templates_per_label"
    }:
        raise ValueError(f"{path}: template_matching has missing or unknown keys")
    labels = matching["labels"]
    if not isinstance(labels, list) or any(not isinstance(label, str) for label in labels):
        raise ValueError(f"{path}: template_matching.labels must be a list of strings")
    for label in labels:
        try:
            validate_label(label)
        except ValueError as exc:
            raise ValueError(f"{path}: template_matching.labels: {exc}") from exc

    values = {}
    for key in ("threshold", "nms_iou"):
        value = matching[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"{path}: template_matching.{key} must be a finite number")
        if not 0 < float(value) <= 1:
            raise ValueError(f"{path}: template_matching.{key} must be in (0, 1]")
        values[key] = float(value)
    max_templates = matching["max_templates_per_label"]
    if isinstance(max_templates, bool) or not isinstance(max_templates, int) or max_templates <= 0:
        raise ValueError(f"{path}: max_templates_per_label must be a positive integer")

    return {
        "fixed_boxes": fixed_boxes,
        "labels": labels,
        "threshold": values["threshold"],
        "nms_iou": values["nms_iou"],
        "max_templates_per_label": max_templates,
    }


def _read_image(path: Path) -> np.ndarray:
    """OpenCV で画像を読み、失敗時は対象パスを示して止める。

    壊れた画像から空の下書きを作らない。
    """
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"{path}: cannot read image")
    return image


def collect_templates(work_root: Path | str, labels: list[str], max_per_label: int) -> dict[str, list[np.ndarray]]:
    """work-root 内の確認済み矩形を、決定的な順で画像見本へ変換する。

    セッション、フレーム、shape の順に走査し、各ラベルの上限数で止める。
    """
    templates = {label: [] for label in labels}
    work_root = Path(work_root)
    for session_dir in sorted(path for path in work_root.iterdir() if path.is_dir()):
        for _, png_path, json_path in iter_frame_files(session_dir):
            if json_path is None:
                continue
            boxes, checked = read_label_file(json_path)
            if not checked:
                continue
            if png_path is None:
                raise ValueError(f"{json_path}: checked label has no matching PNG")
            if not any(box.label in templates for box in boxes):
                continue
            image = _read_image(png_path)
            height, width = image.shape[:2]
            for box in boxes:
                if box.label not in templates or len(templates[box.label]) >= max_per_label:
                    continue
                clipped = clip_box(box, image_width=width, image_height=height)
                if clipped is None:
                    continue
                left, top = math.floor(clipped.left), math.floor(clipped.top)
                right, bottom = math.ceil(clipped.right), math.ceil(clipped.bottom)
                templates[box.label].append(image[top:bottom, left:right].copy())
    return templates


def _match_templates(
    image: np.ndarray,
    templates: dict[str, list[np.ndarray]],
    labels: list[str],
    *,
    threshold: float,
    nms_iou: float,
) -> list[LabelBox]:
    """テンプレート照合の一致位置を NMS でまとめた矩形として返す。

    閾値以上の候補だけを残し、同じ物体に重なる検出を一つにまとめる。
    """
    image_height, image_width = image.shape[:2]
    boxes: list[LabelBox] = []
    for label in labels:
        candidates = []
        scores = []
        for template in templates[label]:
            height, width = template.shape[:2]
            if width > image_width or height > image_height:
                continue
            response = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
            ys, xs = np.where(response >= threshold)
            candidates.extend((int(x), int(y), width, height) for x, y in zip(xs, ys))
            scores.extend(float(response[y, x]) for x, y in zip(xs, ys))
        if not candidates:
            continue
        # OpenCV の NMSBoxes は score_threshold より大きい候補を残すため、等値を含める。
        nms_score_threshold = math.nextafter(threshold, -math.inf)
        indices = cv2.dnn.NMSBoxes(candidates, scores, nms_score_threshold, nms_iou)
        for index in np.asarray(indices).reshape(-1):
            left, top, width, height = candidates[int(index)]
            box = clip_box(
                LabelBox(label, left, top, left + width, top + height, scores[int(index)]),
                image_width=image_width,
                image_height=image_height,
            )
            if box is not None:
                boxes.append(box)
    return boxes


def _parser() -> argparse.ArgumentParser:
    """下書き CLI の work-root、対象 session、設定ファイル引数を定義する。

    既定設定は同梱 YAML を使い、session ID は呼び出し側が指定する。
    """
    parser = argparse.ArgumentParser(description="Create draft labels for Survivors candidate frames.")
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser


def main(argv: list[str] | None = None) -> int:
    """対象セッションの未ラベル PNG に下書きを書き、件数を表示する。

    JSON がすでにあるフレームは読み書きせず、そのまま残す。
    """
    args = _parser().parse_args(argv)
    if not args.session_id or Path(args.session_id).name != args.session_id or args.session_id in {".", ".."}:
        print("error: session-id must be a single directory name", file=sys.stderr)
        return 1
    try:
        config = load_config(args.config)
        templates = collect_templates(args.work_root, config["labels"], config["max_templates_per_label"])
        session_dir = args.work_root / args.session_id
        created = skipped = 0
        for _, png_path, json_path in iter_frame_files(session_dir):
            if png_path is None:
                continue
            if json_path is not None:
                skipped += 1
                continue
            image = _read_image(png_path)
            height, width = image.shape[:2]
            boxes = [
                clipped
                for box in config["fixed_boxes"]
                if (clipped := clip_box(box, image_width=width, image_height=height)) is not None
            ]
            boxes.extend(
                _match_templates(
                    image,
                    templates,
                    config["labels"],
                    threshold=config["threshold"],
                    nms_iou=config["nms_iou"],
                )
            )
            write_label_file(
                json_path or png_path.with_suffix(".json"),
                boxes,
                image_width=width,
                image_height=height,
                checked=False,
            )
            created += 1
    except (OSError, ValueError, cv2.error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print("見本数: " + ", ".join(f"{label}={len(templates[label])}" for label in config["labels"]))
    print(f"下書き作成数: {created}")
    print(f"既存 JSON スキップ数: {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
