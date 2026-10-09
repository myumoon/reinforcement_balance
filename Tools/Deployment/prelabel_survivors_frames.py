#!/usr/bin/env python3
"""候補フレームへ固定矩形と下書き用検出器の推論結果を下書きとして付ける。

学習 CLI（train_survivors_prelabel_detector.py）が保存した重みで敵・ジェム等を検出し、
未作成のラベルファイルを生成する。重みが無ければ固定矩形だけで下書きを作る。
`--refresh-unchecked` を付けると、未チェックの既存下書きも最新の検出器で
再生成する（チェック済みのファイルは変更しない）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from survivors.annotation_labels import (
    clip_box,
    iter_frame_files,
    read_label_file,
    write_label_file,
)
from survivors.prelabel_detector import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_WEIGHTS_NAME,
    default_device,
    drop_ignored,
    load_config,
    load_detector,
    predict_boxes,
)


def _read_image(path: Path) -> np.ndarray:
    """OpenCV で画像を読み、失敗時は対象パスを示して止める。

    壊れた画像から空の下書きを作らない。
    """
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"{path}: cannot read image")
    return image


def _parser() -> argparse.ArgumentParser:
    """下書き CLI の work-root、対象 session、設定ファイル、検出器重みの引数を定義する。

    既定設定は同梱 v2 YAML、既定の重みは `<work-root>/prelabel_detector.pt` を使う。
    session ID は呼び出し側が指定する。
    """
    parser = argparse.ArgumentParser(description="Create draft labels for Survivors candidate frames.")
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--detector",
        type=Path,
        default=None,
        help=f"Detector weights (default: <work-root>/{DEFAULT_WEIGHTS_NAME}).",
    )
    parser.add_argument(
        "--refresh-unchecked",
        action="store_true",
        help="Regenerate draft boxes for existing unchecked labels using the latest detector.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """対象セッションの未ラベル PNG に下書きを書き、件数を表示する。

    JSON が無いフレームには新規下書きを作る。`--refresh-unchecked` を付けると、
    既存 JSON のうち `checked: false` のものだけ最新の検出器で再生成し、
    `checked: true` のものは変更せず残す。重みファイルが無ければ警告して固定矩形のみを書く。
    """
    args = _parser().parse_args(argv)
    if not args.session_id or Path(args.session_id).name != args.session_id or args.session_id in {".", ".."}:
        print("error: session-id must be a single directory name", file=sys.stderr)
        return 1
    detector_path = args.detector or args.work_root / DEFAULT_WEIGHTS_NAME
    try:
        config = load_config(args.config)
        device = default_device()
        model = None
        if detector_path.is_file():
            model = load_detector(detector_path, config.detector, device=device)
        else:
            print(
                f"warning: {detector_path}: detector weights not found; drafting fixed boxes only",
                file=sys.stderr,
            )
        session_dir = args.work_root / args.session_id
        created = skipped = 0
        for _, png_path, json_path in iter_frame_files(session_dir):
            if png_path is None:
                continue
            refresh = False
            if json_path is not None:
                if not args.refresh_unchecked:
                    skipped += 1
                    continue
                _, checked = read_label_file(json_path)
                if checked:
                    skipped += 1
                    continue
                refresh = True
            image = _read_image(png_path)
            height, width = image.shape[:2]
            boxes = [
                clipped
                for box in config.fixed_boxes
                if (clipped := clip_box(box, image_width=width, image_height=height)) is not None
            ]
            if model is not None:
                boxes.extend(drop_ignored(predict_boxes(model, image, config.detector, device=device), config.ignore_regions))
            wrote_label = write_label_file(
                json_path or png_path.with_suffix(".json"),
                boxes,
                image_width=width,
                image_height=height,
                checked=False,
                overwrite=refresh,
            )
            if not wrote_label:
                skipped += 1
                continue
            created += 1
    except (OSError, ValueError, RuntimeError, cv2.error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"検出器: {detector_path}" if model is not None else "検出器: なし（固定矩形のみ）")
    print(f"下書き作成数: {created}")
    print(f"既存 JSON スキップ数: {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
