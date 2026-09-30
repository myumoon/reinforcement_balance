#!/usr/bin/env python3
"""確認済みラベルから下書き用検出器を学習し、重みファイルを保存する CLI。

work-root 内の `checked: true` の JSON と PNG を集めて Faster R-CNN を学習する。
保存した重みは prelabel_survivors_frames.py が自動下書きに使う。
初回は torchvision の事前学習重み（約 74MB）をダウンロードする。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from survivors.prelabel_detector import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_WEIGHTS_NAME,
    collect_training_frames,
    default_device,
    load_config,
    save_detector,
    train_detector,
)


def _parser() -> argparse.ArgumentParser:
    """学習 CLI の work-root、設定ファイル、保存先の引数を定義する。

    既定設定は同梱 v2 YAML、既定の保存先は `<work-root>/prelabel_detector.pt`。
    """
    parser = argparse.ArgumentParser(description="Train the Survivors prelabel detector from checked labels.")
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Output weights path (default: <work-root>/{DEFAULT_WEIGHTS_NAME}).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """確認済みフレームを集めて検出器を学習し、重みを保存して件数と保存先を表示する。

    学習に使えるフレームが1枚も無い、または設定・画像・学習で失敗した場合は
    `error:` を表示して終了コード 1 を返す。
    """
    args = _parser().parse_args(argv)
    output = args.output or args.work_root / DEFAULT_WEIGHTS_NAME
    try:
        config = load_config(args.config)
        frames = collect_training_frames(args.work_root, config.detector)
        if not frames:
            print(f"error: {args.work_root}: no checked frames to train on", file=sys.stderr)
            return 1
        model = train_detector(frames, config.detector, device=default_device())
        save_detector(model, output, config.detector)
    except (OSError, ValueError, RuntimeError, cv2.error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"学習フレーム数: {len(frames)}")
    print(f"保存先: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
