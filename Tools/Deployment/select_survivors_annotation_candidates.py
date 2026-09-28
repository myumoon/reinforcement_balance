#!/usr/bin/env python3
"""公開キャプチャからアノテーション候補画像を選ぶ CLI。

検証済み manifest の foreground フレームを入力にする。
選んだ候補画像とクラス一覧を作業ディレクトリへ配置する。
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np

from survivors.annotation_labels import ALL_CLASSES
from survivors.capture_dataset import DatasetWriter, FrameRecord


def _parser() -> argparse.ArgumentParser:
    """候補抽出 CLI の引数を定義する。

    保存元、セッション、出力先と抽出量を受け取る。
    既定値は作業ディレクトリの規約に合わせる。
    """
    parser = argparse.ArgumentParser(description="Select Survivors annotation candidate frames.")
    parser.add_argument("--store-root", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--work-root")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--stride", type=int, default=15)
    return parser


def _select_records(
    records: tuple[FrameRecord, ...],
    session_path: Path,
    *,
    count: int,
    stride: int,
) -> tuple[int, int, int, list[FrameRecord]]:
    """foreground フレームから視覚的に分散した候補を選ぶ。

    frame_id 昇順で hash 重複を除き、stride 間引きを行う。
    縮小画像の k-means は候補数が上限を超えた場合だけ適用する。
    """
    foreground = sorted(
        (record for record in records if record.foreground),
        key=lambda record: record.frame_id,
    )
    unique: list[FrameRecord] = []
    seen_hashes: set[str] = set()
    for record in foreground:
        if record.object_sha256 not in seen_hashes:
            unique.append(record)
            seen_hashes.add(record.object_sha256)

    thinned = unique[::stride]
    features: list[np.ndarray] = []
    for record in thinned:
        source = session_path / record.object_path
        image = cv2.imread(str(source))
        if image is None:
            raise ValueError(f"cannot read frame image: {source}")
        resized = cv2.resize(image, (64, 36), interpolation=cv2.INTER_AREA)
        features.append(resized.astype(np.float32).reshape(-1) / 255.0)

    if len(thinned) <= count:
        return len(foreground), len(unique), len(thinned), thinned

    samples = np.stack(features)
    cv2.setRNGSeed(0)
    _, labels, centers = cv2.kmeans(
        samples,
        count,
        None,
        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
        1,
        cv2.KMEANS_PP_CENTERS,
    )
    chosen: list[FrameRecord] = []
    for cluster_id, center in enumerate(centers):
        members = np.flatnonzero(labels.ravel() == cluster_id)
        distances = np.sum((samples[members] - center) ** 2, axis=1)
        chosen.append(thinned[int(members[int(np.argmin(distances))])])
    chosen.sort(key=lambda record: record.frame_id)
    return len(foreground), len(unique), len(thinned), chosen


def _place_candidate(source: Path, destination: Path) -> bool:
    """候補画像を既存ファイルを守って配置する。

    hard link を試し、使えない場合は新規ファイルへコピーする。
    既存の候補は触らず、作成できた場合だけ真を返す。
    """
    if destination.exists():
        return False
    try:
        os.link(source, destination)
    except FileExistsError:
        return False
    except OSError:
        try:
            with destination.open("xb"):
                pass
        except FileExistsError:
            return False
        try:
            shutil.copy2(source, destination)
        except Exception:
            destination.unlink(missing_ok=True)
            raise
    return True


def main(argv: list[str] | None = None) -> int:
    """検証済みキャプチャから候補 PNG とクラス一覧を出力する。

    既存 PNG は保持し、ラベル JSON には触れずに件数を表示する。
    入力や画像を読めない場合はエラーを表示して終了コード 1 を返す。
    """
    parser = _parser()
    args = parser.parse_args(argv)
    if args.count < 1 or args.stride < 1:
        parser.error("--count and --stride must be positive")

    store_root = Path(args.store_root)
    work_root = Path(args.work_root) if args.work_root else store_root / "annotation_work"
    try:
        manifest = DatasetWriter.restore(store_root, args.session_id, metadata_only=True)
        foreground_count, deduplicated_count, thinned_count, selected = _select_records(
            manifest.frame_records,
            manifest.session_path,
            count=args.count,
            stride=args.stride,
        )

        output_dir = work_root / manifest.session_id
        output_dir.mkdir(parents=True, exist_ok=True)
        newly_placed = 0
        for record in selected:
            source = manifest.session_path / record.object_path
            destination = output_dir / f"{record.frame_id:08d}.png"
            newly_placed += _place_candidate(source, destination)

        work_root.mkdir(parents=True, exist_ok=True)
        (work_root / "classes.txt").write_text("\n".join(ALL_CLASSES) + "\n", encoding="utf-8")
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"総数: {len(manifest.frame_records)}")
    print(f"foreground: {foreground_count}")
    print(f"重複除去後: {deduplicated_count}")
    print(f"間引き後: {thinned_count}")
    print(f"選択数: {len(selected)}")
    print(f"新規配置数: {newly_placed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
