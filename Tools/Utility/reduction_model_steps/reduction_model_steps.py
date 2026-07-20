#!/usr/bin/env python3
"""
model_steps フォルダ内の model_<steps>_steps.zip を間引く。

REDUCTION_STEPS の各倍数境界に最も近いファイルを1つ残し、
最新（最大ステップ）ファイルも常に残す。それ以外を削除する。

Usage:
    python reduction_model_steps.py --dir <model_steps_folder> [--reduction-steps N] [--yes]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_FILENAME_RE = re.compile(r"^model_(\d+)_steps\.zip$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="model_steps フォルダの zip を間引く"
    )
    parser.add_argument("--dir", required=True, help="model_steps フォルダのパス")
    parser.add_argument(
        "--reduction-steps",
        type=int,
        default=1_000_000,
        help="残す間隔（デフォルト: 1000000）",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="確認プロンプトをスキップして即削除する",
    )
    return parser.parse_args()


def collect_files(folder: Path) -> list[tuple[int, Path]]:
    """folder 直下の model_*_steps.zip を (steps, path) のリストで返す（昇順）。"""
    entries: list[tuple[int, Path]] = []
    for p in folder.iterdir():
        if not p.is_file():
            continue
        m = _FILENAME_RE.match(p.name)
        if m:
            entries.append((int(m.group(1)), p))
    entries.sort(key=lambda x: x[0])
    return entries


def select_keep(
    entries: list[tuple[int, Path]], reduction_steps: int
) -> set[Path]:
    """残すファイルのパス集合を返す。"""
    if not entries:
        return set()

    steps_list = [s for s, _ in entries]
    min_step, max_step = steps_list[0], steps_list[-1]

    keep: set[Path] = set()

    # 最新ファイルは必ず残す
    keep.add(entries[-1][1])

    # REDUCTION_STEPS の各倍数に最も近いファイルを選ぶ
    first_k = (min_step // reduction_steps) + 1
    last_k = max_step // reduction_steps

    for k in range(first_k, last_k + 1):
        boundary = k * reduction_steps
        # boundary に最も近いエントリを線形探索（件数が数百程度なのでOK）
        closest = min(entries, key=lambda x: abs(x[0] - boundary))
        keep.add(closest[1])

    return keep


def format_size(size_bytes: int) -> str:
    if size_bytes >= 1_073_741_824:
        return f"{size_bytes / 1_073_741_824:.2f} GB"
    return f"{size_bytes / 1_048_576:.1f} MB"


def main() -> None:
    args = parse_args()
    folder = Path(args.dir)

    if not folder.is_dir():
        print(f"[ERROR] フォルダが見つかりません: {folder}", file=sys.stderr)
        sys.exit(1)

    entries = collect_files(folder)
    if len(entries) <= 1:
        print(f"[INFO] 対象ファイルが {len(entries)} 件のため間引き不要です。")
        return

    keep = select_keep(entries, args.reduction_steps)
    delete_entries = [(s, p) for s, p in entries if p not in keep]

    keep_count = len(keep)
    delete_count = len(delete_entries)
    delete_size = sum(p.stat().st_size for _, p in delete_entries)

    print(f"\n--- 間引き計画 (REDUCTION_STEPS={args.reduction_steps:,}) ---")
    print(f"  対象ファイル数  : {len(entries)}")
    print(f"  残すファイル数  : {keep_count}")
    print(f"  削除ファイル数  : {delete_count}")
    print(f"  削減サイズ      : {format_size(delete_size)}")

    if delete_count == 0:
        print("\n[INFO] 削除対象がありません。")
        return

    print("\n[削除対象ファイル]")
    for _, p in delete_entries:
        print(f"  {p.name}")

    print()
    if not args.yes:
        answer = input("続行しますか? [y/N] ").strip().lower()
        if answer != "y":
            print("[INFO] キャンセルしました。")
            return

    deleted = 0
    for _, p in delete_entries:
        try:
            p.unlink()
            deleted += 1
        except OSError as e:
            print(f"[WARN] 削除失敗: {p.name} ({e})", file=sys.stderr)

    print(f"\n[INFO] {deleted} 件削除しました（{format_size(delete_size)} 解放）。")


if __name__ == "__main__":
    main()
