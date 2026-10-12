#!/usr/bin/env python3
"""自己一致の二回目 annotation 用に空の work-root を準備する。

確認済み frame を seed 固定で選び、画像だけの B root と blind manifest を作る。
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile

from survivors.annotation_labels import iter_frame_files, read_label_file


def prepare_self_agreement(
    work_root_a: str | Path,
    work_root_b: str | Path,
    *,
    count: int = 300,
    seed: int = 0,
) -> dict:
    """checked frame を厳密に N 枚選び、画像のみを B root へ配置する。

    既存の B 内容を避け、staging directory から manifest と画像をまとめて切り替える。
    """
    if type(count) is not int or count < 1:
        raise ValueError("count must be a positive integer")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    root_a = Path(work_root_a).expanduser().resolve()
    root_b = Path(work_root_b).expanduser().resolve()
    if not root_a.is_dir():
        raise ValueError("work-root-a must be an existing directory")
    if root_a == root_b or root_a in root_b.parents or root_b in root_a.parents:
        raise ValueError("work-root-a and work-root-b must be separate roots")
    if root_b.exists() and (not root_b.is_dir() or any(root_b.iterdir())):
        raise ValueError("work-root-b already contains files")

    candidates = []
    for session_dir in sorted(root_a.iterdir()):
        if not session_dir.is_dir() or session_dir.is_symlink():
            continue
        for frame_id, image_path, label_path in iter_frame_files(session_dir):
            if label_path is None or not read_label_file(label_path)[1]:
                continue
            if image_path is None or image_path.is_symlink():
                raise ValueError(f"checked frame image is missing or is a symlink: {session_dir.name}/{frame_id:08d}")
            candidates.append((session_dir.name, frame_id, image_path))
    if len(candidates) < count:
        raise ValueError(f"need {count} checked frames; found {len(candidates)}")

    selected = sorted(random.Random(seed).sample(candidates, count), key=lambda row: (row[0], row[1]))
    manifest = {
        "schema_version": "survivors_self_agreement.v1",
        "source_work_root": str(root_a),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "selected_frames": [
            {"session_id": session_id, "frame_id": frame_id,
             "image_file": image_path.name}
            for session_id, frame_id, image_path in selected
        ],
        "blind": True,
    }
    root_b.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{root_b.name}.", dir=root_b.parent))
    try:
        for session_id, _, image_path in selected:
            session_target = staging / session_id
            session_target.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(image_path, session_target / image_path.name)
        (staging / "self_agreement_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if root_b.exists():
            root_b.rmdir()
        os.replace(staging, root_b)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def _parser() -> argparse.ArgumentParser:
    """A root、空の B root、seed、選択枚数を受け取る。

    B 側にラベルを複製しない用途をコマンド説明に明示する。
    """
    parser = argparse.ArgumentParser(description="Prepare a blind Survivors self-agreement work-root.")
    parser.add_argument("--work-root-a", required=True, type=Path)
    parser.add_argument("--work-root-b", required=True, type=Path)
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    """自己一致用 B root を作り、選択枚数と保存先を表示する。

    入力不備や filesystem error は stderr と終了コード 1 で通知する。
    """
    args = _parser().parse_args(argv)
    try:
        manifest = prepare_self_agreement(
            args.work_root_a, args.work_root_b, count=args.count, seed=args.seed)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"work_root_b": str(args.work_root_b),
                      "selected_frame_count": len(manifest["selected_frames"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
