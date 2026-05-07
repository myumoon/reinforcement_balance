"""SAM2 を使って recordings/ のフレームに疑似マスクラベルを付与するスケルトン。

役割:
    Tools/Perception/recordings/ 配下の動画/画像を入力に、SAM2 推論で
    インスタンスマスクを生成し Tools/Perception/labels/ に書き出す。

セットアップ:
    1. SAM2 公式リポジトリからチェックポイントを落として
       Tools/Perception/sam2_weights/ に配置する。
    2. 依存パッケージは Tools/Perception/requirements_perception.txt を
       インストールしたあと、SAM2 本体を別途インストールする:
           pip install "git+https://github.com/facebookresearch/sam2.git@<commit_sha>"

使い方（実装後のイメージ）:
    python -m Tools.Perception.scripts.annotate_with_sam2 \
        --recordings Tools/Perception/recordings \
        --labels Tools/Perception/labels \
        --checkpoint Tools/Perception/sam2_weights/sam2_hiera_large.pt
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SAM2 でフレームに疑似ラベルを付与する")
    p.add_argument("--recordings", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True, help="SAM2 重みファイル")
    p.add_argument("--config", type=str, default=None, help="SAM2 構成名（例: sam2_hiera_l）")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--prompt-mode", type=str, default="auto",
                   choices=("auto", "point", "box"),
                   help="マスク生成のプロンプト方式")
    p.add_argument("--max-frames", type=int, default=None)
    return p


def annotate(
    recordings_dir: Path,
    labels_dir: Path,
    checkpoint: Path,
    config: Optional[str] = None,
    device: str = "cuda",
    prompt_mode: str = "auto",
    max_frames: Optional[int] = None,
) -> None:
    """recordings_dir 内のフレームに対し SAM2 でマスクを生成し labels_dir に保存する。

    実装方針（後続フェーズで埋める）:
        1. SAM2 チェックポイントと config から predictor を構築する。
        2. recordings_dir を再帰的に走査し、画像/動画を列挙する。
        3. 各フレームに prompt_mode に従ってプロンプトを与えてマスクを得る。
        4. labels_dir 配下に PNG（マスク）+ JSON（メタ情報）として保存する。
    """
    raise NotImplementedError("annotate_with_sam2 is a skeleton")


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _build_arg_parser().parse_args(argv)
    annotate(
        recordings_dir=args.recordings,
        labels_dir=args.labels,
        checkpoint=args.checkpoint,
        config=args.config,
        device=args.device,
        prompt_mode=args.prompt_mode,
        max_frames=args.max_frames,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
