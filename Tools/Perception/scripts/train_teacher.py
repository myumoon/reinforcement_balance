"""SAM2 疑似ラベルを教師信号に高精度な教師セグメンテーションモデルを訓練するスケルトン。

役割:
    Tools/Perception/labels/ に蓄積された SAM2 疑似マスクを使って、
    高容量バックボーンの教師モデルを学習する。学習結果は
    Tools/Perception/models/teacher/ に保存し、後段の蒸留ステップで使う。

使い方（実装後のイメージ）:
    python -m Tools.Perception.scripts.train_teacher \
        --labels Tools/Perception/labels \
        --output Tools/Perception/models/teacher \
        --epochs 50
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="教師セグメンテーションモデルを訓練する")
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--backbone", type=str, default="resnet50")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    return p


def train_teacher(
    labels_dir: Path,
    output_dir: Path,
    backbone: str = "resnet50",
    epochs: int = 50,
    batch_size: int = 8,
    lr: float = 1e-4,
    device: str = "cuda",
    seed: int = 42,
) -> None:
    """疑似マスクから教師セグメンテーションモデルを学習する。

    実装方針（後続フェーズで埋める）:
        1. labels_dir からフレーム/マスクペアの DataLoader を構築する。
        2. backbone を持つセグメンテーションモデルを初期化する。
        3. CrossEntropy + Dice 等の損失で epochs 分回す。
        4. output_dir に重み・メタ情報・学習ログを保存する。
    """
    raise NotImplementedError("train_teacher is a skeleton")


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _build_arg_parser().parse_args(argv)
    train_teacher(
        labels_dir=args.labels,
        output_dir=args.output,
        backbone=args.backbone,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
