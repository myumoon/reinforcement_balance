"""教師モデルから軽量生徒モデルへ蒸留するスケルトン。

役割:
    Tools/Perception/models/teacher/ の教師モデルを使って、UE5 推論で
    走らせる軽量バックボーンの生徒モデルを蒸留する。最終的には ONNX に
    エクスポートして ReinBalance/Content/Models/ へ持っていく想定。

使い方（実装後のイメージ）:
    python -m Tools.Perception.scripts.distill_student \
        --teacher Tools/Perception/models/teacher \
        --output Tools/Perception/models/student \
        --recordings Tools/Perception/recordings \
        --epochs 30
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="教師→生徒モデルへの蒸留を実行する")
    p.add_argument("--teacher", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--recordings", type=Path, required=True,
                   help="蒸留用の入力フレーム（labels/ は不要）")
    p.add_argument("--student-backbone", type=str, default="mobilenetv3_small")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--temperature", type=float, default=4.0)
    p.add_argument("--alpha", type=float, default=0.5,
                   help="蒸留損失と教師信号無し損失のブレンド比")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--export-onnx", action="store_true",
                   help="蒸留完了後に ONNX をエクスポートする")
    return p


def distill_student(
    teacher_dir: Path,
    output_dir: Path,
    recordings_dir: Path,
    student_backbone: str = "mobilenetv3_small",
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 2e-4,
    temperature: float = 4.0,
    alpha: float = 0.5,
    device: str = "cuda",
    export_onnx: bool = False,
) -> None:
    """教師モデルの予測を soft label として生徒モデルを学習する。

    実装方針（後続フェーズで埋める）:
        1. teacher_dir から教師モデルをロードし eval モードで凍結する。
        2. recordings_dir からラベル不要の DataLoader を構築する。
        3. 教師ロジットを温度 T でソフト化し、生徒との KL 損失で学習する。
        4. epoch ごとに validation IoU を測定し best を output_dir に保存。
        5. export_onnx=True なら output_dir/student.onnx を書き出す。
    """
    raise NotImplementedError("distill_student is a skeleton")


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _build_arg_parser().parse_args(argv)
    distill_student(
        teacher_dir=args.teacher,
        output_dir=args.output,
        recordings_dir=args.recordings,
        student_backbone=args.student_backbone,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        temperature=args.temperature,
        alpha=args.alpha,
        device=args.device,
        export_onnx=args.export_onnx,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
