"""セグメンテーション精度を評価するスケルトン。

役割:
    教師 / 生徒モデルを Tools/Perception/labels/ の正解マスクと突き合わせ、
    IoU・mIoU・Pixel Accuracy などを計測する。蒸留前後の比較やゲーム別の
    弱点把握に使う。

使い方（実装後のイメージ）:
    python -m Tools.Perception.scripts.eval_segmentation \
        --model Tools/Perception/models/student \
        --labels Tools/Perception/labels \
        --report Tools/Perception/models/student/eval.json
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="セグメンテーションモデルを評価する")
    p.add_argument("--model", type=Path, required=True,
                   help="評価対象のモデルディレクトリ（teacher/student）")
    p.add_argument("--labels", type=Path, required=True,
                   help="正解マスク（SAM2 疑似ラベル or 手動アノテ）")
    p.add_argument("--recordings", type=Path, default=None,
                   help="入力フレームディレクトリ（省略時は labels/ の対と整合）")
    p.add_argument("--report", type=Path, required=True,
                   help="評価結果 JSON の出力先")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num-classes", type=int, default=None)
    return p


def evaluate_segmentation(
    model_dir: Path,
    labels_dir: Path,
    report_path: Path,
    recordings_dir: Optional[Path] = None,
    device: str = "cuda",
    num_classes: Optional[int] = None,
) -> dict:
    """モデル予測と正解マスクから IoU 系指標を算出し JSON で保存する。

    実装方針（後続フェーズで埋める）:
        1. model_dir からモデルをロードし eval モードに固定する。
        2. labels_dir / recordings_dir からペアを構築する。
        3. クラス別 IoU、mIoU、Pixel Accuracy を計算する。
        4. report_path に JSON で書き出す（ゲーム別・クラス別ブレイクダウン付き）。
        5. 集計結果を dict で返す（呼び出し側でロギングしやすくする）。
    """
    raise NotImplementedError("evaluate_segmentation is a skeleton")


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _build_arg_parser().parse_args(argv)
    evaluate_segmentation(
        model_dir=args.model,
        labels_dir=args.labels,
        report_path=args.report,
        recordings_dir=args.recordings,
        device=args.device,
        num_classes=args.num_classes,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
