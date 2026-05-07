"""Phase1 SAM2 知覚パイプライン本体（スケルトン）。

UE5 ゲーム画面 → SAM2 で疑似ラベル → 教師モデル学習 → 軽量生徒モデルへの
蒸留 → セグメンテーション評価、という一連のフローを `PerceptionPipeline`
クラスに集約する。

各ステージは Tools/Perception/scripts/ 配下のスクリプトに対応する:

    annotate        → scripts/annotate_with_sam2.py
    train_teacher   → scripts/train_teacher.py
    distill_student → scripts/distill_student.py
    evaluate        → scripts/eval_segmentation.py

このファイル自体は CLI とプログラムからの呼び出し両方を想定したエントリ
ポイント。中身はスケルトン実装であり、ステージ本体は後続フェーズで実装する。
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

_DEFAULT_ROOT = Path(__file__).resolve().parent

_STAGES: tuple[str, ...] = (
    "annotate",
    "train_teacher",
    "distill_student",
    "evaluate",
)


@dataclasses.dataclass
class PerceptionPipelineConfig:
    """`PerceptionPipeline` に渡す設定値。

    すべてのパスはデフォルトで Tools/Perception/ 配下の標準ディレクトリ。
    """

    root: Path = _DEFAULT_ROOT
    sam2_weights_dir: Path = _DEFAULT_ROOT / "sam2_weights"
    recordings_dir: Path = _DEFAULT_ROOT / "recordings"
    labels_dir: Path = _DEFAULT_ROOT / "labels"
    models_dir: Path = _DEFAULT_ROOT / "models"

    sam2_checkpoint: Optional[Path] = None
    sam2_config: Optional[str] = None

    teacher_model_name: str = "teacher"
    student_model_name: str = "student"

    device: str = "cuda"
    seed: int = 42

    def ensure_dirs(self) -> None:
        for d in (
            self.sam2_weights_dir,
            self.recordings_dir,
            self.labels_dir,
            self.models_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


class PerceptionPipeline:
    """SAM2 知覚パイプラインの統合エントリポイント。

    ステージは以下の順で実行される:
        1. annotate          : recordings/ に対し SAM2 で疑似マスクを生成し labels/ へ
        2. train_teacher     : labels/ を使って高精度な教師セグメンテーションモデルを学習
        3. distill_student   : 教師モデルから軽量生徒モデルへ蒸留
        4. evaluate          : 生徒モデルのセグメンテーション精度（IoU 等）を評価
    """

    def __init__(self, config: Optional[PerceptionPipelineConfig] = None) -> None:
        self.config = config or PerceptionPipelineConfig()
        self.config.ensure_dirs()

    def run(self, stages: Optional[Iterable[str]] = None) -> None:
        selected = tuple(stages) if stages else _STAGES
        for stage in selected:
            if stage not in _STAGES:
                raise ValueError(f"unknown stage: {stage}")
        for stage in selected:
            logger.info("=== stage start: %s ===", stage)
            getattr(self, stage)()
            logger.info("=== stage done : %s ===", stage)

    def annotate(self) -> None:
        """SAM2 で recordings/ → labels/ に疑似マスクを生成する。

        実装は scripts/annotate_with_sam2.py に委譲予定。
        """
        raise NotImplementedError("annotate stage is not implemented yet")

    def train_teacher(self) -> None:
        """labels/ を教師信号として高精度教師モデルを訓練する。

        実装は scripts/train_teacher.py に委譲予定。
        """
        raise NotImplementedError("train_teacher stage is not implemented yet")

    def distill_student(self) -> None:
        """教師モデルから軽量生徒モデルへ蒸留する。

        実装は scripts/distill_student.py に委譲予定。
        """
        raise NotImplementedError("distill_student stage is not implemented yet")

    def evaluate(self) -> None:
        """生徒モデルのセグメンテーション精度を評価する。

        実装は scripts/eval_segmentation.py に委譲予定。
        """
        raise NotImplementedError("evaluate stage is not implemented yet")


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SAM2 知覚パイプラインを実行する")
    p.add_argument(
        "--stages",
        nargs="+",
        choices=_STAGES,
        default=None,
        help="実行するステージ（省略時は全ステージを順次実行）",
    )
    p.add_argument("--root", type=Path, default=_DEFAULT_ROOT)
    p.add_argument("--sam2-checkpoint", type=Path, default=None)
    p.add_argument("--sam2-config", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _build_arg_parser().parse_args(argv)
    config = PerceptionPipelineConfig(
        root=args.root,
        sam2_weights_dir=args.root / "sam2_weights",
        recordings_dir=args.root / "recordings",
        labels_dir=args.root / "labels",
        models_dir=args.root / "models",
        sam2_checkpoint=args.sam2_checkpoint,
        sam2_config=args.sam2_config,
        device=args.device,
        seed=args.seed,
    )
    pipeline = PerceptionPipeline(config)
    pipeline.run(args.stages)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
