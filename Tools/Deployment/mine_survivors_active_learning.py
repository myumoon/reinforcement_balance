"""active learning マイニング CLI。

entropy / track break / density disagreement / rare state の 4 軸から
アノテーション優先度スコアを計算し、次に annotate すべきフレームを出力する。

weight がなくても score 計算パスを検証できる。

使用例:
    python mine_survivors_active_learning.py \\
        --annotations data/world_annotations.json \\
        --candidate-frames data/unlabeled_frames.json \\
        --top-k 100 \\
        --output data/mining_results.json
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from dataclasses import dataclass

import numpy as np


# ---- score components ----

def entropy_score(class_probs: np.ndarray) -> float:
    """予測確率分布のエントロピー（不確かさ）。

    class_probs は shape (num_classes,) の確率ベクトル（合計 1）。
    エントロピーが高いほどアノテーション価値が高い。
    """
    eps = 1e-9
    p = np.clip(class_probs, eps, 1.0)
    return float(-np.sum(p * np.log(p)))


def track_break_score(n_breaks: int, max_breaks: int = 10) -> float:
    """トラック切れ回数をスコア化する（0~1）。

    切れが多いほどハードケースでアノテーション価値が高い。
    """
    return min(n_breaks / max(max_breaks, 1), 1.0)


def density_disagreement_score(gt_count: int, pred_count: int, max_error: int = 20) -> float:
    """GT とモデル予測のエンティティ数の差をスコア化する（0~1）。"""
    return min(abs(gt_count - pred_count) / max(max_error, 1), 1.0)


def rare_state_score(class_id: int, class_counts: dict[int, int], total: int) -> float:
    """クラスの出現頻度が低いほど高スコア（long-tail 補正）。"""
    if total == 0:
        return 0.0
    freq = class_counts.get(class_id, 0) / total
    return float(1.0 - freq)


def compute_active_learning_score(
    class_probs: np.ndarray,
    n_track_breaks: int,
    gt_count: int,
    pred_count: int,
    class_id: int,
    class_counts: dict[int, int],
    total_samples: int,
    *,
    w_entropy: float = 0.3,
    w_track: float = 0.3,
    w_density: float = 0.2,
    w_rare: float = 0.2,
) -> float:
    """4 軸スコアの加重和を返す（0~1）。

    デフォルト重みは perception_feasibility_v1.yaml の budget と一致させた初期値。
    """
    s_entropy = entropy_score(class_probs) / math.log(max(len(class_probs), 2))
    s_track = track_break_score(n_track_breaks)
    s_density = density_disagreement_score(gt_count, pred_count)
    s_rare = rare_state_score(class_id, class_counts, total_samples)
    return (
        w_entropy * s_entropy
        + w_track * s_track
        + w_density * s_density
        + w_rare * s_rare
    )


# ---- mining result ----

@dataclass
class MiningResult:
    """1 フレームの mining スコアと優先度情報。"""

    image_id: int
    file_name: str
    score: float
    components: dict[str, float]


def mine_top_k(
    candidates: list[dict],
    class_counts: dict[int, int],
    total_samples: int,
    top_k: int,
) -> list[MiningResult]:
    """candidates からスコア上位 top_k のフレームを返す。

    candidates の各要素は image_id, file_name, class_probs, n_track_breaks,
    gt_count, pred_count, dominant_class_id を持つ dict。
    """
    results: list[MiningResult] = []
    for cand in candidates:
        class_probs = np.array(cand.get("class_probs", [1.0 / 12] * 12), dtype=np.float32)
        score = compute_active_learning_score(
            class_probs=class_probs,
            n_track_breaks=cand.get("n_track_breaks", 0),
            gt_count=cand.get("gt_count", 0),
            pred_count=cand.get("pred_count", 0),
            class_id=cand.get("dominant_class_id", 1),
            class_counts=class_counts,
            total_samples=total_samples,
        )
        results.append(
            MiningResult(
                image_id=cand["image_id"],
                file_name=cand.get("file_name", ""),
                score=score,
                components={
                    "entropy": entropy_score(class_probs),
                    "track_break": track_break_score(cand.get("n_track_breaks", 0)),
                    "density": density_disagreement_score(
                        cand.get("gt_count", 0), cand.get("pred_count", 0)
                    ),
                    "rare": rare_state_score(
                        cand.get("dominant_class_id", 1), class_counts, total_samples
                    ),
                },
            )
        )

    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_k]


# ---- CLI ----

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="active learning マイニング CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--annotations", required=True, help="既存 GT COCO JSON（クラス分布の計算に使用）")
    p.add_argument("--candidate-frames", required=True, help="候補フレーム JSON（image_id, class_probs 等を含む）")
    p.add_argument("--top-k", type=int, default=100, help="出力する上位フレーム数")
    p.add_argument("--output", help="mining 結果 JSON の出力先")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    ann_path = pathlib.Path(args.annotations)
    cand_path = pathlib.Path(args.candidate_frames)

    gt_data = json.loads(ann_path.read_text(encoding="utf-8"))
    gt_anns = gt_data.get("annotations", [])

    class_counts: dict[int, int] = {}
    for ann in gt_anns:
        cid = ann.get("category_id", 0)
        class_counts[cid] = class_counts.get(cid, 0) + 1
    total_samples = len(gt_anns)

    candidates = json.loads(cand_path.read_text(encoding="utf-8"))
    results = mine_top_k(candidates, class_counts, total_samples, args.top_k)

    output = [
        {
            "image_id": r.image_id,
            "file_name": r.file_name,
            "score": r.score,
            "components": r.components,
        }
        for r in results
    ]

    if args.output:
        pathlib.Path(args.output).write_text(
            json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[INFO] top-{args.top_k} フレームを {args.output} へ保存しました。")
    else:
        print(json.dumps(output[:5], indent=2, ensure_ascii=False))
        print(f"... ({len(results)} 件中 5 件を表示)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
