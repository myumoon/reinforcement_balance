"""world detector 評価ハーネス CLI。

holdout mAP50:95 / class recall / density/count error / nearest-distance error /
track ID switches / latency を synthetic prediction で計算する metric harness。

weight がなくても metric 計算パスを検証できる。

使用例:
    python eval_survivors_world_detector.py \\
        --annotations data/world_val.json \\
        --config configs/world_detector_v1.yaml \\
        --class-map configs/world_class_map_v1.yaml
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import yaml


# ---- metric ----

# track_id_switches は静的 eval では計算不能。None で unsupported を明示する。
_UNSUPPORTED = "unsupported"


@dataclass
class EvalMetrics:
    """評価メトリクスの集計結果。

    synthetic prediction でも exact テストできる形式で保持する。
    未実装指標は "unsupported" を出力し、0 と区別する。
    """

    map50_95: float = 0.0
    class_recall: dict[int, float] = field(default_factory=dict)
    # density_error: エンティティ密度誤差（count_error と同じ集計軸）
    density_error: float = 0.0
    count_error: float = 0.0
    # nearest_distance_error: GT と予測の最近傍中心距離（正規化 px）
    nearest_distance_error: float = 0.0
    # track_id_switches: 静的 eval では計算不能。CLI は "unsupported" を出力する。
    track_id_switches: str | int = _UNSUPPORTED
    # mean_latency_ms: metric 集計処理時間（検出器推論時間ではない）。
    # 推論レイテンシは 04-08 の benchmark で計測する。
    mean_latency_ms: float = 0.0

    def to_json(self) -> str:
        return json.dumps(
            {
                "map50_95": self.map50_95,
                "class_recall": {str(k): v for k, v in self.class_recall.items()},
                "density_error": self.density_error,
                "count_error": self.count_error,
                "nearest_distance_error": self.nearest_distance_error,
                "track_id_switches": self.track_id_switches,
                "mean_latency_ms_metric_only": self.mean_latency_ms,
            },
            indent=2,
        )


def compute_iou_matrix(gt_boxes: np.ndarray, pred_boxes: np.ndarray) -> np.ndarray:
    """gt と pred の IoU 行列を計算する (|gt| x |pred|)。

    boxes は (N, 4) float32 [x1, y1, x2, y2]。
    """
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return np.zeros((len(gt_boxes), len(pred_boxes)), dtype=np.float32)

    ix1 = np.maximum(gt_boxes[:, 0:1], pred_boxes[:, 0])
    iy1 = np.maximum(gt_boxes[:, 1:2], pred_boxes[:, 1])
    ix2 = np.minimum(gt_boxes[:, 2:3], pred_boxes[:, 2])
    iy2 = np.minimum(gt_boxes[:, 3:4], pred_boxes[:, 3])
    inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
    area_gt = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
    area_pred = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    union = area_gt[:, None] + area_pred[None, :] - inter
    return (inter / np.maximum(union, 1e-6)).astype(np.float32)


def compute_ap50_95(
    gt_boxes: np.ndarray,
    gt_classes: np.ndarray,
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    pred_classes: np.ndarray,
) -> float:
    """AP50:95 を簡易計算する（COCO 定義の iou_thresholds=[0.5,...,0.95,step=0.05]）。

    ponytail: per-class AP の mean として実装。precision-recall curve の正式実装は 04-07。
    """
    iou_thresholds = np.arange(0.50, 1.00, 0.05)
    aps: list[float] = []

    for iou_thr in iou_thresholds:
        if len(gt_boxes) == 0:
            aps.append(1.0 if len(pred_boxes) == 0 else 0.0)
            continue
        if len(pred_boxes) == 0:
            aps.append(0.0)
            continue

        iou_mat = compute_iou_matrix(gt_boxes, pred_boxes)
        sorted_idx = np.argsort(-pred_scores)
        tp = 0
        matched_gt: set[int] = set()
        for pi in sorted_idx:
            if pred_classes[pi] == 0:
                continue
            best_iou = 0.0
            best_gi = -1
            for gi in range(len(gt_boxes)):
                if gi in matched_gt:
                    continue
                if gt_classes[gi] != pred_classes[pi]:
                    continue
                if iou_mat[gi, pi] > best_iou:
                    best_iou = iou_mat[gi, pi]
                    best_gi = gi
            if best_iou >= iou_thr and best_gi >= 0:
                tp += 1
                matched_gt.add(best_gi)

        precision = tp / max(len(pred_boxes), 1)
        recall = tp / max(len(gt_boxes), 1)
        aps.append(precision * recall)  # 簡易 AP（正式は PR 曲線積分）

    return float(np.mean(aps)) if aps else 0.0


def _box_center(box_xyxy: np.ndarray) -> np.ndarray:
    """(N,4) box → (N,2) 中心座標。"""
    return np.stack([(box_xyxy[:, 0] + box_xyxy[:, 2]) / 2,
                     (box_xyxy[:, 1] + box_xyxy[:, 3]) / 2], axis=1)


def evaluate_from_predictions(
    gt_annotations: list[dict],
    pred_annotations: list[dict],
    num_classes: int,
) -> EvalMetrics:
    """GT と予測から EvalMetrics を計算する。

    synthetic fixture でも動作する。
    gt_annotations / pred_annotations は COCO annotation dict のリスト。

    class recall は confidence 降順・一対一 matching で集計し、
    同一予測 box を複数 GT に対応付けない。
    track_id_switches は静的 eval で計算不能なため "unsupported" を返す。
    """
    gt_by_image: dict[int, list[dict]] = {}
    for ann in gt_annotations:
        gt_by_image.setdefault(ann["image_id"], []).append(ann)

    pred_by_image: dict[int, list[dict]] = {}
    for ann in pred_annotations:
        pred_by_image.setdefault(ann["image_id"], []).append(ann)

    all_image_ids = set(gt_by_image) | set(pred_by_image)

    ap_scores: list[float] = []
    class_tp: dict[int, int] = {c: 0 for c in range(1, num_classes)}
    class_gt: dict[int, int] = {c: 0 for c in range(1, num_classes)}
    count_errors: list[float] = []
    nearest_dist_errors: list[float] = []

    for iid in all_image_ids:
        gts = gt_by_image.get(iid, [])
        preds = pred_by_image.get(iid, [])

        gt_boxes = np.array([[g["bbox"][0], g["bbox"][1],
                               g["bbox"][0] + g["bbox"][2],
                               g["bbox"][1] + g["bbox"][3]] for g in gts], dtype=np.float32).reshape(-1, 4)
        gt_classes = np.array([g["category_id"] for g in gts], dtype=np.int32)
        pred_boxes = np.array([[p["bbox"][0], p["bbox"][1],
                                p["bbox"][0] + p["bbox"][2],
                                p["bbox"][1] + p["bbox"][3]] for p in preds], dtype=np.float32).reshape(-1, 4)
        pred_scores = np.array([p.get("score", 1.0) for p in preds], dtype=np.float32)
        pred_classes = np.array([p["category_id"] for p in preds], dtype=np.int32)

        ap = compute_ap50_95(gt_boxes, gt_classes, pred_boxes, pred_scores, pred_classes)
        ap_scores.append(ap)
        count_errors.append(abs(len(gts) - len(preds)))

        # nearest_distance_error: GT ごとに最近傍の予測中心との距離を集計する
        if len(gt_boxes) > 0 and len(pred_boxes) > 0:
            gt_centers = _box_center(gt_boxes)  # (G, 2)
            pred_centers = _box_center(pred_boxes)  # (P, 2)
            # (G, P) 距離行列
            diff = gt_centers[:, None, :] - pred_centers[None, :, :]
            dists = np.sqrt((diff ** 2).sum(axis=2))  # (G, P)
            nearest_dist_errors.append(float(dists.min(axis=1).mean()))
        elif len(gt_boxes) > 0:
            # GT があるが予測がない → 最大距離を誤差として記録
            nearest_dist_errors.append(float(np.hypot(1920, 1080)))

        # class recall: confidence 降順・一対一 matching（使用済み予測を再利用しない）
        for c in range(1, num_classes):
            class_gt[c] += int((gt_classes == c).sum())
            if len(gt_boxes) > 0 and len(pred_boxes) > 0:
                iou_mat = compute_iou_matrix(gt_boxes, pred_boxes)
                # confidence 降順でソートした予測インデックス（class=c のみ）
                pred_c_idxs = [pi for pi in np.argsort(-pred_scores) if pred_classes[pi] == c]
                used_pred: set[int] = set()
                for gi, gc in enumerate(gt_classes):
                    if gc != c:
                        continue
                    for pi in pred_c_idxs:
                        if pi in used_pred:
                            continue
                        if iou_mat[gi, pi] >= 0.5:
                            class_tp[c] += 1
                            used_pred.add(pi)
                            break

    class_recall = {
        c: class_tp[c] / class_gt[c] if class_gt[c] > 0 else 0.0
        for c in range(1, num_classes)
    }

    density_error = float(np.mean(count_errors)) if count_errors else 0.0

    return EvalMetrics(
        map50_95=float(np.mean(ap_scores)) if ap_scores else 0.0,
        class_recall=class_recall,
        density_error=density_error,
        count_error=density_error,
        nearest_distance_error=float(np.mean(nearest_dist_errors)) if nearest_dist_errors else 0.0,
        track_id_switches=_UNSUPPORTED,  # 静的 eval では計算不能
    )


# ---- CLI ----

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="world detector 評価ハーネス",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--annotations", required=True, help="GT COCO JSON")
    p.add_argument("--predictions", help="予測 COCO JSON（省略時は synthetic zero prediction）")
    p.add_argument("--config", default="configs/world_detector_v1.yaml", help="detector config YAML")
    p.add_argument("--class-map", default="configs/world_class_map_v1.yaml", help="class map YAML")
    p.add_argument("--output", help="メトリクス JSON の出力先")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    from survivors.vision.world_dataset import WorldDataset, load_class_map
    from survivors.vision.world_detector import load_detector_config

    ann_path = pathlib.Path(args.annotations)
    cfg_path = pathlib.Path(args.config)
    cm_path = pathlib.Path(args.class_map)

    cfg = load_detector_config(cfg_path)
    num_classes = cfg["model"]["num_classes"]

    gt_data = json.loads(ann_path.read_text(encoding="utf-8"))
    gt_annotations = gt_data.get("annotations", [])

    if args.predictions:
        pred_data = json.loads(pathlib.Path(args.predictions).read_text(encoding="utf-8"))
        pred_annotations = pred_data.get("annotations", [])
    else:
        pred_annotations = []

    t0 = time.perf_counter()
    metrics = evaluate_from_predictions(gt_annotations, pred_annotations, num_classes)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    metrics.mean_latency_ms = elapsed_ms

    print(metrics.to_json())

    if args.output:
        pathlib.Path(args.output).write_text(metrics.to_json(), encoding="utf-8")
        print(f"[INFO] メトリクスを {args.output} へ保存しました。")

    return 0


if __name__ == "__main__":
    sys.exit(main())
