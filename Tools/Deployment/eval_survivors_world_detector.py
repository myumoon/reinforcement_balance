"""world detector 評価ハーネス CLI。

holdout mAP50:95 / class recall / density/count error / nearest-distance error /
track ID switches / latency を synthetic prediction で計算する metric harness。

weight がなくても metric 計算パスを検証できる。

使用例:
    python eval_survivors_world_detector.py \\
        --annotations data/world_val.json \\
        --config configs/world_detector_v2.yaml \\
        --class-map configs/world_class_map_v2.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import sys
import time
from dataclasses import dataclass, field
from typing import Any

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

    # proxy_ap50_95: PR 曲線台形積分の IoU 閾値平均（development diagnostic）。COCO 公式 101 点補間とは異なる。
    proxy_ap50_95: float = 0.0
    class_recall: dict[int, float] = field(default_factory=dict)
    # density_error: エンティティ密度誤差（count_error と同じ集計軸）
    density_error: float = 0.0
    count_error: float = 0.0
    # nearest_distance_error: GT と予測の最近傍中心距離（正規化 px）
    nearest_distance_error: float = 0.0
    # density_correlation: directional bin ごとの GT-予測密度 Pearson 相関（0〜1）。
    density_correlation: float = 1.0
    # track_id_switches: 静的 eval では計算不能。CLI は "unsupported" を出力する。
    track_id_switches: str | int = _UNSUPPORTED
    # mean_latency_ms: metric 集計処理時間（検出器推論時間ではない）。
    # 推論レイテンシは 04-08 の benchmark で計測する。
    mean_latency_ms: float = 0.0

    def to_json(self) -> str:
        """development metric を JSON 表現へ変換する。\n\n未対応指標は既定値と区別できる wire 値のまま出力する。"""
        return json.dumps(
            {
                "proxy_ap50_95": self.proxy_ap50_95,
                "class_recall": {str(k): v for k, v in self.class_recall.items()},
                "density_error": self.density_error,
                "count_error": self.count_error,
                "nearest_distance_error": self.nearest_distance_error,
                "density_correlation": self.density_correlation,
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


def compute_ap_coco101(
    gt_boxes: np.ndarray,
    gt_classes: np.ndarray,
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    pred_classes: np.ndarray,
    *,
    gt_image_ids: np.ndarray | None = None,
    pred_image_ids: np.ndarray | None = None,
) -> float:
    """COCO の IoU 0.50..0.95 と recall 101 点で foreground AP を計算する。

    image id ごとに GT を照合し、class 0 と GT のない class を集計から外す。
    """
    gt_boxes = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 4)
    pred_boxes = np.asarray(pred_boxes, dtype=np.float64).reshape(-1, 4)
    gt_classes = np.asarray(gt_classes, dtype=np.int64).reshape(-1)
    pred_classes = np.asarray(pred_classes, dtype=np.int64).reshape(-1)
    pred_scores = np.asarray(pred_scores, dtype=np.float64).reshape(-1)
    gt_image_ids = np.zeros(len(gt_boxes), dtype=np.int64) if gt_image_ids is None else np.asarray(gt_image_ids).reshape(-1)
    pred_image_ids = np.zeros(len(pred_boxes), dtype=np.int64) if pred_image_ids is None else np.asarray(pred_image_ids).reshape(-1)
    if len(gt_classes) != len(gt_boxes) or len(pred_classes) != len(pred_boxes) or len(pred_scores) != len(pred_boxes):
        raise ValueError("COCO AP 入力の box/class/score 件数が一致しません")
    if len(gt_image_ids) != len(gt_boxes) or len(pred_image_ids) != len(pred_boxes):
        raise ValueError("COCO AP 入力の image id 件数が一致しません")
    if not np.isfinite(gt_boxes).all() or not np.isfinite(pred_boxes).all() or not np.isfinite(pred_scores).all():
        raise ValueError("COCO AP 入力に NaN / Infinity があります")
    if np.any(pred_scores < 0) or np.any(pred_scores > 1):
        raise ValueError("COCO AP score は [0, 1] の範囲が必要です")
    classes = sorted(set(gt_classes.tolist()) - {0})
    if not classes:
        raise ValueError("COCO AP に foreground GT がありません")
    recall_points = np.linspace(0.0, 1.0, 101)
    class_aps: list[float] = []
    for class_id in classes:
        class_gt = np.flatnonzero(gt_classes == class_id)
        class_predictions = np.flatnonzero(pred_classes == class_id)
        class_threshold_aps: list[float] = []
        for iou_threshold in np.arange(0.50, 1.00, 0.05):
            matched: dict[Any, set[int]] = {}
            gt_by_image: dict[Any, np.ndarray] = {}
            for image_id in dict.fromkeys(gt_image_ids[class_gt].tolist()):
                indexes = class_gt[gt_image_ids[class_gt] == image_id]
                gt_by_image[image_id] = indexes
                matched[image_id] = set()
            ordered = sorted(class_predictions.tolist(), key=lambda index: (-pred_scores[index], index))
            tp = np.zeros(len(ordered), dtype=np.float64)
            fp = np.zeros(len(ordered), dtype=np.float64)
            for rank, prediction_index in enumerate(ordered):
                image_id = pred_image_ids[prediction_index]
                candidates = gt_by_image.get(image_id)
                if candidates is None:
                    fp[rank] = 1
                    continue
                available = np.asarray([index for index in candidates if index not in matched[image_id]], dtype=np.int64)
                if not len(available):
                    fp[rank] = 1
                    continue
                overlaps = compute_iou_matrix(gt_boxes[available], pred_boxes[prediction_index:prediction_index + 1])[:, 0]
                best = int(np.argmax(overlaps))
                if overlaps[best] >= iou_threshold:
                    tp[rank] = 1
                    matched[image_id].add(int(available[best]))
                else:
                    fp[rank] = 1
            if not len(ordered):
                class_threshold_aps.append(0.0)
                continue
            cumulative_tp = np.cumsum(tp)
            recall = cumulative_tp / len(class_gt)
            precision = cumulative_tp / np.maximum(cumulative_tp + np.cumsum(fp), 1.0)
            precision = np.maximum.accumulate(precision[::-1])[::-1]
            sampled = [float(np.max(precision[recall >= point])) if np.any(recall >= point) else 0.0
                       for point in recall_points]
            class_threshold_aps.append(float(np.mean(sampled)))
        class_aps.append(float(np.mean(class_threshold_aps)))
    return float(np.mean(class_aps))


def compute_ap50_95(
    gt_boxes: np.ndarray,
    gt_classes: np.ndarray,
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    pred_classes: np.ndarray,
) -> float:
    """proxy AP50:95 を PR 曲線台形積分で計算する（development diagnostic 用）。

    IoU 閾値 [0.5,...,0.95, step=0.05] ごとに PR 曲線を台形積分し平均を返す。
    COCO 公式 101 点補間とは異なる（proxy）。正式 metric は 04-08 で実装する。
    class 0 は背景として無視する。1 GT + 1 TP(高 conf) + 1 FP(低 conf) → AP=1.0。
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
        n_gt = len(gt_boxes)
        tp_list: list[float] = []
        fp_list: list[float] = []
        matched_gt: set[int] = set()
        for pi in sorted_idx:
            if pred_classes[pi] == 0:
                continue  # 背景クラスは TP/FP に含めない
            best_gi, best_iou = -1, iou_thr - 1e-9
            for gi in range(n_gt):
                if gi not in matched_gt and gt_classes[gi] == pred_classes[pi]:
                    if iou_mat[gi, pi] > best_iou:
                        best_iou, best_gi = iou_mat[gi, pi], gi
            if best_gi >= 0:
                tp_list.append(1.0)
                fp_list.append(0.0)
                matched_gt.add(best_gi)
            else:
                tp_list.append(0.0)
                fp_list.append(1.0)

        if not tp_list:
            aps.append(0.0)
            continue

        cum_tp = np.cumsum(tp_list)
        cum_fp = np.cumsum(fp_list)
        prec = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
        rec = cum_tp / n_gt
        # sentinel を追加して単調減少 precision で台形積分
        prec = np.concatenate([[1.0], prec, [0.0]])
        rec = np.concatenate([[0.0], rec, [rec[-1]]])
        for i in range(len(prec) - 2, -1, -1):
            prec[i] = max(prec[i], prec[i + 1])
        aps.append(float(np.trapz(prec, rec)))

    return float(np.mean(aps)) if aps else 0.0


def _box_center(box_xyxy: np.ndarray) -> np.ndarray:
    """xyxy box 群を中心座標へ変換する。\n\n距離 metric が各 box の中心だけを比較できる形に揃える。"""
    return np.stack([(box_xyxy[:, 0] + box_xyxy[:, 2]) / 2,
                     (box_xyxy[:, 1] + box_xyxy[:, 3]) / 2], axis=1)


def _compute_class_ap_coco(
    gt_by_image: dict[int, list[dict]],
    pred_by_image: dict[int, list[dict]],
    class_id: int,
) -> float:
    """1 クラス分の開発用 AP50:95 を台形積分で計算する（confidence降順・同一image内GTのみ1:1マッチ）。

    全画像の予測を confidence 降順で走査し、各予測は同じ画像の GT のみとマッチできる。
    画像をまたいだ誤 TP を防ぐ。
    注意: 台形積分（np.trapz）を使用した開発用 diagnostic。COCO 公式 101 点補間とは異なる。
    COCO 公式準拠の 101 点補間は 04-08 で実装する。
    """
    n_gt = sum(
        1 for gts in gt_by_image.values()
        for g in gts if g["category_id"] == class_id
    )
    if n_gt == 0:
        return 0.0

    all_preds: list[tuple[float, int, dict]] = sorted(
        (
            (p.get("score", 1.0), iid, p)
            for iid, preds in pred_by_image.items()
            for p in preds if p["category_id"] == class_id
        ),
        key=lambda x: -x[0],
    )
    if not all_preds:
        return 0.0

    # 各予測について同一 image 内 class_id GT との IoU を事前計算し降順保持
    iid_class_gts: dict[int, list[dict]] = {
        iid: [g for g in gts if g["category_id"] == class_id]
        for iid, gts in gt_by_image.items()
    }
    pred_gt_ious: list[list[tuple[int, float]]] = []
    for _score, iid, p in all_preds:
        px1, py1, pw, ph = p["bbox"]
        pred_xyxy = np.array([[px1, py1, px1 + pw, py1 + ph]], dtype=np.float32)
        gts = iid_class_gts.get(iid, [])
        ious: list[tuple[int, float]] = []
        for gi, g in enumerate(gts):
            x1, y1, w, h = g["bbox"]
            iou_val = float(compute_iou_matrix(
                np.array([[x1, y1, x1 + w, y1 + h]], dtype=np.float32), pred_xyxy,
            )[0, 0])
            ious.append((gi, iou_val))
        pred_gt_ious.append(sorted(ious, key=lambda x: -x[1]))

    iou_thresholds = np.arange(0.50, 1.00, 0.05)
    aps: list[float] = []
    for iou_thr in iou_thresholds:
        matched: dict[int, set[int]] = {iid: set() for iid in gt_by_image}
        tp_arr: list[int] = []
        fp_arr: list[int] = []
        for idx, (_score, iid, _p) in enumerate(all_preds):
            best_gi = -1
            for gi, iou_val in pred_gt_ious[idx]:
                if gi not in matched[iid] and iou_val >= iou_thr:
                    best_gi = gi
                    break
            if best_gi >= 0:
                matched[iid].add(best_gi)
                tp_arr.append(1); fp_arr.append(0)
            else:
                tp_arr.append(0); fp_arr.append(1)

        cum_tp = np.cumsum(tp_arr, dtype=float)
        cum_fp = np.cumsum(fp_arr, dtype=float)
        prec = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
        rec = cum_tp / n_gt
        prec = np.concatenate([[1.0], prec, [0.0]])
        rec = np.concatenate([[0.0], rec, [rec[-1]]])
        for i in range(len(prec) - 2, -1, -1):
            prec[i] = max(prec[i], prec[i + 1])
        aps.append(float(np.trapz(prec, rec)))

    return float(np.mean(aps))


def _directional_density_correlation(
    gt_by_image: dict[int, list[dict]],
    pred_by_image: dict[int, list[dict]],
    n_bins: int = 8,
) -> float:
    """画像ごとの directional bin ベクトルを対にして GT-予測密度 Pearson 相関を計算する。

    全画像を単一ヒストグラムに集約せず、画像ごとに GT/予測 bin ベクトルを作成して
    対で連結し Pearson 相関を計算する。画像間の位置入れ替えを正しく検出できる。
    """
    all_image_ids = set(gt_by_image) | set(pred_by_image)
    if not all_image_ids:
        return 1.0

    gt_vec_parts: list[np.ndarray] = []
    pred_vec_parts: list[np.ndarray] = []

    for iid in all_image_ids:
        gts = gt_by_image.get(iid, [])
        preds = pred_by_image.get(iid, [])

        def _centers(anns: list[dict]) -> np.ndarray:
            """一枚の画像の xyxy annotation 群から中心座標を作る。\n\n空 annotation は 0x2 配列で返し concat の shape を保つ。"""
            if not anns:
                return np.zeros((0, 2), np.float32)
            return np.array(
                [[a["bbox"][0] + a["bbox"][2] / 2, a["bbox"][1] + a["bbox"][3] / 2]
                 for a in anns], dtype=np.float32,
            )

        gt_c = _centers(gts)
        pred_c = _centers(preds)

        all_c = np.concatenate([gt_c, pred_c]) if len(gt_c) > 0 and len(pred_c) > 0 \
            else (gt_c if len(gt_c) > 0 else pred_c)
        if len(all_c) == 0:
            continue
        cx, cy = float(np.mean(all_c[:, 0])), float(np.mean(all_c[:, 1]))

        def _bin_vec(pts: np.ndarray) -> np.ndarray:
            """中心点を基準に角度 bin の個数を集計する。\n\nGT と prediction の双方へ同じ中心と bin 数を適用する。"""
            if len(pts) == 0:
                return np.zeros(n_bins, dtype=float)
            angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
            bins = ((angles + np.pi) / (2 * np.pi) * n_bins).astype(int) % n_bins
            return np.bincount(bins, minlength=n_bins).astype(float)

        gt_vec_parts.append(_bin_vec(gt_c))
        pred_vec_parts.append(_bin_vec(pred_c))

    if not gt_vec_parts:
        return 1.0

    gt_vec = np.concatenate(gt_vec_parts)
    pred_vec = np.concatenate(pred_vec_parts)

    if np.std(gt_vec) < 1e-9 or np.std(pred_vec) < 1e-9:
        return 1.0 if np.allclose(gt_vec, pred_vec) else 0.0

    corr = float(np.corrcoef(gt_vec, pred_vec)[0, 1])
    return max(0.0, corr)


def evaluate_from_predictions(
    gt_annotations: list[dict],
    pred_annotations: list[dict],
    num_classes: int,
    *,
    viewport_diagonal: float = float(np.hypot(1920, 1080)),
) -> EvalMetrics:
    """GT と予測から EvalMetrics を計算する。

    synthetic fixture でも動作する。
    gt_annotations / pred_annotations は COCO annotation dict のリスト。

    class recall は confidence 降順・一対一 matching で集計し、
    同一予測 box を複数 GT に対応付けない。
    nearest_distance_error は viewport_diagonal で正規化した値（0〜1）を返す。
    track_id_switches は静的 eval で計算不能なため "unsupported" を返す。
    """
    gt_by_image: dict[int, list[dict]] = {}
    for ann in gt_annotations:
        gt_by_image.setdefault(ann["image_id"], []).append(ann)

    pred_by_image: dict[int, list[dict]] = {}
    for ann in pred_annotations:
        pred_by_image.setdefault(ann["image_id"], []).append(ann)

    all_image_ids = set(gt_by_image) | set(pred_by_image)

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

        count_errors.append(abs(len(gts) - len(preds)))

        # nearest_distance_error: GT ごとに最近傍の予測中心との距離を個別に集計する
        if len(gt_boxes) > 0 and len(pred_boxes) > 0:
            gt_centers = _box_center(gt_boxes)
            pred_centers = _box_center(pred_boxes)
            diff = gt_centers[:, None, :] - pred_centers[None, :, :]
            dists = np.sqrt((diff ** 2).sum(axis=2))
            nearest_dist_errors.extend((dists.min(axis=1) / viewport_diagonal).tolist())
        elif len(gt_boxes) > 0:
            nearest_dist_errors.extend([1.0] * len(gt_boxes))

        # class recall: confidence 降順・一対一 matching
        for c in range(1, num_classes):
            class_gt[c] += int((gt_classes == c).sum())
            if len(gt_boxes) > 0 and len(pred_boxes) > 0:
                iou_mat = compute_iou_matrix(gt_boxes, pred_boxes)
                pred_c_idxs = [pi for pi in np.argsort(-pred_scores) if pred_classes[pi] == c]
                matched_gt: set[int] = set()
                for pi in pred_c_idxs:
                    best_gi, best_iou = -1, 0.5 - 1e-9
                    for gi in range(len(gt_boxes)):
                        if gi in matched_gt or gt_classes[gi] != c:
                            continue
                        if iou_mat[gi, pi] > best_iou:
                            best_iou = iou_mat[gi, pi]
                            best_gi = gi
                    if best_gi >= 0:
                        matched_gt.add(best_gi)
                        class_tp[c] += 1

    class_recall = {
        c: class_tp[c] / class_gt[c] if class_gt[c] > 0 else 0.0
        for c in range(1, num_classes)
    }
    density_error = float(np.mean(count_errors)) if count_errors else 0.0

    # proxy mAP（開発 diagnostic）: 同一画像内 GT のみとマッチする image-aware 台形積分
    class_aps: list[float] = []
    for c in range(1, num_classes):
        if any(g["category_id"] == c for gts in gt_by_image.values() for g in gts):
            class_aps.append(_compute_class_ap_coco(gt_by_image, pred_by_image, c))

    density_corr = _directional_density_correlation(gt_by_image, pred_by_image)

    return EvalMetrics(
        proxy_ap50_95=float(np.mean(class_aps)) if class_aps else 0.0,
        class_recall=class_recall,
        density_error=density_error,
        count_error=density_error,
        nearest_distance_error=float(np.median(nearest_dist_errors)) if nearest_dist_errors else 0.0,
        density_correlation=density_corr,
        track_id_switches=_UNSUPPORTED,
    )


# ---- performance gate ----

@dataclass
class SliceGateResult:
    """1 slice の development diagnostic 結果。

    recall や instance 数が基準を満たさない場合は passed=False となる。
    formal 合否ではなく開発用 diagnostic。
    """

    slice_name: str
    recall: float | None
    recall_min: float
    n_instances: int
    min_instances: int
    n_sessions: int
    min_sessions: int
    passed: bool

    def to_dict(self) -> dict:
        """slice gate の metric と判定を JSON mapping にする。\n\n不足 session 数も数値と閾値を一緒に出して追跡できる。"""
        return {
            "slice": self.slice_name,
            "recall": self.recall,
            "recall_min": self.recall_min,
            "n_instances": self.n_instances,
            "min_instances": self.min_instances,
            "n_sessions": self.n_sessions,
            "min_sessions": self.min_sessions,
            "passed": self.passed,
        }


@dataclass
class DevDiagnosticsResult:
    """開発用 diagnostics の集計結果。

    実装済み指標（class recall / density / nearest-distance / slice recall）の
    development diagnostic 合否を示す。formal gate / release gate ではない。
    actual weight での formal 合否判定は 04-08 に委譲する。

    gpu_p95_latency_ms は情報表示のみ（unsupported）。passed に影響しない。
    """

    passed: bool
    map50_95: float
    map50_95_min: float
    class_recall_results: dict[str, dict]  # class_name -> {recall, min, passed}
    density_correlation: float
    density_correlation_min: float
    nearest_distance_median: float
    nearest_distance_median_max: float
    # GPU latency は unsupported（04-08 で実装）。passed には影響しない。
    gpu_p95_latency_ms: str | float | None
    slice_results: list[SliceGateResult]

    def to_dict(self) -> dict:
        """development diagnostics の集計結果を JSON mapping にする。\n\n個別 slice の結果も同じ形で含めて CLI 出力に使う。"""
        return {
            "passed": self.passed,
            "map50_95": self.map50_95,
            "map50_95_min": self.map50_95_min,
            "class_recall_results": self.class_recall_results,
            "density_correlation": self.density_correlation,
            "density_correlation_min": self.density_correlation_min,
            "nearest_distance_median": self.nearest_distance_median,
            "nearest_distance_median_max": self.nearest_distance_median_max,
            "gpu_p95_latency_ms": self.gpu_p95_latency_ms,
            "slice_results": [s.to_dict() for s in self.slice_results],
        }


# 後方互換エイリアス（テスト・既存コードが参照している場合のみ）
PerformanceGateResult = DevDiagnosticsResult


def _compute_slice_recall(
    gt_annotations: list[dict],
    pred_annotations: list[dict],
    iou_threshold: float = 0.5,
) -> float:
    """slice の GT / prediction リストから recall を計算する（一対一 matching）。

    GT が 0 件のとき recall=1.0（定義の通り）。
    predict が 0 件のとき recall=0.0。
    GT と prediction は {"image_id", "category_id", "bbox": [x,y,w,h]} を持つ dict。
    """
    if not gt_annotations:
        return 1.0
    if not pred_annotations:
        return 0.0

    gt_by_image: dict[int, list[dict]] = {}
    for a in gt_annotations:
        gt_by_image.setdefault(a["image_id"], []).append(a)

    pred_by_image: dict[int, list[dict]] = {}
    for a in pred_annotations:
        pred_by_image.setdefault(a["image_id"], []).append(a)

    total_gt = 0
    total_tp = 0

    for iid, gts in gt_by_image.items():
        preds = pred_by_image.get(iid, [])
        total_gt += len(gts)

        if not preds:
            continue

        gt_boxes = np.array(
            [[g["bbox"][0], g["bbox"][1], g["bbox"][0] + g["bbox"][2], g["bbox"][1] + g["bbox"][3]]
             for g in gts], dtype=np.float32
        )
        pred_boxes = np.array(
            [[p["bbox"][0], p["bbox"][1], p["bbox"][0] + p["bbox"][2], p["bbox"][1] + p["bbox"][3]]
             for p in preds], dtype=np.float32
        )
        pred_scores = np.array([p.get("score", 1.0) for p in preds], dtype=np.float32)

        iou_mat = compute_iou_matrix(gt_boxes, pred_boxes)
        sorted_pred = np.argsort(-pred_scores)
        matched_gt: set[int] = set()

        for pi in sorted_pred:
            best_iou = 0.0
            best_gi = -1
            for gi in range(len(gts)):
                if gi in matched_gt:
                    continue
                if gts[gi].get("category_id") != preds[pi].get("category_id"):
                    continue
                if iou_mat[gi, pi] > best_iou:
                    best_iou = iou_mat[gi, pi]
                    best_gi = gi
            if best_gi >= 0 and best_iou >= iou_threshold:
                matched_gt.add(best_gi)
                total_tp += 1

    return float(total_tp / total_gt) if total_gt > 0 else 1.0



def compute_dev_diagnostics(
    metrics: EvalMetrics,
    gate_cfg: dict,
    class_name_by_id: dict[int, str],
    *,
    slice_annotations: dict[str, list[dict]] | None = None,
    slice_predictions: dict[str, list[dict]] | None = None,
    gpu_p95_latency_ms: float | None = None,
) -> DevDiagnosticsResult:
    """EvalMetrics と diagnostic 設定を照合し、実装済み指標の合否を返す（development diagnostics）。

    formal gate / release gate ではない。actual weight での formal 合否は 04-08 に委譲する。
    GPU latency / track ID switch は unsupported のため passed に影響しない。
    slice_annotations は {slice_name: gt_annotations} の dict。
    slice_predictions は {slice_name: pred_annotations} の dict（省略時は空予測）。
    """
    all_passed = True

    # --- metric 有限値・有効範囲チェック（fail-closed）---
    # AP/recall は [0, 1]、correlation は [-1, 1]、distance は >= 0 を要求する。
    def _check_metric(name: str, val: float, lo: float, hi: float) -> bool:
        """metric が有限値かつ許容範囲内か判定する。\n\nbool は数値として通さず、壊れた metric を gate 集計から除く。"""
        if isinstance(val, bool) or not (math.isfinite(val) and lo <= val <= hi):
            return False
        return True

    if not _check_metric("proxy_ap50_95", metrics.proxy_ap50_95, 0.0, 1.0):
        all_passed = False
    if not _check_metric("density_correlation", metrics.density_correlation, -1.0, 1.0):
        all_passed = False
    if isinstance(metrics.nearest_distance_error, bool) or not (math.isfinite(metrics.nearest_distance_error) and metrics.nearest_distance_error >= 0.0):
        all_passed = False

    # --- threshold パラメータ自体の有限性・型・範囲を事前検証（fail-closed）---
    # bool は int の派生型のため isinstance(True, int) は True になる。明示的に除外する。
    map_min = gate_cfg.get("map50_95_min", 0.0)
    if not (not isinstance(map_min, bool) and isinstance(map_min, (int, float)) and math.isfinite(map_min) and 0.0 <= map_min <= 1.0):
        all_passed = False
        map_min = 0.0  # 無効な閾値でのさらなる比較を避ける

    density_min = gate_cfg.get("density_correlation_min", 0.85)
    if not (not isinstance(density_min, bool) and isinstance(density_min, (int, float)) and math.isfinite(density_min) and -1.0 <= density_min <= 1.0):
        all_passed = False
        density_min = 1.1  # 無効なので通過不可にする

    nd_max = gate_cfg.get("nearest_distance_median_max", 0.04)
    if not (
        not isinstance(nd_max, bool)
        and isinstance(nd_max, (int, float))
        and math.isfinite(nd_max)
        and 0.0 <= nd_max <= 1.0   # 正規化 distance は [0,1] なので閾値も同範囲
    ):
        all_passed = False
        nd_max = -1.0  # 無効なので通過不可にする

    # --- overall mAP50:95 ---
    if not _check_metric("proxy_ap50_95 vs threshold", metrics.proxy_ap50_95, map_min, 1.0):
        all_passed = False

    # --- class recall ---
    recall_cfg: dict[str, float] = gate_cfg.get("class_recall_min", {})
    class_recall_results: dict[str, dict] = {}
    for class_name, recall_min in recall_cfg.items():
        recall_min_valid = (
            not isinstance(recall_min, bool)
            and isinstance(recall_min, (int, float))
            and math.isfinite(recall_min)
            and 0.0 <= recall_min <= 1.0
        )
        cid = next((k for k, v in class_name_by_id.items() if v == class_name), None)
        recall = metrics.class_recall.get(cid, 0.0) if cid is not None else 0.0
        in_range = not isinstance(recall, bool) and math.isfinite(recall) and 0.0 <= recall <= 1.0
        passed = recall_min_valid and in_range and recall >= recall_min
        if not (recall_min_valid and in_range and passed):
            all_passed = False
        class_recall_results[class_name] = {
            "recall": recall,
            "recall_min": recall_min,
            "passed": passed,
        }

    # --- density correlation ---
    density_corr = metrics.density_correlation
    if not _check_metric("density_correlation vs threshold", density_corr, density_min, 1.0):
        all_passed = False

    # --- nearest-distance median error（正規化 [0,1] を要求） ---
    nd_median = metrics.nearest_distance_error
    if not _check_metric("nearest_distance_error", nd_median, 0.0, 1.0):
        all_passed = False
    elif nd_median > nd_max:
        all_passed = False

    # --- GPU p95 latency: unsupported（passed に影響しない） ---
    # GPU latency は 04-08 で実装する。現在は情報表示のみ。
    gpu_latency_display: str | float | None = (
        "unsupported" if gpu_p95_latency_ms is None else gpu_p95_latency_ms
    )

    # --- slice gates ---
    slice_cfg: dict[str, dict] = gate_cfg.get("slice_gate", {})
    slice_results: list[SliceGateResult] = []
    for slice_name, s_cfg in slice_cfg.items():
        s_recall_min = s_cfg.get("recall_min", 0.80)
        s_min_instances = s_cfg.get("min_instances", 4)
        s_min_sessions = s_cfg.get("min_sessions", 4)

        # slice パラメータ型・範囲を fail-closed で検証する
        s_recall_min_valid = (
            isinstance(s_recall_min, (int, float))
            and not isinstance(s_recall_min, bool)
            and math.isfinite(s_recall_min)
            and 0.0 <= s_recall_min <= 1.0
        )
        s_min_inst_valid = (
            isinstance(s_min_instances, int)
            and not isinstance(s_min_instances, bool)
            and s_min_instances >= 1
        )
        s_min_sess_valid = (
            isinstance(s_min_sessions, int)
            and not isinstance(s_min_sessions, bool)
            and s_min_sessions >= 1
        )
        if not (s_recall_min_valid and s_min_inst_valid and s_min_sess_valid):
            all_passed = False
            slice_results.append(SliceGateResult(
                slice_name=slice_name,
                recall=None,
                recall_min=s_recall_min,
                n_instances=0,
                min_instances=s_min_instances,
                n_sessions=0,
                min_sessions=s_min_sessions,
                passed=False,
            ))
            continue

        s_anns = (slice_annotations or {}).get(slice_name, [])
        n_instances = len(s_anns)
        n_sessions = len({a.get("session_id", "") for a in s_anns if a.get("session_id")})

        # 04-07: 単純な development slice recall。session-cluster CI は 04-08 で実装する。
        if n_instances >= s_min_instances and n_sessions >= s_min_sessions:
            s_preds = (slice_predictions or {}).get(slice_name, [])
            s_recall: float | None = _compute_slice_recall(s_anns, s_preds)
        else:
            s_recall = None

        s_passed = (
            s_recall is not None
            and s_recall >= s_recall_min
            and n_instances >= s_min_instances
            and n_sessions >= s_min_sessions
        )
        if not s_passed:
            all_passed = False

        slice_results.append(SliceGateResult(
            slice_name=slice_name,
            recall=s_recall,
            recall_min=s_recall_min,
            n_instances=n_instances,
            min_instances=s_min_instances,
            n_sessions=n_sessions,
            min_sessions=s_min_sessions,
            passed=s_passed,
        ))

    return DevDiagnosticsResult(
        passed=all_passed,
        map50_95=metrics.proxy_ap50_95,
        map50_95_min=map_min,
        class_recall_results=class_recall_results,
        density_correlation=density_corr,
        density_correlation_min=density_min,
        nearest_distance_median=nd_median,
        nearest_distance_median_max=nd_max,
        gpu_p95_latency_ms=gpu_latency_display,
        slice_results=slice_results,
    )


# 後方互換エイリアス（既存コードが check_performance_gate を直接呼んでいる場合）
check_performance_gate = compute_dev_diagnostics


# ---- CLI ----

def _build_arg_parser() -> argparse.ArgumentParser:
    """development と formal evaluation の CLI 引数を定義する。\n\nformal mode では checkpoint、latency、gates、preflight report を必須にする。"""
    p = argparse.ArgumentParser(
        description="world detector 評価ハーネス",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--annotations", required=True, help="GT COCO JSON")
    p.add_argument("--predictions", help="予測 COCO JSON（省略時は synthetic zero prediction）")
    p.add_argument("--config", default="configs/world_detector_v2.yaml", help="detector config YAML")
    p.add_argument("--class-map", default="configs/world_class_map_v2.yaml", help="class map YAML")
    p.add_argument("--output", help="メトリクス JSON の出力先")
    p.add_argument("--formal", action="store_true", help="checkpoint を in-process 推論して formal verdict を出す")
    p.add_argument("--preflight-report", help="training が発行した canonical preflight report")
    p.add_argument("--checkpoint-dir", help="formal checkpoint directory")
    p.add_argument("--latency", help="target GPU latency measurement JSON")
    p.add_argument("--gates", help="formal gate YAML")
    p.add_argument("--verdict-out", help="formal validation verdict JSON")
    return p


def _formal_eval(args: argparse.Namespace) -> int:
    """checkpoint の実 weight で validation frame を推論して verdict を保存する。

    predictions JSON を一切読まず、raw score から formal threshold を選ぶ。
    """
    import hashlib
    from survivors.vision.world_dataset import WorldDataset, load_class_map
    from survivors.vision.world_detector import CheckpointManifest, WorldDetector, drop_ignored_detections, load_detector_config
    from survivors.vision.world_formal_release import (
        DEFAULT_FORMAL_GATES,
        GpuLatencyMeasurement,
        build_validation_verdict,
        read_preflight_report,
        verify_validation_annotations,
    )

    preflight = read_preflight_report(args.preflight_report)
    verify_validation_annotations(preflight, args.annotations)
    checkpoint_dir = pathlib.Path(args.checkpoint_dir)
    manifest_path = checkpoint_dir / "manifest.json"
    weight_path = checkpoint_dir / "model.pt"
    manifest_bytes = manifest_path.read_bytes()
    manifest = CheckpointManifest.load(manifest_path)
    def file_hash(path: pathlib.Path) -> str:
        """checkpoint file の content hash を計算する。

        manifest と actual weight の identity を読み込み前に照合する。
        """
        return hashlib.sha256(path.read_bytes()).hexdigest()
    weight_hash = file_hash(weight_path)
    config_path = pathlib.Path(args.config)
    class_map_path = pathlib.Path(args.class_map)
    if (
        manifest.training_mode != "formal"
        or manifest.development_only is not False
        or manifest.formal_detector_eligible is not False
        or manifest.model_hash != weight_hash
        or manifest.data_hash != preflight.dataset_identity
        or manifest.build_hash != preflight.build_identity
        or manifest.resolved_config_hash != preflight.config_identity
        or manifest.split_hash != preflight.split_hash
        or manifest.config_hash != file_hash(config_path)
        or manifest.class_map_hash != file_hash(class_map_path)
    ):
        raise ValueError("checkpoint manifest/weight does not match preflight identity")
    cfg = dict(preflight.resolved_config)
    if load_detector_config(config_path)["model"]["architecture"] != cfg["model"]["architecture"]:
        raise ValueError("eval config architecture differs from preflight")
    cm = load_class_map(class_map_path)
    detector = WorldDetector.from_config(cfg, class_map_path)
    try:
        import torch
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        raise RuntimeError("formal eval requires torch") from exc
    try:
        state = torch.load(weight_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(weight_path, map_location="cpu")
    detector._model.load_state_dict(state, strict=True)
    detector._model.eval()
    if hasattr(detector._model, "score_thresh"):
        detector._model.score_thresh = 0.0
    caps = {
        "score_thresh": getattr(detector._model, "score_thresh", 0.0),
        "topk_candidates": getattr(detector._model, "topk_candidates", "unknown"),
        "detections_per_img": getattr(detector._model, "detections_per_img", "unknown"),
    }
    dataset = WorldDataset(args.annotations, class_map_path, validate_bounds=True)
    validation_sessions = set(preflight.validation_sessions)
    detections: dict[str, list[dict[str, Any]]] = {session: [] for session in validation_sessions}
    annotations: dict[str, list[dict[str, Any]]] = {session: [] for session in validation_sessions}
    import cv2
    for sample in dataset:
        if sample.session_id not in validation_sessions:
            continue
        frame = cv2.imread(sample.file_name)
        if frame is None:
            raise OSError(f"formal eval image decode failed: {sample.file_name}")
        result = drop_ignored_detections(
            detector.infer(frame, score_threshold=0.0),
            cfg["input"].get("ignore_regions", []),
        )
        detections[sample.session_id].append({
            "image_id": sample.image_id,
            "image_width": sample.image_width,
            "image_height": sample.image_height,
            "boxes_xyxy": result.boxes_xyxy,
            "scores": result.scores,
            "class_ids": result.class_ids,
        })
        for annotation in sample.annotations:
            annotations[sample.session_id].append({
                "image_id": sample.image_id,
                "category_id": annotation.category_id,
                "bbox": list(annotation.bbox_xywh),
                "image_width": sample.image_width,
                "image_height": sample.image_height,
                "track_id": annotation.track_id,
                "late_game": annotation.late_game,
                "time_band": annotation.time_band,
            })
    try:
        with pathlib.Path(args.latency).open(encoding="utf-8") as stream:
            latency = GpuLatencyMeasurement.from_dict(json.load(stream))
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"formal latency JSON is invalid: {exc}") from exc
    with pathlib.Path(args.gates).open(encoding="utf-8") as stream:
        gates = yaml.safe_load(stream) or {}
    if not isinstance(gates, dict):
        raise ValueError("formal gates YAML must be a mapping")
    threshold_rule = {
        "kind": "max_mean_recall_subject_to_precision",
        "precision_floor": 0.80,
        "candidates": [round(value, 2) for value in np.arange(0.05, 1.0, 0.05)],
    }
    verdict = build_validation_verdict(
        detections_by_session=detections,
        annotations_by_session=annotations,
        class_map=cm,
        gates=gates or {key: DEFAULT_FORMAL_GATES[key] for key in ()},
        latency=latency,
        preflight=preflight,
        checkpoint_manifest=json.loads(manifest_bytes),
        checkpoint_manifest_hash=hashlib.sha256(manifest_bytes).hexdigest(),
        weight_hash=weight_hash,
        threshold_rule=threshold_rule,
        inference_caps=caps,
    )
    verdict.write(args.verdict_out)
    print(json.dumps(verdict.to_dict(), ensure_ascii=False, indent=2))
    return 0 if verdict.status == "PASS" else 1


def main(argv: list[str] | None = None) -> int:
    """CLI を解析し development metric または formal verdict を実行する。\n\nformal 入力の欠落や外部 prediction の指定は処理前に拒否する。"""
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    formal_arguments = (args.preflight_report, args.checkpoint_dir, args.latency, args.gates, args.verdict_out)
    if args.formal:
        if args.predictions:
            parser.error("--formal は外部 --predictions を受け付けません")
        if any(value is None for value in formal_arguments):
            parser.error("--formal には --preflight-report --checkpoint-dir --latency --gates --verdict-out が必要です")
        try:
            return _formal_eval(args)
        except (OSError, ValueError, TypeError, RuntimeError, json.JSONDecodeError) as exc:
            print(f"[FORMAL EVAL FAIL] {exc}", file=sys.stderr)
            return 1
    if any(value is not None for value in formal_arguments):
        parser.error("formal eval inputs require --formal")

    from survivors.vision.world_dataset import WorldDataset, load_class_map
    from survivors.vision.world_detector import load_detector_config

    ann_path = pathlib.Path(args.annotations)
    cfg_path = pathlib.Path(args.config)
    cm_path = pathlib.Path(args.class_map)

    cfg = load_detector_config(cfg_path)
    num_classes = cfg["model"]["num_classes"]

    cm = load_class_map(cm_path)
    if cm.num_classes != num_classes:
        raise ValueError(
            f"config num_classes={num_classes} と class_map num_classes={cm.num_classes} が不一致。"
        )

    gt_data = json.loads(ann_path.read_text(encoding="utf-8"))
    gt_annotations = gt_data.get("annotations", [])
    valid_cat_ids = set(range(cm.num_classes))
    invalid_cats = {ann["category_id"] for ann in gt_annotations} - valid_cat_ids
    if invalid_cats:
        raise ValueError(f"GT に未知の category_id があります: {sorted(invalid_cats)}")

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
