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
import math
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

    # AP50:95: PR 曲線台形積分の IoU 閾値平均（COCO 定義）。
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


def compute_ap50_95(
    gt_boxes: np.ndarray,
    gt_classes: np.ndarray,
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    pred_classes: np.ndarray,
) -> float:
    """AP50:95 を PR 曲線台形積分で計算する（COCO 定義: iou_thresholds=[0.5,...,0.95,step=0.05]）。

    各 IoU 閾値で precision-recall 曲線を作り台形積分し、その平均を返す。
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
    """(N,4) box → (N,2) 中心座標。"""
    return np.stack([(box_xyxy[:, 0] + box_xyxy[:, 2]) / 2,
                     (box_xyxy[:, 1] + box_xyxy[:, 3]) / 2], axis=1)


def _compute_class_ap_coco(
    gt_by_image: dict[int, list[dict]],
    pred_by_image: dict[int, list[dict]],
    class_id: int,
) -> float:
    """1 クラス分の AP50:95 を台形積分で計算する（confidence降順・同一image内GTのみ1:1マッチ）。

    全画像の予測を confidence 降順で走査し、各予測は同じ画像の GT のみとマッチできる。
    画像をまたいだ誤 TP を防ぐ。
    注意: 台形積分（np.trapz）を使用。COCO 公式の 101 点補間とは値が僅かに異なる。
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

    # COCO mAP: 同一画像内 GT のみとマッチする image-aware 計算
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
    """1 slice の performance gate 結果。

    recall や instance 数が基準を満たさない場合は passed=False となる。
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
        """JSON 出力用 dict。"""
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
class PerformanceGateResult:
    """全体 performance gate の合否とその根拠。

    synthetic metrics でゲートロジックを検証できる。
    actual weight での合格は 04-08 に委譲する。
    """

    passed: bool
    map50_95: float
    map50_95_min: float
    class_recall_results: dict[str, dict]  # class_name -> {recall, min, passed}
    density_correlation: float
    density_correlation_min: float
    nearest_distance_median: float
    nearest_distance_median_max: float
    gpu_p95_latency_ms: float | None
    gpu_p95_latency_max_ms: float
    slice_results: list[SliceGateResult]

    def to_dict(self) -> dict:
        """JSON 出力用 dict。"""
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
            "gpu_p95_latency_max_ms": self.gpu_p95_latency_max_ms,
            "slice_results": [s.to_dict() for s in self.slice_results],
        }


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


def _compute_session_ci_recall(
    gt_annotations: list[dict],
    pred_annotations: list[dict],
    iou_threshold: float = 0.5,
    z: float = 1.645,
) -> float | None:
    """セッション単位の recall 分布から cluster CI 下限を返す。

    全インスタンスを pool した recall ではなく、セッションごとの recall 平均の
    信頼区間下限（z=1.645 で 90% 片側）を返す。1 つのセッションだけが好成績で
    他が全滅するモデルでも gate を通過できなくなる。
    セッションが 1 件のみの場合は recall をそのまま返す。
    """
    # session ごとに GT をグループ化（image_id でもインデックス化）
    session_gts: dict[str, list[dict]] = {}
    session_images: dict[str, set[int]] = {}
    for a in gt_annotations:
        sid = a.get("session_id", "")
        session_gts.setdefault(sid, []).append(a)
        session_images.setdefault(sid, set()).add(a["image_id"])

    if not session_gts:
        return None

    # per-session recall: prediction は session 内 image_id でマッチ（session_id 不要）
    recalls: list[float] = []
    for sid, s_gts in session_gts.items():
        s_img_ids = session_images[sid]
        s_preds = [a for a in pred_annotations if a.get("image_id") in s_img_ids]
        r = _compute_slice_recall(s_gts, s_preds, iou_threshold)
        recalls.append(r)

    n = len(recalls)
    if n == 1:
        return recalls[0]

    mean_r = sum(recalls) / n
    # 不偏標本分散 → SE → t/z 近似 CI 下限
    variance = sum((r - mean_r) ** 2 for r in recalls) / (n - 1)
    se = (variance / n) ** 0.5
    # ponytail: z 近似（n>=4 前提）。n<4 なら min_sessions 条件で gate 自体が FAIL する。
    return max(0.0, mean_r - z * se)


def check_performance_gate(
    metrics: EvalMetrics,
    gate_cfg: dict,
    class_name_by_id: dict[int, str],
    *,
    slice_annotations: dict[str, list[dict]] | None = None,
    slice_predictions: dict[str, list[dict]] | None = None,
    gpu_p95_latency_ms: float | None = None,
    num_classes: int = 12,
) -> PerformanceGateResult:
    """EvalMetrics と gate_cfg を照合してパス / フェイルを判定する。

    全ゲートを実行して個別結果を返す。1 つでも FAIL なら全体 passed=False。
    slice_annotations は {slice_name: gt_annotations} の dict。
    slice_predictions は {slice_name: pred_annotations} の dict（省略時は空予測）。
    slice recall は一対一 IoU matching で計算する。
    synthetic 値でゲートロジック自体を検証できるよう設計する。
    """
    all_passed = True

    # --- 必須 metric の存在・有限値チェック（fail-closed） ---
    _required_finite = [
        ("proxy_ap50_95", metrics.proxy_ap50_95),
        ("density_correlation", metrics.density_correlation),
        ("nearest_distance_error", metrics.nearest_distance_error),
    ]
    for _name, _val in _required_finite:
        if not math.isfinite(_val):
            all_passed = False

    # GPU latency が gate で要求される場合は None も FAIL
    latency_max = gate_cfg.get("gpu_p95_latency_max_ms", 25.0)
    if "gpu_p95_latency_max_ms" in gate_cfg and gpu_p95_latency_ms is None:
        all_passed = False
    elif gpu_p95_latency_ms is not None and not math.isfinite(gpu_p95_latency_ms):
        all_passed = False

    # --- overall mAP50:95 ---
    map_min = gate_cfg.get("map50_95_min", 0.0)
    if metrics.proxy_ap50_95 < map_min:
        all_passed = False

    # --- class recall ---
    recall_cfg: dict[str, float] = gate_cfg.get("class_recall_min", {})
    class_recall_results: dict[str, dict] = {}
    for class_name, recall_min in recall_cfg.items():
        # class_id を class_name から逆引きする
        cid = next((k for k, v in class_name_by_id.items() if v == class_name), None)
        recall = metrics.class_recall.get(cid, 0.0) if cid is not None else 0.0
        passed = math.isfinite(recall) and recall >= recall_min
        if not passed:
            all_passed = False
        class_recall_results[class_name] = {
            "recall": recall,
            "recall_min": recall_min,
            "passed": passed,
        }

    # --- density correlation ---
    density_corr = metrics.density_correlation
    density_min = gate_cfg.get("density_correlation_min", 0.85)
    if density_corr < density_min:
        all_passed = False

    # --- nearest-distance median error ---
    nd_median = metrics.nearest_distance_error  # median（全対象距離の実 median）
    nd_max = gate_cfg.get("nearest_distance_median_max", 0.04)
    if nd_median > nd_max:
        all_passed = False

    # --- GPU p95 latency （NaN チェックは冒頭で実施済み）---
    if gpu_p95_latency_ms is not None and math.isfinite(gpu_p95_latency_ms) \
            and gpu_p95_latency_ms > latency_max:
        all_passed = False

    # --- slice gates ---
    slice_cfg: dict[str, dict] = gate_cfg.get("slice_gate", {})
    slice_results: list[SliceGateResult] = []
    for slice_name, s_cfg in slice_cfg.items():
        s_recall_min = s_cfg.get("recall_min", 0.80)
        s_min_instances = s_cfg.get("min_instances", 4)
        s_min_sessions = s_cfg.get("min_sessions", 4)

        s_anns = (slice_annotations or {}).get(slice_name, [])
        n_instances = len(s_anns)
        n_sessions = len({a.get("session_id", "") for a in s_anns if a.get("session_id")})

        # recall はセッション分布の CI 下限で判定する（pooled recall では大規模 session が支配的）
        if n_instances >= s_min_instances and n_sessions >= s_min_sessions:
            s_preds = (slice_predictions or {}).get(slice_name, [])
            s_recall: float | None = _compute_session_ci_recall(s_anns, s_preds)
        else:
            s_recall = None  # インスタンス/セッション不足 → gate 自体を FAIL

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

    return PerformanceGateResult(
        passed=all_passed,
        map50_95=metrics.proxy_ap50_95,
        map50_95_min=map_min,
        class_recall_results=class_recall_results,
        density_correlation=density_corr,
        density_correlation_min=density_min,
        nearest_distance_median=nd_median,
        nearest_distance_median_max=nd_max,
        gpu_p95_latency_ms=gpu_p95_latency_ms,
        gpu_p95_latency_max_ms=latency_max,
        slice_results=slice_results,
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
