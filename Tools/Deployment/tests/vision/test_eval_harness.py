"""eval_survivors_world_detector の metric harness テスト。

class recall の一対一 matching、density_error / nearest_distance_error の計算、
track_id_switches の unsupported 明示を synthetic prediction で検証する。
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from eval_survivors_world_detector import evaluate_from_predictions, EvalMetrics, _UNSUPPORTED


def _gt(image_id: int, bbox_xywh: list, category_id: int) -> dict:
    x, y, w, h = bbox_xywh
    return {"image_id": image_id, "category_id": category_id, "bbox": [x, y, w, h], "id": image_id * 100 + category_id}


def _pred(image_id: int, bbox_xywh: list, category_id: int, score: float = 0.9) -> dict:
    x, y, w, h = bbox_xywh
    return {"image_id": image_id, "category_id": category_id, "bbox": [x, y, w, h], "score": score, "id": image_id * 200 + category_id}


class TestClassRecallOneToOne:
    """class recall の一対一 matching を検証する（P1 修正）。"""

    def test_one_pred_matches_only_one_gt(self):
        """GT が 2 件・予測 1 件のとき recall は最大 0.5。"""
        gts = [
            _gt(0, [100, 100, 50, 50], 2),
            _gt(0, [110, 110, 50, 50], 2),  # 同クラス GT 2 件
        ]
        preds = [_pred(0, [100, 100, 50, 50], 2, score=0.9)]  # 予測 1 件
        metrics = evaluate_from_predictions(gts, preds, num_classes=12)
        # 一対一 matching → recall = 1/2 = 0.5、1.0 にはならない
        assert metrics.class_recall[2] <= 0.5 + 1e-6

    def test_perfect_recall_when_all_gt_matched(self):
        """GT ≡ 予測 1 件ずつのとき recall = 1.0。"""
        gts = [_gt(0, [100, 100, 50, 50], 3)]
        preds = [_pred(0, [100, 100, 50, 50], 3)]
        metrics = evaluate_from_predictions(gts, preds, num_classes=12)
        assert abs(metrics.class_recall[3] - 1.0) < 1e-6


class TestDensityAndNearestDistance:
    """density_error と nearest_distance_error が計算されることを確認する（P1 修正）。"""

    def test_density_error_nonzero_on_mismatch(self):
        gts = [_gt(0, [0, 0, 50, 50], 1), _gt(0, [100, 0, 50, 50], 1)]
        preds = [_pred(0, [0, 0, 50, 50], 1)]  # 1 件少ない
        metrics = evaluate_from_predictions(gts, preds, num_classes=12)
        assert metrics.density_error > 0.0

    def test_nearest_distance_error_nonzero(self):
        gts = [_gt(0, [0, 0, 50, 50], 2)]
        preds = [_pred(0, [500, 500, 50, 50], 2)]  # 遠い場所に予測
        metrics = evaluate_from_predictions(gts, preds, num_classes=12)
        assert metrics.nearest_distance_error > 0.0

    def test_nearest_distance_zero_on_perfect_match(self):
        gts = [_gt(0, [100, 100, 50, 50], 3)]
        preds = [_pred(0, [100, 100, 50, 50], 3)]
        metrics = evaluate_from_predictions(gts, preds, num_classes=12)
        assert metrics.nearest_distance_error < 1e-3


class TestComputeAp50_95:
    """PR 曲線台形積分による AP 計算を検証する。"""

    def test_perfect_tp_gives_ap_one(self):
        """1 GT + 高 conf TP + 低 conf FP → AP=1.0（precision*recall では 0.5 になるケース）。"""
        from eval_survivors_world_detector import compute_ap50_95
        import numpy as np
        gt_boxes = np.array([[0, 0, 50, 50]], dtype=np.float32)
        gt_classes = np.array([1], dtype=np.int32)
        pred_boxes = np.array([[0, 0, 50, 50], [200, 200, 250, 250]], dtype=np.float32)
        pred_scores = np.array([0.9, 0.1], dtype=np.float32)
        pred_classes = np.array([1, 1], dtype=np.int32)
        ap = compute_ap50_95(gt_boxes, gt_classes, pred_boxes, pred_scores, pred_classes)
        assert abs(ap - 1.0) < 1e-4, f"期待 AP=1.0、実際={ap}"

    def test_no_preds_gives_ap_zero(self):
        from eval_survivors_world_detector import compute_ap50_95
        import numpy as np
        gt_boxes = np.array([[0, 0, 50, 50]], dtype=np.float32)
        gt_classes = np.array([1], dtype=np.int32)
        ap = compute_ap50_95(gt_boxes, gt_classes, np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int32))
        assert ap == 0.0

    def test_no_gt_no_pred_gives_ap_one(self):
        from eval_survivors_world_detector import compute_ap50_95
        import numpy as np
        ap = compute_ap50_95(np.zeros((0, 4), np.float32), np.zeros(0, np.int32), np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int32))
        assert ap == 1.0


class TestNearestDistanceNormalization:
    """nearest_distance_error が viewport 正規化されることを検証する。"""

    def test_large_offset_below_one(self):
        """1920×1080 内の最大距離（対角線 ≈ 2203px）でも 1.0 以下になる。"""
        import numpy as np
        from eval_survivors_world_detector import evaluate_from_predictions
        gts = [{"image_id": 0, "category_id": 1, "bbox": [0, 0, 10, 10]}]
        preds = [{"image_id": 0, "category_id": 1, "bbox": [1910, 1070, 10, 10], "score": 0.9}]
        metrics = evaluate_from_predictions(gts, preds, num_classes=12)
        assert 0.0 < metrics.nearest_distance_error <= 1.0, \
            f"viewport 正規化後は 0〜1 の範囲を期待、実際={metrics.nearest_distance_error}"

    def test_no_prediction_gives_one(self):
        """GT ありで予測なし → 正規化後の最大誤差 1.0。"""
        import numpy as np
        from eval_survivors_world_detector import evaluate_from_predictions
        gts = [{"image_id": 0, "category_id": 1, "bbox": [0, 0, 50, 50]}]
        metrics = evaluate_from_predictions(gts, [], num_classes=12)
        assert abs(metrics.nearest_distance_error - 1.0) < 1e-6

    def test_perfect_match_gives_zero(self):
        from eval_survivors_world_detector import evaluate_from_predictions
        gts = [{"image_id": 0, "category_id": 1, "bbox": [100, 100, 50, 50]}]
        preds = [{"image_id": 0, "category_id": 1, "bbox": [100, 100, 50, 50], "score": 0.9}]
        metrics = evaluate_from_predictions(gts, preds, num_classes=12)
        assert metrics.nearest_distance_error < 1e-6


class TestMapCrossImage:
    """cross-image mAP 集計の正確性を検証する（P1 修正）。"""

    def test_fp_in_different_image_does_not_lower_ap(self):
        """画像0にTP(高conf)、画像1に空GT+FP(低conf) → dataset全体のAP=1.0。"""
        gts = [_gt(0, [0, 0, 50, 50], 1)]
        preds = [
            _pred(0, [0, 0, 50, 50], 1, score=0.9),   # image 0: TP
            _pred(1, [200, 200, 50, 50], 1, score=0.1),  # image 1: FP (no GT)
        ]
        metrics = evaluate_from_predictions(gts, preds, num_classes=12)
        assert abs(metrics.proxy_ap50_95 - 1.0) < 1e-4, (
            f"クロス画像でFPが後続 → AP=1.0 を期待、実際={metrics.proxy_ap50_95}"
        )

    def test_per_class_map_average(self):
        """class1を完全検出・class2を全欠落 → class平均mAP=0.5。"""
        gts = [_gt(0, [0, 0, 50, 50], 1), _gt(0, [200, 0, 50, 50], 2)]
        preds = [_pred(0, [0, 0, 50, 50], 1, score=0.9)]  # class2の予測なし
        metrics = evaluate_from_predictions(gts, preds, num_classes=12)
        assert abs(metrics.proxy_ap50_95 - 0.5) < 1e-4, (
            f"class1=AP1.0, class2=AP0.0 → 平均0.5を期待、実際={metrics.proxy_ap50_95}"
        )

    def test_cross_image_position_swap_gives_zero_map(self):
        """image0とimage1でGT/予測の位置を入れ替えたとき、mAP=0.0（class recall=0.0）。"""
        # image 0: GT が左、Pred が右 → IoU=0
        # image 1: GT が右、Pred が左 → IoU=0
        gts = [_gt(0, [50, 200, 50, 50], 1), _gt(1, [1800, 200, 50, 50], 1)]
        preds = [_pred(0, [1800, 200, 50, 50], 1, score=0.9),
                 _pred(1, [50, 200, 50, 50], 1, score=0.8)]
        metrics = evaluate_from_predictions(gts, preds, num_classes=12)
        assert metrics.proxy_ap50_95 < 1e-4, (
            f"位置入れ替えは全FP → mAP≈0を期待、実際={metrics.proxy_ap50_95}"
        )
        assert metrics.class_recall.get(1, 0.0) < 1e-4


class TestDensityCorrelationImageAware:
    """density_correlation が画像ごとに計算されることを検証する（P1 修正）。"""

    def test_cross_image_position_swap_gives_low_correlation(self):
        """image0/1 で GT/予測の左右を入れ替えると correlation は低くなる。"""
        # image 0: GT 左 (x≈100)、Pred 右 (x≈1800)
        # image 1: GT 右 (x≈1800)、Pred 左 (x≈100)
        gts = [_gt(0, [100, 500, 50, 50], 1), _gt(1, [1800, 500, 50, 50], 1)]
        preds = [_pred(0, [1800, 500, 50, 50], 1, score=0.9),
                 _pred(1, [100, 500, 50, 50], 1, score=0.8)]
        metrics = evaluate_from_predictions(gts, preds, num_classes=12)
        assert metrics.density_correlation < 0.5, (
            f"位置が全画像でずれているので correlation は低くなるはず、実際={metrics.density_correlation}"
        )


class TestNearestDistanceMedian:
    """nearest_distance_error が実 median で計算されることを検証する（P1 修正）。"""

    def test_median_not_mean_on_skewed_distances(self):
        """正規化距離 [0, 0, large] → median≈0 で通過、mean は large/3 になる。"""
        vp_diag = float(np.hypot(1920, 1080))
        # 3 GT、pred1 は完全一致×2、pred3 は遠い場所
        gts = [
            _gt(0, [100, 100, 10, 10], 1),
            _gt(0, [200, 100, 10, 10], 1),
            _gt(0, [300, 100, 10, 10], 1),
        ]
        preds = [
            _pred(0, [100, 100, 10, 10], 1, score=0.9),  # GT1 に完全一致
            _pred(0, [200, 100, 10, 10], 1, score=0.8),  # GT2 に完全一致
            _pred(0, [1900, 1070, 10, 10], 1, score=0.7),  # GT3 から遠い
        ]
        metrics = evaluate_from_predictions(gts, preds, num_classes=12)
        # distances: [0, 0, ~vp_diag] / vp_diag → median=0
        assert metrics.nearest_distance_error < 1e-6, (
            f"median は 0 を期待（外れ値 1 つは median を変えない）、実際={metrics.nearest_distance_error}"
        )


class TestTrackIdSwitchesUnsupported:
    """track_id_switches は静的 eval で計算不能 → unsupported を明示する（P1 修正）。"""

    def test_track_id_switches_is_unsupported(self):
        gts = [_gt(0, [0, 0, 50, 50], 1)]
        preds = [_pred(0, [0, 0, 50, 50], 1)]
        metrics = evaluate_from_predictions(gts, preds, num_classes=12)
        assert metrics.track_id_switches == _UNSUPPORTED

    def test_to_json_contains_unsupported_marker(self):
        gts = [_gt(0, [0, 0, 50, 50], 1)]
        preds = []
        metrics = evaluate_from_predictions(gts, preds, num_classes=12)
        js = json.loads(metrics.to_json())
        assert js["track_id_switches"] == _UNSUPPORTED
