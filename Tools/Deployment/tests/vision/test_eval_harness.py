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
