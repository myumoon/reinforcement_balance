"""アノテーション集計とデータセット予算の契約テスト。

正常な集計の再現性と、無効な件数・速度・品質値が拒否されることを検証します。
"""
from __future__ import annotations

import pytest

from reinbalance_survivors_contracts.ui_intent import ContractValidationError
from spikes.annotation_throughput import (
    AnnotationEvent,
    DatasetSplitRequest,
    estimate_dataset_budget,
    summarize_annotation,
)
from spikes.perception_probe import make_synthetic_fixture


def test_annotation_timing_and_qa_summary_is_reproducible():
    """作業速度と QA 指標が期待値どおり集計されることを確認する。

    固定した作業実績を渡し、毎時処理数、手直し率、一致度を比較します。
    """
    events = [
        AnnotationEvent("frame", 10, 36.0, 0, False),
        AnnotationEvent("entity", 20, 120.0, 2, True),
        AnnotationEvent("ui_event", 4, 48.0, 0, False),
    ]
    summary = summarize_annotation(events, qa_ious=[0.9, 0.8], class_matches=[True, False])
    assert summary["entities_per_hour"] == 600.0
    assert summary["qa_rework_rate"] == 2 / 34
    assert summary["bbox_qa_iou"] == pytest.approx(0.85)
    assert summary["class_agreement"] == 0.5


def test_budget_covers_every_split_and_resource_dimension():
    """全分割と全資源項目が予算へ含まれることを確認する。

    三つの用途別要求を見積もり、合計件数と必要な指標を検査します。
    """
    requests = [
        DatasetSplitRequest("development", 3, 1200, 24000, 180),
        DatasetSplitRequest("calibration", 2, 600, 12000, 90),
        DatasetSplitRequest("untouched_final", 2, 600, 12000, 90),
    ]
    budget = estimate_dataset_budget(
        requests,
        entities_per_hour=400,
        ui_events_per_hour=120,
        frame_storage_bytes=1920 * 1080 * 4,
        gpu_seconds_per_frame=0.05,
        parallel_worker_limit=4,
    )
    assert set(budget["splits"]) == {"development", "calibration", "untouched_final"}
    assert budget["totals"]["sessions"] == 7
    for key in ("frames", "entities", "ui_events", "annotation_hours",
                "gpu_hours", "storage_gb", "wall_clock_hours", "parallel_worker_limit"):
        assert key in budget["totals"]
    assert budget["totals"]["parallel_worker_limit"] == 4


def test_synthetic_fixture_contains_timer_icon_entity_and_annotation_data():
    """合成データが必要な対象種別と注釈時間を持つことを確認する。

    タイマー、アイコン、物体が含まれ、各観測の作業時間が正であるか調べます。
    """
    samples = make_synthetic_fixture(seed=5)
    assert {sample.kind for sample in samples} >= {"timer", "icon", "entity"}
    assert all(sample.annotation_seconds > 0 for sample in samples)


@pytest.mark.parametrize(
    "args",
    [
        ("entity", -1, 1.0),
        ("entity", 1, float("nan")),
        ("entity", 1, 1.0, -1),
        ("", 1, 1.0),
        ("entity", True, 1.0),
    ],
)
def test_annotation_event_rejects_invalid_fields_at_construction(args):
    """不正な作業実績が生成時に拒否されることを確認する。

    負数、NaN、空の種別、真偽値を件数に使う例を試します。
    """
    with pytest.raises(ContractValidationError):
        AnnotationEvent(*args)


@pytest.mark.parametrize("qa", [[-0.1], [1.1], [float("inf")], [True]])
def test_annotation_summary_rejects_invalid_qa_inputs(qa):
    """範囲外または有限でない QA 値が拒否されることを確認する。

    IoU として扱えない値を集計へ渡し、契約エラーになるか検査します。
    """
    with pytest.raises(ContractValidationError):
        summarize_annotation([AnnotationEvent("entity", 1, 1.0)], qa_ious=qa)


@pytest.mark.parametrize(
    "args",
    [
        ("", 1, 1, 1, 1),
        ("dev", -1, 1, 1, 1),
        ("dev", 1, True, 1, 1),
    ],
)
def test_dataset_split_rejects_invalid_fields_at_construction(args):
    """不正なデータ分割要求が生成時に拒否されることを確認する。

    空名、負数、件数としての真偽値をそれぞれ試します。
    """
    with pytest.raises(ContractValidationError):
        DatasetSplitRequest(*args)


@pytest.mark.parametrize(
    "override",
    [
        {"entities_per_hour": float("nan")},
        {"ui_events_per_hour": float("inf")},
        {"frame_storage_bytes": -1},
        {"gpu_seconds_per_frame": -0.1},
        {"parallel_worker_limit": 1.5},
    ],
)
def test_budget_rejects_invalid_rates_and_resources(override):
    """不正な処理速度や資源値が予算計算で拒否されることを確認する。

    NaN、無限大、負数、整数でない並列数を一項目ずつ差し替えます。
    """
    kwargs = {
        "entities_per_hour": 400,
        "ui_events_per_hour": 120,
        "frame_storage_bytes": 100,
        "gpu_seconds_per_frame": 0.05,
        "parallel_worker_limit": 2,
    }
    kwargs.update(override)
    with pytest.raises(ContractValidationError):
        estimate_dataset_budget([DatasetSplitRequest("dev", 1, 1, 1, 1)], **kwargs)
