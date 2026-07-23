from __future__ import annotations

import pytest

from spikes.annotation_throughput import (
    AnnotationEvent,
    DatasetSplitRequest,
    estimate_dataset_budget,
    summarize_annotation,
)
from spikes.perception_probe import make_synthetic_fixture


def test_annotation_timing_and_qa_summary_is_reproducible():
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
    samples = make_synthetic_fixture(seed=5)
    assert {sample.kind for sample in samples} >= {"timer", "icon", "entity"}
    assert all(sample.annotation_seconds > 0 for sample in samples)
