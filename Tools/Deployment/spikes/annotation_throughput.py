"""Annotation throughput and downstream dataset budget calculations."""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Iterable
import numpy as np
from reinbalance_survivors_contracts.ui_intent import ensure


@dataclass(frozen=True)
class AnnotationEvent:
    kind: str
    units: int
    elapsed_seconds: float
    rework_units: int = 0
    dense_frame: bool = False


def summarize_annotation(events: Iterable[AnnotationEvent], *, qa_ious=(),
                         class_matches=()) -> dict[str, float]:
    values = tuple(events)
    ensure(values, "annotation events required")
    total_units = sum(v.units for v in values)
    entity = [v for v in values if v.kind == "entity"]
    entity_units, entity_seconds = sum(v.units for v in entity), sum(v.elapsed_seconds for v in entity)
    dense = [v for v in entity if v.dense_frame]
    dense_units, dense_seconds = sum(v.units for v in dense), sum(v.elapsed_seconds for v in dense)
    return {
        "entities_per_hour": entity_units * 3600 / entity_seconds if entity_seconds else 0.0,
        "dense_entities_per_hour": dense_units * 3600 / dense_seconds if dense_seconds else 0.0,
        "qa_rework_rate": sum(v.rework_units for v in values) / total_units,
        "bbox_qa_iou": float(np.mean(tuple(qa_ious))) if tuple(qa_ious) else 0.0,
        "class_agreement": float(np.mean(tuple(class_matches))) if tuple(class_matches) else 0.0,
        "annotation_hours": sum(v.elapsed_seconds for v in values) / 3600,
    }


@dataclass(frozen=True)
class DatasetSplitRequest:
    name: str
    sessions: int
    frames: int
    entities: int
    ui_events: int


def estimate_dataset_budget(requests: Iterable[DatasetSplitRequest], *,
                            entities_per_hour: float, ui_events_per_hour: float,
                            frame_storage_bytes: int, gpu_seconds_per_frame: float,
                            parallel_worker_limit: int) -> dict:
    values = tuple(requests)
    ensure(values and len({v.name for v in values}) == len(values), "unique splits required")
    ensure(entities_per_hour > 0 and ui_events_per_hour > 0 and
           parallel_worker_limit > 0, "positive rates/workers required")
    splits = {}
    for item in values:
        annotation = item.entities/entities_per_hour + item.ui_events/ui_events_per_hour
        gpu = item.frames*gpu_seconds_per_frame/3600
        splits[item.name] = {
            "sessions": item.sessions, "frames": item.frames, "entities": item.entities,
            "ui_events": item.ui_events, "annotation_hours": annotation,
            "gpu_hours": gpu, "storage_gb": item.frames*frame_storage_bytes/1e9,
            "wall_clock_hours": max(annotation/parallel_worker_limit, gpu),
        }
    totals = {key: sum(value[key] for value in splits.values()) for key in (
        "sessions", "frames", "entities", "ui_events", "annotation_hours",
        "gpu_hours", "storage_gb")}
    totals["parallel_worker_limit"] = parallel_worker_limit
    totals["wall_clock_hours"] = max(totals["annotation_hours"]/parallel_worker_limit,
                                     totals["gpu_hours"])
    return {"splits": splits, "totals": totals}
