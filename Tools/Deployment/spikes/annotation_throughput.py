"""アノテーション処理量とデータセット予算を算出するモジュール。

画像への正解付けにかかる時間や品質を集計し、後続のデータ作成に必要な
作業時間・GPU時間・保存容量の見積もりを分割単位で作ります。
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Any, Iterable
import numpy as np
from reinbalance_survivors_contracts.ui_intent import ensure, is_strict_number


def _non_negative_int(value: Any, label: str) -> None:
    """値が非負の整数であることを検証する。

    件数や枚数として使う値に、負数や小数、真偽値が紛れ込むのを防ぎます。
    """
    ensure(type(value) is int and value >= 0, f"{label} must be a non-negative integer")


def _finite(value: Any, label: str, *, positive: bool = False) -> None:
    """値が有限の非負数または正数であることを検証する。

    時間や処理速度の計算を壊す NaN・無限大・不適切な負数を早めに拒否します。
    """
    ensure(is_strict_number(value) and math.isfinite(value) and
           (value > 0 if positive else value >= 0),
           f"{label} must be {'positive' if positive else 'non-negative'} and finite")


@dataclass(frozen=True)
class AnnotationEvent:
    """ひとまとまりのアノテーション作業実績を表す。

    作業種別、処理数、所要時間、手直し数、密集フレームかどうかを保持します。
    """
    kind: str
    units: int
    elapsed_seconds: float
    rework_units: int = 0
    dense_frame: bool = False

    def __post_init__(self) -> None:
        """生成直後に全フィールドを検証する。

        不正な作業実績が集計処理へ渡らないよう、作成時点で確認します。
        """
        self.validate()

    def validate(self) -> None:
        """アノテーション実績の型と値の範囲を検証する。

        作業種別が空でないことや、手直し数が処理数を超えないことを確かめます。
        """
        ensure(isinstance(self.kind, str) and self.kind, "annotation kind must be non-empty")
        _non_negative_int(self.units, "annotation units")
        _finite(self.elapsed_seconds, "annotation elapsed seconds")
        _non_negative_int(self.rework_units, "annotation rework units")
        ensure(self.rework_units <= self.units, "rework units cannot exceed annotation units")
        ensure(type(self.dense_frame) is bool, "dense frame must be boolean")


def summarize_annotation(events: Iterable[AnnotationEvent], *, qa_ious=(),
                         class_matches=()) -> dict[str, float]:
    """アノテーション速度と品質指標を集計する。

    複数の作業実績から毎時処理数、手直し率、矩形一致度、クラス一致率、
    合計作業時間を再現可能な形で求めます。
    """
    values = tuple(events)
    ensure(values and all(isinstance(v, AnnotationEvent) for v in values),
           "annotation events required")
    for value in values:
        value.validate()
    qa_values, match_values = tuple(qa_ious), tuple(class_matches)
    ensure(all(is_strict_number(v) and math.isfinite(v) and 0 <= v <= 1
               for v in qa_values), "QA IoUs must be finite numbers between zero and one")
    ensure(all(type(v) is bool for v in match_values), "class matches must be booleans")
    total_units = sum(v.units for v in values)
    ensure(total_units > 0, "annotation events must contain at least one unit")
    entity = [v for v in values if v.kind == "entity"]
    entity_units, entity_seconds = sum(v.units for v in entity), sum(v.elapsed_seconds for v in entity)
    dense = [v for v in entity if v.dense_frame]
    dense_units, dense_seconds = sum(v.units for v in dense), sum(v.elapsed_seconds for v in dense)
    return {
        "entities_per_hour": entity_units * 3600 / entity_seconds if entity_seconds else 0.0,
        "dense_entities_per_hour": dense_units * 3600 / dense_seconds if dense_seconds else 0.0,
        "qa_rework_rate": sum(v.rework_units for v in values) / total_units,
        "bbox_qa_iou": float(np.mean(qa_values)) if qa_values else 0.0,
        "class_agreement": float(np.mean(match_values)) if match_values else 0.0,
        "annotation_hours": sum(v.elapsed_seconds for v in values) / 3600,
    }


@dataclass(frozen=True)
class DatasetSplitRequest:
    """データセット分割ごとの作成量を表す。

    開発用や最終評価用などの区分ごとに、セッション・画像・物体・UIイベントの
    必要数をまとめます。
    """
    name: str
    sessions: int
    frames: int
    entities: int
    ui_events: int

    def __post_init__(self) -> None:
        """生成直後に分割要求を検証する。

        不正な名前や負の件数を持つ要求を、予算計算の前に拒否します。
        """
        self.validate()

    def validate(self) -> None:
        """分割名と各必要数を検証する。

        名前が空でなく、すべての数量が非負の整数であることを確かめます。
        """
        ensure(isinstance(self.name, str) and self.name, "split name must be non-empty")
        for label in ("sessions", "frames", "entities", "ui_events"):
            _non_negative_int(getattr(self, label), f"split {label}")


def estimate_dataset_budget(requests: Iterable[DatasetSplitRequest], *,
                            entities_per_hour: float, ui_events_per_hour: float,
                            frame_storage_bytes: int, gpu_seconds_per_frame: float,
                            parallel_worker_limit: int) -> dict:
    """データセット作成に必要な資源と時間を見積もる。

    各分割の数量と処理速度から、アノテーション時間、GPU時間、保存容量、
    並列作業を考慮した経過時間を分割別・合計で算出します。
    """
    values = tuple(requests)
    ensure(values and all(isinstance(v, DatasetSplitRequest) for v in values),
           "dataset split requests required")
    for value in values:
        value.validate()
    ensure(len({v.name for v in values}) == len(values), "unique splits required")
    _finite(entities_per_hour, "entities per hour", positive=True)
    _finite(ui_events_per_hour, "UI events per hour", positive=True)
    _non_negative_int(frame_storage_bytes, "frame storage bytes")
    _finite(gpu_seconds_per_frame, "GPU seconds per frame")
    ensure(type(parallel_worker_limit) is int and parallel_worker_limit > 0,
           "parallel worker limit must be a positive integer")
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
