"""Target telemetry と simulator telemetry を整列して fidelity 差と cluster CI を計算する。

実 video がない開発環境でも、抽出済み frame fixture から同じ計測ロジックを検証できます。
計測不能値は null と理由で保持し、policy 入力へ推測値を入れません。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class TelemetrySample:
    """単一時点の screen-measurable telemetry。

    session/time と任意 metric を検証し、非 finite 値を集計へ渡しません。
    """
    session_id: str
    time_seconds: float
    metrics: Mapping[str, float | None]
    uncertainties: Mapping[str, str]

    def __post_init__(self) -> None:
        """全 metric と uncertainty の型・有限性・対応を検証する。

        None 値には必ず accepted uncertainty を要求し、数値には uncertainty を禁止します。
        """
        if not isinstance(self.session_id, str) or not self.session_id:
            raise ValueError("session_id must be non-empty")
        if isinstance(self.time_seconds, bool) or not isinstance(self.time_seconds, (int, float)) or not math.isfinite(self.time_seconds) or self.time_seconds < 0:
            raise ValueError("time_seconds must be finite and non-negative")
        if not isinstance(self.metrics, Mapping) or not isinstance(self.uncertainties, Mapping):
            raise ValueError("metrics and uncertainties must be objects")
        for name, value in self.metrics.items():
            if not isinstance(name, str) or not name:
                raise ValueError("metric name must be non-empty")
            if value is None:
                if not isinstance(self.uncertainties.get(name), str) or not self.uncertainties[name]:
                    raise ValueError("unmeasurable metric requires accepted uncertainty")
            elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("metric value must be finite numeric or null")
            elif name in self.uncertainties:
                raise ValueError("measurable metric cannot have uncertainty")
        if set(self.uncertainties) - {name for name, value in self.metrics.items() if value is None}:
            raise ValueError("orphan uncertainty")

    @classmethod
    def from_wire(cls, value: Any) -> "TelemetrySample":
        """frame wire object を exact-key 検証して sample に変換する。

        未知 key や欠落 key を黙認しません。
        """
        if not isinstance(value, Mapping) or set(value) != {"session_id", "time_seconds", "metrics", "uncertainties"}:
            raise ValueError("telemetry sample keys mismatch")
        return cls(value["session_id"], value["time_seconds"], dict(value["metrics"]), dict(value["uncertainties"]))


@dataclass(frozen=True)
class MetricDifference:
    """target-simulator 差と session-cluster CI。

    session 平均を標本として扱い、frame 数の多い session に偏らない集計を表します。
    """
    metric: str
    mean_difference: float
    ci_low: float
    ci_high: float
    session_count: int


def extract_action_telemetry(frames: Sequence[Mapping[str, Any]]) -> tuple[TelemetrySample, ...]:
    """抽出済み video/telemetry frame から監査 sample を生成する。

    direction/speed/cadence/timer/level/density/offer/chest/terminal を入力 metric 名のまま保持します。
    """
    if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)):
        raise ValueError("frames must be an array")
    return tuple(TelemetrySample.from_wire(frame) for frame in frames)


def align_and_compare(target: Sequence[TelemetrySample], simulator: Sequence[TelemetrySample], *, time_tolerance: float = 0.1) -> tuple[MetricDifference, ...]:
    """同 session/time band の sample を整列し viewport/unit 換算後の差を集計する。

    `*_px` は各 sample の viewport_width で正規化し、None metric は policy 比較から除外します。
    """
    if not math.isfinite(time_tolerance) or time_tolerance < 0:
        raise ValueError("time_tolerance must be finite and non-negative")
    by_session: dict[str, list[dict[str, float]]] = {}
    for left in target:
        candidates = [right for right in simulator if right.session_id == left.session_id and abs(right.time_seconds - left.time_seconds) <= time_tolerance]
        if not candidates:
            continue
        right = min(candidates, key=lambda row: abs(row.time_seconds - left.time_seconds))
        row: dict[str, float] = {}
        for name in set(left.metrics) & set(right.metrics):
            lv, rv = left.metrics[name], right.metrics[name]
            if lv is None or rv is None or name == "viewport_width":
                continue
            if name.endswith("_px"):
                lw, rw = left.metrics.get("viewport_width"), right.metrics.get("viewport_width")
                if not lw or not rw:
                    continue
                row[name] = float(lv) / float(lw) - float(rv) / float(rw)
            else:
                row[name] = float(lv) - float(rv)
        by_session.setdefault(left.session_id, []).append(row)
    metric_sessions: dict[str, list[float]] = {}
    for rows in by_session.values():
        for metric in set().union(*(row.keys() for row in rows)):
            values = [row[metric] for row in rows if metric in row]
            if values:
                metric_sessions.setdefault(metric, []).append(float(np.mean(values)))
    results = []
    for metric, values in sorted(metric_sessions.items()):
        mean = float(np.mean(values))
        half = 0.0 if len(values) == 1 else 1.96 * float(np.std(values, ddof=1)) / math.sqrt(len(values))
        results.append(MetricDifference(metric, mean, mean - half, mean + half, len(values)))
    return tuple(results)
