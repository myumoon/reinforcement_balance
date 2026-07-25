"""Target telemetry と simulator telemetry を整列して fidelity 差と cluster CI を計算する。

実 video がない開発環境でも、抽出済み frame fixture から同じ計測ロジックを検証できます。
計測不能値は null と理由で保持し、policy 入力へ推測値を入れません。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

_VIDEO_FIELDS = frozenset({"direction_x", "direction_y", "speed_px", "viewport_width", "enemy_density", "chest_visible"})
_TELEMETRY_FIELDS = frozenset({"cadence_hz", "timer_seconds", "level", "offer_count", "terminal_event"})


@dataclass(frozen=True)
class AuditRunProfile:
    """simulator 再実行を拘束する target profile/time band。

    runner が別 profile や別時間帯を暗黙に選べないよう、比較条件を一つの不変値で渡します。
    """
    profile_hash: str
    time_band_start: float
    time_band_end: float

    def __post_init__(self) -> None:
        """profile identity と時間帯を fail-closed 検証する。

        空 profile、非有限値、逆転した区間を simulator 起動前に拒否します。
        """
        if not isinstance(self.profile_hash, str) or not self.profile_hash:
            raise ValueError("profile_hash must be non-empty")
        values = (self.time_band_start, self.time_band_end)
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in values):
            raise ValueError("time band must be finite numeric")
        if self.time_band_start < 0 or self.time_band_end <= self.time_band_start:
            raise ValueError("invalid time band")


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

    def __post_init__(self) -> None:
        """差分と CI の集計結果を構築時に検証する。

        非有限な統計値や空 session 集計を verdict へ流しません。
        """
        if not isinstance(self.metric, str) or not self.metric:
            raise ValueError("metric must be non-empty")
        if any(not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(x)
               for x in (self.mean_difference, self.ci_low, self.ci_high)):
            raise ValueError("difference statistics must be finite")
        if type(self.session_count) is not int or self.session_count <= 0:
            raise ValueError("session_count must be positive")


@dataclass(frozen=True)
class ExtractedVideoFrame:
    """画像 frame から抽出した screen-measurable metric。

    fixture でも実 capture と同じ interface を通し、既に計算済みの metric を video として
    素通しする経路を作りません。
    """
    metrics: Mapping[str, float]

    def __post_init__(self) -> None:
        """抽出済み metric 集合を直接構築時にも検証する。

        extractor factory を迂回して未知 field や非有限値を監査入力へ混ぜることを
        防ぎます。
        """
        if not isinstance(self.metrics, Mapping) or set(self.metrics) != _VIDEO_FIELDS:
            raise ValueError("extracted video metric keys mismatch")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in self.metrics.values()):
            raise ValueError("extracted video metrics must be finite numeric")

    @classmethod
    def from_wire(cls, value: Any) -> "ExtractedVideoFrame":
        """frame の幾何情報から方向・速度・密度・chest 可視性を抽出する。

        player の現在/直前位置と経過時間を使い、画面内 entity は中心点の列として
        fail-closed に検証します。
        """
        keys = {"player_xy", "previous_player_xy", "elapsed_seconds", "viewport_width", "enemy_centers", "chest_centers"}
        if not isinstance(value, Mapping) or set(value) != keys:
            raise ValueError("video frame keys mismatch")
        for name in ("player_xy", "previous_player_xy"):
            point = value[name]
            if not isinstance(point, list) or len(point) != 2 or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in point):
                raise ValueError(f"{name} must be a finite xy pair")
        elapsed, width = value["elapsed_seconds"], value["viewport_width"]
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0 for x in (elapsed, width)):
            raise ValueError("elapsed_seconds and viewport_width must be positive finite")
        for name in ("enemy_centers", "chest_centers"):
            centers = value[name]
            if not isinstance(centers, list):
                raise ValueError(f"{name} must be an array")
            for point in centers:
                if not isinstance(point, list) or len(point) != 2 or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in point):
                    raise ValueError(f"{name} entries must be finite xy pairs")
        dx = float(value["player_xy"][0]) - float(value["previous_player_xy"][0])
        dy = float(value["player_xy"][1]) - float(value["previous_player_xy"][1])
        distance = math.hypot(dx, dy)
        return cls({
            "direction_x": 0.0 if distance == 0 else dx / distance,
            "direction_y": 0.0 if distance == 0 else dy / distance,
            "speed_px": distance / float(elapsed),
            "viewport_width": float(width),
            "enemy_density": float(len(value["enemy_centers"])),
            "chest_visible": float(bool(value["chest_centers"])),
        })


def extract_action_telemetry(frames: Sequence[Mapping[str, Any]]) -> tuple[TelemetrySample, ...]:
    """抽出済み video/telemetry frame から監査 sample を生成する。

    direction/speed/cadence/timer/level/density/offer/chest/terminal を入力 metric 名のまま保持します。
    """
    if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)):
        raise ValueError("frames must be an array")
    return tuple(TelemetrySample.from_wire(frame) for frame in frames)


def _extract_source_rows(rows: Sequence[Mapping[str, Any]], required: frozenset[str], label: str) -> dict[tuple[str, float], dict[str, float | None]]:
    """video/telemetry source の必須 field を session/time ごとに抽出する。

    未計測値は null のまま保持し、欠落 field や未知 field は source schema 違反として拒否します。
    """
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        raise ValueError(f"{label} rows must be a non-empty array")
    output: dict[tuple[str, float], dict[str, float | None]] = {}
    allowed = required | {"session_id", "time_seconds", "uncertainties"}
    for row in rows:
        if not isinstance(row, Mapping) or set(row) - allowed or not required <= set(row):
            raise ValueError(f"{label} row keys mismatch")
        sample = TelemetrySample.from_wire({
            "session_id": row.get("session_id"),
            "time_seconds": row.get("time_seconds"),
            "metrics": {name: row[name] for name in required},
            "uncertainties": dict(row.get("uncertainties", {})),
        })
        key = (sample.session_id, float(sample.time_seconds))
        if key in output:
            raise ValueError(f"duplicate {label} session/time row")
        output[key] = dict(sample.metrics)
    return output


def _extract_video_rows(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, float], dict[str, float | None]]:
    """raw frame 列を video extraction interface 経由で metric 行へ変換する。

    session/time は capture metadata として保持し、frame payload 以外の数値 metric を
    video source に直接置く入力は拒否します。
    """
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        raise ValueError("video rows must be a non-empty array")
    output = {}
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {"session_id", "time_seconds", "frame"}:
            raise ValueError("video row keys mismatch")
        extracted = ExtractedVideoFrame.from_wire(row["frame"])
        sample = TelemetrySample(row["session_id"], row["time_seconds"], extracted.metrics, {})
        key = (sample.session_id, float(sample.time_seconds))
        if key in output:
            raise ValueError("duplicate video session/time row")
        output[key] = dict(sample.metrics)
    return output


def extract_target_session(
    video_frames: Sequence[Mapping[str, Any]],
    telemetry_rows: Sequence[Mapping[str, Any]],
) -> tuple[TelemetrySample, ...]:
    """target video と telemetry から必須監査 metric を結合する。

    direction/speed/density/chest は画面、cadence/timer/level/offer/terminal は telemetry 由来とし、
    同じ session/time に揃わない入力を比較対象へ混ぜません。
    """
    video = _extract_video_rows(video_frames)
    telemetry = _extract_source_rows(telemetry_rows, _TELEMETRY_FIELDS, "telemetry")
    if set(video) != set(telemetry):
        raise ValueError("video/telemetry session-time keys differ")
    result = []
    for session_id, time_seconds in sorted(video):
        metrics = {**video[(session_id, time_seconds)], **telemetry[(session_id, time_seconds)]}
        uncertainties = {}
        for source in (*video_frames, *telemetry_rows):
            if source.get("session_id") == session_id and float(source.get("time_seconds")) == time_seconds:
                uncertainties.update(source.get("uncertainties", {}))
        result.append(TelemetrySample(session_id, time_seconds, metrics, uncertainties))
    return tuple(result)


def run_fidelity_audit(
    target: Sequence[TelemetrySample],
    profile: AuditRunProfile,
    simulator_runner: Callable[[AuditRunProfile], Sequence[TelemetrySample]],
) -> tuple[MetricDifference, ...]:
    """同一 profile/time band で simulator を実行して target 差を返す。

    runner を注入可能にして、実 UE5 がない環境でも fixture で抽出・実行・比較を検証できます。
    """
    if not isinstance(profile, AuditRunProfile) or not callable(simulator_runner):
        raise ValueError("validated profile and callable simulator_runner required")
    checked_target = tuple(TelemetrySample.from_wire({
        "session_id": row.session_id, "time_seconds": row.time_seconds,
        "metrics": dict(row.metrics), "uncertainties": dict(row.uncertainties),
    }) for row in target)
    simulator = simulator_runner(profile)
    if not isinstance(simulator, Sequence) or isinstance(simulator, (str, bytes)):
        raise ValueError("simulator runner must return samples")
    checked_simulator = tuple(TelemetrySample.from_wire({
        "session_id": row.session_id, "time_seconds": row.time_seconds,
        "metrics": dict(row.metrics), "uncertainties": dict(row.uncertainties),
    }) for row in simulator)
    for row in (*checked_target, *checked_simulator):
        if not profile.time_band_start <= row.time_seconds <= profile.time_band_end:
            raise ValueError("sample outside requested time band")
    return align_and_compare(checked_target, checked_simulator)


def _maximum_unique_pairs(candidates: Sequence[tuple[Any, Any, Any]]) -> tuple[tuple[Any, Any], ...]:
    """候補辺から最大 cardinality・最小コストの二部 matching を作る。

    最短 augmenting path を反復し、sample 再利用を防いだまま対応数を最大化します。
    さらに同じ対応数では総時間差を最小化するため、交差対応による CI の歪みも防ぎます。
    """
    left_nodes = tuple(sorted({left for _, left, _ in candidates}, key=str))
    right_nodes = tuple(sorted({right for _, _, right in candidates}, key=str))
    source, sink = ("source",), ("sink",)
    graph: dict[Any, list[list[Any]]] = {}

    def add_edge(start: Any, end: Any, capacity: int, cost: float) -> None:
        """残余グラフへ正方向と逆方向の辺を対で追加する。

        逆辺により、後続の経路が以前の対応を取り消して総コストを改善できます。
        """
        forward = [end, capacity, cost, None]
        reverse = [start, 0, -cost, forward]
        forward[3] = reverse
        graph.setdefault(start, []).append(forward)
        graph.setdefault(end, []).append(reverse)

    for left in left_nodes:
        add_edge(source, ("left", left), 1, 0.0)
    for right in right_nodes:
        add_edge(("right", right), sink, 1, 0.0)
    for distance, left, right in sorted(candidates, key=lambda row: (float(row[0]), str(row[1]), str(row[2]))):
        add_edge(("left", left), ("right", right), 1, float(distance))

    while True:
        distances = {node: math.inf for node in graph}
        distances[source] = 0.0
        previous: dict[Any, tuple[Any, list[Any]]] = {}
        for _ in range(len(graph) - 1):
            changed = False
            for start in sorted(graph, key=str):
                if not math.isfinite(distances[start]):
                    continue
                for edge in graph[start]:
                    end, capacity, cost, _ = edge
                    candidate = distances[start] + cost
                    if capacity and candidate < distances[end] - 1e-15:
                        distances[end] = candidate
                        previous[end] = (start, edge)
                        changed = True
            if not changed:
                break
        if sink not in previous:
            break
        node = sink
        while node != source:
            start, edge = previous[node]
            edge[1] -= 1
            edge[3][1] += 1
            node = start

    pairs = []
    for left in left_nodes:
        for edge in graph[("left", left)]:
            end, capacity, _, _ = edge
            if isinstance(end, tuple) and len(end) == 2 and end[0] == "right" and capacity == 0:
                pairs.append((left, end[1]))
    return tuple(sorted(pairs, key=lambda pair: str(pair)))


def align_and_compare(target: Sequence[TelemetrySample], simulator: Sequence[TelemetrySample], *, time_tolerance: float = 0.1) -> tuple[MetricDifference, ...]:
    """独立 session を time band で整列し viewport/unit 換算後の差を集計する。

    target と simulator の session identity は共有せず、一対一の session/sample 対応を作ります。
    `*_px` は viewport_width で正規化し、None metric は policy 比較から除外します。
    """
    if not math.isfinite(time_tolerance) or time_tolerance < 0:
        raise ValueError("time_tolerance must be finite and non-negative")
    target_sessions: dict[str, list[TelemetrySample]] = {}
    simulator_sessions: dict[str, list[TelemetrySample]] = {}
    for sample in target:
        target_sessions.setdefault(sample.session_id, []).append(sample)
    for sample in simulator:
        simulator_sessions.setdefault(sample.session_id, []).append(sample)
    session_candidates = []
    for target_id, left_rows in target_sessions.items():
        for simulator_id, right_rows in simulator_sessions.items():
            distance = min(abs(left.time_seconds - right.time_seconds) for left in left_rows for right in right_rows)
            if distance <= time_tolerance:
                session_candidates.append((distance, target_id, simulator_id))
    session_pairs = _maximum_unique_pairs(session_candidates)

    by_session: dict[str, list[dict[str, float]]] = {}
    for target_id, simulator_id in session_pairs:
        sample_candidates = [
            (abs(left.time_seconds - right.time_seconds), left_index, right_index)
            for left_index, left in enumerate(target_sessions[target_id])
            for right_index, right in enumerate(simulator_sessions[simulator_id])
            if abs(left.time_seconds - right.time_seconds) <= time_tolerance
        ]
        for left_index, right_index in _maximum_unique_pairs(sample_candidates):
            left = target_sessions[target_id][left_index]
            right = simulator_sessions[simulator_id][right_index]
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
            by_session.setdefault(target_id, []).append(row)
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
