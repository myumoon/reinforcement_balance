"""sealed perception 評価用の事前登録窓を生成・検証する。"""

from __future__ import annotations

import bisect
import math
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Mapping, Sequence

from reinbalance_survivors_contracts.canonical_json import canonical_hash


SCHEMA_VERSION = "evaluation_windows.v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_TIME_BAND_SECONDS = 600.0


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    """窓規則と frame index の整数値を厳密に確認する。

    bool は Python で int の派生型なので、wire 入力では明示的に拒否する。
    """
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _number(value: object, label: str, *, minimum: float = 0.0) -> float:
    """秒・score の有限数を確認する。

    数値文字列と bool を拒否し、非有限値が窓選択へ混入しないようにする。
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number >= {minimum}")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{label} must be a finite number >= {minimum}")
    return result


def _sha256(value: object, label: str) -> str:
    """plan 規則に含まれる artifact identity を確認する。
    不正な hash は canonical plan を作る前に拒否する。
    """
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def validate_rule(rule: object) -> dict[str, Any]:
    """事前登録ルールの必須要素と値域を検証して複製する。

    未知キーを拒否し、plan 生成後に規則の意味が変わる形を許さない。
    """
    if not isinstance(rule, Mapping) or set(rule) != {
        "fixed_offsets_per_time_band", "run_tail", "event_windows", "boss_windows"
    }:
        raise ValueError("rule must contain exactly the four evaluation window rules")
    fixed = rule["fixed_offsets_per_time_band"]
    tail = rule["run_tail"]
    events = rule["event_windows"]
    bosses = rule["boss_windows"]
    if not isinstance(fixed, Mapping) or set(fixed) != {"window_frames", "offsets_seconds"}:
        raise ValueError("fixed_offsets_per_time_band has invalid fields")
    offsets = fixed["offsets_seconds"]
    if not isinstance(offsets, Mapping) or set(offsets) != {"early", "mid", "late"}:
        raise ValueError("offsets_seconds must define early, mid, and late")
    clean_offsets = {
        name: _number(offsets[name], f"offsets_seconds.{name}")
        for name in ("early", "mid", "late")
    }
    if not isinstance(tail, Mapping) or set(tail) != {"seconds"}:
        raise ValueError("run_tail has invalid fields")
    if not isinstance(events, Mapping) or set(events) != {
        "detector", "states", "pre_frames", "post_frames", "max_per_band",
        "parser_artifact_hash",
    }:
        raise ValueError("event_windows has invalid fields")
    states = events["states"]
    if (
        events["detector"] != "development_hud_parser"
        or not isinstance(states, list) or not states
        or any(type(state) is not str or not state for state in states)
        or len(states) != len(set(states))
    ):
        raise ValueError("event_windows detector/states are invalid")
    if not isinstance(bosses, Mapping) or set(bosses) != {
        "detector", "checkpoint_hash", "score_threshold", "pre_frames",
        "post_frames", "max_per_session",
    }:
        raise ValueError("boss_windows has invalid fields")
    if bosses["detector"] != "development_world_detector":
        raise ValueError("boss_windows detector is invalid")
    score_threshold = _number(bosses["score_threshold"], "score_threshold")
    if not 0 <= score_threshold <= 1:
        raise ValueError("score_threshold must be between 0 and 1")
    return {
        "fixed_offsets_per_time_band": {
            "window_frames": _integer(fixed["window_frames"], "window_frames", minimum=1),
            "offsets_seconds": clean_offsets,
        },
        "run_tail": {"seconds": _number(tail.get("seconds"), "run_tail.seconds")},
        "event_windows": {
            "detector": events["detector"],
            "states": list(states),
            "pre_frames": _integer(events["pre_frames"], "event_windows.pre_frames"),
            "post_frames": _integer(events["post_frames"], "event_windows.post_frames"),
            "max_per_band": _integer(events["max_per_band"], "event_windows.max_per_band", minimum=1),
            "parser_artifact_hash": _sha256(events["parser_artifact_hash"], "parser_artifact_hash"),
        },
        "boss_windows": {
            "detector": bosses["detector"],
            "checkpoint_hash": _sha256(bosses["checkpoint_hash"], "checkpoint_hash"),
            "score_threshold": score_threshold,
            "pre_frames": _integer(bosses["pre_frames"], "boss_windows.pre_frames"),
            "post_frames": _integer(bosses["post_frames"], "boss_windows.post_frames"),
            "max_per_session": _integer(bosses["max_per_session"], "boss_windows.max_per_session", minimum=1),
        },
    }


def validate_windows(
    windows: Sequence[Sequence[int]], *, frame_count: int,
) -> tuple[tuple[int, int], ...]:
    """session 内の閉区間窓が昇順・非重複であることを確認する。

    session_frame_index は 0 始まりの連続 index とし、end は frame_count 未満に限る。
    """
    _integer(frame_count, "frame_count", minimum=1)
    if not isinstance(windows, Sequence) or isinstance(windows, (str, bytes)):
        raise ValueError("windows must be a sequence")
    result: list[tuple[int, int]] = []
    previous_end = -1
    for index, pair in enumerate(windows):
        if not isinstance(pair, Sequence) or isinstance(pair, (str, bytes)) or len(pair) != 2:
            raise ValueError(f"windows[{index}] must be a [start, end] pair")
        start = _integer(pair[0], f"windows[{index}].start")
        end = _integer(pair[1], f"windows[{index}].end")
        if start > end or end >= frame_count:
            raise ValueError(f"windows[{index}] is outside the session frame range")
        if start <= previous_end:
            raise ValueError("windows must be sorted and non-overlapping")
        result.append((start, end))
        previous_end = end
    return tuple(result)


def _merge_windows(windows: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """規則ごとの重複・隣接窓をひと続きの区間にまとめる。
    出力は昇順かつ閉区間の非重複となる。
    """
    merged: list[list[int]] = []
    for start, end in sorted(windows):
        if merged and start <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def generate_windows(
    manifest: Any,
    rule: Mapping[str, Any],
    *,
    frame_loader: Callable[[Any], Mapping[str, Any]],
) -> list[tuple[int, int]]:
    """session manifest と固定規則から評価窓だけを決定する。

    内容を人が選ばないよう、HUD state と boss score は frame_loader の機械判定を使う。
    """
    normalized_rule = validate_rule(rule)
    records = tuple(getattr(manifest, "frame_records", ()))
    if not records:
        raise ValueError("evaluation window session has no frame records")
    ids = [record.frame_id for record in records]
    timestamps = [record.captured_monotonic_ns for record in records]
    if any(type(frame_id) is not int or frame_id < 0 for frame_id in ids):
        raise ValueError("session frame ids must be nonnegative integers")
    if ids != list(range(ids[0], ids[0] + len(ids))) or ids[0] != 0:
        raise ValueError("session frame ids must be contiguous and start at zero")
    if any(type(value) is not int for value in timestamps) or any(
        right <= left for left, right in zip(timestamps, timestamps[1:])
    ):
        raise ValueError("session timestamps must increase strictly")

    windows: list[tuple[int, int]] = []
    fixed = normalized_rule["fixed_offsets_per_time_band"]
    elapsed = [(value - timestamps[0]) / 1_000_000_000 for value in timestamps]
    for offset in fixed["offsets_seconds"].values():
        index = bisect.bisect_left(elapsed, offset)
        if index < len(records):
            windows.append((index, min(len(records) - 1, index + fixed["window_frames"] - 1)))

    tail_seconds = normalized_rule["run_tail"]["seconds"]
    cutoff = elapsed[-1] - tail_seconds
    tail_start = bisect.bisect_left(elapsed, cutoff)
    windows.append((tail_start, len(records) - 1))

    event_rule = normalized_rule["event_windows"]
    boss_rule = normalized_rule["boss_windows"]
    detections: list[Mapping[str, Any]] = []
    if event_rule["states"] or boss_rule["max_per_session"]:
        for record in records:
            detection = frame_loader(record)
            if not isinstance(detection, Mapping):
                raise ValueError("frame_loader must return a mapping")
            detections.append(detection)

    event_counts = {"early": 0, "mid": 0, "late": 0}
    previous_state: str | None = None
    for index, detection in enumerate(detections):
        state = detection.get("screen_state")
        if type(state) is not str:
            raise ValueError("frame_loader screen_state must be a string")
        band = "early" if elapsed[index] < _TIME_BAND_SECONDS else (
            "mid" if elapsed[index] < 2 * _TIME_BAND_SECONDS else "late"
        )
        if state in event_rule["states"] and state != previous_state:
            if event_counts[band] < event_rule["max_per_band"]:
                windows.append((
                    max(0, index - event_rule["pre_frames"]),
                    min(len(records) - 1, index + event_rule["post_frames"]),
                ))
                event_counts[band] += 1
        previous_state = state

    boss_count = 0
    previous_boss = False
    for index, detection in enumerate(detections):
        score = _number(detection.get("boss_score", 0.0), "boss_score")
        present = score >= boss_rule["score_threshold"]
        if present and not previous_boss and boss_count < boss_rule["max_per_session"]:
            windows.append((
                max(0, index - boss_rule["pre_frames"]),
                min(len(records) - 1, index + boss_rule["post_frames"]),
            ))
            boss_count += 1
        previous_boss = present
    merged = _merge_windows(windows)
    validate_windows(merged, frame_count=len(records))
    return merged


@dataclass(frozen=True, slots=True)
class EvaluationWindowPlan:
    """窓規則・session 区間・生成時刻を hash で束縛した plan。
    wire 化後の plan hash は annotation と formal replay で共通に使う。
    """

    sessions: Mapping[str, tuple[tuple[int, int], ...]]
    rule: Mapping[str, Any]
    stride: int
    generated_at_utc: str
    plan_canonical_hash: str
    schema_version: str = SCHEMA_VERSION

    @classmethod
    def create(
        cls,
        sessions: Mapping[str, Sequence[Sequence[int]]],
        *,
        rule: Mapping[str, Any],
        stride: int = 3,
        generated_at_utc: str,
    ) -> "EvaluationWindowPlan":
        """session 窓を検証し、canonical identity を付けて plan を作る。
        生成時刻も identity に含めて事前登録後の差し替えを検出する。
        """
        _integer(stride, "stride", minimum=1)
        if not isinstance(sessions, Mapping) or not sessions:
            raise ValueError("sessions must be a non-empty mapping")
        clean_sessions: dict[str, tuple[tuple[int, int], ...]] = {}
        for session_id, windows in sessions.items():
            if type(session_id) is not str or not session_id:
                raise ValueError("session ids must be non-empty strings")
            if not isinstance(windows, Sequence) or not windows:
                raise ValueError(f"session {session_id!r} must have at least one window")
            pairs: list[tuple[int, int]] = []
            previous_end = -1
            for pair in windows:
                if not isinstance(pair, Sequence) or isinstance(pair, (str, bytes)) or len(pair) != 2:
                    raise ValueError("each window must contain start and end frame index")
                start = _integer(pair[0], "window.start")
                end = _integer(pair[1], "window.end")
                if start > end or start <= previous_end:
                    raise ValueError("session windows must be sorted and non-overlapping")
                pairs.append((start, end))
                previous_end = end
            clean_sessions[session_id] = tuple(pairs)
        clean_rule = validate_rule(rule)
        try:
            parsed_time = datetime.fromisoformat(generated_at_utc.replace("Z", "+00:00"))
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("generated_at_utc must be an ISO-8601 timestamp") from exc
        if parsed_time.utcoffset() is None:
            raise ValueError("generated_at_utc must include a timezone")
        body = {
            "schema_version": SCHEMA_VERSION,
            "generated_at_utc": generated_at_utc,
            "sessions": {
                name: [list(pair) for pair in clean_sessions[name]]
                for name in sorted(clean_sessions)
            },
            "rule": clean_rule,
            "stride": stride,
        }
        return cls(
            clean_sessions, clean_rule, stride, generated_at_utc,
            canonical_hash(body),
        )

    def to_wire(self) -> dict[str, Any]:
        """descriptor と annotation tool が使う JSON wire 形式を返す。
        session と窓は canonical な配列形式に揃えて出力する。
        """
        return {
            "schema_version": self.schema_version,
            "generated_at_utc": self.generated_at_utc,
            "sessions": {
                name: [list(pair) for pair in windows]
                for name, windows in sorted(self.sessions.items())
            },
            "rule": validate_rule(self.rule),
            "stride": self.stride,
            "plan_canonical_hash": self.plan_canonical_hash,
        }


def validate_plan_wire(value: object) -> EvaluationWindowPlan:
    """evaluation_windows.v1 wire の schema と canonical hash を検証する。
    wire が宣言した hash と再計算値の不一致を拒否する。
    """
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version", "generated_at_utc", "sessions", "rule", "stride",
        "plan_canonical_hash",
    }:
        raise ValueError("evaluation window plan fields do not match evaluation_windows.v1")
    if value["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported evaluation window plan schema_version")
    claimed_hash = _sha256(value["plan_canonical_hash"], "plan_canonical_hash")
    plan = EvaluationWindowPlan.create(
        value["sessions"], rule=value["rule"], stride=value["stride"],
        generated_at_utc=value["generated_at_utc"],
    )
    if plan.plan_canonical_hash != claimed_hash:
        raise ValueError("evaluation window plan hash mismatch")
    return plan
