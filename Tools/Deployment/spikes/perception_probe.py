"""NumPy-only perception feasibility core.

Capture/detector backends are intentionally optional; codecs, transforms and
probe aggregation remain testable in the minimal Deployment environment.
"""
from __future__ import annotations

from dataclasses import dataclass
import binascii
import math
from pathlib import Path
import struct
from types import MappingProxyType
from typing import Any, Iterable, Mapping
import zlib

import numpy as np
import yaml

from reinbalance_survivors_contracts.ui_intent import (
    ContractValidationError, ensure, is_strict_number, require_schema_version,
)

CONFIG_VERSION = "perception_feasibility.v1"
CONFIG = Path(__file__).parents[1] / "configs" / "perception_feasibility_v1.yaml"
_CONFIG_KEYS = frozenset({"schema_version", "pilot", "thresholds", "architectures", "budget"})
_PILOT_KEYS = frozenset({"min_sessions", "max_sessions", "min_minutes_per_session",
                         "min_frames", "required_slices", "min_sessions_per_slice"})
_THRESHOLD_KEYS = frozenset({"ssdlite320_p10_short_side_px", "oracle_recall",
                             "single_pass_p95_ms", "bbox_qa_iou",
                             "class_agreement", "dense_entities_per_hour"})
_ARCH_KEYS = frozenset({"ssdlite320", "ssdlite640_multiscale", "tile_2x2", "coarse_density"})
_BUDGET_KEYS = frozenset({"parallel_worker_limit", "frame_storage_bytes",
                          "gpu_seconds_per_frame"})


def _closed(data: Mapping[str, Any], keys: frozenset[str], label: str) -> None:
    ensure(isinstance(data, Mapping), f"{label} must be a mapping")
    unknown, missing = set(data) - keys, keys - set(data)
    ensure(not unknown, f"unknown {label} fields: {sorted(unknown)}")
    ensure(not missing, f"missing {label} fields: {sorted(missing)}")


@dataclass(frozen=True)
class FeasibilityConfig:
    pilot: Mapping[str, Any]
    thresholds: Mapping[str, float]
    architectures: tuple[str, ...]
    budget: Mapping[str, float]
    schema_version: str = CONFIG_VERSION

    def __post_init__(self) -> None:
        # A frozen dataclass does not freeze dictionaries held by its fields.
        # Copy before wrapping so callers cannot retain an alias to mutable state.
        pilot = dict(self.pilot)
        if "required_slices" in pilot:
            pilot["required_slices"] = tuple(pilot["required_slices"])
        object.__setattr__(self, "pilot", MappingProxyType(pilot))
        object.__setattr__(self, "thresholds", MappingProxyType(dict(self.thresholds)))
        object.__setattr__(self, "architectures", tuple(self.architectures))
        object.__setattr__(self, "budget", MappingProxyType(dict(self.budget)))
        ensure(self.schema_version == CONFIG_VERSION, "feasibility config schema mismatch")
        _closed(self.pilot, _PILOT_KEYS, "pilot")
        _closed(self.thresholds, _THRESHOLD_KEYS, "thresholds")
        _closed(self.budget, _BUDGET_KEYS, "budget")
        ensure(set(self.architectures) == _ARCH_KEYS and
               len(self.architectures) == len(set(self.architectures)),
               "architecture matrix must be complete and unique")
        ensure(all(type(self.pilot[k]) is int and self.pilot[k] > 0 for k in
                   ("min_sessions", "max_sessions", "min_minutes_per_session",
                    "min_frames", "min_sessions_per_slice")) and
               self.pilot["min_sessions"] <= self.pilot["max_sessions"],
               "pilot counts/durations must be positive with a valid session range")
        ensure(isinstance(self.pilot["required_slices"], tuple) and
               all(isinstance(v, str) and v for v in self.pilot["required_slices"]) and
               len(self.pilot["required_slices"]) == len(set(self.pilot["required_slices"])),
               "required slices must be unique non-empty strings")
        ensure(all(is_strict_number(v) and math.isfinite(v) and v > 0
                   for v in self.thresholds.values()) and
               all(self.thresholds[k] <= 1 for k in
                   ("oracle_recall", "bbox_qa_iou", "class_agreement")),
               "thresholds must be finite, positive, and bounded where applicable")
        ensure(type(self.budget["parallel_worker_limit"]) is int and
               self.budget["parallel_worker_limit"] > 0 and
               type(self.budget["frame_storage_bytes"]) is int and
               self.budget["frame_storage_bytes"] > 0 and
               is_strict_number(self.budget["gpu_seconds_per_frame"]) and
               math.isfinite(self.budget["gpu_seconds_per_frame"]) and
               self.budget["gpu_seconds_per_frame"] > 0,
               "budget resources must be positive finite values of the required type")

    @classmethod
    def from_wire(cls, data: Mapping[str, Any]) -> "FeasibilityConfig":
        _closed(data, _CONFIG_KEYS, "feasibility config")
        require_schema_version(data, CONFIG_VERSION, "feasibility config")
        _closed(data["pilot"], _PILOT_KEYS, "pilot")
        _closed(data["thresholds"], _THRESHOLD_KEYS, "thresholds")
        _closed(data["budget"], _BUDGET_KEYS, "budget")
        ensure(set(data["architectures"]) == _ARCH_KEYS, "architecture matrix is incomplete")
        ensure(len(data["architectures"]) == len(set(data["architectures"])),
               "architectures must be unique")
        pilot = data["pilot"]
        ensure(all(type(pilot[k]) is int and pilot[k] > 0 for k in
                   ("min_sessions", "max_sessions", "min_minutes_per_session",
                    "min_frames", "min_sessions_per_slice")),
               "pilot counts/durations must be positive integers")
        ensure(pilot["min_sessions"] <= pilot["max_sessions"],
               "pilot session range is invalid")
        ensure(isinstance(pilot["required_slices"], list) and
               all(isinstance(v, str) and v for v in pilot["required_slices"]) and
               len(pilot["required_slices"]) == len(set(pilot["required_slices"])),
               "required slices must be unique non-empty strings")
        thresholds = data["thresholds"]
        ensure(all(is_strict_number(v) and math.isfinite(v) and v > 0
                   for v in thresholds.values()), "thresholds must be positive finite numbers")
        ensure(all(thresholds[k] <= 1 for k in
                   ("oracle_recall", "bbox_qa_iou", "class_agreement")),
               "recall/agreement thresholds must be at most one")
        budget = data["budget"]
        ensure(type(budget["parallel_worker_limit"]) is int and
               budget["parallel_worker_limit"] > 0,
               "parallel worker limit must be a positive integer")
        ensure(type(budget["frame_storage_bytes"]) is int and
               budget["frame_storage_bytes"] > 0,
               "frame storage bytes must be a positive integer")
        ensure(is_strict_number(budget["gpu_seconds_per_frame"]) and
               math.isfinite(budget["gpu_seconds_per_frame"]) and
               budget["gpu_seconds_per_frame"] > 0,
               "GPU seconds per frame must be a positive finite number")
        return cls(dict(data["pilot"]), dict(data["thresholds"]),
                   tuple(data["architectures"]), dict(data["budget"]))

    def validate(self) -> None:
        # Re-enter the closed, finite-number wire validator at every trust boundary.
        self.from_wire(self.to_wire())

    def to_wire(self) -> dict[str, Any]:
        pilot = dict(self.pilot)
        if "required_slices" in pilot:
            pilot["required_slices"] = list(pilot["required_slices"])
        return {"schema_version": self.schema_version, "pilot": pilot,
                "thresholds": dict(self.thresholds), "architectures": list(self.architectures),
                "budget": dict(self.budget)}


def load_feasibility_config(source: Path | Mapping[str, Any] = CONFIG) -> FeasibilityConfig:
    if isinstance(source, Mapping):
        data = source
    else:
        with Path(source).open(encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
    return FeasibilityConfig.from_wire(data)


@dataclass(frozen=True)
class Box:
    x1: float; y1: float; x2: float; y2: float
    def __post_init__(self) -> None:
        ensure(all(is_strict_number(v) and math.isfinite(v)
                   for v in (self.x1, self.y1, self.x2, self.y2)),
               "box coordinates must be finite numbers")
        ensure(self.x2 >= self.x1 and self.y2 >= self.y1,
               "box coordinates must be ordered")

    def almost_equals(self, other: "Box", tolerance: float = 1e-6) -> bool:
        ensure(isinstance(other, Box), "comparison target must be a box")
        ensure(is_strict_number(tolerance) and math.isfinite(tolerance) and tolerance >= 0,
               "box tolerance must be finite and non-negative")
        return bool(np.allclose((self.x1, self.y1, self.x2, self.y2),
                                (other.x1, other.y1, other.x2, other.y2),
                                atol=tolerance, rtol=0))


@dataclass(frozen=True)
class CoordinateTransform:
    scale_x: float; scale_y: float; offset_x: float; offset_y: float

    def __post_init__(self) -> None:
        ensure(all(is_strict_number(v) and math.isfinite(v)
                   for v in (self.scale_x, self.scale_y, self.offset_x, self.offset_y)),
               "transform values must be finite numbers")
        ensure(self.scale_x > 0 and self.scale_y > 0, "transform scale must be positive")

    @classmethod
    def roi(cls, x: float, y: float) -> "CoordinateTransform":
        return cls(1, 1, -x, -y)

    @classmethod
    def resize(cls, source: tuple[int, int], target: tuple[int, int]) -> "CoordinateTransform":
        _validate_dimensions(source, "source")
        _validate_dimensions(target, "target")
        return cls(target[0] / source[0], target[1] / source[1], 0, 0)

    @classmethod
    def letterbox(cls, source: tuple[int, int], target: tuple[int, int]) -> "CoordinateTransform":
        _validate_dimensions(source, "source")
        _validate_dimensions(target, "target")
        scale = min(target[0] / source[0], target[1] / source[1])
        return cls(scale, scale, (target[0] - source[0] * scale) / 2,
                   (target[1] - source[1] * scale) / 2)

    @classmethod
    def tile(cls, column: int, row: int, *, tile_width: int,
             tile_height: int) -> "CoordinateTransform":
        ensure(type(column) is int and column >= 0 and type(row) is int and row >= 0,
               "tile indices must be non-negative integers")
        _validate_dimensions((tile_width, tile_height), "tile")
        return cls(1, 1, -column * tile_width, -row * tile_height)

    def forward_box(self, box: Box) -> Box:
        return Box(box.x1*self.scale_x+self.offset_x, box.y1*self.scale_y+self.offset_y,
                   box.x2*self.scale_x+self.offset_x, box.y2*self.scale_y+self.offset_y)

    def inverse_box(self, box: Box) -> Box:
        return Box((box.x1-self.offset_x)/self.scale_x, (box.y1-self.offset_y)/self.scale_y,
                   (box.x2-self.offset_x)/self.scale_x, (box.y2-self.offset_y)/self.scale_y)


def _chunk(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(
        ">I", binascii.crc32(kind + payload) & 0xffffffff)


def _validate_dimensions(value: Any, label: str) -> None:
    ensure(isinstance(value, tuple) and len(value) == 2 and
           all(type(v) is int and v > 0 for v in value),
           f"{label} dimensions must be two positive integers")


def encode_png(frame: np.ndarray) -> bytes:
    """Encode uint8 BGRA as a standards-compliant, filter-0 RGBA PNG."""
    _validate_frame(frame)
    rgba = frame[..., [2, 1, 0, 3]]
    scanlines = b"".join(b"\0" + row.tobytes() for row in rgba)
    ihdr = struct.pack(">IIBBBBB", frame.shape[1], frame.shape[0], 8, 6, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr) + _chunk(
        b"IDAT", zlib.compress(scanlines, 9)) + _chunk(b"IEND", b"")


def _decode_png(payload: bytes) -> np.ndarray:
    ensure(payload.startswith(b"\x89PNG\r\n\x1a\n"), "invalid PNG signature")
    offset, width, height, compressed = 8, None, None, bytearray()
    while offset < len(payload):
        length = struct.unpack(">I", payload[offset:offset+4])[0]
        kind, data = payload[offset+4:offset+8], payload[offset+8:offset+8+length]
        ensure(binascii.crc32(kind + data) & 0xffffffff ==
               struct.unpack(">I", payload[offset+8+length:offset+12+length])[0],
               "PNG CRC mismatch")
        if kind == b"IHDR":
            width, height, depth, color, comp, filt, interlace = struct.unpack(">IIBBBBB", data)
            ensure((depth, color, comp, filt, interlace) == (8, 6, 0, 0, 0),
                   "only non-interlaced RGBA8 PNG is supported")
        elif kind == b"IDAT":
            compressed.extend(data)
        elif kind == b"IEND":
            break
        offset += 12 + length
    ensure(width is not None and height is not None, "PNG missing IHDR")
    raw = zlib.decompress(bytes(compressed))
    stride = width * 4
    ensure(len(raw) == height * (stride + 1), "invalid PNG data length")
    rows = []
    for row in range(height):
        start = row * (stride + 1)
        ensure(raw[start] == 0, "only PNG filter 0 is supported")
        rows.append(np.frombuffer(raw[start+1:start+1+stride], dtype=np.uint8))
    rgba = np.stack(rows).reshape(height, width, 4)
    return rgba[..., [2, 1, 0, 3]].copy()


def encode_near_lossless(frame: np.ndarray) -> bytes:
    """Reversible spike codec; production codecs remain an optional runtime concern."""
    _validate_frame(frame)
    header = struct.pack(">4sII", b"NLS1", frame.shape[1], frame.shape[0])
    return header + zlib.compress(frame.tobytes(), 6)


def _validate_frame(frame: np.ndarray) -> None:
    ensure(isinstance(frame, np.ndarray) and frame.dtype == np.uint8 and
           frame.ndim == 3 and frame.shape[2] == 4, "frame must be uint8 BGRA")


def decode_frame(payload: bytes, encoding: str, *, width: int | None = None,
                 height: int | None = None) -> np.ndarray:
    ensure(isinstance(payload, bytes), "encoded frame payload must be bytes")
    ensure(isinstance(encoding, str), "frame encoding must be text")
    if encoding == "raw_bgra":
        ensure(type(width) is int and width > 0 and type(height) is int and height > 0 and
               len(payload) == width * height * 4,
               "raw BGRA dimensions mismatch")
        return np.frombuffer(payload, dtype=np.uint8).reshape(height, width, 4).copy()
    if encoding == "png":
        return _decode_png(payload)
    if encoding == "near_lossless":
        magic, actual_width, actual_height = struct.unpack(">4sII", payload[:12])
        ensure(magic == b"NLS1", "invalid near-lossless header")
        raw = zlib.decompress(payload[12:])
        return decode_frame(raw, "raw_bgra", width=actual_width, height=actual_height)
    raise ContractValidationError(f"unknown frame encoding: {encoding!r}")


@dataclass(frozen=True)
class ProbeSample:
    kind: str
    short_side_px: float
    slices: tuple[str, ...]
    oracle_detectable: bool
    architecture_latency_ms: Mapping[str, float]
    annotation_seconds: float
    architecture_detectable: Mapping[str, bool] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "slices", tuple(self.slices))
        object.__setattr__(self, "architecture_latency_ms",
                           MappingProxyType(dict(self.architecture_latency_ms)))
        if self.architecture_detectable is not None:
            object.__setattr__(self, "architecture_detectable",
                               MappingProxyType(dict(self.architecture_detectable)))
        self.validate()

    def validate(self) -> None:
        ensure(isinstance(self.kind, str) and self.kind, "probe kind must be non-empty")
        ensure(is_strict_number(self.short_side_px) and math.isfinite(self.short_side_px) and
               self.short_side_px >= 0, "probe short side must be finite and non-negative")
        ensure(self.slices and all(isinstance(v, str) and v for v in self.slices),
               "probe slices must be non-empty strings")
        ensure(type(self.oracle_detectable) is bool, "oracle detectable must be boolean")
        ensure(set(self.architecture_latency_ms) == _ARCH_KEYS and
               all(is_strict_number(v) and math.isfinite(v) and v > 0
                   for v in self.architecture_latency_ms.values()),
               "architecture latency matrix must contain positive finite values")
        ensure(is_strict_number(self.annotation_seconds) and
               math.isfinite(self.annotation_seconds) and self.annotation_seconds >= 0,
               "annotation seconds must be finite and non-negative")
        ensure(self.architecture_detectable is None or
               (set(self.architecture_detectable) == _ARCH_KEYS and
                all(type(v) is bool for v in self.architecture_detectable.values())),
               "architecture detection matrix is incomplete")


def make_synthetic_fixture(seed: int = 0) -> tuple[ProbeSample, ...]:
    rng = np.random.default_rng(seed)
    slices = ("small", "occluded", "late", "heavy", "boss", "gem")
    kinds = ("timer", "icon", "entity")
    result = []
    for index in range(36):
        lat = {"ssdlite320": 11.0, "ssdlite640_multiscale": 23.0,
               "tile_2x2": 37.0, "coarse_density": 5.0}
        result.append(ProbeSample(
            kinds[index % 3], float(rng.integers(3, 25)),
            (slices[index % len(slices)],), bool(index % 7),
            {key: value + float(rng.uniform(0, 2)) for key, value in lat.items()},
            float(rng.uniform(1, 5)),
            {name: bool((index + offset) % 7)
             for offset, name in enumerate(sorted(_ARCH_KEYS))}))
    return tuple(result)


def evaluate_probe(samples: Iterable[ProbeSample]) -> dict[str, Any]:
    values = tuple(samples)
    ensure(values and all(isinstance(v, ProbeSample) for v in values),
           "at least one probe sample is required")
    for value in values:
        value.validate()
    ensure(all(isinstance(v.kind, str) and v.kind and
               is_strict_number(v.short_side_px) and math.isfinite(v.short_side_px) and
               v.short_side_px >= 0 and
               is_strict_number(v.annotation_seconds) and
               math.isfinite(v.annotation_seconds) and v.annotation_seconds >= 0 and
               type(v.oracle_detectable) is bool and
               all(isinstance(s, str) and s for s in v.slices)
               for v in values), "probe sample metrics must be finite and non-negative")
    ensure(all(v.architecture_detectable is None or
               (set(v.architecture_detectable) == _ARCH_KEYS and
                all(type(detected) is bool
                    for detected in v.architecture_detectable.values()))
               for v in values), "architecture detection matrix is incomplete")
    short = np.array([v.short_side_px for v in values])
    slice_names = ("small", "occluded", "late", "heavy", "boss", "gem")
    recall = {}
    for name in slice_names:
        subset = [v for v in values if name in v.slices]
        recall[name] = float(sum(v.oracle_detectable for v in subset) / len(subset)) if subset else 0.0
    architectures = {}
    all_latency = []
    for name in _ARCH_KEYS:
        ensure(all(set(v.architecture_latency_ms) == _ARCH_KEYS for v in values),
               "architecture latency matrix is incomplete")
        latency = np.array([v.architecture_latency_ms[name] for v in values])
        ensure(all(is_strict_number(x) and math.isfinite(x) and x > 0 for x in latency),
               "architecture latency must be positive and finite")
        slice_recall = {}
        slice_metrics = {}
        for slice_name in slice_names:
            subset = [v for v in values if slice_name in v.slices]
            detected = [
                (v.architecture_detectable[name] if v.architecture_detectable is not None
                 else v.oracle_detectable)
                for v in subset
            ]
            slice_recall[slice_name] = float(sum(detected) / len(detected)) if detected else 0.0
            slice_metrics[slice_name] = {
                "recall_upper_bound": slice_recall[slice_name],
                "latency_p95_ms": float(np.percentile(
                    [v.architecture_latency_ms[name] for v in subset], 95
                )) if subset else None,
            }
        architectures[name] = {"latency_p95_ms": float(np.percentile(latency, 95)),
                               "utility_per_latency": float(np.mean([v.oracle_detectable for v in values])
                                                            / np.percentile(latency, 95)),
                               "recall_upper_bound": slice_recall,
                               "slice_metrics": slice_metrics}
        all_latency.extend(latency)
    latency_array = np.array(all_latency)
    return {"pixel_size": {"p10_short_side": float(np.percentile(short, 10)),
                           "median_short_side": float(np.median(short))},
            "recall_upper_bound": recall,
            "latency_ms": {"median": float(np.median(latency_array)),
                           "p95": float(np.percentile(latency_array, 95))},
            "annotation_seconds": float(sum(v.annotation_seconds for v in values)),
            "architectures": architectures}


def normalized_displacement(start_xy: tuple[float, float], end_xy: tuple[float, float],
                            viewport_wh: tuple[int, int]) -> tuple[float, float]:
    ensure(all(isinstance(v, tuple) and len(v) == 2 for v in (start_xy, end_xy)),
           "screen points must contain two coordinates")
    ensure(all(is_strict_number(x) and math.isfinite(x) for v in (start_xy, end_xy) for x in v),
           "screen coordinates must be finite numbers")
    _validate_dimensions(viewport_wh, "viewport")
    return ((end_xy[0]-start_xy[0])/viewport_wh[0],
            (end_xy[1]-start_xy[1])/viewport_wh[1])


def template_score(image: np.ndarray, template: np.ndarray) -> float:
    """Small HUD digit/icon baseline without OpenCV."""
    ensure(isinstance(image, np.ndarray) and isinstance(template, np.ndarray) and
           image.shape == template.shape and image.size > 0 and
           image.dtype == np.uint8 and template.dtype == np.uint8,
           "template inputs must be matching non-empty uint8 arrays")
    delta = np.abs(image.astype(np.float32) - template.astype(np.float32))
    return float(1.0 - np.mean(delta) / 255.0)
