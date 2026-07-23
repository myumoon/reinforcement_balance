"""Capture → perception → read-only action replay feasibility harness."""
from __future__ import annotations
from dataclasses import dataclass, replace
import json
import math
import os
from pathlib import Path
import time
from types import MappingProxyType
from typing import Any, Callable, Mapping
import uuid

from reinbalance_survivors_contracts.canonical_json import canonical_hash, canonical_json_bytes
from reinbalance_survivors_contracts.ui_intent import (
    ContractValidationError, ensure, is_strict_number, require_schema_version,
)
from .annotation_throughput import DatasetSplitRequest, estimate_dataset_budget
from .perception_probe import FeasibilityConfig, load_feasibility_config
from survivors.action_semantics import load_action_contract, validate_golden_fixture

EVIDENCE_VERSION = "perception_gate_evidence.v1"
_EVIDENCE_KEYS = frozenset({
    "schema_version", "session_ids", "build_ids", "profile_ids", "target_audit_pass",
    "session_minutes", "slice_counts", "representative_frames", "independent_annotators",
    "p10_short_side_px", "late_recall", "heavy_recall", "single_pass_p95_ms",
    "bbox_qa_iou", "class_agreement", "dense_entities_per_hour",
    "architecture_metrics", "unresolved_risks",
})
_REQUIRED_SLICES = ("early", "mid", "late", "heavy", "level_up", "chest", "death_result")


@dataclass(frozen=True)
class GateEvidence:
    session_ids: tuple[str, ...]
    build_ids: tuple[str, ...]
    profile_ids: tuple[str, ...]
    target_audit_pass: bool
    session_minutes: Mapping[str, float]
    slice_counts: Mapping[str, int]
    representative_frames: int
    independent_annotators: tuple[str, ...]
    p10_short_side_px: float
    late_recall: float
    heavy_recall: float
    single_pass_p95_ms: float
    bbox_qa_iou: float
    class_agreement: float
    dense_entities_per_hour: float
    architecture_metrics: Mapping[str, Mapping[str, float]]
    unresolved_risks: tuple[str, ...] = ()
    schema_version: str = EVIDENCE_VERSION

    def __post_init__(self) -> None:
        for name in ("session_ids", "build_ids", "profile_ids",
                     "independent_annotators", "unresolved_risks"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(self, "session_minutes",
                           MappingProxyType(dict(self.session_minutes)))
        object.__setattr__(self, "slice_counts", MappingProxyType(dict(self.slice_counts)))
        object.__setattr__(
            self, "architecture_metrics",
            MappingProxyType({key: MappingProxyType(dict(value))
                              for key, value in self.architecture_metrics.items()}))
        self.validate()

    def replace(self, **changes: Any) -> "GateEvidence":
        return replace(self, **changes)

    def validate(self, config: FeasibilityConfig | None = None) -> None:
        ensure(self.schema_version == EVIDENCE_VERSION, "gate evidence schema mismatch")
        ensure(type(self.target_audit_pass) is bool, "target audit pass must be boolean")
        ensure(all(isinstance(v, str) and v for group in
                   (self.session_ids, self.build_ids, self.profile_ids,
                    self.independent_annotators, self.unresolved_risks) for v in group),
               "evidence identifiers/risks must be non-empty strings")
        ensure(len(self.session_ids) == len(set(self.session_ids)),
               "session identifiers must be unique")
        ensure(set(self.session_minutes) == set(self.session_ids),
               "session duration identities must exactly match sessions")
        ensure(all(isinstance(k, str) and is_strict_number(v) and math.isfinite(v) and v >= 0
                   for k, v in self.session_minutes.items()),
               "session durations must be finite and non-negative")
        ensure(set(self.slice_counts) <= set(_REQUIRED_SLICES) and
               all(type(v) is int and v >= 0 for v in self.slice_counts.values()),
               "slice counts must be known non-negative integers")
        ensure(type(self.representative_frames) is int and self.representative_frames >= 0,
               "representative frames must be a non-negative integer")
        for name, value in {
            "p10 short side": self.p10_short_side_px,
            "single pass latency": self.single_pass_p95_ms,
            "dense entities/hour": self.dense_entities_per_hour,
        }.items():
            ensure(is_strict_number(value) and math.isfinite(value) and value >= 0,
                   f"{name} must be finite and non-negative")
        for name, value in {
            "late recall": self.late_recall, "heavy recall": self.heavy_recall,
            "bbox QA IoU": self.bbox_qa_iou, "class agreement": self.class_agreement,
        }.items():
            ensure(is_strict_number(value) and math.isfinite(value) and 0 <= value <= 1,
                   f"{name} must be finite and between zero and one")
        expected_architectures = set(config.architectures) if config else {
            "ssdlite320", "ssdlite640_multiscale", "tile_2x2", "coarse_density"}
        ensure(set(self.architecture_metrics) == expected_architectures,
               "unknown/missing architecture metrics")
        for metrics in self.architecture_metrics.values():
            ensure(isinstance(metrics, Mapping) and
                   set(metrics) == {"utility", "latency_p95_ms"},
                   "unknown/missing architecture metric fields")
            ensure(is_strict_number(metrics["utility"]) and
                   math.isfinite(metrics["utility"]) and 0 <= metrics["utility"] <= 1,
                   "architecture utility must be finite and between zero and one")
            ensure(is_strict_number(metrics["latency_p95_ms"]) and
                   math.isfinite(metrics["latency_p95_ms"]) and
                   metrics["latency_p95_ms"] > 0,
                   "architecture latency must be positive and finite")

    @classmethod
    def valid_fixture(cls) -> "GateEvidence":
        sessions = ("s1", "s2", "s3")
        slices = {name: 2 for name in _REQUIRED_SLICES}
        metrics = {
            "ssdlite320": {"utility": .80, "latency_p95_ms": 20.0},
            "ssdlite640_multiscale": {"utility": .91, "latency_p95_ms": 24.0},
            "tile_2x2": {"utility": .94, "latency_p95_ms": 39.0},
            "coarse_density": {"utility": .70, "latency_p95_ms": 6.0},
        }
        return cls(sessions, ("build-a",), ("profile-a",), True,
                   {s: 10.0 for s in sessions}, slices, 300, ("alice", "bob"),
                   6.0, .90, .88, 24.0, .85, .97, 400.0, metrics)

    def to_wire(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version, "session_ids": list(self.session_ids),
            "build_ids": list(self.build_ids), "profile_ids": list(self.profile_ids),
            "target_audit_pass": self.target_audit_pass,
            "session_minutes": dict(self.session_minutes), "slice_counts": dict(self.slice_counts),
            "representative_frames": self.representative_frames,
            "independent_annotators": list(self.independent_annotators),
            "p10_short_side_px": self.p10_short_side_px, "late_recall": self.late_recall,
            "heavy_recall": self.heavy_recall, "single_pass_p95_ms": self.single_pass_p95_ms,
            "bbox_qa_iou": self.bbox_qa_iou, "class_agreement": self.class_agreement,
            "dense_entities_per_hour": self.dense_entities_per_hour,
            "architecture_metrics": {k: dict(v) for k, v in self.architecture_metrics.items()},
            "unresolved_risks": list(self.unresolved_risks),
        }

    @classmethod
    def from_wire(cls, data: Mapping[str, Any]) -> "GateEvidence":
        ensure(isinstance(data, Mapping), "gate evidence must be a mapping")
        ensure(set(data) == _EVIDENCE_KEYS,
               f"unknown/missing gate evidence fields: {sorted(set(data) ^ _EVIDENCE_KEYS)}")
        require_schema_version(data, EVIDENCE_VERSION, "gate evidence")
        ensure(set(data["slice_counts"]) == set(_REQUIRED_SLICES),
               "unknown/missing slice count fields")
        result = cls(tuple(data["session_ids"]), tuple(data["build_ids"]), tuple(data["profile_ids"]),
                   data["target_audit_pass"], dict(data["session_minutes"]),
                   dict(data["slice_counts"]), data["representative_frames"],
                   tuple(data["independent_annotators"]), data["p10_short_side_px"],
                   data["late_recall"], data["heavy_recall"], data["single_pass_p95_ms"],
                   data["bbox_qa_iou"], data["class_agreement"],
                   data["dense_entities_per_hour"],
                   {k: dict(v) for k, v in data["architecture_metrics"].items()},
                   tuple(data["unresolved_risks"]))
        result.validate()
        return result


@dataclass(frozen=True)
class ActionReplayRecord:
    index: int
    proposal_vector: tuple[float, float]
    measured_screen_displacement: tuple[float, float]
    source_parent_hash: str
    capture_to_observation_latency_ms: float
    proposal_cadence_hz: float
    live_input_sent: bool = False

    def __post_init__(self) -> None:
        ensure(type(self.index) is int and self.index >= 0, "action index must be non-negative")
        for label, vector in (("proposal", self.proposal_vector),
                              ("measured displacement", self.measured_screen_displacement)):
            ensure(isinstance(vector, tuple) and len(vector) == 2 and
                   all(is_strict_number(v) and math.isfinite(v) for v in vector),
                   f"{label} vector must contain two finite numbers")
        ensure(isinstance(self.source_parent_hash, str) and len(self.source_parent_hash) == 64 and
               all(character in "0123456789abcdef" for character in self.source_parent_hash),
               "source parent hash must be SHA-256 text")
        ensure(is_strict_number(self.capture_to_observation_latency_ms) and
               math.isfinite(self.capture_to_observation_latency_ms) and
               self.capture_to_observation_latency_ms >= 0,
               "capture latency must be finite and non-negative")
        ensure(is_strict_number(self.proposal_cadence_hz) and
               math.isfinite(self.proposal_cadence_hz) and self.proposal_cadence_hz > 0,
               "proposal cadence must be positive and finite")
        ensure(type(self.live_input_sent) is bool and not self.live_input_sent,
               "spike action replay cannot send live input")


def replay_action_displacement(path: Path, *, latency_ms: float = 0.0,
                               proposal_cadence_hz: float = 15.0) -> tuple[ActionReplayRecord, ...]:
    """Read 00-03 telemetry only. This function has no input-driver dependency."""
    ensure(is_strict_number(latency_ms) and math.isfinite(latency_ms) and latency_ms >= 0,
           "latency must be finite and non-negative")
    ensure(is_strict_number(proposal_cadence_hz) and math.isfinite(proposal_cadence_hz) and
           proposal_cadence_hz > 0, "proposal cadence must be positive and finite")
    contract = load_action_contract()
    validate_golden_fixture(contract, path)
    payload = path.read_bytes()
    data = json.loads(payload)
    parent_hash = canonical_hash(data)
    records = []
    for sample in data["samples"]:
        ensure(set(sample) == {"index", "sim_delta_sign", "screen_delta_sign", "keys"},
               "unknown/missing action sample fields")
        records.append(ActionReplayRecord(
            sample["index"], tuple(float(x) for x in sample["sim_delta_sign"]),
            tuple(float(x) for x in sample["screen_delta_sign"]), parent_hash,
            float(latency_ms), float(proposal_cadence_hz), False))
    return tuple(records)


@dataclass(frozen=True)
class FrameMetadata:
    sequence: int
    captured_monotonic_ns: int
    width: int
    height: int
    dropped_since_previous: int
    duplicate: bool
    encoded_bytes: int

    def __post_init__(self) -> None:
        for label in ("sequence", "captured_monotonic_ns", "dropped_since_previous",
                      "encoded_bytes"):
            ensure(type(getattr(self, label)) is int and getattr(self, label) >= 0,
                   f"frame {label} must be a non-negative integer")
        ensure(type(self.width) is int and self.width > 0 and
               type(self.height) is int and self.height > 0,
               "frame dimensions must be positive integers")
        ensure(type(self.duplicate) is bool, "frame duplicate must be boolean")


class LatestFrameSampler:
    """Backend-neutral 30 FPS latest-frame sampler with drop/duplicate accounting."""
    def __init__(self, getter: Callable[[], Any], fps: float = 30.0):
        ensure(callable(getter), "frame getter must be callable")
        ensure(is_strict_number(fps) and math.isfinite(fps) and fps > 0,
               "fps must be positive and finite")
        self.getter, self.period_ns, self.sequence = getter, int(1e9/fps), 0
        self._last_hash: str | None = None
        self._last_capture_ns: int | None = None

    def capture(self) -> tuple[Any, FrameMetadata]:
        frame = self.getter()
        ensure(frame is not None and getattr(frame, "ndim", None) == 3, "capture returned no frame")
        now = time.monotonic_ns()
        frame_hash = canonical_hash({"bytes_hex": frame.tobytes().hex()})
        dropped = 0 if self._last_capture_ns is None else max(
            0, (now-self._last_capture_ns)//self.period_ns-1)
        meta = FrameMetadata(self.sequence, now, frame.shape[1], frame.shape[0],
                             int(dropped), frame_hash == self._last_hash, int(frame.nbytes))
        self.sequence += 1
        self._last_capture_ns, self._last_hash = now, frame_hash
        return frame, meta

    @classmethod
    def from_dxcam(cls, fps: float = 30.0) -> "LatestFrameSampler":
        try:
            import dxcam  # type: ignore
        except ImportError as exc:
            raise RuntimeError("DXcam is optional and required only for live Windows capture") from exc
        camera = dxcam.create(output_color="BGRA")
        camera.start(target_fps=int(fps), video_mode=True)
        return cls(camera.get_latest_frame, fps)


def _gate_failures(e: GateEvidence, c: FeasibilityConfig) -> list[str]:
    p, t = c.pilot, c.thresholds
    failures = []
    if not e.target_audit_pass or len(set(e.build_ids)) != 1 or len(set(e.profile_ids)) != 1:
        failures.append("mixed_identity_or_target_audit_not_pass")
    if not (p["min_sessions"] <= len(set(e.session_ids)) <= p["max_sessions"]):
        failures.append("session_count_out_of_range")
    if set(e.session_minutes) != set(e.session_ids) or any(
            minutes < p["min_minutes_per_session"] for minutes in e.session_minutes.values()):
        failures.append("session_duration_insufficient")
    if set(e.slice_counts) != set(p["required_slices"]) or any(
            e.slice_counts.get(name, 0) < p["min_sessions_per_slice"] for name in _REQUIRED_SLICES):
        failures.append("missing_required_slices")
    if e.representative_frames < p["min_frames"]:
        failures.append("representative_frames_insufficient")
    if len(set(e.independent_annotators)) < 2:
        failures.append("independent_annotation_reviewer_missing")
    if set(e.architecture_metrics) != set(c.architectures):
        failures.append("architecture_matrix_incomplete")
    return failures


def issue_verdict(evidence: GateEvidence, config: FeasibilityConfig) -> dict[str, Any]:
    ensure(isinstance(evidence, GateEvidence), "validated gate evidence is required")
    ensure(isinstance(config, FeasibilityConfig), "validated feasibility config is required")
    config.validate()
    evidence.validate(config)
    failures = _gate_failures(evidence, config)
    t = config.thresholds
    reject_320 = (evidence.p10_short_side_px < t["ssdlite320_p10_short_side_px"] or
                  evidence.late_recall < t["oracle_recall"] or
                  evidence.heavy_recall < t["oracle_recall"])
    qa_alternative = (evidence.bbox_qa_iou < t["bbox_qa_iou"] or
                      evidence.class_agreement < t["class_agreement"] or
                      evidence.dense_entities_per_hour < t["dense_entities_per_hour"])
    candidates = dict(evidence.architecture_metrics)
    if reject_320:
        candidates.pop("ssdlite320", None)
    # Slow single pass does not force tiling: compare utility / latency for every candidate.
    selected = max(candidates, key=lambda k: candidates[k]["utility"] /
                   candidates[k]["latency_p95_ms"]) if candidates else None
    if selected is None:
        failures.append("architecture_switch_condition_unsatisfied")
    status = "PASS" if not failures else "FAIL"
    rejected = [name for name in config.architectures if name != selected]
    return {
        "schema_version": "perception_feasibility_verdict.v1",
        "status": status, "subject_evidence_hash": canonical_hash(evidence.to_wire()),
        "selected_architecture": selected if status == "PASS" else None,
        "provisional_architecture": selected,
        "rejected_alternatives": rejected,
        "switch_conditions": {
            "reject_ssdlite320": reject_320,
            "compare_utility_per_latency": evidence.single_pass_p95_ms > t["single_pass_p95_ms"],
            "consider_segmentation_density_count": qa_alternative,
        },
        "fail_reasons": failures, "unresolved_risks": list(evidence.unresolved_risks),
        "downstream": {"allow_04_01": status == "PASS",
                       "allow_long_run_student": status == "PASS"},
    }


def render_verdict_markdown(verdict: Mapping[str, Any], budget: Mapping[str, Any]) -> str:
    rejected = ", ".join(verdict["rejected_alternatives"])
    risks = ", ".join(verdict["unresolved_risks"]) or "none recorded"
    return f"""# Survivors perception feasibility gate

Status: **{verdict['status']}**

- Selected architecture: `{verdict['selected_architecture']}`
- Provisional architecture (diagnostic only on FAIL): `{verdict['provisional_architecture']}`
- Rejected alternatives: {rejected}
- Switch conditions: `{json.dumps(verdict['switch_conditions'], sort_keys=True)}`
- Unresolved risks: {risks}
- Budget totals: `{json.dumps(budget['totals'], sort_keys=True)}`

This verdict is bound to evidence hash `{verdict['subject_evidence_hash']}`. A FAIL
does not authorize a production/formal verdict: do not start 04-01 or later
perception work, and do not start long-run student training until a current,
target-audited PASS is issued.

Action displacement is read only from the 00-03 parent telemetry. Every replay
record keeps `proposal_vector` separate from `measured_screen_displacement`; this
spike never sends live input.
"""


def default_budget(config: FeasibilityConfig) -> dict[str, Any]:
    ensure(isinstance(config, FeasibilityConfig), "validated feasibility config is required")
    config.validate()
    splits = (
        DatasetSplitRequest("development", 3, 1800, 36000, 240),
        DatasetSplitRequest("calibration", 2, 900, 18000, 120),
        DatasetSplitRequest("untouched_final", 2, 900, 18000, 120),
    )
    return estimate_dataset_budget(
        splits, entities_per_hour=400, ui_events_per_hour=120,
        frame_storage_bytes=int(config.budget["frame_storage_bytes"]),
        gpu_seconds_per_frame=float(config.budget["gpu_seconds_per_frame"]),
        parallel_worker_limit=int(config.budget["parallel_worker_limit"]))


def write_verdict(evidence: GateEvidence, json_path: Path, markdown_path: Path,
                  config: FeasibilityConfig | None = None) -> dict[str, Any]:
    config = config or load_feasibility_config()
    verdict, budget = issue_verdict(evidence, config), default_budget(config)
    output = {**verdict, "budget": budget}
    json_path, markdown_path = Path(json_path), Path(markdown_path)
    ensure(json_path != markdown_path, "JSON and Markdown verdict paths must differ")
    token = uuid.uuid4().hex
    json_temp = json_path.with_name(f".{json_path.name}.{token}.tmp")
    markdown_temp = markdown_path.with_name(f".{markdown_path.name}.{token}.tmp")
    paths = ((json_path, json_temp), (markdown_path, markdown_temp))
    backups: list[tuple[Path, Path]] = []
    installed: list[Path] = []
    try:
        json_temp.write_bytes(canonical_json_bytes(output) + b"\n")
        markdown_temp.write_text(render_verdict_markdown(verdict, budget), encoding="utf-8")
        for final, _ in paths:
            if final.exists():
                backup = final.with_name(f".{final.name}.{token}.bak")
                os.replace(final, backup)
                backups.append((final, backup))
        for final, temporary in paths:
            os.replace(temporary, final)
            installed.append(final)
    except BaseException:
        for final in installed:
            try:
                final.unlink()
            except FileNotFoundError:
                pass
        for final, backup in reversed(backups):
            if backup.exists():
                os.replace(backup, final)
        raise
    finally:
        for _, temporary in paths:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        for _, backup in backups:
            try:
                backup.unlink()
            except FileNotFoundError:
                pass
    return output
