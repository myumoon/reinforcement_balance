"""Pure contracts for a fixed-size Survivors live-canary campaign."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from reinbalance_survivors_contracts.canonical_json import (
    canonical_hash,
    canonical_json_bytes,
    sha256_hex,
)

CAMPAIGN_SCHEMA_VERSION = "survivors.campaign.v1"
CAMPAIGN_SLOT_COUNT = 20
REQUIRED_PREREQUISITES = (
    "exact_runtime",
    "target",
    "save",
    "build",
    "hardware",
    "perception_final",
    "shadow",
    "replay",
    "input_safety",
    "restore",
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_FORBIDDEN_CLAIMS = frozenset(
    {
        "seed",
        "real_seed",
        "same_seed",
        "same_seed_claim",
        "independent",
        "statistical_independence",
        "independence_claim",
        "population_success_probability",
    }
)


class EventType(StrEnum):
    FORMAL_SLOT_RESERVED = "FORMAL_SLOT_RESERVED"
    ATTEMPT_PREFLIGHT = "ATTEMPT_PREFLIGHT"
    PREFLIGHT_FAILED = "PREFLIGHT_FAILED"
    LAUNCH_INTENT_COMMITTED = "LAUNCH_INTENT_COMMITTED"
    BROKER_PROCESS_ATTESTED = "BROKER_PROCESS_ATTESTED"
    PROCESS_LAUNCH_CONFIRMED = "PROCESS_LAUNCH_CONFIRMED"
    LAUNCH_GATE_FAILED = "LAUNCH_GATE_FAILED"
    LAUNCH_UNCERTAIN = "LAUNCH_UNCERTAIN"
    FORMAL_RUN_ACTIVATED = "FORMAL_RUN_ACTIVATED"
    SUCCESS = "SUCCESS"
    GAMEPLAY_FAILURE = "GAMEPLAY_FAILURE"
    SAFETY_FAILURE = "SAFETY_FAILURE"
    ARTIFACT_FAILURE = "ARTIFACT_FAILURE"


TERMINAL_OUTCOMES = frozenset(
    {
        EventType.SUCCESS,
        EventType.GAMEPLAY_FAILURE,
        EventType.SAFETY_FAILURE,
        EventType.ARTIFACT_FAILURE,
    }
)
_ALL_EVENT_TYPES = frozenset(EventType)
_EVENT_FIELDS = {
    EventType.FORMAL_SLOT_RESERVED: frozenset(),
    EventType.ATTEMPT_PREFLIGHT: frozenset({"attempt_id"}),
    EventType.PREFLIGHT_FAILED: frozenset({"attempt_id", "failure_reason"}),
    EventType.LAUNCH_INTENT_COMMITTED: frozenset(
        {"attempt_id", "reserved_run_id", "gameplay_attempt_id", "launch_nonce"}
    ),
    EventType.BROKER_PROCESS_ATTESTED: frozenset({"process_ref", "job_ref"}),
    EventType.PROCESS_LAUNCH_CONFIRMED: frozenset(),
    EventType.LAUNCH_GATE_FAILED: frozenset({"failure_reason"}),
    EventType.LAUNCH_UNCERTAIN: frozenset({"failure_reason"}),
    EventType.FORMAL_RUN_ACTIVATED: frozenset({"activation_source"}),
    EventType.SUCCESS: frozenset(),
    EventType.GAMEPLAY_FAILURE: frozenset({"failure_reason"}),
    EventType.SAFETY_FAILURE: frozenset({"failure_reason"}),
    EventType.ARTIFACT_FAILURE: frozenset({"failure_reason"}),
}
_WIRE_FIELDS = frozenset(
    {
        "event_type",
        "slot_id",
        "attempt_id",
        "reserved_run_id",
        "gameplay_attempt_id",
        "launch_nonce",
        "process_ref",
        "job_ref",
        "activation_source",
        "failure_reason",
        "details",
    }
)


def _reject_claim_fields(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if isinstance(key, str) and key.lower() in _FORBIDDEN_CLAIMS:
                raise ValueError(f"statistical claim field is forbidden: {key}")
            _reject_claim_fields(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _reject_claim_fields(child)


def _exact_keys(value: Mapping[str, Any], expected: set[str] | frozenset[str], name: str) -> None:
    missing = expected - value.keys()
    unknown = value.keys() - expected
    if missing:
        raise ValueError(f"{name} missing fields: {', '.join(sorted(missing))}")
    if unknown:
        raise ValueError(f"{name} has unknown fields: {', '.join(sorted(unknown))}")


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(child) for key, child in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(child) for child in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_thaw(child) for child in value]
    return value


@dataclass(frozen=True, slots=True)
class PrerequisiteSet:
    hashes: Mapping[str, str]
    parents: Mapping[str, str]
    statuses: Mapping[str, str]
    cloud_sync_status: str
    backup_hash: str
    pre_save_contract_hash: str
    post_save_contract_hash: str
    development_only: bool = False

    def __post_init__(self) -> None:
        for name in ("hashes", "parents", "statuses"):
            values = getattr(self, name)
            if not isinstance(values, Mapping) or set(values) != set(REQUIRED_PREREQUISITES):
                raise ValueError(f"{name} must contain the exact prerequisite set")
            copied = dict(values)
            object.__setattr__(self, name, MappingProxyType(copied))
        for name, value in self.hashes.items():
            _digest(value, f"hashes.{name}")
        for name, value in self.parents.items():
            _digest(value, f"parents.{name}")
        if len(set(self.parents.values())) != 1:
            raise ValueError("prerequisites have mixed parents")
        if any(value != "PASS" for value in self.statuses.values()):
            raise ValueError("all prerequisites must be current PASS evidence")
        if self.cloud_sync_status != "verified":
            raise ValueError("cloud sync status must be verified")
        for name in ("backup_hash", "pre_save_contract_hash", "post_save_contract_hash"):
            _digest(getattr(self, name), name)
        if type(self.development_only) is not bool or self.development_only:
            raise ValueError("formal prerequisites cannot be development-only")

    def to_wire(self) -> dict[str, Any]:
        return {
            "hashes": dict(self.hashes),
            "parents": dict(self.parents),
            "statuses": dict(self.statuses),
            "cloud_sync_status": self.cloud_sync_status,
            "backup_hash": self.backup_hash,
            "pre_save_contract_hash": self.pre_save_contract_hash,
            "post_save_contract_hash": self.post_save_contract_hash,
            "development_only": self.development_only,
        }


def validate_prerequisites(
    value: Mapping[str, Any], *, expected_parent_hash: str
) -> PrerequisiteSet:
    if not isinstance(value, Mapping):
        raise ValueError("prerequisites must be an object")
    _reject_claim_fields(value)
    fields = {
        "hashes",
        "parents",
        "statuses",
        "cloud_sync_status",
        "backup_hash",
        "pre_save_contract_hash",
        "post_save_contract_hash",
        "development_only",
    }
    _exact_keys(value, fields, "prerequisites")
    expected_parent_hash = _digest(expected_parent_hash, "expected_parent_hash")
    for name in ("hashes", "parents", "statuses"):
        item = value[name]
        if not isinstance(item, Mapping):
            raise ValueError(f"{name} must be an object")
        _exact_keys(item, frozenset(REQUIRED_PREREQUISITES), name)
    parents = value["parents"]
    if any(parent != expected_parent_hash for parent in parents.values()):
        raise ValueError("prerequisite parent is stale or mixed")
    return PrerequisiteSet(
        hashes=value["hashes"],
        parents=parents,
        statuses=value["statuses"],
        cloud_sync_status=value["cloud_sync_status"],
        backup_hash=value["backup_hash"],
        pre_save_contract_hash=value["pre_save_contract_hash"],
        post_save_contract_hash=value["post_save_contract_hash"],
        development_only=value["development_only"],
    )


@dataclass(frozen=True, slots=True)
class CampaignManifest:
    campaign_id: str
    schema_version: str = CAMPAIGN_SCHEMA_VERSION
    mode: str = "synthetic"
    expected_slots: int = CAMPAIGN_SLOT_COUNT
    rng_control: str = "uncontrolled"
    trial_separation: str = "unique_run_id_separate_process"
    development_only: bool = True
    prerequisites: PrerequisiteSet | None = None
    prerequisite_parent_hash: str | None = None

    def __post_init__(self) -> None:
        _text(self.campaign_id, "campaign_id")
        if self.schema_version != CAMPAIGN_SCHEMA_VERSION:
            raise ValueError("unsupported campaign schema version")
        if type(self.expected_slots) is not int or self.expected_slots != CAMPAIGN_SLOT_COUNT:
            raise ValueError(f"expected_slots must be {CAMPAIGN_SLOT_COUNT}")
        if self.mode not in {"synthetic", "formal"}:
            raise ValueError("mode must be synthetic or formal")
        if self.rng_control != "uncontrolled":
            raise ValueError('rng_control must be "uncontrolled"')
        if self.trial_separation != "unique_run_id_separate_process":
            raise ValueError('trial_separation must be "unique_run_id_separate_process"')
        if type(self.development_only) is not bool or self.development_only != (self.mode == "synthetic"):
            raise ValueError("development_only must match campaign mode")
        if self.mode == "synthetic":
            if self.prerequisites is not None or self.prerequisite_parent_hash is not None:
                raise ValueError("synthetic fixtures cannot issue formal prerequisites")
        else:
            if not isinstance(self.prerequisites, PrerequisiteSet):
                raise ValueError("formal campaign requires verified prerequisites")
            parent = _digest(self.prerequisite_parent_hash, "prerequisite_parent_hash")
            validate_prerequisites(
                self.prerequisites.to_wire(), expected_parent_hash=parent
            )

    def to_wire(self) -> dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "schema_version": self.schema_version,
            "mode": self.mode,
            "expected_slots": self.expected_slots,
            "rng_control": self.rng_control,
            "trial_separation": self.trial_separation,
            "development_only": self.development_only,
            "prerequisites": None if self.prerequisites is None else self.prerequisites.to_wire(),
            "prerequisite_parent_hash": self.prerequisite_parent_hash,
        }

    @classmethod
    def from_wire(cls, value: Mapping[str, Any]) -> CampaignManifest:
        if not isinstance(value, Mapping):
            raise ValueError("campaign manifest must be an object")
        _reject_claim_fields(value)
        fields = {
            "campaign_id",
            "schema_version",
            "mode",
            "expected_slots",
            "rng_control",
            "trial_separation",
            "development_only",
            "prerequisites",
            "prerequisite_parent_hash",
        }
        _exact_keys(value, fields, "campaign manifest")
        prerequisites = value["prerequisites"]
        parent_hash = value["prerequisite_parent_hash"]
        if prerequisites is not None:
            prerequisites = validate_prerequisites(
                prerequisites, expected_parent_hash=parent_hash
            )
        return cls(**{**value, "prerequisites": prerequisites})


@dataclass(frozen=True, slots=True)
class CampaignEvent:
    event_type: EventType | str
    slot_id: int
    attempt_id: str | None = None
    reserved_run_id: str | None = None
    gameplay_attempt_id: str | None = None
    launch_nonce: str | None = None
    process_ref: str | None = None
    job_ref: str | None = None
    activation_source: str | None = None
    failure_reason: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        try:
            event_type = EventType(self.event_type)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"unknown campaign event: {self.event_type!r}") from exc
        object.__setattr__(self, "event_type", event_type)
        if type(self.slot_id) is not int or self.slot_id < 0:
            raise ValueError("slot_id must be a non-negative integer")
        present = {
            name for name in _EVENT_FIELDS[event_type] if getattr(self, name) is not None
        }
        missing = _EVENT_FIELDS[event_type] - present
        if missing:
            raise ValueError(f"{event_type.value} missing fields: {', '.join(sorted(missing))}")
        optional = {
            "attempt_id",
            "reserved_run_id",
            "gameplay_attempt_id",
            "launch_nonce",
            "process_ref",
            "job_ref",
            "activation_source",
            "failure_reason",
        }
        unexpected = {name for name in optional if getattr(self, name) is not None} - _EVENT_FIELDS[event_type]
        if unexpected:
            raise ValueError(f"{event_type.value} has unexpected fields: {', '.join(sorted(unexpected))}")
        for name in _EVENT_FIELDS[event_type]:
            _text(getattr(self, name), name)
        if event_type is EventType.FORMAL_RUN_ACTIVATED and self.activation_source not in {
            "normal",
            "reconciliation",
        }:
            raise ValueError("activation_source must be normal or reconciliation")
        if not isinstance(self.details, Mapping):
            raise ValueError("details must be an object")
        _reject_claim_fields(self.details)
        try:
            details = json.loads(canonical_json_bytes(dict(self.details)))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"details must contain canonical JSON values: {exc}") from exc
        object.__setattr__(self, "details", _freeze(details))

    def to_wire(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "event_type": self.event_type.value,
            "slot_id": self.slot_id,
        }
        for name in _WIRE_FIELDS - {"event_type", "slot_id", "details"}:
            item = getattr(self, name)
            if item is not None:
                value[name] = item
        if self.details:
            value["details"] = _thaw(self.details)
        return value

    @classmethod
    def from_wire(cls, value: Mapping[str, Any]) -> CampaignEvent:
        if not isinstance(value, Mapping):
            raise ValueError("campaign event must be an object")
        _reject_claim_fields(value)
        unknown = value.keys() - _WIRE_FIELDS
        if unknown:
            raise ValueError(f"campaign event has unknown fields: {', '.join(sorted(unknown))}")
        if not {"event_type", "slot_id"} <= value.keys():
            raise ValueError("campaign event requires event_type and slot_id")
        return cls(**value)


def canonical_event_jsonl(events: Sequence[CampaignEvent]) -> bytes:
    lines = []
    for item in events:
        event = item if isinstance(item, CampaignEvent) else CampaignEvent.from_wire(item)
        lines.append(canonical_json_bytes(event.to_wire()) + b"\n")
    return b"".join(lines)


def campaign_event_hash(events: Sequence[CampaignEvent]) -> str:
    normalized = [
        (item if isinstance(item, CampaignEvent) else CampaignEvent.from_wire(item)).to_wire()
        for item in events
    ]
    return canonical_hash(normalized)


def campaign_jsonl_hash(events: Sequence[CampaignEvent]) -> str:
    return sha256_hex(canonical_event_jsonl(events))


def campaign_manifest_hash(manifest: CampaignManifest) -> str:
    return canonical_hash(manifest.to_wire())


def validate_campaign_events(
    events: Sequence[CampaignEvent | Mapping[str, Any]], *, expected_slots: int = CAMPAIGN_SLOT_COUNT
) -> dict[int, tuple[CampaignEvent, ...]]:
    if type(expected_slots) is not int or expected_slots <= 0:
        raise ValueError("expected_slots must be a positive integer")
    if isinstance(events, (str, bytes)) or not isinstance(events, Sequence):
        raise ValueError("events must be a sequence")
    grouped: dict[int, list[CampaignEvent]] = {}
    states: dict[int, str] = {}
    identities: dict[str, set[str]] = {
        "attempt_id": set(),
        "reserved_run_id": set(),
        "gameplay_attempt_id": set(),
        "launch_nonce": set(),
        "process_ref": set(),
        "job_ref": set(),
    }
    bound: dict[int, dict[str, str]] = {}
    for raw in events:
        event = raw if isinstance(raw, CampaignEvent) else CampaignEvent.from_wire(raw)
        if event.slot_id >= expected_slots:
            raise ValueError(f"slot_id {event.slot_id} is outside campaign capacity")
        slot = event.slot_id
        kind = event.event_type
        state = states.get(slot)
        if state == "terminal":
            raise ValueError(f"slot {slot} has an event after its terminal outcome")
        if kind is EventType.FORMAL_SLOT_RESERVED:
            if state is not None:
                raise ValueError(f"slot {slot} was reused or replaced")
            states[slot] = "reserved"
        elif kind is EventType.ATTEMPT_PREFLIGHT:
            if state != "reserved":
                raise ValueError(f"slot {slot} preflight requires one reserved slot")
            _unique(identities["attempt_id"], event.attempt_id, "attempt_id")
            bound[slot] = {"attempt_id": event.attempt_id}
            states[slot] = "preflight"
        elif kind is EventType.PREFLIGHT_FAILED:
            _require_bound_attempt(event, bound, state, "preflight")
            states[slot] = "terminal"
        elif kind is EventType.LAUNCH_INTENT_COMMITTED:
            _require_bound_attempt(event, bound, state, "preflight")
            for name in ("reserved_run_id", "gameplay_attempt_id", "launch_nonce"):
                _unique(identities[name], getattr(event, name), name)
            bound[slot].update(
                reserved_run_id=event.reserved_run_id,
                gameplay_attempt_id=event.gameplay_attempt_id,
            )
            states[slot] = "launch_intent"
        elif kind is EventType.BROKER_PROCESS_ATTESTED:
            if state != "launch_intent":
                raise ValueError(f"slot {slot} process attestation requires launch intent")
            _unique(identities["process_ref"], event.process_ref, "process_ref")
            _unique(identities["job_ref"], event.job_ref, "job_ref")
            bound[slot].update(process_ref=event.process_ref, job_ref=event.job_ref)
            states[slot] = "attested"
        elif kind is EventType.PROCESS_LAUNCH_CONFIRMED:
            if state != "attested":
                raise ValueError(f"slot {slot} launch confirmation requires process attestation")
            states[slot] = "confirmed"
        elif kind is EventType.FORMAL_RUN_ACTIVATED:
            if state != "confirmed":
                raise ValueError(f"slot {slot} activation requires attempt, process, and gameplay identity")
            states[slot] = "activated"
        elif kind in {EventType.PREFLIGHT_FAILED, EventType.LAUNCH_GATE_FAILED, EventType.LAUNCH_UNCERTAIN}:
            if kind is EventType.PREFLIGHT_FAILED:
                continue
            if state not in {"preflight", "launch_intent", "attested", "confirmed"}:
                raise ValueError(f"slot {slot} launch gate failure is out of order")
            states[slot] = "terminal"
        elif kind in TERMINAL_OUTCOMES:
            if state != "activated":
                raise ValueError(f"slot {slot} terminal outcome requires activation")
            states[slot] = "terminal"
        grouped.setdefault(slot, []).append(event)
    return {slot: tuple(items) for slot, items in grouped.items()}


def _unique(values: set[str], value: str | None, name: str) -> None:
    if value is None:
        raise ValueError(f"{name} is required")
    if value in values:
        raise ValueError(f"duplicate {name}: {value}")
    values.add(value)


def _require_bound_attempt(
    event: CampaignEvent, bound: Mapping[int, Mapping[str, str]], state: str | None, expected_state: str
) -> None:
    if state != expected_state or event.slot_id not in bound:
        raise ValueError(f"slot {event.slot_id} event requires preflight")
    if event.attempt_id != bound[event.slot_id]["attempt_id"]:
        raise ValueError(f"slot {event.slot_id} attempt identity changed")
