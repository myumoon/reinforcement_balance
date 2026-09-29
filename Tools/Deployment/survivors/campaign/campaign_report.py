"""Deterministic campaign aggregation using only Python's standard library."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

from reinbalance_survivors_contracts.canonical_json import canonical_hash

from .campaign_schema import (
    CAMPAIGN_SLOT_COUNT,
    TERMINAL_OUTCOMES,
    CampaignEvent,
    CampaignManifest,
    EventType,
    STAGE_POLICIES,
    StagePolicy,
    campaign_event_hash,
    campaign_manifest_hash,
    validate_campaign_events,
)

_WILSON_Z_95 = 1.959963984540054


def wilson_score_interval(successes: int, denominator: int) -> tuple[float, float] | None:
    """Return a two-sided 95% Wilson score interval for the observed rate."""
    if type(successes) is not int or type(denominator) is not int:
        raise ValueError("successes and denominator must be integers")
    if denominator < 0 or successes < 0 or successes > denominator:
        raise ValueError("require 0 <= successes <= denominator")
    if denominator == 0:
        return None
    p = successes / denominator
    z2 = _WILSON_Z_95**2
    scale = 1 + z2 / denominator
    center = (p + z2 / (2 * denominator)) / scale
    margin = (
        _WILSON_Z_95
        * math.sqrt(p * (1 - p) / denominator + z2 / (4 * denominator**2))
        / scale
    )
    return max(0.0, center - margin), min(1.0, center + margin)


@dataclass(frozen=True, slots=True)
class CampaignReport:
    campaign_id: str
    stage: str
    duration_seconds: int
    planned_slots: int
    promotion_floor: int
    denominator: int
    successes: int
    incomplete_slot_ids: tuple[int, ...]
    observed_rate: float | None
    wilson_ci: tuple[float, float] | None
    preflight_failures: int
    launch_gate_failures: int
    uncertain_launches: int
    activated_failure_counts: Mapping[str, int]
    blocked_slot_ids: tuple[int, ...]
    stage_blocked: bool
    promotion_eligible: bool
    replacement_allowed: bool
    support_outside_ui: tuple[str, ...]
    failure_taxonomy: Mapping[str, int]
    campaign_chain: Mapping[str, tuple[str, ...]]
    development_only: bool
    formal_parent_eligible: bool
    manifest_hash: str
    prerequisite_parent_hash: str | None
    event_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "activated_failure_counts", MappingProxyType(dict(self.activated_failure_counts)))
        object.__setattr__(self, "failure_taxonomy", MappingProxyType(dict(self.failure_taxonomy)))
        object.__setattr__(
            self,
            "campaign_chain",
            MappingProxyType({name: tuple(ids) for name, ids in self.campaign_chain.items()}),
        )

    def to_wire(self) -> dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "stage": self.stage,
            "duration_seconds": self.duration_seconds,
            "planned_slots": self.planned_slots,
            "promotion_floor": self.promotion_floor,
            "denominator": self.denominator,
            "successes": self.successes,
            "incomplete_slot_ids": list(self.incomplete_slot_ids),
            "observed_rate": self.observed_rate,
            "wilson_ci": None if self.wilson_ci is None else list(self.wilson_ci),
            "preflight_failures": self.preflight_failures,
            "launch_gate_failures": self.launch_gate_failures,
            "uncertain_launches": self.uncertain_launches,
            "activated_failure_counts": dict(self.activated_failure_counts),
            "blocked_slot_ids": list(self.blocked_slot_ids),
            "stage_blocked": self.stage_blocked,
            "promotion_eligible": self.promotion_eligible,
            "replacement_allowed": self.replacement_allowed,
            "support_outside_ui": list(self.support_outside_ui),
            "failure_taxonomy": dict(self.failure_taxonomy),
            "campaign_chain": {name: list(ids) for name, ids in self.campaign_chain.items()},
            "development_only": self.development_only,
            "formal_parent_eligible": self.formal_parent_eligible,
            "manifest_hash": self.manifest_hash,
            "prerequisite_parent_hash": self.prerequisite_parent_hash,
            "event_hash": self.event_hash,
        }

    @property
    def report_hash(self) -> str:
        return canonical_hash(self.to_wire())


def generate_campaign_report(
    manifest: CampaignManifest,
    events: Sequence[CampaignEvent | Mapping[str, Any]],
    *,
    event_manifest_hash: str,
    stage: str | None = None,
    support_outside_ui: Sequence[str] = (),
    blocked_campaign_ids: Sequence[str] = (),
    superseded_campaign_ids: Sequence[str] = (),
) -> CampaignReport:
    if not isinstance(manifest, CampaignManifest):
        raise ValueError("manifest must be a validated CampaignManifest")
    if stage is not None and stage != manifest.stage:
        raise ValueError("report stage does not match manifest stage")
    stage = manifest.stage
    try:
        policy = STAGE_POLICIES[stage]
    except KeyError as exc:
        raise ValueError(f"unknown campaign stage: {stage}") from exc
    manifest_hash = campaign_manifest_hash(manifest)
    if event_manifest_hash != manifest_hash:
        raise ValueError("event manifest hash does not match manifest")
    normalized = tuple(
        event if isinstance(event, CampaignEvent) else CampaignEvent.from_wire(event)
        for event in events
    )
    grouped = validate_campaign_events(normalized, expected_slots=manifest.expected_slots)
    if manifest.expected_slots != policy.slot_count:
        raise ValueError("manifest expected_slots does not match stage policy")
    support = _unique_texts(support_outside_ui, "support_outside_ui")
    blocked = _unique_texts(blocked_campaign_ids, "blocked_campaign_ids")
    superseded = _unique_texts(superseded_campaign_ids, "superseded_campaign_ids")

    activated = sum(event.event_type is EventType.FORMAL_RUN_ACTIVATED for event in normalized)
    successes = sum(event.event_type is EventType.SUCCESS for event in normalized)
    preflight = sum(event.event_type is EventType.PREFLIGHT_FAILED for event in normalized)
    launch_gate = sum(event.event_type is EventType.LAUNCH_GATE_FAILED for event in normalized)
    uncertain = sum(event.event_type is EventType.LAUNCH_UNCERTAIN for event in normalized)
    incomplete = tuple(
        slot
        for slot in range(policy.slot_count)
        if not grouped.get(slot)
        or grouped[slot][-1].event_type
        not in TERMINAL_OUTCOMES
        | {
            EventType.PREFLIGHT_FAILED,
            EventType.LAUNCH_GATE_FAILED,
            EventType.LAUNCH_UNCERTAIN,
        }
    )
    activated_failures = Counter(
        event.event_type.value
        for event in normalized
        if event.event_type in TERMINAL_OUTCOMES and event.event_type is not EventType.SUCCESS
    )
    taxonomy: Counter[str] = Counter()
    blocked_slots: set[int] = set()
    for event in normalized:
        if event.event_type in {
            EventType.PREFLIGHT_FAILED,
            EventType.LAUNCH_GATE_FAILED,
            EventType.LAUNCH_UNCERTAIN,
            EventType.GAMEPLAY_FAILURE,
            EventType.SAFETY_FAILURE,
            EventType.ARTIFACT_FAILURE,
        }:
            prefix = {
                EventType.PREFLIGHT_FAILED: "preflight",
                EventType.LAUNCH_GATE_FAILED: "launch_gate",
                EventType.LAUNCH_UNCERTAIN: "uncertain_launch",
                EventType.GAMEPLAY_FAILURE: "gameplay_failure",
                EventType.SAFETY_FAILURE: "safety_failure",
                EventType.ARTIFACT_FAILURE: "artifact_failure",
            }[event.event_type]
            taxonomy[f"{prefix}:{event.failure_reason}"] += 1
        if event.event_type in {
            EventType.PREFLIGHT_FAILED,
            EventType.LAUNCH_GATE_FAILED,
            EventType.LAUNCH_UNCERTAIN,
        }:
            blocked_slots.add(event.slot_id)

    rate = successes / activated if activated else None
    blocked_stage = bool(preflight or launch_gate or uncertain or incomplete)
    eligible = (
        not blocked_stage
        and activated == policy.slot_count
        and successes >= policy.promotion_floor
    )
    return CampaignReport(
        campaign_id=manifest.campaign_id,
        stage=stage,
        duration_seconds=policy.duration_seconds,
        planned_slots=policy.slot_count,
        promotion_floor=policy.promotion_floor,
        denominator=activated,
        successes=successes,
        incomplete_slot_ids=incomplete,
        observed_rate=rate,
        wilson_ci=wilson_score_interval(successes, activated),
        preflight_failures=preflight,
        launch_gate_failures=launch_gate,
        uncertain_launches=uncertain,
        activated_failure_counts=activated_failures,
        blocked_slot_ids=tuple(sorted(blocked_slots)),
        stage_blocked=blocked_stage,
        promotion_eligible=eligible,
        replacement_allowed=False,
        support_outside_ui=support,
        failure_taxonomy=taxonomy,
        campaign_chain={"blocked": blocked, "superseded": superseded},
        development_only=manifest.development_only,
        formal_parent_eligible=(manifest.mode == "formal" and not manifest.development_only),
        manifest_hash=manifest_hash,
        prerequisite_parent_hash=manifest.prerequisite_parent_hash,
        event_hash=campaign_event_hash(normalized),
    )


def _unique_texts(values: Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{name} must be a sequence of strings")
    items = tuple(values)
    if any(not isinstance(item, str) or not item.strip() for item in items):
        raise ValueError(f"{name} must contain non-empty strings")
    if len(set(items)) != len(items):
        raise ValueError(f"{name} cannot contain duplicates")
    return tuple(sorted(items))
