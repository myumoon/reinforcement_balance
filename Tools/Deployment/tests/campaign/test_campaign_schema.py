from __future__ import annotations

from dataclasses import replace

import pytest

from survivors.campaign.campaign_schema import (
    CampaignEvent,
    CampaignManifest,
    EventType,
    REQUIRED_PREREQUISITES,
    campaign_event_hash,
    campaign_jsonl_hash,
    campaign_manifest_hash,
    canonical_event_jsonl,
    validate_campaign_events,
    validate_prerequisites,
)


def _hash(char: str) -> str:
    return char * 64


def _prerequisite_wire(*, parent: str = "b", development_only: bool = False) -> dict[str, object]:
    names = REQUIRED_PREREQUISITES
    return {
        "hashes": {name: _hash("a") for name in names},
        "parents": {name: _hash(parent) for name in names},
        "statuses": {name: "PASS" for name in names},
        "cloud_sync_status": "verified",
        "backup_hash": _hash("c"),
        "pre_save_contract_hash": _hash("d"),
        "post_save_contract_hash": _hash("e"),
        "development_only": development_only,
    }


def _events(*, slot: int = 0, terminal: EventType = EventType.SUCCESS,
            activation_source: str = "normal", attempt: str = "a0",
            run: str = "r0", gameplay: str = "g0", process: str = "p0") -> list[CampaignEvent]:
    return [
        CampaignEvent(EventType.FORMAL_SLOT_RESERVED, slot),
        CampaignEvent(EventType.ATTEMPT_PREFLIGHT, slot, attempt_id=attempt),
        CampaignEvent(
            EventType.LAUNCH_INTENT_COMMITTED,
            slot,
            attempt_id=attempt,
            reserved_run_id=run,
            gameplay_attempt_id=gameplay,
            launch_nonce=f"n{slot}",
        ),
        CampaignEvent(EventType.BROKER_PROCESS_ATTESTED, slot, process_ref=process,
                      job_ref=f"j{slot}"),
        CampaignEvent(EventType.PROCESS_LAUNCH_CONFIRMED, slot),
        CampaignEvent(EventType.FORMAL_RUN_ACTIVATED, slot,
                      activation_source=activation_source),
        CampaignEvent(terminal, slot,
                      failure_reason=None if terminal is EventType.SUCCESS else "fixture_failure"),
    ]


@pytest.mark.parametrize(
    "events",
    [
        [CampaignEvent(EventType.FORMAL_SLOT_RESERVED, 0)],
        [CampaignEvent(EventType.FORMAL_SLOT_RESERVED, 0),
         CampaignEvent(EventType.ATTEMPT_PREFLIGHT, 0, attempt_id="a0"),
         CampaignEvent(EventType.PREFLIGHT_FAILED, 0, attempt_id="a0", failure_reason="missing_target")],
        _events(activation_source="reconciliation"),
    ],
    ids=["reserved-without-attempt", "attempt-without-reserved-run", "full-activation"],
)
def test_valid_campaign_event_cardinalities(events: list[CampaignEvent]) -> None:
    validate_campaign_events(events, expected_slots=20)


@pytest.mark.parametrize(
    "events",
    [
        _events() + [CampaignEvent(EventType.FORMAL_RUN_ACTIVATED, 0,
                                   activation_source="normal")],
        _events()[:-1] + [CampaignEvent(EventType.FORMAL_RUN_ACTIVATED, 0,
                                        activation_source="normal")],
        [CampaignEvent(EventType.FORMAL_RUN_ACTIVATED, 0,
                       activation_source="normal")],
        [CampaignEvent(EventType.FORMAL_SLOT_RESERVED, 0),
         CampaignEvent(EventType.FORMAL_SLOT_RESERVED, 0)],
    ],
    ids=["terminal-overwrite", "second-activation", "activation-without-identity", "slot-reuse"],
)
def test_cardinality_violations_are_rejected(events: list[CampaignEvent]) -> None:
    with pytest.raises(ValueError):
        validate_campaign_events(events, expected_slots=20)


def test_duplicate_attempt_gameplay_and_process_ids_are_rejected() -> None:
    duplicate_attempt = _events(slot=0) + _events(slot=1, attempt="a0")
    duplicate_gameplay = _events(slot=0) + _events(slot=1, gameplay="g0")
    duplicate_process = _events(slot=0) + _events(slot=1, process="p0")
    for events in (duplicate_attempt, duplicate_gameplay, duplicate_process):
        with pytest.raises(ValueError):
            validate_campaign_events(events, expected_slots=20)


def test_second_gameplay_attempt_cannot_replace_an_activated_run() -> None:
    events = _events()[:-1] + [
        CampaignEvent(
            EventType.LAUNCH_INTENT_COMMITTED,
            0,
            attempt_id="a0",
            reserved_run_id="r1",
            gameplay_attempt_id="g1",
            launch_nonce="n1",
        )
    ]
    with pytest.raises(ValueError):
        validate_campaign_events(events, expected_slots=20)


def test_preflight_failure_cannot_be_retried_or_replaced() -> None:
    events = [
        CampaignEvent(EventType.FORMAL_SLOT_RESERVED, 0),
        CampaignEvent(EventType.ATTEMPT_PREFLIGHT, 0, attempt_id="a0"),
        CampaignEvent(EventType.PREFLIGHT_FAILED, 0, attempt_id="a0", failure_reason="stale_build"),
        CampaignEvent(EventType.ATTEMPT_PREFLIGHT, 0, attempt_id="a1"),
    ]
    with pytest.raises(ValueError):
        validate_campaign_events(events, expected_slots=20)


def test_campaign_manifest_round_trip_and_canonical_hash() -> None:
    manifest = CampaignManifest(campaign_id="synthetic-01")
    decoded = CampaignManifest.from_wire(manifest.to_wire())
    assert decoded == manifest
    assert campaign_manifest_hash(decoded) == campaign_manifest_hash(manifest)
    events = _events()
    round_tripped = [CampaignEvent.from_wire(event.to_wire()) for event in events]
    assert campaign_event_hash(round_tripped) == campaign_event_hash(events)
    assert campaign_jsonl_hash(round_tripped) == campaign_jsonl_hash(events)
    assert canonical_event_jsonl(events).endswith(b"\n")


@pytest.mark.parametrize("field", ["seed", "same_seed", "independent", "statistical_independence"])
def test_manifest_rejects_seed_and_independence_claims(field: str) -> None:
    wire = CampaignManifest(campaign_id="synthetic-01").to_wire()
    wire[field] = True
    with pytest.raises(ValueError):
        CampaignManifest.from_wire(wire)


def test_manifest_requires_separate_rng_and_trial_fields() -> None:
    wire = CampaignManifest(campaign_id="synthetic-01").to_wire()
    del wire["trial_separation"]
    with pytest.raises(ValueError):
        CampaignManifest.from_wire(wire)
    with pytest.raises(ValueError):
        replace(CampaignManifest(campaign_id="synthetic-01"), rng_control="seeded")


def test_manifest_rejects_unknown_wire_fields() -> None:
    wire = CampaignManifest(campaign_id="synthetic-01").to_wire()
    wire["legacy_issuance_id"] = "old"
    with pytest.raises(ValueError):
        CampaignManifest.from_wire(wire)


@pytest.mark.parametrize(
    "change",
    [
        lambda wire: wire["hashes"].pop("exact_runtime"),
        lambda wire: wire["statuses"].update({"target": "STALE"}),
        lambda wire: wire["parents"].update({"target": _hash("f")}),
        lambda wire: wire.update({"cloud_sync_status": "unknown"}),
        lambda wire: wire.update({"backup_hash": None}),
        lambda wire: wire.update({"pre_save_contract_hash": None}),
        lambda wire: wire.update({"post_save_contract_hash": None}),
        lambda wire: wire.update({"development_only": True}),
    ],
    ids=["missing-runtime", "stale-target", "mixed-parent", "unknown-cloud-sync",
         "missing-backup", "missing-pre-save", "missing-post-save", "development-fixture"],
)
def test_prerequisite_validator_fails_closed(change) -> None:
    wire = _prerequisite_wire()
    change(wire)
    with pytest.raises(ValueError):
        validate_prerequisites(wire, expected_parent_hash=_hash("b"))


def test_prerequisite_validator_rejects_stale_parent_and_unknown_field() -> None:
    with pytest.raises(ValueError, match="stale"):
        validate_prerequisites(_prerequisite_wire(), expected_parent_hash=_hash("f"))
    wire = _prerequisite_wire()
    wire["legacy_issuance"] = True
    with pytest.raises(ValueError, match="unknown"):
        validate_prerequisites(wire, expected_parent_hash=_hash("b"))


def test_formal_manifest_requires_verified_non_development_prerequisites() -> None:
    good = validate_prerequisites(_prerequisite_wire(), expected_parent_hash=_hash("b"))
    formal = CampaignManifest(
        campaign_id="formal-01", mode="formal", development_only=False,
        prerequisites=good, prerequisite_parent_hash=_hash("b"),
    )
    assert CampaignManifest.from_wire(formal.to_wire()) == formal
    with pytest.raises(ValueError):
        CampaignManifest(campaign_id="formal-02", mode="formal", development_only=False)
    with pytest.raises(ValueError):
        validate_prerequisites(
            _prerequisite_wire(development_only=True), expected_parent_hash=_hash("b")
        )


def test_event_wire_rejects_unknown_required_and_claim_fields() -> None:
    wire = CampaignEvent(EventType.FORMAL_SLOT_RESERVED, 0).to_wire()
    wire["legacy_issuance_field"] = "old"
    with pytest.raises(ValueError):
        CampaignEvent.from_wire(wire)


@pytest.mark.parametrize(
    "claim",
    [
        {"same-seed": True},
        {"rng_seed": 123},
        {"realSeed": 123},
        {"is_independent": True},
        {"metadata": {"same seed": True}},
    ],
    ids=["hyphen", "snake", "camel", "independence", "nested-spaces"],
)
def test_event_details_reject_normalized_and_nested_claim_fields(claim) -> None:
    with pytest.raises(ValueError, match="statistical claim"):
        CampaignEvent(
            EventType.FORMAL_SLOT_RESERVED,
            0,
            details=claim,
        )


@pytest.mark.parametrize(
    "field,identity",
    [
        ("reserved_run_id", "r0"),
        ("launch_nonce", "n0"),
        ("job_ref", "j0"),
    ],
)
def test_all_identity_kinds_reject_duplicates(field, identity) -> None:
    first = _events(slot=0)
    second = _events(slot=1)
    index = {"reserved_run_id": 2, "launch_nonce": 2, "job_ref": 3}[field]
    first[index] = replace(first[index], **{field: identity})
    second[index] = replace(second[index], **{field: identity})
    with pytest.raises(ValueError, match=f"duplicate {field}"):
        validate_campaign_events(first + second, expected_slots=20)
    wire = {
        "event_type": EventType.FORMAL_SLOT_RESERVED.value,
        "slot_id": 0,
        "details": {"independent": True},
    }
    with pytest.raises(ValueError):
        CampaignEvent.from_wire(wire)
