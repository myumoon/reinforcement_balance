from __future__ import annotations

import json
from pathlib import Path

import pytest
from reinbalance_survivors_contracts.canonical_json import canonical_hash

from survivors.campaign.campaign_report import (
    STAGE_POLICIES,
    generate_campaign_report,
    wilson_score_interval,
)
from survivors.campaign.campaign_schema import (
    CampaignEvent,
    CampaignManifest,
    EventType,
    campaign_event_hash,
    canonical_event_jsonl,
    campaign_manifest_hash,
)


def _run(slot: int, outcome: EventType = EventType.SUCCESS,
         activation_source: str = "normal") -> list[CampaignEvent]:
    prefix = f"{slot}"
    return [
        CampaignEvent(EventType.FORMAL_SLOT_RESERVED, slot),
        CampaignEvent(EventType.ATTEMPT_PREFLIGHT, slot, attempt_id=f"attempt-{prefix}"),
        CampaignEvent(
            EventType.LAUNCH_INTENT_COMMITTED,
            slot,
            attempt_id=f"attempt-{prefix}",
            reserved_run_id=f"run-{prefix}",
            gameplay_attempt_id=f"gameplay-{prefix}",
            launch_nonce=f"nonce-{prefix}",
        ),
        CampaignEvent(EventType.BROKER_PROCESS_ATTESTED, slot,
                      process_ref=f"process-{prefix}", job_ref=f"job-{prefix}"),
        CampaignEvent(EventType.PROCESS_LAUNCH_CONFIRMED, slot),
        CampaignEvent(EventType.FORMAL_RUN_ACTIVATED, slot,
                      activation_source=activation_source),
        CampaignEvent(outcome, slot,
                      failure_reason=None if outcome is EventType.SUCCESS else "fixture_failure"),
    ]


def _campaign(successes: int) -> list[CampaignEvent]:
    return [
        event
        for slot in range(20)
        for event in _run(
            slot,
            EventType.SUCCESS if slot < successes else EventType.GAMEPLAY_FAILURE,
        )
    ]


def _manifest(stage: str, campaign_id: str = "synthetic-test") -> CampaignManifest:
    policy = STAGE_POLICIES[stage]
    return CampaignManifest(
        campaign_id=campaign_id,
        stage=stage,
        expected_slots=policy.slot_count,
    )


def _report(manifest: CampaignManifest, events, **kwargs):
    return generate_campaign_report(
        manifest,
        events,
        event_manifest_hash=campaign_manifest_hash(manifest),
        **kwargs,
    )


def test_stage_policies_freeze_duration_slots_and_promotion_floors() -> None:
    assert {
        key: (policy.duration_seconds, policy.slot_count, policy.promotion_floor)
        for key, policy in STAGE_POLICIES.items()
    } == {
        "C0": (1800, 2, 2),
        "C1": (3600, 4, 3),
        "C2": (7200, 8, 6),
        "C3": (14400, 16, 12),
        "C4": (28800, 20, 16),
    }


def test_campaign_contract_doc_records_stage_and_denominator_rules() -> None:
    path = Path(__file__).resolve().parents[4] / "docs" / "deployment" / "campaign_contract.md"
    document = path.read_text(encoding="utf-8")
    for term in ("FORMAL_SLOT_RESERVED", "FORMAL_RUN_ACTIVATED", "promotion floor", "Wilson score interval"):
        assert term.casefold() in document.casefold()


def test_six_golden_jsonl_contract_fixtures_are_canonical_and_reproducible() -> None:
    path = Path(__file__).parent / "fixtures" / "golden_campaigns.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    fixtures = [json.loads(line) for line in lines]
    assert {item["fixture_name"] for item in fixtures} == {
        "normal_16_of_20",
        "normal_15_of_20",
        "reconciliation_activation",
        "preflight_failure",
        "uncertain_launch",
        "duplicate_process",
    }
    for line, fixture in zip(lines, fixtures):
        assert json.dumps(fixture, sort_keys=True, separators=(",", ":"), ensure_ascii=False) == line
        manifest = CampaignManifest.from_wire(fixture["manifest"])
        events = [CampaignEvent.from_wire(item) for item in fixture["events"]]
        assert CampaignManifest.from_wire(manifest.to_wire()) == manifest
        if fixture["fixture_name"] == "duplicate_process":
            with pytest.raises(ValueError, match="duplicate process_ref"):
                generate_campaign_report(manifest, events, stage=fixture["stage"])
            continue
        report = generate_campaign_report(manifest, events, stage=fixture["stage"])
        assert report.denominator == fixture["expected"]["denominator"]
        assert report.successes == fixture["expected"]["successes"]
        assert report.stage_blocked is fixture["expected"]["stage_blocked"]
        assert campaign_event_hash(events) == campaign_event_hash(
            [CampaignEvent.from_wire(event.to_wire()) for event in events]
        )
        assert canonical_event_jsonl(events).endswith(b"\n")


def test_sixteen_of_twenty_report_uses_observed_rate_and_wilson_interval() -> None:
    report = generate_campaign_report(
        CampaignManifest(campaign_id="synthetic-16"), _campaign(16), stage="C4"
    )
    assert report.denominator == 20
    assert report.successes == 16
    assert report.observed_rate == pytest.approx(0.8)
    assert report.wilson_ci == pytest.approx((0.583, 0.919), abs=0.002)
    assert report.promotion_eligible is True
    assert report.development_only is True
    assert report.formal_parent_eligible is False
    assert "population_success_probability" not in report.to_wire()
    assert report.report_hash == canonical_hash(report.to_wire())


def test_fifteen_of_twenty_does_not_meet_the_frozen_c4_floor() -> None:
    report = generate_campaign_report(
        CampaignManifest(campaign_id="synthetic-15"), _campaign(15), stage="C4"
    )
    assert report.denominator == 20
    assert report.observed_rate == pytest.approx(0.75)
    assert report.promotion_eligible is False


@pytest.mark.parametrize(
    "failure_type",
    [EventType.PREFLIGHT_FAILED, EventType.LAUNCH_GATE_FAILED],
)
def test_pre_activation_failures_are_excluded_and_block_the_stage(failure_type: EventType) -> None:
    events = [CampaignEvent(EventType.FORMAL_SLOT_RESERVED, 0)]
    events.append(CampaignEvent(EventType.ATTEMPT_PREFLIGHT, 0, attempt_id="a0"))
    if failure_type is EventType.PREFLIGHT_FAILED:
        events.append(CampaignEvent(failure_type, 0, attempt_id="a0", failure_reason="stale_build"))
    else:
        events.append(CampaignEvent(failure_type, 0, failure_reason="gate_closed"))
    events.extend(event for slot in range(1, 20) for event in _run(slot))
    report = generate_campaign_report(
        CampaignManifest(campaign_id="synthetic-gate"), events, stage="C4"
    )
    assert report.denominator == 19
    assert report.preflight_failures == (1 if failure_type is EventType.PREFLIGHT_FAILED else 0)
    assert report.launch_gate_failures == (1 if failure_type is EventType.LAUNCH_GATE_FAILED else 0)
    assert report.stage_blocked is True
    assert report.promotion_eligible is False
    assert report.blocked_slot_ids == (0,)


def test_activated_failure_stays_in_denominator_and_cannot_be_replaced() -> None:
    events = _run(0, EventType.SAFETY_FAILURE)
    report = generate_campaign_report(
        CampaignManifest(campaign_id="synthetic-safety"), events, stage="C0"
    )
    assert report.denominator == 1
    assert report.successes == 0
    assert report.activated_failure_counts["SAFETY_FAILURE"] == 1
    assert report.replacement_allowed is False


def test_uncertain_launch_is_a_blocked_pre_activation_slot() -> None:
    events = [
        CampaignEvent(EventType.FORMAL_SLOT_RESERVED, 0),
        CampaignEvent(EventType.ATTEMPT_PREFLIGHT, 0, attempt_id="a0"),
        CampaignEvent(EventType.LAUNCH_INTENT_COMMITTED, 0, attempt_id="a0",
                      reserved_run_id="r0", gameplay_attempt_id="g0", launch_nonce="n0"),
        CampaignEvent(EventType.LAUNCH_UNCERTAIN, 0, failure_reason="observer_timeout"),
    ]
    report = generate_campaign_report(
        CampaignManifest(campaign_id="synthetic-uncertain"), events, stage="C0"
    )
    assert report.denominator == 0
    assert report.uncertain_launches == 1
    assert report.stage_blocked is True
    assert report.replacement_allowed is False


def test_report_includes_unsupported_ui_failure_taxonomy_and_campaign_chain() -> None:
    events = _run(0, EventType.ARTIFACT_FAILURE)
    report = generate_campaign_report(
        CampaignManifest(campaign_id="synthetic-chain"),
        events,
        stage="C0",
        support_outside_ui=("pause_overlay",),
        blocked_campaign_ids=("campaign-old",),
        superseded_campaign_ids=("campaign-prev",),
    )
    assert report.support_outside_ui == ("pause_overlay",)
    assert report.failure_taxonomy["artifact_failure:fixture_failure"] == 1
    assert report.campaign_chain == {
        "blocked": ("campaign-old",),
        "superseded": ("campaign-prev",),
    }


def test_two_sided_wilson_interval_handles_empty_and_extreme_samples() -> None:
    assert wilson_score_interval(0, 0) is None
    assert wilson_score_interval(0, 20)[0] == 0.0
    assert wilson_score_interval(20, 20)[1] == 1.0
    with pytest.raises(ValueError):
        wilson_score_interval(21, 20)


@pytest.mark.parametrize(
    "events,incomplete",
    [
        (_run(0)[:-1] + _run(1), (0,)),
        ([CampaignEvent(EventType.FORMAL_SLOT_RESERVED, 0)] + _run(1), (0,)),
        (
            [CampaignEvent(EventType.FORMAL_SLOT_RESERVED, 0),
             CampaignEvent(EventType.ATTEMPT_PREFLIGHT, 0, attempt_id="pending")] + _run(1),
            (0,),
        ),
        (_run(0), (1,)),
    ],
    ids=["activated-no-terminal", "reserved-only", "preflight-in-progress", "missing-slot"],
)
def test_incomplete_slots_block_stage_and_prevent_promotion(events, incomplete) -> None:
    report = _report(_manifest("C0"), events)
    assert report.incomplete_slot_ids == incomplete
    assert report.stage_blocked is True
    assert report.promotion_eligible is False


def test_manifest_stage_and_event_binding_are_enforced_and_reported() -> None:
    manifest = _manifest("C0", "bound-campaign")
    events = _run(0) + _run(1)
    report = _report(manifest, events)
    assert report.stage == "C0"
    assert report.manifest_hash == campaign_manifest_hash(manifest)
    assert report.prerequisite_parent_hash is None
    with pytest.raises(ValueError, match="manifest hash"):
        generate_campaign_report(
            manifest,
            events,
            event_manifest_hash="0" * 64,
        )
    with pytest.raises(ValueError, match="does not match manifest stage"):
        generate_campaign_report(
            manifest,
            events,
            event_manifest_hash=campaign_manifest_hash(manifest),
            stage="C4",
        )
