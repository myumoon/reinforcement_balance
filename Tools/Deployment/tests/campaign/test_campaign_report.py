"""campaign report の分母と stage 集計を検証します。

golden fixture と synthetic run で再現可能な結果を固定します。
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from reinbalance_survivors_contracts.canonical_json import canonical_hash, canonical_json_bytes

from survivors.campaign.campaign_report import (
    STAGE_POLICIES,
    generate_campaign_report,
    wilson_score_interval,
)
from survivors.campaign.campaign_schema import (
    CampaignEvent,
    CampaignManifest,
    EventType,
    REQUIRED_PREREQUISITES,
    campaign_event_hash,
    campaign_jsonl_hash,
    campaign_manifest_hash,
    canonical_event_jsonl,
)


def _run(slot: int, outcome: EventType = EventType.SUCCESS,
         activation_source: str = "normal") -> list[CampaignEvent]:
    """一 slot 分の activated run event を作ります。

    terminal outcome と activation source を切り替えられます。
    """
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
    """20-slot synthetic campaign event 列を作ります。

    先頭の指定数を success、残りを gameplay failure にします。
    """
    return [
        event
        for slot in range(20)
        for event in _run(
            slot,
            EventType.SUCCESS if slot < successes else EventType.GAMEPLAY_FAILURE,
        )
    ]


def _manifest(stage: str, campaign_id: str = "synthetic-test") -> CampaignManifest:
    """指定 stage に合う synthetic manifest を作ります。

    expected slot 数は固定 stage policy から取得します。
    """
    policy = STAGE_POLICIES[stage]
    return CampaignManifest(
        campaign_id=campaign_id,
        stage=stage,
        expected_slots=policy.slot_count,
    )


def _report(manifest: CampaignManifest, events, **kwargs):
    """event 列を manifest hash に束縛して report します。

    test ごとに同じ report binding を適用します。
    """
    manifest_hash = campaign_manifest_hash(manifest)
    return generate_campaign_report(
        manifest,
        [replace(event, campaign_manifest_hash=manifest_hash) for event in events],
        event_manifest_hash=manifest_hash,
        **kwargs,
    )


def test_stage_policies_freeze_duration_slots_and_promotion_floors() -> None:
    """C0 から C4 の duration、slot、floor を固定します。

    policy 値の変更が test で検出されることを確認します。
    """
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
    """contract 文書に stage と denominator 用語があることを確認します。

    実装と運用文書の主要語彙を同期させます。
    """
    path = Path(__file__).resolve().parents[4] / "docs" / "deployment" / "campaign_contract.md"
    document = path.read_text(encoding="utf-8")
    for term in ("FORMAL_SLOT_RESERVED", "FORMAL_RUN_ACTIVATED", "promotion floor", "Wilson score interval"):
        assert term.casefold() in document.casefold()


def test_six_golden_jsonl_contract_fixtures_are_canonical_and_reproducible() -> None:
    """六つの golden fixture の wire と pinned hash を検証します。

    event、JSONL、manifest、report の各 digest を照合します。
    """
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
        assert canonical_json_bytes(fixture) == line.encode("utf-8")
        manifest = CampaignManifest.from_wire(fixture["manifest"])
        events = [CampaignEvent.from_wire(item) for item in fixture["events"]]
        assert CampaignManifest.from_wire(manifest.to_wire()) == manifest
        assert [item.to_wire() for item in events] == fixture["events"]
        expected_hashes = fixture["expected_hashes"]
        assert fixture["event_manifest_hash"] == expected_hashes["manifest_hash"]
        assert campaign_manifest_hash(manifest) == expected_hashes["manifest_hash"]
        assert campaign_event_hash(events) == expected_hashes["event_hash"]
        assert campaign_jsonl_hash(events) == expected_hashes["jsonl_hash"]
        if fixture["fixture_name"] == "duplicate_process":
            with pytest.raises(ValueError, match="duplicate process_ref"):
                generate_campaign_report(
                    manifest,
                    events,
                    event_manifest_hash=fixture["event_manifest_hash"],
                )
            assert expected_hashes["report_hash"] is None
            continue
        report = generate_campaign_report(
            manifest,
            events,
            event_manifest_hash=fixture["event_manifest_hash"],
        )
        assert report.denominator == fixture["expected"]["denominator"]
        assert report.successes == fixture["expected"]["successes"]
        assert report.stage_blocked is fixture["expected"]["stage_blocked"]
        assert report.report_hash == expected_hashes["report_hash"]
        assert canonical_event_jsonl(events).endswith(b"\n")


def test_sixteen_of_twenty_report_uses_observed_rate_and_wilson_interval() -> None:
    """16/20 report が観測 rate と Wilson interval を出すことを確認します。

    synthetic report に population probability claim が無いことも確認します。
    """
    report = _report(_manifest("C4", "synthetic-16"), _campaign(16))
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
    """15/20 が C4 promotion floor に届かないことを確認します。

    observed rate と promotion 判定を別々に検証します。
    """
    report = _report(_manifest("C4", "synthetic-15"), _campaign(15))
    assert report.denominator == 20
    assert report.observed_rate == pytest.approx(0.75)
    assert report.promotion_eligible is False


@pytest.mark.parametrize(
    "failure_type",
    [EventType.PREFLIGHT_FAILED, EventType.LAUNCH_GATE_FAILED],
)
def test_pre_activation_failures_are_excluded_and_block_the_stage(failure_type: EventType) -> None:
    """pre-activation failure を分母外にして stage を block します。

    preflight と launch gate の両経路に同じ規則を適用します。
    """
    events = [CampaignEvent(EventType.FORMAL_SLOT_RESERVED, 0)]
    events.append(CampaignEvent(EventType.ATTEMPT_PREFLIGHT, 0, attempt_id="a0"))
    if failure_type is EventType.PREFLIGHT_FAILED:
        events.append(CampaignEvent(failure_type, 0, attempt_id="a0", failure_reason="stale_build"))
    else:
        events.append(CampaignEvent(failure_type, 0, failure_reason="gate_closed"))
    events.extend(event for slot in range(1, 20) for event in _run(slot))
    report = _report(_manifest("C4", "synthetic-gate"), events)
    assert report.denominator == 19
    assert report.preflight_failures == (1 if failure_type is EventType.PREFLIGHT_FAILED else 0)
    assert report.launch_gate_failures == (1 if failure_type is EventType.LAUNCH_GATE_FAILED else 0)
    assert report.stage_blocked is True
    assert report.promotion_eligible is False
    assert report.blocked_slot_ids == (0,)


def test_activated_failure_stays_in_denominator_and_cannot_be_replaced() -> None:
    """activated failure を denominator に残して replacement を拒否します。

    safety failure の分類と slot identity を保持します。
    """
    events = _run(0, EventType.SAFETY_FAILURE)
    report = _report(_manifest("C0", "synthetic-safety"), events)
    assert report.denominator == 1
    assert report.successes == 0
    assert report.activated_failure_counts["SAFETY_FAILURE"] == 1
    assert report.replacement_allowed is False


def test_uncertain_launch_is_a_blocked_pre_activation_slot() -> None:
    """uncertain launch を分母外の blocked slot として扱います。

    activation 前で止まった slot に replacement を許しません。
    """
    events = [
        CampaignEvent(EventType.FORMAL_SLOT_RESERVED, 0),
        CampaignEvent(EventType.ATTEMPT_PREFLIGHT, 0, attempt_id="a0"),
        CampaignEvent(EventType.LAUNCH_INTENT_COMMITTED, 0, attempt_id="a0",
                      reserved_run_id="r0", gameplay_attempt_id="g0", launch_nonce="n0"),
        CampaignEvent(EventType.LAUNCH_UNCERTAIN, 0, failure_reason="observer_timeout"),
    ]
    report = _report(_manifest("C0", "synthetic-uncertain"), events)
    assert report.denominator == 0
    assert report.uncertain_launches == 1
    assert report.stage_blocked is True
    assert report.replacement_allowed is False


def test_report_includes_unsupported_ui_failure_taxonomy_and_campaign_chain() -> None:
    """UI support、failure taxonomy、campaign chain を report します。

    artifact failure と blocked/superseded id を wire に残します。
    """
    events = _run(0, EventType.ARTIFACT_FAILURE)
    report = _report(
        _manifest("C0", "synthetic-chain"),
        events,
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
    """Wilson interval の空標本と極端な標本を検証します。

    invalid successes 数は入力境界で拒否されます。
    """
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
    """未完了 slot を列挙して stage promotion を止めます。

    activation 後、reserved、preflight、中抜けの各状態を確認します。
    """
    report = _report(_manifest("C0"), events)
    assert report.incomplete_slot_ids == incomplete
    assert report.stage_blocked is True
    assert report.promotion_eligible is False


def test_manifest_stage_and_event_binding_are_enforced_and_reported() -> None:
    """manifest stage と event hash binding を検証します。

    report wire に parent identity を含めることも確認します。
    """
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


def test_synthetic_golden_events_cannot_be_relabelled_as_formal_parent() -> None:
    """synthetic golden event を formal manifest に付け替えられません。

    保存された event manifest hash と formal manifest hash を照合します。
    """
    prerequisites_wire = {
        "hashes": {name: "a" * 64 for name in REQUIRED_PREREQUISITES},
        "parents": {name: "b" * 64 for name in REQUIRED_PREREQUISITES},
        "statuses": {name: "PASS" for name in REQUIRED_PREREQUISITES},
        "cloud_sync_status": "verified",
        "backup_hash": "c" * 64,
        "pre_save_contract_hash": "d" * 64,
        "post_save_contract_hash": "e" * 64,
        "development_only": False,
    }
    from survivors.campaign.campaign_schema import validate_prerequisites

    formal = CampaignManifest(
        campaign_id="formal-fake",
        mode="formal",
        stage="C0",
        expected_slots=2,
        development_only=False,
        prerequisites=validate_prerequisites(prerequisites_wire, expected_parent_hash="b" * 64),
        prerequisite_parent_hash="b" * 64,
    )
    golden_path = Path(__file__).parent / "fixtures" / "golden_campaigns.jsonl"
    synthetic = next(
        fixture
        for fixture in (json.loads(line) for line in golden_path.read_text(encoding="utf-8").splitlines())
        if fixture["fixture_name"] == "reconciliation_activation"
    )
    events = [CampaignEvent.from_wire(item) for item in synthetic["events"]]
    with pytest.raises(ValueError, match="manifest hash"):
        generate_campaign_report(
            formal,
            events,
            event_manifest_hash=synthetic["event_manifest_hash"],
        )


def test_synthetic_events_cannot_be_rebound_to_formal_manifest() -> None:
    prerequisites = {
        "hashes": {name: "a" * 64 for name in REQUIRED_PREREQUISITES},
        "parents": {name: "b" * 64 for name in REQUIRED_PREREQUISITES},
        "statuses": {name: "PASS" for name in REQUIRED_PREREQUISITES},
        "cloud_sync_status": "verified",
        "backup_hash": "c" * 64,
        "pre_save_contract_hash": "d" * 64,
        "post_save_contract_hash": "e" * 64,
        "development_only": False,
    }
    from survivors.campaign.campaign_schema import validate_prerequisites

    formal = CampaignManifest(
        campaign_id="formal-c4",
        mode="formal",
        stage="C4",
        expected_slots=20,
        development_only=False,
        prerequisites=validate_prerequisites(prerequisites, expected_parent_hash="b" * 64),
        prerequisite_parent_hash="b" * 64,
    )
    golden_path = Path(__file__).parent / "fixtures" / "golden_campaigns.jsonl"
    fixture = next(
        row
        for row in (json.loads(line) for line in golden_path.read_text(encoding="utf-8").splitlines())
        if row["fixture_name"] == "normal_16_of_20"
    )
    events = [CampaignEvent.from_wire(item) for item in fixture["events"]]

    with pytest.raises(ValueError, match="event.*manifest"):
        generate_campaign_report(
            formal,
            events,
            event_manifest_hash=campaign_manifest_hash(formal),
        )
