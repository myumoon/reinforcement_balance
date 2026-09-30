"""Survivors goal evidence builder の受け入れ条件を検証します。

synthetic campaign fixture を使い、再計算・失敗条件・復元経路を固定します。
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
from reinbalance_survivors_contracts.canonical_json import canonical_hash, canonical_json_bytes, sha256_hex

from survivors.campaign.campaign_runner import ArtifactStore
from survivors.campaign.campaign_schema import (
    CAMPAIGN_SCHEMA_VERSION,
    CampaignEvent,
    CampaignManifest,
    EventType,
    STAGE_POLICIES,
    campaign_manifest_hash,
)
from survivors.campaign.durable_launch_store import (
    Classification,
    LaunchHistory,
    LaunchIntent,
    LaunchStage,
    ProcessAttestation,
    ProcessIdentity,
    StorageVerdict,
)
from survivors.release.evidence_builder import (
    EvidenceError,
    build_goal_evidence,
    verify_evidence_bundle,
    write_evidence_bundle,
)


BROKER_SCHEMA = "survivors.launch_broker.v1"
CHAIN_SCHEMA = "survivors.goal_release_chain.v1"
CHAIN_NAMES = (
    "target_profile", "game_build", "combat_model", "vecnormalize", "deploy_schema",
    "error_profile", "selector", "parser", "detector", "controller", "training",
    "teacher", "dataset", "evaluation",
)


@dataclass
class FakeLedger:
    """耐久ledger APIを模すテスト用の記録集合です。

    campaignごとのattempt履歴とstorage verdictをメモリ上で返します。
    """

    histories: dict[str, LaunchHistory]
    events: dict[str, tuple[CampaignEvent, ...]]
    directory: Path
    verdict: StorageVerdict
    pragmas: dict[str, str]

    def attempt_ids(self) -> tuple[str, ...]:
        """登録されたattempt idを返します。

        evidence builder がledger全体を走査できるようにします。
        """
        return tuple(self.histories)

    def history(self, attempt_id: str) -> LaunchHistory:
        """指定attemptのlifecycle履歴を返します。

        fixtureが登録した完全な履歴をそのまま公開します。
        """
        return self.histories[attempt_id]

    def campaign_events(self, attempt_id: str) -> tuple[CampaignEvent, ...]:
        """指定attemptに記録されたcampaign eventを返します。

        event列は実ledger consumerと同じ読み出し形にそろえます。
        """
        return self.events[attempt_id]


def _digest(text: str) -> str:
    """fixture値から安定したSHA-256を作ります。

    各テストで同じ入力が同じ参照値になるようにします。
    """
    return hashlib.sha256(text.encode()).hexdigest()


def _chain(build_hash: str) -> dict:
    """テスト用release chainを生成します。

    全nodeと親hashを正規hashで結び、buildの混在を検証可能にします。
    """
    nodes = {}
    contents = {
        "target_profile": {"profile_sha256": _digest("target-profile")},
        "game_build": {"build_id": "mad-forest-build-1", "build_sha256": build_hash},
    }
    for name in CHAIN_NAMES[2:10]:
        contents[name] = {"artifact_sha256": _digest(name)}
    contents.update({name: {"artifact_sha256": _digest(name), "verdict": "PASS"}
                     for name in CHAIN_NAMES[10:]})

    parents = {
        "target_profile": [],
        "game_build": [],
        **{name: ["target_profile", "game_build"] for name in CHAIN_NAMES[2:10]},
        "training": ["combat_model", "vecnormalize", "target_profile", "game_build"],
        "teacher": ["training", "target_profile", "game_build"],
        "dataset": ["teacher", "target_profile", "game_build"],
        "evaluation": ["training", "teacher", "dataset", "target_profile", "game_build"],
    }
    for name in CHAIN_NAMES:
        parent_hashes = [nodes[parent]["sha256"] for parent in parents[name]]
        body = {
            "schema_version": f"survivors.{name}.v1",
            "content": contents[name],
            "parents": parent_hashes,
        }
        nodes[name] = {**body, "sha256": canonical_hash(body)}
    body = {
        "schema_version": CHAIN_SCHEMA,
        "target_profile_sha256": nodes["target_profile"]["sha256"],
        "game_build_sha256": nodes["game_build"]["sha256"],
        "nodes": nodes,
    }
    return {**body, "chain_sha256": canonical_hash(body)}


def _put_stage(
    artifacts: ArtifactStore,
    *,
    execution_id: str,
    plan_hash: str,
    outcomes: tuple[CampaignEvent, ...],
    histories: dict[str, LaunchHistory],
    ledger_events: dict[str, tuple[CampaignEvent, ...]],
    backup_sha256: str,
    build_hash: str,
    stage: str = "C4",
    omit_rng_control: bool = False,
) -> dict:
    """指定stageのmanifest、run、ledger fixtureを保存します。

    受け取ったoutcomeごとにactivationとartifactを組み立てます。
    """
    manifest = CampaignManifest(
        campaign_id=execution_id,
        schema_version=CAMPAIGN_SCHEMA_VERSION,
        mode="synthetic",
        stage=stage,
        expected_slots=STAGE_POLICIES[stage].slot_count,
        development_only=True,
    )
    manifest_wire = manifest.to_wire()
    if omit_rng_control:
        del manifest_wire["rng_control"]
    manifest_hash = campaign_manifest_hash(manifest)
    artifacts.put_json(
        f"stages/{execution_id}/manifest.json",
        {
            "manifest": manifest_wire,
            "plan_hash": plan_hash,
            "parent_execution_id": None,
            "blocked_execution_ids": [],
            "campaign_run_mode": "formal_single_attempt",
            "development_only": True,
            "formal_campaign_eligible": False,
        },
    )
    run_records = []
    preflight_records = []
    outcome_records = []
    gate_records = []
    for event in outcomes:
        slot = event.slot_id
        attempt_id = f"attempt-{execution_id}-{slot:02d}"
        reserved_run_id = f"reserved-{execution_id}-{slot:02d}"
        gameplay_attempt_id = f"gameplay-{execution_id}-{slot:02d}"
        nonce = _digest(f"nonce-{execution_id}-{slot:02d}")
        pid = 5000 + slot + (100 if execution_id.endswith("x2") else 0)
        identity = ProcessIdentity(pid, 1_330_000_000_000_000 + pid, sys.executable, 1, pid)
        attestation = ProcessAttestation(identity, f"job-{pid}", f"broker-{execution_id}")
        intent = LaunchIntent(
            campaign_manifest_hash=manifest_hash,
            slot_id=slot,
            attempt_id=attempt_id,
            reserved_run_id=reserved_run_id,
            gameplay_attempt_id=gameplay_attempt_id,
            launch_nonce=nonce,
            executable_path=str(Path(sys.executable).resolve()),
            executable_hash=_digest("executable"),
            build_hash=build_hash,
            config_hash=_digest("config"),
            argv=(str(Path(sys.executable).resolve()), "-c", "pass"),
        )
        history = LaunchHistory(
            attempt_id,
            intent,
            (LaunchStage.LAUNCH_INTENT, LaunchStage.PROCESS_ATTESTED,
             LaunchStage.RESUME_INTENT, LaunchStage.PROCESS_LAUNCH_CONFIRMED,
             LaunchStage.FORMAL_RUN_ACTIVATED),
            tuple(_digest(f"row-{execution_id}-{slot}-{index}") for index in range(1, 6)),
            attestation,
            "normal",
        )
        histories[attempt_id] = history
        ledger_events[attempt_id] = (
            CampaignEvent(EventType.LAUNCH_INTENT_COMMITTED, slot, attempt_id,
                          reserved_run_id, gameplay_attempt_id, nonce,
                          campaign_manifest_hash=manifest_hash),
            CampaignEvent(EventType.BROKER_PROCESS_ATTESTED, slot,
                          process_ref=identity.process_ref, job_ref=attestation.job_ref,
                          campaign_manifest_hash=manifest_hash),
            CampaignEvent(EventType.PROCESS_LAUNCH_CONFIRMED, slot,
                          campaign_manifest_hash=manifest_hash),
            CampaignEvent(EventType.FORMAL_RUN_ACTIVATED, slot, activation_source="normal",
                          campaign_manifest_hash=manifest_hash),
        )
        preflight_records.extend((
            {"stage_execution_id": execution_id,
             "event": CampaignEvent(EventType.FORMAL_SLOT_RESERVED, slot,
                                     campaign_manifest_hash=manifest_hash).to_wire()},
            {"stage_execution_id": execution_id,
             "event": CampaignEvent(EventType.ATTEMPT_PREFLIGHT, slot, attempt_id=attempt_id,
                                     details={"launch_nonce": nonce},
                                     campaign_manifest_hash=manifest_hash).to_wire()},
        ))
        pre_hash, post_hash = _digest(f"pre-{attempt_id}"), _digest(f"post-{attempt_id}")
        artifacts.put_json(f"save/pre/{attempt_id}.json", {"attempt_id": attempt_id, "sha256": pre_hash})
        artifacts.put_json(f"save/post/{attempt_id}.json", {"attempt_id": attempt_id, "sha256": post_hash})
        name = f"stages/{execution_id}/runs/slot-{slot:02d}.json"
        artifacts.put_json(name, {
            "campaign_id": "fixture-c4",
            "plan_hash": plan_hash,
            "stage": stage,
            "stage_execution_id": execution_id,
            "campaign_manifest_hash": manifest_hash,
            "slot_id": slot,
            "attempt_id": attempt_id,
            "reserved_run_id": reserved_run_id,
            "gameplay_attempt_id": gameplay_attempt_id,
            "campaign_run_mode": "formal_single_attempt",
            "activation_source": "normal",
            "run_spec": {
                "campaign_id": "fixture-c4", "stage_execution_id": execution_id,
                "stage": stage, "slot_id": slot, "reserved_run_id": reserved_run_id,
                "gameplay_attempt_id": gameplay_attempt_id,
                "campaign_run_mode": "formal_single_attempt",
                "process_ref": identity.process_ref, "job_ref": attestation.job_ref,
                "target_pid": pid, "duration_seconds": 28800, "ui_restart_enabled": False,
                "development_only": True,
            },
            "operator_checkpoints": ["arm", "focus", "manual_start"],
            "broker": {
                "process_ref": identity.process_ref, "job_ref": attestation.job_ref,
                "broker_ref": attestation.broker_ref,
                "ledger_row_hashes": list(history.row_hashes),
            },
            "controller_pid": 6000 + slot,
            "helper_pid": pid,
            "release_observation": None,
            "result": None,
            "telemetry_refs": [],
            "video_refs": [],
            "lease_audit_ref": None,
            "save": {"pre_sha256": pre_hash, "post_sha256": post_hash,
                     "backup_sha256": backup_sha256, "cloud_sync_attested": True},
            "outcome": event.to_wire(),
            "development_only": True,
            "formal_campaign_eligible": False,
            "save_hashes_complete": False,
        })
        run_records.append(name)
        outcome_records.append({"stage_execution_id": execution_id, "event": event.to_wire(),
                                "run_manifest": name})
        gate_records.append({
            "stage_execution_id": execution_id, "slot_id": slot, "attempt_id": attempt_id,
            "events": [item.to_wire() for item in ledger_events[attempt_id]],
            "ledger_row_hashes": list(history.row_hashes), "broker_response": None,
        })
    artifacts.append("preflight_attempts", preflight_records)
    artifacts.append("launched_outcomes", outcome_records)
    artifacts.append("launch_gates", gate_records)
    execution = {
        "plan_hash": plan_hash,
        "stage": stage,
        "stage_execution_id": execution_id,
        "state": "stage_passed" if sum(e.event_type is EventType.SUCCESS for e in outcomes) >= 16
        else "stage_not_promoted",
        "report_hash": _digest(f"report-{execution_id}"),
        "manifest_hash": manifest_hash,
        "parent_execution_id": None,
        "blocked_execution_ids": [],
        "threshold_parent_hash": _digest("threshold-parent"),
        "thresholds_hash": _digest("thresholds"),
        "development_only": True,
        "formal_campaign_eligible": False,
    }
    artifacts.put_json(f"stages/{execution_id}/execution.json", execution)
    return {"execution": execution, "run_records": run_records, "manifest_hash": manifest_hash}


def make_fixture(
    root: Path,
    *,
    successes: int = 16,
    include_c4: bool = True,
    omit_rng_control: bool = False,
    build_hash_override: str | None = None,
    missing_backup: bool = False,
    missing_restore: bool = False,
    duplicate_reserved: bool = False,
    duplicate_gameplay: bool = False,
) -> tuple[ArtifactStore, FakeLedger]:
    """保存・restore・ledgerを含むsynthetic campaignを作ります。

    既定fixtureはC4の16/20成功例で、引数に応じて拒否条件を差し替えます。
    """
    root.mkdir(parents=True, exist_ok=True)
    artifacts = ArtifactStore(root)
    histories: dict[str, LaunchHistory] = {}
    ledger_events: dict[str, tuple[CampaignEvent, ...]] = {}
    backup = b"synthetic original save"
    backup_hash = sha256_hex(backup)
    canonical = b"synthetic canonical save"
    canonical_hash_value = sha256_hex(canonical)
    build_hash = build_hash_override or _digest("game-build")
    plan = {
        "campaign_id": "fixture-c4", "mode": "synthetic",
        "executable_path": "C:/fixture/game.exe", "executable_hash": _digest("exe"),
        "build_hash": build_hash, "config_hash": _digest("config"),
        "argv": ["C:/fixture/game.exe", "-windowed"],
        "canonical_save_hash": canonical_hash_value,
        "thresholds": {"success": 0.8}, "threshold_parent_hash": _digest("threshold-parent"),
        "prerequisites": None, "prerequisite_parent_hash": None, "grace_seconds": 120,
        "campaign_run_mode": "formal_single_attempt", "ui_restart_enabled": False,
        "development_only": True, "formal_campaign_eligible": False,
    }
    plan_hash = canonical_hash(plan)
    artifacts.put_json("campaign/plan.json", plan)
    artifacts.put_json("campaign/goal_release_chain.json", _chain(_digest("game-build")))
    if not missing_backup:
        artifacts.put_immutable("save/original_backup.bin", backup)
        artifacts.put_json("save/original_backup.json", {
            "sha256": backup_hash, "size": len(backup), "source": "C:/private/original.sav",
        })
    artifacts.put_immutable("save/canonical.bin", canonical)
    artifacts.put_json("save/cloud_sync_attestation.json", {"cloud_sync_disabled": True})
    if not missing_restore:
        artifacts.put_json("save/restore_verdict.json", {
            "status": "PASS", "backup_sha256": backup_hash, "restored_sha256": backup_hash,
        })

    executions = []
    if include_c4:
        events = []
        for slot in range(20):
            kind = EventType.SUCCESS if slot < successes else EventType.GAMEPLAY_FAILURE
            events.append(CampaignEvent(
                kind, slot,
                failure_reason=None if kind is EventType.SUCCESS else "fixture_gameplay_failure",
                campaign_manifest_hash="f" * 64,
            ))
        # The stage manifest hash is bound after it is written below.
        stage_id = "fixture-c4.C4.x1"
        stage = _put_stage(artifacts, execution_id=stage_id, plan_hash=plan_hash,
                           outcomes=tuple(events), histories=histories, ledger_events=ledger_events,
                           backup_sha256=backup_hash, build_hash=build_hash,
                           omit_rng_control=omit_rng_control)
        # Repair the outcome manifest binding with the generated stage hash before finalizing streams.
        # Artifacts are immutable; the fixture creates final outcomes in the same manifest hash below.
        manifest_hash = stage["manifest_hash"]
        outcome_records = []
        for slot, event in enumerate(events):
            final_event = CampaignEvent(
                event.event_type, slot, failure_reason=event.failure_reason,
                campaign_manifest_hash=manifest_hash,
            )
            name = f"stages/{stage_id}/runs/slot-{slot:02d}.json"
            run = artifacts.read_json(name)
            run["outcome"] = final_event.to_wire()
            artifacts._path(name).write_bytes(canonical_json_bytes(run))
            attempt_id = run["attempt_id"]
            outcome_records.append({"stage_execution_id": stage_id, "event": final_event.to_wire(), "run_manifest": name})
            events = list(ledger_events[attempt_id])
            # Existing immutable files are only adjusted during fixture construction.
        outcome_path = artifacts._path("streams/launched_outcomes.jsonl")
        outcome_path.write_bytes(b"".join(
            canonical_json_bytes(record) + b"\n"
            for record in outcome_records
        ))
        executions.append(stage["execution"])
        if duplicate_reserved or duplicate_gameplay:
            # The ledger contract objects remain individually valid; cross-attempt identity must fail closed.
            attempt_ids = list(histories)
            first, second = attempt_ids[:2]
            for attempt_id, field in ((second, "reserved_run_id" if duplicate_reserved else "gameplay_attempt_id"),):
                old = histories[attempt_id]
                changed = {name: getattr(old.intent, name) for name in old.intent.__dataclass_fields__}
                changed[field] = getattr(histories[first].intent, field)
                replacement_intent = LaunchIntent(**changed)
                histories[attempt_id] = LaunchHistory(
                    old.attempt_id, replacement_intent, old.stages, old.row_hashes,
                    old.attestation, old.activation_source, old.failure_reason,
                )
                previous_events = ledger_events[attempt_id]
                ledger_events[attempt_id] = (
                    CampaignEvent(
                        EventType.LAUNCH_INTENT_COMMITTED, replacement_intent.slot_id,
                        attempt_id=replacement_intent.attempt_id,
                        reserved_run_id=replacement_intent.reserved_run_id,
                        gameplay_attempt_id=replacement_intent.gameplay_attempt_id,
                        launch_nonce=replacement_intent.launch_nonce,
                        campaign_manifest_hash=replacement_intent.campaign_manifest_hash,
                    ),
                    *previous_events[1:],
                )
                if duplicate_reserved:
                    path = artifacts._path(f"stages/{stage_id}/runs/slot-01.json")
                    run = artifacts.read_json(f"stages/{stage_id}/runs/slot-01.json")
                    run["reserved_run_id"] = replacement_intent.reserved_run_id
                    run["run_spec"]["reserved_run_id"] = replacement_intent.reserved_run_id
                    path.write_bytes(canonical_json_bytes(run))
                else:
                    path = artifacts._path(f"stages/{stage_id}/runs/slot-01.json")
                    run = artifacts.read_json(f"stages/{stage_id}/runs/slot-01.json")
                    run["gameplay_attempt_id"] = replacement_intent.gameplay_attempt_id
                    run["run_spec"]["gameplay_attempt_id"] = replacement_intent.gameplay_attempt_id
                    path.write_bytes(canonical_json_bytes(run))

    save_evidence = {
        "backup_sha256": None if missing_backup else backup_hash,
        "cloud_sync_attested": True,
        "canonical_sha256": canonical_hash_value,
        "restore_status": None if missing_restore else "PASS",
    }
    summary = {
        "campaign_id": "fixture-c4", "plan_hash": plan_hash,
        "executions": executions, "in_progress_execution_ids": [],
        "save_evidence": save_evidence,
        "restore_verdict": None if missing_restore else {
            "status": "PASS", "backup_sha256": backup_hash, "restored_sha256": backup_hash,
        },
        "development_only": True, "formal_campaign_eligible": False,
        "formal_evidence_eligible": False, "formal_evidence_run_manifests": [],
    }
    artifacts.put_json("campaign/summary.json", summary)
    facts = {"drive_type": "fixed", "filesystem": "NTFS", "volume_serial": 1}
    verdict = StorageVerdict(True, (), facts)
    ledger = FakeLedger(histories, ledger_events, root / "ledger", verdict,
                        {"journal_mode": "wal", "synchronous": "FULL",
                         "integrity_check": "ok", "sqlite_version": "3.45.1"})
    return artifacts, ledger


def _close_issue(artifacts: ArtifactStore, issue_id: str) -> None:
    """remediation issueに独立検証済みのclose recordを追加します。

    evidence objectのhash、作成者、別のclose担当者をclosureへ結びます。
    """
    evidence_path = f"campaign/remediation/{_digest(issue_id)}.json"
    evidence = issue_id.encode("utf-8")
    artifacts.put_immutable(evidence_path, evidence)
    path = "campaign/remediation_closures.json"
    payload = (artifacts.read_json(path) if artifacts.exists(path) else {
        "schema_version": "survivors.campaign_remediation_closures.v1", "closures": [],
    })
    payload["closures"].append({
        "issue_id": issue_id, "evidence_path": evidence_path,
        "evidence_sha256": sha256_hex(evidence), "author_id": "operator-a",
        "closed_by": "reviewer-b", "status": "closed",
    })
    artifacts._path(path).write_bytes(canonical_json_bytes(payload))


def _add_c4_execution(
    artifacts: ArtifactStore,
    ledger: FakeLedger,
    execution_id: str,
    *,
    successes: int,
    outcome_count: int,
    blocked_after: int | None = None,
) -> None:
    """C4 executionを追加し、必要ならpreflight failureで止めます。

    各executionのstreamとrun outcomeを固有manifest hashへ結び直します。
    """
    plan = artifacts.read_json("campaign/plan.json")
    backup_hash = artifacts.read_json("save/original_backup.json")["sha256"]
    outcomes = tuple(CampaignEvent(
        EventType.SUCCESS if slot < successes else EventType.GAMEPLAY_FAILURE,
        slot,
        failure_reason=None if slot < successes else "fixture_gameplay_failure",
        campaign_manifest_hash="f" * 64,
    ) for slot in range(outcome_count))
    stage = _put_stage(
        artifacts, execution_id=execution_id, plan_hash=canonical_hash(plan), outcomes=outcomes,
        histories=ledger.histories, ledger_events=ledger.events, backup_sha256=backup_hash,
        build_hash=plan["build_hash"],
    )
    records = artifacts.stream("launched_outcomes")
    for record in records:
        if record["stage_execution_id"] != execution_id:
            continue
        event_wire = dict(record["event"])
        event_wire["campaign_manifest_hash"] = stage["manifest_hash"]
        record["event"] = event_wire
        run = artifacts.read_json(record["run_manifest"])
        run["outcome"] = event_wire
        artifacts._path(record["run_manifest"]).write_bytes(canonical_json_bytes(run))
    artifacts._path("streams/launched_outcomes.jsonl").write_bytes(b"".join(
        canonical_json_bytes(record) + b"\n" for record in records
    ))
    execution = dict(stage["execution"])
    if blocked_after is not None:
        slot = blocked_after
        attempt_id = f"blocked-{execution_id}-{slot:02d}"
        nonce = _digest(f"blocked-nonce-{execution_id}-{slot:02d}")
        failed = (
            CampaignEvent(EventType.FORMAL_SLOT_RESERVED, slot,
                          campaign_manifest_hash=stage["manifest_hash"]),
            CampaignEvent(EventType.ATTEMPT_PREFLIGHT, slot, attempt_id=attempt_id,
                          details={"launch_nonce": nonce}, campaign_manifest_hash=stage["manifest_hash"]),
            CampaignEvent(EventType.PREFLIGHT_FAILED, slot, attempt_id=attempt_id,
                          failure_reason="preflight_failed", campaign_manifest_hash=stage["manifest_hash"]),
        )
        preflight = artifacts.stream("preflight_attempts")
        preflight.extend({"stage_execution_id": execution_id, "event": event.to_wire()} for event in failed)
        artifacts._path("streams/preflight_attempts.jsonl").write_bytes(b"".join(
            canonical_json_bytes(record) + b"\n" for record in preflight
        ))
        execution["state"] = "preflight_blocked"
        _close_issue(artifacts, f"{execution_id}.preflight.slot-{slot:02d}")
    artifacts._path(f"stages/{execution_id}/execution.json").write_bytes(canonical_json_bytes(execution))
    summary = artifacts.read_json("campaign/summary.json")
    summary["executions"].append(execution)
    artifacts._path("campaign/summary.json").write_bytes(canonical_json_bytes(summary))


def _permit_test_storage(monkeypatch, module) -> None:
    """storage検査をfixture専用のNTFS/WAL verdictへ置き換えます。

    本番のcheck_storageを避け、ledger artifactの検証経路に集中します。
    """
    monkeypatch.setattr(module, "check_storage", lambda _path: StorageVerdict(
        True, (), {"drive_type": "fixed", "filesystem": "NTFS", "volume_serial": 1}
    ))


def test_blocked_c4_executions_do_not_combine_into_promoted_metrics(tmp_path, monkeypatch):
    """別々のblocked executionの成功slotを合算しても昇格させません。

    remediationがclose済みでもpromoted C4 executionが無ければ拒否します。
    """
    from survivors.release import evidence_builder

    artifacts, ledger = make_fixture(tmp_path / "artifacts", include_c4=False)
    _add_c4_execution(artifacts, ledger, "fixture-c4.C4.x1", successes=10,
                      outcome_count=10, blocked_after=10)
    _add_c4_execution(artifacts, ledger, "fixture-c4.C4.x2", successes=10,
                      outcome_count=10, blocked_after=10)
    _permit_test_storage(monkeypatch, evidence_builder)
    with pytest.raises(EvidenceError, match="promoted C4"):
        build_goal_evidence(artifacts, ledger)


def test_blocked_c4_history_is_excluded_from_the_single_promoted_execution(tmp_path, monkeypatch):
    """blocked C4履歴を残し、唯一のpromoted executionだけを集計します。

    先行executionのattempt数に影響されず、成功16件と分母20件を返します。
    """
    from survivors.release import evidence_builder

    artifacts, ledger = make_fixture(tmp_path / "artifacts", include_c4=False)
    _add_c4_execution(artifacts, ledger, "fixture-c4.C4.x1", successes=3,
                      outcome_count=3, blocked_after=3)
    _add_c4_execution(artifacts, ledger, "fixture-c4.C4.x2", successes=16,
                      outcome_count=20)
    _permit_test_storage(monkeypatch, evidence_builder)
    evidence = build_goal_evidence(artifacts, ledger)
    assert evidence.report["successes"] == 16
    assert evidence.report["denominator"] == 20
    assert len(evidence.report["stage_history"]) == 2
    assert evidence.report["preflight_failures"] == 1


def test_c3_safety_failure_requires_and_records_remediation_close(tmp_path, monkeypatch):
    """C3のsafety failureにもcloseを要求し、履歴を出力へ残します。

    C4以外のstageでcloseを追加する前後を同じfixtureで確認します。
    """
    from survivors.release import evidence_builder

    artifacts, ledger = make_fixture(tmp_path / "artifacts")
    plan = artifacts.read_json("campaign/plan.json")
    execution_id = "fixture-c4.C3.x1"
    stage = _put_stage(
        artifacts, execution_id=execution_id, plan_hash=canonical_hash(plan), outcomes=(),
        histories=ledger.histories, ledger_events=ledger.events,
        backup_sha256=artifacts.read_json("save/original_backup.json")["sha256"],
        build_hash=plan["build_hash"], stage="C3",
    )
    execution = dict(stage["execution"])
    execution["state"] = "stage_passed"
    artifacts._path(f"stages/{execution_id}/execution.json").write_bytes(canonical_json_bytes(execution))
    summary = artifacts.read_json("campaign/summary.json")
    summary["executions"].append(execution)
    artifacts._path("campaign/summary.json").write_bytes(canonical_json_bytes(summary))
    safety = CampaignEvent(EventType.SAFETY_FAILURE, 0, failure_reason="fixture_safety_failure",
                           campaign_manifest_hash=stage["manifest_hash"])
    artifacts.append("launched_outcomes", [{
        "stage_execution_id": execution_id, "event": safety.to_wire(),
        "run_manifest": f"stages/{execution_id}/runs/slot-00.json",
    }])
    _permit_test_storage(monkeypatch, evidence_builder)
    with pytest.raises(EvidenceError, match="remediation"):
        build_goal_evidence(artifacts, ledger)

    issue_id = f"{execution_id}.safety.slot-00"
    _close_issue(artifacts, issue_id)
    evidence = build_goal_evidence(artifacts, ledger)
    assert issue_id in {entry["issue_id"] for entry in evidence.manifest["remediated_history"]}


def test_superseded_campaign_is_rejected_before_evidence_output(tmp_path, monkeypatch):
    """superseded campaign自体をdevelopment evidenceの対象から外します。

    close済みfailureの有無に依存せず、後継campaignを持つrootを拒否します。
    """
    from survivors.release import evidence_builder

    artifacts, ledger = make_fixture(tmp_path / "artifacts")
    artifacts.put_json("campaign/superseded.json", {
        "kind": "superseded", "campaign_id": "fixture-c4",
        "successor_campaign_id": "fixture-c4-next", "reason": "superseded fixture",
    })
    _permit_test_storage(monkeypatch, evidence_builder)
    with pytest.raises(EvidenceError, match="superseded"):
        build_goal_evidence(artifacts, ledger)


def test_valid_synthetic_sixteen_of_twenty_recomputes_metrics_and_round_trips(tmp_path, monkeypatch):
    """有効なsynthetic fixtureからmetricsを再計算して復元します。

    16/20、別々のcontrol field、安全なmanifest、backup restoreを確認します。
    """
    from survivors.release import evidence_builder

    artifacts, ledger = make_fixture(tmp_path / "artifacts")
    _permit_test_storage(monkeypatch, evidence_builder)
    evidence = build_goal_evidence(artifacts, ledger)

    assert evidence.manifest["development_only"] is True
    assert evidence.manifest["goal_release_eligible"] is False
    assert evidence.report["activation_count"] == 20
    assert evidence.report["denominator"] == 20
    assert evidence.report["terminal_outcomes"]["SUCCESS"] == 16
    assert evidence.report["successes"] == 16
    assert evidence.report["promotion_floor"] == 16
    assert evidence.report["promotion_eligible"] is True
    assert evidence.report["wilson_95_interval"] == pytest.approx((0.5838, 0.9192), abs=0.001)
    assert evidence.report["rng_control"] == "uncontrolled"
    assert evidence.report["trial_separation"] == "unique_run_id_separate_process"

    with tempfile.TemporaryDirectory(prefix="survivors-evidence-test-") as temp_dir:
        primary, backup = Path(temp_dir) / "primary", Path(temp_dir) / "backup"
        write_evidence_bundle(evidence, artifacts, ledger, primary, backup)
        assert (primary / "manifest.json").is_file()
        assert (primary / "report.json").is_file()
        assert (backup / "bundle" / "index.json").is_file()
        restored = verify_evidence_bundle(backup, Path(temp_dir) / "empty-root")
        assert restored.manifest == evidence.manifest
        assert restored.report == evidence.report
        output_text = (primary / "manifest.json").read_text(encoding="utf-8") + (primary / "report.json").read_text(encoding="utf-8")
        assert "C:/private/original.sav" not in output_text


def test_fifteen_of_twenty_is_rejected_before_output(tmp_path, monkeypatch):
    """15/20のcampaignは出力前に拒否します。

    observed promotion floorを下回る境界を固定します。
    """
    from survivors.release import evidence_builder

    artifacts, ledger = make_fixture(tmp_path / "artifacts", successes=15)
    _permit_test_storage(monkeypatch, evidence_builder)
    with pytest.raises(EvidenceError, match="16/20"):
        build_goal_evidence(artifacts, ledger)


def test_missing_c4_is_rejected(tmp_path, monkeypatch):
    """C4 executionが無いcampaignを拒否します。

    下位stageだけではgoal evidenceを生成できないことを確認します。
    """
    from survivors.release import evidence_builder

    artifacts, ledger = make_fixture(tmp_path / "artifacts", include_c4=False)
    _permit_test_storage(monkeypatch, evidence_builder)
    with pytest.raises(EvidenceError, match="C4"):
        build_goal_evidence(artifacts, ledger)


def test_cli_exposes_the_campaign_and_output_directories():
    """CLI helpが入力rootと一時出力先を公開することを確認します。

    指定したcampaign、ledger、backup、primaryの引数を検証します。
    """
    repo_root = Path(__file__).resolve().parents[4]
    script = repo_root / "Tools" / "Deployment" / "build_survivors_goal_evidence.py"
    result = subprocess.run([sys.executable, str(script), "--help"], capture_output=True,
                            encoding="utf-8", check=False)
    assert result.returncode == 0
    assert "--artifacts-root" in result.stdout
    assert "--ledger-directory" in result.stdout
    assert "--backup-directory" in result.stdout
    assert "--output-directory" in result.stdout


def test_missing_rng_control_is_rejected(tmp_path, monkeypatch):
    """manifestからRNG-control fieldが欠けた場合に拒否します。

    field欠落をdefault値で補わずfail-closedにします。
    """
    from survivors.release import evidence_builder

    artifacts, ledger = make_fixture(tmp_path / "artifacts", omit_rng_control=True)
    _permit_test_storage(monkeypatch, evidence_builder)
    with pytest.raises(EvidenceError, match="manifest"):
        build_goal_evidence(artifacts, ledger)


def test_mixed_build_chain_is_rejected(tmp_path, monkeypatch):
    """campaign buildとrelease chainの不一致を拒否します。

    同じfixture内の別build hashが混ざるケースを固定します。
    """
    from survivors.release import evidence_builder

    artifacts, ledger = make_fixture(tmp_path / "artifacts", build_hash_override=_digest("other-build"))
    _permit_test_storage(monkeypatch, evidence_builder)
    with pytest.raises(EvidenceError, match="build"):
        build_goal_evidence(artifacts, ledger)


def test_tampered_release_chain_digest_is_rejected(tmp_path, monkeypatch):
    """release chain nodeの改ざんをhash照合で拒否します。

    内容を変えて元のdigestを残したcounterexampleを使います。
    """
    from survivors.release import evidence_builder

    artifacts, ledger = make_fixture(tmp_path / "artifacts")
    chain = artifacts.read_json("campaign/goal_release_chain.json")
    chain["nodes"]["combat_model"]["content"]["artifact_sha256"] = "0" * 64
    artifacts._path("campaign/goal_release_chain.json").write_bytes(canonical_json_bytes(chain))
    _permit_test_storage(monkeypatch, evidence_builder)
    with pytest.raises(EvidenceError, match="hash mismatch"):
        build_goal_evidence(artifacts, ledger)


def test_population_independence_claim_is_rejected(tmp_path, monkeypatch):
    """独立性を主張するrelease chainを拒否します。

    forbidden claimがgoal outputへ流れないことを確認します。
    """
    from survivors.release import evidence_builder

    artifacts, ledger = make_fixture(tmp_path / "artifacts")
    chain = artifacts.read_json("campaign/goal_release_chain.json")
    chain["nodes"]["training"]["content"]["claims"] = "independent trials"
    artifacts._path("campaign/goal_release_chain.json").write_bytes(canonical_json_bytes(chain))
    _permit_test_storage(monkeypatch, evidence_builder)
    with pytest.raises(EvidenceError, match="claim"):
        build_goal_evidence(artifacts, ledger)


def test_synthetic_fixture_cannot_be_promoted_by_formal_root(tmp_path, monkeypatch):
    """synthetic fixtureをformal C4 rootとして受け入れません。

    development-only labelを正式適格性へ昇格できないことを確認します。
    """
    from survivors.release import evidence_builder

    artifacts, ledger = make_fixture(tmp_path / "artifacts")
    _permit_test_storage(monkeypatch, evidence_builder)
    with pytest.raises(EvidenceError, match="synthetic|formal"):
        build_goal_evidence(artifacts, ledger, formal_c4_root=artifacts.root)


@pytest.mark.parametrize("damage", ["missing", "tampered"])
@pytest.mark.parametrize("object_name", ["manifest.json", "source-0000.json"])
def test_empty_root_restore_rejects_missing_or_tampered_objects(tmp_path, monkeypatch, damage, object_name):
    """bundle objectの欠落と改ざんをempty-root restoreで検出します。

    manifestとsource referenceの両方へ破損を適用します。
    """
    from survivors.release import evidence_builder

    artifacts, ledger = make_fixture(tmp_path / "artifacts")
    _permit_test_storage(monkeypatch, evidence_builder)
    evidence = build_goal_evidence(artifacts, ledger)
    with tempfile.TemporaryDirectory(prefix="survivors-evidence-test-") as temp_dir:
        root = Path(temp_dir)
        backup = root / "backup"
        write_evidence_bundle(evidence, artifacts, ledger, root / "primary", backup)
        manifest_object = backup / "bundle" / "objects" / object_name
        if damage == "missing":
            manifest_object.unlink()
        else:
            manifest_object.write_bytes(b"{}\n")
        with pytest.raises(EvidenceError, match="object|hash"):
            verify_evidence_bundle(backup, root / "empty")


@pytest.mark.parametrize("missing", ["backup", "restore"])
def test_missing_save_backup_or_restore_is_rejected(tmp_path, monkeypatch, missing):
    """backupまたはrestore verdictが無いcampaignを拒否します。

    どちらか片方だけ揃ったsave identityも正式証跡にしません。
    """
    from survivors.release import evidence_builder

    artifacts, ledger = make_fixture(
        tmp_path / "artifacts", missing_backup=missing == "backup", missing_restore=missing == "restore"
    )
    _permit_test_storage(monkeypatch, evidence_builder)
    with pytest.raises(EvidenceError, match="save|restore|backup"):
        build_goal_evidence(artifacts, ledger)


def test_preflight_or_launch_gate_contamination_is_rejected(tmp_path, monkeypatch):
    """launch gate failureの混入を拒否します。

    CREATE_PROCESS_FAILED相当のeventがterminal outcomeを置き換えません。
    """
    from survivors.release import evidence_builder

    artifacts, ledger = make_fixture(tmp_path / "artifacts")
    records = artifacts.stream("preflight_attempts")
    failure = CampaignEvent(EventType.LAUNCH_GATE_FAILED, 0, failure_reason="create_process_failed",
                            campaign_manifest_hash=records[0]["event"]["campaign_manifest_hash"])
    records.append({"stage_execution_id": "fixture-c4.C4.x1", "event": failure.to_wire()})
    path = artifacts._path("streams/preflight_attempts.jsonl")
    path.write_bytes(b"".join(
        canonical_json_bytes(record) + b"\n"
        for record in records
    ))
    _permit_test_storage(monkeypatch, evidence_builder)
    with pytest.raises(EvidenceError):
        build_goal_evidence(artifacts, ledger)


@pytest.mark.parametrize("duplicate", ["reserved", "gameplay"])
def test_duplicate_c4_identity_is_rejected(tmp_path, monkeypatch, duplicate):
    """重複reservedまたはgameplay identityを拒否します。

    C4全attemptを横断するidentity cardinalityを検証します。
    """
    from survivors.release import evidence_builder

    artifacts, ledger = make_fixture(
        tmp_path / "artifacts",
        duplicate_reserved=duplicate == "reserved",
        duplicate_gameplay=duplicate == "gameplay",
    )
    _permit_test_storage(monkeypatch, evidence_builder)
    with pytest.raises(EvidenceError, match="duplicate|identity|reserved|gameplay"):
        build_goal_evidence(artifacts, ledger)


@pytest.mark.parametrize("failure", ["ntfs", "wal"])
def test_missing_ntfs_or_wal_attestation_is_rejected(tmp_path, monkeypatch, failure):
    """NTFSまたはWAL/FULL attestationが欠けたledgerを拒否します。

    storage verdictと実SQLite設定の両方の失敗経路を固定します。
    """
    from survivors.release import evidence_builder

    artifacts, ledger = make_fixture(tmp_path / "artifacts")
    if failure == "ntfs":
        monkeypatch.setattr(evidence_builder, "check_storage", lambda _path: StorageVerdict(
            False, ("filesystem_not_ntfs",), {"drive_type": "fixed", "filesystem": "ReFS"}
        ))
    else:
        _permit_test_storage(monkeypatch, evidence_builder)
        ledger.pragmas["synchronous"] = "NORMAL"
    with pytest.raises(EvidenceError, match="storage|NTFS|WAL|FULL|integrity"):
        build_goal_evidence(artifacts, ledger)


def test_unclosed_preflight_remediation_is_rejected(tmp_path, monkeypatch):
    """PREFLIGHT_FAILED eventにcloseが無いcampaignを拒否します。

    通常campaignでもblocked historyをremediationなしで通しません。
    """
    from survivors.release import evidence_builder

    artifacts, ledger = make_fixture(tmp_path / "artifacts")
    artifacts.put_json("campaign/remediation_closures.json", {
        "schema_version": "survivors.campaign_remediation_closures.v1",
        "closures": [],
    })
    failure = CampaignEvent(EventType.PREFLIGHT_FAILED, 19, attempt_id="failed-attempt",
                            failure_reason="preflight_failed",
                            campaign_manifest_hash=ledger.history(next(iter(ledger.histories))).intent.campaign_manifest_hash)
    records = artifacts.stream("preflight_attempts")
    records.extend([
        {"stage_execution_id": "fixture-c4.C4.x1",
         "event": CampaignEvent(EventType.FORMAL_SLOT_RESERVED, 19,
                                 campaign_manifest_hash=failure.campaign_manifest_hash).to_wire()},
        {"stage_execution_id": "fixture-c4.C4.x1",
         "event": CampaignEvent(EventType.ATTEMPT_PREFLIGHT, 19, attempt_id="failed-attempt",
                                 campaign_manifest_hash=failure.campaign_manifest_hash).to_wire()},
        {"stage_execution_id": "fixture-c4.C4.x1", "event": failure.to_wire()},
    ])
    artifacts._path("streams/preflight_attempts.jsonl").write_bytes(b"".join(
        canonical_json_bytes(record) + b"\n"
        for record in records
    ))
    _permit_test_storage(monkeypatch, evidence_builder)
    with pytest.raises(EvidenceError, match="remediation"):
        build_goal_evidence(artifacts, ledger)


def test_safety_failure_without_remediation_close_is_rejected(tmp_path, monkeypatch):
    """C4 SAFETY_FAILURE eventにcloseが無い場合を拒否します。

    run outcomeとlaunched outcome streamに同じfailureを置きます。
    """
    from survivors.release import evidence_builder

    artifacts, ledger = make_fixture(tmp_path / "artifacts")
    path = "stages/fixture-c4.C4.x1/runs/slot-19.json"
    run = artifacts.read_json(path)
    safety = CampaignEvent(EventType.SAFETY_FAILURE, 19, failure_reason="fixture_safety_failure",
                           campaign_manifest_hash=run["campaign_manifest_hash"])
    run["outcome"] = safety.to_wire()
    artifacts._path(path).write_bytes(canonical_json_bytes(run))
    outcomes = artifacts.stream("launched_outcomes")
    outcomes[-1]["event"] = safety.to_wire()
    artifacts._path("streams/launched_outcomes.jsonl").write_bytes(b"".join(
        canonical_json_bytes(record) + b"\n" for record in outcomes
    ))
    _permit_test_storage(monkeypatch, evidence_builder)
    with pytest.raises(EvidenceError, match="remediation"):
        build_goal_evidence(artifacts, ledger)
