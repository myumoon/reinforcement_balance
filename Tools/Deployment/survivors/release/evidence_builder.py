"""Survivors C4 campaign artifacts and durable launch history into goal evidence.

Inputs are re-read through the campaign artifact and ledger APIs. The generated
manifest contains only logical artifact names and digests, never source payloads.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reinbalance_survivors_contracts.canonical_json import canonical_hash, canonical_json_bytes, sha256_hex

from survivors.campaign.campaign_report import wilson_score_interval
from survivors.campaign.campaign_runner import ArtifactStore
from survivors.campaign.campaign_schema import (
    CAMPAIGN_SLOT_COUNT,
    STAGE_POLICIES,
    CampaignEvent,
    CampaignManifest,
    EventType,
    TERMINAL_OUTCOMES,
    campaign_manifest_hash,
    validate_campaign_events,
    validate_prerequisites,
)
from survivors.campaign.durable_launch_store import (
    LEDGER_SCHEMA_ID,
    LEDGER_SCHEMA_VERSION,
    DurableLaunchStore,
    LaunchStage,
    StorageVerdict,
    check_storage,
)
from survivors.campaign.win32_launch_broker import BROKER_SCHEMA


EVIDENCE_SCHEMA = "survivors.goal_evidence.v1"
RELEASE_CHAIN_SCHEMA = "survivors.goal_release_chain.v1"
BUNDLE_SCHEMA = "survivors.goal_evidence_bundle.v1"
REMEDIATION_SCHEMA = "survivors.campaign_remediation_closures.v1"
CHAIN_NODES = (
    "target_profile", "game_build", "combat_model", "vecnormalize", "deploy_schema",
    "error_profile", "selector", "parser", "detector", "controller", "training",
    "teacher", "dataset", "evaluation",
)
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SCHEMA_VERSION = re.compile(r"[A-Za-z][A-Za-z0-9_.-]*\.v[1-9][0-9]*\Z")
_FORBIDDEN_CLAIM_KEYS = frozenset({
    "seed", "real_seed", "same_seed", "same_seed_claim", "independent",
    "statistical_independence", "independence_claim", "population_success_probability",
})
_ACTIVATED = frozenset({EventType.FORMAL_RUN_ACTIVATED})
_LEDGER_CLOSED = frozenset({
    LaunchStage.FORMAL_RUN_ACTIVATED,
    LaunchStage.LAUNCH_GATE_FAILED,
    LaunchStage.LAUNCH_UNCERTAIN,
})


class EvidenceError(ValueError):
    """Evidence is absent or inconsistent.

    Callers should stop the build instead of converting this validation failure
    into a partial report.
    """


@dataclass(frozen=True, slots=True)
class GoalEvidence:
    """Sanitized manifest and independently recomputed campaign report.

    Source object bodies are omitted; the manifest keeps their logical ids and
    SHA-256 references for audit and restore checks.
    """

    manifest: dict[str, Any]
    report: dict[str, Any]


class _Artifacts:
    def __init__(self, store: ArtifactStore) -> None:
        self.store = store
        self.inventory: dict[str, str] = {}

    def exists(self, name: str) -> bool:
        return self.store.exists(name)

    def read(self, name: str) -> bytes:
        try:
            data = self.store.read(name)
        except OSError as exc:
            raise EvidenceError(f"missing artifact: {name}") from exc
        self.inventory[name] = sha256_hex(data)
        return data

    def json(self, name: str) -> Any:
        try:
            value = json.loads(self.read(name))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise EvidenceError(f"invalid JSON artifact: {name}") from exc
        return value

    def stream(self, name: str) -> list[dict[str, Any]]:
        logical_id = f"streams/{name}.jsonl"
        if not self.exists(logical_id):
            return []
        try:
            records = self.store.stream(name)
            raw = self.store.read(logical_id)
        except (OSError, ValueError) as exc:
            raise EvidenceError(f"invalid artifact stream: {logical_id}") from exc
        self.inventory[logical_id] = sha256_hex(raw)
        return records


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EvidenceError(f"{name} must be an object")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise EvidenceError(f"{name} must be non-empty text")
    return value


def _id(value: Any, name: str) -> str:
    text = _text(value, name)
    if _ID.fullmatch(text) is None:
        raise EvidenceError(f"{name} must be a sanitized logical id")
    return text


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise EvidenceError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _reject_claims(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if isinstance(key, str) and key.casefold() in _FORBIDDEN_CLAIM_KEYS:
                raise EvidenceError(f"unsupported population claim field: {key}")
            if isinstance(key, str) and key.casefold() in {"claim", "claims", "interpretation"}:
                if isinstance(child, str) and any(term in child.casefold() for term in ("independent", "独立")):
                    raise EvidenceError("population-independence claims are not supported")
            _reject_claims(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _reject_claims(child)


def _events(records: Sequence[Mapping[str, Any]], execution_id: str) -> list[CampaignEvent]:
    result = []
    for record in records:
        if record.get("stage_execution_id") == execution_id:
            try:
                event = CampaignEvent.from_wire(record["event"])
                _reject_claims(event.to_wire())
                result.append(event)
            except (KeyError, TypeError, ValueError) as exc:
                raise EvidenceError(f"invalid campaign event in {execution_id}") from exc
    return result


def _storage_attestation(store: DurableLaunchStore) -> dict[str, Any]:
    try:
        verdict = check_storage(store.directory)
        opened_verdict = store.verdict
        pragmas = store.pragmas
    except (AttributeError, OSError, TypeError) as exc:
        raise EvidenceError("durable store storage attestation is unavailable") from exc
    if not isinstance(verdict, StorageVerdict) or not verdict.eligible:
        raise EvidenceError("ledger must reside on local fixed NTFS storage")
    if verdict.facts.get("drive_type") != "fixed" or verdict.facts.get("filesystem") != "NTFS":
        raise EvidenceError("ledger storage facts do not attest fixed NTFS")
    if not isinstance(opened_verdict, StorageVerdict) or not opened_verdict.eligible:
        raise EvidenceError("ledger ACL or storage checks did not pass when opened")
    if not isinstance(pragmas, Mapping) or any(
        pragmas.get(key) != value for key, value in (
            ("journal_mode", "wal"), ("synchronous", "FULL"), ("integrity_check", "ok")
        )
    ):
        raise EvidenceError("ledger WAL/FULL/integrity attestation is missing")
    if not isinstance(pragmas.get("sqlite_version"), str) or not pragmas["sqlite_version"]:
        raise EvidenceError("ledger SQLite version attestation is missing")
    attestation = {
        "storage": "local_fixed_ntfs",
        "acl_verified_on_open": True,
        "journal_mode": "wal",
        "synchronous": "FULL",
        "integrity_check": "ok",
        "sqlite_version": pragmas["sqlite_version"],
        "ledger_schema": LEDGER_SCHEMA_ID,
        "ledger_schema_version": LEDGER_SCHEMA_VERSION,
        "broker_schema": BROKER_SCHEMA,
    }
    attestation["attestation_sha256"] = canonical_hash(attestation)
    return attestation


def _validate_chain(chain: Any, plan: Mapping[str, Any]) -> tuple[str, list[dict[str, str]]]:
    chain = _mapping(chain, "goal release chain")
    required = {"schema_version", "target_profile_sha256", "game_build_sha256", "nodes", "chain_sha256"}
    if set(chain) != required or chain.get("schema_version") != RELEASE_CHAIN_SCHEMA:
        raise EvidenceError("goal release chain schema or fields are invalid")
    nodes = _mapping(chain["nodes"], "goal release chain nodes")
    if set(nodes) != set(CHAIN_NODES):
        raise EvidenceError("goal release chain is missing or mixes required artifacts")
    hashes: dict[str, str] = {}
    values: dict[str, Mapping[str, Any]] = {}
    for name in CHAIN_NODES:
        node = _mapping(nodes[name], f"chain node {name}")
        if set(node) != {"schema_version", "content", "parents", "sha256"}:
            raise EvidenceError(f"chain node {name} has unexpected fields")
        version = node["schema_version"]
        if (not isinstance(version, str) or _SCHEMA_VERSION.fullmatch(version) is None
                or version != f"survivors.{name}.v1"):
            raise EvidenceError(f"chain node {name} has no supported schema version")
        content = _mapping(node["content"], f"chain node {name} content")
        parents = node["parents"]
        if not isinstance(parents, list) or any(not isinstance(parent, str) for parent in parents):
            raise EvidenceError(f"chain node {name} parents must be a digest list")
        body = {"schema_version": version, "content": dict(content), "parents": parents}
        digest = _sha(node["sha256"], f"chain node {name} sha256")
        if canonical_hash(body) != digest:
            raise EvidenceError(f"chain node {name} hash mismatch")
        hashes[name] = digest
        values[name] = content

    expected_parents: dict[str, list[str]] = {
        "target_profile": [],
        "game_build": [],
        **{name: [hashes["target_profile"], hashes["game_build"]] for name in CHAIN_NODES[2:10]},
        "training": [hashes["combat_model"], hashes["vecnormalize"], hashes["target_profile"], hashes["game_build"]],
        "teacher": [hashes["training"], hashes["target_profile"], hashes["game_build"]],
        "dataset": [hashes["teacher"], hashes["target_profile"], hashes["game_build"]],
        "evaluation": [hashes["training"], hashes["teacher"], hashes["dataset"],
                       hashes["target_profile"], hashes["game_build"]],
    }
    for name in CHAIN_NODES:
        if nodes[name]["parents"] != expected_parents[name]:
            raise EvidenceError(f"chain node {name} has a mixed artifact/profile/build parent")
    _sha(values["target_profile"].get("profile_sha256"), "target profile SHA-256")
    build_id = _id(values["game_build"].get("build_id"), "game build id")
    build_hash = _sha(values["game_build"].get("build_sha256"), "game build SHA-256")
    if build_hash != plan.get("build_hash"):
        raise EvidenceError(f"game build hash does not match campaign plan ({build_id})")
    for name in CHAIN_NODES[2:]:
        _sha(values[name].get("artifact_sha256"), f"{name} artifact SHA-256")
    for name in CHAIN_NODES[10:]:
        if values[name].get("verdict") != "PASS":
            raise EvidenceError(f"{name} verdict is not PASS")

    target_digest = _sha(chain["target_profile_sha256"], "target profile binding")
    build_digest = _sha(chain["game_build_sha256"], "game build binding")
    if target_digest != hashes["target_profile"] or build_digest != hashes["game_build"]:
        raise EvidenceError("goal release chain profile/build binding mismatch")
    body = {key: chain[key] for key in required - {"chain_sha256"}}
    chain_digest = _sha(chain["chain_sha256"], "goal release chain hash")
    if canonical_hash(body) != chain_digest:
        raise EvidenceError("goal release chain hash mismatch")

    prerequisites = plan.get("prerequisites")
    parent_hash = plan.get("prerequisite_parent_hash")
    if prerequisites is not None:
        try:
            validated = validate_prerequisites(prerequisites, expected_parent_hash=parent_hash)
        except (TypeError, ValueError) as exc:
            raise EvidenceError("campaign prerequisite chain is invalid") from exc
        if validated.hashes["target"] != values["target_profile"].get("profile_sha256"):
            raise EvidenceError("campaign target profile hash differs from the release chain")
        if validated.hashes["build"] != build_hash:
            raise EvidenceError("campaign prerequisite build hash differs from the release chain")
    elif not plan.get("development_only"):
        raise EvidenceError("formal campaign is missing its verified prerequisites")
    chain_refs = [{"name": name, "schema_version": nodes[name]["schema_version"], "sha256": hashes[name]}
                  for name in CHAIN_NODES]
    return chain_digest, chain_refs


def _validate_save(reader: _Artifacts, plan: Mapping[str, Any], summary: Mapping[str, Any]) -> dict[str, Any]:
    backup = reader.read("save/original_backup.bin")
    backup_hash = sha256_hex(backup)
    backup_record = _mapping(reader.json("save/original_backup.json"), "original backup record")
    if set(backup_record) != {"sha256", "size", "source"}:
        raise EvidenceError("original backup record has an unexpected shape")
    if backup_record.get("sha256") != backup_hash or backup_record.get("size") != len(backup):
        raise EvidenceError("original save backup hash or size mismatch")
    cloud = _mapping(reader.json("save/cloud_sync_attestation.json"), "cloud sync attestation")
    if cloud.get("cloud_sync_disabled") is not True:
        raise EvidenceError("cloud sync is not attested disabled")
    canonical_hash_value = sha256_hex(reader.read("save/canonical.bin"))
    if canonical_hash_value != plan.get("canonical_save_hash"):
        raise EvidenceError("canonical save identity does not match campaign plan")
    restore = _mapping(reader.json("save/restore_verdict.json"), "restore verdict")
    if set(restore) != {"status", "backup_sha256", "restored_sha256"}:
        raise EvidenceError("restore verdict has an unexpected shape")
    if (restore.get("status") != "PASS" or restore.get("backup_sha256") != backup_hash
            or restore.get("restored_sha256") != backup_hash):
        raise EvidenceError("original save restore verdict failed verification")
    summary_evidence = _mapping(summary.get("save_evidence"), "campaign save evidence")
    if set(summary_evidence) != {"backup_sha256", "cloud_sync_attested", "canonical_sha256", "restore_status"}:
        raise EvidenceError("summary save evidence has an unexpected shape")
    expected_evidence = {
        "backup_sha256": backup_hash,
        "cloud_sync_attested": True,
        "canonical_sha256": canonical_hash_value,
        "restore_status": "PASS",
    }
    if dict(summary_evidence) != expected_evidence or dict(summary.get("restore_verdict", {})) != dict(restore):
        raise EvidenceError("summary save/restore claims differ from the saved artifacts")
    return expected_evidence


def _read_run(reader: _Artifacts, path: str, plan: Mapping[str, Any], backup_hash: str) -> dict[str, Any]:
    run = _mapping(reader.json(path), f"run manifest {path}")
    required = {
        "campaign_id", "plan_hash", "stage", "stage_execution_id", "campaign_manifest_hash",
        "slot_id", "attempt_id", "reserved_run_id", "gameplay_attempt_id", "campaign_run_mode",
        "activation_source", "run_spec", "operator_checkpoints", "broker", "controller_pid",
        "helper_pid", "release_observation", "result", "telemetry_refs", "video_refs",
        "lease_audit_ref", "save", "outcome", "development_only", "formal_campaign_eligible",
        "save_hashes_complete",
    }
    if not required <= set(run) or "formal_evidence_eligible" in run:
        raise EvidenceError(f"run manifest fields do not match the campaign contract: {path}")
    if run["plan_hash"] != plan.get("plan_hash") or run["campaign_run_mode"] != "formal_single_attempt":
        raise EvidenceError(f"run manifest plan or run mode mismatch: {path}")
    save = _mapping(run["save"], f"run save identity {path}")
    _sha(save.get("pre_sha256"), f"{path} pre-save SHA-256")
    _sha(save.get("post_sha256"), f"{path} post-save SHA-256")
    if save.get("backup_sha256") != backup_hash or save.get("cloud_sync_attested") is not True:
        raise EvidenceError(f"run save backup/cloud identity is incomplete: {path}")
    for kind in ("pre", "post"):
        record = _mapping(reader.json(f"save/{kind}/{run['attempt_id']}.json"), f"{kind}-save record")
        if record.get("attempt_id") != run["attempt_id"] or record.get("sha256") != save[f"{kind}_sha256"]:
            raise EvidenceError(f"run {kind}-save hash is missing or does not match: {path}")
    development_only = plan.get("development_only") is True
    save_complete = (not development_only and save["pre_sha256"] is not None
                     and save["post_sha256"] is not None and backup_hash is not None
                     and save["cloud_sync_attested"] is True)
    if (run.get("development_only") is not development_only
            or run.get("formal_campaign_eligible") is not (not development_only)
            or run.get("save_hashes_complete") is not save_complete):
        raise EvidenceError(f"run manifest eligibility fields are inconsistent: {path}")
    return dict(run)


def _load_closures(
    reader: _Artifacts,
    required_issues: set[str],
) -> list[dict[str, str]]:
    path = "campaign/remediation_closures.json"
    if not reader.exists(path):
        if required_issues:
            raise EvidenceError("remediation closure is required for blocked campaign history")
        return []
    payload = _mapping(reader.json(path), "remediation closures")
    if set(payload) != {"schema_version", "closures"} or payload.get("schema_version") != REMEDIATION_SCHEMA:
        raise EvidenceError("remediation closure schema is invalid")
    closures = payload["closures"]
    if not isinstance(closures, list):
        raise EvidenceError("remediation closures must be an array")
    validated: dict[str, dict[str, str]] = {}
    for closure in closures:
        closure = _mapping(closure, "remediation closure")
        if set(closure) != {"issue_id", "evidence_path", "evidence_sha256", "author_id", "closed_by", "status"}:
            raise EvidenceError("remediation closure has unexpected fields")
        issue_id = _id(closure["issue_id"], "remediation issue id")
        evidence_path = _text(closure["evidence_path"], "remediation evidence path")
        if any(_ID.fullmatch(part) is None or part in {".", ".."} for part in evidence_path.split("/")):
            raise EvidenceError("remediation evidence path must be repository-relative")
        evidence_hash = _sha(closure["evidence_sha256"], "remediation evidence SHA-256")
        actual_hash = sha256_hex(reader.read(evidence_path))
        author = _text(closure["author_id"], "remediation author")
        verifier = _text(closure["closed_by"], "remediation verifier")
        if (actual_hash != evidence_hash or closure["status"] != "closed"
                or author == verifier or issue_id in validated):
            raise EvidenceError(f"remediation closure is invalid: {issue_id}")
        validated[issue_id] = {
            "issue_id": issue_id,
            "evidence_path": evidence_path,
            "evidence_sha256": evidence_hash,
            "verifier_sha256": sha256_hex(verifier.encode("utf-8")),
        }
    missing = required_issues - validated.keys()
    if missing:
        raise EvidenceError(f"remediation closure is missing: {', '.join(sorted(missing))}")
    return [validated[key] for key in sorted(validated)]


def _safe_output(value: Any, key: str = "") -> None:
    if isinstance(value, Mapping):
        for name, child in value.items():
            if not isinstance(name, str) or re.search(r"secret|password|token|frame|payload|authorization", name, re.I):
                raise EvidenceError("output contains a secret or raw payload field")
            _safe_output(child, name)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _safe_output(child, key)
    elif isinstance(value, str):
        if re.search(r"^(?:[A-Za-z]:[\\/]|[\\/]{2,}|/)", value):
            raise EvidenceError("output contains an absolute path")
        lowered = value.casefold()
        if any(term in lowered for term in ("independent", "statistical_independence", "population_success_probability", "独立")):
            raise EvidenceError("output contains a population-independence claim")


def _check_c4_events(
    reader: _Artifacts,
    store: DurableLaunchStore,
    summary: Mapping[str, Any],
    plan: Mapping[str, Any],
    plan_hash: str,
    backup_hash: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    executions = summary.get("executions")
    if not isinstance(executions, list):
        raise EvidenceError("summary executions must be an array")
    c4_executions = []
    all_execution_ids: set[str] = set()
    for raw in executions:
        execution = _mapping(raw, "summary execution")
        execution_id = _id(execution.get("stage_execution_id"), "stage execution id")
        if execution_id in all_execution_ids:
            raise EvidenceError("summary has duplicate stage execution ids")
        all_execution_ids.add(execution_id)
        if execution.get("stage") == "C4":
            c4_executions.append(dict(execution))
    if not c4_executions:
        raise EvidenceError("C4 stage is missing")

    stage_data: dict[str, dict[str, Any]] = {}
    for execution in c4_executions:
        execution_id = execution["stage_execution_id"]
        path = f"stages/{execution_id}"
        actual_execution = _mapping(reader.json(f"{path}/execution.json"), "stage execution artifact")
        if dict(actual_execution) != execution:
            raise EvidenceError(f"summary execution differs from its artifact: {execution_id}")
        if actual_execution.get("plan_hash") != plan_hash or actual_execution.get("stage") != "C4":
            raise EvidenceError(f"C4 execution plan/stage mismatch: {execution_id}")
        envelope = _mapping(reader.json(f"{path}/manifest.json"), "stage manifest artifact")
        manifest_wire = _mapping(envelope.get("manifest"), "campaign manifest")
        try:
            manifest = CampaignManifest.from_wire(manifest_wire)
        except (TypeError, ValueError) as exc:
            raise EvidenceError(f"invalid campaign manifest: {execution_id}") from exc
        digest = campaign_manifest_hash(manifest)
        if (manifest.stage != "C4" or manifest.expected_slots != STAGE_POLICIES["C4"].slot_count
                or manifest.campaign_id != execution_id or envelope.get("plan_hash") != plan_hash
                or actual_execution.get("manifest_hash") != digest):
            raise EvidenceError(f"C4 manifest chain mismatch: {execution_id}")
        if (manifest.mode != plan.get("mode") or manifest.development_only != plan.get("development_only")
                or envelope.get("development_only") is not manifest.development_only
                or envelope.get("formal_campaign_eligible") is not (not manifest.development_only)):
            raise EvidenceError(f"C4 eligibility differs from campaign plan: {execution_id}")
        stage_data[execution_id] = {"manifest": manifest, "hash": digest, "events": []}

    executions_by_hash = {value["hash"]: key for key, value in stage_data.items()}
    if len(executions_by_hash) != len(stage_data):
        raise EvidenceError("C4 stage executions share a campaign manifest identity")

    preflight_records = reader.stream("preflight_attempts")
    outcome_records = reader.stream("launched_outcomes")
    gate_records = reader.stream("launch_gates")
    chain_records = reader.stream("campaign_chain")
    known_stream_ids = all_execution_ids
    for records in (preflight_records, outcome_records, gate_records, chain_records):
        for record in records:
            execution_id = record.get("stage_execution_id")
            if execution_id in known_stream_ids or not isinstance(execution_id, str):
                continue
            if execution_id.startswith(f"{plan['campaign_id']}.C4."):
                raise EvidenceError(f"unlisted C4 artifact history: {execution_id}")

    superseded = None
    if reader.exists("campaign/superseded.json"):
        superseded = _mapping(reader.json("campaign/superseded.json"), "superseded campaign record")
        if (set(superseded) != {"kind", "campaign_id", "successor_campaign_id", "reason"}
                or superseded.get("kind") != "superseded"
                or superseded.get("campaign_id") != plan.get("campaign_id")
                or _id(superseded.get("successor_campaign_id"), "successor campaign id") == plan.get("campaign_id")):
            raise EvidenceError("superseded campaign record is invalid")
    required_remediations: set[str] = set()
    raw_preflight_by_exec: dict[str, list[CampaignEvent]] = defaultdict(list)
    for execution_id in stage_data:
        events = _events(preflight_records, execution_id)
        raw_preflight_by_exec[execution_id] = events
        for event in events:
            if event.event_type is EventType.PREFLIGHT_FAILED:
                required_remediations.add(f"{execution_id}.preflight.slot-{event.slot_id:02d}")
    closures = _load_closures(reader, required_remediations)
    ledger_attempts: dict[str, dict[int, tuple[Any, tuple[CampaignEvent, ...]]]] = defaultdict(dict)
    global_ids: dict[str, set[str]] = {name: set() for name in (
        "attempt_id", "reserved_run_id", "gameplay_attempt_id", "launch_nonce", "process_ref", "job_ref"
    )}
    all_attempt_ids = tuple(store.attempt_ids())
    if len(set(all_attempt_ids)) != len(all_attempt_ids):
        raise EvidenceError("durable ledger returned duplicate attempt identities")
    for attempt_id in all_attempt_ids:
        try:
            history = store.history(attempt_id)
            intent = history.intent
            final_stage = history.last_stage
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise EvidenceError("durable ledger history is invalid") from exc
        if final_stage not in _LEDGER_CLOSED:
            raise EvidenceError(f"unresolved LAUNCH_INTENT/resume boundary: {attempt_id}")
        if not isinstance(history.row_hashes, tuple) or any(_SHA256.fullmatch(row) is None for row in history.row_hashes):
            raise EvidenceError(f"ledger reconciliation reference is missing: {attempt_id}")
        if intent.campaign_manifest_hash not in executions_by_hash:
            continue
        execution_id = executions_by_hash[intent.campaign_manifest_hash]
        if attempt_id != intent.attempt_id:
            raise EvidenceError("ledger attempt id differs from launch intent")
        events = tuple(store.campaign_events(attempt_id))
        normalized = []
        for raw in events:
            try:
                event = raw if isinstance(raw, CampaignEvent) else CampaignEvent.from_wire(raw)
            except (TypeError, ValueError) as exc:
                raise EvidenceError(f"invalid durable campaign event: {attempt_id}") from exc
            if event.campaign_manifest_hash != intent.campaign_manifest_hash or event.slot_id != intent.slot_id:
                raise EvidenceError(f"ledger event is bound to a different C4 slot: {attempt_id}")
            normalized.append(event)
        if sum(event.event_type is EventType.LAUNCH_INTENT_COMMITTED for event in normalized) != 1:
            raise EvidenceError(f"attempt must contain one launch intent: {attempt_id}")
        intent_event = next(event for event in normalized if event.event_type is EventType.LAUNCH_INTENT_COMMITTED)
        if (intent_event.attempt_id != intent.attempt_id or intent_event.reserved_run_id != intent.reserved_run_id
                or intent_event.gameplay_attempt_id != intent.gameplay_attempt_id
                or intent_event.launch_nonce != intent.launch_nonce):
            raise EvidenceError(f"ledger event identity differs from launch intent: {attempt_id}")
        activation_count = sum(event.event_type is EventType.FORMAL_RUN_ACTIVATED for event in normalized)
        if activation_count > 1:
            raise EvidenceError(f"reserved run id was activated more than once: {intent.reserved_run_id}")
        if any(event.event_type in {EventType.LAUNCH_GATE_FAILED, EventType.LAUNCH_UNCERTAIN} for event in normalized):
            raise EvidenceError(f"C4 launch gate failure/uncertain history: {attempt_id}")
        activation = next((event for event in normalized if event.event_type is EventType.FORMAL_RUN_ACTIVATED), None)
        if (activation is not None) != (final_stage is LaunchStage.FORMAL_RUN_ACTIVATED):
            raise EvidenceError(f"ledger activation event differs from attempt lifecycle: {attempt_id}")
        intent_identities = {
            "attempt_id": intent.attempt_id,
            "reserved_run_id": intent.reserved_run_id,
            "gameplay_attempt_id": intent.gameplay_attempt_id,
            "launch_nonce": intent.launch_nonce,
        }
        for kind, value in intent_identities.items():
            if value in global_ids[kind]:
                raise EvidenceError(f"duplicate C4 {kind}: {value}")
            global_ids[kind].add(value)
        if activation is not None:
            attestation = history.attestation
            if attestation is None or history.activation_source not in {"normal", "reconciliation"}:
                raise EvidenceError(f"activated run has no broker process attestation: {attempt_id}")
            if (sum(event.event_type is EventType.BROKER_PROCESS_ATTESTED for event in normalized) != 1
                    or sum(event.event_type is EventType.PROCESS_LAUNCH_CONFIRMED for event in normalized) != 1):
                raise EvidenceError(f"activated run has an incomplete process lifecycle: {attempt_id}")
            attestation_event = next(event for event in normalized
                                     if event.event_type is EventType.BROKER_PROCESS_ATTESTED)
            if (attestation_event.process_ref != attestation.identity.process_ref
                    or attestation_event.job_ref != attestation.job_ref
                    or activation.activation_source != history.activation_source):
                raise EvidenceError(f"ledger activation differs from broker attestation: {attempt_id}")
            if (len(history.row_hashes) < 4 or not intent.reserved_run_id or not intent.gameplay_attempt_id
                    or not intent.launch_nonce):
                raise EvidenceError(f"activated run has incomplete ledger reconciliation references: {attempt_id}")
            identities = {
                "process_ref": attestation.identity.process_ref,
                "job_ref": attestation.job_ref,
            }
            for kind, value in identities.items():
                if value in global_ids[kind]:
                    raise EvidenceError(f"duplicate C4 {kind}: {value}")
                global_ids[kind].add(value)
        if intent.slot_id in ledger_attempts[execution_id]:
            raise EvidenceError(f"C4 slot has multiple launch attempts: {execution_id}:{intent.slot_id}")
        ledger_attempts[execution_id][intent.slot_id] = (history, tuple(normalized))

    gates_by_attempt: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in gate_records:
        execution_id = record.get("stage_execution_id")
        if execution_id not in stage_data:
            continue
        attempt_id = _text(record.get("attempt_id"), "launch gate attempt id")
        gates_by_attempt[attempt_id].append(record)
    outcomes_by_exec_slot: dict[tuple[str, int], list[tuple[CampaignEvent, Mapping[str, Any]]]] = defaultdict(list)
    for record in outcome_records:
        execution_id = record.get("stage_execution_id")
        if execution_id not in stage_data:
            continue
        try:
            event = CampaignEvent.from_wire(record["event"])
        except (KeyError, TypeError, ValueError) as exc:
            raise EvidenceError("invalid launched outcome record") from exc
        if event.event_type is EventType.SAFETY_FAILURE:
            required_remediations.add(f"{execution_id}.safety.slot-{event.slot_id:02d}")
        outcomes_by_exec_slot[(execution_id, event.slot_id)].append((event, record))
    closures = _load_closures(reader, required_remediations)

    activated_runs: list[dict[str, Any]] = []
    stage_history: list[dict[str, Any]] = []
    terminal_counts: Counter[str] = Counter()
    preflight_failures = 0
    for execution_id, data in stage_data.items():
        manifest: CampaignManifest = data["manifest"]
        manifest_hash = data["hash"]
        preflight = raw_preflight_by_exec[execution_id]
        preflight_failures += sum(event.event_type is EventType.PREFLIGHT_FAILED for event in preflight)
        slot_preflight: dict[int, list[CampaignEvent]] = defaultdict(list)
        for event in preflight:
            slot_preflight[event.slot_id].append(event)
        stage_outcomes = 0
        stage_successes = 0
        for slot in range(CAMPAIGN_SLOT_COUNT):
            prefix = f"stages/{execution_id}/runs/slot-{slot:02d}.json"
            history_item = ledger_attempts[execution_id].get(slot)
            ledger_slot_events = [] if history_item is None else list(history_item[1])
            terminal_records = outcomes_by_exec_slot.get((execution_id, slot), [])
            terminal_list = [item[0] for item in terminal_records]
            if len(terminal_list) > 1:
                raise EvidenceError(f"duplicate terminal outcome for {execution_id}:slot-{slot:02d}")
            if history_item is not None:
                history, raw_events = history_item
                gate_records_for_attempt = gates_by_attempt.get(history.attempt_id, [])
                if len(gate_records_for_attempt) != 1:
                    raise EvidenceError(f"launch gate stream reference missing or duplicated: {history.attempt_id}")
                gate = gate_records_for_attempt[0]
                if (gate.get("slot_id") != slot or gate.get("ledger_row_hashes") != list(history.row_hashes)
                        or gate.get("events") != [event.to_wire() for event in raw_events]):
                    raise EvidenceError(f"launch gate stream differs from durable ledger: {history.attempt_id}")
                preflight_attempts = [event for event in slot_preflight[slot]
                                      if event.event_type is EventType.ATTEMPT_PREFLIGHT]
                if (len(preflight_attempts) != 1 or preflight_attempts[0].attempt_id != history.attempt_id
                        or preflight_attempts[0].details.get("launch_nonce") != history.intent.launch_nonce):
                    raise EvidenceError(f"attempt preflight does not bind to durable launch intent: {history.attempt_id}")
                activation_count = sum(event.event_type in _ACTIVATED for event in raw_events)
                if activation_count == 1:
                    if len(terminal_list) != 1 or not reader.exists(prefix):
                        raise EvidenceError(f"activated run is missing a terminal artifact: {history.attempt_id}")
                    run = _read_run(reader, prefix, {**plan, "plan_hash": plan_hash}, backup_hash)
                    event = terminal_list[0]
                    outcome_record = terminal_records[0][1]
                    try:
                        manifest_outcome = CampaignEvent.from_wire(run["outcome"])
                    except (TypeError, ValueError) as exc:
                        raise EvidenceError(f"invalid terminal outcome in run manifest: {prefix}") from exc
                    if (event.event_type not in TERMINAL_OUTCOMES or manifest_outcome.to_wire() != event.to_wire()
                            or outcome_record.get("run_manifest") != prefix):
                        raise EvidenceError(f"run manifest outcome differs from launched outcome: {prefix}")
                    intent = history.intent
                    attestation = history.attestation
                    process_ref = attestation.identity.process_ref
                    broker = _mapping(run["broker"], f"run broker {prefix}")
                    spec = _mapping(run["run_spec"], f"run spec {prefix}")
                    if (run["campaign_id"] != plan["campaign_id"] or run["stage"] != "C4"
                            or run["stage_execution_id"] != execution_id or run["campaign_manifest_hash"] != manifest_hash
                            or run["slot_id"] != slot or run["attempt_id"] != intent.attempt_id
                            or run["reserved_run_id"] != intent.reserved_run_id
                            or run["gameplay_attempt_id"] != intent.gameplay_attempt_id
                            or run["activation_source"] != history.activation_source
                            or spec.get("reserved_run_id") != intent.reserved_run_id
                            or spec.get("gameplay_attempt_id") != intent.gameplay_attempt_id
                            or spec.get("process_ref") != process_ref or spec.get("job_ref") != attestation.job_ref
                            or broker.get("process_ref") != process_ref or broker.get("job_ref") != attestation.job_ref
                            or broker.get("broker_ref") != attestation.broker_ref
                            or broker.get("ledger_row_hashes") != list(history.row_hashes)):
                        raise EvidenceError(f"run manifest identity differs from durable lifecycle: {prefix}")
                    terminal_counts[event.event_type.value] += 1
                    stage_outcomes += 1
                    stage_successes += event.event_type is EventType.SUCCESS
                    activated_runs.append({
                        "artifact_id": prefix,
                        "sha256": reader.inventory[prefix],
                        "attempt_id_sha256": sha256_hex(intent.attempt_id.encode()),
                        "reserved_run_id_sha256": sha256_hex(intent.reserved_run_id.encode()),
                        "gameplay_attempt_id_sha256": sha256_hex(intent.gameplay_attempt_id.encode()),
                        "process_ref_sha256": sha256_hex(process_ref.encode()),
                        "job_ref_sha256": sha256_hex(attestation.job_ref.encode()),
                        "broker_ref_sha256": sha256_hex(attestation.broker_ref.encode()),
                        "ledger_row_hashes": list(history.row_hashes),
                    })
                elif terminal_list or reader.exists(prefix):
                    raise EvidenceError(f"inactive launch has terminal run evidence: {execution_id}:slot-{slot:02d}")
            elif terminal_list or reader.exists(prefix):
                raise EvidenceError(f"run artifact has no durable activated identity: {execution_id}:slot-{slot:02d}")
        # Reconstruct in canonical slot order, preserving each stream's lifecycle order.
        ordered_events = []
        for slot in range(CAMPAIGN_SLOT_COUNT):
            ordered_events.extend(slot_preflight[slot])
            history_item = ledger_attempts[execution_id].get(slot)
            if history_item is not None:
                ordered_events.extend(history_item[1])
            ordered_events.extend(event for event, _record in outcomes_by_exec_slot.get((execution_id, slot), ()))
        try:
            grouped = validate_campaign_events(
                ordered_events,
                expected_manifest_hash=manifest_hash,
                expected_slots=STAGE_POLICIES["C4"].slot_count,
            )
        except ValueError as exc:
            raise EvidenceError(f"C4 event lifecycle failed validation: {execution_id}: {exc}") from exc
        if any(event.event_type in {EventType.LAUNCH_GATE_FAILED, EventType.LAUNCH_UNCERTAIN}
               for items in grouped.values() for event in items):
            raise EvidenceError(f"C4 contains a pre-activation launch gate failure: {execution_id}")
        expected_state = (
            "preflight_blocked" if any(e.event_type is EventType.PREFLIGHT_FAILED for e in ordered_events)
            else "stage_passed" if len([e for e in ordered_events if e.event_type in TERMINAL_OUTCOMES]) == 20
            and stage_successes >= STAGE_POLICIES["C4"].promotion_floor
            else "stage_not_promoted"
        )
        if next(e for e in c4_executions if e["stage_execution_id"] == execution_id)["state"] != expected_state:
            raise EvidenceError(f"C4 runner state differs from ledger/artifact events: {execution_id}")
        stage_history.append({
            "stage_execution_id": execution_id,
            "campaign_manifest_sha256": manifest_hash,
            "activated": sum(e.event_type is EventType.FORMAL_RUN_ACTIVATED for e in ordered_events),
            "terminal_outcomes": stage_outcomes,
            "preflight_failures": sum(e.event_type is EventType.PREFLIGHT_FAILED for e in ordered_events),
            "state": expected_state,
        })

    for attempt_id, records in gates_by_attempt.items():
        if attempt_id not in all_attempt_ids or len(records) != 1:
            raise EvidenceError(f"duplicate launch gate record: {attempt_id}")
    activation_count = len(activated_runs)
    if activation_count != 20 or len(global_ids["reserved_run_id"]) != 20:
        raise EvidenceError(f"C4 requires 20 unique activated reserved-run ids; found {activation_count}/20")
    if len(global_ids["attempt_id"]) != 20 or len(global_ids["gameplay_attempt_id"]) != 20:
        raise EvidenceError("C4 attempt/gameplay identity cardinality must be 20")
    if len(global_ids["process_ref"]) != 20 or len(global_ids["job_ref"]) != 20:
        raise EvidenceError("C4 process/job lifecycle cardinality must be 20")
    if sum(terminal_counts.values()) != 20:
        raise EvidenceError("C4 activated outcome denominator must be 20")
    stage_history.sort(key=lambda entry: entry["stage_execution_id"])
    return {
        "activation_count": activation_count,
        "denominator": activation_count,
        "terminal_outcomes": {name: terminal_counts.get(name, 0) for name in (
            "SUCCESS", "GAMEPLAY_FAILURE", "SAFETY_FAILURE", "ARTIFACT_FAILURE"
        )},
        "successes": terminal_counts[EventType.SUCCESS.value],
        "preflight_failures": preflight_failures,
        "stage_history": stage_history,
    }, {"run_manifests": activated_runs, "required_remediations": required_remediations,
        "remediated_history": closures}


def _summary_formal_eligibility(
    reader: _Artifacts,
    plan: Mapping[str, Any],
    summary: Mapping[str, Any],
    backup_hash: str,
) -> bool:
    executions = summary["executions"]
    run_paths: list[str] = []
    runs_ready = True
    for raw in executions:
        execution = _mapping(raw, "summary execution")
        stage = execution.get("stage")
        execution_id = _id(execution.get("stage_execution_id"), "stage execution id")
        policy = STAGE_POLICIES.get(stage)
        if policy is None:
            raise EvidenceError(f"unknown campaign stage in summary: {stage}")
        for slot in range(policy.slot_count):
            path = f"stages/{execution_id}/runs/slot-{slot:02d}.json"
            if reader.exists(path):
                run = _read_run(reader, path, {**plan, "plan_hash": summary["plan_hash"]}, backup_hash)
                run_paths.append(path)
                runs_ready &= run["save_hashes_complete"] is True
    in_progress = summary.get("in_progress_execution_ids")
    evidence = _mapping(summary.get("save_evidence"), "campaign save evidence")
    restore = _mapping(summary.get("restore_verdict"), "restore verdict")
    eligible = (
        plan.get("development_only") is False and bool(executions) and runs_ready
        and isinstance(in_progress, list) and not in_progress
        and evidence.get("backup_sha256") == backup_hash
        and evidence.get("cloud_sync_attested") is True and restore.get("status") == "PASS"
    )
    expected_runs = run_paths if eligible else []
    if summary.get("formal_evidence_eligible") is not eligible:
        raise EvidenceError("summary formal evidence flag differs from independently verified save evidence")
    if summary.get("formal_evidence_run_manifests") != expected_runs:
        raise EvidenceError("summary formal run manifest list differs from independently enumerated artifacts")
    return eligible


def build_goal_evidence(
    artifacts: ArtifactStore,
    store: DurableLaunchStore,
    *,
    formal_c4_root: str | os.PathLike[str] | None = None,
) -> GoalEvidence:
    """Recompute C4 outcomes from campaign artifacts and durable ledger history.

    Without an explicit formal C4 root, only a synthetic development result is
    returned. The output always labels the observed 16/20 criterion separately
    from RNG control and trial separation.
    """
    reader = _Artifacts(artifacts)
    storage = _storage_attestation(store)
    try:
        plan = _mapping(reader.json("campaign/plan.json"), "campaign plan")
        summary = _mapping(reader.json("campaign/summary.json"), "campaign summary")
        plan_id = _id(plan.get("campaign_id"), "campaign id")
        plan_hash = canonical_hash(dict(plan))
        if (summary.get("campaign_id") != plan_id or summary.get("plan_hash") != plan_hash
                or plan.get("campaign_run_mode") != "formal_single_attempt"
                or plan.get("ui_restart_enabled") is not False):
            raise EvidenceError("campaign plan and summary binding is invalid")
        development_only = plan.get("development_only") is True
        if (plan.get("mode") not in {"synthetic", "formal"}
                or plan.get("development_only") is not development_only
                or development_only != (plan.get("mode") == "synthetic")
                or plan.get("formal_campaign_eligible") is not (not development_only)):
            raise EvidenceError("campaign eligibility claims are inconsistent")
        if summary.get("development_only") is not development_only:
            raise EvidenceError("summary development status differs from campaign plan")
        if summary.get("formal_campaign_eligible") is not (not development_only):
            raise EvidenceError("summary formal campaign status differs from campaign plan")
        if summary.get("in_progress_execution_ids") != []:
            raise EvidenceError("campaign has unresolved in-progress executions")

        formal_root = None if formal_c4_root is None else Path(formal_c4_root).resolve()
        if formal_root is None:
            if not development_only:
                raise EvidenceError("formal campaign requires the explicit formal C4 root")
        else:
            if formal_root != Path(artifacts.root).resolve():
                raise EvidenceError("formal C4 root does not match the artifact root")
            if development_only:
                raise EvidenceError("synthetic fixtures cannot be loaded as formal C4 evidence")

        _reject_claims(plan)
        _reject_claims(summary)
        chain = reader.json("campaign/goal_release_chain.json")
        _reject_claims(chain)
        chain_digest, chain_refs = _validate_chain(chain, plan)
        save = _validate_save(reader, plan, summary)
        metrics, identity = _check_c4_events(reader, store, summary, plan, plan_hash, save["backup_sha256"])
        closures = identity["remediated_history"]
        formal_evidence_eligible = _summary_formal_eligibility(reader, plan, summary, save["backup_sha256"])
        promotion_floor = STAGE_POLICIES["C4"].promotion_floor
        promotion_eligible = metrics["successes"] >= promotion_floor
        if not promotion_eligible:
            raise EvidenceError(f"C4 observed campaign criterion requires 16/20; found {metrics['successes']}/20")
        if formal_root is not None and not formal_evidence_eligible:
            raise EvidenceError("formal summary is not independently eligible for goal release")

        intervals = wilson_score_interval(metrics["successes"], metrics["denominator"])
        goal_release_eligible = formal_root is not None and not development_only
        source_artifacts = [
            {"logical_id": name, "sha256": digest}
            for name, digest in sorted(reader.inventory.items())
        ]
        run_refs = sorted(identity["run_manifests"], key=lambda entry: entry["artifact_id"])
        manifest = {
            "schema_version": EVIDENCE_SCHEMA,
            "campaign_id": plan_id,
            "stage": "C4",
            "campaign_plan_sha256": plan_hash,
            "campaign_summary_sha256": reader.inventory["campaign/summary.json"],
            "goal_release_chain_sha256": chain_digest,
            "chain_nodes": chain_refs,
            "campaign_manifest_sha256": sorted({
                entry["campaign_manifest_sha256"] for entry in metrics["stage_history"]
            }),
            "source_artifacts": source_artifacts,
            "run_manifests": run_refs,
            "save": {
                "backup_sha256": save["backup_sha256"],
                "canonical_sha256": save["canonical_sha256"],
                "restore_status": "PASS",
                "cloud_sync_attested": True,
            },
            "ledger_attestation": storage,
            "remediated_history": closures,
            "development_only": development_only,
            "goal_release_eligible": goal_release_eligible,
            "rng_control": "uncontrolled",
            "trial_separation": "unique_run_id_separate_process",
        }
        report = {
            "schema_version": EVIDENCE_SCHEMA,
            "campaign_id": plan_id,
            "stage": "C4",
            "activation_count": metrics["activation_count"],
            "denominator": metrics["denominator"],
            "terminal_outcomes": metrics["terminal_outcomes"],
            "successes": metrics["successes"],
            "observed_success_rate": metrics["successes"] / metrics["denominator"],
            "promotion_floor": promotion_floor,
            "promotion_eligible": promotion_eligible,
            "wilson_95_interval": None if intervals is None else list(intervals),
            "preflight_failures": metrics["preflight_failures"],
            "stage_history": metrics["stage_history"],
            "development_only": development_only,
            "goal_release_eligible": goal_release_eligible,
            "rng_control": "uncontrolled",
            "trial_separation": "unique_run_id_separate_process",
        }
        _safe_output(manifest)
        _safe_output(report)
        if report["rng_control"] == report["trial_separation"]:
            raise EvidenceError("RNG control and trial separation must remain separate fields")
        return GoalEvidence(manifest, report)
    except EvidenceError:
        raise
    except (OSError, KeyError, TypeError, ValueError, AssertionError) as exc:
        raise EvidenceError(f"campaign evidence validation failed: {exc}") from exc


def _bundle_entries(evidence: GoalEvidence) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_data = canonical_json_bytes(evidence.manifest)
    report_data = canonical_json_bytes(evidence.report)
    sources = evidence.manifest["source_artifacts"]
    source_objects = [
        {"object_id": f"bundle/objects/source-{index:04d}.json", "logical_id": item["logical_id"],
         "sha256": item["sha256"]}
        for index, item in enumerate(sources)
    ]
    objects = [
        {"object_id": "bundle/objects/manifest.json", "sha256": sha256_hex(manifest_data)},
        {"object_id": "bundle/objects/report.json", "sha256": sha256_hex(report_data)},
        *[{"object_id": item["object_id"], "sha256": canonical_hash({
            "logical_id": item["logical_id"], "sha256": item["sha256"]
        })} for item in source_objects],
    ]
    index = {
        "schema_version": BUNDLE_SCHEMA,
        "manifest_sha256": canonical_hash(evidence.manifest),
        "report_sha256": canonical_hash(evidence.report),
        "source_artifacts_sha256": canonical_hash(sources),
        "objects": objects,
    }
    return index, {"source_objects": source_objects, "manifest": evidence.manifest, "report": evidence.report}


def verify_evidence_bundle(backup_directory: str | os.PathLike[str], empty_root: str | os.PathLike[str]) -> GoalEvidence:
    """Restore a sanitized bundle into an empty root and verify every object.

    The manifest is reconstructed from its bundled source references and hashes;
    missing or changed objects stop the restore.
    """
    backup_path, restore_path = Path(backup_directory), Path(empty_root)
    if not backup_path.is_dir():
        raise EvidenceError("evidence backup store is missing")
    if restore_path.exists() and any(restore_path.iterdir()):
        raise EvidenceError("evidence restore root must be empty")
    backup = ArtifactStore(backup_path)
    try:
        index = _mapping(backup.read_json("bundle/index.json"), "evidence bundle index")
        if set(index) != {"schema_version", "manifest_sha256", "report_sha256", "source_artifacts_sha256", "objects"}:
            raise EvidenceError("evidence bundle index has unexpected fields")
        if index["schema_version"] != BUNDLE_SCHEMA or not isinstance(index["objects"], list):
            raise EvidenceError("evidence bundle index schema is invalid")
        restore = ArtifactStore(restore_path)
        object_ids = set()
        for record in index["objects"]:
            record = _mapping(record, "evidence bundle object"); object_id = _text(record.get("object_id"), "object id")
            if object_id in object_ids or not object_id.startswith("bundle/objects/"):
                raise EvidenceError("evidence bundle object id is invalid or duplicated")
            object_ids.add(object_id)
            data = backup.read(object_id)
            if sha256_hex(data) != _sha(record.get("sha256"), "bundle object hash"):
                raise EvidenceError(f"evidence bundle object is missing or altered: {object_id}")
            restore.put_immutable(object_id, data)
        manifest = _mapping(restore.read_json("bundle/objects/manifest.json"), "restored evidence manifest")
        report = _mapping(restore.read_json("bundle/objects/report.json"), "restored evidence report")
        if canonical_hash(manifest) != index["manifest_sha256"] or canonical_hash(report) != index["report_sha256"]:
            raise EvidenceError("restored manifest/report hash mismatch")
        source_objects = []
        for object_id in sorted(object_ids):
            if not object_id.endswith(".json") or "source-" not in object_id:
                continue
            value = _mapping(restore.read_json(object_id), "restored source reference")
            if set(value) != {"logical_id", "sha256"}:
                raise EvidenceError("restored source reference has unexpected fields")
            source_objects.append({"logical_id": _text(value["logical_id"], "source logical id"),
                                   "sha256": _sha(value["sha256"], "source SHA-256")})
        if (source_objects != manifest.get("source_artifacts")
                or canonical_hash(source_objects) != index["source_artifacts_sha256"]
                or report.get("campaign_id") != manifest.get("campaign_id")):
            raise EvidenceError("restored source inventory does not reconstruct the manifest")
        _safe_output(manifest)
        _safe_output(report)
        return GoalEvidence(dict(manifest), dict(report))
    except EvidenceError:
        raise
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise EvidenceError(f"evidence bundle restore failed: {exc}") from exc


def write_evidence_bundle(
    evidence: GoalEvidence,
    source_artifacts: ArtifactStore,
    store: DurableLaunchStore,
    primary_directory: str | os.PathLike[str],
    backup_directory: str | os.PathLike[str],
    *,
    formal_c4_root: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Write safe manifest/report copies to sibling temp stores and verify backup restore.

    Evidence is rebuilt from the input APIs immediately before writing, so a
    changed source cannot reuse a previously cached PASS result.
    """
    rebuilt = build_goal_evidence(source_artifacts, store, formal_c4_root=formal_c4_root)
    if (canonical_hash(rebuilt.manifest) != canonical_hash(evidence.manifest)
            or canonical_hash(rebuilt.report) != canonical_hash(evidence.report)):
        raise EvidenceError("source artifacts changed after goal evidence was built")
    primary, backup = Path(primary_directory).resolve(), Path(backup_directory).resolve()
    if primary == backup or primary.parent != backup.parent:
        raise EvidenceError("primary and backup output stores must be distinct siblings")
    temp_root = Path(tempfile.gettempdir()).resolve()
    if not primary.is_relative_to(temp_root) or not backup.is_relative_to(temp_root):
        raise EvidenceError("goal evidence output stores must be under the temporary directory")
    if primary.exists() or backup.exists():
        raise EvidenceError("goal evidence output stores must not already exist")
    if evidence.manifest.get("schema_version") != EVIDENCE_SCHEMA or evidence.report.get("schema_version") != EVIDENCE_SCHEMA:
        raise EvidenceError("evidence result schema is invalid")
    index, contents = _bundle_entries(evidence)
    primary.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="survivors-evidence-", dir=primary.parent) as temporary:
        staging = Path(temporary)
        staged_primary = staging / "primary"
        staged_backup = staging / "backup"
        for target in (staged_primary, staged_backup):
            output = ArtifactStore(target)
            output.put_json("manifest.json", contents["manifest"])
            output.put_json("report.json", contents["report"])
            output.put_json("bundle/objects/manifest.json", contents["manifest"])
            output.put_json("bundle/objects/report.json", contents["report"])
            for item in contents["source_objects"]:
                output.put_json(item["object_id"], {"logical_id": item["logical_id"], "sha256": item["sha256"]})
            output.put_json("bundle/index.json", index)
        restored_primary = verify_evidence_bundle(staged_primary, staging / "empty-primary")
        restored_backup = verify_evidence_bundle(staged_backup, staging / "empty-backup")
        if (canonical_hash(restored_primary.manifest) != canonical_hash(evidence.manifest)
                or canonical_hash(restored_primary.report) != canonical_hash(evidence.report)
                or canonical_hash(restored_backup.manifest) != canonical_hash(evidence.manifest)
                or canonical_hash(restored_backup.report) != canonical_hash(evidence.report)):
            raise EvidenceError("empty-root restore changed generated goal evidence")
        staged_primary.replace(primary)
        try:
            staged_backup.replace(backup)
        except OSError:
            if primary.exists():
                import shutil
                shutil.rmtree(primary)
            raise
    return {
        "manifest_sha256": canonical_hash(evidence.manifest),
        "report_sha256": canonical_hash(evidence.report),
        "restored": True,
    }
