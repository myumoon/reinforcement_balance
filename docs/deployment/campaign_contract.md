# Survivors live canary campaign contract

## Scope and identity

This package defines the pure Python contract for a 20-slot Survivors campaign. It does not launch processes, perform controller I/O, or persist a durable ledger. Those remain the responsibility of later phases.

Every manifest uses schema `survivors.campaign.v1`, a fixed `stage` (`C0`–`C4`), and an `expected_slots` value matching that stage's frozen policy. It also requires `rng_control: "uncontrolled"` and `trial_separation: "unique_run_id_separate_process"` as separate fields. A distinct run ID and process do not establish statistical independence; seed values and independence claims are rejected across manifest, prerequisite, and event details.

The event path is:

```text
FORMAL_SLOT_RESERVED
  -> ATTEMPT_PREFLIGHT(attempt_id)
  -> PREFLIGHT_FAILED
  -> LAUNCH_INTENT_COMMITTED(reserved_run_id, gameplay_attempt_id, launch_nonce)
  -> BROKER_PROCESS_ATTESTED(process_ref, job_ref)
  -> PROCESS_LAUNCH_CONFIRMED
  -> FORMAL_RUN_ACTIVATED(activation_source=normal|reconciliation)
  -> SUCCESS | GAMEPLAY_FAILURE | SAFETY_FAILURE | ARTIFACT_FAILURE
```

`PREFLIGHT_FAILED`, `LAUNCH_GATE_FAILED`, and `LAUNCH_UNCERTAIN` end that slot before activation. A slot can have one preflight attempt and at most one launch intent. IDs are unique within their identity type. A slot cannot be reserved again, and a terminal event cannot be overwritten. Each activation is bound to one attempt, one attested process/job, and one gameplay attempt.

## Formal prerequisites

Formal prerequisites contain exactly these evidence keys: `exact_runtime`, `target`, `save`, `build`, `hardware`, `perception_final`, `shadow`, `replay`, `input_safety`, and `restore`. Each has a lowercase SHA-256 hash, a parent hash, and `PASS` status. All parents must match the manifest's expected parent hash.

The bundle also requires verified cloud sync, a backup hash, and pre-save and post-save contract hashes. Missing, stale, mixed-parent, non-PASS, development-only, or unknown fields are rejected. A synthetic manifest is always `development_only: true` and cannot contain formal prerequisite evidence. A formal manifest requires a validated non-development prerequisite bundle.

This phase validates evidence hash shape and shared parent binding. Comparing those evidence hashes with independently observed expected values belongs to the formal issuance flow in 06-04; this pure contract does not issue a formal campaign from synthetic fixtures.

## Frozen stage policies

This implementation freezes these values because the supplied phase contract named the fields but did not provide numeric values. Changing them requires an explicit contract revision.

| Stage | Duration | Slots | Promotion floor (successes) |
|---|---:|---:|---:|
| C0 | 30 minutes | 2 | 2 |
| C1 | 60 minutes | 4 | 3 |
| C2 | 120 minutes | 8 | 6 |
| C3 | 240 minutes | 16 | 12 |
| C4 | 480 minutes | 20 | 16 |

## Denominator and report rules

The denominator is the number of `FORMAL_RUN_ACTIVATED` events. Success rate is `SUCCESS / activated runs`; preflight, launch-gate, and uncertain-launch failures are excluded and block stage promotion. They do not create replacements. Gameplay, safety, and artifact failures after activation remain in the denominator and cannot be replaced.

Every planned slot must end in a pre-activation failure or an activated terminal outcome. Missing slots, reserved-only slots, in-progress preflight, and activated slots without a terminal outcome appear in `incomplete_slot_ids`; any such slot blocks the stage and makes promotion ineligible. An activated slot without a terminal outcome still counts in the denominator.

The report includes the observed rate and a two-sided 95% Wilson score interval. For 16 successes from 20 activations, it reports 0.8 with an interval of approximately 0.583–0.919. It makes no population success probability claim. Fifteen of twenty successes report 0.75 and do not meet C4's frozen floor of 16.

Reports include `support_outside_ui`, a failure taxonomy, blocked/superseded campaign IDs, and the manifest's canonical hash. Report generation requires an `event_manifest_hash` matching that manifest; the report wire also records `prerequisite_parent_hash` when present. Synthetic reports carry `development_only: true` and `formal_parent_eligible: false`; they cannot serve as a formal C0 parent.

## Canonical fixtures

`Tools/Deployment/tests/campaign/fixtures/golden_campaigns.jsonl` contains six canonical scenario envelopes: normal 16/20, normal 15/20, reconciliation activation, preflight failure, uncertain launch, and duplicate process. Each row pins its stage, event-to-manifest binding, and expected event, JSONL, manifest, and report hashes (the duplicate-process report hash is null because validation rejects it). The fixtures use only synthetic values and are parsed through the same strict event and manifest schemas as generated reports.
