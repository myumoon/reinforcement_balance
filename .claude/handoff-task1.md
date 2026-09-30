# 06-04 サブタスク1 handoff（campaign_runner / save_lifecycle）

サブタスク2（`run_survivors_campaign.py` / `mad_forest_canary_v1.yaml` / `docs/deployment/live_canary_runbook.md`）はこのファイルの API だけに依存してください。

## `Tools/Deployment/survivors/campaign/campaign_runner.py` 公開 API（`grep -n "^def \|^class "` の実測値）

```
54:class RunnerError(RuntimeError):
61:class PrerequisiteMismatch(RunnerError):
68:class PlanMismatch(RunnerError):
75:class StageOrderError(RunnerError):
82:class CampaignBlocked(RunnerError):
89:class UnreconciledLaunch(RunnerError):
96:class ArtifactConflict(RunnerError):
103:class RunnerState(StrEnum):
140:class ArtifactStore:
250:def threshold_parent_hash(prerequisite_hashes: Mapping[str, str]) -> str:
261:class CampaignPlan:
361:class SlotIdentity(NamedTuple):
372:def slot_identity(stage_execution_id: str, slot_id: int) -> SlotIdentity:
382:class RunSpec:
411:class RunResult:
479:class ReleaseObservation:
498:class StageResult:
511:def result_failures(spec: RunSpec, result: RunResult) -> list[tuple[EventType, str]]:
531:def outcome_event(slot_id, manifest_hash, result, failures, release) -> CampaignEvent
562:def validate_parent(plan: CampaignPlan, stage: str, record: Mapping[str, Any], report: Mapping[str, Any]) -> None:
617:class CampaignRunner:
```
（`_digest` / `_json_value` / `_Execution` は private）

定数: `CAMPAIGN_RUN_MODE = "formal_single_attempt"`、`STAGES = ("C0","C1","C2","C3","C4")`（06-02 `STAGE_POLICIES` の順）、
stream 名 `PREFLIGHT_STREAM="preflight_attempts"` / `GATE_STREAM="launch_gates"` / `OUTCOME_STREAM="launched_outcomes"` / `CHAIN_STREAM="campaign_chain"`。

### 主要シグネチャ

- `CampaignPlan(campaign_id, mode("formal"|"synthetic"), executable_path, executable_hash, build_hash, config_hash, argv, canonical_save_hash, thresholds: Mapping[str,float], threshold_parent_hash, prerequisites=None, prerequisite_parent_hash=None, grace_seconds=120)`
  - `campaign_id` は `[A-Za-z0-9][A-Za-z0-9._-]*`（file 名に使う）。`argv[0] == executable_path`（絶対 path）。
  - synthetic は prerequisites 無し・`development_only=True`・`formal_campaign_eligible=False`。
  - formal は `threshold_parent_hash == threshold_parent_hash(prerequisites["hashes"])` が必須（target / perception_final evidence に束縛）。
  - `.to_wire()` / `.plan_hash` / `.development_only`。
- `CampaignRunner(plan, *, artifacts, ledger, broker, probe, controller, helper, target, operator, save, validator=campaign_schema)`
  - `begin()` : prerequisite / threshold parent / canonical save 検証（不一致は `PrerequisiteMismatch`、何も書かない）→ formal なら development_only port を拒否 → plan を write-once 保存（差分は `PlanMismatch`）→ 元 save backup → `operator.checkpoint("cloud_sync_disabled", ctx)` → attestation 記録。
  - `run_stage(stage, *, max_slots=None) -> StageResult` : 開始 / 再開。`max_slots` 到達で `RunnerState.STOPPED`（再度呼ぶと続きから）。完了済み stage は `StageOrderError`（`PREFLIGHT_BLOCKED` 後だけ新 execution id `x{n+1}` で再実行可）。下位 stage 未 pass は `StageOrderError`。uncertain / supersede 後は `CampaignBlocked`。未 reconcile intent があれば `UnreconciledLaunch`。
  - `finish() -> Mapping` : 元 save 復元（`restore_original`）→ `campaign/summary.json` write-once。`formal_evidence_eligible` は formal かつ backup・cloud sync・全 run の pre/post hash・restore PASS・in-progress 無しのときだけ True。
  - `supersede(successor_campaign_id, reason)` : 以後の stage を拒否。
  - `state: RunnerState`（IDLE / READY / PREREQUISITE_MISMATCH / STOPPED / STAGE_PASSED / STAGE_NOT_PROMOTED / PREFLIGHT_BLOCKED / LAUNCH_GATE_BLOCKED / CAMPAIGN_BLOCKED / SUPERSEDED / FINISHED）。
- `StageResult(stage, stage_execution_id, state, report_hash, slot_events: Mapping[int, tuple[str,...]])`
- `ArtifactStore(root)` : `exists/read/read_json/put_immutable/put_json/append/stream`。write-once file（別内容は `ArtifactConflict`）と append-only JSONL stream（torn 行は ValueError）。

### port 契約（duck typing）

| port | 呼び出し | 備考 |
|---|---|---|
| ledger | 06-03 `DurableLaunchStore`（`commit_intent` / `activate` / `reconcile` / `commit_uncertain` / `history` / `attempt_ids` / `campaign_events`） | runner は SQLite を直接触らない |
| broker | `broker(message: dict) -> dict` | production は `lambda m: request_broker(pipe, m, broker_pid=pid, timeout_s=...)`。OSError/ValueError は「応答なし」として ledger reconcile で分類 |
| probe | `probe(ProcessIdentity) -> ProbeResult` | production は `win32_launch_broker.probe_identity` |
| controller | `start(RunSpec) -> handle`、handle に `controller_pid` / `helper_pid` / `wait(timeout_s) -> RunResult | None` / `terminate()` | None は timeout |
| helper | `confirm_release(helper_pid, spec_wire) -> ReleaseObservation` | external observer。未確認は SAFETY_FAILURE |
| target | `stopped() -> bool`（game/launcher 停止）、`stop(spec_wire) -> bool` | save lifecycle の `processes_stopped` にも `target.stopped` を渡す |
| operator | `checkpoint(name, context) -> bool` | name は `cloud_sync_disabled` / `arm` / `focus` / `manual_start` |
| 全 port | `development_only` 属性（無ければ True 扱い） | formal plan は True の port を拒否 |

### artifact layout（artifact root 配下）

- `campaign/plan.json`（write-once）、`campaign/summary.json`、`campaign/superseded.json`
- `stages/<campaign_id>.<stage>.x<n>/manifest.json`（06-02 manifest wire + plan_hash / parent / blocked ids）、`report.json`（06-02 reporter の `to_wire()` そのまま）、`execution.json`（state / report_hash / threshold parent）、`runs/slot-NN.json`（per-run manifest）
- `save/original_backup.bin|json`、`save/canonical.bin`、`save/cloud_sync_attestation.json`、`save/pre|post/<attempt_id>.json`、`save/restore_verdict.json`
- `streams/preflight_attempts.jsonl` / `launch_gates.jsonl` / `launched_outcomes.jsonl` / `campaign_chain.jsonl` / `save_restore_attempts.jsonl`

## `Tools/Deployment/survivors/campaign/save_lifecycle.py` 公開 API（実測）

```
25:class SaveLifecycleError(RuntimeError):
40:def atomic_replace(path: str | os.PathLike[str], data: bytes, expected_hash: str) -> None:
66:class SaveLifecycle:
```
`SaveLifecycle(save_path, artifacts, *, processes_stopped)` の method: `backup_original()` / `register_canonical(data)` / `attest_cloud_sync(attestation)` / `install_canonical(attempt_id, expected_hash)` / `post_run_hash(attempt_id)` / `slot_hash(kind, attempt_id)` / `restore_original()` / `evidence()`。
CLI は `begin()` 前に `save.register_canonical(bytes)` を呼び、hash を `CampaignPlan.canonical_save_hash` に入れる。

## テスト一覧

`Tools/Deployment/tests/campaign/test_campaign_runner.py`（30 件、parametrize 込み）:
test_effect_order_preflight_launch_activate_terminal, test_controller_receives_only_ledger_confirmed_identity,
test_report_is_reporter_output_and_streams_are_separate, test_run_manifest_collects_pids_release_refs_and_save_hashes,
test_prerequisite_mismatch_touches_nothing[3], test_preflight_failure_blocks_stage_without_reserved_identity,
test_broker_timeout_after_confirm_uses_reconciliation_activation, test_launch_failures_are_consumed_without_replacement[6],
test_lower_stage_must_pass_before_upper_stage, test_passed_lower_stage_gates_next_stage, test_validate_parent_rejections,
test_plan_changes_after_live_results_are_rejected, test_completed_slots_and_reserved_identities_cannot_be_reused,
test_outcome_overwrite_is_rejected, test_resume_after_crash_post_activation_consumes_failure,
test_resume_with_pending_intent_reconciles_without_relaunch, test_unreconciled_foreign_intent_blocks_next_slot,
test_timeout_terminates_controller_then_confirms_helper_release[2], test_checkpoints_gate_controller_and_menu_input_is_safety_failure,
test_outcome_event_conversion, test_formal_plan_rejects_development_ports, test_formal_evidence_requires_every_save_record

`Tools/Deployment/tests/campaign/test_save_lifecycle.py`（10 件）:
test_backup_is_stored_and_not_retaken_on_resume, test_backup_refuses_while_game_running_or_missing,
test_install_requires_backup_and_cloud_sync_first, test_install_is_verified_atomic_and_records_pre_hash,
test_install_refuses_while_running_or_on_hash_mismatch, test_atomic_replace_temp_verification_failure_keeps_original,
test_post_run_hash_requires_stop_and_is_write_once, test_restore_original_passes_and_is_idempotent,
test_restore_failure_is_recorded_without_verdict, test_canonical_registration_is_write_once

テスト用 fake（FakeBroker / FakeController / FakeHelper / FakeTarget / FakeOperator / RecordingStore / RecordingSave / LedgerProxy）は
`test_campaign_runner.py` にあり、CLI の dry-run test から import して再利用できます（FakeBroker は実 ledger 上で 06-03 protocol を再現）。

## 設計上の注意（サブタスク2向け）

- 06-02 `STAGE_POLICIES` は C0=1800s×2, C1=3600s×4, C2=7200s×8, C3=14400s×16, C4=28800s×20。plan 本文の「C0 60秒×3 …」とは異なるが、runner は 06-02 を唯一の正として使う（変更禁止）。
- slot の ledger 行は ATTEMPT_PREFLIGHT の `details.launch_nonce` と一致するときだけ自分のものとして扱う（同じ campaign id を別 artifact root で走らせても他人の intent を採用しない）。
- broker response は分類に使わない（confirmed でも ledger attestation と process_ref / job_ref が一致したときだけ normal activation）。timeout は ledger reconcile で CONFIRMED なら `activation_source="reconciliation"`、生存を証明できない confirmed は LAUNCH_UNCERTAIN（campaign block）。
- Windows 以外では `DurableLaunchStore` が開けないので runner test は skip（06-03 と同じ方針）。

## 実行した検証コマンド

（下に追記）
