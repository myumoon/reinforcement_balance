"""campaign runner の effect 順序・state 写像・fail-closed 境界を検証します。

06-03 の実 DurableLaunchStore と 06-02 の実 schema/reporter を使い、broker・controller・helper・
target・operator は effect を記録する fake にします。preflight → intent → launch → activation →
run → terminal の順序、broker timeout / orphan / uncertain の扱い、slot 再利用の拒否、
save 証跡と development-only 表示を固定します。
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest
from reinbalance_survivors_contracts.canonical_json import canonical_hash

from survivors.campaign import campaign_schema
from survivors.campaign.campaign_report import generate_campaign_report
from survivors.campaign.campaign_runner import (
    ArtifactConflict,
    ArtifactStore,
    CAMPAIGN_RUN_MODE,
    CampaignBlocked,
    CampaignPlan,
    CampaignRunner,
    PlanMismatch,
    PrerequisiteMismatch,
    ReleaseObservation,
    RunnerError,
    RunnerState,
    RunResult,
    RunSpec,
    StageOrderError,
    UnreconciledLaunch,
    outcome_event,
    result_failures,
    slot_identity,
    threshold_parent_hash,
    validate_parent,
)
from survivors.campaign.campaign_schema import (
    REQUIRED_PREREQUISITES,
    CampaignEvent,
    CampaignManifest,
    EventType,
)
from survivors.campaign.durable_launch_store import (
    DurableLaunchStore,
    LaunchIntent,
    ProbeResult,
    ProcessAttestation,
    ProcessIdentity,
    UnsupportedStorageError,
    new_launch_nonce,
)
from survivors.campaign.save_lifecycle import SaveLifecycle, SaveLifecycleError
from survivors.campaign.win32_launch_broker import parse_launch_request

ORIGINAL = b"original-save"
CANONICAL = b"canonical-save"
CANONICAL_HASH = hashlib.sha256(CANONICAL).hexdigest()
EXE = sys.executable
FORMAL_PARENT = "f" * 64
FORMAL_HASHES = {name: hashlib.sha256(name.encode()).hexdigest() for name in REQUIRED_PREREQUISITES}
# 同じ ledger を共有する複数 runner でも fake process identity が衝突しないよう test 全体で採番します。
_PROCESS_NUMBERS = itertools.count(1)


class Crash(BaseException):
    """runner process の強制終了を再現する例外です。

    runner の `except Exception` を通り抜けて呼び出し元まで届きます。
    """


def _plan(**overrides) -> CampaignPlan:
    """synthetic plan を作ります。

    overrides で field を差し替えます。
    """
    values = dict(
        campaign_id="canary-dev", mode="synthetic", executable_path=EXE, executable_hash="b" * 64,
        build_hash="c" * 64, config_hash="d" * 64, argv=(EXE, "-c", "pass"), canonical_save_hash=CANONICAL_HASH,
        thresholds={"perception_recall": 0.9}, threshold_parent_hash="e" * 64, grace_seconds=30,
    )
    values.update(overrides)
    return CampaignPlan(**values)


def _formal_prerequisites(parent: str = FORMAL_PARENT) -> dict:
    """formal prerequisite wire を作ります。

    parent を変えると stale parent を再現できます。
    """
    return {
        "hashes": dict(FORMAL_HASHES),
        "parents": {name: parent for name in REQUIRED_PREREQUISITES},
        "statuses": {name: "PASS" for name in REQUIRED_PREREQUISITES},
        "cloud_sync_status": "verified",
        "backup_hash": "1" * 64,
        "pre_save_contract_hash": "2" * 64,
        "post_save_contract_hash": "3" * 64,
        "development_only": False,
    }


def _formal_plan(**overrides) -> CampaignPlan:
    """formal plan を作ります。

    threshold parent は prerequisite の target/perception_final から導きます。
    """
    values = dict(
        campaign_id="canary-formal", mode="formal", prerequisites=_formal_prerequisites(),
        prerequisite_parent_hash=FORMAL_PARENT, threshold_parent_hash=threshold_parent_hash(FORMAL_HASHES),
    )
    values.update(overrides)
    return _plan(**values)


def _result(spec: RunSpec, **overrides) -> RunResult:
    """confirmed success の run 結果を作ります。

    overrides で death や menu input などを再現します。
    """
    values = dict(
        gameplay_attempt_id=spec.gameplay_attempt_id, gameplay_entries=1, target_success="confirmed", died=False,
        level=21, gems=340, kills=812, choices=20, unknown_frames=3, fallback_decisions=1,
        latency_ms={"p50": 9.5, "p95": 14.0}, menu_inputs_sent=0, telemetry_refs=("telemetry.jsonl",),
        video_refs=("run.mp4",), lease_audit_ref="lease_audit.jsonl",
    )
    values.update(overrides)
    return RunResult(**values)


class RecordingStore(ArtifactStore):
    """書き込みを effect として記録する artifact store です。

    stream 追記と write-once 保存の順序を観測します。
    """

    def __init__(self, root, effects):
        """root と effect list を受け取ります。"""
        super().__init__(root)
        self.effects = effects

    def put_immutable(self, name, data):
        """write-once 保存を記録します。"""
        self.effects.append(f"put:{name}")
        return super().put_immutable(name, data)

    def append(self, stream, records):
        """stream 追記を記録します。"""
        self.effects.append(f"stream.{stream}")
        super().append(stream, records)


class RecordingSave(SaveLifecycle):
    """save 操作を effect として記録する SaveLifecycle です。

    実 file への backup・差し替え・復元はそのまま行います。
    """

    def __init__(self, *args, effects, **kwargs):
        """effect list を受け取ります。"""
        super().__init__(*args, **kwargs)
        self.effects = effects

    def backup_original(self):
        """backup を記録します。"""
        self.effects.append("save.backup_original")
        return super().backup_original()

    def attest_cloud_sync(self, attestation):
        """cloud sync attestation を記録します。"""
        self.effects.append("save.attest_cloud_sync")
        return super().attest_cloud_sync(attestation)

    def install_canonical(self, attempt_id, expected_hash):
        """canonical 差し替えを記録します。"""
        self.effects.append("save.install_canonical")
        return super().install_canonical(attempt_id, expected_hash)

    def post_run_hash(self, attempt_id):
        """post hash を記録します。"""
        self.effects.append("save.post_run_hash")
        return super().post_run_hash(attempt_id)

    def restore_original(self):
        """復元を記録します。"""
        self.effects.append("save.restore_original")
        return super().restore_original()


class LedgerProxy:
    """実 DurableLaunchStore の mutation 呼び出しを記録します。

    読み取りはそのまま委譲します。
    """

    def __init__(self, store, effects):
        """store と effect list を受け取ります。"""
        self._store = store
        self._effects = effects

    def __getattr__(self, name):
        """mutation method だけ記録付きで返します。"""
        attribute = getattr(self._store, name)
        if name not in {"commit_intent", "activate", "reconcile", "commit_uncertain"}:
            return attribute

        def recorded(*args, **kwargs):
            self._effects.append(f"ledger.{name}")
            return attribute(*args, **kwargs)

        return recorded


class FakeBroker:
    """06-03 broker の launch protocol を ledger 上で再現する fake です。

    scenario 列を launch ごとに1つ消費し、confirmed / no-process / uncertain / timeout / orphan を作ります。
    """

    def __init__(self, store, effects, probe_results, scenarios=()):
        """ledger・effect list・probe 結果表・scenario 列を受け取ります。"""
        self.store = store
        self.effects = effects
        self.probe_results = probe_results
        self.scenarios = list(scenarios)
        self.calls: list[str] = []

    def __call__(self, message):
        """launch request を strict に検証して scenario を実行します。"""
        request = parse_launch_request(message)
        attempt = request["attempt_id"]
        self.effects.append("broker.launch")
        self.calls.append(attempt)
        scenario = self.scenarios.pop(0) if self.scenarios else "confirmed"
        if scenario == "crash":
            raise Crash()
        if scenario == "no_process":
            self.store.commit_gate_failure(attempt, "create_process_failed")
            return {"kind": "launch_result", "attempt_id": attempt, "status": "proven_no_process"}
        if scenario == "timeout_pending":
            raise TimeoutError("broker pipe timed out")
        number = next(_PROCESS_NUMBERS)
        identity = ProcessIdentity(4000 + number, 133_000_000_000_000_000 + number, EXE, 7, 99)
        job = f"job-{attempt}"
        self.store.commit_attestation(attempt, ProcessAttestation(identity, job, "pid:1:ct:1"))
        self.store.commit_resume_intent(attempt, identity.process_ref)
        if scenario == "uncertain":
            self.store.commit_uncertain(attempt, "post_resume_probe_exited")
            return {"kind": "launch_result", "attempt_id": attempt, "status": "uncertain"}
        if scenario == "timeout_resumed":
            raise TimeoutError("broker pipe timed out after resume")
        self.store.commit_confirmation(attempt, identity.process_ref)
        if scenario == "timeout_confirmed":
            raise TimeoutError("broker pipe timed out after confirm")
        if scenario == "confirmed_dead":
            self.probe_results[identity.process_ref] = ProbeResult.EXITED
        ref = "pid:9:ct:9" if scenario == "wrong_ref" else identity.process_ref
        return {"kind": "launch_result", "attempt_id": attempt, "status": "confirmed", "process_ref": ref,
                "job_ref": job}


class FakeHandle:
    """controller process handle の fake です。

    wait は登録された結果関数を呼び、None なら timeout を表します。
    """

    def __init__(self, controller, spec):
        """spec から controller/helper PID を割り当てます。"""
        self.controller = controller
        self.spec = spec
        self.controller_pid = 5000 + spec.slot_id
        self.helper_pid = 6000 + spec.slot_id

    def wait(self, timeout_s):
        """run の終了を待つ代わりに結果を返します。"""
        self.controller.effects.append("controller.wait")
        self.controller.timeouts.append(timeout_s)
        behavior = self.controller.behaviors.pop(0) if self.controller.behaviors else _result
        return behavior(self.spec)

    def terminate(self):
        """controller terminate を記録します。"""
        self.controller.effects.append("controller.terminate")


class FakeController:
    """05-04 controller の起動 port の fake です。

    受け取った RunSpec を保存し、start 時の crash も再現できます。
    """

    def __init__(self, effects, development_only=True):
        """effect list を受け取ります。"""
        self.effects = effects
        self.development_only = development_only
        self.specs: list[RunSpec] = []
        self.behaviors: list = []
        self.timeouts: list = []
        self.crash_on_start = False

    def start(self, spec):
        """controller 起動を記録して handle を返します。"""
        self.effects.append("controller.start")
        self.specs.append(spec)
        if self.crash_on_start:
            raise Crash()
        return FakeHandle(self, spec)


class FakeHelper:
    """helper lease release の external observer の fake です。

    confirmed を False にすると release 未確認を再現します。
    """

    def __init__(self, effects, development_only=True):
        """effect list を受け取ります。"""
        self.effects = effects
        self.development_only = development_only
        self.confirmed = True

    def confirm_release(self, helper_pid, spec):
        """release 観測を記録して返します。"""
        self.effects.append("helper.confirm_release")
        return ReleaseObservation(self.confirmed, f"audit:{helper_pid}", 4.0)


class FakeTarget:
    """game / launcher process 状態の fake です。

    running を True にすると停止確認が失敗します。
    """

    def __init__(self, effects, development_only=True):
        """effect list を受け取ります。"""
        self.effects = effects
        self.development_only = development_only
        self.running = False
        self.stop_confirmed = True

    def stopped(self):
        """停止確認を記録します。"""
        self.effects.append("target.stopped")
        return not self.running

    def stop(self, spec):
        """起動した target の停止を記録します。"""
        self.effects.append("target.stop")
        return self.stop_confirmed


class FakeOperator:
    """operator checkpoint の fake です。

    decline に入れた checkpoint は確認されません。
    """

    def __init__(self, effects, development_only=True):
        """effect list を受け取ります。"""
        self.effects = effects
        self.development_only = development_only
        self.decline: set[str] = set()

    def checkpoint(self, name, context):
        """checkpoint を記録して結果を返します。"""
        self.effects.append(f"operator.{name}")
        return name not in self.decline


def _validator(effects):
    """06-02 validator へ委譲しつつ呼び出しを記録する fake validator です。

    判定は実 campaign_schema が行います。
    """

    def prerequisites(*args, **kwargs):
        effects.append("validator.prerequisites")
        return campaign_schema.validate_prerequisites(*args, **kwargs)

    def events(*args, **kwargs):
        effects.append("validator.events")
        return campaign_schema.validate_campaign_events(*args, **kwargs)

    return SimpleNamespace(validate_prerequisites=prerequisites, validate_campaign_events=events)


@pytest.fixture
def ledger(tmp_path):
    """current-user ACL 付きの実 durable ledger を開きます。

    Windows 以外では 06-03 と同じく検証不能として skip します。
    """
    if os.name != "nt":
        with pytest.raises(UnsupportedStorageError):
            DurableLaunchStore(tmp_path / "ledger")
        pytest.skip("formal_ineligible: platform_not_windows (verdict asserted)")
    store = DurableLaunchStore(tmp_path / "ledger")
    yield store
    store.close()


@pytest.fixture
def save_path(tmp_path):
    """元 save file を作ります。

    runner の backup / 差し替え / 復元対象です。
    """
    path = tmp_path / "game" / "SaveData.sav"
    path.parent.mkdir()
    path.write_bytes(ORIGINAL)
    return path


def _build(tmp_path, ledger, save_path, *, plan=None, root="artifacts", scenarios=(), development=True):
    """fake port と実 ledger / artifact store / save lifecycle で runner を組み立てます。

    effect list と各 fake を namespace で返します。
    """
    effects: list[str] = []
    probe_results: dict = {}
    artifacts = RecordingStore(tmp_path / root, effects)
    target = FakeTarget(effects, development)
    save = RecordingSave(save_path, artifacts, processes_stopped=target.stopped, effects=effects)
    save.register_canonical(CANONICAL)
    broker = FakeBroker(ledger, effects, probe_results, scenarios)
    controller = FakeController(effects, development)
    helper = FakeHelper(effects, development)
    operator = FakeOperator(effects, development)
    effects.clear()
    runner = CampaignRunner(
        plan or _plan(), artifacts=artifacts, ledger=LedgerProxy(ledger, effects), broker=broker,
        probe=lambda identity: probe_results.get(identity.process_ref, ProbeResult.ALIVE),
        controller=controller, helper=helper, target=target, operator=operator, save=save,
        validator=_validator(effects),
    )
    return SimpleNamespace(runner=runner, effects=effects, artifacts=artifacts, broker=broker, controller=controller,
                           helper=helper, target=target, operator=operator, save=save, ledger=ledger)


def _assert_order(effects, expected):
    """expected が effects の部分列として順に現れることを確認します。

    間に他の effect が挟まってもよいです。
    """
    index = 0
    for item in expected:
        try:
            index = effects.index(item, index) + 1
        except ValueError:
            pytest.fail(f"{item!r} missing after position {index} in {effects}")


def _outcomes(env, execution_id):
    """outcome stream から stage 実行の terminal event を返します。"""
    return [CampaignEvent.from_wire(record["event"]) for record in env.artifacts.stream("launched_outcomes")
            if record["stage_execution_id"] == execution_id]


SLOT_EFFECTS = [
    "validator.events", "stream.preflight_attempts", "target.stopped", "save.install_canonical",
    "ledger.commit_intent", "broker.launch", "ledger.activate", "stream.launch_gates", "operator.arm",
    "operator.focus", "operator.manual_start", "controller.start", "controller.wait", "helper.confirm_release",
    "target.stop", "save.post_run_hash",
]


def test_effect_order_preflight_launch_activate_terminal(tmp_path, ledger, save_path):
    """begin から finish までの effect が固定順序で起きます。

    各 slot は preflight 記録 → 停止確認 → save 差し替え → intent commit → broker → activation →
    checkpoint → controller → helper release → target 停止 → post hash → run manifest → outcome の順です。
    """
    env = _build(tmp_path, ledger, save_path)
    env.runner.begin()
    result = env.runner.run_stage("C0")
    env.runner.finish()
    assert result.state is RunnerState.STAGE_PASSED
    run = lambda slot: f"put:stages/canary-dev.C0.x1/runs/slot-{slot:02d}.json"  # noqa: E731
    _assert_order(env.effects, [
        "put:campaign/plan.json", "save.backup_original", "operator.cloud_sync_disabled", "save.attest_cloud_sync",
        "put:stages/canary-dev.C0.x1/manifest.json",
        *SLOT_EFFECTS, run(0), "validator.events", "stream.launched_outcomes",
        *SLOT_EFFECTS, run(1), "validator.events", "stream.launched_outcomes",
        "put:stages/canary-dev.C0.x1/report.json", "put:stages/canary-dev.C0.x1/execution.json",
        "save.restore_original", "put:campaign/summary.json",
    ])
    assert len(env.broker.calls) == 2 and len(set(env.broker.calls)) == 2
    assert save_path.read_bytes() == ORIGINAL


def test_controller_receives_only_ledger_confirmed_identity(tmp_path, ledger, save_path):
    """controller には ledger の attestation 由来の identity と固定 run mode だけが渡ります。

    gameplay attempt は slot ごとに deterministic な1件です。
    """
    env = _build(tmp_path, ledger, save_path)
    env.runner.begin()
    env.runner.run_stage("C0")
    for spec in env.controller.specs:
        ids = slot_identity("canary-dev.C0.x1", spec.slot_id)
        history = ledger.history(ids.attempt_id)
        assert spec.process_ref == history.attestation.identity.process_ref
        assert spec.job_ref == history.attestation.job_ref
        assert spec.target_pid == history.attestation.identity.pid
        assert (spec.reserved_run_id, spec.gameplay_attempt_id) == (ids.reserved_run_id, ids.gameplay_attempt_id)
        assert spec.campaign_run_mode == CAMPAIGN_RUN_MODE == "formal_single_attempt"
        assert spec.ui_restart_enabled is False and spec.development_only is True
        assert history.activation_source == "normal"
    assert env.controller.timeouts == [1800 + 30, 1800 + 30]


def test_report_is_reporter_output_and_streams_are_separate(tmp_path, ledger, save_path):
    """report は 06-02 reporter の canonical output で、4つの stream は種類ごとに分かれます。

    stream と ledger から再構成した event で reporter を呼び直すと保存済み report と一致します。
    """
    env = _build(tmp_path, ledger, save_path)
    env.runner.begin()
    result = env.runner.run_stage("C0")
    env.runner.finish()
    execution = "canary-dev.C0.x1"
    preflight = [record["event"]["event_type"] for record in env.artifacts.stream("preflight_attempts")]
    assert preflight == ["FORMAL_SLOT_RESERVED", "ATTEMPT_PREFLIGHT"] * 2
    gates = env.artifacts.stream("launch_gates")
    assert [[event["event_type"] for event in record["events"]] for record in gates] == [[
        "LAUNCH_INTENT_COMMITTED", "BROKER_PROCESS_ATTESTED", "PROCESS_LAUNCH_CONFIRMED", "FORMAL_RUN_ACTIVATED",
    ]] * 2
    assert all(record["broker_response"]["status"] == "confirmed" and record["ledger_row_hashes"] for record in gates)
    assert [event.event_type for event in _outcomes(env, execution)] == [EventType.SUCCESS] * 2
    assert env.artifacts.stream("campaign_chain") == []
    events = []
    for slot in range(2):
        events += [CampaignEvent.from_wire(r["event"]) for r in env.artifacts.stream("preflight_attempts")
                   if r["event"]["slot_id"] == slot]
        events += [CampaignEvent.from_wire(e) for r in gates if r["slot_id"] == slot for e in r["events"]]
        events += [event for event in _outcomes(env, execution) if event.slot_id == slot]
    stored = env.artifacts.read_json(f"stages/{execution}/manifest.json")
    manifest = CampaignManifest.from_wire(stored["manifest"])
    expected = generate_campaign_report(manifest, events, event_manifest_hash=canonical_hash(manifest.to_wire()))
    assert env.artifacts.read_json(f"stages/{execution}/report.json") == expected.to_wire()
    assert result.report_hash == expected.report_hash
    assert env.artifacts.read_json(f"stages/{execution}/execution.json")["report_hash"] == expected.report_hash


def test_run_manifest_collects_pids_release_refs_and_save_hashes(tmp_path, ledger, save_path):
    """per-run manifest は PID・release 観測・telemetry/video/broker 参照・save hash を持ちます。

    synthetic run は development_only=true かつ formal_campaign_eligible=false です。
    """
    env = _build(tmp_path, ledger, save_path)
    env.runner.begin()
    env.runner.run_stage("C0")
    summary = env.runner.finish()
    run = env.artifacts.read_json("stages/canary-dev.C0.x1/runs/slot-01.json")
    assert (run["controller_pid"], run["helper_pid"]) == (5001, 6001)
    assert run["release_observation"] == {"confirmed": True, "observer_ref": "audit:6001", "release_ms": 4.0}
    assert run["telemetry_refs"] == ["telemetry.jsonl"] and run["video_refs"] == ["run.mp4"]
    assert run["lease_audit_ref"] == "lease_audit.jsonl"
    assert run["broker"]["process_ref"].startswith("pid:") and run["broker"]["ledger_row_hashes"]
    assert run["save"]["pre_sha256"] == CANONICAL_HASH and run["save"]["post_sha256"] == CANONICAL_HASH
    assert run["campaign_run_mode"] == "formal_single_attempt"
    assert run["operator_checkpoints"] == ["arm", "focus", "manual_start"]
    assert (run["development_only"], run["formal_campaign_eligible"], run["save_hashes_complete"]) == (
        True, False, False)
    assert "formal_evidence_eligible" not in run
    plan = env.artifacts.read_json("campaign/plan.json")
    assert (plan["development_only"], plan["formal_campaign_eligible"], plan["ui_restart_enabled"]) == (
        True, False, False)
    assert (summary["development_only"], summary["formal_evidence_eligible"]) == (True, False)
    assert summary["restore_verdict"]["status"] == "PASS"


@pytest.mark.parametrize(
    ("plan_factory", "message"),
    [
        (lambda: _formal_plan(prerequisites=_formal_prerequisites("9" * 64)), "stale or mixed"),
        (lambda: _formal_plan(threshold_parent_hash="7" * 64), "threshold_parent_hash"),
        (lambda: _plan(canonical_save_hash="8" * 64), "canonical save"),
    ],
)
def test_prerequisite_mismatch_touches_nothing(tmp_path, ledger, save_path, plan_factory, message):
    """prerequisite / threshold parent / canonical save の不一致は何も書かずに止まります。

    runner.state は PREREQUISITE_MISMATCH になり、stage も開始できません。
    """
    env = _build(tmp_path, ledger, save_path, plan=plan_factory())
    with pytest.raises(PrerequisiteMismatch, match=message):
        env.runner.begin()
    assert env.runner.state is RunnerState.PREREQUISITE_MISMATCH
    assert not [effect for effect in env.effects if effect.startswith(("put:", "save.", "ledger.", "broker."))]
    assert ledger.attempt_ids() == ()
    with pytest.raises(RunnerError, match="begin"):
        env.runner.run_stage("C0")


def test_preflight_failure_blocks_stage_without_reserved_identity(tmp_path, ledger, save_path):
    """preflight failure は slot を terminal にし、reserved identity を作らず stage を止めます。

    是正後は新しい stage 実行 id で最初からやり直し、blocked chain は report に残ります。
    """
    env = _build(tmp_path, ledger, save_path)
    env.runner.begin()
    stopped = env.runner.run_stage("C0", max_slots=1)
    assert stopped.state is RunnerState.STOPPED and env.runner.state is RunnerState.STOPPED
    env.target.running = True
    installs = env.effects.count("save.install_canonical")
    blocked = env.runner.run_stage("C0")
    assert blocked.state is RunnerState.PREFLIGHT_BLOCKED
    assert blocked.slot_events[1] == ("FORMAL_SLOT_RESERVED", "ATTEMPT_PREFLIGHT", "PREFLIGHT_FAILED")
    assert env.effects.count("save.install_canonical") == installs
    x1 = slot_identity("canary-dev.C0.x1", 1)
    assert x1.attempt_id not in ledger.attempt_ids() and len(env.broker.calls) == 1
    report = env.artifacts.read_json("stages/canary-dev.C0.x1/report.json")
    assert report["preflight_failures"] == 1 and report["stage_blocked"] is True
    assert env.artifacts.stream("campaign_chain")[0]["state"] == "preflight_blocked"
    with pytest.raises(StageOrderError, match="C0 has not passed"):
        env.runner.run_stage("C1")
    env.target.running = False
    rerun = env.runner.run_stage("C0")
    assert (rerun.stage_execution_id, rerun.state) == ("canary-dev.C0.x2", RunnerState.STAGE_PASSED)
    report = env.artifacts.read_json("stages/canary-dev.C0.x2/report.json")
    assert report["campaign_chain"]["blocked"] == ["canary-dev.C0.x1"]
    assert env.artifacts.read_json("stages/canary-dev.C0.x2/manifest.json")["blocked_execution_ids"] == [
        "canary-dev.C0.x1"]


def test_broker_timeout_after_confirm_uses_reconciliation_activation(tmp_path, ledger, save_path):
    """確認済み launch の応答が timeout したら reconcile で CONFIRMED を確かめて activation します。

    broker への再送はせず、activation_source は reconciliation になります。
    """
    env = _build(tmp_path, ledger, save_path, scenarios=["timeout_confirmed"])
    env.runner.begin()
    result = env.runner.run_stage("C0")
    assert result.state is RunnerState.STAGE_PASSED
    first = slot_identity("canary-dev.C0.x1", 0).attempt_id
    assert ledger.history(first).activation_source == "reconciliation"
    assert env.broker.calls.count(first) == 1
    _assert_order(env.effects, ["broker.launch", "ledger.reconcile", "ledger.activate", "controller.start"])


@pytest.mark.parametrize(
    ("scenario", "last", "state"),
    [
        ("no_process", "LAUNCH_GATE_FAILED", RunnerState.LAUNCH_GATE_BLOCKED),
        ("timeout_pending", "LAUNCH_GATE_FAILED", RunnerState.LAUNCH_GATE_BLOCKED),
        ("uncertain", "LAUNCH_UNCERTAIN", RunnerState.CAMPAIGN_BLOCKED),
        ("timeout_resumed", "LAUNCH_UNCERTAIN", RunnerState.CAMPAIGN_BLOCKED),
        ("confirmed_dead", "LAUNCH_UNCERTAIN", RunnerState.CAMPAIGN_BLOCKED),
        ("wrong_ref", "LAUNCH_UNCERTAIN", RunnerState.CAMPAIGN_BLOCKED),
    ],
)
def test_launch_failures_are_consumed_without_replacement(tmp_path, ledger, save_path, scenario, last, state):
    """no-process / timeout / uncertain / confirmed orphan は置き換えずに slot を消費します。

    controller は起動されず、後続 slot も予約されず、同じ stage は再実行できません。
    uncertain と orphan は campaign 全体を block します。
    """
    env = _build(tmp_path, ledger, save_path, scenarios=[scenario])
    env.runner.begin()
    result = env.runner.run_stage("C0")
    assert result.state is state and env.runner.state is state
    assert result.slot_events[0][-1] == last and 1 not in result.slot_events
    assert env.controller.specs == [] and len(env.broker.calls) == 1
    assert env.artifacts.stream("campaign_chain")[0]["state"] == state.value
    expected = CampaignBlocked if state is RunnerState.CAMPAIGN_BLOCKED else StageOrderError
    with pytest.raises(expected):
        env.runner.run_stage("C0")
    assert len(env.broker.calls) == 1


def test_lower_stage_must_pass_before_upper_stage(tmp_path, ledger, save_path):
    """下位 stage 未実行・未 promotion の場合は上位 stage を開始しません。

    death を含む C0 は not promoted で、C1 は拒否されます。
    """
    env = _build(tmp_path, ledger, save_path)
    env.runner.begin()
    with pytest.raises(StageOrderError, match="C0 has not passed"):
        env.runner.run_stage("C1")
    env.controller.behaviors = [lambda spec: _result(spec, died=True, target_success="not_reached")]
    result = env.runner.run_stage("C0")
    assert result.state is RunnerState.STAGE_NOT_PROMOTED
    assert _outcomes(env, "canary-dev.C0.x1")[0].failure_reason == "death"
    with pytest.raises(StageOrderError):
        env.runner.run_stage("C1")
    with pytest.raises(StageOrderError, match="already concluded"):
        env.runner.run_stage("C0")


def test_passed_lower_stage_gates_next_stage(tmp_path, ledger, save_path):
    """promotion 済みの C0 を parent にして C1 を開始できます。

    C1 manifest は parent 実行 id を記録します。
    """
    env = _build(tmp_path, ledger, save_path)
    env.runner.begin()
    env.runner.run_stage("C0")
    result = env.runner.run_stage("C1", max_slots=1)
    assert result.state is RunnerState.STOPPED
    stored = env.artifacts.read_json("stages/canary-dev.C1.x1/manifest.json")
    assert stored["parent_execution_id"] == "canary-dev.C0.x1"
    assert stored["manifest"]["expected_slots"] == campaign_schema.STAGE_POLICIES["C1"].slot_count


def test_validate_parent_rejections(tmp_path, ledger, save_path):
    """parent 検証は未 pass・report 改変・threshold parent 変更・development-only parent を拒否します。

    formal plan は development-only の下位 stage を parent にできません。
    """
    env = _build(tmp_path, ledger, save_path)
    env.runner.begin()
    env.runner.run_stage("C0")
    record = env.artifacts.read_json("stages/canary-dev.C0.x1/execution.json")
    report = env.artifacts.read_json("stages/canary-dev.C0.x1/report.json")
    plan = env.runner.plan
    validate_parent(plan, "C1", record, report)
    with pytest.raises(StageOrderError, match="requires a C1 parent"):
        validate_parent(plan, "C2", record, report)
    with pytest.raises(StageOrderError, match="has not passed"):
        validate_parent(plan, "C1", {**record, "state": "stage_not_promoted"}, report)
    with pytest.raises(PlanMismatch, match="report hash"):
        validate_parent(plan, "C1", record, {**report, "successes": 99})
    with pytest.raises(PlanMismatch, match="threshold parent"):
        validate_parent(replace(plan, threshold_parent_hash="6" * 64), "C1",
                        {**record, "plan_hash": replace(plan, threshold_parent_hash="6" * 64).plan_hash}, report)
    formal = _formal_plan()
    forged_report = {**report, "prerequisite_parent_hash": FORMAL_PARENT}
    forged = {**record, "plan_hash": formal.plan_hash, "threshold_parent_hash": formal.threshold_parent_hash,
              "report_hash": canonical_hash(forged_report)}
    with pytest.raises(StageOrderError, match="development-only parent"):
        validate_parent(formal, "C1", forged, forged_report)


def test_plan_changes_after_live_results_are_rejected(tmp_path, ledger, save_path):
    """live 結果の後に threshold や threshold parent を変えた plan では再開できません。

    新しい campaign id で C0 からやり直す必要があります。
    """
    env = _build(tmp_path, ledger, save_path)
    env.runner.begin()
    env.runner.run_stage("C0", max_slots=1)
    changed = _build(tmp_path, ledger, save_path, plan=_plan(thresholds={"perception_recall": 0.5}))
    with pytest.raises(PlanMismatch, match="thresholds cannot change after live results"):
        changed.runner.begin()
    reparented = _build(tmp_path, ledger, save_path, plan=_plan(threshold_parent_hash="5" * 64))
    with pytest.raises(PlanMismatch, match="threshold_parent_hash"):
        reparented.runner.begin()


def test_completed_slots_and_reserved_identities_cannot_be_reused(tmp_path, ledger, save_path):
    """完了 slot の再実行と、別 runner からの同一 reserved / gameplay identity の再予約を拒否します。

    ledger が identity claim を拒否すると preflight failure となり broker は呼ばれません。
    """
    env = _build(tmp_path, ledger, save_path)
    env.runner.begin()
    env.runner.run_stage("C0", max_slots=1)
    env.runner.run_stage("C0")
    first = slot_identity("canary-dev.C0.x1", 0).attempt_id
    assert env.broker.calls.count(first) == 1
    with pytest.raises(StageOrderError, match="already concluded"):
        env.runner.run_stage("C0")
    save_path.write_bytes(ORIGINAL)
    other = _build(tmp_path, ledger, save_path, root="artifacts-copy")
    other.runner.begin()
    result = other.runner.run_stage("C0")
    assert result.state is RunnerState.PREFLIGHT_BLOCKED
    assert other.broker.calls == []
    # 同じ attempt id の ledger 行は nonce が違うので、other の slot として採用されません。
    assert result.slot_events[0] == ("FORMAL_SLOT_RESERVED", "ATTEMPT_PREFLIGHT", "PREFLIGHT_FAILED")
    reason = [r for r in other.artifacts.stream("preflight_attempts") if r["event"]["event_type"] == "PREFLIGHT_FAILED"]
    assert reason[0]["event"]["failure_reason"].startswith("launch_intent_commit_failed")


def test_outcome_overwrite_is_rejected(tmp_path, ledger, save_path):
    """terminal outcome と per-run manifest は上書きできません。

    stream に2件目の terminal があれば resume 時に 06-02 validator が fail-closed で拒否します。
    """
    env = _build(tmp_path, ledger, save_path)
    env.runner.begin()
    env.runner.run_stage("C0", max_slots=1)
    name = "stages/canary-dev.C0.x1/runs/slot-00.json"
    with pytest.raises(ArtifactConflict):
        env.artifacts.put_json(name, {**env.artifacts.read_json(name), "controller_pid": 1})
    manifest_hash = env.artifacts.read_json(name)["campaign_manifest_hash"]
    duplicate = CampaignEvent(EventType.GAMEPLAY_FAILURE, slot_id=0, failure_reason="rewritten",
                              campaign_manifest_hash=manifest_hash)
    env.artifacts.append("launched_outcomes", [{"stage_execution_id": "canary-dev.C0.x1",
                                                "event": duplicate.to_wire()}])
    with pytest.raises(ValueError, match="after its terminal outcome"):
        env.runner.run_stage("C0")


def test_resume_after_crash_post_activation_consumes_failure(tmp_path, ledger, save_path):
    """activation 後に runner が落ちたら、再開時に safety failure として消費し再 launch しません。

    target 停止と post-run hash は再開時にも取り、後続 slot は通常どおり進みます。
    """
    env = _build(tmp_path, ledger, save_path)
    env.runner.begin()
    env.controller.crash_on_start = True
    with pytest.raises(Crash):
        env.runner.run_stage("C0")
    resumed = _build(tmp_path, ledger, save_path)
    resumed.runner.begin()
    result = resumed.runner.run_stage("C0")
    outcomes = _outcomes(resumed, "canary-dev.C0.x1")
    assert [(event.event_type, event.failure_reason) for event in outcomes] == [
        (EventType.SAFETY_FAILURE, "runner_interrupted_after_activation"), (EventType.SUCCESS, None)]
    assert resumed.broker.calls == [slot_identity("canary-dev.C0.x1", 1).attempt_id]
    assert "target.stop" in resumed.effects and "save.post_run_hash" in resumed.effects
    assert result.state is RunnerState.STAGE_NOT_PROMOTED


def test_resume_with_pending_intent_reconciles_without_relaunch(tmp_path, ledger, save_path):
    """broker 送信中に runner が落ちた intent は再開時に reconcile され、再送されません。

    resume intent 前の intent は proven no-process の launch gate failure になります。
    """
    env = _build(tmp_path, ledger, save_path, scenarios=["crash"])
    env.runner.begin()
    with pytest.raises(Crash):
        env.runner.run_stage("C0")
    resumed = _build(tmp_path, ledger, save_path)
    resumed.runner.begin()
    result = resumed.runner.run_stage("C0")
    assert result.state is RunnerState.LAUNCH_GATE_BLOCKED
    assert resumed.broker.calls == []
    assert result.slot_events[0] == ("FORMAL_SLOT_RESERVED", "ATTEMPT_PREFLIGHT", "LAUNCH_INTENT_COMMITTED",
                                     "LAUNCH_GATE_FAILED")


def test_unreconciled_foreign_intent_blocks_next_slot(tmp_path, ledger, save_path):
    """ledger に未 reconcile intent があれば次 slot を予約しません。

    別 campaign の intent でも replacement 防止のため開始を拒否します。
    """
    ledger.commit_intent(LaunchIntent(
        campaign_manifest_hash="9" * 64, slot_id=0, attempt_id="foreign-attempt", reserved_run_id="foreign-run",
        gameplay_attempt_id="foreign-gameplay", launch_nonce=new_launch_nonce(), executable_path=EXE,
        executable_hash="b" * 64, build_hash="c" * 64, config_hash="d" * 64, argv=(EXE,),
    ))
    env = _build(tmp_path, ledger, save_path)
    env.runner.begin()
    with pytest.raises(UnreconciledLaunch, match="foreign-attempt"):
        env.runner.run_stage("C0")
    assert env.artifacts.stream("preflight_attempts") == [] and env.broker.calls == []


@pytest.mark.parametrize(
    ("confirmed", "kind", "reason"),
    [(True, EventType.ARTIFACT_FAILURE, "controller_timeout"),
     (False, EventType.SAFETY_FAILURE, "helper_release_unconfirmed")],
)
def test_timeout_terminates_controller_then_confirms_helper_release(tmp_path, ledger, save_path, confirmed, kind,
                                                                     reason):
    """timeout は controller terminate の後に helper release を external observer で確認します。

    release を確認できなければ safety failure です。
    """
    env = _build(tmp_path, ledger, save_path)
    env.helper.confirmed = confirmed
    env.controller.behaviors = [lambda spec: None]
    env.runner.begin()
    env.runner.run_stage("C0", max_slots=1)
    _assert_order(env.effects, ["controller.wait", "controller.terminate", "helper.confirm_release", "target.stop"])
    event = _outcomes(env, "canary-dev.C0.x1")[0]
    assert (event.event_type, event.failure_reason) == (kind, reason)


def test_checkpoints_gate_controller_and_menu_input_is_safety_failure(tmp_path, ledger, save_path):
    """arm / focus / manual start が確認されるまで controller を起動しません。

    UI restart 無効の run で menu input が送られていれば safety failure です。
    """
    env = _build(tmp_path, ledger, save_path)
    env.operator.decline = {"focus"}
    env.runner.begin()
    env.runner.run_stage("C0", max_slots=1)
    assert env.controller.specs == []
    assert "operator.manual_start" not in env.effects and "target.stop" in env.effects
    event = _outcomes(env, "canary-dev.C0.x1")[0]
    assert (event.event_type, event.failure_reason) == (EventType.SAFETY_FAILURE, "focus_checkpoint_not_confirmed")
    env.operator.decline = set()
    env.controller.behaviors = [lambda spec: _result(spec, menu_inputs_sent=2)]
    env.runner.run_stage("C0")
    event = _outcomes(env, "canary-dev.C0.x1")[1]
    assert (event.event_type, event.failure_reason) == (
        EventType.SAFETY_FAILURE, "menu_input_while_ui_restart_disabled")


def test_outcome_event_conversion():
    """run 結果を 06-02 reporter 入力の terminal event へ変換します。

    confirmed のみ SUCCESS、pending / death は gameplay failure、safety > artifact の優先順です。
    """
    spec = RunSpec("c", "c.C0.x1", "C0", 0, "r", "g", CAMPAIGN_RUN_MODE, "pid:1:ct:1", "j", 1, 1800, False, True)
    release = ReleaseObservation(True, "audit", 3.5)
    manifest_hash = "a" * 64
    success = outcome_event(0, manifest_hash, _result(spec), [], release)
    assert success.event_type is EventType.SUCCESS
    details = dict(success.details)
    assert {key: details[key] for key in ("level", "gems", "kills", "choices", "unknown_frames",
                                          "fallback_decisions", "target_success")} == {
        "level": 21, "gems": 340, "kills": 812, "choices": 20, "unknown_frames": 3, "fallback_decisions": 1,
        "target_success": "confirmed"}
    assert dict(details["latency_ms"]) == {"p50": 9.5, "p95": 14.0}
    assert dict(details["release"])["release_ms"] == 3.5
    pending = outcome_event(0, manifest_hash, _result(spec, target_success="pending"), [], release)
    assert (pending.event_type, pending.failure_reason) == (EventType.GAMEPLAY_FAILURE, "target_success_pending")
    died = outcome_event(0, manifest_hash, _result(spec, died=True), [], release)
    assert (died.event_type, died.failure_reason) == (EventType.GAMEPLAY_FAILURE, "death")
    mixed = outcome_event(0, manifest_hash, _result(spec), [(EventType.ARTIFACT_FAILURE, "a"),
                                                           (EventType.SAFETY_FAILURE, "s")], release)
    assert (mixed.event_type, mixed.failure_reason) == (EventType.SAFETY_FAILURE, "s")
    assert list(dict(mixed.details)["failures"]) == ["ARTIFACT_FAILURE:a", "SAFETY_FAILURE:s"]
    bad = _result(spec, gameplay_attempt_id="other", gameplay_entries=2, telemetry_refs=())
    assert result_failures(spec, bad) == [
        (EventType.SAFETY_FAILURE, "gameplay_attempt_mismatch"),
        (EventType.SAFETY_FAILURE, "gameplay_entries_2"),
        (EventType.ARTIFACT_FAILURE, "telemetry_missing"),
    ]


def test_formal_plan_rejects_development_ports(tmp_path, ledger, save_path):
    """formal plan は development-only の fake controller / helper では開始できません。

    save や plan には何も書きません。
    """
    env = _build(tmp_path, ledger, save_path, plan=_formal_plan())
    with pytest.raises(RunnerError, match="development-only ports"):
        env.runner.begin()
    assert not env.artifacts.exists("campaign/plan.json") and "save.backup_original" not in env.effects


def test_formal_evidence_requires_every_save_record(tmp_path, ledger, save_path):
    """formal 証跡は backup・cloud sync・各 run の pre/post hash・restore verdict がそろったときだけです。

    post-run hash を取れない run が1件でもあれば campaign summary は formal_evidence_eligible=false です。
    run manifest は save hash の完全性だけを持ち、formal 適格は summary の列挙だけで表します。
    """
    env = _build(tmp_path, ledger, save_path, plan=_formal_plan(), development=False)
    env.runner.begin()
    assert "validator.prerequisites" in env.effects
    env.runner.run_stage("C0")
    run = env.artifacts.read_json("stages/canary-formal.C0.x1/runs/slot-00.json")
    assert (run["development_only"], run["save_hashes_complete"]) == (False, True)
    assert _formal_claims(env.artifacts) == []
    summary = env.runner.finish()
    assert summary["formal_evidence_eligible"] is True and summary["formal_campaign_eligible"] is True
    assert summary["formal_evidence_run_manifests"] == [
        "stages/canary-formal.C0.x1/runs/slot-00.json", "stages/canary-formal.C0.x1/runs/slot-01.json"]
    assert _formal_claims(env.artifacts) == ["campaign/summary.json"]

    save_path.write_bytes(ORIGINAL)
    broken = _build(tmp_path, ledger, save_path, root="artifacts-broken",
                    plan=_formal_plan(campaign_id="canary-formal-2"), development=False)
    broken.target.stop_confirmed = False
    broken.runner.begin()
    broken.runner.run_stage("C0", max_slots=1)
    run = broken.artifacts.read_json("stages/canary-formal-2.C0.x1/runs/slot-00.json")
    assert run["save"]["post_sha256"] is None and run["save_hashes_complete"] is False
    assert run["outcome"]["failure_reason"] == "target_process_not_stopped"
    summary = broken.runner.finish()
    assert summary["formal_evidence_eligible"] is False and summary["formal_evidence_run_manifests"] == []
    assert _formal_claims(broken.artifacts) == []


def _formal_claims(artifacts) -> list[str]:
    """formal_evidence_eligible=true を主張する JSON artifact の相対 path を列挙します。

    artifact root 配下の全 JSON を再帰的に調べ、入れ子の値も含めて探します。
    """
    def claims(value) -> bool:
        if isinstance(value, dict):
            return value.get("formal_evidence_eligible") is True or any(map(claims, value.values()))
        return isinstance(value, list) and any(map(claims, value))

    root = artifacts.root
    return sorted(path.relative_to(root).as_posix() for path in root.rglob("*.json")
                  if claims(json.loads(path.read_text(encoding="utf-8"))))


def test_failed_restore_leaves_no_formal_evidence_claim(tmp_path, ledger, save_path):
    """restore が失敗すると summary は作られず、formal 適格を主張する artifact が1件もありません。

    run 実行後(restore 未実行)と restore 失敗後の両方で確認し、元 save が戻っていないことも固定します。
    """
    env = _build(tmp_path, ledger, save_path, plan=_formal_plan(), development=False)
    env.runner.begin()
    env.runner.run_stage("C0")
    assert _formal_claims(env.artifacts) == []
    env.artifacts._path("save/original_backup.bin").write_bytes(b"corrupted-backup")
    with pytest.raises(SaveLifecycleError, match="restore failed"):
        env.runner.finish()
    assert not env.artifacts.exists("campaign/summary.json")
    assert _formal_claims(env.artifacts) == []
    for slot in (0, 1):
        run = env.artifacts.read_json(f"stages/canary-formal.C0.x1/runs/slot-{slot:02d}.json")
        assert run["save_hashes_complete"] is True and "formal_evidence_eligible" not in run
    assert save_path.read_bytes() == CANONICAL


@pytest.mark.parametrize("fresh_instance", [False, True])
def test_finalized_campaign_rejects_begin_and_run_stage(tmp_path, ledger, save_path, fresh_instance):
    """finish 済み campaign は同じ instance でも同じ root の新 instance でも begin / run_stage を拒否します。

    canonical save が再び入らず、元 save のままであることを確認します。
    """
    env = _build(tmp_path, ledger, save_path, plan=_formal_plan(), development=False)
    env.runner.begin()
    env.runner.run_stage("C0")
    assert env.runner.finish()["formal_evidence_eligible"] is True
    assert save_path.read_bytes() == ORIGINAL
    probe = _build(tmp_path, ledger, save_path, plan=_formal_plan(), development=False) if fresh_instance else env
    probe.effects.clear()
    with pytest.raises(RunnerError, match="already finalized"):
        probe.runner.run_stage("C1")
    with pytest.raises(RunnerError, match="already finalized"):
        probe.runner.begin()
    assert "save.install_canonical" not in probe.effects and "save.backup_original" not in probe.effects
    assert save_path.read_bytes() == ORIGINAL
    assert not env.artifacts.exists("stages/canary-formal.C1.x1/manifest.json")
