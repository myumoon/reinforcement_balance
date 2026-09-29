"""controller / broker / target helper を実 process で動かし、全 crash 境界の recovery を検証します。

broker を各 protocol 境界で停止させ、テスト process(別 process)から controller・broker を
TerminateProcess で kill します。再起動後の reconcile が confirmed / proven-no-process / uncertain を
正しく分け、replacement launch・二重 activation・PID reuse substitution が0件であることを確認します。
support envelope 外の環境は skip せず formal-ineligible verdict を記録します。本家 game は起動しません。
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from survivors.campaign.campaign_schema import CampaignEvent, EventType, validate_campaign_events
from survivors.campaign.durable_launch_store import (
    Classification,
    DurableLaunchStore,
    LaunchIntent,
    LaunchStage,
    LedgerIdentityError,
    LedgerOrderError,
    ProbeResult,
    ProcessIdentity,
    UnsupportedStorageError,
    check_storage,
    new_launch_nonce,
)

MANIFEST = "7" * 64
DEPLOYMENT_ROOT = Path(__file__).resolve().parents[2]
TARGET_CODE = (
    "import os,sys,time;"
    "open(sys.argv[1],'w').write(os.environ.get('REINBALANCE_LAUNCH_NONCE',''));"
    "time.sleep(600)"
)
CONTROLLER_CODE = (
    "import json,sys\n"
    "from survivors.campaign.durable_launch_store import DurableLaunchStore, LaunchIntent\n"
    "from survivors.campaign.win32_launch_broker import build_launch_request, request_broker\n"
    "ledger, pipe, broker_pid, payload, mode = sys.argv[1:6]\n"
    "intent = LaunchIntent.from_payload(json.loads(payload))\n"
    "with DurableLaunchStore(ledger) as store:\n"
    "    store.commit_intent(intent)\n"
    "if mode == 'intent_only':\n"
    "    sys.exit(0)\n"
    "print(json.dumps(request_broker(pipe, build_launch_request(intent), broker_pid=int(broker_pid), timeout_s=60)))\n"
)
# (hold 境界, kill 対象, 期待分類, helper が resume 済み(marker あり)か)
SCENARIOS = [
    ("intent_commit", "controller_exit", Classification.PROVEN_NO_PROCESS, False),
    ("after_intent_readback", "both", Classification.PROVEN_NO_PROCESS, False),
    ("after_create_process", "both", Classification.PROVEN_NO_PROCESS, False),
    ("after_attestation", "both", Classification.PROVEN_NO_PROCESS, False),
    ("after_resume_intent", "both", Classification.UNCERTAIN, False),
    ("after_resume_intent", "broker", Classification.UNCERTAIN, False),
    ("after_resume", "both", Classification.UNCERTAIN, True),
    ("after_confirm", "both", Classification.UNCERTAIN, True),
    ("after_attestation", "controller", Classification.CONFIRMED, True),
    ("after_confirm", "controller", Classification.CONFIRMED, True),
    ("none", "none", Classification.CONFIRMED, True),
]


def _env() -> dict[str, str]:
    """子 process 用の環境変数を作ります。

    survivors package を import できるよう Deployment root を PYTHONPATH の先頭に置きます。
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(DEPLOYMENT_ROOT), env.get("PYTHONPATH", "")])
    return env


def _wait_for(predicate, timeout: float = 30.0) -> bool:
    """条件が真になるまで短い間隔で待ちます。

    時間内に真にならなければ False を返します。
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def _eligible_or_record(tmp_path: Path) -> bool:
    """support envelope 外なら formal-ineligible verdict を記録して False を返します。

    skip はせず、ledger が開けないこと(fail closed)を assert してから結果を保存します。
    """
    verdict = check_storage(tmp_path / "ledger")
    if verdict.eligible:
        return True
    with pytest.raises(UnsupportedStorageError):
        DurableLaunchStore(tmp_path / "ledger")
    result = {"verdict": verdict.label, "reasons": list(verdict.reasons), "facts": dict(verdict.facts)}
    (tmp_path / "integration_result.json").write_text(json.dumps(result, default=str), encoding="utf-8")
    assert verdict.label == "formal_ineligible"
    return False


def _intent(tmp_path: Path, **overrides) -> LaunchIntent:
    """marker を書く harmless target helper の intent を作ります。

    helper は resume 後にだけ marker へ nonce を書き、その後 sleep します。
    """
    exe = sys.executable
    values = dict(
        campaign_manifest_hash=MANIFEST, slot_id=0, attempt_id="attempt-0", reserved_run_id="run-0",
        gameplay_attempt_id="gameplay-0", launch_nonce=new_launch_nonce(), executable_path=exe,
        executable_hash=hashlib.sha256(Path(exe).read_bytes()).hexdigest(), build_hash="1" * 64,
        config_hash="2" * 64, argv=(exe, "-c", TARGET_CODE, str(tmp_path / "resume-marker.txt")),
    )
    values.update(overrides)
    return LaunchIntent(**values)


def _pipe_name(tag: str) -> str:
    """test ごとに一意な local named pipe 名を返します。

    並行実行や前回の残骸と衝突しないよう PID と時刻を含めます。
    """
    from survivors.campaign.win32_launch_broker import PIPE_PREFIX

    return f"{PIPE_PREFIX}reinbalance-it-{tag}-{os.getpid()}-{time.monotonic_ns()}"


def _spawn_broker(ledger: Path, pipe: str, hold_at: str | None = None, fault_dir: Path | None = None):
    """broker を別 process として起動します。

    hold_at を渡すとその protocol 境界で停止し、テストからの kill を待ちます。
    """
    command = [sys.executable, "-m", "survivors.campaign.win32_launch_broker", "--ledger", str(ledger),
               "--pipe", pipe]
    if hold_at is not None:
        command += ["--test-fault-hold-at", hold_at, "--test-fault-dir", str(fault_dir)]
    return subprocess.Popen(command, env=_env(), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def _spawn_controller(ledger: Path, pipe: str, broker_pid: int, intent: LaunchIntent, mode: str):
    """intent を commit して broker へ request する controller を別 process で起動します。

    mode='intent_only' は intent commit 直後に request を送らず終了します。
    """
    payload = json.dumps(intent.to_payload())
    return subprocess.Popen([sys.executable, "-c", CONTROLLER_CODE, str(ledger), pipe, str(broker_pid), payload, mode],
                            env=_env(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _probe():
    """Win32 kernel probe を返します。

    import を Windows 実行時まで遅らせます。
    """
    from survivors.campaign.win32_launch_broker import probe_identity

    return probe_identity


def _validate_0602(store: DurableLaunchStore, attempt: str) -> None:
    """ledger 由来 event 列を 06-02 validator に通します。

    reserved/preflight を前置し campaign_manifest_hash 束縛も検査します。
    """
    prefix = [CampaignEvent(EventType.FORMAL_SLOT_RESERVED, 0, campaign_manifest_hash=MANIFEST),
              CampaignEvent(EventType.ATTEMPT_PREFLIGHT, 0, attempt_id=attempt, campaign_manifest_hash=MANIFEST)]
    validate_campaign_events([*prefix, *store.campaign_events(attempt)], expected_manifest_hash=MANIFEST)


def _stop(process: subprocess.Popen) -> None:
    """process を TerminateProcess で kill して終了を待ちます。

    既に終了していれば何もしません。
    """
    if process.poll() is None:
        process.kill()
    process.wait(timeout=30)


@pytest.mark.parametrize(("boundary", "kill", "expected", "resumed"), SCENARIOS,
                         ids=[f"{b}-{k}" for b, k, _e, _r in SCENARIOS])
def test_crash_boundary_recovery_with_real_processes(tmp_path, boundary, kill, expected, resumed) -> None:
    """各 protocol 境界で controller / broker を kill し、reconcile の分類と0件条件を確かめます。

    resume intent 前の crash は helper が一度も走らず(marker なし)proven-no-process、
    resume intent 以降は uncertain、broker が生き残って confirm した launch だけが confirmed で
    idempotent に1回だけ activation できます。結果は環境 hash と共に JSON へ保存します。
    """
    if not _eligible_or_record(tmp_path):
        return
    from survivors.campaign.win32_launch_broker import (
        build_launch_request,
        integration_environment,
        reconcile_ledger,
        request_broker,
    )

    probe = _probe()
    ledger, fault = tmp_path / "ledger", tmp_path / "fault"
    fault.mkdir()
    DurableLaunchStore(ledger).close()
    intent = _intent(tmp_path)
    marker = Path(intent.argv[-1])
    pipe = _pipe_name("crash")
    hold = boundary if boundary not in {"intent_commit", "none"} else None
    broker = _spawn_broker(ledger, pipe, hold, fault)
    controller = _spawn_controller(ledger, pipe, broker.pid, intent,
                                   "intent_only" if boundary == "intent_commit" else "launch")
    facts: dict = {}
    try:
        if hold is not None:
            reached = fault / f"{hold}.reached"
            assert _wait_for(reached.exists), broker.stderr.read().decode(errors="replace") if broker.poll() else hold
            facts = json.loads(reached.read_text(encoding="utf-8"))
            if resumed and kill == "both":
                assert _wait_for(marker.exists)
        if kill == "both":
            _stop(controller)
            _stop(broker)
        elif kill == "broker":
            _stop(broker)
            controller.wait(timeout=60)
        elif kill == "controller":
            _stop(controller)
            (fault / "release").write_text("go", encoding="utf-8")
        elif kill == "controller_exit":
            assert controller.wait(timeout=60) == 0
        else:
            output, _ = controller.communicate(timeout=60)
            assert json.loads(output.splitlines()[-1])["status"] == "confirmed"
        if expected is Classification.CONFIRMED:
            with DurableLaunchStore(ledger) as watch:
                assert _wait_for(lambda: watch.history(intent.attempt_id).last_stage
                                 is LaunchStage.PROCESS_LAUNCH_CONFIRMED)
        if "process" in facts and kill in {"both", "broker"}:
            killed = ProcessIdentity.from_payload(facts["process"])
            assert _wait_for(lambda: probe(killed) is not ProbeResult.ALIVE), "kill-on-job-close did not stop helper"
        if not resumed:
            time.sleep(0.3)
            assert not marker.exists(), "helper ran although it was never resumed"
        else:
            assert _wait_for(marker.exists) and marker.read_text() == intent.launch_nonce

        with DurableLaunchStore(ledger) as store:
            (result,) = reconcile_ledger(store)
            assert result.classification is expected, result
            assert reconcile_ledger(store)[0].classification is expected
            history = store.history(intent.attempt_id)
            if expected is Classification.CONFIRMED:
                assert store.activate(intent.attempt_id, "reconciliation", probe) is True
                assert store.activate(intent.attempt_id, "reconciliation", probe) is False
                identity = history.attestation.identity
                forged = ProcessIdentity(identity.pid, identity.create_time + 1, identity.image_path,
                                         identity.volume_serial, identity.file_index)
                assert probe(forged) is ProbeResult.PID_REUSED
            else:
                with pytest.raises(LedgerOrderError):
                    store.activate(intent.attempt_id, "reconciliation", probe)
            activations = store._conn.execute("SELECT COUNT(*) FROM launch_rows WHERE stage=?",
                                              (LaunchStage.FORMAL_RUN_ACTIVATED.value,)).fetchone()[0]
            assert activations == (1 if expected is Classification.CONFIRMED else 0)
            with pytest.raises(LedgerIdentityError, match="duplicate slot"):
                store.commit_intent(_intent(tmp_path, attempt_id="attempt-replacement",
                                            reserved_run_id="run-r", gameplay_attempt_id="gameplay-r"))
            assert store.attempt_ids() == (intent.attempt_id,)
            _validate_0602(store, intent.attempt_id)
            environment = integration_environment(store)
            stages = [stage.value for stage in store.history(intent.attempt_id).stages]

        relaunch_pipe = _pipe_name("relaunch")
        relaunch = _spawn_broker(ledger, relaunch_pipe)
        try:
            again = request_broker(relaunch_pipe, build_launch_request(intent), broker_pid=relaunch.pid)
            assert again["status"] == "rejected", again
            assert request_broker(relaunch_pipe, {"kind": "shutdown"}, broker_pid=relaunch.pid)["kind"] == "shutdown_ack"
            assert relaunch.wait(timeout=30) == 0
        finally:
            _stop(relaunch)

        if broker.poll() is None:
            assert request_broker(pipe, {"kind": "shutdown"}, broker_pid=broker.pid)["kind"] == "shutdown_ack"
            broker.wait(timeout=30)
            if history.attestation is not None:
                assert _wait_for(lambda: probe(history.attestation.identity) is not ProbeResult.ALIVE)
        record = {
            "verdict": environment["storage_verdict"], "environment": environment, "boundary": boundary,
            "kill": kill, "classification": result.classification.value, "reason": result.reason,
            "stages": stages, "resumed_marker": marker.exists(), "replacement_launches": 0,
            "activation_rows": activations,
        }
        path = tmp_path / "integration_result.json"
        path.write_text(json.dumps(record, sort_keys=True, default=str), encoding="utf-8")
        saved = json.loads(path.read_text(encoding="utf-8"))
        assert saved["verdict"] == "formal_eligible"
        assert saved["environment"]["filesystem"]["filesystem"] == "NTFS"
        assert saved["environment"]["sqlite"]["journal_mode"] == "wal"
        assert set(saved["environment"]["build_hashes"]) == {"win32_launch_broker", "durable_launch_store",
                                                             "campaign_schema"}
    finally:
        _stop(controller)
        _stop(broker)


def test_broker_refuses_unsupported_storage_as_formal_ineligible(tmp_path) -> None:
    """UNC 上の ledger を指定された broker は起動せず、pipe も作りません。

    support 外 storage を skip ではなく formal-ineligible(非0終了)として扱います。
    """
    unc = "\\\\unsupported-host\\share\\ledger"
    verdict = check_storage(unc)
    assert verdict.label == "formal_ineligible"
    broker = subprocess.run(
        [sys.executable, "-m", "survivors.campaign.win32_launch_broker", "--ledger", unc,
         "--pipe", "\\\\.\\pipe\\reinbalance-it-unsupported"],
        env=_env(), capture_output=True, timeout=60)
    assert broker.returncode != 0
    assert b"formal-ineligible storage" in broker.stderr
