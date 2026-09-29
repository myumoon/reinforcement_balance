"""Win32 launch broker の request 検証・read-back・attestation・失敗時 terminate を検証します。

harmless な Python helper を実際に CREATE_SUSPENDED + kill-on-job-close Job で起動し、
kernel identity・resume marker・activation 可否を確認します。本家 game は起動しません。
"""

from __future__ import annotations

import hashlib
import os
import sys
import threading
import time
from pathlib import Path

import pytest

from survivors.campaign import win32_launch_broker as broker_module
from survivors.campaign.campaign_schema import CampaignEvent, EventType, validate_campaign_events
from survivors.campaign.durable_launch_store import (
    Classification,
    DurableLaunchStore,
    LaunchIntent,
    LaunchStage,
    LedgerCommitError,
    LedgerIdentityError,
    LedgerOrderError,
    ProbeResult,
    ProcessIdentity,
    UnsupportedStorageError,
    acl_findings,
    new_launch_nonce,
)
from survivors.campaign.win32_launch_broker import (
    PIPE_PREFIX,
    AttestationError,
    LaunchBroker,
    build_launch_request,
    parse_launch_request,
    pipe_sddl,
    probe_identity,
    request_broker,
    serve,
)

MANIFEST = "e" * 64
TARGET_CODE = (
    "import os,sys,time;"
    "open(sys.argv[1],'w').write(os.environ.get('REINBALANCE_LAUNCH_NONCE',''));"
    "time.sleep(600)"
)


@pytest.fixture
def store(tmp_path):
    """current-user ACL 付きの新規 ledger を開きます。

    Windows 以外では formal-ineligible verdict を確認した上で、この環境では検証不能として扱います。
    """
    ledger = tmp_path / "ledger"
    if os.name != "nt":
        with pytest.raises(UnsupportedStorageError, match="platform_not_windows"):
            DurableLaunchStore(ledger)
        pytest.skip("formal_ineligible: platform_not_windows (verdict asserted)")
    opened = DurableLaunchStore(ledger)
    yield opened
    opened.close()


@pytest.fixture
def broker(store):
    """ledger に結び付いた in-process broker を作ります。

    test 終了時に close し、Job 内の helper を必ず終了させます。
    """
    instance = LaunchBroker(store)
    yield instance
    instance.close()


def _exe_hash() -> str:
    """test helper として使う python.exe の SHA-256 です。

    intent の executable_hash に入れ、broker が実 file と照合します。
    """
    return hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest()


def _intent(tmp_path: Path, slot: int = 0, **overrides) -> LaunchIntent:
    """resume marker を書く harmless helper の intent を作ります。

    helper は resume 後にだけ marker へ受け取った nonce を書き、その後 sleep します。
    """
    exe = sys.executable
    values = dict(
        campaign_manifest_hash=MANIFEST, slot_id=slot, attempt_id=f"attempt-{slot}", reserved_run_id=f"run-{slot}",
        gameplay_attempt_id=f"gameplay-{slot}", launch_nonce=new_launch_nonce(), executable_path=exe,
        executable_hash=_exe_hash(), build_hash="1" * 64, config_hash="2" * 64,
        argv=(exe, "-c", TARGET_CODE, str(tmp_path / f"marker-{slot}.txt")),
    )
    values.update(overrides)
    return LaunchIntent(**values)


def _wait_for(predicate, timeout: float = 10.0) -> bool:
    """条件が真になるまで短い間隔で待ちます。

    時間内に真にならなければ False を返します。
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def _validate_0602(store: DurableLaunchStore, attempt: str, slot: int = 0) -> None:
    """ledger 由来 event 列を 06-02 validator に通します。

    reserved/preflight を前置して slot lifecycle 全体を検査します。
    """
    prefix = [CampaignEvent(EventType.FORMAL_SLOT_RESERVED, slot, campaign_manifest_hash=MANIFEST),
              CampaignEvent(EventType.ATTEMPT_PREFLIGHT, slot, attempt_id=attempt, campaign_manifest_hash=MANIFEST)]
    validate_campaign_events([*prefix, *store.campaign_events(attempt)], expected_manifest_hash=MANIFEST)


def test_parse_launch_request_is_strict() -> None:
    """launch request は key 完全一致・schema・digest 形式を要求します。

    欠落・未知 key・別 schema・短い nonce を全部拒否します。
    """
    exe = sys.executable
    intent = LaunchIntent(MANIFEST, 0, "a", "r", "g", new_launch_nonce(), exe, "3" * 64, "1" * 64, "2" * 64, (exe,))
    request = build_launch_request(intent)
    assert parse_launch_request(request) == request
    for broken in ({**request, "extra": 1}, {k: v for k, v in request.items() if k != "config_hash"},
                   {**request, "schema": "v0"}, {**request, "launch_nonce": "ab"}, {**request, "kind": "resume"}):
        with pytest.raises(ValueError):
            parse_launch_request(broken)


def test_pipe_security_is_current_user_only_and_local() -> None:
    """named pipe DACL は current user だけの protected DACL です。

    remote pipe 名や入れ子名は受け付けません。
    """
    sid = "S-1-5-21-1-2-3-1001"
    assert acl_findings(pipe_sddl(sid), sid, require_owner=False, require_protected=True) == []
    for name in ("\\\\server\\pipe\\x", PIPE_PREFIX + "a\\b", "pipe"):
        with pytest.raises(ValueError):
            broker_module._validate_pipe_name(name)


@pytest.mark.parametrize("field", ["launch_nonce", "config_hash", "executable_hash", "command_hash"])
def test_read_back_mismatch_is_rejected_without_mutation(store, broker, tmp_path, field) -> None:
    """request の nonce / config / executable / command hash が ledger と違えば起動しません。

    ledger は LAUNCH_INTENT のまま変わらず、process も作られません。
    """
    intent = _intent(tmp_path)
    store.commit_intent(intent)
    request = {**build_launch_request(intent), field: "f" * 64}
    result = broker.handle_message(request)
    assert result["status"] == "rejected" and field in result["failure_reason"]
    assert store.history(intent.attempt_id).stages == (LaunchStage.LAUNCH_INTENT,)


def test_executable_hash_mismatch_on_disk_is_durable_no_process(store, broker, tmp_path) -> None:
    """disk 上の executable が intent の hash と違えば process を作らず gate failure です。

    その launch は activation できず、06-02 の LAUNCH_GATE_FAILED に対応します。
    """
    intent = _intent(tmp_path, executable_hash="9" * 64)
    store.commit_intent(intent)
    result = broker.launch(parse_launch_request(build_launch_request(intent)))
    assert result["status"] == Classification.PROVEN_NO_PROCESS.value
    assert result["failure_reason"] == "executable_hash_mismatch"
    assert store.history(intent.attempt_id).last_stage is LaunchStage.LAUNCH_GATE_FAILED
    with pytest.raises(LedgerOrderError):
        store.activate(intent.attempt_id, "normal", probe_identity)
    _validate_0602(store, intent.attempt_id)


def test_confirmed_launch_is_attested_resumed_and_killed_with_job(store, broker, tmp_path) -> None:
    """正常系: kernel identity で attest され、resume 後に marker が書かれ、Job close で終了します。

    activation は1回だけで、同じ attempt への再 launch request は rejected です。
    """
    intent = _intent(tmp_path)
    store.commit_intent(intent)
    result = broker.launch(parse_launch_request(build_launch_request(intent)))
    assert result["status"] == "confirmed"
    history = store.history(intent.attempt_id)
    identity = history.attestation.identity
    assert result["process_ref"] == identity.process_ref == f"pid:{identity.pid}:ct:{identity.create_time}"
    assert os.path.normcase(identity.image_path) == os.path.normcase(os.path.realpath(sys.executable))
    marker = Path(intent.argv[-1])
    assert _wait_for(marker.exists) and _wait_for(lambda: marker.read_text() == intent.launch_nonce)
    assert probe_identity(identity) is ProbeResult.ALIVE
    assert store.activate(intent.attempt_id, "normal", probe_identity) is True
    assert store.activate(intent.attempt_id, "normal", probe_identity) is False
    again = broker.launch(parse_launch_request(build_launch_request(intent)))
    assert again["status"] == "rejected"
    _validate_0602(store, intent.attempt_id)
    broker.close()
    assert _wait_for(lambda: probe_identity(identity) is not ProbeResult.ALIVE)


def test_probe_identity_detects_pid_reuse_and_exit(store, broker, tmp_path) -> None:
    """PID が同じでも create time / file identity が違えば PID_REUSED です。

    Job close 後の process は ALIVE になりません。
    """
    intent = _intent(tmp_path)
    store.commit_intent(intent)
    broker.launch(parse_launch_request(build_launch_request(intent)))
    identity = store.history(intent.attempt_id).attestation.identity
    reused = ProcessIdentity(identity.pid, identity.create_time + 1, identity.image_path, identity.volume_serial,
                             identity.file_index)
    other_file = ProcessIdentity(identity.pid, identity.create_time, identity.image_path, identity.volume_serial,
                                 identity.file_index + 1)
    assert probe_identity(reused) is ProbeResult.PID_REUSED
    assert probe_identity(other_file) is ProbeResult.PID_REUSED
    me = broker_module._own_identity()
    assert probe_identity(me) is ProbeResult.ALIVE
    with pytest.raises(LedgerIdentityError, match="pid_reused"):
        store.activate(intent.attempt_id, "normal", lambda _identity: probe_identity(reused))
    broker.close()
    assert _wait_for(lambda: probe_identity(identity) in {ProbeResult.EXITED, ProbeResult.PID_REUSED})
    assert store.reconcile(intent.attempt_id, probe_identity).classification is Classification.UNCERTAIN


def _capture_attest(monkeypatch, captured: list, error: Exception | None = None) -> None:
    """_attest を包み、生成された suspended process の identity を記録します。

    error を渡すとその例外で attestation 失敗を再現します。
    """
    original = broker_module._attest

    def wrapped(info, job, expected_file, intent):
        captured.append(broker_module.process_identity(info.process, info.pid))
        if error is not None:
            raise error
        return original(info, job, expected_file, intent)

    monkeypatch.setattr(broker_module, "_attest", wrapped)


def test_attestation_failure_terminates_suspended_helper(store, broker, tmp_path, monkeypatch) -> None:
    """attestation 失敗時は suspended helper を terminate し、gate failure にします。

    helper は一度も resume されないので marker は書かれず、activation もできません。
    """
    captured: list[ProcessIdentity] = []
    _capture_attest(monkeypatch, captured, AttestationError("forced"))
    intent = _intent(tmp_path)
    store.commit_intent(intent)
    result = broker.launch(parse_launch_request(build_launch_request(intent)))
    assert result["status"] == Classification.PROVEN_NO_PROCESS.value
    assert result["failure_reason"].startswith("attestation_failed")
    assert probe_identity(captured[0]) is not ProbeResult.ALIVE
    assert not Path(intent.argv[-1]).exists()
    with pytest.raises(LedgerOrderError):
        store.activate(intent.attempt_id, "normal", probe_identity)
    _validate_0602(store, intent.attempt_id)


def test_resume_intent_commit_failure_never_resumes(store, broker, tmp_path, monkeypatch) -> None:
    """resume intent を commit できなければ resume せず terminate して gate failure です。

    ledger に resume intent が無いので no-process と証明できます。
    """
    captured: list[ProcessIdentity] = []
    _capture_attest(monkeypatch, captured)

    def fail(*_args) -> None:
        raise LedgerCommitError("disk full")

    monkeypatch.setattr(store, "commit_resume_intent", fail)
    intent = _intent(tmp_path)
    store.commit_intent(intent)
    result = broker.launch(parse_launch_request(build_launch_request(intent)))
    assert result["status"] == Classification.PROVEN_NO_PROCESS.value
    assert probe_identity(captured[0]) is not ProbeResult.ALIVE
    time.sleep(0.3)
    assert not Path(intent.argv[-1]).exists()
    assert store.history(intent.attempt_id).last_stage is LaunchStage.LAUNCH_GATE_FAILED


@pytest.mark.parametrize("failure", ["confirm_commit", "post_resume_probe"])
def test_failure_after_resume_is_uncertain_and_terminated(store, broker, tmp_path, monkeypatch, failure) -> None:
    """resume 後の失敗は uncertain(campaign block)になり、helper は Job ごと終了します。

    process が走った可能性があるため no-process とは分類しません。
    """
    captured: list[ProcessIdentity] = []
    _capture_attest(monkeypatch, captured)
    if failure == "confirm_commit":
        def fail(*_args) -> None:
            raise LedgerCommitError("commit failed")

        monkeypatch.setattr(store, "commit_confirmation", fail)
    else:
        monkeypatch.setattr(broker_module, "probe_identity", lambda _identity: ProbeResult.EXITED)
    intent = _intent(tmp_path)
    store.commit_intent(intent)
    result = broker.launch(parse_launch_request(build_launch_request(intent)))
    assert result["status"] == Classification.UNCERTAIN.value
    assert store.history(intent.attempt_id).last_stage is LaunchStage.LAUNCH_UNCERTAIN
    assert _wait_for(lambda: probe_identity(captured[0]) is not ProbeResult.ALIVE)
    with pytest.raises(LedgerOrderError):
        store.activate(intent.attempt_id, "normal", probe_identity)
    _validate_0602(store, intent.attempt_id)


def test_named_pipe_checks_server_identity_and_serves_launch(store, broker, tmp_path) -> None:
    """pipe 越しの launch は server PID と SID を照合してから処理されます。

    server PID が期待と違えば client は request を送らず、ledger は変わりません。
    """
    pipe = f"{PIPE_PREFIX}reinbalance-test-{os.getpid()}-{time.monotonic_ns()}"
    thread = threading.Thread(target=serve, args=(pipe, broker), daemon=True)
    thread.start()
    intent = _intent(tmp_path)
    store.commit_intent(intent)
    with pytest.raises(PermissionError):
        request_broker(pipe, build_launch_request(intent), broker_pid=os.getpid() + 4, timeout_s=10)
    assert request_broker(pipe, {"kind": "nope"}, broker_pid=os.getpid())["kind"] == "rejected"
    assert store.history(intent.attempt_id).stages == (LaunchStage.LAUNCH_INTENT,)
    confirmed = request_broker(pipe, build_launch_request(intent), broker_pid=os.getpid())
    assert confirmed["status"] == "confirmed"
    assert request_broker(pipe, {"kind": "shutdown"}, broker_pid=os.getpid()) == {"kind": "shutdown_ack"}
    thread.join(10)
    assert not thread.is_alive()


def test_server_rejects_client_with_foreign_sid(store, tmp_path, monkeypatch) -> None:
    """client token の SID が current user でなければ request を読まずに拒否します。

    server 側の SID 照合だけを差し替え、ledger が変わらないことを確認します。
    """
    broker = LaunchBroker(store)
    pipe = f"{PIPE_PREFIX}reinbalance-test-sid-{os.getpid()}-{time.monotonic_ns()}"
    real = broker_module.process_user_sid
    me = os.getpid()
    monkeypatch.setattr(broker_module, "process_user_sid",
                        lambda pid=None: "S-1-5-21-0-0-0-500" if pid == me and threading.current_thread().name ==
                        "broker-server" else real(pid))
    thread = threading.Thread(target=serve, args=(pipe, broker), kwargs={"max_requests": 1}, daemon=True,
                              name="broker-server")
    thread.start()
    intent = _intent(tmp_path)
    store.commit_intent(intent)
    response = request_broker(pipe, build_launch_request(intent), broker_pid=me)
    assert response == {"kind": "rejected", "error": "client is not the current user"}
    thread.join(10)
    broker.close()
    assert store.history(intent.attempt_id).stages == (LaunchStage.LAUNCH_INTENT,)
