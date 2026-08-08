"""helper expiry と大量 action 解放の fault-injection テスト。
controller が止まっても短い lease の期限で key-up され、長時間の操作でキーが残らないことを検査します。
"""
from __future__ import annotations
import multiprocessing
import random
import time
from survivors.input.audit_log import AuditLog
from survivors.input.dry_run_backend import DryRunBackend, run_dry_run_helper_for_test
from survivors.input.helper import HelperRuntime
from survivors.input.lease_protocol import Lease, LeaseValidator
def _lease(sequence: int, action: int, now_ns: int) -> Lease:
    """75ms の有効期限を持つ fault test lease を作る。
    毎回同じ target binding を使い、停止・遅延時の解放時間だけを測ります。
    """
    return Lease(
        session_nonce="1" * 32,
        sequence=sequence,
        issued_monotonic_ns=now_ns,
        expires_monotonic_ns=now_ns + 75_000_000,
        target_hash="a" * 64,
        action_hash="b" * 64,
        action_index=action,
        target_pid=42,
        target_hwnd=84,
    )
def test_helper_subprocess_releases_expired_lease_with_bounded_observed_latency(tmp_path) -> None:
    """独立 helper が75ms lease期限後に held key を全解放する。
    user-space process が動ける fault 条件で100回測り、p99 100ms以下・最大150ms以下を確認します。
    """
    parent, child = multiprocessing.Pipe()
    operations = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=run_dry_run_helper_for_test,
        args=(child, operations, str(tmp_path / "subprocess-audit.jsonl"),
              "1" * 32, "a" * 64, "b" * 64, 42, 84),
    )
    process.start()
    child.close()
    latencies_ms: list[float] = []
    try:
        assert parent.recv()["kind"] == "ready"
        for sequence in range(1, 101):
            issued = time.monotonic_ns()
            parent.send(_lease(sequence, sequence % 8, issued).to_wire())
            assert parent.recv()["kind"] == "ack"
            while True:
                operation = operations.get(timeout=0.2)
                if operation["kind"] == "release" and operation["sequence"] == sequence:
                    break
            assert operation["held_count"] == 0
            latencies_ms.append((operation["monotonic_ns"] - issued) / 1_000_000)
        parent.send({"kind": "emergency_release", "session_nonce": "1" * 32})
        assert parent.recv()["kind"] == "released"
    finally:
        parent.close()
        process.join(timeout=1.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=1.0)
    ordered = sorted(latencies_ms)
    p99 = ordered[98]
    assert p99 <= 100.0
    assert max(latencies_ms) <= 150.0
def _dry_run_helper_for_controller_fault(
    connection, audit_path: str, session_nonce: str,
    target_hash: str, action_hash: str, target_pid: int, target_hwnd: int,
) -> None:
    """controller_fault テスト用の helper_main インターフェース互換 dry-run entry。
    process_target として InputLeaseController へ渡せるようにシグネチャを合わせます。
    """
    from survivors.input.audit_log import AuditLog
    from survivors.input.dry_run_backend import DryRunBackend
    from survivors.input.helper import HelperRuntime, run_helper_loop
    from survivors.input.lease_protocol import LeaseValidator
    backend = DryRunBackend(
        foreground_pid=target_pid, foreground_hwnd=target_hwnd,
        focused=True, armed=True,
    )
    runtime = HelperRuntime(
        LeaseValidator(session_nonce, target_hash, action_hash, target_pid, target_hwnd),
        backend, AuditLog(audit_path),
    )
    run_helper_loop(connection, runtime, session_nonce)
def test_controller_atexit_close_pipe_triggers_helper_exit_and_release(tmp_path) -> None:
    """controller が atexit で pipe を閉じると helper が 500ms 内に exit し release audit を残す。
    context manager を使わず _atexit_close_pipe() を直接呼ぶことで sys.exit シナリオを模擬します。
    """
    import json, pathlib
    from survivors.input.controller import InputLeaseController
    target_hash, action_hash = "a" * 64, "b" * 64
    target_pid, target_hwnd = 42, 84
    audit_path = str(tmp_path / "atexit-audit.jsonl")
    ctrl = InputLeaseController(
        target_hash=target_hash, action_hash=action_hash,
        target_pid=target_pid, target_hwnd=target_hwnd, audit_path=audit_path,
        process_target=_dry_run_helper_for_controller_fault,
    )
    deadline = time.monotonic() + 1.0
    ready = False
    while time.monotonic() < deadline and not ready:
        if ctrl._connection.poll(0.005):
            msg = ctrl._connection.recv()
            if isinstance(msg, dict) and msg.get("kind") == "ready":
                ready = True
        time.sleep(0.001)
    assert ready, "helper must send ready"
    # 1 keydown lease を適用する（action 0 = N, key W）
    ctrl.send_action(0)
    # sys.exit の atexit と同等の pipe close でデッドロックなく helper が exit することを確認する
    t_close = time.monotonic()
    ctrl._atexit_close_pipe()
    ctrl._process.join(timeout=0.5)
    elapsed_ms = (time.monotonic() - t_close) * 1000
    assert not ctrl._process.is_alive(), "helper must exit after pipe close"
    assert elapsed_ms < 500, f"helper took {elapsed_ms:.0f}ms > 500ms"
    # audit に release event が含まれること
    lines = pathlib.Path(audit_path).read_text().splitlines()
    events = [json.loads(ln) for ln in lines if ln.strip()]
    assert any(e.get("event") == "release" for e in events), "release event must appear in audit"
def test_dry_run_10k_random_actions_has_no_pressed_without_release(tmp_path) -> None:
    """1万件のランダム action 後に pressed key が残らないことを確認する。
    action 切替の差分 key-up と最後の emergency release を含め、漏れ件数を厳密に0とします。
    """
    backend = DryRunBackend(
        foreground_pid=42, foreground_hwnd=84, focused=True, armed=True
    )
    runtime = HelperRuntime(
        LeaseValidator("1" * 32, "a" * 64, "b" * 64, 42, 84),
        backend,
        AuditLog(tmp_path / "random-audit.jsonl"),
    )
    generator = random.Random(20260809)
    for sequence in range(1, 10_001):
        now_ns = 1_000_000_000 + sequence * 1_000_000
        runtime.handle_lease(_lease(sequence, generator.randrange(9), now_ns), now_ns)
    runtime.emergency_release(sequence=10_000)
    assert backend.held_leak_count == 0
    assert backend.pressed == set()
