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
