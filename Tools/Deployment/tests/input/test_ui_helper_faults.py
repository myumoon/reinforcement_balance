"""UI lease(ROIクリック・Enter・Escape)のsubprocess往復・未知kind拒否・大量fault-injectionテスト。
movementと同じsubprocess境界・拒否経路をUI操作でも独立に検査し、held leakや同時保持がないことを確認します。
"""
from __future__ import annotations
import multiprocessing
import random
import threading
import time
from survivors.input.audit_log import AuditLog
from survivors.input.dry_run_backend import DryRunBackend, run_dry_run_helper_for_test
from survivors.input.helper import HelperRuntime, run_helper_loop
from survivors.input.lease_protocol import Lease, LeaseValidator, UiLease, ui_action_contract_hash
def _movement_lease(sequence: int, action: int, now_ns: int) -> Lease:
    """75msの有効期限を持つ fault test 用 movement lease を作る。
    UIとの混在fault-injectionで、movement側の生成規則をUI側と揃えます。
    """
    return Lease(
        session_nonce="1" * 32, sequence=sequence, issued_monotonic_ns=now_ns,
        expires_monotonic_ns=now_ns + 75_000_000, target_hash="a" * 64, action_hash="b" * 64,
        action_index=action, target_pid=42, target_hwnd=84,
    )
def _ui_lease(sequence: int, now_ns: int, ui_action: str = "CLICK") -> UiLease:
    """75msの有効期限を持つ fault test 用 UiLease を作る。
    CLICK以外は座標fieldを含めず、閉じた行動空間のままfault-injectionできるようにします。
    """
    coords = {"normalized_x": 0.5, "normalized_y": 0.5} if ui_action == "CLICK" else {}
    return UiLease(
        session_nonce="1" * 32, sequence=sequence, issued_monotonic_ns=now_ns,
        expires_monotonic_ns=now_ns + 75_000_000, target_hash="a" * 64,
        ui_action_hash=ui_action_contract_hash(), ui_action=ui_action,
        target_pid=42, target_hwnd=84, **coords,
    )
def test_ui_action_subprocess_round_trip_has_bounded_latency(tmp_path) -> None:
    """独立helper subprocessでUI lease(CLICK)のACK往復が100msの目標内に収まる。
    movementの解放latency計測と対になる、UI側のsubprocess境界越え応答性の確認です。
    """
    parent, child = multiprocessing.Pipe()
    operations = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=run_dry_run_helper_for_test,
        args=(child, operations, str(tmp_path / "ui-subprocess-audit.jsonl"),
              "1" * 32, "a" * 64, "b" * 64, 42, 84),
    )
    process.start()
    child.close()
    latencies_ms: list[float] = []
    try:
        assert parent.recv()["kind"] == "ready"
        for sequence in range(1, 101):
            issued = time.monotonic_ns()
            parent.send(_ui_lease(sequence, issued).to_wire())
            reply = parent.recv()
            assert reply["kind"] == "ack" and reply["applied"] is True
            latencies_ms.append((time.monotonic_ns() - issued) / 1_000_000)
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
def test_run_helper_loop_rejects_unknown_wire_kind_without_degrading_to_lease(tmp_path) -> None:
    """`kind`が既知の集合以外ならLeaseへ縮退させず拒否する(I4)。
    subprocessを起こさずthread内でrun_helper_loopを動かし、応答とSendInput 0件を確認します。
    """
    parent, child = multiprocessing.Pipe()
    backend = DryRunBackend(foreground_pid=42, foreground_hwnd=84, focused=True, armed=True)
    runtime = HelperRuntime(
        LeaseValidator("1" * 32, "a" * 64, "b" * 64, 42, 84),
        backend, AuditLog(tmp_path / "unknown-kind-audit.jsonl"),
    )
    thread = threading.Thread(target=run_helper_loop, args=(child, runtime, "1" * 32), daemon=True)
    thread.start()
    try:
        assert parent.recv()["kind"] == "ready"
        parent.send({"kind": "mystery", "session_nonce": "1" * 32})
        reply = parent.recv()
        assert reply["kind"] == "rejected"
        assert backend.send_input_calls == 0
    finally:
        parent.send({"kind": "shutdown"})
        thread.join(timeout=1.0)
        parent.close()
def test_dry_run_10k_interleaved_movement_and_ui_actions_never_double_hold(tmp_path) -> None:
    """1万件のmovement/UI混在random送信後もheld leakがなく、同時保持も一度も起きない(I5)。
    各件の直後にUI action時のpressed空集合を毎回確認し、境界での見落としを防ぎます。
    """
    backend = DryRunBackend(foreground_pid=42, foreground_hwnd=84, focused=True, armed=True)
    runtime = HelperRuntime(
        LeaseValidator("1" * 32, "a" * 64, "b" * 64, 42, 84),
        backend, AuditLog(tmp_path / "ui-random-audit.jsonl"),
    )
    generator = random.Random(20260924)
    sequence = 0
    for sequence in range(1, 10_001):
        now_ns = 1_000_000_000 + sequence * 1_000_000
        if generator.random() < 0.5:
            lease: Lease | UiLease = _movement_lease(sequence, generator.randrange(9), now_ns)
        else:
            ui_action = generator.choice(["CLICK", "ENTER", "ESCAPE"])
            lease = _ui_lease(sequence, now_ns, ui_action)
        result = runtime.handle_lease(lease, now_ns)
        assert result["kind"] == "ack"
        if isinstance(lease, UiLease):
            assert backend.pressed == set(), "UI action must never coexist with a held movement chord"
    runtime.emergency_release(sequence=sequence)
    assert backend.held_leak_count == 0
    assert backend.pressed == set()
