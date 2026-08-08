"""入力 helper の arm・focus・target gate と allowlist の受入テスト。
対象ゲームへ安全に結び付かない状況では、OS 入力処理を一度も呼ばないことを確認します。
"""
from __future__ import annotations
import pytest
from survivors.input.audit_log import AuditLog
from survivors.input.dry_run_backend import DryRunBackend
from survivors.input.helper import HelperRuntime
from survivors.input.lease_protocol import Lease, LeaseValidator
def _runtime(tmp_path, *, armed: bool, focused: bool, pid: int = 42, hwnd: int = 84):
    """指定 gate 状態の helper runtime を作る。
    OS API を使わず、呼び出し回数を観察できる backend を組み合わせます。
    """
    backend = DryRunBackend(foreground_pid=pid, foreground_hwnd=hwnd,
                            focused=focused, armed=armed)
    runtime = HelperRuntime(
        validator=LeaseValidator("1" * 32, "a" * 64, "b" * 64, 42, 84),
        backend=backend,
        audit=AuditLog(tmp_path / "audit.jsonl"),
    )
    return runtime, backend
def _lease(sequence: int = 1, *, pid: int = 42, hwnd: int = 84, action: int = 0) -> Lease:
    """gate テスト用の有効な semantic lease を返す。
    target binding 以外の検証条件を固定し、注入 gate だけを観察します。
    """
    return Lease(
        session_nonce="1" * 32,
        sequence=sequence,
        issued_monotonic_ns=1_000_000_000,
        expires_monotonic_ns=1_075_000_000,
        target_hash="a" * 64,
        action_hash="b" * 64,
        action_index=action,
        target_pid=pid,
        target_hwnd=hwnd,
    )
@pytest.mark.parametrize(
    "armed, focused, lease_pid, lease_hwnd, foreground_pid, foreground_hwnd",
    [
        (False, True, 42, 84, 42, 84),
        (True, False, 42, 84, 42, 84),
        (True, True, 43, 84, 42, 84),
        (True, True, 42, 85, 42, 84),
        (True, True, 42, 84, 43, 84),
        (True, True, 42, 84, 42, 85),
    ],
)
def test_unsafe_gate_never_calls_send_input(
    tmp_path,
    armed: bool,
    focused: bool,
    lease_pid: int,
    lease_hwnd: int,
    foreground_pid: int,
    foreground_hwnd: int,
) -> None:
    """disarm・非focus・PID/HWND不一致では SendInput 相当を0件にする。
    lease 自身と現在の前面 window の両側を照合し、片側だけ正しいケースも拒否します。
    """
    runtime, backend = _runtime(
        tmp_path,
        armed=armed,
        focused=focused,
        pid=foreground_pid,
        hwnd=foreground_hwnd,
    )
    runtime.handle_lease(_lease(pid=lease_pid, hwnd=lease_hwnd), now_ns=1_001_000_000)
    assert backend.send_input_calls == 0
def test_arm_is_default_off_and_toggle_is_edge_triggered() -> None:
    """arm は既定 off で、Ctrl+Shift+F12 chord の立上りだけで反転する。
    hotkey を押し続けた poll では連続反転せず、離して再押下した時だけ変わります。
    """
    backend = DryRunBackend(foreground_pid=42, foreground_hwnd=84, focused=True)
    assert backend.armed is False
    backend.set_arm_chord_for_test(True)
    backend.poll_arm_toggle()
    assert backend.armed is True
    backend.poll_arm_toggle()
    assert backend.armed is True
    backend.set_arm_chord_for_test(False)
    backend.poll_arm_toggle()
    backend.set_arm_chord_for_test(True)
    backend.poll_arm_toggle()
    assert backend.armed is False
def test_backend_allowlist_rejects_every_nonapproved_key() -> None:
    """WASD・Enter・Escape・left click 以外を backend で拒否する。
    wire が閉じていても内部バグから危険キーが届く可能性に備え、注入直前にも検査します。
    """
    backend = DryRunBackend(foreground_pid=42, foreground_hwnd=84, focused=True)
    for unsafe in ("Q", "F12", "CTRL", "RIGHT_CLICK"):
        with pytest.raises(ValueError):
            backend.apply_inputs({unsafe})
    assert backend.send_input_calls == 0
