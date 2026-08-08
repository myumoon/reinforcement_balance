"""テスト専用の in-memory input backend と subprocess entry point。
production の代替には使わず、SendInput 相当の呼出し・press/release 漏れ・時刻を観察します。
"""
from __future__ import annotations
from multiprocessing.connection import Connection
from queue import Queue
import time
from typing import Iterable
ALLOWED_INPUTS = frozenset({"W", "A", "S", "D", "ENTER", "ESCAPE", "LEFT_CLICK"})
class DryRunBackend:
    """OSへ触れず入力状態だけを再現する test double。
    allowlist と差分 press/release は production backend と同じ規則で動作します。
    """
    test_only = True
    def __init__(
        self, *, foreground_pid: int, foreground_hwnd: int, focused: bool,
        armed: bool = False, operations: Queue | None = None,
    ) -> None:
        """前面 target と初期 gate 状態をテスト fixture として設定する。
        armed の既定値は production と同じ false です。
        """
        self.foreground_pid = foreground_pid
        self.foreground_hwnd = foreground_hwnd
        self.focused = focused
        self.armed = armed
        self.pressed: set[str] = set()
        self._press_count = self._release_count = 0
        self.send_input_calls = 0
        self._arm_chord = False
        self._arm_chord_previous = False
        self._operations = operations
    @property
    def held_leak_count(self) -> int:
        """release されていない入力数を返す。
        fault/random test の終了時 invariant を単一の数値で検査できます。
        """
        return self._press_count - self._release_count
    def set_arm_chord_for_test(self, pressed: bool) -> None:
        """Ctrl+Shift+F12 chord 状態をテストから設定する。
        wire command にはせず backend fixture 内だけで hotkey edge を模擬します。
        """
        self._arm_chord = bool(pressed)
    def poll_arm_toggle(self) -> bool:
        """hotkey の false→true edge だけで arm を反転する。
        押しっぱなしの poll で連続 toggle しない実装を production と揃えます。
        """
        if self._arm_chord and not self._arm_chord_previous:
            self.armed = not self.armed
        self._arm_chord_previous = self._arm_chord
        return self.armed
    def target_is_safe(self, pid: int, hwnd: int) -> bool:
        """focus と PID/HWND の完全一致を返す。
        lease target と現在の foreground target の片方でも違えば false です。
        """
        return self.focused and (pid, hwnd) == (self.foreground_pid, self.foreground_hwnd)
    def apply_inputs(
        self, inputs: Iterable[str], *, sequence: int | None = None, monotonic_ns: int | None = None,
    ) -> None:
        """allowlisted 入力への差分操作だけを記録する。
        未許可入力は状態変更前に拒否し、同じ chord の再適用は SendInput 0件として扱います。
        """
        desired = set(inputs)
        if not desired <= ALLOWED_INPUTS:
            raise ValueError("input outside allowlist")
        if desired == self.pressed:
            return
        released, newly_pressed = self.pressed - desired, desired - self.pressed
        self.send_input_calls += 1
        self._press_count += len(newly_pressed); self._release_count += len(released)
        self.pressed = desired
        if released and self._operations is not None:
            self._operations.put({
                "kind": "release", "sequence": sequence,
                "held_count": len(self.pressed),
                "monotonic_ns": monotonic_ns if monotonic_ns is not None else time.monotonic_ns(),
            })
def run_dry_run_helper_for_test(
    connection: Connection, operations: Queue, audit_path: str, session_nonce: str,
    target_hash: str, action_hash: str, target_pid: int, target_hwnd: int,
) -> None:
    """fault test 専用 subprocess で generic helper loop を動かす。
    production launcher から参照せず、実 OS 注入なしで lease expiry を実時間計測します。
    """
    from .audit_log import AuditLog
    from .helper import HelperRuntime, run_helper_loop
    from .lease_protocol import LeaseValidator
    backend = DryRunBackend(
        foreground_pid=target_pid, foreground_hwnd=target_hwnd,
        focused=True, armed=True, operations=operations,
    )
    runtime = HelperRuntime(
        LeaseValidator(session_nonce, target_hash, action_hash, target_pid, target_hwnd),
        backend, AuditLog(audit_path),
    )
    run_helper_loop(connection, runtime, session_nonce)
