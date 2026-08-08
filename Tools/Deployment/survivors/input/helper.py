"""semantic lease を検証して OS backend を所有する helper process runtime。
短い poll loop が focus/arm/expiry を監視し、controller 停止時にも held input を解放します。
"""
from __future__ import annotations
from multiprocessing.connection import Connection
import time
from typing import Any
from survivors.action_semantics import load_action_contract
from .audit_log import AuditLog
from .lease_protocol import Lease, LeaseValidator
class HelperRuntime:
    """protocol・target gate・chord 適用・解放を束ねる state machine。
    backend は production Win32 または明示的な test double ですが、同じ安全判断を通ります。
    """
    def __init__(self, validator: LeaseValidator, backend: Any, audit: AuditLog) -> None:
        """validator と backend を固定して action contract を読み込む。
        chord は既存 ActionContract だけから構築し、独自の方向表を持ちません。
        """
        self._validator = validator
        self._backend = backend
        self._audit = audit
        self._chords = {row.index: frozenset(row.wasd_chord) for row in load_action_contract().rows}
        self._active_expiry_ns: int | None = None
        self._active_sequence: int | None = None
        self._active_target: tuple[int, int] | None = None
    def handle_lease(self, lease: Lease, now_ns: int | None = None) -> dict[str, object]:
        """lease を全検証し、安全 gate が開いた時だけ semantic chord を適用する。
        protocol 受理と OS 注入可否を分けて audit し、gate 不一致時は新規 keydown を発生させません。
        """
        now = time.monotonic_ns() if now_ns is None else now_ns
        event_timestamp = time.time_ns()
        try:
            self._validator.accept(lease, now)
        except ValueError as exc:
            # protocol/binding 違反は gate と区別してそのまま rejected で返し状態を変えない
            return {"kind": "rejected", "sequence": lease.sequence, "error": str(exc)}
        self._backend.poll_arm_toggle()
        safe = self._backend.armed and self._backend.target_is_safe(lease.target_pid, lease.target_hwnd)
        if safe:
            self._backend.apply_inputs(self._chords[lease.action_index], sequence=lease.sequence, monotonic_ns=now)
            self._active_expiry_ns = lease.expires_monotonic_ns
            self._active_sequence = lease.sequence
            self._active_target = (lease.target_pid, lease.target_hwnd)
        else:
            self._release("gate_closed", lease.sequence, now, record_when_empty=False)
        ack_timestamp = time.time_ns()
        self._audit.write(
            "lease_ack", sequence=lease.sequence, action_index=lease.action_index,
            applied=safe, event_timestamp_ns=event_timestamp, ack_timestamp_ns=ack_timestamp,
        )
        return {"kind": "ack", "sequence": lease.sequence, "applied": safe}
    def tick(self, now_ns: int | None = None) -> None:
        """arm/focus/expiry を poll し、閉じた gate の held input を解放する。
        75ms は user-space helper が scheduling される fault 条件での観測目標で、OS-wide 保証ではありません。
        """
        now = time.monotonic_ns() if now_ns is None else now_ns
        self._backend.poll_arm_toggle()
        if self._active_sequence is None:
            return
        target_safe = self._active_target is not None and self._backend.target_is_safe(*self._active_target)
        if not self._backend.armed or not target_safe:
            self._release("gate_closed", self._active_sequence, now)
        elif self._active_expiry_ns is not None and now >= self._active_expiry_ns:
            self._release("lease_expired", self._active_sequence, now)
    def emergency_release(self, sequence: int | None = None) -> None:
        """保持中の allowlisted input を即時 key-up する。
        controller から許可される唯一の action 以外の command で、新しい keydown は生成しません。
        """
        self._release("emergency", sequence, time.monotonic_ns())
    def _release(
        self, reason: str, sequence: int | None, now_ns: int, *, record_when_empty: bool = True,
    ) -> None:
        """held input を空集合へ遷移し release audit を残す。
        gate test の未保持状態では backend を呼ばず、SendInput 0件の不変条件を維持します。
        """
        had_held = bool(self._backend.pressed)
        if had_held:
            self._backend.apply_inputs(set(), sequence=sequence, monotonic_ns=now_ns)
        if had_held or record_when_empty:
            self._audit.write(
                "release", sequence=sequence, reason=reason,
                release_timestamp_ns=time.time_ns(), release_monotonic_ns=now_ns,
            )
        self._active_expiry_ns = None
        self._active_sequence = None
        self._active_target = None
def run_helper_loop(connection: Connection, runtime: HelperRuntime, session_nonce: str) -> None:
    """IPC を2ms間隔で待ち、lease と emergency release だけを処理する。
    EOF・例外・process 終了の finally でも release を試み、controller 障害から独立して cleanup します。
    """
    connection.send({"kind": "ready"})
    try:
        while True:
            runtime.tick()
            if not connection.poll(0.002):
                continue
            try:
                message = connection.recv()
            except EOFError:
                break
            # fork で parent fd を継承した場合 EOF が届かないため sentinel で終了する
            if isinstance(message, dict) and message.get("kind") == "shutdown":
                break
            if isinstance(message, dict) and set(message) == {"kind", "session_nonce"} and message.get("kind") == "emergency_release":
                if message["session_nonce"] != session_nonce:
                    connection.send({"kind": "error", "error": "wrong session nonce"})
                    continue
                runtime.emergency_release()
                connection.send({"kind": "released"})
                continue
            try:
                lease = Lease.from_wire(message)
                connection.send(runtime.handle_lease(lease))
            except (TypeError, ValueError) as exc:
                connection.send({"kind": "rejected", "error": str(exc)})
    finally:
        runtime.emergency_release()
        connection.close()
def helper_main(
    connection: Connection, audit_path: str, session_nonce: str, target_hash: str,
    action_hash: str, target_pid: int, target_hwnd: int,
) -> None:
    """production helper process 内だけで Win32 SendInput backend を生成する。
    controller へ backend object を渡さず、OS 入力 API の所有権を process 境界で隔離します。
    """
    from .win32_backend import Win32InputBackend
    runtime = HelperRuntime(
        LeaseValidator(session_nonce, target_hash, action_hash, target_pid, target_hwnd),
        Win32InputBackend(), AuditLog(audit_path),
    )
    run_helper_loop(connection, runtime, session_nonce)
def emergency_release_main(audit_path: str) -> None:
    """異常終了した helper に代わる release-only process entry point。
    新しい keydown は作らず、allowlist 全体の key-up と audit だけを実行します。
    """
    from .win32_backend import Win32InputBackend
    backend = Win32InputBackend()
    backend.emergency_release_all()
    AuditLog(audit_path).write("release", sequence=None, reason="helper_death_emergency",
                               release_timestamp_ns=time.time_ns(), release_monotonic_ns=time.monotonic_ns())
