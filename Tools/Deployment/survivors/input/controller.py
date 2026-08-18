"""helper process へ semantic action lease だけを送る controller。
SendInput を所有せず、helper 死亡を検出した後は通常 action を永久に fail-closed にします。
"""
from __future__ import annotations
import atexit
import multiprocessing
from pathlib import Path
import secrets
import time
from typing import Any, Callable
from .helper import emergency_release_main, helper_main
from .lease_protocol import Lease
class HelperUnavailable(RuntimeError):
    """helper の死亡・timeout・IPC failure を表す例外。
    この状態へ遷移した controller では以後の semantic action を拒否します。
    """
class InputLeaseController:
    """semantic action と emergency key-up だけを公開する IPC client。
    arbitrary VK・text・shortcut・座標 click の method や wire payload は提供しません。
    """
    def __init__(
        self, *, target_hash: str, action_hash: str, target_pid: int, target_hwnd: int,
        audit_path: Path | str, process_target: Callable[..., None] = helper_main,
        emergency_process_target: Callable[..., None] = emergency_release_main,
    ) -> None:
        """session nonce を作り独立 helper process を default-disarmed で起動する。
        process_target 差替えは死亡検出 test 用で、production default は Win32 helper 固定です。
        """
        self._nonce = secrets.token_hex(16)
        self._target_hash = target_hash
        self._action_hash = action_hash
        self._target_pid = target_pid
        self._target_hwnd = target_hwnd
        self._sequence = 0
        self._failed = False
        self._audit_path = str(audit_path)
        self._emergency_process_target = emergency_process_target
        self._pipe_closed = False
        self._connection, child = multiprocessing.Pipe()
        self._process = multiprocessing.Process(
            target=process_target,
            args=(child, str(audit_path), self._nonce, target_hash, action_hash, target_pid, target_hwnd),
            daemon=False,  # ponytail: daemon=False で controller exit 後も helper が cleanup できる。全プロセス強制 kill (SIGKILL) は非保証、runbook 参照
        )
        self._process.start()
        child.close()
        # atexit は LIFO なので multiprocessing の non-daemon join より先に実行され、
        # pipe を閉じることで helper が EOF を受けて cleanup できる
        atexit.register(self._atexit_close_pipe)
    def send_action(self, action_index: int) -> bool:
        """0〜8 の semantic action を75ms lease として helper へ送る。
        helper が死んだ・応答しない・reject した場合は controller を sealing し、再送を禁止します。
        """
        if type(action_index) is not int or not 0 <= action_index <= 8:
            raise ValueError("action_index must be from ActionContract")
        self._ensure_available()
        self._sequence += 1
        now = time.monotonic_ns()
        lease = Lease(
            session_nonce=self._nonce, sequence=self._sequence,
            issued_monotonic_ns=now, expires_monotonic_ns=now + 75_000_000,
            target_hash=self._target_hash, action_hash=self._action_hash,
            action_index=action_index, target_pid=self._target_pid, target_hwnd=self._target_hwnd,
        )
        response = self._exchange(lease.to_wire())
        if response.get("kind") != "ack" or response.get("sequence") != self._sequence:
            self._failed = True
            raise HelperUnavailable("helper rejected semantic action")
        return response.get("applied") is True
    def emergency_release(self) -> bool:
        """helper が生存中なら emergency key-up を依頼し、死亡済みなら安全に false を返す。
        controller 自身は入力 API を呼ばず、新しい keydown を生成しません。
        """
        if not self._process.is_alive():
            self._failed = True
            process = multiprocessing.Process(target=self._emergency_process_target, args=(self._audit_path,))
            process.start()
            process.join(timeout=2.0)
            if process.is_alive():
                process.terminate(); process.join(timeout=0.2)
                return False
            return process.exitcode == 0
        try:
            response = self._exchange({"kind": "emergency_release", "session_nonce": self._nonce})
        except HelperUnavailable:
            return False
        return response.get("kind") == "released"
    def _ensure_available(self) -> None:
        """helper process と controller seal の両方を確認する。
        一度 failure を観測した instance は process が偶然見えても再利用しません。
        """
        if self._failed or not self._process.is_alive():
            self._failed = True
            raise HelperUnavailable("input helper is unavailable")
    def _exchange(self, message: dict[str, object]) -> dict[str, Any]:
        """1 command を送信し ready を除く response を期限付きで待つ。
        broken pipe・EOF・200ms timeout は helper death と同じ fail-closed 状態にします。
        """
        try:
            self._connection.send(message)
            deadline = time.monotonic() + 0.2
            while time.monotonic() < deadline:
                if not self._process.is_alive() and not self._connection.poll():
                    raise HelperUnavailable("input helper died")
                if self._connection.poll(0.005):
                    response = self._connection.recv()
                    if isinstance(response, dict) and response.get("kind") == "ready":
                        continue
                    if not isinstance(response, dict):
                        raise HelperUnavailable("invalid helper response")
                    return response
        except (BrokenPipeError, EOFError, OSError) as exc:
            self._failed = True
            raise HelperUnavailable("input helper IPC failed") from exc
        self._failed = True
        raise HelperUnavailable("input helper response timed out")
    def _atexit_close_pipe(self) -> None:
        """atexit と _shutdown 共用の pipe close。LIFO により multiprocessing join より先に実行される。
        fork で parent fd が引き継がれると EOF が届かないため shutdown sentinel を先送りしてから閉じる。
        既に閉じていれば何もしない。
        """
        if not self._pipe_closed:
            try:
                self._connection.send({"kind": "shutdown"})
            except Exception:
                pass
            try:
                self._connection.close()
            except Exception:
                pass
            self._pipe_closed = True
    def _shutdown(self) -> None:
        """内部 lifecycle cleanup として release・EOF・join を順に行う。
        public input API を増やさず、launcher/context manager 終了時だけ使用します。
        """
        self.emergency_release()
        self._atexit_close_pipe()
        self._process.join(timeout=0.3)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=0.3)
    def __enter__(self) -> "InputLeaseController":
        """launcher の with block へ controller を返す。
        block 終了時の確実な cleanup と対にする lifecycle helper です。
        """
        return self
    def __exit__(self, *_exc: object) -> None:
        """with block 終了時に emergency release と helper 停止を行う。
        通常終了・例外終了のどちらでも同じ key-up 経路を通します。
        """
        self._shutdown()
