"""controller の helper 死亡時制約と audit 完全性の受入テスト。
helper が利用不能になった後に通常 keydown が進まず、時刻付き記録が残ることを確認します。
"""
from __future__ import annotations
import json
import time
import pytest
from survivors.input.audit_log import AuditLog
from survivors.input.controller import HelperUnavailable, InputLeaseController
from survivors.input.dry_run_backend import DryRunBackend
from survivors.input.helper import HelperRuntime
from survivors.input.lease_protocol import Lease, LeaseValidator
def _exit_immediately(_connection, *_args) -> None:
    """起動直後に落ちる helper を模擬する。
    controller の死亡検出を deterministic に試すための subprocess entry point です。
    """
    return None
def _record_emergency(audit_path: str) -> None:
    """release-only helper の起動を marker file に記録する。
    OS API を使わず controller の死亡後 key-up 経路を検査します。
    """
    __import__("pathlib").Path(audit_path + ".released").touch()
def test_controller_blocks_actions_after_helper_death_but_emergency_release_is_safe(tmp_path) -> None:
    """helper 死亡後は semantic action を拒否し emergency release だけを安全に許す。
    controller 自身が OS 入力を代行せず、死んだ IPC への通常 action を確実に止めます。
    """
    controller = InputLeaseController(
        target_hash="a" * 64,
        action_hash="b" * 64,
        target_pid=42,
        target_hwnd=84,
        audit_path=tmp_path / "dead-audit.jsonl",
        process_target=_exit_immediately,
        emergency_process_target=_record_emergency,
    )
    deadline = time.monotonic() + 1.0
    while controller._process.is_alive() and time.monotonic() < deadline:
        time.sleep(0.001)
    with pytest.raises(HelperUnavailable):
        controller.send_action(0)
    assert controller.emergency_release() is True
    assert (tmp_path / "dead-audit.jsonl.released").is_file()
def test_audit_jsonl_records_event_ack_and_release_timestamps(tmp_path) -> None:
    """audit JSONL に event・ack・release の各 timestamp を残す。
    受信、適用確認、解放の順序を後から追えるよう、全3時刻の存在と単調な順番を確認します。
    """
    path = tmp_path / "audit.jsonl"
    backend = DryRunBackend(
        foreground_pid=42, foreground_hwnd=84, focused=True, armed=True
    )
    runtime = HelperRuntime(
        LeaseValidator("1" * 32, "a" * 64, "b" * 64, 42, 84), backend, AuditLog(path)
    )
    lease = Lease(
        session_nonce="1" * 32,
        sequence=1,
        issued_monotonic_ns=1_000_000_000,
        expires_monotonic_ns=1_075_000_000,
        target_hash="a" * 64,
        action_hash="b" * 64,
        action_index=0,
        target_pid=42,
        target_hwnd=84,
    )
    runtime.handle_lease(lease, 1_001_000_000)
    runtime.tick(1_075_000_000)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    lease_row = next(row for row in rows if row["event"] == "lease_ack")
    release_row = next(row for row in rows if row["event"] == "release")
    assert lease_row["event_timestamp_ns"] <= lease_row["ack_timestamp_ns"]
    assert release_row["release_timestamp_ns"] >= lease_row["event_timestamp_ns"]
