"""durable launch ledger と組み合わせる Win32 launch broker(別 process)です。

runner とは別 process の broker だけが `CreateProcessW(CREATE_SUSPENDED)` と kill-on-job-close
Job Object を所有します。current-user SID 限定の named pipe で launch request を受け、ledger の
intent を read-back して nonce / config / executable / command hash を照合し、kernel から読んだ
PID / create time / image / file identity で attest してから resume します。
resume intent の commit 前に失敗した場合は suspended process を terminate し、activation を禁止します。
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import hmac
import json
import os
import platform
import sqlite3
import struct
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from . import campaign_schema, durable_launch_store
from .durable_launch_store import (
    Classification,
    DurableLaunchStore,
    LaunchIntent,
    LaunchStage,
    LedgerError,
    ProbeResult,
    ProcessAttestation,
    ProcessIdentity,
    current_user_sid,
    process_user_sid,
    security_descriptor_from_sddl,
    win32_check,
    win32_function,
)

BROKER_SCHEMA = "survivors.launch_broker.v1"
PIPE_PREFIX = "\\\\.\\pipe\\"
NONCE_ENV = "REINBALANCE_LAUNCH_NONCE"
FAULT_BOUNDARIES = (
    "after_intent_readback",
    "after_create_process",
    "after_attestation",
    "after_resume_intent",
    "after_resume",
    "after_confirm",
)
_REQUEST_KEYS = frozenset(
    {"kind", "schema", "attempt_id", "launch_nonce", "config_hash", "executable_hash", "command_hash"}
)
_MAX_FRAME = 65536
_H = ctypes.c_void_p
_DW = ctypes.c_uint32
_BOOL = ctypes.c_int
_LPW = ctypes.c_wchar_p
_PDW = ctypes.POINTER(ctypes.c_uint32)
_INVALID_HANDLE = ctypes.c_void_p(-1).value
_CREATE_SUSPENDED = 0x4
_CREATE_UNICODE_ENVIRONMENT = 0x400
_EXTENDED_STARTUPINFO_PRESENT = 0x80000
_CREATE_NO_WINDOW = 0x08000000
_PROC_THREAD_ATTRIBUTE_JOB_LIST = 0x2000D
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_SYNCHRONIZE = 0x100000
_GENERIC_READ = 0x80000000
_FILE_READ_ATTRIBUTES = 0x80
_FILE_SHARE_ALL = 0x7
_FILE_SHARE_READ = 0x1
_OPEN_EXISTING = 3
_FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
_PIPE_ACCESS_DUPLEX = 0x3
_FILE_FLAG_FIRST_PIPE_INSTANCE = 0x00080000
_PIPE_REJECT_REMOTE_CLIENTS = 0x8
_ERROR_INVALID_PARAMETER = 87
_ERROR_PIPE_CONNECTED = 535
_ERROR_NO_DATA = 232
_WAIT_OBJECT_0 = 0


class _FileTime(ctypes.Structure):
    """Win32 FILETIME の ctypes layout です。

    100ns 単位の時刻を上位/下位 32bit に分けて持ちます。
    """

    _fields_ = [("low", _DW), ("high", _DW)]

    @property
    def value(self) -> int:
        """64bit 整数へ結合した時刻です。

        process create time の比較に使います。
        """
        return (self.high << 32) | self.low


class _ByHandleFileInformation(ctypes.Structure):
    """Win32 BY_HANDLE_FILE_INFORMATION の ctypes layout です。

    volume serial と file index で file の実体 identity を表します。
    """

    _fields_ = [("attributes", _DW), ("created", _FileTime), ("accessed", _FileTime), ("written", _FileTime),
                ("volume_serial", _DW), ("size_high", _DW), ("size_low", _DW), ("links", _DW),
                ("index_high", _DW), ("index_low", _DW)]


class _StartupInfo(ctypes.Structure):
    """Win32 STARTUPINFOW の ctypes layout です。

    window 表示などは既定値のまま使います。
    """

    _fields_ = [("cb", _DW), ("reserved", _LPW), ("desktop", _LPW), ("title", _LPW), ("x", _DW), ("y", _DW),
                ("x_size", _DW), ("y_size", _DW), ("x_chars", _DW), ("y_chars", _DW), ("fill", _DW),
                ("flags", _DW), ("show_window", ctypes.c_ushort), ("reserved2_size", ctypes.c_ushort),
                ("reserved2", _H), ("stdin", _H), ("stdout", _H), ("stderr", _H)]


class _StartupInfoEx(ctypes.Structure):
    """Win32 STARTUPINFOEXW の ctypes layout です。

    Job list attribute を渡し、process を生成と同時に Job へ入れます。
    """

    _fields_ = [("startup", _StartupInfo), ("attributes", _H)]


class _ProcessInformation(ctypes.Structure):
    """Win32 PROCESS_INFORMATION の ctypes layout です。

    CreateProcessW が返す process/thread handle と id を受け取ります。
    """

    _fields_ = [("process", _H), ("thread", _H), ("pid", _DW), ("tid", _DW)]


class _IoCounters(ctypes.Structure):
    """Win32 IO_COUNTERS の ctypes layout です。

    Job extended limit 構造体の一部として必要なだけで値は使いません。
    """

    _fields_ = [(name, ctypes.c_ulonglong) for name in ("r_ops", "w_ops", "o_ops", "r_bytes", "w_bytes", "o_bytes")]


class _JobBasicLimit(ctypes.Structure):
    """Win32 JOBOBJECT_BASIC_LIMIT_INFORMATION の ctypes layout です。

    LimitFlags に kill-on-job-close を立てます。
    """

    _fields_ = [("process_time", ctypes.c_longlong), ("job_time", ctypes.c_longlong), ("flags", _DW),
                ("min_ws", ctypes.c_size_t), ("max_ws", ctypes.c_size_t), ("active_limit", _DW),
                ("affinity", ctypes.c_size_t), ("priority", _DW), ("scheduling", _DW)]


class _JobExtendedLimit(ctypes.Structure):
    """Win32 JOBOBJECT_EXTENDED_LIMIT_INFORMATION の ctypes layout です。

    basic limit 以外の memory 制限などは 0(無制限)のままです。
    """

    _fields_ = [("basic", _JobBasicLimit), ("io", _IoCounters), ("process_memory", ctypes.c_size_t),
                ("job_memory", ctypes.c_size_t), ("peak_process", ctypes.c_size_t), ("peak_job", ctypes.c_size_t)]


class _SecurityAttributes(ctypes.Structure):
    """Win32 SECURITY_ATTRIBUTES の ctypes layout です。

    named pipe に current-user 専用 DACL を付けるために使います。
    """

    _fields_ = [("length", _DW), ("descriptor", _H), ("inherit", _BOOL)]


class AttestationError(RuntimeError):
    """kernel identity attestation が成立しなかったことを表す例外です。

    image・file identity・Job 所属・suspend 状態のどれかが期待と違う場合に送出します。
    """


def _k(name: str, restype: Any, *argtypes: Any) -> Any:
    """kernel32 関数を signature 付きで取得します。

    durable_launch_store の共通 binder を使います。
    """
    return win32_function("kernel32", name, restype, *argtypes)


def _close(handle: Any) -> None:
    """handle を閉じます。

    None / 0 は無視します。
    """
    if handle:
        _k("CloseHandle", _BOOL, _H)(handle)


def _open_file(path: str, access: int, share: int, flags: int = 0) -> Any:
    """CreateFileW で既存 file の handle を開きます。

    失敗時は OSError です。
    """
    handle = _k("CreateFileW", _H, _LPW, _DW, _DW, _H, _DW, _DW, _H)(path, access, share, None, _OPEN_EXISTING,
                                                                     flags, None)
    if handle in (None, _INVALID_HANDLE):
        error = ctypes.get_last_error()
        raise OSError(error, f"CreateFileW failed for {path} (winerror {error})")
    return handle


def _file_identity(handle: Any) -> tuple[int, int]:
    """開いた file handle の (volume serial, file index) を返します。

    path 文字列ではなく NTFS 上の実体で file を識別します。
    """
    info = _ByHandleFileInformation()
    win32_check(_k("GetFileInformationByHandle", _BOOL, _H, ctypes.POINTER(_ByHandleFileInformation))(
        handle, ctypes.byref(info)), "GetFileInformationByHandle")
    return info.volume_serial, (info.index_high << 32) | info.index_low


def file_identity_of_path(path: str) -> tuple[int, int]:
    """path が指す file の (volume serial, file index) を返します。

    属性読み取り権限だけで開くので、実行中の image にも使えます。
    """
    handle = _open_file(path, _FILE_READ_ATTRIBUTES, _FILE_SHARE_ALL, _FILE_FLAG_BACKUP_SEMANTICS)
    try:
        return _file_identity(handle)
    finally:
        _close(handle)


def process_identity(handle: Any, pid: int) -> ProcessIdentity:
    """process handle から kernel identity を読みます。

    create time は GetProcessTimes、image は QueryFullProcessImageNameW、file identity は
    image file の volume serial / file index です。環境変数の自己申告は使いません。
    """
    created, exited, kernel, user = _FileTime(), _FileTime(), _FileTime(), _FileTime()
    ptr = ctypes.POINTER(_FileTime)
    win32_check(_k("GetProcessTimes", _BOOL, _H, ptr, ptr, ptr, ptr)(
        handle, ctypes.byref(created), ctypes.byref(exited), ctypes.byref(kernel), ctypes.byref(user)),
        "GetProcessTimes")
    image = ctypes.create_unicode_buffer(32768)
    size = _DW(32768)
    win32_check(_k("QueryFullProcessImageNameW", _BOOL, _H, _DW, _LPW, _PDW)(handle, 0, image, ctypes.byref(size)),
                "QueryFullProcessImageNameW")
    serial, index = file_identity_of_path(image.value)
    return ProcessIdentity(pid, created.value, image.value, serial, index)


def _own_identity() -> ProcessIdentity:
    """broker 自身の kernel identity を返します。

    attestation の broker_ref と job_ref の一意化に使います。
    """
    return process_identity(_k("GetCurrentProcess", _H)(), os.getpid())


def probe_identity(identity: ProcessIdentity) -> ProbeResult:
    """attested identity の process が今も同じ実体で生きているかを kernel に問い合わせます。

    PID が存在しない・終了済みなら EXITED、create time / image / file identity が違えば PID_REUSED、
    権限などで確かめられなければ UNKNOWN です。ALIVE 以外は activation できません。
    """
    handle = _k("OpenProcess", _H, _DW, _BOOL, _DW)(_PROCESS_QUERY_LIMITED_INFORMATION | _SYNCHRONIZE, 0,
                                                    identity.pid)
    if not handle:
        return ProbeResult.EXITED if ctypes.get_last_error() == _ERROR_INVALID_PARAMETER else ProbeResult.UNKNOWN
    try:
        try:
            current = process_identity(handle, identity.pid)
        except (OSError, ValueError):
            return ProbeResult.UNKNOWN
        if current.create_time != identity.create_time or current != identity:
            return ProbeResult.PID_REUSED
        if _k("WaitForSingleObject", _DW, _H, _DW)(handle, 0) == _WAIT_OBJECT_0:
            return ProbeResult.EXITED
        return ProbeResult.ALIVE
    finally:
        _close(handle)


def _hold_executable(path: str) -> tuple[Any, tuple[int, int], str]:
    """executable を書込み拒否で開いたまま hash と file identity を取ります。

    返した handle を閉じるまで他 process は file を書き換えられないので、
    hash 照合から CreateProcessW までの差し替え(TOCTOU)を防ぎます。
    """
    handle = _open_file(path, _GENERIC_READ, _FILE_SHARE_READ)
    try:
        identity = _file_identity(handle)
        digest = hashlib.sha256()
        buffer = ctypes.create_string_buffer(1 << 20)
        read = _DW()
        reader = _k("ReadFile", _BOOL, _H, _H, _DW, _PDW, _H)
        while True:
            win32_check(reader(handle, buffer, len(buffer), ctypes.byref(read), None), "ReadFile")
            if read.value == 0:
                break
            digest.update(buffer.raw[: read.value])
        return handle, identity, digest.hexdigest()
    except BaseException:
        _close(handle)
        raise


def _create_job() -> Any:
    """kill-on-job-close の Job Object を作ります。

    broker が異常終了して handle が閉じると、Job 内の process は OS が必ず終了させます。
    """
    job = win32_check(_k("CreateJobObjectW", _H, _H, _LPW)(None, None), "CreateJobObjectW")
    info = _JobExtendedLimit()
    info.basic.flags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    try:
        win32_check(_k("SetInformationJobObject", _BOOL, _H, ctypes.c_int, _H, _DW)(
            job, _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION, ctypes.addressof(info), ctypes.sizeof(info)),
            "SetInformationJobObject")
    except BaseException:
        _close(job)
        raise
    return job


def _create_suspended(intent: LaunchIntent, job: Any) -> _ProcessInformation:
    """intent の argv を CREATE_SUSPENDED で起動し、生成と同時に Job へ入れます。

    PROC_THREAD_ATTRIBUTE_JOB_LIST を使うので、生成から Job 所属までの隙間がありません。
    nonce は環境変数でも渡しますが、attestation には使いません。
    """
    size = ctypes.c_size_t()
    initialize = _k("InitializeProcThreadAttributeList", _BOOL, _H, _DW, _DW, ctypes.POINTER(ctypes.c_size_t))
    initialize(None, 1, 0, ctypes.byref(size))
    attributes = ctypes.create_string_buffer(size.value)
    win32_check(initialize(attributes, 1, 0, ctypes.byref(size)), "InitializeProcThreadAttributeList")
    try:
        jobs = (_H * 1)(job)
        win32_check(_k("UpdateProcThreadAttribute", _BOOL, _H, _DW, ctypes.c_size_t, _H, ctypes.c_size_t, _H, _H)(
            attributes, 0, _PROC_THREAD_ATTRIBUTE_JOB_LIST, ctypes.addressof(jobs), ctypes.sizeof(jobs), None, None),
            "UpdateProcThreadAttribute")
        startup = _StartupInfoEx()
        startup.startup.cb = ctypes.sizeof(startup)
        startup.attributes = ctypes.addressof(attributes)
        environment = {**os.environ, NONCE_ENV: intent.launch_nonce}
        block = "".join(f"{key}={value}\0" for key, value in sorted(environment.items(), key=lambda kv: kv[0].upper()))
        env_buffer = ctypes.create_unicode_buffer(block + "\0")
        command = ctypes.create_unicode_buffer(subprocess.list2cmdline(intent.argv))
        info = _ProcessInformation()
        flags = _CREATE_SUSPENDED | _CREATE_UNICODE_ENVIRONMENT | _EXTENDED_STARTUPINFO_PRESENT | _CREATE_NO_WINDOW
        win32_check(_k("CreateProcessW", _BOOL, _LPW, _LPW, _H, _H, _BOOL, _DW, _H, _LPW,
                       ctypes.POINTER(_StartupInfoEx), ctypes.POINTER(_ProcessInformation))(
            intent.executable_path, command, None, None, 0, flags, ctypes.addressof(env_buffer), None,
            ctypes.byref(startup), ctypes.byref(info)), "CreateProcessW")
        return info
    finally:
        _k("DeleteProcThreadAttributeList", None, _H)(attributes)


def _attest(info: _ProcessInformation, job: Any, expected_file: tuple[int, int], intent: LaunchIntent) -> ProcessIdentity:
    """生成直後の process を kernel 値だけで attest します。

    image file identity が書込み拒否で保持中の executable と一致すること、Job 所属、
    未終了、initial thread の suspend count が 1(一度も実行されていない)ことを確かめます。
    """
    identity = process_identity(info.process, info.pid)
    if (identity.volume_serial, identity.file_index) != expected_file:
        raise AttestationError("process image is not the held executable file")
    if os.path.normcase(os.path.realpath(identity.image_path)) != os.path.normcase(
            os.path.realpath(intent.executable_path)):
        raise AttestationError("process image path differs from intent executable")
    in_job = _BOOL()
    win32_check(_k("IsProcessInJob", _BOOL, _H, _H, ctypes.POINTER(_BOOL))(info.process, job, ctypes.byref(in_job)),
                "IsProcessInJob")
    if not in_job.value:
        raise AttestationError("process is not in the broker job")
    if _k("WaitForSingleObject", _DW, _H, _DW)(info.process, 0) == _WAIT_OBJECT_0:
        raise AttestationError("process exited before attestation")
    previous = _k("SuspendThread", _DW, _H)(info.thread)
    if previous != 0xFFFFFFFF:
        _k("ResumeThread", _DW, _H)(info.thread)
    if previous != 1:
        raise AttestationError(f"initial thread suspend count is {previous}, not 1")
    return identity


def pipe_sddl(sid: str) -> str:
    """named pipe 用の current-user 専用 DACL を SDDL で返します。

    継承なしの protected DACL で、current user の SID にだけ全権を与えます。
    """
    return f"D:P(A;;GA;;;{sid})"


def _validate_pipe_name(name: str) -> str:
    """local named pipe 名だけを受け付けます。

    `\\\\.\\pipe\\` 以外(remote pipe 含む)は拒否します。
    """
    if not isinstance(name, str) or not name.startswith(PIPE_PREFIX) or "\\" in name[len(PIPE_PREFIX):]:
        raise ValueError(f"pipe name must be {PIPE_PREFIX}<name>")
    return name


def build_launch_request(intent: LaunchIntent) -> dict[str, Any]:
    """controller が broker へ送る launch request を intent から作ります。

    broker はこの値を ledger の intent と read-back 照合します。
    """
    return {
        "kind": "launch", "schema": BROKER_SCHEMA, "attempt_id": intent.attempt_id,
        "launch_nonce": intent.launch_nonce, "config_hash": intent.config_hash,
        "executable_hash": intent.executable_hash, "command_hash": intent.command_hash,
    }


def parse_launch_request(message: Any) -> dict[str, str]:
    """launch request を strict に検証します。

    key の完全一致・schema・digest 形式を要求し、違反は ValueError です。
    """
    if not isinstance(message, Mapping) or set(message) != _REQUEST_KEYS:
        raise ValueError(f"launch request must have exactly keys {sorted(_REQUEST_KEYS)}")
    if message["kind"] != "launch" or message["schema"] != BROKER_SCHEMA:
        raise ValueError("unsupported launch request kind or schema")
    if not isinstance(message["attempt_id"], str) or not message["attempt_id"]:
        raise ValueError("attempt_id must be a non-empty string")
    for name in ("launch_nonce", "config_hash", "executable_hash", "command_hash"):
        durable_launch_store._digest(message[name], name)
    return dict(message)


def _result(attempt_id: str, status: str, *, process_ref: str | None = None, job_ref: str | None = None,
            failure_reason: str | None = None) -> dict[str, Any]:
    """broker response を組み立てます。

    status は confirmed / proven_no_process / uncertain / rejected のどれかです。
    """
    return {"kind": "launch_result", "attempt_id": attempt_id, "status": status, "process_ref": process_ref,
            "job_ref": job_ref, "failure_reason": failure_reason}


class LaunchBroker:
    """ledger と Win32 process/job API を束ねる launch protocol 実装です。

    intent read-back → suspended 生成 → attestation commit → resume intent commit → resume →
    生存確認 → confirm commit の順に進み、どこで失敗しても activation 可能な状態を作りません。
    confirm 済み process の Job handle は broker が close まで保持します。
    """

    def __init__(self, store: DurableLaunchStore, *, fault_hold_at: str | None = None,
                 fault_dir: str | os.PathLike[str] | None = None) -> None:
        """ledger と fault injection 設定を受け取ります。

        fault_hold_at は crash 境界の再現テスト専用で、指定境界で停止して外部 kill を待ちます。
        """
        if fault_hold_at is not None and (fault_hold_at not in FAULT_BOUNDARIES or fault_dir is None):
            raise ValueError("fault_hold_at must be a known boundary and requires fault_dir")
        self._store = store
        self._fault_hold_at = fault_hold_at
        self._fault_dir = Path(fault_dir) if fault_dir is not None else None
        self._launched: dict[str, tuple[Any, Any, Any]] = {}
        self._self = _own_identity()
        self.broker_ref = self._self.process_ref

    def close(self) -> None:
        """保持中の Job / process handle を閉じます。

        kill-on-job-close により confirm 済み helper もここで終了します。
        """
        for job, process, thread in self._launched.values():
            _close(thread)
            _close(process)
            _close(job)
        self._launched.clear()

    def handle_message(self, message: Any) -> dict[str, Any]:
        """named pipe から受けた1 message を処理します。

        launch と shutdown 以外の kind・不正 request は状態を変えずに rejected を返します。
        """
        kind = message.get("kind") if isinstance(message, Mapping) else None
        if kind == "shutdown" and set(message) == {"kind"}:
            return {"kind": "shutdown_ack"}
        try:
            request = parse_launch_request(message)
        except ValueError as exc:
            return {"kind": "rejected", "error": str(exc)}
        return self.launch(request)

    def _hold(self, boundary: str, attempt_id: str, info: _ProcessInformation | None = None) -> None:
        """fault injection 用に指定境界で停止します。

        到達 marker(生成済み process の kernel identity を含む)を書いて release file を待つ間に、
        テストが別 process から kill します。指定境界以外では何もしません。
        """
        if boundary != self._fault_hold_at or self._fault_dir is None:
            return
        facts = {"process": process_identity(info.process, info.pid).to_payload()} if info is not None else {}
        marker = self._fault_dir / f"{boundary}.reached"
        temporary = marker.with_suffix(".tmp")
        temporary.write_text(json.dumps({"boundary": boundary, "attempt_id": attempt_id,
                                         "broker_pid": os.getpid(), **facts}), encoding="utf-8")
        os.replace(temporary, marker)
        deadline = time.monotonic() + 120.0
        while not (self._fault_dir / "release").exists():
            if time.monotonic() > deadline:
                raise TimeoutError(f"fault hold at {boundary} was never released")
            time.sleep(0.01)

    def _fail(self, attempt_id: str, job: Any, info: _ProcessInformation | None, reason: str,
              commit: Callable[[str, str], None]) -> dict[str, Any]:
        """失敗時に Job ごと process を終了させ、terminal row を試み、ledger の分類を返します。

        commit 自体が失敗しても reconcile で ledger 上の段階から分類し直すので、
        resume 前なら proven_no_process、resume 後なら uncertain に必ず落ちます。
        """
        if job:
            _k("TerminateJobObject", _BOOL, _H, ctypes.c_uint)(job, 1)
        if info is not None and info.process:
            _k("WaitForSingleObject", _DW, _H, _DW)(info.process, 10000)
        try:
            commit(attempt_id, reason)
        except LedgerError:
            pass
        try:
            status = self._store.reconcile(attempt_id, probe_identity).classification
        except LedgerError:
            status = Classification.UNCERTAIN
        if status is Classification.CONFIRMED:
            status = Classification.UNCERTAIN
        return _result(attempt_id, status.value, failure_reason=reason)

    def launch(self, request: Mapping[str, str]) -> dict[str, Any]:
        """検証済み request で durable launch protocol を1回実行します。

        ledger の attempt が LAUNCH_INTENT で止まっていない(重複・replacement)場合や
        request と intent の hash が一致しない場合は、ledger を変えずに rejected を返します。
        """
        attempt = request["attempt_id"]
        try:
            history = self._store.history(attempt)
        except LedgerError as exc:
            return _result(attempt, "rejected", failure_reason=f"ledger read-back failed: {exc}")
        if history.last_stage is not LaunchStage.LAUNCH_INTENT:
            return _result(attempt, "rejected", failure_reason=f"launch is not pending ({history.last_stage.value})")
        intent = history.intent
        for name, expected in (("launch_nonce", intent.launch_nonce), ("config_hash", intent.config_hash),
                               ("executable_hash", intent.executable_hash), ("command_hash", intent.command_hash)):
            if not hmac.compare_digest(request[name], expected):
                return _result(attempt, "rejected", failure_reason=f"{name} read-back mismatch")
        self._hold("after_intent_readback", attempt)
        executable = job = info = None
        gate = self._store.commit_gate_failure
        uncertain = self._store.commit_uncertain
        try:
            try:
                executable, expected_file, digest = _hold_executable(intent.executable_path)
            except OSError as exc:
                return self._fail(attempt, None, None, f"executable_unreadable: {exc}", gate)
            if not hmac.compare_digest(digest, intent.executable_hash):
                return self._fail(attempt, None, None, "executable_hash_mismatch", gate)
            try:
                job = _create_job()
                info = _create_suspended(intent, job)
            except OSError as exc:
                return self._fail(attempt, job, None, f"create_process_failed: {exc}", gate)
            self._hold("after_create_process", attempt, info)
            job_ref = f"job:{self.broker_ref}:{attempt}"
            try:
                identity = _attest(info, job, expected_file, intent)
                self._store.commit_attestation(attempt, ProcessAttestation(identity, job_ref, self.broker_ref))
            except (OSError, ValueError, AttestationError, LedgerError) as exc:
                return self._fail(attempt, job, info, f"attestation_failed: {exc}", gate)
            self._hold("after_attestation", attempt, info)
            try:
                self._store.commit_resume_intent(attempt, identity.process_ref)
            except LedgerError as exc:
                return self._fail(attempt, job, info, f"resume_intent_commit_failed: {exc}", gate)
            self._hold("after_resume_intent", attempt, info)
            if _k("ResumeThread", _DW, _H)(info.thread) != 1:
                return self._fail(attempt, job, info, "resume_thread_failed", uncertain)
            self._hold("after_resume", attempt, info)
            state = probe_identity(identity)
            if state is not ProbeResult.ALIVE:
                return self._fail(attempt, job, info, f"post_resume_probe_{state.value}", uncertain)
            try:
                self._store.commit_confirmation(attempt, identity.process_ref)
            except LedgerError as exc:
                return self._fail(attempt, job, info, f"confirm_commit_failed: {exc}", uncertain)
            self._hold("after_confirm", attempt, info)
            self._launched[attempt] = (job, info.process, info.thread)
            job = info = None
            return _result(attempt, Classification.CONFIRMED.value, process_ref=identity.process_ref, job_ref=job_ref)
        finally:
            _close(executable)
            if info is not None:
                _close(info.thread)
                _close(info.process)
            _close(job)


def _read_exact(read: Callable[[int], bytes], size: int) -> bytes:
    """指定 byte 数を読み切ります。

    途中で EOF になったら ValueError です。
    """
    data = b""
    while len(data) < size:
        chunk = read(size - len(data))
        if not chunk:
            raise ValueError("pipe closed mid-frame")
        data += chunk
    return data


def _decode_frame(read: Callable[[int], bytes]) -> Any:
    """4byte little-endian 長 + UTF-8 JSON の frame を1つ読みます。

    上限を超える長さは拒否します。
    """
    (size,) = struct.unpack("<I", _read_exact(read, 4))
    if size > _MAX_FRAME:
        raise ValueError("frame too large")
    return json.loads(_read_exact(read, size).decode("utf-8"))


def _encode_frame(message: Mapping[str, Any]) -> bytes:
    """message を長さ prefix 付き frame に変換します。

    上限を超える message は送りません。
    """
    body = json.dumps(message, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if len(body) > _MAX_FRAME:
        raise ValueError("frame too large")
    return struct.pack("<I", len(body)) + body


def _pipe_reader(pipe: Any) -> Callable[[int], bytes]:
    """server 側 pipe handle から読む関数を作ります。

    ReadFile の同期呼び出しで要求 byte 数まで読みます。
    """
    reader = _k("ReadFile", _BOOL, _H, _H, _DW, _PDW, _H)

    def read(size: int) -> bytes:
        buffer = ctypes.create_string_buffer(size)
        count = _DW()
        if not reader(pipe, buffer, size, ctypes.byref(count), None):
            return b""
        return buffer.raw[: count.value]

    return read


def serve(pipe_name: str, broker: LaunchBroker, *, max_requests: int | None = None) -> None:
    """current-user 専用 named pipe で request を順に処理します。

    pipe は FIRST_PIPE_INSTANCE・REJECT_REMOTE_CLIENTS・protected DACL で作り、接続ごとに
    client process の token SID を照合します。shutdown request か max_requests 到達で終了します。
    """
    _validate_pipe_name(pipe_name)
    sid = current_user_sid()
    descriptor = security_descriptor_from_sddl(pipe_sddl(sid))
    try:
        attributes = _SecurityAttributes(ctypes.sizeof(_SecurityAttributes), descriptor, 0)
        pipe = _k("CreateNamedPipeW", _H, _LPW, _DW, _DW, _DW, _DW, _DW, _DW, ctypes.POINTER(_SecurityAttributes))(
            pipe_name, _PIPE_ACCESS_DUPLEX | _FILE_FLAG_FIRST_PIPE_INSTANCE, _PIPE_REJECT_REMOTE_CLIENTS, 1,
            _MAX_FRAME, _MAX_FRAME, 0, ctypes.byref(attributes))
    finally:
        durable_launch_store._local_free(descriptor)
    if pipe in (None, _INVALID_HANDLE):
        error = ctypes.get_last_error()
        raise OSError(error, f"CreateNamedPipeW failed (winerror {error})")
    connect = _k("ConnectNamedPipe", _BOOL, _H, _H)
    writer = _k("WriteFile", _BOOL, _H, ctypes.c_char_p, _DW, _PDW, _H)
    handled = 0
    try:
        while max_requests is None or handled < max_requests:
            if not connect(pipe, None):
                error = ctypes.get_last_error()
                if error == _ERROR_NO_DATA:
                    # client が接続前に open→close した。切断して次の接続を待つ(しないと毎回 232 で busy-loop)。
                    _k("DisconnectNamedPipe", _BOOL, _H)(pipe)
                    continue
                if error != _ERROR_PIPE_CONNECTED:
                    raise OSError(error, f"ConnectNamedPipe failed (winerror {error})")
            stop = False
            try:
                client = _DW()
                win32_check(_k("GetNamedPipeClientProcessId", _BOOL, _H, _PDW)(pipe, ctypes.byref(client)),
                            "GetNamedPipeClientProcessId")
                if process_user_sid(client.value) != sid:
                    response: dict[str, Any] = {"kind": "rejected", "error": "client is not the current user"}
                else:
                    message = _decode_frame(_pipe_reader(pipe))
                    response = broker.handle_message(message)
                    stop = response.get("kind") == "shutdown_ack"
                frame = _encode_frame(response)
                written = _DW()
                writer(pipe, frame, len(frame), ctypes.byref(written), None)
                _k("FlushFileBuffers", _BOOL, _H)(pipe)
            except (OSError, ValueError):
                pass
            finally:
                _k("DisconnectNamedPipe", _BOOL, _H)(pipe)
            handled += 1
            if stop:
                break
    finally:
        _close(pipe)


def request_broker(pipe_name: str, message: Mapping[str, Any], *, broker_pid: int,
                   timeout_s: float = 30.0) -> dict[str, Any]:
    """broker へ1 request を送り response を返します(controller 側 client)。

    接続先 pipe の server PID が起動した broker の PID と一致し、その token SID が
    current user であることを確かめてから request を書きます。
    """
    import msvcrt

    _validate_pipe_name(pipe_name)
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            stream = open(pipe_name, "r+b", buffering=0)
            break
        except OSError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.02)
    with stream:
        server = _DW()
        win32_check(_k("GetNamedPipeServerProcessId", _BOOL, _H, _PDW)(
            msvcrt.get_osfhandle(stream.fileno()), ctypes.byref(server)), "GetNamedPipeServerProcessId")
        if server.value != broker_pid or process_user_sid(server.value) != current_user_sid():
            raise PermissionError(f"pipe server {server.value} is not the expected broker {broker_pid}")
        stream.write(_encode_frame(message))
        return _decode_frame(stream.read)


def reconcile_ledger(store: DurableLaunchStore) -> tuple[Any, ...]:
    """再起動後に ledger 全体を kernel probe 付きで reconcile します。

    controller / broker の crash 後に呼び、confirmed / proven-no-process / uncertain を確定します。
    """
    return store.reconcile_all(probe_identity)


def _sha256_file(path: str) -> str:
    """file の SHA-256 を返します。

    broker build hash の記録に使います。
    """
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def integration_environment(store: DurableLaunchStore) -> dict[str, Any]:
    """integration result に残す OS / filesystem / SQLite / broker build hash を集めます。

    どの環境・どの build で durability を検証したかを後から照合できるようにします。
    """
    windows = list(sys.getwindowsversion()[:4]) if os.name == "nt" else None
    return {
        "os": {"platform": platform.platform(), "machine": platform.machine(), "windows_version": windows},
        "filesystem": dict(store.verdict.facts),
        "storage_verdict": store.verdict.label,
        "sqlite": {**store.pragmas, "module_sqlite_version": sqlite3.sqlite_version},
        "python": sys.version,
        "build_hashes": {
            "win32_launch_broker": _sha256_file(__file__),
            "durable_launch_store": _sha256_file(durable_launch_store.__file__),
            "campaign_schema": _sha256_file(campaign_schema.__file__),
        },
    }


def main(argv: list[str] | None = None) -> int:
    """broker process の entry point です。

    ledger を開けない(support 外 storage・破損)場合は起動しません。
    `--test-fault-hold-at` は crash 境界テスト専用で、formal run では指定しません。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--pipe", required=True)
    parser.add_argument("--test-fault-hold-at", choices=FAULT_BOUNDARIES)
    parser.add_argument("--test-fault-dir")
    args = parser.parse_args(argv)
    with DurableLaunchStore(args.ledger) as store:
        broker = LaunchBroker(store, fault_hold_at=args.test_fault_hold_at, fault_dir=args.test_fault_dir)
        try:
            serve(args.pipe, broker)
        finally:
            broker.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
