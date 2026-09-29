"""Survivors launch の durable ledger(SQLite WAL/FULL)を提供します。

launch intent → process attestation → resume intent → launch confirm を append-only に記録し、
crash 後は confirmed / proven-no-process / uncertain の3分類へ reconcile します。
保存先は local fixed NTFS かつ current-user ACL だけを受け付け、それ以外は
formal-ineligible として例外なく fail closed にします。Win32 API は Windows 上でだけ呼びます。
"""

from __future__ import annotations

import ctypes
import functools
import json
import os
import re
import secrets
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from reinbalance_survivors_contracts.canonical_json import canonical_hash, canonical_json_bytes

from .campaign_schema import CampaignEvent, EventType

LEDGER_SCHEMA_ID = "survivors.durable_launch.v1"
LEDGER_SCHEMA_VERSION = 1
LEDGER_FILENAME = "launch_ledger.sqlite3"
_GENESIS = "0" * 64
_SHA256 = re.compile(r"[0-9a-f]{64}")
_DRIVE_NAMES = {0: "unknown", 1: "no_root_dir", 2: "removable", 3: "fixed", 4: "remote", 5: "cdrom", 6: "ramdisk"}
_OWNER_SECURITY_INFORMATION = 0x1
_DACL_SECURITY_INFORMATION = 0x4
_PROTECTED_DACL_SECURITY_INFORMATION = 0x80000000
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_TOKEN_QUERY = 0x8
_TOKEN_USER = 1
_H = ctypes.c_void_p
_DW = ctypes.c_uint32
_BOOL = ctypes.c_int
_LPW = ctypes.c_wchar_p
_PDW = ctypes.POINTER(ctypes.c_uint32)


class LedgerError(RuntimeError):
    """durable launch ledger の全失敗の基底例外です。

    呼び出し側はこの型を捕まえたら launch を activation してはいけません。
    """


class UnsupportedStorageError(LedgerError):
    """support envelope 外の保存先を formal-ineligible として拒否した例外です。

    UNC・network・removable・non-NTFS・ACL 不一致などが該当し、verdict に理由が残ります。
    """

    def __init__(self, verdict: StorageVerdict) -> None:
        """拒否理由を含む verdict を保持します。

        message には理由の一覧を並べ、ログだけでも原因を追えるようにします。
        """
        super().__init__(f"formal-ineligible storage: {', '.join(verdict.reasons)}")
        self.verdict = verdict


class LedgerIntegrityError(LedgerError):
    """PRAGMA read-back・integrity_check・hash chain・row 形式の破損を表す例外です。

    ledger を信用できない状態なので、reconcile も含め全操作を止めます。
    """


class LedgerOrderError(LedgerError):
    """duplicate または順序外の mutation を拒否した例外です。

    intent→attestation→resume→confirm の順序と一回限りの記録を守ります。
    """


class LedgerIdentityError(LedgerError):
    """identity の重複・束縛不一致を拒否した例外です。

    同じ slot の replacement launch や PID reuse による差し替えがここで止まります。
    """


class LedgerCommitError(LedgerError):
    """SQLite の commit・書込み失敗(disk full 等)を表す例外です。

    transaction は rollback 済みで、失敗した段階の row は残りません。
    """


class LaunchStage(StrEnum):
    """ledger row の段階名です。

    enum 値が SQLite の stage 列にそのまま保存されます。
    """

    LAUNCH_INTENT = "LAUNCH_INTENT"
    PROCESS_ATTESTED = "PROCESS_ATTESTED"
    RESUME_INTENT = "RESUME_INTENT"
    PROCESS_LAUNCH_CONFIRMED = "PROCESS_LAUNCH_CONFIRMED"
    FORMAL_RUN_ACTIVATED = "FORMAL_RUN_ACTIVATED"
    LAUNCH_GATE_FAILED = "LAUNCH_GATE_FAILED"
    LAUNCH_UNCERTAIN = "LAUNCH_UNCERTAIN"


_S = LaunchStage
# resume intent 前だけが gate failure(= process は一度も実行されていない)を取れる。
# resume intent 以降は process が走った可能性があるため uncertain しか取れない。
_NEXT: dict[LaunchStage | None, frozenset[LaunchStage]] = {
    None: frozenset({_S.LAUNCH_INTENT}),
    _S.LAUNCH_INTENT: frozenset({_S.PROCESS_ATTESTED, _S.LAUNCH_GATE_FAILED}),
    _S.PROCESS_ATTESTED: frozenset({_S.RESUME_INTENT, _S.LAUNCH_GATE_FAILED}),
    _S.RESUME_INTENT: frozenset({_S.PROCESS_LAUNCH_CONFIRMED, _S.LAUNCH_UNCERTAIN}),
    _S.PROCESS_LAUNCH_CONFIRMED: frozenset({_S.FORMAL_RUN_ACTIVATED, _S.LAUNCH_UNCERTAIN}),
}
_INTENT_KEYS = frozenset(
    {
        "campaign_manifest_hash", "slot_id", "attempt_id", "reserved_run_id", "gameplay_attempt_id",
        "launch_nonce", "executable_path", "executable_hash", "build_hash", "config_hash", "argv",
        "command_hash",
    }
)
_PAYLOAD_KEYS = {
    _S.LAUNCH_INTENT: _INTENT_KEYS,
    _S.PROCESS_ATTESTED: frozenset({"process", "job_ref", "broker_ref"}),
    _S.RESUME_INTENT: frozenset({"process_ref"}),
    _S.PROCESS_LAUNCH_CONFIRMED: frozenset({"process_ref"}),
    _S.FORMAL_RUN_ACTIVATED: frozenset({"activation_source"}),
    _S.LAUNCH_GATE_FAILED: frozenset({"failure_reason"}),
    _S.LAUNCH_UNCERTAIN: frozenset({"failure_reason"}),
}
_SCHEMA_SQL = f"""
CREATE TABLE IF NOT EXISTS ledger_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS launch_rows(
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    attempt_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    payload TEXT NOT NULL,
    prev_hash TEXT NOT NULL,
    row_hash TEXT NOT NULL UNIQUE,
    UNIQUE(attempt_id, stage)
);
CREATE TABLE IF NOT EXISTS identity_claims(
    kind TEXT NOT NULL, value TEXT NOT NULL, attempt_id TEXT NOT NULL, PRIMARY KEY(kind, value)
);
CREATE TRIGGER IF NOT EXISTS launch_rows_no_update BEFORE UPDATE ON launch_rows
    BEGIN SELECT RAISE(ABORT, 'launch ledger is append-only'); END;
CREATE TRIGGER IF NOT EXISTS launch_rows_no_delete BEFORE DELETE ON launch_rows
    BEGIN SELECT RAISE(ABORT, 'launch ledger is append-only'); END;
CREATE TRIGGER IF NOT EXISTS identity_claims_no_update BEFORE UPDATE ON identity_claims
    BEGIN SELECT RAISE(ABORT, 'launch ledger is append-only'); END;
CREATE TRIGGER IF NOT EXISTS identity_claims_no_delete BEFORE DELETE ON identity_claims
    BEGIN SELECT RAISE(ABORT, 'launch ledger is append-only'); END;
CREATE TRIGGER IF NOT EXISTS ledger_meta_no_update BEFORE UPDATE ON ledger_meta
    BEGIN SELECT RAISE(ABORT, 'launch ledger is append-only'); END;
CREATE TRIGGER IF NOT EXISTS ledger_meta_no_delete BEFORE DELETE ON ledger_meta
    BEGIN SELECT RAISE(ABORT, 'launch ledger is append-only'); END;
INSERT OR IGNORE INTO ledger_meta(key, value) VALUES('schema', '{LEDGER_SCHEMA_ID}');
PRAGMA user_version = {LEDGER_SCHEMA_VERSION};
"""


def _digest(value: Any, name: str) -> str:
    """lowercase SHA-256 hex の形式を検証します。

    256-bit nonce もこの形式(64桁 hex)で受け付けます。
    """
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase 64-digit hex digest")
    return value


def _text(value: Any, name: str) -> str:
    """空でない文字列だけを受け付けます。

    identity や failure reason の空文字・非文字列を拒否します。
    """
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _exact(value: Any, keys: frozenset[str], name: str) -> Mapping[str, Any]:
    """object の key 集合が期待と完全一致することを確かめます。

    欠落も未知 key も partial row / 改竄として拒否します。
    """
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{name} must have exactly keys {sorted(keys)}")
    return value


def _int(value: Any, name: str, minimum: int) -> int:
    """bool を除く整数と下限を検証します。

    PID・create time・file index などの kernel 値に使います。
    """
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def new_launch_nonce() -> str:
    """launch intent 用の 256-bit nonce を生成します。

    OS の暗号学的乱数から 32 byte を取り、64桁の hex にします。
    """
    return secrets.token_hex(32)


def command_hash(argv: Sequence[str]) -> str:
    """argv 配列の canonical hash を返します。

    broker は ledger の argv からこの値を再計算して read-back 照合します。
    """
    return canonical_hash(list(argv))


@dataclass(frozen=True)
class LaunchIntent:
    """LAUNCH_INTENT row に束縛する launch 条件一式です。

    slot・予約 id・nonce・executable/build/config/command hash を1回だけ commit します。
    argv[0] は executable_path と一致しなければなりません。
    """

    campaign_manifest_hash: str
    slot_id: int
    attempt_id: str
    reserved_run_id: str
    gameplay_attempt_id: str
    launch_nonce: str
    executable_path: str
    executable_hash: str
    build_hash: str
    config_hash: str
    argv: tuple[str, ...]

    def __post_init__(self) -> None:
        """全 field の形式と argv/executable の一致を検証します。

        argv は tuple に正規化して immutable にします。
        """
        for name in ("campaign_manifest_hash", "launch_nonce", "executable_hash", "build_hash", "config_hash"):
            _digest(getattr(self, name), name)
        _int(self.slot_id, "slot_id", 0)
        for name in ("attempt_id", "reserved_run_id", "gameplay_attempt_id", "executable_path"):
            _text(getattr(self, name), name)
        if not os.path.isabs(self.executable_path):
            raise ValueError("executable_path must be absolute")
        if isinstance(self.argv, (str, bytes)) or not isinstance(self.argv, Sequence):
            raise ValueError("argv must be a sequence of strings")
        argv = tuple(self.argv)
        if not argv or not all(isinstance(item, str) for item in argv) or argv[0] != self.executable_path:
            raise ValueError("argv must be non-empty strings starting with executable_path")
        object.__setattr__(self, "argv", argv)

    @property
    def command_hash(self) -> str:
        """argv の canonical hash です。

        request と ledger の read-back 照合に使います。
        """
        return command_hash(self.argv)

    def to_payload(self) -> dict[str, Any]:
        """ledger payload(canonical JSON 化前の dict)を返します。

        command_hash も保存し、読み戻し時に argv と再照合します。
        """
        payload = {name: getattr(self, name) for name in _INTENT_KEYS - {"argv", "command_hash"}}
        payload.update(argv=list(self.argv), command_hash=self.command_hash)
        return payload

    @classmethod
    def from_payload(cls, value: Mapping[str, Any]) -> LaunchIntent:
        """ledger payload から intent を復元します。

        key の完全一致と command_hash の再計算一致を要求します。
        """
        _exact(value, _INTENT_KEYS, "launch intent")
        intent = cls(**{name: value[name] for name in _INTENT_KEYS - {"command_hash"}})
        if value["command_hash"] != intent.command_hash:
            raise ValueError("command_hash does not match argv")
        return intent


@dataclass(frozen=True)
class ProcessIdentity:
    """kernel から読んだ process identity(PID/create time/image/file identity)です。

    PID だけでは再利用で別 process を指しうるため、create time と image file の
    volume serial / file index まで一致したときだけ同じ process とみなします。
    """

    pid: int
    create_time: int
    image_path: str
    volume_serial: int
    file_index: int

    def __post_init__(self) -> None:
        """kernel 値の型と範囲を検証します。

        0 や負値の PID / create time は identity 未証明として拒否します。
        """
        _int(self.pid, "pid", 1)
        _int(self.create_time, "create_time", 1)
        _text(self.image_path, "image_path")
        _int(self.volume_serial, "volume_serial", 0)
        _int(self.file_index, "file_index", 0)

    @property
    def process_ref(self) -> str:
        """06-02 event の process_ref に使う一意文字列です。

        PID と create time の組なので PID 再利用後も衝突しません。
        """
        return f"pid:{self.pid}:ct:{self.create_time}"

    def to_payload(self) -> dict[str, Any]:
        """ledger payload 用の dict を返します。

        field 名をそのまま key にします。
        """
        return {name: getattr(self, name) for name in ("pid", "create_time", "image_path", "volume_serial", "file_index")}

    @classmethod
    def from_payload(cls, value: Mapping[str, Any]) -> ProcessIdentity:
        """ledger payload から identity を復元します。

        key の過不足は partial row として拒否します。
        """
        _exact(value, frozenset({"pid", "create_time", "image_path", "volume_serial", "file_index"}), "process")
        return cls(**value)


@dataclass(frozen=True)
class ProcessAttestation:
    """broker が suspended process と Job Object を attest した結果です。

    identity は kernel から読んだ値で、process 環境内 nonce の自己申告は含みません。
    """

    identity: ProcessIdentity
    job_ref: str
    broker_ref: str

    def __post_init__(self) -> None:
        """identity 型と参照文字列を検証します。

        job_ref / broker_ref は空文字を許しません。
        """
        if not isinstance(self.identity, ProcessIdentity):
            raise ValueError("identity must be a ProcessIdentity")
        _text(self.job_ref, "job_ref")
        _text(self.broker_ref, "broker_ref")

    def to_payload(self) -> dict[str, Any]:
        """ledger payload 用の dict を返します。

        identity は入れ子 object として保存します。
        """
        return {"process": self.identity.to_payload(), "job_ref": self.job_ref, "broker_ref": self.broker_ref}

    @classmethod
    def from_payload(cls, value: Mapping[str, Any]) -> ProcessAttestation:
        """ledger payload から attestation を復元します。

        入れ子 identity も完全一致検証します。
        """
        _exact(value, _PAYLOAD_KEYS[_S.PROCESS_ATTESTED], "attestation")
        return cls(ProcessIdentity.from_payload(value["process"]), value["job_ref"], value["broker_ref"])


class ProbeResult(StrEnum):
    """attested identity を kernel へ問い合わせた結果です。

    ALIVE 以外はすべて「同じ process が生きていると証明できない」扱いです。
    """

    ALIVE = "alive"
    EXITED = "exited"
    PID_REUSED = "pid_reused"
    UNKNOWN = "unknown"


class Classification(StrEnum):
    """crash 後 reconcile の3分類です。

    CONFIRMED だけが activation 可能で、他は launch gate failure / campaign block になります。
    """

    CONFIRMED = "confirmed"
    PROVEN_NO_PROCESS = "proven_no_process"
    UNCERTAIN = "uncertain"


# 06-02 CampaignEvent 契約との対応: confirmed は activation 前提、no-process は gate failure、
# uncertain は LAUNCH_UNCERTAIN(campaign block)。
CLASSIFICATION_EVENTS = {
    Classification.CONFIRMED: EventType.PROCESS_LAUNCH_CONFIRMED,
    Classification.PROVEN_NO_PROCESS: EventType.LAUNCH_GATE_FAILED,
    Classification.UNCERTAIN: EventType.LAUNCH_UNCERTAIN,
}
ProcessProbe = Callable[[ProcessIdentity], ProbeResult]


@dataclass(frozen=True)
class LaunchHistory:
    """1 attempt の検証済み ledger 履歴です。

    hash chain・順序・payload を全行検証したあとだけ作られます。
    """

    attempt_id: str
    intent: LaunchIntent
    stages: tuple[LaunchStage, ...]
    row_hashes: tuple[str, ...]
    attestation: ProcessAttestation | None = None
    activation_source: str | None = None
    failure_reason: str | None = None

    @property
    def last_stage(self) -> LaunchStage:
        """最後に commit された段階です。

        次に許される mutation はこの値で決まります。
        """
        return self.stages[-1]


@dataclass(frozen=True)
class Reconciliation:
    """reconcile の結果です。

    classification と理由、activation 済みかどうかを返します。
    """

    attempt_id: str
    classification: Classification
    reason: str
    activated: bool = False

    @property
    def event_type(self) -> EventType:
        """06-02 CampaignEvent 契約上の対応 event 種別です。

        campaign 側はこの種別で slot lifecycle を記録します。
        """
        return CLASSIFICATION_EVENTS[self.classification]


@dataclass(frozen=True)
class StorageVerdict:
    """保存先の machine-check 結果です。

    eligible が False なら formal-ineligible で、ledger を開きません。
    """

    eligible: bool
    reasons: tuple[str, ...]
    facts: Mapping[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        """integration result に保存する verdict 名です。

        formal_eligible か formal_ineligible のどちらかです。
        """
        return "formal_eligible" if self.eligible else "formal_ineligible"


@functools.cache
def win32_function(dll: str, name: str, restype: Any, *argtypes: Any) -> Any:
    """Win32 関数を signature 付きで1回だけ結び付けます。

    64-bit handle を切り詰めないよう restype / argtypes を必ず設定します。
    Windows 以外では呼ばれない前提です。
    """
    function = getattr(ctypes.WinDLL(dll, use_last_error=True), name)
    function.restype = restype
    function.argtypes = argtypes
    return function


def win32_check(result: Any, what: str) -> Any:
    """Win32 の失敗戻り値を OSError に変換します。

    0 / None を失敗とみなし、GetLastError を例外へ載せます。
    """
    if not result:
        error = ctypes.get_last_error()
        raise OSError(error, f"{what} failed (winerror {error})")
    return result


def _close_handle(handle: Any) -> None:
    """handle を閉じます。

    None は無視します。
    """
    if handle:
        win32_function("kernel32", "CloseHandle", _BOOL, _H)(handle)


def _local_free(pointer: Any) -> None:
    """LocalAlloc 系で返された buffer を解放します。

    SDDL 変換 API の戻り値の後始末に使います。
    """
    if pointer:
        win32_function("kernel32", "LocalFree", _H, _H)(pointer)


def process_user_sid(pid: int | None = None) -> str:
    """process token の user SID を文字列で返します。

    pid を省略すると自 process の SID です。named pipe の相手確認にも使います。
    """
    kernel = "kernel32"
    if pid is None:
        process = win32_function(kernel, "GetCurrentProcess", _H)()
    else:
        process = win32_check(
            win32_function(kernel, "OpenProcess", _H, _DW, _BOOL, _DW)(_PROCESS_QUERY_LIMITED_INFORMATION, 0, pid),
            "OpenProcess",
        )
    token = _H()
    try:
        win32_check(win32_function("advapi32", "OpenProcessToken", _BOOL, _H, _DW, ctypes.POINTER(_H))(
            process, _TOKEN_QUERY, ctypes.byref(token)), "OpenProcessToken")
        info = win32_function("advapi32", "GetTokenInformation", _BOOL, _H, ctypes.c_int, _H, _DW, _PDW)
        size = _DW()
        info(token, _TOKEN_USER, None, 0, ctypes.byref(size))
        buffer = ctypes.create_string_buffer(size.value)
        win32_check(info(token, _TOKEN_USER, buffer, size, ctypes.byref(size)), "GetTokenInformation")
        text = _H()
        win32_check(win32_function("advapi32", "ConvertSidToStringSidW", _BOOL, _H, ctypes.POINTER(_H))(
            ctypes.c_void_p.from_buffer(buffer).value, ctypes.byref(text)), "ConvertSidToStringSidW")
        try:
            return ctypes.wstring_at(text.value)
        finally:
            _local_free(text)
    finally:
        _close_handle(token)
        if pid is not None:
            _close_handle(process)


def current_user_sid() -> str:
    """現在の process user の SID を返します。

    ledger ACL と named pipe DACL の唯一の許可先になります。
    """
    return process_user_sid(None)


def security_descriptor_from_sddl(sddl: str) -> Any:
    """SDDL から self-relative security descriptor を作ります。

    戻り値は LocalFree で解放する必要があります。
    """
    pointer = _H()
    win32_check(win32_function("advapi32", "ConvertStringSecurityDescriptorToSecurityDescriptorW",
                               _BOOL, _LPW, _DW, ctypes.POINTER(_H), _PDW)(sddl, 1, ctypes.byref(pointer), None),
                "ConvertStringSecurityDescriptorToSecurityDescriptorW")
    return pointer


def apply_private_acl(path: str, sid: str) -> None:
    """directory の protected DACL を current user 専用にします。

    継承 ACE を切り、配下に作られる DB/WAL/SHM も current user だけが触れるようにします。
    owner は作成者(current user)のままで、WRITE_OWNER 権限を要求しません。
    """
    descriptor = security_descriptor_from_sddl(f"D:P(A;OICI;FA;;;{sid})")
    try:
        win32_check(win32_function("advapi32", "SetFileSecurityW", _BOOL, _LPW, _DW, _H)(
            path, _DACL_SECURITY_INFORMATION | _PROTECTED_DACL_SECURITY_INFORMATION,
            descriptor), "SetFileSecurityW")
    finally:
        _local_free(descriptor)


def read_sddl(path: str) -> str:
    """file / directory の owner と DACL を SDDL 文字列で読み戻します。

    ACL を設定値ではなく実際の OS 状態で検査するための read-back です。
    """
    flags = _OWNER_SECURITY_INFORMATION | _DACL_SECURITY_INFORMATION
    get = win32_function("advapi32", "GetFileSecurityW", _BOOL, _LPW, _DW, _H, _DW, _PDW)
    needed = _DW()
    get(path, flags, None, 0, ctypes.byref(needed))
    buffer = ctypes.create_string_buffer(max(needed.value, 1))
    win32_check(get(path, flags, buffer, needed, ctypes.byref(needed)), "GetFileSecurityW")
    text = _H()
    win32_check(win32_function("advapi32", "ConvertSecurityDescriptorToStringSecurityDescriptorW",
                               _BOOL, _H, _DW, _DW, ctypes.POINTER(_H), _PDW)(
        buffer, 1, flags, ctypes.byref(text), None), "ConvertSecurityDescriptorToStringSecurityDescriptorW")
    try:
        return ctypes.wstring_at(text.value)
    finally:
        _local_free(text)


def acl_findings(sddl: str, sid: str, *, require_owner: bool, require_protected: bool) -> list[str]:
    """SDDL が current-user 専用 ACL かを検査し、違反理由を返します。

    全 ACE が current user への allow だけであること、必要なら owner と継承遮断も要求します。
    空リストなら合格です。NULL DACL や空 DACL も違反として扱います。
    """
    findings: list[str] = []
    owner = re.search(r"O:(.+?)(?=[GDS]:|$)", sddl)
    if require_owner and (owner is None or owner.group(1) != sid):
        findings.append("owner_not_current_user")
    dacl = re.search(r"D:([A-Z]*)((?:\([^)]*\))*)", sddl)
    aces = re.findall(r"\(([^)]*)\)", dacl.group(2)) if dacl else []
    if not aces:
        findings.append("dacl_missing_or_empty")
    if require_protected and (dacl is None or "P" not in dacl.group(1)):
        findings.append("dacl_not_protected")
    for ace in aces:
        parts = ace.split(";")
        if len(parts) < 6 or parts[0] != "A" or parts[5] != sid:
            findings.append(f"foreign_ace:{ace}")
    return findings


def _is_unc(path: str) -> bool:
    """UNC / device namespace 形式の path かを判定します。

    `\\\\server\\share`・`\\\\?\\`・`\\\\.\\` はすべて support 外として扱います。
    """
    return path.replace("/", "\\").startswith("\\\\")


def _volume_facts(path: str) -> dict[str, Any]:
    """path が属する volume の root・drive type・filesystem を OS から読みます。

    mount point も GetVolumePathNameW で実 volume に解決します。
    """
    root = ctypes.create_unicode_buffer(1024)
    win32_check(win32_function("kernel32", "GetVolumePathNameW", _BOOL, _LPW, _LPW, _DW)(path, root, 1024),
                "GetVolumePathNameW")
    drive = win32_function("kernel32", "GetDriveTypeW", ctypes.c_uint, _LPW)(root.value)
    filesystem = ctypes.create_unicode_buffer(64)
    serial = _DW()
    win32_check(win32_function("kernel32", "GetVolumeInformationW", _BOOL, _LPW, _LPW, _DW, _PDW, _PDW, _PDW,
                               _LPW, _DW)(root.value, None, 0, ctypes.byref(serial), None, None, filesystem, 64),
                "GetVolumeInformationW")
    return {
        "volume_root": root.value,
        "drive_type": _DRIVE_NAMES.get(drive, str(drive)),
        "filesystem": filesystem.value,
        "volume_serial": serial.value,
    }


def check_storage(directory: str | os.PathLike[str]) -> StorageVerdict:
    """ledger 保存先が local fixed NTFS かを machine-check します。

    UNC・network・removable・non-NTFS・Windows 以外・OS 照会失敗はすべて formal-ineligible です。
    symlink/junction は実体へ解決してから判定します。
    """
    raw = os.fspath(directory)
    facts: dict[str, Any] = {"requested_path": raw, "os_name": os.name}
    if os.name != "nt":
        return StorageVerdict(False, ("platform_not_windows",), facts)
    if _is_unc(raw):
        return StorageVerdict(False, ("unc_path",), facts)
    resolved = os.path.realpath(raw)
    facts["resolved_path"] = resolved
    if _is_unc(resolved):
        return StorageVerdict(False, ("unc_path",), facts)
    try:
        facts.update(_volume_facts(resolved))
    except OSError as exc:
        facts["error"] = str(exc)
        return StorageVerdict(False, ("volume_query_failed",), facts)
    reasons = []
    if facts["drive_type"] != "fixed":
        reasons.append(f"drive_{facts['drive_type']}")
    if facts["filesystem"] != "NTFS":
        reasons.append("filesystem_not_ntfs")
    return StorageVerdict(not reasons, tuple(reasons), facts)


def _read_pragma(connection: sqlite3.Connection, statement: str) -> Any:
    """PRAGMA を実行して先頭値を読み戻します。

    設定値ではなく SQLite が実際に採用した値を返します。
    """
    return connection.execute(statement).fetchone()[0]


def _row_hash(attempt_id: str, stage: str, body: str, prev_hash: str) -> str:
    """row の hash chain 値を計算します。

    直前 row の hash を含めるので、途中 row の欠落・差し替えを検出できます。
    """
    return canonical_hash({"attempt_id": attempt_id, "stage": stage, "payload": body, "prev_hash": prev_hash})


def _replay(attempt_id: str, rows: Sequence[tuple[Any, ...]]) -> LaunchHistory:
    """row 列を先頭から再検証して履歴を組み立てます。

    canonical payload・hash chain・段階遷移・process_ref 束縛のどれかが崩れたら ValueError です。
    """
    prev = _GENESIS
    last: LaunchStage | None = None
    intent = attestation = None
    source = reason = None
    stages: list[LaunchStage] = []
    hashes: list[str] = []
    for _seq, stage_text, body, prev_hash, row_hash in rows:
        stage = LaunchStage(stage_text)
        payload = json.loads(body)
        if not isinstance(payload, dict) or canonical_json_bytes(payload).decode("utf-8") != body:
            raise ValueError("payload is not canonical JSON")
        if prev_hash != prev or row_hash != _row_hash(attempt_id, stage.value, body, prev_hash):
            raise ValueError("hash chain is broken")
        if stage not in _NEXT.get(last, frozenset()):
            raise ValueError(f"illegal transition {last} -> {stage.value}")
        _exact(payload, _PAYLOAD_KEYS[stage], stage.value)
        if stage is _S.LAUNCH_INTENT:
            intent = LaunchIntent.from_payload(payload)
            if intent.attempt_id != attempt_id:
                raise ValueError("intent attempt_id does not match row")
        elif stage is _S.PROCESS_ATTESTED:
            attestation = ProcessAttestation.from_payload(payload)
        elif stage in (_S.RESUME_INTENT, _S.PROCESS_LAUNCH_CONFIRMED):
            if attestation is None or payload["process_ref"] != attestation.identity.process_ref:
                raise ValueError(f"{stage.value} process_ref is not the attested process")
        elif stage is _S.FORMAL_RUN_ACTIVATED:
            if payload["activation_source"] not in {"normal", "reconciliation"}:
                raise ValueError("activation_source must be normal or reconciliation")
            source = payload["activation_source"]
        else:
            reason = _text(payload["failure_reason"], "failure_reason")
        prev, last = row_hash, stage
        stages.append(stage)
        hashes.append(row_hash)
    assert intent is not None
    return LaunchHistory(attempt_id, intent, tuple(stages), tuple(hashes), attestation, source, reason)


class DurableLaunchStore:
    """SQLite WAL/FULL の append-only launch ledger です。

    開くときに保存先・ACL・PRAGMA・integrity・schema version を全部読み戻して検査し、
    1つでも合わなければ例外で止めます。controller と broker が別 process から同じ ledger を
    開いても BEGIN IMMEDIATE で直列化され、順序外の書込みは拒否されます。
    """

    def __init__(self, directory: str | os.PathLike[str]) -> None:
        """保存先を検査して ledger を開きます(無ければ作成)。

        新規 directory には current-user 専用 ACL を付け、既存 directory は ACL を読み戻して検査します。
        """
        self.directory = Path(directory)
        self.verdict = check_storage(self.directory)
        if not self.verdict.eligible:
            raise UnsupportedStorageError(self.verdict)
        sid = current_user_sid()
        if not self.directory.exists():
            self.directory.mkdir(parents=True)
            apply_private_acl(str(self.directory), sid)
        findings = acl_findings(read_sddl(str(self.directory)), sid, require_owner=True, require_protected=True)
        if findings:
            raise UnsupportedStorageError(StorageVerdict(False, tuple(findings), self.verdict.facts))
        self.path = self.directory / LEDGER_FILENAME
        connection = sqlite3.connect(self.path, isolation_level=None, timeout=10.0, check_same_thread=False)
        try:
            self.pragmas = self._configure(connection)
            self._ensure_schema(connection)
            for item in (self.path, Path(f"{self.path}-wal"), Path(f"{self.path}-shm")):
                if item.exists():
                    bad = acl_findings(read_sddl(str(item)), sid, require_owner=False, require_protected=False)
                    if bad:
                        raise UnsupportedStorageError(StorageVerdict(False, tuple(bad), self.verdict.facts))
        except sqlite3.DatabaseError as exc:
            connection.close()
            raise LedgerIntegrityError(f"ledger database is unreadable: {exc}") from exc
        except BaseException:
            connection.close()
            raise
        self._conn = connection

    @staticmethod
    def _configure(connection: sqlite3.Connection) -> dict[str, Any]:
        """WAL/FULL を設定し、read-back と integrity_check を検査します。

        SQLite が WAL / FULL を採用しなかった場合や integrity が ok でない場合は拒否します。
        """
        connection.execute("PRAGMA busy_timeout=10000")
        mode = str(_read_pragma(connection, "PRAGMA journal_mode=WAL")).lower()
        connection.execute("PRAGMA synchronous=FULL")
        synchronous = _read_pragma(connection, "PRAGMA synchronous")
        if mode != "wal" or synchronous != 2:
            raise LedgerIntegrityError(f"PRAGMA read-back mismatch: journal_mode={mode} synchronous={synchronous}")
        integrity = connection.execute("PRAGMA integrity_check").fetchall()
        if integrity != [("ok",)]:
            raise LedgerIntegrityError(f"integrity_check failed: {integrity[:3]}")
        return {"journal_mode": mode, "synchronous": "FULL", "integrity_check": "ok",
                "sqlite_version": sqlite3.sqlite_version}

    @staticmethod
    def _ensure_schema(connection: sqlite3.Connection) -> None:
        """空 DB なら schema を作り、既存 DB は schema id と version を照合します。

        未知 version・別 schema の DB は開きません。
        """
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if _read_pragma(connection, "PRAGMA user_version") == 0 and not tables - {"sqlite_sequence"}:
            connection.executescript("BEGIN IMMEDIATE;" + _SCHEMA_SQL + "COMMIT;")
        version = _read_pragma(connection, "PRAGMA user_version")
        meta = connection.execute("SELECT value FROM ledger_meta WHERE key='schema'").fetchone()
        if version != LEDGER_SCHEMA_VERSION or meta != (LEDGER_SCHEMA_ID,):
            raise LedgerIntegrityError(f"unsupported ledger schema: version={version} meta={meta}")

    def close(self) -> None:
        """ledger 接続を閉じます。

        最後の接続なら SQLite が WAL を checkpoint します。
        """
        self._conn.close()

    def __enter__(self) -> DurableLaunchStore:
        """with 文で使えるよう自身を返します。

        抜けるときに close します。
        """
        return self

    def __exit__(self, *_exc: object) -> None:
        """with 文の終了時に接続を閉じます。

        例外は握りつぶしません。
        """
        self.close()

    def _commit(self) -> None:
        """transaction を commit します。

        fault injection で commit 失敗を再現するための差し替え点でもあります。
        """
        self._conn.execute("COMMIT")

    def _rows(self, attempt_id: str) -> list[tuple[Any, ...]]:
        """attempt の row を seq 順に読みます。

        検証は呼び出し側の _replay が行います。
        """
        return self._conn.execute(
            "SELECT seq, stage, payload, prev_hash, row_hash FROM launch_rows WHERE attempt_id=? ORDER BY seq",
            (attempt_id,),
        ).fetchall()

    def _load(self, attempt_id: str) -> LaunchHistory | None:
        """attempt の履歴を検証付きで読みます。

        row が無ければ None、破損していれば LedgerIntegrityError です。
        """
        rows = self._rows(attempt_id)
        if not rows:
            return None
        try:
            return _replay(attempt_id, rows)
        except (ValueError, TypeError, KeyError, AssertionError) as exc:
            raise LedgerIntegrityError(f"ledger rows for {attempt_id} are invalid: {exc}") from exc

    def history(self, attempt_id: str) -> LaunchHistory:
        """attempt の検証済み履歴を返します。

        未知 attempt は LedgerOrderError です。
        """
        loaded = self._load(attempt_id)
        if loaded is None:
            raise LedgerOrderError(f"unknown launch attempt: {attempt_id}")
        return loaded

    def attempt_ids(self) -> tuple[str, ...]:
        """ledger に intent がある attempt id を commit 順で返します。

        reconcile_all の走査対象です。
        """
        rows = self._conn.execute("SELECT attempt_id FROM launch_rows WHERE stage=? ORDER BY seq",
                                  (_S.LAUNCH_INTENT.value,)).fetchall()
        return tuple(row[0] for row in rows)

    def _append(
        self,
        attempt_id: str,
        stage: LaunchStage,
        payload: Mapping[str, Any],
        *,
        claims: Sequence[tuple[str, str]] = (),
        expect: LaunchStage | None | str = "any",
    ) -> None:
        """1段階を1 transaction で append します。

        BEGIN IMMEDIATE で他 process と直列化し、既存履歴の検証・遷移検査・identity claim・
        row insert・commit のどれかが失敗したら全体を rollback します。
        """
        try:
            body = canonical_json_bytes(dict(payload)).decode("utf-8")
        except (TypeError, ValueError) as exc:
            raise LedgerIdentityError(f"payload is not canonical JSON: {exc}") from exc
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                history = self._load(attempt_id)
                last = history.last_stage if history else None
                if expect != "any" and last != expect:
                    raise LedgerOrderError(f"{attempt_id}: expected {expect} but ledger is at {last}")
                if stage not in _NEXT.get(last, frozenset()):
                    raise LedgerOrderError(f"{attempt_id}: {last} -> {stage.value} is duplicate or out of order")
                rows = self._rows(attempt_id)
                prev = rows[-1][4] if rows else _GENESIS
                row = (None, stage.value, body, prev, _row_hash(attempt_id, stage.value, body, prev))
                try:
                    _replay(attempt_id, [*rows, row])
                except (ValueError, TypeError, KeyError, AssertionError) as exc:
                    raise LedgerIdentityError(f"{attempt_id}: {stage.value} payload rejected: {exc}") from exc
                for kind, value in claims:
                    try:
                        self._conn.execute("INSERT INTO identity_claims(kind, value, attempt_id) VALUES(?, ?, ?)",
                                           (kind, value, attempt_id))
                    except sqlite3.IntegrityError as exc:
                        raise LedgerIdentityError(f"duplicate {kind}: {value}") from exc
                self._conn.execute(
                    "INSERT INTO launch_rows(attempt_id, stage, payload, prev_hash, row_hash) VALUES(?, ?, ?, ?, ?)",
                    (attempt_id, *row[1:]),
                )
                self._commit()
            except BaseException:
                if self._conn.in_transaction:
                    self._conn.execute("ROLLBACK")
                raise
        except sqlite3.IntegrityError as exc:
            raise LedgerOrderError(f"{attempt_id}: {stage.value} violates ledger uniqueness: {exc}") from exc
        except sqlite3.DatabaseError as exc:
            raise LedgerCommitError(f"{attempt_id}: {stage.value} commit failed: {exc}") from exc

    def commit_intent(self, intent: LaunchIntent) -> None:
        """LAUNCH_INTENT を commit します。

        slot・reserved run id・gameplay attempt id・nonce を ledger 全体で一意に claim するので、
        同じ slot への replacement launch はここで拒否されます。
        """
        if not isinstance(intent, LaunchIntent):
            raise LedgerIdentityError("intent must be a LaunchIntent")
        claims = [
            ("slot", f"{intent.campaign_manifest_hash}:{intent.slot_id}"),
            ("reserved_run_id", intent.reserved_run_id),
            ("gameplay_attempt_id", intent.gameplay_attempt_id),
            ("launch_nonce", intent.launch_nonce),
        ]
        self._append(intent.attempt_id, _S.LAUNCH_INTENT, intent.to_payload(), claims=claims)

    def commit_attestation(self, attempt_id: str, attestation: ProcessAttestation) -> None:
        """PROCESS_ATTESTED を commit します。

        process_ref(PID+create time)と job_ref を一意 claim し、PID reuse による差し替えを拒否します。
        """
        if not isinstance(attestation, ProcessAttestation):
            raise LedgerIdentityError("attestation must be a ProcessAttestation")
        claims = [("process_ref", attestation.identity.process_ref), ("job_ref", attestation.job_ref)]
        self._append(attempt_id, _S.PROCESS_ATTESTED, attestation.to_payload(), claims=claims)

    def commit_resume_intent(self, attempt_id: str, process_ref: str) -> None:
        """RESUME_INTENT を commit します。

        broker はこの commit 成功後にだけ ResumeThread を呼んでよいです。
        """
        self._append(attempt_id, _S.RESUME_INTENT, {"process_ref": process_ref})

    def commit_confirmation(self, attempt_id: str, process_ref: str) -> None:
        """PROCESS_LAUNCH_CONFIRMED を commit します。

        resume 後に同じ kernel identity の生存を確認してから呼びます。
        """
        self._append(attempt_id, _S.PROCESS_LAUNCH_CONFIRMED, {"process_ref": process_ref})

    def commit_gate_failure(self, attempt_id: str, reason: str) -> None:
        """LAUNCH_GATE_FAILED(durable no-process)を commit します。

        resume intent 前にしか書けないので、process が一度も実行されていないことを意味します。
        """
        self._append(attempt_id, _S.LAUNCH_GATE_FAILED, {"failure_reason": reason})

    def commit_uncertain(self, attempt_id: str, reason: str) -> None:
        """LAUNCH_UNCERTAIN(campaign block)を commit します。

        resume intent 以降の失敗で、process が走ったかどうかを証明できない状態です。
        """
        self._append(attempt_id, _S.LAUNCH_UNCERTAIN, {"failure_reason": reason})

    def activate(self, attempt_id: str, source: str, probe: ProcessProbe) -> bool:
        """confirmed launch を idempotent に activation します。

        初回は attested identity の生存を probe で確かめて FORMAL_RUN_ACTIVATED を append し True、
        既に activation 済みなら row を増やさず False を返します。confirmed 以外は拒否します。
        """
        history = self.history(attempt_id)
        if history.last_stage is _S.FORMAL_RUN_ACTIVATED:
            return False
        if history.last_stage is not _S.PROCESS_LAUNCH_CONFIRMED or history.attestation is None:
            raise LedgerOrderError(f"{attempt_id}: activation requires a confirmed launch")
        result = ProbeResult(probe(history.attestation.identity))
        if result is not ProbeResult.ALIVE:
            raise LedgerIdentityError(f"{attempt_id}: attested process is not alive ({result.value})")
        try:
            self._append(attempt_id, _S.FORMAL_RUN_ACTIVATED, {"activation_source": source},
                         expect=_S.PROCESS_LAUNCH_CONFIRMED)
        except LedgerOrderError:
            if self.history(attempt_id).last_stage is _S.FORMAL_RUN_ACTIVATED:
                return False
            raise
        return True

    def reconcile(self, attempt_id: str, probe: ProcessProbe) -> Reconciliation:
        """crash 後の attempt を3分類し、必要なら terminal row を append します。

        resume intent 前 → proven no-process(gate failure)、resume intent 後で未 confirm →
        uncertain、confirm 済みは attested identity が ALIVE のときだけ confirmed です。
        broker が同時に進めた場合は読み直して再判定します。
        """
        for _ in range(3):
            history = self.history(attempt_id)
            last = history.last_stage
            if last is _S.LAUNCH_GATE_FAILED:
                return Reconciliation(attempt_id, Classification.PROVEN_NO_PROCESS, history.failure_reason or "")
            if last is _S.LAUNCH_UNCERTAIN:
                return Reconciliation(attempt_id, Classification.UNCERTAIN, history.failure_reason or "")
            if last is _S.FORMAL_RUN_ACTIVATED:
                return Reconciliation(attempt_id, Classification.CONFIRMED, "already_activated", activated=True)
            try:
                if last in (_S.LAUNCH_INTENT, _S.PROCESS_ATTESTED):
                    reason = f"reconciled_before_resume_intent_at_{last.value.lower()}"
                    self._append(attempt_id, _S.LAUNCH_GATE_FAILED, {"failure_reason": reason}, expect=last)
                    return Reconciliation(attempt_id, Classification.PROVEN_NO_PROCESS, reason)
                if last is _S.RESUME_INTENT:
                    reason = "resume_boundary_ambiguous"
                    self._append(attempt_id, _S.LAUNCH_UNCERTAIN, {"failure_reason": reason}, expect=last)
                    return Reconciliation(attempt_id, Classification.UNCERTAIN, reason)
                assert history.attestation is not None
                result = ProbeResult(probe(history.attestation.identity))
                if result is ProbeResult.ALIVE:
                    return Reconciliation(attempt_id, Classification.CONFIRMED, "confirmed_process_alive")
                reason = f"confirmed_process_{result.value}"
                self._append(attempt_id, _S.LAUNCH_UNCERTAIN, {"failure_reason": reason}, expect=last)
                return Reconciliation(attempt_id, Classification.UNCERTAIN, reason)
            except LedgerOrderError:
                continue
        raise LedgerOrderError(f"{attempt_id}: reconciliation did not converge")

    def reconcile_all(self, probe: ProcessProbe) -> tuple[Reconciliation, ...]:
        """ledger 上の全 attempt を commit 順に reconcile します。

        1件でも破損していれば LedgerIntegrityError で止まります。
        """
        return tuple(self.reconcile(attempt_id, probe) for attempt_id in self.attempt_ids())

    def campaign_events(self, attempt_id: str) -> tuple[CampaignEvent, ...]:
        """ledger 履歴を 06-02 CampaignEvent 列へ写します。

        RESUME_INTENT は ledger 内部の境界なので event にはしません。
        campaign_manifest_hash は intent に束縛された値を必ず付けます。
        """
        history = self.history(attempt_id)
        intent = history.intent
        common = {"slot_id": intent.slot_id, "campaign_manifest_hash": intent.campaign_manifest_hash}
        events: list[CampaignEvent] = []
        for stage in history.stages:
            if stage is _S.LAUNCH_INTENT:
                events.append(CampaignEvent(
                    EventType.LAUNCH_INTENT_COMMITTED, attempt_id=intent.attempt_id,
                    reserved_run_id=intent.reserved_run_id, gameplay_attempt_id=intent.gameplay_attempt_id,
                    launch_nonce=intent.launch_nonce, **common))
            elif stage is _S.PROCESS_ATTESTED:
                assert history.attestation is not None
                events.append(CampaignEvent(
                    EventType.BROKER_PROCESS_ATTESTED, process_ref=history.attestation.identity.process_ref,
                    job_ref=history.attestation.job_ref, **common))
            elif stage is _S.PROCESS_LAUNCH_CONFIRMED:
                events.append(CampaignEvent(EventType.PROCESS_LAUNCH_CONFIRMED, **common))
            elif stage is _S.FORMAL_RUN_ACTIVATED:
                events.append(CampaignEvent(EventType.FORMAL_RUN_ACTIVATED,
                                            activation_source=history.activation_source, **common))
            elif stage in (_S.LAUNCH_GATE_FAILED, _S.LAUNCH_UNCERTAIN):
                events.append(CampaignEvent(EventType(stage.value), failure_reason=history.failure_reason, **common))
        return tuple(events)
