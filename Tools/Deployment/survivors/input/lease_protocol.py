"""閉じた semantic lease wire protocol と再送防止 validator。
任意キーではなく9 actionだけを運び、nonce・順序・期限・対象 hash を注入前に一括検証します。
"""
from __future__ import annotations
from dataclasses import dataclass
import json
import re
from typing import Any, Mapping
from survivors.action_semantics import ActionContract
SCHEMA_VERSION = "survivors_input_lease.v1"
_NONCE, _HASH = re.compile(r"[0-9a-f]{32}"), re.compile(r"[0-9a-f]{64}")
_FIELDS = frozenset({"kind", "schema_version", "session_nonce", "sequence",
                     "issued_monotonic_ns", "expires_monotonic_ns", "target_hash",
                     "action_hash", "action_index", "target_pid", "target_hwnd"})
def _strict_positive_int(value: object) -> bool:
    """bool を除いた正の整数だけを認める。
    JSON の true が 1 として binding 検証を抜ける曖昧さを防ぎます。
    """
    return type(value) is int and value > 0
@dataclass(frozen=True)
class Lease:
    """1回の9方向 action を最大150msだけ許可する immutable command。
    PID/HWND と2つの contract hash を含み、helper が注入直前に再検証できます。
    """
    session_nonce: str
    sequence: int
    issued_monotonic_ns: int
    expires_monotonic_ns: int
    target_hash: str
    action_hash: str
    action_index: int
    target_pid: int
    target_hwnd: int
    def __post_init__(self) -> None:
        """型・値域・期限上限を fail-closed に検査する。
        wire と直接生成のどちらにも同じ不変条件を適用します。
        """
        ints = (self.sequence, self.issued_monotonic_ns, self.expires_monotonic_ns,
                self.target_pid, self.target_hwnd)
        if not all(_strict_positive_int(value) for value in ints):
            raise ValueError("lease integer fields must be positive strict integers")
        if type(self.action_index) is not int or not 0 <= self.action_index <= 8:
            raise ValueError("action_index must be a semantic action from 0 through 8")
        if not isinstance(self.session_nonce, str) or not _NONCE.fullmatch(self.session_nonce):
            raise ValueError("invalid session nonce")
        if not isinstance(self.target_hash, str) or not _HASH.fullmatch(self.target_hash):
            raise ValueError("invalid target hash")
        if not isinstance(self.action_hash, str) or not _HASH.fullmatch(self.action_hash):
            raise ValueError("invalid action hash")
        duration = self.expires_monotonic_ns - self.issued_monotonic_ns
        if not 0 < duration <= 150_000_000:
            raise ValueError("lease duration must be positive and at most 150ms")
    def to_wire(self) -> dict[str, object]:
        """closed schema の JSON-compatible mapping を返す。
        protocol field を明示列挙し、内部状態や任意入力指定を wire に漏らしません。
        """
        return {
            "kind": "lease", "schema_version": SCHEMA_VERSION,
            "session_nonce": self.session_nonce, "sequence": self.sequence,
            "issued_monotonic_ns": self.issued_monotonic_ns,
            "expires_monotonic_ns": self.expires_monotonic_ns,
            "target_hash": self.target_hash, "action_hash": self.action_hash,
            "action_index": self.action_index, "target_pid": self.target_pid,
            "target_hwnd": self.target_hwnd,
        }
    @classmethod
    def from_wire(cls, data: Mapping[str, Any]) -> "Lease":
        """未知・欠落 field を拒否して Lease を復元する。
        全 wire reader をこの1か所へ集約し、別経路の緩い解釈を作りません。
        """
        if not isinstance(data, Mapping) or set(data) != _FIELDS:
            raise ValueError("unknown or missing lease fields")
        if data.get("kind") != "lease" or data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("invalid lease envelope")
        return cls(**{key: data[key] for key in _FIELDS - {"kind", "schema_version"}})
class LeaseValidator:
    """session binding と strictly increasing sequence を保持する validator。
    すべての binding が通った後だけ sequence を進め、拒否 payload が状態を汚染しないようにします。
    """
    def __init__(self, session_nonce: str, target_hash: str, action_hash: str) -> None:
        """期待する session・target・action identity を固定する。
        起動時に controller と共有した値以外を後から受理しません。
        """
        if not _NONCE.fullmatch(session_nonce) or not _HASH.fullmatch(target_hash) or not _HASH.fullmatch(action_hash):
            raise ValueError("invalid validator binding")
        self._nonce = session_nonce
        self._target_hash = target_hash
        self._action_hash = action_hash
        self._last_sequence = 0
    def accept(self, lease: Lease, now_ns: int) -> None:
        """nonce・sequence・時刻・両 hash を対称に検査して受理する。
        stale/replay/expired/wrong-target/wrong-action のどれか1つでも一致しなければ状態変更前に拒否します。
        """
        if type(now_ns) is not int or now_ns <= 0:
            raise ValueError("invalid validation time")
        failures = (
            lease.session_nonce != self._nonce,
            lease.sequence <= self._last_sequence,
            lease.issued_monotonic_ns > now_ns,
            lease.expires_monotonic_ns <= now_ns,
            lease.target_hash != self._target_hash,
            lease.action_hash != self._action_hash,
        )
        if any(failures):
            raise ValueError("lease binding, order, or lifetime rejected")
        self._last_sequence = lease.sequence
def action_golden_bytes(contract: ActionContract) -> bytes:
    """ActionContract の sim vector と WASD chord を正準 JSON bytes にする。
    golden fixture と byte 単位で比較でき、9 action の並び替えや chord drift を検出できます。
    """
    rows = [
        {"index": row.index, "sim_vector": list(row.sim_vector), "wasd_chord": list(row.wasd_chord)}
        for row in contract.rows
    ]
    return json.dumps(rows, ensure_ascii=True, separators=(",", ":")).encode("ascii")
