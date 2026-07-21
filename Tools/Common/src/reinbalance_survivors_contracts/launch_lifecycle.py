"""Formal run identity and durable launch transition contract."""
from __future__ import annotations

import enum
from dataclasses import dataclass, replace
from typing import Any, Mapping

from .canonical_json import canonical_hash

SCHEMA_VERSION = "launch_lifecycle.v1"


class LaunchState(str, enum.Enum):
    PREFLIGHT = "PREFLIGHT"
    PREFLIGHT_FAILED = "PREFLIGHT_FAILED"
    LAUNCH_INTENT = "LAUNCH_INTENT"
    FORMAL_RUN_ACTIVATED = "FORMAL_RUN_ACTIVATED"
    LAUNCH_GATE_FAILED = "LAUNCH_GATE_FAILED"
    LAUNCH_UNCERTAIN = "LAUNCH_UNCERTAIN"
    TERMINAL = "TERMINAL"


@dataclass(frozen=True)
class PlatformGate:
    local_fixed_volume: bool
    ntfs: bool
    wal: bool
    synchronous_full: bool
    integrity_ok: bool
    flush_ok: bool
    broker_attested: bool

    def validate(self) -> None:
        failed = [name for name, value in self.__dict__.items() if value is not True]
        if failed:
            raise ValueError("formal launch platform gate failed: " + ", ".join(failed))


@dataclass(frozen=True)
class LaunchLifecycle:
    attempt_id: str
    state: LaunchState = LaunchState.PREFLIGHT
    reserved_run_id: str | None = None
    gameplay_attempt_id: str | None = None
    launch_nonce: str | None = None
    process_ref: str | None = None
    activation_source: str | None = None
    failure_reason: str | None = None
    schema_version: str = SCHEMA_VERSION

    _KEYS = frozenset({"schema_version", "attempt_id", "state", "reserved_run_id", "gameplay_attempt_id", "launch_nonce", "process_ref", "activation_source", "failure_reason"})

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION or not self.attempt_id:
            raise ValueError("invalid launch lifecycle identity/version")
        reserved = (self.reserved_run_id, self.gameplay_attempt_id, self.launch_nonce)
        if any(v is not None for v in reserved) and not all(isinstance(v, str) and v for v in reserved):
            raise ValueError("reserved identity fields must be all present or all absent")
        if self.process_ref is not None and self.state not in {LaunchState.FORMAL_RUN_ACTIVATED, LaunchState.TERMINAL}:
            raise ValueError("process ref requires activation")

    @classmethod
    def begin(cls, attempt_id: str) -> "LaunchLifecycle": return cls(attempt_id)

    def preflight_failure(self, reason: str) -> "LaunchLifecycle":
        if self.state is not LaunchState.PREFLIGHT: raise ValueError("preflight already terminal")
        return replace(self, state=LaunchState.PREFLIGHT_FAILED, failure_reason=reason)

    def reserve(self, run_id: str, gameplay_attempt_id: str, nonce: str, *, durable: bool) -> "LaunchLifecycle":
        if self.state is not LaunchState.PREFLIGHT or not durable:
            raise ValueError("durable LAUNCH_INTENT must be the first reservation")
        return replace(self, state=LaunchState.LAUNCH_INTENT, reserved_run_id=run_id, gameplay_attempt_id=gameplay_attempt_id, launch_nonce=nonce)

    def activate(self, process_ref: str, *, source: str = "observer") -> "LaunchLifecycle":
        if self.state is not LaunchState.LAUNCH_INTENT or source not in {"observer", "reconciliation"}:
            raise ValueError("activation requires one durable launch intent")
        return replace(self, state=LaunchState.FORMAL_RUN_ACTIVATED, process_ref=process_ref, activation_source=source)

    def create_process_failed(self) -> "LaunchLifecycle":
        if self.state is not LaunchState.LAUNCH_INTENT: raise ValueError("no launch intent")
        return replace(self, state=LaunchState.LAUNCH_GATE_FAILED, failure_reason="CREATE_PROCESS_FAILED")

    def launch_uncertain(self) -> "LaunchLifecycle":
        if self.state is not LaunchState.LAUNCH_INTENT: raise ValueError("no launch intent")
        return replace(self, state=LaunchState.LAUNCH_UNCERTAIN, failure_reason="PROCESS_IDENTITY_UNCERTAIN")

    @property
    def counts_toward_outcome_denominator(self) -> bool:
        return self.state in {LaunchState.FORMAL_RUN_ACTIVATED, LaunchState.TERMINAL} and self.process_ref is not None

    @property
    def campaign_blocked(self) -> bool: return self.state is LaunchState.LAUNCH_UNCERTAIN
    @property
    def replacement_allowed(self) -> bool: return False

    def to_wire(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "attempt_id": self.attempt_id, "state": self.state.value,
                "reserved_run_id": self.reserved_run_id, "gameplay_attempt_id": self.gameplay_attempt_id,
                "launch_nonce": self.launch_nonce, "process_ref": self.process_ref,
                "activation_source": self.activation_source, "failure_reason": self.failure_reason}

    @property
    def record_hash(self) -> str: return canonical_hash(self.to_wire())

    @classmethod
    def from_wire(cls, data: Mapping[str, Any]) -> "LaunchLifecycle":
        if set(data) != cls._KEYS: raise ValueError("unknown or missing launch lifecycle fields")
        try: state = LaunchState(data["state"])
        except (ValueError, TypeError) as exc: raise ValueError("unknown launch state") from exc
        return cls(attempt_id=data["attempt_id"], state=state, reserved_run_id=data["reserved_run_id"],
                   gameplay_attempt_id=data["gameplay_attempt_id"], launch_nonce=data["launch_nonce"],
                   process_ref=data["process_ref"], activation_source=data["activation_source"],
                   failure_reason=data["failure_reason"], schema_version=data["schema_version"])
