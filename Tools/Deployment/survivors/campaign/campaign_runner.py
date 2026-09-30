"""Survivors live canary campaign の stage 実行を順序固定で orchestration します。

06-02 の campaign 契約で prerequisite と event 列を検証し、06-03 の durable ledger へ
launch intent を commit してから broker へ launch を1回だけ依頼します。ledger で確認済みの
kernel identity だけを controller へ渡し、save lifecycle・operator checkpoint・controller/helper
停止確認・artifact flush を決まった順序で実行します。report は 06-02 reporter の出力をそのまま保存します。
runner 自身は process 生成も SQLite 行の操作も行わず、06-03 の public API だけを呼びます。
"""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, NamedTuple

from reinbalance_survivors_contracts.canonical_json import canonical_hash, canonical_json_bytes, sha256_hex

from . import campaign_schema
from .campaign_report import generate_campaign_report
from .campaign_schema import STAGE_POLICIES, CampaignEvent, CampaignManifest, EventType, campaign_manifest_hash
from .durable_launch_store import Classification, LaunchIntent, LaunchStage, LedgerError, new_launch_nonce
from .save_lifecycle import SaveLifecycleError
from .win32_launch_broker import build_launch_request

CAMPAIGN_RUN_MODE = "formal_single_attempt"
STAGES = tuple(STAGE_POLICIES)
PLAN = "campaign/plan.json"
SUPERSEDED = "campaign/superseded.json"
SUMMARY = "campaign/summary.json"
PREFLIGHT_STREAM = "preflight_attempts"
GATE_STREAM = "launch_gates"
OUTCOME_STREAM = "launched_outcomes"
CHAIN_STREAM = "campaign_chain"
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}")
_DIGEST = re.compile(r"[0-9a-f]{64}")
_PENDING = frozenset(
    {
        LaunchStage.LAUNCH_INTENT,
        LaunchStage.PROCESS_ATTESTED,
        LaunchStage.RESUME_INTENT,
        LaunchStage.PROCESS_LAUNCH_CONFIRMED,
    }
)
_BLOCKING = frozenset({EventType.PREFLIGHT_FAILED, EventType.LAUNCH_GATE_FAILED, EventType.LAUNCH_UNCERTAIN})
_TARGET_SUCCESS = frozenset({"confirmed", "pending", "not_reached"})


class RunnerError(RuntimeError):
    """runner が fail-closed で処理を止めたことを表す基底例外です。

    どの派生例外でも、slot の置き換えや再利用は行われていません。
    """


class PrerequisiteMismatch(RunnerError):
    """formal prerequisite・threshold parent・canonical save の束縛が一致しないことを表します。

    save や ledger に触れる前に検出され、campaign は開始されません。
    """


class PlanMismatch(RunnerError):
    """記録済み campaign plan と異なる plan で再開しようとしたことを表します。

    threshold や parent を変えた場合は新しい campaign id で C0 からやり直します。
    """


class StageOrderError(RunnerError):
    """下位 stage 未 pass・完了済み stage の再実行など stage 順序違反を表します。

    上位 stage は下位 stage の promotion を確認してからしか開始できません。
    """


class CampaignBlocked(RunnerError):
    """uncertain launch や supersede により campaign が停止済みであることを表します。

    新しい campaign id で最初からやり直す必要があります。
    """


class UnreconciledLaunch(RunnerError):
    """ledger に未 reconcile の launch intent が残っていることを表します。

    reconcile するまで replacement も次 slot の開始も禁止します。
    """


class ArtifactConflict(RunnerError):
    """write-once artifact を別内容で上書きしようとしたことを表します。

    manifest・outcome・hash 記録の改変はこの例外で拒否されます。
    """


class RunnerState(StrEnum):
    """runner と stage 実行の状態名です。

    stage 結果と runner.state の両方でこの値を使います。
    """

    IDLE = "idle"
    READY = "ready"
    PREREQUISITE_MISMATCH = "prerequisite_mismatch"
    STOPPED = "stopped"
    STAGE_PASSED = "stage_passed"
    STAGE_NOT_PROMOTED = "stage_not_promoted"
    PREFLIGHT_BLOCKED = "preflight_blocked"
    LAUNCH_GATE_BLOCKED = "launch_gate_blocked"
    CAMPAIGN_BLOCKED = "campaign_blocked"
    SUPERSEDED = "superseded"
    FINISHED = "finished"


def _digest(value: Any, name: str) -> str:
    """lowercase SHA-256 hex を検証します。

    plan の hash field に共通で使います。
    """
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _json_value(value: Any) -> Any:
    """canonical JSON で表せる値の独立 copy を返します。

    tuple は list になり、呼び出し側の mutable 値と切り離されます。
    """
    return json.loads(canonical_json_bytes(value))


class ArtifactStore:
    """campaign artifact を write-once file と append-only JSONL stream で保存します。

    file は一時 file へ書いて fsync してから hard link で作成するので、既存 file を
    別内容で置き換えることはできません。stream は追記と fsync だけを行い、読み込み時に
    壊れた行があれば fail-closed で例外にします。
    """

    def __init__(self, root: str | os.PathLike[str]) -> None:
        """artifact root directory を作成して保持します。

        root 外への path は受け付けません。
        """
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        """artifact 名を root 配下の path に変換します。

        `/` 区切りの各要素は英数字・`.`・`_`・`-` だけを許し、`..` などを拒否します。
        """
        parts = name.split("/") if isinstance(name, str) else []
        if not parts or any(_ID.fullmatch(part) is None or set(part) <= {"."} for part in parts):
            raise ValueError(f"invalid artifact name: {name!r}")
        return self.root.joinpath(*parts)

    def exists(self, name: str) -> bool:
        """artifact が存在するかを返します。

        stream ではなく write-once file を対象にします。
        """
        return self._path(name).is_file()

    def read(self, name: str) -> bytes:
        """artifact の bytes を読みます。

        存在しなければ OSError です。
        """
        return self._path(name).read_bytes()

    def read_json(self, name: str) -> Any:
        """JSON artifact を読みます。

        write-once file の canonical JSON を dict などへ戻します。
        """
        return json.loads(self.read(name))

    def put_immutable(self, name: str, data: bytes) -> str:
        """write-once file を作成して SHA-256 を返します。

        同じ内容が既にあれば何もせず hash を返し、別内容なら ArtifactConflict です。
        """
        path = self._path(name)
        digest = sha256_hex(bytes(data))
        if path.exists():
            if sha256_hex(path.read_bytes()) != digest:
                raise ArtifactConflict(f"artifact {name} is immutable and already has different content")
            return digest
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
        try:
            with open(temporary, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                return self.put_immutable(name, data)
        finally:
            temporary.unlink(missing_ok=True)
        return digest

    def put_json(self, name: str, value: Any) -> str:
        """JSON 値を canonical JSON の write-once file として保存します。

        同じ値の再保存は idempotent です。
        """
        return self.put_immutable(name, canonical_json_bytes(value))

    def append(self, stream: str, records: Sequence[Mapping[str, Any]]) -> None:
        """stream へ record を追記して fsync します。

        複数 record は1回の write でまとめて書きます。
        """
        path = self._path(f"streams/{stream}.jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        data = b"".join(canonical_json_bytes(dict(record)) + b"\n" for record in records)
        with open(path, "ab") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())

    def stream(self, stream: str) -> list[dict[str, Any]]:
        """stream の全 record を読みます。

        途中で切れた行や JSON object 以外の行があれば ValueError です。
        """
        path = self._path(f"streams/{stream}.jsonl")
        if not path.exists():
            return []
        data = path.read_bytes()
        if data and not data.endswith(b"\n"):
            raise ValueError(f"stream {stream} has a torn trailing record")
        records = [json.loads(line) for line in data.splitlines()]
        if any(not isinstance(record, dict) for record in records):
            raise ValueError(f"stream {stream} has a non-object record")
        return records


def threshold_parent_hash(prerequisite_hashes: Mapping[str, str]) -> str:
    """metric floor の parent hash を exact-target と final perception evidence から導きます。

    formal plan の threshold_parent_hash はこの値と一致しなければなりません。
    """
    return canonical_hash(
        {"target": prerequisite_hashes["target"], "perception_final": prerequisite_hashes["perception_final"]}
    )


@dataclass(frozen=True)
class CampaignPlan:
    """campaign 全体で固定する launch 条件・threshold・prerequisite の束です。

    artifact store へ write-once で保存され、code/model/config/threshold を変える場合は
    新しい campaign id が必要です。synthetic plan は development-only で formal 対象外です。
    """

    campaign_id: str
    mode: str
    executable_path: str
    executable_hash: str
    build_hash: str
    config_hash: str
    argv: tuple[str, ...]
    canonical_save_hash: str
    thresholds: Mapping[str, float]
    threshold_parent_hash: str
    prerequisites: Mapping[str, Any] | None = None
    prerequisite_parent_hash: str | None = None
    grace_seconds: int = 120

    def __post_init__(self) -> None:
        """plan field の形式を検証して独立 copy にします。

        prerequisite の内容検証は runner の begin で 06-02 validator が行います。
        """
        if not isinstance(self.campaign_id, str) or _ID.fullmatch(self.campaign_id) is None:
            raise ValueError("campaign_id must match [A-Za-z0-9][A-Za-z0-9._-]*")
        if self.mode not in {"formal", "synthetic"}:
            raise ValueError("mode must be formal or synthetic")
        for name in ("executable_hash", "build_hash", "config_hash", "canonical_save_hash", "threshold_parent_hash"):
            _digest(getattr(self, name), name)
        argv = tuple(self.argv) if not isinstance(self.argv, (str, bytes)) else ()
        if not os.path.isabs(self.executable_path) or not argv or argv[0] != self.executable_path:
            raise ValueError("argv must start with the absolute executable_path")
        if not all(isinstance(item, str) for item in argv):
            raise ValueError("argv must contain strings")
        object.__setattr__(self, "argv", argv)
        if not isinstance(self.thresholds, Mapping) or not all(
            isinstance(key, str) and isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value)
            for key, value in self.thresholds.items()
        ):
            raise ValueError("thresholds must map names to finite numbers")
        object.__setattr__(self, "thresholds", _json_value(dict(self.thresholds)))
        if self.mode == "synthetic":
            if self.prerequisites is not None or self.prerequisite_parent_hash is not None:
                raise ValueError("synthetic plans cannot carry formal prerequisites")
        else:
            if not isinstance(self.prerequisites, Mapping):
                raise ValueError("formal plans require prerequisites")
            _digest(self.prerequisite_parent_hash, "prerequisite_parent_hash")
            object.__setattr__(self, "prerequisites", _json_value(dict(self.prerequisites)))
        if type(self.grace_seconds) is not int or self.grace_seconds < 0:
            raise ValueError("grace_seconds must be a non-negative integer")

    @property
    def development_only(self) -> bool:
        """synthetic plan なら True です。

        development-only の run は formal campaign の証跡になりません。
        """
        return self.mode == "synthetic"

    def to_wire(self) -> dict[str, Any]:
        """plan を canonical JSON 用の object にします。

        run mode・UI restart 無効・formal 可否も固定値として含めます。
        """
        return _json_value(
            {
                "campaign_id": self.campaign_id,
                "mode": self.mode,
                "executable_path": self.executable_path,
                "executable_hash": self.executable_hash,
                "build_hash": self.build_hash,
                "config_hash": self.config_hash,
                "argv": list(self.argv),
                "canonical_save_hash": self.canonical_save_hash,
                "thresholds": dict(self.thresholds),
                "threshold_parent_hash": self.threshold_parent_hash,
                "prerequisites": self.prerequisites,
                "prerequisite_parent_hash": self.prerequisite_parent_hash,
                "grace_seconds": self.grace_seconds,
                "campaign_run_mode": CAMPAIGN_RUN_MODE,
                "ui_restart_enabled": False,
                "development_only": self.development_only,
                "formal_campaign_eligible": not self.development_only,
            }
        )

    @property
    def plan_hash(self) -> str:
        """plan wire object の canonical hash です。

        stage 実行記録と per-run manifest がこの値へ束縛されます。
        """
        return canonical_hash(self.to_wire())


class SlotIdentity(NamedTuple):
    """1 slot に1件だけ予約する deterministic identity です。

    同じ stage 実行・slot からは常に同じ id が作られ、ledger が再利用を拒否します。
    """

    attempt_id: str
    reserved_run_id: str
    gameplay_attempt_id: str


def slot_identity(stage_execution_id: str, slot_id: int) -> SlotIdentity:
    """stage 実行 id と slot 番号から attempt / run / gameplay attempt id を作ります。

    process 内 restart 用の2件目の gameplay attempt id は作りません。
    """
    base = f"{stage_execution_id}.s{slot_id:02d}"
    return SlotIdentity(f"{base}.attempt", f"{base}.run", f"{base}.gameplay-0")


@dataclass(frozen=True)
class RunSpec:
    """controller へ渡す、ledger で確認済みの run identity です。

    process_ref / job_ref / target_pid は broker response ではなく ledger の attestation から作ります。
    """

    campaign_id: str
    stage_execution_id: str
    stage: str
    slot_id: int
    reserved_run_id: str
    gameplay_attempt_id: str
    campaign_run_mode: str
    process_ref: str
    job_ref: str
    target_pid: int
    duration_seconds: int
    ui_restart_enabled: bool
    development_only: bool

    def to_wire(self) -> dict[str, Any]:
        """spec を JSON object にします。

        operator checkpoint と per-run manifest に使います。
        """
        return asdict(self)


@dataclass(frozen=True)
class RunResult:
    """controller が返す1 run の結果 summary です。

    target success の pending/confirmed、death、level/gems/kills/choices、unknown/fallback、
    latency、menu input 数、telemetry/video/lease audit の参照を持ちます。
    """

    gameplay_attempt_id: str
    gameplay_entries: int
    target_success: str
    died: bool
    level: int
    gems: int
    kills: int
    choices: int
    unknown_frames: int
    fallback_decisions: int
    latency_ms: Mapping[str, float]
    menu_inputs_sent: int
    safety_stop: str | None = None
    telemetry_refs: tuple[str, ...] = ()
    video_refs: tuple[str, ...] = ()
    lease_audit_ref: str | None = None

    def __post_init__(self) -> None:
        """結果値の型と範囲を検証します。

        不正な結果は run 側の safety failure として扱われます。
        """
        if self.target_success not in _TARGET_SUCCESS:
            raise ValueError(f"target_success must be one of {sorted(_TARGET_SUCCESS)}")
        for name in ("gameplay_entries", "level", "gems", "kills", "choices", "unknown_frames",
                     "fallback_decisions", "menu_inputs_sent"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        object.__setattr__(self, "latency_ms", _json_value(dict(self.latency_ms)))
        object.__setattr__(self, "telemetry_refs", tuple(self.telemetry_refs))
        object.__setattr__(self, "video_refs", tuple(self.video_refs))

    def metrics(self) -> dict[str, Any]:
        """06-02 reporter の event details に載せる gameplay 指標です。

        telemetry 参照などの artifact 情報は含めません。
        """
        return {
            "target_success": self.target_success,
            "died": self.died,
            "level": self.level,
            "gems": self.gems,
            "kills": self.kills,
            "choices": self.choices,
            "unknown_frames": self.unknown_frames,
            "fallback_decisions": self.fallback_decisions,
            "latency_ms": dict(self.latency_ms),
            "menu_inputs_sent": self.menu_inputs_sent,
            "gameplay_entries": self.gameplay_entries,
        }

    def to_wire(self) -> dict[str, Any]:
        """結果全体を JSON object にします。

        per-run manifest に保存します。
        """
        return _json_value(asdict(self))


@dataclass(frozen=True)
class ReleaseObservation:
    """external observer が確認した helper lease release の結果です。

    confirmed が True でない限り、その run は safety failure です。
    """

    confirmed: bool
    observer_ref: str
    release_ms: float | None = None

    def to_wire(self) -> dict[str, Any]:
        """観測結果を JSON object にします。

        per-run manifest と event details に使います。
        """
        return _json_value(asdict(self))


@dataclass(frozen=True)
class StageResult:
    """1 stage 実行の結果です。

    report_hash は 06-02 reporter の canonical output の hash で、停止中なら None です。
    """

    stage: str
    stage_execution_id: str
    state: RunnerState
    report_hash: str | None
    slot_events: Mapping[int, tuple[str, ...]] = field(default_factory=dict)


def result_failures(spec: RunSpec, result: RunResult) -> list[tuple[EventType, str]]:
    """controller 結果から safety / artifact failure を抽出します。

    gameplay attempt の差し替え、process 内 restart、UI restart 無効時の menu input、
    controller safety stop、telemetry 欠落を検出します。
    """
    failures: list[tuple[EventType, str]] = []
    if result.gameplay_attempt_id != spec.gameplay_attempt_id:
        failures.append((EventType.SAFETY_FAILURE, "gameplay_attempt_mismatch"))
    if result.gameplay_entries != 1:
        failures.append((EventType.SAFETY_FAILURE, f"gameplay_entries_{result.gameplay_entries}"))
    if result.menu_inputs_sent and not spec.ui_restart_enabled:
        failures.append((EventType.SAFETY_FAILURE, "menu_input_while_ui_restart_disabled"))
    if result.safety_stop:
        failures.append((EventType.SAFETY_FAILURE, f"controller_safety_stop:{result.safety_stop}"))
    if not result.telemetry_refs:
        failures.append((EventType.ARTIFACT_FAILURE, "telemetry_missing"))
    return failures


def outcome_event(
    slot_id: int,
    manifest_hash: str,
    result: RunResult | None,
    failures: Sequence[tuple[EventType, str]],
    release: ReleaseObservation | None,
) -> CampaignEvent:
    """run の結果を 06-02 reporter 入力の terminal event に変換します。

    safety > artifact > gameplay の優先順で種別を決め、確認済み target success かつ
    死亡なしのときだけ SUCCESS にします。指標と全 failure は details に残します。
    """
    details: dict[str, Any] = {
        "failures": [f"{kind.value}:{reason}" for kind, reason in failures],
        "release": None if release is None else release.to_wire(),
    }
    if result is not None:
        details.update(result.metrics())
    common = {"slot_id": slot_id, "campaign_manifest_hash": manifest_hash, "details": details}
    for kind in (EventType.SAFETY_FAILURE, EventType.ARTIFACT_FAILURE):
        reasons = [reason for item, reason in failures if item is kind]
        if reasons:
            return CampaignEvent(kind, failure_reason=reasons[0], **common)
    if result is None:
        return CampaignEvent(EventType.SAFETY_FAILURE, failure_reason="run_result_missing", **common)
    if result.target_success == "confirmed" and not result.died:
        return CampaignEvent(EventType.SUCCESS, **common)
    reason = "death" if result.died else f"target_success_{result.target_success}"
    return CampaignEvent(EventType.GAMEPLAY_FAILURE, failure_reason=reason, **common)


def validate_parent(plan: CampaignPlan, stage: str, record: Mapping[str, Any], report: Mapping[str, Any]) -> None:
    """上位 stage を開始してよい下位 stage 実行かを検証します。

    直下の stage であること、promotion 済みであること、同じ plan・threshold parent・
    prerequisite parent に束縛されていること、formal plan なら development-only でないこと、
    保存済み report hash と一致することを要求します。
    """
    if stage not in STAGES or stage == STAGES[0]:
        raise StageOrderError(f"{stage} has no lower stage parent")
    lower = STAGES[STAGES.index(stage) - 1]
    if record.get("stage") != lower or report.get("stage") != lower:
        raise StageOrderError(f"{stage} requires a {lower} parent")
    if record.get("state") != RunnerState.STAGE_PASSED.value or report.get("promotion_eligible") is not True:
        raise StageOrderError(f"lower stage {lower} has not passed")
    if canonical_hash(dict(report)) != record.get("report_hash"):
        raise PlanMismatch("parent report does not match its recorded report hash")
    if record.get("plan_hash") != plan.plan_hash:
        raise PlanMismatch("parent stage belongs to a different campaign plan")
    if (record.get("threshold_parent_hash") != plan.threshold_parent_hash
            or record.get("thresholds_hash") != canonical_hash(dict(plan.thresholds))):
        raise PlanMismatch("threshold parent changed since the lower stage; start a new campaign id")
    if report.get("prerequisite_parent_hash") != plan.prerequisite_parent_hash:
        raise PlanMismatch("prerequisite parent changed since the lower stage")
    if not plan.development_only and (
        report.get("development_only") is not False or report.get("formal_parent_eligible") is not True
    ):
        raise StageOrderError("a development-only parent cannot gate a formal stage")


@dataclass
class _Execution:
    """実行中 stage の in-memory 状態です。

    stream から読み戻した event を slot ごとに保持します。
    """

    stage: str
    execution_id: str
    manifest: CampaignManifest
    manifest_hash: str
    blocked_ids: tuple[str, ...]
    parent_id: str | None
    preflight: dict[int, list[CampaignEvent]] = field(default_factory=dict)
    outcomes: dict[int, list[CampaignEvent]] = field(default_factory=dict)
    flushed_attempts: set[str] = field(default_factory=set)

    @property
    def slots(self) -> int:
        """stage policy の slot 数です。

        06-02 manifest の expected_slots と同じ値です。
        """
        return self.manifest.expected_slots


class CampaignRunner:
    """campaign plan を C0〜C4 の stage 実行として進める orchestrator です。

    port は duck typing で受け取ります。
    - ledger: 06-03 `DurableLaunchStore`(public method のみ使用)
    - broker: launch request dict を受けて response dict を返す callable(`request_broker` の client)
    - probe: `ProcessIdentity -> ProbeResult`
    - controller: `start(RunSpec) -> handle`(handle は controller_pid / helper_pid / wait / terminate)
    - helper: `confirm_release(helper_pid, spec_wire) -> ReleaseObservation`(external observer)
    - target: `stopped() -> bool`(game/launcher 停止)と `stop(spec_wire) -> bool`
    - operator: `checkpoint(name, context) -> bool`(cloud sync / arm / focus / manual start)
    - save: `SaveLifecycle`、validator: 既定は 06-02 `campaign_schema` module
    formal plan では development_only な port を拒否します。
    """

    def __init__(
        self,
        plan: CampaignPlan,
        *,
        artifacts: ArtifactStore,
        ledger: Any,
        broker: Callable[[Mapping[str, Any]], Mapping[str, Any]],
        probe: Callable[[Any], Any],
        controller: Any,
        helper: Any,
        target: Any,
        operator: Any,
        save: Any,
        validator: Any = campaign_schema,
    ) -> None:
        """plan と port を保持します。

        artifact・save・ledger への書き込みは begin 以降にだけ行います。
        """
        if not isinstance(plan, CampaignPlan):
            raise TypeError("plan must be a CampaignPlan")
        self.plan = plan
        self._artifacts = artifacts
        self._ledger = ledger
        self._broker = broker
        self._probe = probe
        self._controller = controller
        self._helper = helper
        self._target = target
        self._operator = operator
        self._save = save
        self._validator = validator
        self._prerequisites: Any = None
        self.state = RunnerState.IDLE

    # ---- campaign lifecycle -------------------------------------------------

    def begin(self) -> None:
        """campaign を開始可能にします。

        prerequisite・threshold parent・canonical save を検証し(不一致なら何も書かない)、
        plan を write-once で保存し、元 save の backup と cloud sync 無効化 checkpoint を記録します。
        """
        try:
            self._prerequisites = self._validated_prerequisites()
        except (PrerequisiteMismatch, ValueError) as exc:
            self.state = RunnerState.PREREQUISITE_MISMATCH
            raise PrerequisiteMismatch(str(exc)) from exc
        if not self.plan.development_only:
            ports = {"controller": self._controller, "helper": self._helper, "target": self._target,
                     "operator": self._operator}
            development = sorted(name for name, port in ports.items() if getattr(port, "development_only", True))
            if development:
                raise RunnerError(f"formal campaigns cannot use development-only ports: {development}")
        self._write_plan()
        self._save.backup_original()
        if not self._save.evidence()["cloud_sync_attested"]:
            context = {"campaign_id": self.plan.campaign_id, "plan_hash": self.plan.plan_hash}
            if self._operator.checkpoint("cloud_sync_disabled", context) is not True:
                raise RunnerError("operator did not attest that cloud sync is disabled")
            self._save.attest_cloud_sync(
                {"cloud_sync_disabled": True, "operator_checkpoint": "cloud_sync_disabled", **context}
            )
        self.state = RunnerState.READY

    def _validated_prerequisites(self) -> Any:
        """formal prerequisite と threshold parent・canonical save を検証します。

        synthetic plan は prerequisite を持たず None を返します。
        """
        evidence = self._save.evidence()
        if evidence["canonical_sha256"] != self.plan.canonical_save_hash:
            raise PrerequisiteMismatch("registered canonical save does not match the plan canonical_save_hash")
        if self.plan.development_only:
            return None
        validated = self._validator.validate_prerequisites(
            self.plan.prerequisites, expected_parent_hash=self.plan.prerequisite_parent_hash
        )
        if threshold_parent_hash(validated.hashes) != self.plan.threshold_parent_hash:
            raise PrerequisiteMismatch("threshold_parent_hash does not match target/perception_final evidence")
        return validated

    def _write_plan(self) -> None:
        """plan を write-once で保存します。

        記録済み plan と違えば、変更 field と live 結果の有無を示して拒否します。
        """
        wire = self.plan.to_wire()
        try:
            self._artifacts.put_json(PLAN, wire)
        except ArtifactConflict:
            stored = self._artifacts.read_json(PLAN)
            changed = sorted(key for key in set(stored) | set(wire) if stored.get(key) != wire.get(key))
            message = f"campaign plan is immutable; changed fields {changed}"
            if self._executions() and {"thresholds", "threshold_parent_hash"} & set(changed):
                message += "; thresholds cannot change after live results"
            raise PlanMismatch(message + "; start a new campaign id from C0") from None

    def supersede(self, successor_campaign_id: str, reason: str) -> None:
        """campaign を後継 campaign id で置き換えたことを記録します。

        以後この campaign では stage を開始できません。
        """
        record = {"kind": "superseded", "campaign_id": self.plan.campaign_id,
                  "successor_campaign_id": successor_campaign_id, "reason": reason}
        self._artifacts.put_json(SUPERSEDED, record)
        self._append_chain_once(record)
        self.state = RunnerState.SUPERSEDED

    def finish(self) -> Mapping[str, Any]:
        """元 save を復元し、campaign summary を write-once で保存します。

        backup・cloud sync attestation・全 run の pre/post hash・restore verdict が
        そろった formal campaign だけを formal_evidence_eligible にします。
        """
        verdict = self._save.restore_original()
        executions = []
        runs_ready = True
        in_progress = []
        for stage, execution_id in self._executions():
            name = f"stages/{execution_id}/execution.json"
            if not self._artifacts.exists(name):
                in_progress.append(execution_id)
                continue
            executions.append(self._artifacts.read_json(name))
            for slot in range(STAGE_POLICIES[stage].slot_count):
                run = f"stages/{execution_id}/runs/slot-{slot:02d}.json"
                if self._artifacts.exists(run):
                    runs_ready &= self._artifacts.read_json(run)["formal_evidence_eligible"] is True
        evidence = self._save.evidence()
        summary = {
            "campaign_id": self.plan.campaign_id,
            "plan_hash": self.plan.plan_hash,
            "executions": executions,
            "in_progress_execution_ids": in_progress,
            "save_evidence": evidence,
            "restore_verdict": dict(verdict),
            "development_only": self.plan.development_only,
            "formal_campaign_eligible": not self.plan.development_only,
            "formal_evidence_eligible": (
                not self.plan.development_only and bool(executions) and runs_ready and not in_progress
                and evidence["backup_sha256"] is not None and evidence["cloud_sync_attested"]
                and verdict.get("status") == "PASS"
            ),
        }
        self._artifacts.put_json(SUMMARY, summary)
        self.state = RunnerState.FINISHED
        return summary

    # ---- stage execution ----------------------------------------------------

    def run_stage(self, stage: str, *, max_slots: int | None = None) -> StageResult:
        """stage を開始または再開し、slot を順に実行します。

        preflight failure・launch gate failure・uncertain launch が出た時点で stage を止めます。
        max_slots を指定すると、その数の slot を処理した時点で STOPPED として返します(resume 可能)。
        """
        if self.state in {RunnerState.IDLE, RunnerState.PREREQUISITE_MISMATCH}:
            raise RunnerError("begin() must succeed before running a stage")
        if stage not in STAGES:
            raise StageOrderError(f"unknown stage: {stage}")
        self._require_not_blocked()
        ctx = self._open_execution(stage)
        processed = 0
        for slot in range(ctx.slots):
            last = self._slot_last(ctx, slot)
            if last is not None and (last in _BLOCKING or last in campaign_schema.TERMINAL_OUTCOMES):
                if last in _BLOCKING:
                    break
                continue
            if max_slots is not None and processed >= max_slots:
                self.state = RunnerState.STOPPED
                return StageResult(stage, ctx.execution_id, RunnerState.STOPPED, None, self._slot_events(ctx))
            processed += 1
            try:
                last = self._run_slot(ctx, slot)
            except CampaignBlocked:
                self.state = RunnerState.CAMPAIGN_BLOCKED
                raise
            if last in _BLOCKING:
                break
        return self._close_execution(ctx)

    def _executions(self) -> list[tuple[str, str]]:
        """記録済みの全 stage 実行を (stage, 実行 id) で返します。

        各 stage の x1, x2, ... を manifest の存在で列挙します。
        """
        found = []
        for stage in STAGES:
            number = 1
            while self._artifacts.exists(f"stages/{self._execution_id(stage, number)}/manifest.json"):
                found.append((stage, self._execution_id(stage, number)))
                number += 1
        return found

    def _execution_id(self, stage: str, number: int) -> str:
        """stage 実行 id を作ります。

        preflight_blocked 後の再実行は番号を進めた新しい id になります。
        """
        return f"{self.plan.campaign_id}.{stage}.x{number}"

    def _require_not_blocked(self) -> None:
        """campaign が block / supersede 済みでないことを確認します。

        uncertain launch 後は同じ campaign id で stage を開始できません。
        """
        if self._artifacts.exists(SUPERSEDED):
            raise CampaignBlocked("campaign was superseded; use the successor campaign id")
        for _stage, execution_id in self._executions():
            name = f"stages/{execution_id}/execution.json"
            if (self._artifacts.exists(name)
                    and self._artifacts.read_json(name)["state"] == RunnerState.CAMPAIGN_BLOCKED.value):
                raise CampaignBlocked(f"campaign is blocked by {execution_id}; start a new campaign id")

    def _open_execution(self, stage: str) -> _Execution:
        """stage 実行を再開するか、条件を検証して新規作成します。

        完了済み stage の再実行は preflight_blocked 後だけ許し、上位 stage は
        promotion 済みの直下 stage を parent に要求します。
        """
        existing = [execution_id for item, execution_id in self._executions() if item == stage]
        if existing and not self._artifacts.exists(f"stages/{existing[-1]}/execution.json"):
            return self._load_execution(stage, existing[-1])
        if existing:
            state = self._artifacts.read_json(f"stages/{existing[-1]}/execution.json")["state"]
            if state != RunnerState.PREFLIGHT_BLOCKED.value:
                raise StageOrderError(f"{stage} already concluded as {state}; start a new campaign id to rerun it")
        parent_id = None
        if stage != STAGES[0]:
            lower = STAGES[STAGES.index(stage) - 1]
            passed = [
                execution_id for item, execution_id in self._executions()
                if item == lower and self._artifacts.exists(f"stages/{execution_id}/execution.json")
                and self._artifacts.read_json(f"stages/{execution_id}/execution.json")["state"]
                == RunnerState.STAGE_PASSED.value
            ]
            if not passed:
                raise StageOrderError(f"lower stage {lower} has not passed; {stage} cannot start")
            parent_id = passed[-1]
            validate_parent(
                self.plan, stage,
                self._artifacts.read_json(f"stages/{parent_id}/execution.json"),
                self._artifacts.read_json(f"stages/{parent_id}/report.json"),
            )
        execution_id = self._execution_id(stage, len(existing) + 1)
        policy = STAGE_POLICIES[stage]
        manifest = CampaignManifest(
            campaign_id=execution_id,
            mode=self.plan.mode,
            stage=stage,
            expected_slots=policy.slot_count,
            development_only=self.plan.development_only,
            prerequisites=self._prerequisites,
            prerequisite_parent_hash=self.plan.prerequisite_parent_hash,
        )
        self._artifacts.put_json(
            f"stages/{execution_id}/manifest.json",
            {
                "manifest": manifest.to_wire(),
                "plan_hash": self.plan.plan_hash,
                "parent_execution_id": parent_id,
                "blocked_execution_ids": existing,
                "campaign_run_mode": CAMPAIGN_RUN_MODE,
                "development_only": self.plan.development_only,
                "formal_campaign_eligible": not self.plan.development_only,
            },
        )
        return self._load_execution(stage, execution_id)

    def _load_execution(self, stage: str, execution_id: str) -> _Execution:
        """stage 実行の manifest と stream を読み戻します。

        plan hash の不一致や 06-02 validator が拒否する event 列(outcome の重複など)は fail-closed です。
        """
        stored = self._artifacts.read_json(f"stages/{execution_id}/manifest.json")
        if stored["plan_hash"] != self.plan.plan_hash:
            raise PlanMismatch(f"{execution_id} was created by a different campaign plan")
        manifest = CampaignManifest.from_wire(stored["manifest"])
        ctx = _Execution(stage, execution_id, manifest, campaign_manifest_hash(manifest),
                         tuple(stored["blocked_execution_ids"]), stored["parent_execution_id"])
        for stream, target in ((PREFLIGHT_STREAM, ctx.preflight), (OUTCOME_STREAM, ctx.outcomes)):
            for record in self._artifacts.stream(stream):
                if record.get("stage_execution_id") == execution_id:
                    event = CampaignEvent.from_wire(record["event"])
                    target.setdefault(event.slot_id, []).append(event)
        ctx.flushed_attempts = {
            record["attempt_id"] for record in self._artifacts.stream(GATE_STREAM)
            if record.get("stage_execution_id") == execution_id
        }
        self._validator.validate_campaign_events(
            self._events(ctx), expected_manifest_hash=ctx.manifest_hash, expected_slots=ctx.slots
        )
        return ctx

    def _events(self, ctx: _Execution) -> list[CampaignEvent]:
        """stage 実行の 06-02 event 列を slot 順に組み立てます。

        preflight event は preflight stream、launch/activation event は ledger、
        terminal outcome は outcome stream を正とします。
        """
        attempts = set(self._ledger.attempt_ids())
        events: list[CampaignEvent] = []
        for slot in sorted(set(ctx.preflight) | set(ctx.outcomes)):
            events.extend(ctx.preflight.get(slot, ()))
            if self._owned_history(ctx, slot, attempts) is not None:
                events.extend(self._ledger.campaign_events(slot_identity(ctx.execution_id, slot).attempt_id))
            events.extend(ctx.outcomes.get(slot, ()))
        return events

    def _owned_history(self, ctx: _Execution, slot: int, attempts: set[str] | None = None) -> Any:
        """slot の ledger 履歴を、この runner が記録した launch nonce と一致するときだけ返します。

        同じ attempt id の intent が別 runner のものなら None で、その ledger 行を自分の
        slot として採用・再開しません。nonce は ATTEMPT_PREFLIGHT の details に記録します。
        """
        attempt = slot_identity(ctx.execution_id, slot).attempt_id
        if attempt not in (attempts if attempts is not None else set(self._ledger.attempt_ids())):
            return None
        nonce = next((event.details.get("launch_nonce") for event in ctx.preflight.get(slot, ())
                      if event.event_type is EventType.ATTEMPT_PREFLIGHT), None)
        history = self._ledger.history(attempt)
        return history if nonce is not None and history.intent.launch_nonce == nonce else None

    def _slot_events(self, ctx: _Execution) -> dict[int, tuple[str, ...]]:
        """slot ごとの event 種別列を返します。

        StageResult の要約に使います。
        """
        grouped: dict[int, list[str]] = {}
        for event in self._events(ctx):
            grouped.setdefault(event.slot_id, []).append(event.event_type.value)
        return {slot: tuple(kinds) for slot, kinds in grouped.items()}

    def _slot_last(self, ctx: _Execution, slot: int) -> EventType | None:
        """slot の最後の event 種別を返します。

        まだ予約されていない slot は None です。
        """
        events = [event for event in self._events(ctx) if event.slot_id == slot]
        return events[-1].event_type if events else None

    def _append(self, ctx: _Execution, stream: str, events: Sequence[CampaignEvent], **extra: Any) -> None:
        """event を 06-02 validator で検証してから stream へ flush します。

        terminal 後の event や identity の再利用は validator が拒否し、何も書きません。
        """
        self._validator.validate_campaign_events(
            [*self._events(ctx), *events], expected_manifest_hash=ctx.manifest_hash, expected_slots=ctx.slots
        )
        self._artifacts.append(
            stream, [{"stage_execution_id": ctx.execution_id, "event": event.to_wire(), **extra} for event in events]
        )
        target = ctx.preflight if stream == PREFLIGHT_STREAM else ctx.outcomes
        for event in events:
            target.setdefault(event.slot_id, []).append(event)

    def _append_chain_once(self, record: Mapping[str, Any]) -> None:
        """blocked / superseded record を campaign chain stream へ1回だけ追記します。

        resume で同じ record を重複させません。
        """
        if dict(record) not in self._artifacts.stream(CHAIN_STREAM):
            self._artifacts.append(CHAIN_STREAM, [record])

    def _require_no_pending_launch(self) -> None:
        """ledger に未 reconcile の launch intent が無いことを確認します。

        1件でもあれば replacement や次 slot の開始を拒否します。
        """
        for attempt in self._ledger.attempt_ids():
            last = self._ledger.history(attempt).last_stage
            if last in _PENDING:
                raise UnreconciledLaunch(f"{attempt} is unreconciled at {last.value}; reconcile before launching")

    def _run_slot(self, ctx: _Execution, slot: int) -> EventType:
        """1 slot を preflight → intent → launch → activation → run → terminal まで進めます。

        再開時は ledger を reconcile して続きから進め、broker へは再送しません。
        """
        ids = slot_identity(ctx.execution_id, slot)
        owned = self._owned_history(ctx, slot) is not None
        response: Mapping[str, Any] | None = None
        if slot not in ctx.preflight:
            self._require_no_pending_launch()
            nonce = new_launch_nonce()
            self._append(ctx, PREFLIGHT_STREAM, [
                CampaignEvent(EventType.FORMAL_SLOT_RESERVED, slot_id=slot, campaign_manifest_hash=ctx.manifest_hash),
                CampaignEvent(EventType.ATTEMPT_PREFLIGHT, slot_id=slot, attempt_id=ids.attempt_id,
                              campaign_manifest_hash=ctx.manifest_hash, details={"launch_nonce": nonce}),
            ])
            reason = self._preflight(ids)
            if reason is None:
                intent = LaunchIntent(
                    campaign_manifest_hash=ctx.manifest_hash, slot_id=slot, attempt_id=ids.attempt_id,
                    reserved_run_id=ids.reserved_run_id, gameplay_attempt_id=ids.gameplay_attempt_id,
                    launch_nonce=nonce, executable_path=self.plan.executable_path,
                    executable_hash=self.plan.executable_hash, build_hash=self.plan.build_hash,
                    config_hash=self.plan.config_hash, argv=self.plan.argv,
                )
                try:
                    self._ledger.commit_intent(intent)
                except (LedgerError, ValueError) as exc:
                    if self._owned_history(ctx, slot) is None:
                        reason = f"launch_intent_commit_failed: {exc}"
            if reason is not None:
                self._append(ctx, PREFLIGHT_STREAM, [
                    CampaignEvent(EventType.PREFLIGHT_FAILED, slot_id=slot, attempt_id=ids.attempt_id,
                                  failure_reason=reason, campaign_manifest_hash=ctx.manifest_hash),
                ])
                return EventType.PREFLIGHT_FAILED
            response = self._send_launch(intent)
            last, resumed = self._conclude_launch(ids.attempt_id, response)
        elif not owned:
            self._append(ctx, PREFLIGHT_STREAM, [
                CampaignEvent(EventType.PREFLIGHT_FAILED, slot_id=slot, attempt_id=ids.attempt_id,
                              failure_reason="runner_interrupted_before_owned_launch_intent",
                              campaign_manifest_hash=ctx.manifest_hash),
            ])
            return EventType.PREFLIGHT_FAILED
        else:
            last, resumed = self._conclude_launch(ids.attempt_id, None)
        self._flush_gate(ctx, slot, ids.attempt_id, response)
        if last is not LaunchStage.FORMAL_RUN_ACTIVATED:
            return EventType(last.value)
        return self._finish_activated(ctx, slot, ids, interrupted=resumed)

    def _preflight(self, ids: SlotIdentity) -> str | None:
        """slot の preflight を行い、失敗理由(成功なら None)を返します。

        06-02 validator による prerequisite 再検証、save 証跡、game/launcher 停止、
        canonical save の temp verify → atomic replace を順に確認します。
        """
        if not self.plan.development_only:
            try:
                self._validator.validate_prerequisites(
                    self.plan.prerequisites, expected_parent_hash=self.plan.prerequisite_parent_hash
                )
            except ValueError as exc:
                return f"prerequisites_invalid: {exc}"
        evidence = self._save.evidence()
        if evidence["backup_sha256"] is None or not evidence["cloud_sync_attested"]:
            return "save_backup_or_cloud_sync_attestation_missing"
        if self._target.stopped() is not True:
            return "game_or_launcher_running"
        try:
            self._save.install_canonical(ids.attempt_id, self.plan.canonical_save_hash)
        except (SaveLifecycleError, OSError) as exc:
            return f"canonical_save_install_failed: {exc}"
        return None

    def _send_launch(self, intent: LaunchIntent) -> Mapping[str, Any] | None:
        """broker へ launch request を1回だけ送ります。

        timeout や pipe error は None(応答なし)として返し、分類は ledger から行います。
        """
        try:
            response = self._broker(build_launch_request(intent))
        except (OSError, ValueError) as exc:
            return {"status": "no_response", "error": f"{type(exc).__name__}: {exc}"}
        return response if isinstance(response, Mapping) else {"status": "no_response", "error": "non-object"}

    def _conclude_launch(self, attempt: str, response: Mapping[str, Any] | None) -> tuple[LaunchStage, bool]:
        """launch の結末を ledger から確定し、確認済みなら activation します。

        confirmed response は ledger の attestation と一致したときだけ normal activation にし、
        それ以外(timeout・uncertain・rejected・再開)は reconcile の分類に従います。
        生存を証明できない confirmed launch(orphan)は uncertain として campaign を block します。
        戻り値の bool は、今回ではなく以前に activation 済みだったかどうかです。
        """
        history = self._ledger.history(attempt)
        if history.last_stage is LaunchStage.FORMAL_RUN_ACTIVATED:
            return LaunchStage.FORMAL_RUN_ACTIVATED, True
        if response is not None and response.get("status") == Classification.CONFIRMED.value:
            source = "normal"
            attested = history.attestation
            if (attested is None or response.get("process_ref") != attested.identity.process_ref
                    or response.get("job_ref") != attested.job_ref):
                return self._orphan(attempt, "broker_response_identity_mismatch"), False
        else:
            reconciliation = self._ledger.reconcile(attempt, self._probe)
            if reconciliation.classification is not Classification.CONFIRMED:
                return self._ledger.history(attempt).last_stage, False
            if reconciliation.activated:
                return LaunchStage.FORMAL_RUN_ACTIVATED, True
            source = "reconciliation"
        try:
            self._ledger.activate(attempt, source, self._probe)
        except LedgerError as exc:
            return self._orphan(attempt, f"confirmed_orphan: {exc}"), False
        return LaunchStage.FORMAL_RUN_ACTIVATED, False

    def _orphan(self, attempt: str, reason: str) -> LaunchStage:
        """activation できない launch を uncertain に落とします。

        ledger が uncertain を受け付けない段階なら reconcile の分類に従い、
        それでも terminal にならなければ CampaignBlocked です。replacement は行いません。
        """
        try:
            self._ledger.commit_uncertain(attempt, reason)
        except LedgerError:
            self._ledger.reconcile(attempt, self._probe)
        last = self._ledger.history(attempt).last_stage
        if last not in {LaunchStage.LAUNCH_GATE_FAILED, LaunchStage.LAUNCH_UNCERTAIN}:
            raise CampaignBlocked(f"{attempt}: launch could not be concluded ({last.value})")
        return last

    def _flush_gate(self, ctx: _Execution, slot: int, attempt: str, response: Mapping[str, Any] | None) -> None:
        """ledger の launch gate event を launch gate stream へ1回だけ flush します。

        broker response と ledger row hash も一緒に残します。
        """
        if attempt in ctx.flushed_attempts:
            return
        history = self._ledger.history(attempt)
        self._artifacts.append(GATE_STREAM, [{
            "stage_execution_id": ctx.execution_id,
            "slot_id": slot,
            "attempt_id": attempt,
            "events": [event.to_wire() for event in self._ledger.campaign_events(attempt)],
            "ledger_row_hashes": list(history.row_hashes),
            "broker_response": None if response is None else _json_value(dict(response)),
        }])
        ctx.flushed_attempts.add(attempt)

    def _finish_activated(self, ctx: _Execution, slot: int, ids: SlotIdentity, *, interrupted: bool) -> EventType:
        """activation 済み run を実行し、terminal outcome と per-run manifest を記録します。

        arm → focus → manual start の checkpoint 後にだけ controller を起動し、timeout 時は
        controller terminate → helper release の external 確認を行います。process 停止確認後に
        post-run save hash を取り、per-run manifest を書いてから outcome を flush します。
        runner 中断後の再開では結果が不明なため safety failure として消費し、置き換えません。
        """
        name = f"stages/{ctx.execution_id}/runs/slot-{slot:02d}.json"
        if self._artifacts.exists(name):
            event = CampaignEvent.from_wire(self._artifacts.read_json(name)["outcome"])
            self._append(ctx, OUTCOME_STREAM, [event], run_manifest=name)
            return event.event_type
        history = self._ledger.history(ids.attempt_id)
        attestation = history.attestation
        spec = RunSpec(
            campaign_id=self.plan.campaign_id, stage_execution_id=ctx.execution_id, stage=ctx.stage, slot_id=slot,
            reserved_run_id=history.intent.reserved_run_id, gameplay_attempt_id=history.intent.gameplay_attempt_id,
            campaign_run_mode=CAMPAIGN_RUN_MODE, process_ref=attestation.identity.process_ref,
            job_ref=attestation.job_ref, target_pid=attestation.identity.pid,
            duration_seconds=STAGE_POLICIES[ctx.stage].duration_seconds, ui_restart_enabled=False,
            development_only=self.plan.development_only,
        )
        failures: list[tuple[EventType, str]] = []
        result: RunResult | None = None
        release: ReleaseObservation | None = None
        handle: Any = None
        checkpoints: list[str] = []
        if interrupted:
            failures.append((EventType.SAFETY_FAILURE, "runner_interrupted_after_activation"))
        else:
            for checkpoint in ("arm", "focus", "manual_start"):
                if self._operator.checkpoint(checkpoint, spec.to_wire()) is not True:
                    failures.append((EventType.SAFETY_FAILURE, f"{checkpoint}_checkpoint_not_confirmed"))
                    break
                checkpoints.append(checkpoint)
            else:
                try:
                    handle = self._controller.start(spec)
                    result = handle.wait(spec.duration_seconds + self.plan.grace_seconds)
                    if result is None:
                        handle.terminate()
                        failures.append((EventType.ARTIFACT_FAILURE, "controller_timeout"))
                    elif not isinstance(result, RunResult):
                        failures.append((EventType.SAFETY_FAILURE, "controller_result_invalid"))
                        result = None
                except Exception as exc:  # noqa: BLE001 - controller 障害は必ず terminal outcome にする
                    failures.append((EventType.SAFETY_FAILURE, f"controller_error: {type(exc).__name__}: {exc}"))
                    if handle is not None:
                        try:
                            handle.terminate()
                        except Exception:  # noqa: BLE001
                            pass
                if handle is not None:
                    try:
                        release = self._helper.confirm_release(getattr(handle, "helper_pid", None), spec.to_wire())
                    except Exception:  # noqa: BLE001 - observer 障害は未確認として扱う
                        release = None
                    if not isinstance(release, ReleaseObservation) or release.confirmed is not True:
                        failures.append((EventType.SAFETY_FAILURE, "helper_release_unconfirmed"))
                        release = release if isinstance(release, ReleaseObservation) else None
        if result is not None:
            failures.extend(result_failures(spec, result))
        post_hash = None
        try:
            stopped = self._target.stop(spec.to_wire()) is True
        except Exception:  # noqa: BLE001 - 停止を確認できない場合は save に触れない
            stopped = False
        if stopped:
            try:
                post_hash = self._save.post_run_hash(ids.attempt_id)
            except (SaveLifecycleError, OSError) as exc:
                failures.append((EventType.ARTIFACT_FAILURE, f"post_save_hash_failed: {exc}"))
        else:
            failures.append((EventType.ARTIFACT_FAILURE, "target_process_not_stopped"))
        pre_hash = self._save.slot_hash("pre", ids.attempt_id)
        if pre_hash is None:
            failures.append((EventType.ARTIFACT_FAILURE, "pre_save_hash_missing"))
        event = outcome_event(slot, ctx.manifest_hash, result, failures, release)
        evidence = self._save.evidence()
        history = self._ledger.history(ids.attempt_id)
        manifest = {
            "campaign_id": self.plan.campaign_id,
            "plan_hash": self.plan.plan_hash,
            "stage": ctx.stage,
            "stage_execution_id": ctx.execution_id,
            "campaign_manifest_hash": ctx.manifest_hash,
            "slot_id": slot,
            "attempt_id": ids.attempt_id,
            "reserved_run_id": ids.reserved_run_id,
            "gameplay_attempt_id": ids.gameplay_attempt_id,
            "campaign_run_mode": CAMPAIGN_RUN_MODE,
            "activation_source": history.activation_source,
            "run_spec": spec.to_wire(),
            "operator_checkpoints": checkpoints,
            "broker": {
                "process_ref": attestation.identity.process_ref,
                "job_ref": attestation.job_ref,
                "broker_ref": attestation.broker_ref,
                "ledger_row_hashes": list(history.row_hashes),
            },
            "controller_pid": getattr(handle, "controller_pid", None),
            "helper_pid": getattr(handle, "helper_pid", None),
            "release_observation": None if release is None else release.to_wire(),
            "result": None if result is None else result.to_wire(),
            "telemetry_refs": [] if result is None else list(result.telemetry_refs),
            "video_refs": [] if result is None else list(result.video_refs),
            "lease_audit_ref": None if result is None else result.lease_audit_ref,
            "save": {"pre_sha256": pre_hash, "post_sha256": post_hash, "backup_sha256": evidence["backup_sha256"],
                     "cloud_sync_attested": evidence["cloud_sync_attested"]},
            "outcome": event.to_wire(),
            "development_only": self.plan.development_only,
            "formal_campaign_eligible": not self.plan.development_only,
            "formal_evidence_eligible": (
                not self.plan.development_only and pre_hash is not None and post_hash is not None
                and evidence["backup_sha256"] is not None and evidence["cloud_sync_attested"]
            ),
        }
        self._artifacts.put_json(name, manifest)
        self._append(ctx, OUTCOME_STREAM, [event], run_manifest=name)
        return event.event_type

    def _close_execution(self, ctx: _Execution) -> StageResult:
        """06-02 reporter の report を保存し、stage 実行を terminal にします。

        report は reporter の canonical output をそのまま write-once で保存し、
        blocked な stage 実行は campaign chain stream へも flush します。
        """
        events = self._events(ctx)
        report = generate_campaign_report(
            ctx.manifest, events, event_manifest_hash=ctx.manifest_hash, blocked_campaign_ids=ctx.blocked_ids
        )
        report_wire = report.to_wire()
        self._artifacts.put_json(f"stages/{ctx.execution_id}/report.json", report_wire)
        kinds = {event.event_type for event in events}
        if EventType.LAUNCH_UNCERTAIN in kinds:
            state = RunnerState.CAMPAIGN_BLOCKED
        elif EventType.PREFLIGHT_FAILED in kinds:
            state = RunnerState.PREFLIGHT_BLOCKED
        elif EventType.LAUNCH_GATE_FAILED in kinds:
            state = RunnerState.LAUNCH_GATE_BLOCKED
        elif report.promotion_eligible:
            state = RunnerState.STAGE_PASSED
        else:
            state = RunnerState.STAGE_NOT_PROMOTED
        if state in {RunnerState.CAMPAIGN_BLOCKED, RunnerState.PREFLIGHT_BLOCKED, RunnerState.LAUNCH_GATE_BLOCKED}:
            self._append_chain_once({"kind": "blocked", "stage_execution_id": ctx.execution_id,
                                     "state": state.value, "report_hash": report.report_hash})
        self._artifacts.put_json(f"stages/{ctx.execution_id}/execution.json", {
            "plan_hash": self.plan.plan_hash,
            "stage": ctx.stage,
            "stage_execution_id": ctx.execution_id,
            "state": state.value,
            "report_hash": report.report_hash,
            "manifest_hash": ctx.manifest_hash,
            "parent_execution_id": ctx.parent_id,
            "blocked_execution_ids": list(ctx.blocked_ids),
            "threshold_parent_hash": self.plan.threshold_parent_hash,
            "thresholds_hash": canonical_hash(dict(self.plan.thresholds)),
            "development_only": self.plan.development_only,
            "formal_campaign_eligible": not self.plan.development_only,
        })
        self.state = state
        return StageResult(ctx.stage, ctx.execution_id, state, report.report_hash, self._slot_events(ctx))
