"""AgentRuntime: perception snapshot から typed AgentDecision を返す中心オーケストレータ。

DecisionScheduler で 15 Hz cadence・backlog skip・inference timeout を強制し、
snapshot age と global validity gate を満たす場合だけ movement action を返す。
画面遷移の判定は raw screen_state の重複表ではなく、assembler が型付けした
`ui_policy_input.screen_state` を正本にする。death / result / unknown gap では
combat の actor LSTM state を破棄し、前 run の記憶を次 run へ持ち越さない。
OS input には一切触れず、effect は 05-03 gameplay UI state machine が所有する。
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any, Iterable, Literal

import numpy as np

from reinbalance_survivors_contracts.canonical_json import (
    canonical_hash,
    canonical_json_bytes,
)
from reinbalance_survivors_contracts.deploy_obs import DeployObsSchema
from reinbalance_survivors_contracts.ui_intent import (
    ContractValidationError,
    UiIntentKind,
    UiIntentV1,
)
from reinbalance_survivors_contracts.ui_policy import (
    NonModelUiPolicyConfigV1,
    ScreenState,
    UiPolicyInputV1,
    decide_non_model_ui_intent,
)
from survivors.perception_snapshot import PerceptionSnapshot

from .artifact_bundle import RuntimeBundle
from .combat_session import CombatSession, StaleSnapshotError
from .decision_scheduler import DecisionScheduler, DecisionTiming
from .item_session import ItemSession, ItemSessionError

AgentDecisionKind = Literal["move", "ui", "no_op", "stop"]

AGENT_DECISION_SCHEMA_VERSION = "survivors.agent_decision.v1"

# ITEM_SELECTOR_SESSION が生成する decision の policy 固定値 (05-01 arbiter)
_ITEM_SELECTOR_POLICY_ID = "item_selector_session.v1"
_ITEM_SELECTOR_RULE_ID = "calibrated_argmax_v1"

# episode 境界とみなす raw screen_state。ここに入ると LSTM state を破棄する。
_EPISODE_BOUNDARY_SCREEN_STATES = frozenset({"death", "result", "unknown"})
# combat/UI いずれの判断もせず待機する raw screen_state。
_IDLE_SCREEN_STATES = frozenset({"paused"})

# 観測可能 field の validity 平均がこの値未満なら movement action を返さない。
GLOBAL_VALIDITY_THRESHOLD = 0.5
# age=1.0 は「その field は今 frame で観測できていない」を意味する。
# temporal_inferred は episode 先頭で必ず 1.0 になるため、個別 field の 1.0 は
# 異常ではない。観測可能 field が「すべて」1.0 の場合だけ perception 停止とみなす。
MAX_OBSERVABLE_AGE = 1.0


@dataclass(frozen=True)
class AgentDecision:
    """runtime が 1 tick で生成した typed decision。

    OS input を送信しない — effect は 05-03 gameplay UI state machine が担う。
    kind は move / ui / no_op / stop だけとし、card / fallback / button を
    action index へ詰め込まない。field 構成は plan 05-01 の契約に一致させ、
    05-03 が同じ wire を消費できるようにする。
    """

    decision_id: str
    kind: AgentDecisionKind
    action_index: int | None
    ui_intent: UiIntentV1 | None
    confidence: float
    reason: str
    source_snapshot_id: str
    source_frame_id: str
    source_content_hash: str
    snapshot_timestamp_ns: int
    inference_started_ns: int
    inference_finished_ns: int
    schema_version: str = AGENT_DECISION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """kind と省略可能 field の整合性を検証する。

        kind ごとに存在すべき / してはいけない field を強制する。
        """
        if self.kind == "move":
            if self.action_index is None:
                raise ValueError("move decision requires action_index")
            if self.ui_intent is not None:
                raise ValueError("move decision must not have ui_intent")
        elif self.kind == "ui":
            if self.ui_intent is None:
                raise ValueError("ui decision requires ui_intent")
            if self.action_index is not None:
                raise ValueError("ui decision must not have action_index")
        elif self.kind in ("no_op", "stop"):
            if self.ui_intent is not None:
                raise ValueError(f"{self.kind} must not have ui_intent")
            if self.action_index is not None:
                raise ValueError(f"{self.kind} must not have action_index")
        else:
            raise ValueError(f"unknown AgentDecision kind: {self.kind!r}")
        if not isinstance(self.decision_id, str) or not self.decision_id:
            raise ValueError("decision_id must be non-empty string")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("reason must be non-empty string")
        if isinstance(self.confidence, bool) or not isinstance(self.confidence, (int, float)):
            raise ValueError("confidence must be a real number")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be within [0, 1]")

    @property
    def inference_latency_ns(self) -> int:
        """推論に要した時間をナノ秒で返す。"""
        return self.inference_finished_ns - self.inference_started_ns

    def to_wire(self) -> dict[str, Any]:
        """canonical JSON 化できる dict 表現を返す。

        05-03 と telemetry が同じ wire を読むため、field 名と順序を固定する。
        """
        return {
            "schema_version": self.schema_version,
            "decision_id": self.decision_id,
            "kind": self.kind,
            "action_index": self.action_index,
            "ui_intent": self.ui_intent.to_wire() if self.ui_intent is not None else None,
            "confidence": float(self.confidence),
            "reason": self.reason,
            "source_snapshot_id": self.source_snapshot_id,
            "source_frame_id": self.source_frame_id,
            "source_content_hash": self.source_content_hash,
            "snapshot_timestamp_ns": self.snapshot_timestamp_ns,
            "inference_started_ns": self.inference_started_ns,
            "inference_finished_ns": self.inference_finished_ns,
        }

    def canonical_bytes(self) -> bytes:
        """canonical JSON bytes を返す。JSONL 1 行分に相当する。"""
        return canonical_json_bytes(self.to_wire())

    def decision_hash(self) -> str:
        """decision wire の SHA-256 を返す。replay 照合で使う。"""
        return canonical_hash(self.to_wire())

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> "AgentDecision":
        """canonical wire から AgentDecision を復元する。

        replay / shadow が保存済み JSONL を読み戻すために使う。
        """
        if data.get("schema_version") != AGENT_DECISION_SCHEMA_VERSION:
            raise ValueError("unsupported AgentDecision schema_version")
        intent_wire = data.get("ui_intent")
        return cls(
            decision_id=data["decision_id"],
            kind=data["kind"],
            action_index=data.get("action_index"),
            ui_intent=UiIntentV1.from_wire(intent_wire) if intent_wire is not None else None,
            confidence=data["confidence"],
            reason=data["reason"],
            source_snapshot_id=data["source_snapshot_id"],
            source_frame_id=data["source_frame_id"],
            source_content_hash=data["source_content_hash"],
            snapshot_timestamp_ns=data["snapshot_timestamp_ns"],
            inference_started_ns=data["inference_started_ns"],
            inference_finished_ns=data["inference_finished_ns"],
        )


def decisions_to_jsonl(decisions: Iterable[AgentDecision]) -> bytes:
    """AgentDecision 列を canonical JSONL bytes へ直列化する。

    1 decision = 1 行。同じ入力からは常に byte-identical な出力になる。
    """
    return b"".join(decision.canonical_bytes() + b"\n" for decision in decisions)


def _observable_mask(schema: DeployObsSchema) -> np.ndarray:
    """deploy schema のうち画面から観測できる field の bool mask を返す。

    unobservable field は契約上 validity=0 / age=1 で固定されるため、
    global validity gate の分母から除外しないと常に gate が閉じてしまう。
    """
    mask = np.zeros(schema.dim, dtype=bool)
    layout = schema.layout
    for field_spec in schema.fields:
        if field_spec.source_class == "unobservable":
            continue
        offset, size = layout[field_spec.name]
        mask[offset : offset + size] = True
    return mask


class AgentRuntime:
    """combat / item / UI policy session を一本化して AgentDecision を返す runtime。

    DecisionScheduler が cadence と締切を、validity/age gate が観測の鮮度を、
    typed screen_state が経路を決める。stale / invalid / unknown の場合は
    安全な no_op / stop を返す。OS input は生成しない。
    """

    def __init__(
        self,
        bundle: RuntimeBundle,
        *,
        scheduler: DecisionScheduler | None = None,
    ) -> None:
        """bundle から各 session と scheduler を初期化する。

        ItemSelector が None の場合は item session を skip し non-model policy へ回す。
        """
        if not isinstance(bundle, RuntimeBundle):
            raise ValueError("bundle must be RuntimeBundle")
        self._bundle = bundle
        self._deploy_schema = bundle.deploy_schema
        self._observable_mask = _observable_mask(bundle.deploy_schema)
        self._combat_session = CombatSession(bundle.combat_policy)
        self._item_session: ItemSession | None = (
            ItemSession(bundle.item_selector) if bundle.item_selector is not None else None
        )
        self._ui_policy_config: NonModelUiPolicyConfigV1 = bundle.ui_policy_config
        self._scheduler = scheduler or DecisionScheduler()
        self._last_screen_state: str | None = None
        self._last_timing: DecisionTiming | None = None

    @property
    def scheduler(self) -> DecisionScheduler:
        """統合済み DecisionScheduler を返す。telemetry と test が参照する。"""
        return self._scheduler

    @property
    def combat_session(self) -> CombatSession:
        """combat session を返す。LSTM state の検査に使う。"""
        return self._combat_session

    @property
    def last_timing(self) -> DecisionTiming | None:
        """直近 decision のタイミングメタデータを返す。"""
        return self._last_timing

    def reset_episode(self) -> None:
        """エピソード開始時に combat LSTM 状態と cadence をリセットする。

        新 run / death / result / unknown gap 後に呼ぶ。runtime 内部でも
        画面遷移を検出して自動的に呼ばれる。
        """
        self._combat_session.reset_episode()
        self._scheduler.reset()
        self._last_screen_state = None

    def decide(
        self,
        snapshot: PerceptionSnapshot,
        *,
        now_ns: int | None = None,
        episode_start: bool = False,
    ) -> AgentDecision:
        """1 tick の decision を返す。

        cadence 外の呼び出し、stale / invalid / unknown snapshot では安全な
        no_op / stop を返す。OS input には触れない。
        now_ns は capture 層と同じ単調高分解能時計 (`time.perf_counter_ns`) の値。
        省略時は runtime が同じ時計から取得する。Windows の `time.monotonic_ns` は
        分解能が約 15.6 ms しかなく inference timeout を判定できないため使わない。
        """
        started_ns = time.perf_counter_ns()
        current_ns = int(now_ns) if now_ns is not None else started_ns

        if not isinstance(snapshot, PerceptionSnapshot):
            # 型不明の入力は episode 継続の保証がないため記憶を破棄して停止する。
            self._combat_session.reset_episode()
            self._last_screen_state = None
            return AgentDecision(
                decision_id=str(uuid.uuid4()),
                kind="stop",
                action_index=None,
                ui_intent=None,
                confidence=0.0,
                reason="unknown snapshot type",
                source_snapshot_id="unknown",
                source_frame_id="unknown",
                source_content_hash="unknown",
                snapshot_timestamp_ns=0,
                inference_started_ns=started_ns,
                inference_finished_ns=time.perf_counter_ns(),
            )

        if episode_start:
            self.reset_episode()

        # 15 Hz cadence の外では推論せず、backlog では最新 snapshot だけを使う。
        if not self._scheduler.should_decide(current_ns):
            return self._safety(
                snapshot, "no_op", "off-cadence tick skipped by scheduler", started_ns
            )
        scheduled_ns = self._scheduler.advance(current_ns)

        raw_state = snapshot.screen_state
        # 画面遷移で episode が切れる場合は、判断より先に LSTM state を破棄する。
        if raw_state in _EPISODE_BOUNDARY_SCREEN_STATES:
            self._combat_session.reset_episode()
            self._last_screen_state = raw_state
            return self._safety(
                snapshot,
                "no_op",
                f"episode boundary screen_state={raw_state!r}; recurrent state reset",
                started_ns,
                scheduled_ns=scheduled_ns,
            )
        if raw_state in _IDLE_SCREEN_STATES:
            self._last_screen_state = raw_state
            return self._safety(
                snapshot, "no_op", f"idle screen_state={raw_state!r}", started_ns,
                scheduled_ns=scheduled_ns,
            )

        policy_input = snapshot.ui_policy_input
        if policy_input is None or not isinstance(policy_input, UiPolicyInputV1):
            # 型付き screen_state がない snapshot は経路を決められない。
            self._combat_session.reset_episode()
            self._last_screen_state = raw_state
            return self._safety(
                snapshot, "no_op", "ui_policy_input is missing; cannot route decision",
                started_ns, scheduled_ns=scheduled_ns,
            )

        screen_state = policy_input.screen_state
        if screen_state == ScreenState.UNKNOWN:
            self._combat_session.reset_episode()
            self._last_screen_state = raw_state
            return self._safety(
                snapshot, "no_op", "unknown screen state; recurrent state reset",
                started_ns, scheduled_ns=scheduled_ns,
            )
        self._last_screen_state = raw_state

        if screen_state == ScreenState.GAMEPLAY:
            return self._decide_combat(
                snapshot, started_ns=started_ns, now_ns=current_ns, scheduled_ns=scheduled_ns
            )
        if screen_state == ScreenState.LEVEL_UP:
            return self._decide_item(
                snapshot, policy_input, started_ns=started_ns, scheduled_ns=scheduled_ns
            )
        return self._decide_non_model_ui(
            snapshot, policy_input, started_ns=started_ns, scheduled_ns=scheduled_ns
        )

    def _record_timing(
        self, decision: AgentDecision, scheduled_ns: int | None
    ) -> AgentDecision:
        """decision の timing metadata を記録して返す。"""
        self._last_timing = DecisionTiming(
            decision_id=decision.decision_id,
            scheduled_ns=scheduled_ns if scheduled_ns is not None else decision.inference_started_ns,
            inference_started_ns=decision.inference_started_ns,
            inference_finished_ns=decision.inference_finished_ns,
        )
        return decision

    def _safety(
        self,
        snapshot: PerceptionSnapshot,
        kind: AgentDecisionKind,
        reason: str,
        started_ns: int,
        *,
        scheduled_ns: int | None = None,
    ) -> AgentDecision:
        """no_op / stop の safety decision を組み立てる。"""
        decision = AgentDecision(
            decision_id=str(uuid.uuid4()),
            kind=kind,
            action_index=None,
            ui_intent=None,
            confidence=0.0,
            reason=reason,
            source_snapshot_id=snapshot.snapshot_id,
            source_frame_id=snapshot.frame_id,
            source_content_hash=snapshot.source_content_hash,
            snapshot_timestamp_ns=snapshot.captured_ns,
            inference_started_ns=started_ns,
            inference_finished_ns=time.perf_counter_ns(),
        )
        return self._record_timing(decision, scheduled_ns)

    def _ui_decision(
        self,
        snapshot: PerceptionSnapshot,
        intent: UiIntentV1,
        confidence: float,
        reason: str,
        started_ns: int,
        scheduled_ns: int | None,
    ) -> AgentDecision:
        """UiIntentV1 を持つ ui decision を組み立てる。"""
        decision = AgentDecision(
            decision_id=str(uuid.uuid4()),
            kind="ui",
            action_index=None,
            ui_intent=intent,
            confidence=confidence,
            reason=reason,
            source_snapshot_id=snapshot.snapshot_id,
            source_frame_id=snapshot.frame_id,
            source_content_hash=snapshot.source_content_hash,
            snapshot_timestamp_ns=snapshot.captured_ns,
            inference_started_ns=started_ns,
            inference_finished_ns=time.perf_counter_ns(),
        )
        return self._record_timing(decision, scheduled_ns)

    def _global_validity_gate(self, snapshot: PerceptionSnapshot) -> str | None:
        """観測の global validity / age gate を評価し、不合格なら理由を返す。

        合格なら None。movement action はこの gate を通った場合だけ返す。
        """
        observation = snapshot.deploy_obs
        validity = np.asarray(observation.validity, dtype=np.float64)
        age = np.asarray(observation.age, dtype=np.float64)
        mask = self._observable_mask
        if validity.shape != mask.shape or age.shape != mask.shape:
            return "deploy_obs planes do not match the bundle deploy schema"
        observable_validity = validity[mask]
        observable_age = age[mask]
        if observable_validity.size == 0:
            return "deploy schema declares no observable field"
        mean_validity = float(observable_validity.mean())
        if mean_validity < GLOBAL_VALIDITY_THRESHOLD:
            return (
                f"global validity {mean_validity:.3f} below "
                f"{GLOBAL_VALIDITY_THRESHOLD:.3f}"
            )
        if float(observable_age.min()) >= MAX_OBSERVABLE_AGE:
            return "all observable fields are fully stale (age=1.0)"
        return None

    def _decide_combat(
        self,
        snapshot: PerceptionSnapshot,
        *,
        started_ns: int,
        now_ns: int,
        scheduled_ns: int,
    ) -> AgentDecision:
        """combat observation から move decision を生成する。

        snapshot age / global validity / inference timeout のいずれかが
        未達なら movement action を返さず no_op にする。
        """
        if not self._scheduler.is_snapshot_fresh(snapshot.captured_ns, now_ns):
            age_ns = self._scheduler.snapshot_age_ns(snapshot.captured_ns, now_ns)
            return self._safety(
                snapshot,
                "no_op",
                f"snapshot age gate failed (captured_ns={snapshot.captured_ns}, age_ns={age_ns})",
                started_ns,
                scheduled_ns=scheduled_ns,
            )

        validity_reason = self._global_validity_gate(snapshot)
        if validity_reason is not None:
            return self._safety(
                snapshot, "no_op", f"combat validity gate failed: {validity_reason}",
                started_ns, scheduled_ns=scheduled_ns,
            )

        try:
            obs_array = snapshot.deploy_obs.as_policy_tensor(self._deploy_schema)
        except Exception as exc:  # noqa: BLE001
            return self._safety(
                snapshot, "no_op", f"deploy_obs schema mismatch or invalid: {exc}",
                started_ns, scheduled_ns=scheduled_ns,
            )

        try:
            combat_decision = self._combat_session.decide(obs_array)
        except StaleSnapshotError as exc:
            return self._safety(
                snapshot, "no_op", f"stale combat obs: {exc}", started_ns,
                scheduled_ns=scheduled_ns,
            )
        except Exception as exc:  # noqa: BLE001  # 想定外の model error → stop
            return self._safety(
                snapshot, "stop", f"combat inference error: {exc}", started_ns,
                scheduled_ns=scheduled_ns,
            )

        finished_ns = time.perf_counter_ns()
        if self._scheduler.exceeded_inference_timeout(started_ns, finished_ns):
            # 締切超過の move は古い画面への入力になるため破棄する。
            return self._safety(
                snapshot,
                "no_op",
                f"combat inference exceeded timeout "
                f"({(finished_ns - started_ns) / 1_000_000.0:.2f} ms)",
                started_ns,
                scheduled_ns=scheduled_ns,
            )

        decision = AgentDecision(
            decision_id=str(uuid.uuid4()),
            kind="move",
            action_index=combat_decision.action_index,
            ui_intent=None,
            confidence=combat_decision.confidence,
            reason="combat recurrent policy move",
            source_snapshot_id=snapshot.snapshot_id,
            source_frame_id=snapshot.frame_id,
            source_content_hash=snapshot.source_content_hash,
            snapshot_timestamp_ns=snapshot.captured_ns,
            inference_started_ns=started_ns,
            inference_finished_ns=finished_ns,
        )
        return self._record_timing(decision, scheduled_ns)

    def _decide_item(
        self,
        snapshot: PerceptionSnapshot,
        policy_input: UiPolicyInputV1,
        *,
        started_ns: int,
        scheduled_ns: int,
    ) -> AgentDecision:
        """ItemSelector で level-up 候補を採点して CHOOSE_CARD を返す。

        ItemSession が None または item_context が None → non-model UI policy へ。
        confidence gate 未達 → no_op。target 解決失敗 → stop。
        """
        if self._item_session is None or snapshot.item_context is None:
            return self._decide_non_model_ui(
                snapshot, policy_input, started_ns=started_ns, scheduled_ns=scheduled_ns
            )

        try:
            outcome = self._item_session.decide(
                snapshot.item_context,
                snapshot.ui_presentation,
                decision_policy_id=_ITEM_SELECTOR_POLICY_ID,
                decision_rule_id=_ITEM_SELECTOR_RULE_ID,
                decision_config_hash=self._bundle.deploy_schema_hash,
            )
        except ItemSessionError as exc:
            return self._safety(
                snapshot, "stop", f"item session error: {exc}", started_ns,
                scheduled_ns=scheduled_ns,
            )

        if outcome.intent is None:
            return self._safety(
                snapshot, "no_op", outcome.reason, started_ns, scheduled_ns=scheduled_ns
            )
        return self._ui_decision(
            snapshot, outcome.intent, outcome.confidence, outcome.reason,
            started_ns, scheduled_ns,
        )

    def _decide_non_model_ui(
        self,
        snapshot: PerceptionSnapshot,
        policy_input: UiPolicyInputV1,
        *,
        started_ns: int,
        scheduled_ns: int,
    ) -> AgentDecision:
        """shared non-model UI policy (02-04 / 05-01 共有) から UiIntentV1 を生成する。

        policy が None を返す → no_op。stop intent → stop decision。
        05-03 の関数は呼ばない — intent の所有者は本 runtime 側にある。
        """
        try:
            intent = decide_non_model_ui_intent(policy_input, self._ui_policy_config)
        except ContractValidationError as exc:
            return self._safety(
                snapshot, "stop", f"non-model UI policy error: {exc}", started_ns,
                scheduled_ns=scheduled_ns,
            )

        if intent is None:
            return self._safety(
                snapshot, "no_op", "non-model UI policy returned no intent", started_ns,
                scheduled_ns=scheduled_ns,
            )
        if intent.kind == UiIntentKind.STOP:
            return self._safety(
                snapshot,
                "stop",
                f"non-model UI policy returned stop: {intent.semantic_action}",
                started_ns,
                scheduled_ns=scheduled_ns,
            )
        return self._ui_decision(
            snapshot, intent, 1.0, f"non-model UI policy {intent.kind.value}",
            started_ns, scheduled_ns,
        )
