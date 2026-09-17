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
from typing import Any, Callable, Iterable, Literal

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
)
from survivors.perception_snapshot import PerceptionSnapshot
from survivors.real_obs_assembler import _HUD_TO_SCREEN_STATE

from .artifact_bundle import REQUIRED_ACTION_DIM, RuntimeBundle
from .combat_session import CombatSession, StaleSnapshotError
from .decision_scheduler import DecisionScheduler, DecisionTiming, ScreenDecisionScheduler
from .item_session import ItemSession, ItemSessionError, validate_ui_capability_owners

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
            if type(self.action_index) is not int or not 0 <= self.action_index < REQUIRED_ACTION_DIM:
                raise ValueError(
                    f"move decision action_index must be an int within [0, {REQUIRED_ACTION_DIM})"
                )
            if self.ui_intent is not None:
                raise ValueError("move decision must not have ui_intent")
        elif self.kind == "ui":
            if self.ui_intent is None:
                raise ValueError("ui decision requires ui_intent")
            if self.action_index is not None:
                raise ValueError("ui decision must not have action_index")
            if (
                self.ui_intent.source_snapshot_hash != self.source_snapshot_id
                or self.ui_intent.source_frame_hash != self.source_frame_id
                or self.ui_intent.source_content_hash != self.source_content_hash
            ):
                raise ValueError("ui_intent source binding must match AgentDecision")
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
        if self.inference_finished_ns < self.inference_started_ns:
            raise ValueError("inference timestamps must be ordered")

    @property
    def inference_latency_ns(self) -> int:
        """推論に要した時間をナノ秒で返す。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
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
        """canonical JSON bytes を返す。JSONL 1 行分に相当する。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        return canonical_json_bytes(self.to_wire())

    def decision_hash(self) -> str:
        """decision wire の SHA-256 を返す。replay 照合で使う。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
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
        screen_scheduler: ScreenDecisionScheduler | None = None,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
        require_live: bool = False,
    ) -> None:
        """bundle から各 session と scheduler を初期化する。

        ItemSelector が None の場合は item session を skip し non-model policy へ回す。
        """
        if not isinstance(bundle, RuntimeBundle):
            raise ValueError("bundle must be RuntimeBundle")
        if not callable(clock_ns):
            raise ValueError("clock_ns must be callable")
        if require_live:
            bundle.assert_live_eligible()
        validate_ui_capability_owners(bundle.ui_policy_config)
        self._bundle = bundle
        self._deploy_schema = bundle.deploy_schema
        self._observable_mask = _observable_mask(bundle.deploy_schema)
        self._combat_session = CombatSession(
            bundle.combat_policy,
            action_semantics=bundle.action_semantics.actions,
        )
        self._item_session: ItemSession | None = (
            ItemSession(bundle.item_selector) if bundle.item_selector is not None else None
        )
        self._ui_policy_config: NonModelUiPolicyConfigV1 = bundle.ui_policy_config
        self._scheduler = scheduler or DecisionScheduler()
        self._screen_scheduler = screen_scheduler or ScreenDecisionScheduler()
        self._clock_ns = clock_ns
        self._last_screen_state: str | None = None
        self._last_snapshot_timestamp_ns: int | None = None
        self._last_timing: DecisionTiming | None = None

    @property
    def scheduler(self) -> DecisionScheduler:
        """統合済み DecisionScheduler を返す。telemetry と test が参照する。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        return self._scheduler

    @property
    def screen_scheduler(self) -> ScreenDecisionScheduler:
        """capture frame ごとの UI scheduler を返す。

        やさしい説明: combat decision count と独立した 30 Hz 評価数を確認できます。
        """
        return self._screen_scheduler

    @property
    def combat_session(self) -> CombatSession:
        """combat session を返す。LSTM state の検査に使う。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        return self._combat_session

    @property
    def last_timing(self) -> DecisionTiming | None:
        """直近 decision のタイミングメタデータを返す。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        return self._last_timing

    def reset_episode(self) -> None:
        """エピソード開始時に combat LSTM 状態と cadence をリセットする。

        新 run / death / result / unknown gap 後に呼ぶ。runtime 内部でも
        画面遷移を検出して自動的に呼ばれる。
        """
        self._combat_session.reset_episode("runtime_reset")
        self._scheduler.reset()
        self._screen_scheduler.reset()
        self._last_screen_state = None
        self._last_snapshot_timestamp_ns = None

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
        started_ns = self._clock_ns()
        current_ns = int(now_ns) if now_ns is not None else started_ns

        if not isinstance(snapshot, PerceptionSnapshot):
            # 型不明の入力は episode 継続の保証がないため記憶を破棄して停止する。
            self._combat_session.reset_episode("unknown_snapshot_type")
            self._screen_scheduler.reset()
            self._last_screen_state = None
            self._last_snapshot_timestamp_ns = None
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
                inference_finished_ns=self._clock_ns(),
            )

        raw_state = snapshot.screen_state
        if (
            not episode_start
            and self._last_snapshot_timestamp_ns is not None
            and snapshot.captured_ns <= self._last_snapshot_timestamp_ns
        ):
            return self._safety(
                snapshot,
                "no_op",
                "stale snapshot gate failed",
                started_ns,
                scheduled_ns=current_ns,
            )
        if not self._scheduler.is_snapshot_fresh(snapshot.captured_ns, current_ns):
            age_ns = self._scheduler.snapshot_age_ns(snapshot.captured_ns, current_ns)
            return self._safety(
                snapshot,
                "no_op",
                f"snapshot age gate failed (captured_ns={snapshot.captured_ns}, age_ns={age_ns})",
                started_ns,
                scheduled_ns=current_ns,
            )

        policy_input = snapshot.ui_policy_input
        binding_error = self._ui_policy_binding_error(snapshot, policy_input)
        if binding_error is not None:
            return self._safety(
                snapshot,
                "stop",
                f"ui_policy_input binding gate failed: {binding_error}",
                started_ns,
                scheduled_ns=current_ns,
            )

        # episode_start のreset は checkpoint取得より先に確定させる。
        # 先にcheckpointを取ると、reset後にtimeout/stopした場合の rollback が
        # reset前(前episode)のLSTM stateとepisode_start=Falseを復元してしまう。
        if episode_start:
            self.reset_episode()

        checkpoint = (
            self._combat_session.recurrent_state_copy(),
            self._combat_session.episode_start_pending,
            self._last_snapshot_timestamp_ns,
            self._last_screen_state,
        )

        screen_intent: UiIntentV1 | None = None
        if isinstance(policy_input, UiPolicyInputV1):
            try:
                screen_intent = self._screen_scheduler.evaluate(
                    policy_input, self._ui_policy_config
                )
            except ContractValidationError as exc:
                self._restore_runtime_checkpoint(checkpoint)
                return self._safety(
                    snapshot,
                    "stop",
                    f"screen UI policy error: {exc}",
                    started_ns,
                    scheduled_ns=current_ns,
                )

        # 画面遷移で episode が切れる場合は、判断より先に LSTM state を破棄する。
        if raw_state in _EPISODE_BOUNDARY_SCREEN_STATES:
            self._combat_session.reset_episode(raw_state)
            self._last_screen_state = raw_state
            return self._accept_snapshot(
                snapshot,
                self._safety(
                    snapshot,
                    "no_op",
                    f"episode boundary screen_state={raw_state!r}; recurrent state reset",
                    started_ns,
                    scheduled_ns=current_ns,
                ),
            )
        if raw_state in _IDLE_SCREEN_STATES:
            self._last_screen_state = raw_state
            return self._accept_snapshot(
                snapshot,
                self._safety(
                    snapshot, "no_op", f"idle screen_state={raw_state!r}", started_ns,
                    scheduled_ns=current_ns,
                ),
            )

        if policy_input is None or not isinstance(policy_input, UiPolicyInputV1):
            # 型付き screen_state がない snapshot は経路を決められない。
            self._combat_session.reset_episode("missing_ui_policy_input")
            self._last_screen_state = raw_state
            return self._accept_snapshot(
                snapshot,
                self._safety(
                    snapshot, "no_op", "ui_policy_input is missing; cannot route decision",
                    started_ns, scheduled_ns=current_ns,
                ),
            )

        screen_state = policy_input.screen_state
        if screen_state == ScreenState.UNKNOWN:
            self._combat_session.reset_episode("unknown_screen_state")
            self._last_screen_state = raw_state
            return self._accept_snapshot(
                snapshot,
                self._safety(
                    snapshot, "no_op", "unknown screen state; recurrent state reset",
                    started_ns, scheduled_ns=current_ns,
                ),
            )
        self._last_screen_state = raw_state

        if screen_state == ScreenState.GAMEPLAY:
            # 15 Hz deadline は combat 推論だけに適用する。
            if not self._scheduler.should_decide(current_ns):
                return self._accept_snapshot(
                    snapshot,
                    self._safety(
                        snapshot,
                        "no_op",
                        "off-cadence tick skipped by scheduler",
                        started_ns,
                        scheduled_ns=current_ns,
                    ),
                )
            scheduled_ns = self._scheduler.advance(current_ns)
            decision = self._decide_combat(
                snapshot, started_ns=started_ns, scheduled_ns=scheduled_ns
            )
        elif screen_state == ScreenState.LEVEL_UP:
            scheduled_ns = current_ns
            decision = self._decide_item(
                snapshot,
                policy_input,
                screen_intent,
                started_ns=started_ns,
                scheduled_ns=scheduled_ns,
            )
        else:
            scheduled_ns = current_ns
            decision = self._decide_non_model_ui(
                snapshot,
                screen_intent,
                started_ns=started_ns,
                scheduled_ns=scheduled_ns,
            )
        return self._enforce_inference_timeout(
            snapshot,
            decision,
            started_ns=started_ns,
            scheduled_ns=scheduled_ns,
            checkpoint=checkpoint,
        )

    def _ui_policy_binding_error(
        self,
        snapshot: PerceptionSnapshot,
        policy_input: UiPolicyInputV1 | None,
    ) -> str | None:
        """typed UI 入力が外側 snapshot と同じ frame に束縛されるか調べる。

        やさしい説明: 別 frame の hash や screen state で経路を選ぶ前に不一致を止めます。
        """
        if policy_input is None:
            return None
        expected_screen_state = _HUD_TO_SCREEN_STATE.get(
            snapshot.screen_state, ScreenState.UNKNOWN
        )
        bindings = (
            ("source_snapshot_hash", policy_input.source_snapshot_hash, snapshot.snapshot_id),
            ("source_frame_hash", policy_input.source_frame_hash, snapshot.frame_id),
            (
                "source_content_hash",
                policy_input.source_content_hash,
                snapshot.source_content_hash,
            ),
            ("ui_state_key", policy_input.ui_state_key, snapshot.ui_state_key),
        )
        for name, actual, expected in bindings:
            if actual != expected:
                return f"{name} does not match PerceptionSnapshot"
        # 生成側(real_obs_assembler)は screen_state_confidence が閾値未満だと
        # 既知HUD名でも UNKNOWN に落とす。raw名からの期待値だけで比較すると
        # その正規の低confidence UNKNOWNを不一致として誤検知するため、
        # UNKNOWNは安全側の縮退として個別に許容する(他の値の不一致は従来通り拒否)。
        if (
            policy_input.screen_state != expected_screen_state
            and policy_input.screen_state != ScreenState.UNKNOWN
        ):
            return "screen_state does not match PerceptionSnapshot"
        return None

    def _accept_snapshot(
        self, snapshot: PerceptionSnapshot, decision: AgentDecision
    ) -> AgentDecision:
        """全 gate を通過した snapshot の replay cursor を確定する。

        やさしい説明: 拒否した未来 frame や timeout が次の正常 frame を stale にしないようにします。
        """
        self._last_snapshot_timestamp_ns = snapshot.captured_ns
        return decision

    def _restore_runtime_checkpoint(
        self,
        checkpoint: tuple[np.ndarray | None, bool, int | None, str | None],
    ) -> None:
        """拒否した推論より前の recurrent/replay state を復元する。

        やさしい説明: action を破棄するときは、その action を生んだ内部記憶も一緒に戻します。
        """
        recurrent_state, episode_start, snapshot_timestamp_ns, screen_state = checkpoint
        self._combat_session.restore_recurrent_state(
            recurrent_state, episode_start=episode_start
        )
        self._last_snapshot_timestamp_ns = snapshot_timestamp_ns
        self._last_screen_state = screen_state

    def _record_timing(
        self, decision: AgentDecision, scheduled_ns: int | None
    ) -> AgentDecision:
        """decision の timing metadata を記録して返す。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
        self._last_timing = DecisionTiming(
            decision_id=decision.decision_id,
            scheduled_ns=scheduled_ns if scheduled_ns is not None else decision.inference_started_ns,
            inference_started_ns=decision.inference_started_ns,
            inference_finished_ns=decision.inference_finished_ns,
        )
        return decision

    def _enforce_inference_timeout(
        self,
        snapshot: PerceptionSnapshot,
        decision: AgentDecision,
        *,
        started_ns: int,
        scheduled_ns: int,
        checkpoint: tuple[np.ndarray | None, bool, int | None, str | None],
    ) -> AgentDecision:
        """全 decision 経路へ同じ inference timeout gate を適用する。

        やさしい説明: combat/item/non-model UI のどれでも遅れた結果を古い画面へ返しません。
        """
        if (
            decision.kind != "stop"
            and self._scheduler.exceeded_inference_timeout(
                started_ns, decision.inference_finished_ns
            )
        ):
            self._restore_runtime_checkpoint(checkpoint)
            return self._safety(
                snapshot,
                "no_op",
                "inference timeout gate failed",
                started_ns,
                scheduled_ns=scheduled_ns,
            )
        if decision.kind == "stop":
            self._restore_runtime_checkpoint(checkpoint)
            return decision
        return self._accept_snapshot(snapshot, decision)

    def _safety(
        self,
        snapshot: PerceptionSnapshot,
        kind: AgentDecisionKind,
        reason: str,
        started_ns: int,
        *,
        scheduled_ns: int | None = None,
    ) -> AgentDecision:
        """no_op / stop の safety decision を組み立てる。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
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
            inference_finished_ns=self._clock_ns(),
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
        """UiIntentV1 を持つ ui decision を組み立てる。

        やさしい説明: 呼び出し側が期待する入出力と安全条件を明示します。
        """
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
            inference_finished_ns=self._clock_ns(),
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
        scheduled_ns: int,
    ) -> AgentDecision:
        """combat observation から move decision を生成する。

        central freshness gate 通過後も global validity / tensor 契約が未達なら
        movement action を返さず no_op にする。
        """
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

        finished_ns = self._clock_ns()

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
        screen_intent: UiIntentV1 | None,
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
                snapshot, screen_intent, started_ns=started_ns, scheduled_ns=scheduled_ns
            )

        try:
            outcome = self._item_session.decide(
                snapshot.item_context,
                snapshot.ui_presentation,
                decision_policy_id=_ITEM_SELECTOR_POLICY_ID,
                decision_rule_id=_ITEM_SELECTOR_RULE_ID,
                decision_config_hash=self._bundle.deploy_schema_hash,
                fallback_input=policy_input,
                ui_policy_config=self._ui_policy_config,
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
        if outcome.intent.kind == UiIntentKind.STOP:
            return self._safety(
                snapshot,
                "stop",
                f"item fallback returned stop: {outcome.intent.semantic_action}",
                started_ns,
                scheduled_ns=scheduled_ns,
            )
        return self._ui_decision(
            snapshot, outcome.intent, outcome.confidence, outcome.reason,
            started_ns, scheduled_ns,
        )

    def _decide_non_model_ui(
        self,
        snapshot: PerceptionSnapshot,
        intent: UiIntentV1 | None,
        *,
        started_ns: int,
        scheduled_ns: int,
    ) -> AgentDecision:
        """shared non-model UI policy (02-04 / 05-01 共有) から UiIntentV1 を生成する。

        policy が None を返す → no_op。stop intent → stop decision。
        05-03 の関数は呼ばない — intent の所有者は本 runtime 側にある。
        """
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
