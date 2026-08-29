"""AgentRuntime: perception snapshot から typed AgentDecision を返す中心オーケストレータ。

combat / item / UI policy の各 session を screen_state に応じて振り分け、
stale / invalid / unknown snapshot には no_op / stop を返す。
OS input には一切触れない。
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from reinbalance_survivors_contracts.ui_intent import (
    ContractValidationError,
    DecisionOwner,
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

from .artifact_bundle import RuntimeBundle, BundleLoadError
from .combat_session import CombatSession, StaleSnapshotError
from .item_session import ItemSession, ItemSessionError


# combat が有効な screen_state の集合 — gameplay だけで動く
_COMBAT_SCREEN_STATES = frozenset({"gameplay"})
# ItemSelector が必要な screen_state の集合
_ITEM_SCREEN_STATES = frozenset({"level_up_items"})
# 非モデル UI policy が処理する screen_state の集合
_NON_MODEL_UI_SCREEN_STATES = frozenset({"level_up_fallback", "chest", "confirm"})

AgentDecisionKind = Literal["move", "ui", "no_op", "stop"]

# ITEM_SELECTOR_SESSION が生成する decision の policy 固定値 (05-01 arbiter)
_ITEM_SELECTOR_POLICY_ID = "item_selector_session.v1"
_ITEM_SELECTOR_RULE_ID = "argmax_v1"


@dataclass(frozen=True)
class AgentDecision:
    """runtime が 1 tick で生成した typed decision。

    OS input を送信しない — effect は 05-03 gameplay UI state machine が担う。
    kind は move / ui / no_op / stop だけとし、card / fallback / button を
    choice index へ詰め込まない。
    """

    kind: AgentDecisionKind
    decision_id: str
    ui_intent: UiIntentV1 | None
    move_action_index: int | None
    no_op_reason: str | None
    source_snapshot_id: str
    source_frame_id: str
    source_content_hash: str
    snapshot_timestamp_ns: int
    inference_started_ns: int
    inference_finished_ns: int

    def __post_init__(self) -> None:
        """kind と省略可能 field の整合性を検証する。

        kind ごとに存在すべき / してはいけない field を強制する。
        """
        if self.kind == "move":
            if self.move_action_index is None:
                raise ValueError("move decision requires move_action_index")
            if self.ui_intent is not None:
                raise ValueError("move decision must not have ui_intent")
        elif self.kind == "ui":
            if self.ui_intent is None:
                raise ValueError("ui decision requires ui_intent")
            if self.move_action_index is not None:
                raise ValueError("ui decision must not have move_action_index")
        elif self.kind in ("no_op", "stop"):
            if self.ui_intent is not None:
                raise ValueError(f"{self.kind} must not have ui_intent")
            if self.move_action_index is not None:
                raise ValueError(f"{self.kind} must not have move_action_index")
        else:
            raise ValueError(f"unknown AgentDecision kind: {self.kind!r}")
        if not isinstance(self.decision_id, str) or not self.decision_id:
            raise ValueError("decision_id must be non-empty string")


def _make_no_op(
    snapshot: PerceptionSnapshot, reason: str, started_ns: int, finished_ns: int
) -> AgentDecision:
    """no_op AgentDecision を構築するヘルパー。"""
    return AgentDecision(
        kind="no_op",
        decision_id=str(uuid.uuid4()),
        ui_intent=None,
        move_action_index=None,
        no_op_reason=reason,
        source_snapshot_id=snapshot.snapshot_id,
        source_frame_id=snapshot.frame_id,
        source_content_hash=snapshot.source_content_hash,
        snapshot_timestamp_ns=snapshot.captured_ns,
        inference_started_ns=started_ns,
        inference_finished_ns=finished_ns,
    )


def _make_stop(
    snapshot: PerceptionSnapshot, reason: str, started_ns: int, finished_ns: int
) -> AgentDecision:
    """stop AgentDecision を構築するヘルパー。"""
    return AgentDecision(
        kind="stop",
        decision_id=str(uuid.uuid4()),
        ui_intent=None,
        move_action_index=None,
        no_op_reason=reason,
        source_snapshot_id=snapshot.snapshot_id,
        source_frame_id=snapshot.frame_id,
        source_content_hash=snapshot.source_content_hash,
        snapshot_timestamp_ns=snapshot.captured_ns,
        inference_started_ns=started_ns,
        inference_finished_ns=finished_ns,
    )


def _snap_to_policy_input(snap: PerceptionSnapshot) -> UiPolicyInputV1 | None:
    """PerceptionSnapshot.ui_policy_input を返す。

    None の場合は caller が no_op を生成する。
    """
    return snap.ui_policy_input


class AgentRuntime:
    """combat / item / UI policy session を一本化して AgentDecision を返す runtime。

    screen_state に応じてセッションを切り替え、stale / invalid / unknown の場合は
    安全な no_op / stop を返す。OS input は生成しない。
    """

    def __init__(self, bundle: RuntimeBundle) -> None:
        """bundle から各 session を初期化する。

        ItemSelector が None の場合は item session を skip し no_op を返す。
        """
        if not isinstance(bundle, RuntimeBundle):
            raise ValueError("bundle must be RuntimeBundle")
        self._bundle = bundle
        self._deploy_schema = bundle.deploy_schema
        self._combat_session = CombatSession(bundle.combat_model)
        self._item_session: ItemSession | None = (
            ItemSession(bundle.item_selector)
            if bundle.item_selector is not None
            else None
        )
        self._ui_policy_config: NonModelUiPolicyConfigV1 = bundle.ui_policy_config
        self._episode_started = False

    def reset_episode(self) -> None:
        """エピソード開始時に combat GRU 状態をリセットする。

        新 run / death / result 画面後に必ず呼ぶ。
        """
        self._combat_session.reset_episode()
        self._episode_started = False

    def decide(
        self,
        snapshot: PerceptionSnapshot,
        *,
        episode_start: bool = False,
    ) -> AgentDecision:
        """1 tick の decision を返す。

        stale / invalid / unknown snapshot では安全な no_op / stop を返す。
        OS input には触れない。
        """
        started_ns = time.monotonic_ns()

        if not isinstance(snapshot, PerceptionSnapshot):
            finished_ns = time.monotonic_ns()
            # unknown snapshot — stop
            return AgentDecision(
                kind="stop",
                decision_id=str(uuid.uuid4()),
                ui_intent=None,
                move_action_index=None,
                no_op_reason="unknown snapshot type",
                source_snapshot_id="unknown",
                source_frame_id="unknown",
                source_content_hash="unknown",
                snapshot_timestamp_ns=0,
                inference_started_ns=started_ns,
                inference_finished_ns=finished_ns,
            )

        screen_state = snapshot.screen_state

        # gameplay → combat session
        if screen_state in _COMBAT_SCREEN_STATES:
            return self._decide_combat(snapshot, episode_start=episode_start, started_ns=started_ns)

        # level_up_items → ItemSelector
        if screen_state in _ITEM_SCREEN_STATES:
            return self._decide_item(snapshot, started_ns=started_ns)

        # fallback / chest / confirm → shared non-model UI policy
        if screen_state in _NON_MODEL_UI_SCREEN_STATES:
            return self._decide_non_model_ui(snapshot, started_ns=started_ns)

        # unknown / paused / death / result → no_op
        finished_ns = time.monotonic_ns()
        return _make_no_op(snapshot, f"unhandled screen_state={screen_state!r}", started_ns, finished_ns)

    def _decide_combat(
        self,
        snapshot: PerceptionSnapshot,
        *,
        episode_start: bool,
        started_ns: int,
    ) -> AgentDecision:
        """combat observation から move decision を生成する。

        invalid/stale obs → no_op。非有限 logit → stop。
        """
        try:
            obs_array = snapshot.deploy_obs.as_policy_tensor(self._deploy_schema)
        except Exception as exc:  # noqa: BLE001
            finished_ns = time.monotonic_ns()
            return _make_no_op(
                snapshot, f"deploy_obs schema mismatch or invalid: {exc}", started_ns, finished_ns
            )
        try:
            action_index = self._combat_session.decide(
                obs_array, episode_start=episode_start or not self._episode_started
            )
            self._episode_started = True
        except StaleSnapshotError as exc:
            finished_ns = time.monotonic_ns()
            return _make_no_op(snapshot, f"stale combat obs: {exc}", started_ns, finished_ns)
        except Exception as exc:  # noqa: BLE001  # unexpected model error → stop
            finished_ns = time.monotonic_ns()
            return _make_stop(snapshot, f"combat inference error: {exc}", started_ns, finished_ns)

        finished_ns = time.monotonic_ns()
        return AgentDecision(
            kind="move",
            decision_id=str(uuid.uuid4()),
            ui_intent=None,
            move_action_index=action_index,
            no_op_reason=None,
            source_snapshot_id=snapshot.snapshot_id,
            source_frame_id=snapshot.frame_id,
            source_content_hash=snapshot.source_content_hash,
            snapshot_timestamp_ns=snapshot.captured_ns,
            inference_started_ns=started_ns,
            inference_finished_ns=finished_ns,
        )

    def _decide_item(self, snapshot: PerceptionSnapshot, *, started_ns: int) -> AgentDecision:
        """ItemSelector で level-up 候補を採点して CHOOSE_CARD を返す。

        ItemSession が None または item_context が None → fallback として non-model UI policy へ。
        target 解決失敗 → stop。
        """
        if self._item_session is None or snapshot.item_context is None:
            # ItemSelector 不在または context なし → non-model policy で処理
            return self._decide_non_model_ui(snapshot, started_ns=started_ns)

        try:
            intent = self._item_session.decide(
                snapshot.item_context,
                snapshot.ui_presentation,
                decision_policy_id=_ITEM_SELECTOR_POLICY_ID,
                decision_rule_id=_ITEM_SELECTOR_RULE_ID,
                decision_config_hash=self._bundle.deploy_schema_hash,
            )
        except ItemSessionError as exc:
            finished_ns = time.monotonic_ns()
            return _make_stop(
                snapshot, f"item session error: {exc}", started_ns, finished_ns
            )

        finished_ns = time.monotonic_ns()
        return AgentDecision(
            kind="ui",
            decision_id=str(uuid.uuid4()),
            ui_intent=intent,
            move_action_index=None,
            no_op_reason=None,
            source_snapshot_id=snapshot.snapshot_id,
            source_frame_id=snapshot.frame_id,
            source_content_hash=snapshot.source_content_hash,
            snapshot_timestamp_ns=snapshot.captured_ns,
            inference_started_ns=started_ns,
            inference_finished_ns=finished_ns,
        )

    def _decide_non_model_ui(
        self, snapshot: PerceptionSnapshot, *, started_ns: int
    ) -> AgentDecision:
        """shared non-model UI policy (02-04 / 05-01 共有) から UiIntentV1 を生成する。

        ui_policy_input が None または policy が None を返す → no_op。
        stop intent → stop decision。
        """
        policy_input = _snap_to_policy_input(snapshot)
        if policy_input is None:
            finished_ns = time.monotonic_ns()
            return _make_no_op(
                snapshot, "ui_policy_input is None", started_ns, finished_ns
            )

        try:
            intent = decide_non_model_ui_intent(policy_input, self._ui_policy_config)
        except ContractValidationError as exc:
            finished_ns = time.monotonic_ns()
            return _make_stop(
                snapshot, f"non-model UI policy error: {exc}", started_ns, finished_ns
            )

        finished_ns = time.monotonic_ns()

        if intent is None:
            return _make_no_op(
                snapshot, "non-model UI policy returned None", started_ns, finished_ns
            )
        if intent.kind == UiIntentKind.STOP:
            return _make_stop(
                snapshot, f"non-model UI policy returned stop: {intent.semantic_action}",
                started_ns, finished_ns
            )

        return AgentDecision(
            kind="ui",
            decision_id=str(uuid.uuid4()),
            ui_intent=intent,
            move_action_index=None,
            no_op_reason=None,
            source_snapshot_id=snapshot.snapshot_id,
            source_frame_id=snapshot.frame_id,
            source_content_hash=snapshot.source_content_hash,
            snapshot_timestamp_ns=snapshot.captured_ns,
            inference_started_ns=started_ns,
            inference_finished_ns=finished_ns,
        )
