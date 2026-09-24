"""controller state machine のテストで使う実オブジェクト fixture builder。

やさしい説明:
テストで使う「本物そっくりの `PerceptionSnapshot`/`AgentDecision`/`UiIntentV1`」を
組み立てる道具箱です。`Tools/Deployment/tests/runtime/_runtime_fixtures.py` と同じ
流儀(conftest.py ではなく素の python モジュールを import する方式)で、実装側の
コードを一切呼び出さずに契約が定める「形」だけをテスト側で独立に組み立てます。
"""
from __future__ import annotations

from reinbalance_survivors_contracts.canonical_json import canonical_hash
from reinbalance_survivors_contracts.deploy_obs import DeployObsSchema, DeployObservation
from reinbalance_survivors_contracts.ui_intent import DecisionOwner, UiIntentKind, UiIntentV1

import numpy as np

from survivors.perception_snapshot import (
    UI_PRESENTATION_SCHEMA_HASH,
    NormalizedRoi,
    PerceptionSnapshot,
    UiButtonTargetV1,
    UiCandidateTargetV1,
    UiPresentationSnapshotV1,
)
from survivors.runtime.agent_runtime import AGENT_DECISION_SCHEMA_VERSION, AgentDecision

_SCHEMA = DeployObsSchema.default_v1()


def _hash_of(label: str) -> str:
    """テスト専用の 64 文字ハッシュを、識別しやすい label から決定的に作る。"""
    return canonical_hash({"fixture_label": label})


def make_deploy_observation() -> DeployObservation:
    """schema と整合する、最小の有効な `DeployObservation` を返す。"""
    dim = _SCHEMA.dim
    values = np.zeros(dim, dtype=np.float32)
    validity = np.ones(dim, dtype=np.float32)
    age = np.zeros(dim, dtype=np.float32)
    return DeployObservation(
        values=values,
        validity=validity,
        age=age,
        schema_hash=_SCHEMA.schema_hash,
        timestamp_ns=0,
        provenance="release",
    )


def make_roi(left: float = 0.1, top: float = 0.1, right: float = 0.2, bottom: float = 0.2) -> NormalizedRoi:
    """クリック対象の正規化 ROI を作る。"""
    return NormalizedRoi(left, top, right, bottom)


def make_candidate_target(
    *,
    choice_id: str = "card-0",
    choice_index: int = 0,
    semantic_kind: str = "item_card",
    roi: NormalizedRoi | None = None,
    validity: bool = True,
    confidence: float = 0.9,
) -> UiCandidateTargetV1:
    """候補カード target を作る。"""
    return UiCandidateTargetV1(
        choice_id=choice_id,
        choice_index=choice_index,
        semantic_kind=semantic_kind,
        roi=roi or make_roi(),
        validity=validity,
        confidence=confidence,
    )


def make_button_target(
    *,
    semantic_action: str = "ack_chest",
    roi: NormalizedRoi | None = None,
    validity: bool = True,
    capability: bool = True,
    confidence: float = 0.9,
) -> UiButtonTargetV1:
    """ボタン target を作る。"""
    return UiButtonTargetV1(
        semantic_action=semantic_action,
        roi=roi or make_roi(0.4, 0.4, 0.5, 0.5),
        validity=validity,
        capability=capability,
        confidence=confidence,
    )


def make_ui_presentation(
    *,
    screen_state: str,
    snapshot_id: str,
    frame_id: str,
    source_content_hash: str,
    ui_state_key: str,
    parser_artifact_hash: str | None = None,
    candidate_set_hash: str | None = None,
    inventory_hash: str | None = None,
    candidates: tuple[UiCandidateTargetV1, ...] = (),
    buttons: tuple[UiButtonTargetV1, ...] = (),
) -> UiPresentationSnapshotV1:
    """UI presentation snapshot を作る。identity 系ハッシュは省略時に自動生成する。"""
    return UiPresentationSnapshotV1(
        schema_hash=UI_PRESENTATION_SCHEMA_HASH,
        snapshot_id=snapshot_id,
        frame_id=frame_id,
        parser_artifact_hash=parser_artifact_hash or _hash_of("parser_artifact"),
        screen_state=screen_state,
        candidate_set_hash=candidate_set_hash or _hash_of(f"candidate_set:{snapshot_id}"),
        inventory_hash=inventory_hash or _hash_of("inventory"),
        source_content_hash=source_content_hash,
        ui_state_key=ui_state_key,
        candidates=candidates,
        buttons=buttons,
    )


def make_perception_snapshot(
    *,
    screen_state: str,
    snapshot_id: str = "snap-0",
    frame_id: str = "frame-0",
    captured_ns: int = 0,
    ui_state_key: str | None = None,
    source_content_hash: str | None = None,
    candidates: tuple[UiCandidateTargetV1, ...] = (),
    buttons: tuple[UiButtonTargetV1, ...] = (),
    candidate_set_hash: str | None = None,
    inventory_hash: str | None = None,
) -> PerceptionSnapshot:
    """`PerceptionSnapshot` を、UI presentation との atomic binding を満たしたまま作る。"""
    content_hash = source_content_hash or _hash_of(f"content:{snapshot_id}:{frame_id}")
    resolved_ui_state_key = ui_state_key or _hash_of(f"ui_state_key:{snapshot_id}")
    parser_artifact_hash = _hash_of("parser_artifact")
    ui_presentation = make_ui_presentation(
        screen_state=screen_state,
        snapshot_id=snapshot_id,
        frame_id=frame_id,
        source_content_hash=content_hash,
        ui_state_key=resolved_ui_state_key,
        parser_artifact_hash=parser_artifact_hash,
        candidate_set_hash=candidate_set_hash,
        inventory_hash=inventory_hash,
        candidates=candidates,
        buttons=buttons,
    )
    return PerceptionSnapshot(
        snapshot_id=snapshot_id,
        frame_id=frame_id,
        captured_ns=captured_ns,
        parser_artifact_hash=parser_artifact_hash,
        source_content_hash=content_hash,
        ui_state_key=resolved_ui_state_key,
        screen_state=screen_state,
        deploy_obs=make_deploy_observation(),
        item_context=None,
        choices=(),
        ui_presentation=ui_presentation,
        diagnostics={},
    )


def make_move_decision(
    snapshot: PerceptionSnapshot, *, action_index: int = 8, decision_id: str = "decision-move"
) -> AgentDecision:
    """movement 決定を作る。"""
    return AgentDecision(
        decision_id=decision_id,
        kind="move",
        action_index=action_index,
        ui_intent=None,
        confidence=1.0,
        reason="fixture move decision",
        source_snapshot_id=snapshot.snapshot_id,
        source_frame_id=snapshot.frame_id,
        source_content_hash=snapshot.source_content_hash,
        snapshot_timestamp_ns=snapshot.captured_ns,
        inference_started_ns=snapshot.captured_ns,
        inference_finished_ns=snapshot.captured_ns,
        schema_version=AGENT_DECISION_SCHEMA_VERSION,
    )


def make_no_op_decision(snapshot: PerceptionSnapshot, *, decision_id: str = "decision-no-op") -> AgentDecision:
    """``kind="no_op"`` 決定を作る。

    やさしい説明: 本物の `AgentRuntime` は death/result/unknown/paused の
    episode boundary/idle screen state では ``"no_op"`` を返す
    (``_EPISODE_BOUNDARY_SCREEN_STATES``/``_IDLE_SCREEN_STATES``、
    `survivors/runtime/agent_runtime.py`)。状態遷移テストではこちらを使う。
    """
    return AgentDecision(
        decision_id=decision_id,
        kind="no_op",
        action_index=None,
        ui_intent=None,
        confidence=0.0,
        reason="fixture no_op decision",
        source_snapshot_id=snapshot.snapshot_id,
        source_frame_id=snapshot.frame_id,
        source_content_hash=snapshot.source_content_hash,
        snapshot_timestamp_ns=snapshot.captured_ns,
        inference_started_ns=snapshot.captured_ns,
        inference_finished_ns=snapshot.captured_ns,
        schema_version=AGENT_DECISION_SCHEMA_VERSION,
    )


def make_stop_decision(snapshot: PerceptionSnapshot, *, decision_id: str = "decision-stop") -> AgentDecision:
    """``kind="stop"`` 決定を作る。

    やさしい説明: 本物の `AgentRuntime` はこれを、screen state 分類とは
    無関係な致命的 safety 違反(観測が古すぎる・global validity 未達・
    非モデル UI policy の contract 違反など)のときだけ返す。従って
    state machine 側もこれを「どの状態にいても即座に emergency stop する」
    最優先信号として扱う。
    """
    return AgentDecision(
        decision_id=decision_id,
        kind="stop",
        action_index=None,
        ui_intent=None,
        confidence=0.0,
        reason="fixture stop decision",
        source_snapshot_id=snapshot.snapshot_id,
        source_frame_id=snapshot.frame_id,
        source_content_hash=snapshot.source_content_hash,
        snapshot_timestamp_ns=snapshot.captured_ns,
        inference_started_ns=snapshot.captured_ns,
        inference_finished_ns=snapshot.captured_ns,
        schema_version=AGENT_DECISION_SCHEMA_VERSION,
    )


def make_choose_card_intent(
    snapshot: PerceptionSnapshot,
    *,
    target_index: int = 0,
    candidate_set_hash: str | None = None,
    policy_id: str = "test_item_selector_policy.v1",
    rule_id: str = "test_item_selector_rule.v1",
) -> UiIntentV1:
    """`choose_card` intent を、指定 snapshot に source binding させて作る。"""
    return UiIntentV1(
        kind=UiIntentKind.CHOOSE_CARD,
        semantic_action="choose_card",
        decision_owner=DecisionOwner.ITEM_SELECTOR_SESSION,
        source_snapshot_hash=snapshot.snapshot_id,
        source_frame_hash=snapshot.frame_id,
        source_content_hash=snapshot.source_content_hash,
        ui_state_key=snapshot.ui_state_key,
        target_index=target_index,
        candidate_set_hash=candidate_set_hash or snapshot.ui_presentation.candidate_set_hash,
        decision_policy_id=policy_id,
        decision_rule_id=rule_id,
        decision_config_hash=_hash_of("item_selector_config"),
    )


def make_choose_fallback_intent(
    snapshot: PerceptionSnapshot,
    *,
    target_id: str = "fallback-gold",
    target_index: int = 0,
    semantic: str = "gold",
    policy_id: str = "test_non_model_ui_policy.v1",
    rule_id: str = "test_fallback_rule.v1",
) -> UiIntentV1:
    """`choose_fallback` intent を作る。"""
    return UiIntentV1(
        kind=UiIntentKind.CHOOSE_FALLBACK,
        semantic_action=semantic,
        decision_owner=DecisionOwner.NON_MODEL_UI_POLICY,
        source_snapshot_hash=snapshot.snapshot_id,
        source_frame_hash=snapshot.frame_id,
        source_content_hash=snapshot.source_content_hash,
        ui_state_key=snapshot.ui_state_key,
        target_id=target_id,
        target_index=target_index,
        decision_policy_id=policy_id,
        decision_rule_id=rule_id,
        decision_config_hash=_hash_of("fallback_config"),
    )


def make_button_intent(
    snapshot: PerceptionSnapshot,
    *,
    kind: UiIntentKind,
    semantic_action: str,
    policy_id: str = "test_non_model_ui_policy.v1",
    rule_id: str = "test_button_rule.v1",
) -> UiIntentV1:
    """reroll/skip/banish/ack_chest/confirm intent を作る。"""
    return UiIntentV1(
        kind=kind,
        semantic_action=semantic_action,
        decision_owner=DecisionOwner.NON_MODEL_UI_POLICY,
        source_snapshot_hash=snapshot.snapshot_id,
        source_frame_hash=snapshot.frame_id,
        source_content_hash=snapshot.source_content_hash,
        ui_state_key=snapshot.ui_state_key,
        decision_policy_id=policy_id,
        decision_rule_id=rule_id,
        decision_config_hash=_hash_of("button_config"),
    )


def make_stop_intent(snapshot: PerceptionSnapshot) -> UiIntentV1:
    """runtime safety 由来の `stop` intent を作る。"""
    return UiIntentV1(
        kind=UiIntentKind.STOP,
        semantic_action="stop",
        decision_owner=DecisionOwner.RUNTIME_SAFETY,
        source_snapshot_hash=snapshot.snapshot_id,
        source_frame_hash=snapshot.frame_id,
        source_content_hash=snapshot.source_content_hash,
        ui_state_key=snapshot.ui_state_key,
    )


def make_ui_decision(
    snapshot: PerceptionSnapshot, intent: UiIntentV1, *, decision_id: str = "decision-ui"
) -> AgentDecision:
    """`kind="ui"` の決定を、指定 snapshot/intent に source binding させて作る。"""
    return AgentDecision(
        decision_id=decision_id,
        kind="ui",
        action_index=None,
        ui_intent=intent,
        confidence=1.0,
        reason="fixture ui decision",
        source_snapshot_id=snapshot.snapshot_id,
        source_frame_id=snapshot.frame_id,
        source_content_hash=snapshot.source_content_hash,
        snapshot_timestamp_ns=snapshot.captured_ns,
        inference_started_ns=snapshot.captured_ns,
        inference_finished_ns=snapshot.captured_ns,
        schema_version=AGENT_DECISION_SCHEMA_VERSION,
    )
