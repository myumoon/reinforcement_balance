"""perception snapshot と AgentDecision を、安全な入力 effect の並びへ変換する決定的 hierarchical state machine。

やさしい説明:
このファイルは Survivors を実機で遊ばせるための「唯一の交通整理役」です。
毎フレーム、画面が今どんな状態か(通常プレイ中か、レベルアップ選択中か、
宝箱を開けているところか、死亡/結果画面か、など)を認識結果(screen_state)から
判定し、AI の決定(移動したいのか、UIをクリックしたいのか)と組み合わせて、
「今このtickで送ってよい入力は何か」を1つだけ決めます。

安全のための主な工夫:
- 状態の切り替えは連続 N フレーム一致してから確定する(1フレームの誤認識で
  暴走しない、ヒステリシス/デバウンス)。
- 移動入力と UI 入力を同じ tick に同時に出さない(competing input を作らない)。
- UI クリック前には必ず movement 解放 effect を先に出す(gameplay→level-up の
  「key release が choice click より先」順序)。
- レベルアップ/宝箱/確認画面のクリックは1回だけ再送(retry)を許し、それでも
  変化がなければ input を止めて安全側(emergency stop)へ倒す(fail-closed)。
- unknown は 1s、paused は無期限(input 0 のまま画面変化を待つ)。どちらから
  復帰する場合も RECOVER で gameplay を再確認してから初めて移動を再開する。
  RECOVER/PAUSED/UNKNOWN は、対応する UiIntentV1 を持たない入力(例えば
  自己判断の ESCAPE 連打)を一切発行しない。
- run の成功/失敗/緊急停止は compare-and-set で一度だけ確定し、確定した
  直後に「入力解放 → controller 停止 → プロセス終了要求」の順で必ず1回だけ
  effect を出す。30:00 evidence(``TARGET_REACHED_PENDING_TRANSITION``)を
  観測した後は、何が起きても最終的に成功(``COMPLETE``)へ収束する。

このモジュールは `UiIntentV1` を生成・再分類しません(受け取った決定を
そのまま消費するだけ)。また、画面解析結果の内部表現や非モデル UI 決定ロジック側の
モジュールは import しません(所有権の境界の詳細は
`docs/deployment/controller_states.md` を参照)。入力は `PerceptionSnapshot` と
`AgentDecision`/`UiIntentV1` だけです。
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal

from reinbalance_survivors_contracts.ui_intent import UiIntentKind

from .ui_navigation import Effect, NavigationProfile, resolve_ui_target

if TYPE_CHECKING:  # pragma: no cover - 型注釈だけに使い、実行時 import は避ける。
    from reinbalance_survivors_contracts.ui_intent import UiIntentV1
    from survivors.perception_snapshot import PerceptionSnapshot
    from survivors.runtime.agent_runtime import AgentDecision

__all__ = [
    "CONTROLLER_STATE_SCHEMA_VERSION",
    "ControllerState",
    "CampaignRunMode",
    "RunIdentity",
    "UiAttempt",
    "StateContext",
    "StateMachine",
    "classify_screen_state",
    "reduce",
]

CONTROLLER_STATE_SCHEMA_VERSION = "survivors.controller_state_machine.v1"


class ControllerState(str, enum.Enum):
    """gameplay/UI state machine が取りうる状態。

    やさしい説明: 「今どの画面フェーズにいるか」を表す名札です。
    plan文書(05-03)の遷移表に出てくる状態にそのまま対応します。
    """

    DISARMED = "disarmed"
    ACQUIRE_TARGET = "acquire_target"
    RUN_SETUP = "run_setup"
    GAMEPLAY = "gameplay"
    LEVEL_UP = "level_up"
    CHEST = "chest"
    PAUSED = "paused"
    UNKNOWN = "unknown"
    RECOVER = "recover"
    TARGET_REACHED_PENDING_TRANSITION = "target_reached_pending_transition"
    DEATH_RESULT = "death_result"
    COMPLETE = "complete"
    FORMAL_RUN_TERMINAL_FAILURE = "formal_run_terminal_failure"
    EMERGENCY_STOP = "emergency_stop"


class CampaignRunMode(str, enum.Enum):
    """formal campaign の実行様式。

    やさしい説明: ``FORMAL_SINGLE_ATTEMPT`` は「1プロセスにつき1回の
    正式な run しか許さない」厳格モード、``OPERATOR_DEBUG_RESTART`` は
    「死亡/結果のあと自動で run setup へ戻ってよい」開発者向けモードです。
    """

    FORMAL_SINGLE_ATTEMPT = "formal_single_attempt"
    OPERATOR_DEBUG_RESTART = "operator_debug_restart"


# 生の(HUD由来の) screen_state 文字列を、controller state のカテゴリへ写す表。
# 04-09 parser の raw screen_state 語彙(9値)だけを扱い、未知の文字列や欠落は
# fail-safe に UNKNOWN 扱いにする(05-03 はこの語彙を増やさない)。
_SCREEN_STATE_MAP: dict[str, ControllerState] = {
    "gameplay": ControllerState.GAMEPLAY,
    "level_up_items": ControllerState.LEVEL_UP,
    "level_up_fallback": ControllerState.LEVEL_UP,
    "chest": ControllerState.CHEST,
    "target_reached_transition": ControllerState.TARGET_REACHED_PENDING_TRANSITION,
    "paused": ControllerState.PAUSED,
    "death": ControllerState.DEATH_RESULT,
    "result": ControllerState.DEATH_RESULT,
    "unknown": ControllerState.UNKNOWN,
}

# ControllerState -> NavigationProfile.timeout_ns_for() が使う文字列 key。
# RECOVER には専用 timeout を持たせない(ui_navigation.NavigationProfile 参照)。
_TIMEOUT_KEY: dict[ControllerState, str] = {
    ControllerState.LEVEL_UP: "level_up",
    ControllerState.CHEST: "chest",
    ControllerState.TARGET_REACHED_PENDING_TRANSITION: "target_reached",
}


def classify_screen_state(raw_screen_state: str, *, window_focused: bool = True) -> ControllerState:
    """``PerceptionSnapshot.screen_state`` の生文字列を controller state カテゴリへ分類する。

    やさしい説明: 見知らぬ文字列や window focus 喪失時は、安全側の
    ``UNKNOWN`` として扱います(focus loss は画面の内容を信用できないので
    unknown と同じ扱いにする、という plan 本文の要求に対応します)。
    """
    if not window_focused:
        return ControllerState.UNKNOWN
    return _SCREEN_STATE_MAP.get(raw_screen_state, ControllerState.UNKNOWN)


@dataclass(frozen=True)
class RunIdentity:
    """1 run / 1 gameplay_attempt / 1 process の紐付けを表す不変 identity。"""

    run_id: str
    gameplay_attempt_id: str


@dataclass(frozen=True)
class UiAttempt:
    """進行中の UI click 試行(初回 target・元 snapshot・retry 済みか)を保持する。

    やさしい説明: 「さっき何をクリックしようとしたか」を覚えておく付箋です。
    これがないと、ack(適用確認)が来るまで待つべきか、もう一度送ってよいか
    判断できません。``original_snapshot`` は retry しても更新しません
    (同値性判定は常に *最初に* 送った瞬間の snapshot が基準です)。
    """

    intent_identity: tuple
    original_snapshot: "PerceptionSnapshot"
    sent_ns: int
    retried: bool = False


@dataclass(frozen=True)
class StateContext:
    """state machine の純粋な内部状態(I/O を一切持たない)。

    やさしい説明: 「今の状況」をまるごと1つの不変オブジェクトに詰めたものです。
    ``reduce()`` はこれを受け取り、新しい ``StateContext`` を返すだけの
    純粋関数なので、同じ入力なら必ず同じ結果になり、テストしやすくなります。
    """

    state: ControllerState = ControllerState.DISARMED
    campaign_run_mode: CampaignRunMode = CampaignRunMode.OPERATOR_DEBUG_RESTART
    run_identity: RunIdentity | None = None
    gameplay_entries: int = 0
    pending_category: ControllerState | None = None
    pending_streak: int = 0
    state_entered_ns: int = 0
    success_latched: bool = False
    death_result_reset_emitted: bool = False
    terminal_locked: bool = False
    terminal_state: ControllerState | None = None
    terminal_reason: str | None = None
    ui_attempt: UiAttempt | None = None


def _intent_identity(intent: "UiIntentV1") -> tuple:
    """UiIntentV1 の semantic identity(kind/semantic_action/target 特定情報)を返す。

    やさしい説明: 「同じ意図かどうか」を比較するための短い署名です。
    ``candidate_set_hash`` も含めることで、``choose_card`` が同じ index でも
    別の候補集合を指す状況(reroll による inventory 変化)を別 identity として
    扱い、二重クリックにならないようにします。
    """
    return (
        intent.kind.value,
        intent.semantic_action,
        intent.target_id,
        intent.target_index,
        intent.candidate_set_hash,
    )


def _send_effects(target, *, intent: "UiIntentV1", mode: Literal["initial", "retry"]) -> tuple[Effect, ...]:
    """movement/UI 解放 effect と click/Enter effect を、この順で1組作る。

    やさしい説明: plan の「key release が choice click より先」という順序を
    ここで固定します(``InputLeaseController`` 自体も UI 送信前に movement を
    強制解放しますが、05-03 側の effect 順序としても明示しておきます)。
    ``CONFIRM`` intent だけ Enter キーを使い、それ以外は resolve 済み ROI の
    中心をクリックします。
    """
    release = Effect(kind="release_all", reason="ui_preempt", intent=intent, mode=mode)
    if intent.kind is UiIntentKind.CONFIRM:
        action = Effect(
            kind="ui_key", key="ENTER", target=target, intent=intent, mode=mode,
            reason=f"{intent.kind.value}:{mode}",
        )
    else:
        action = Effect(
            kind="ui_click", target=target, intent=intent, mode=mode,
            reason=f"{intent.kind.value}:{mode}",
        )
    return (release, action)


def _finalize_terminal(
    context: StateContext, terminal_state: ControllerState, *, now_ns: int, reason: str
) -> tuple[StateContext, tuple[Effect, ...]]:
    """formal terminal を compare-and-set で一度だけ確定し、終端 effect 列を返す。

    やさしい説明: 「もう結果は決まった」を1回だけ記録します。既に確定済みなら
    何もせず無視します(二重確定・上書きの防止)。確定した瞬間だけ、
    「入力を全部離す→controllerを止める→プロセス終了を要求する」を
    この順番で必ず1回出します。
    """
    if context.terminal_locked:
        return context, ()
    new_context = replace(
        context,
        state=terminal_state,
        terminal_locked=True,
        terminal_state=terminal_state,
        terminal_reason=reason,
        state_entered_ns=now_ns,
        ui_attempt=None,
    )
    effects = (
        Effect(kind="release_all", reason=reason),
        Effect(kind="controller_stop", reason=reason),
        Effect(kind="process_terminate", reason=reason),
    )
    return new_context, effects


def _emergency_stop(
    context: StateContext, *, now_ns: int, reason: str
) -> tuple[StateContext, tuple[Effect, ...]]:
    """unknown/stuck/retry exhausted を fail-closed に処理する。

    やさしい説明: 「もう安全に続けられない」と判断したときの共通の逃げ道です。
    formal モードでは正式な失敗として確定させ、debug restart モードでは
    入力を全部離して安全に disarmed へ戻します(次の run を待てる状態)。
    """
    if context.campaign_run_mode is CampaignRunMode.FORMAL_SINGLE_ATTEMPT:
        return _finalize_terminal(
            context, ControllerState.FORMAL_RUN_TERMINAL_FAILURE, now_ns=now_ns, reason=reason
        )
    new_context = replace(
        context,
        state=ControllerState.DISARMED,
        pending_category=None,
        pending_streak=0,
        ui_attempt=None,
        state_entered_ns=now_ns,
        terminal_reason=reason,
    )
    return new_context, (Effect(kind="release_all", reason=reason),)


def _advance_debounce(
    context: StateContext, category: ControllerState, frames_required: int
) -> tuple[StateContext, bool]:
    """同じカテゴリが連続何フレーム観測されたかを数え、確定したかを返す。

    やさしい説明: フリッカー(一瞬だけの誤認識)で状態が暴れないよう、
    同じ見え方が既定フレーム数続いてはじめて「確定」とみなします。
    """
    streak = context.pending_streak + 1 if category == context.pending_category else 1
    confirmed = streak >= frames_required
    new_context = replace(context, pending_category=category, pending_streak=streak)
    return new_context, confirmed


def _reset_pending(context: StateContext) -> StateContext:
    """新しい状態へ入った直後、debounce カウンタを初期化する。"""
    return replace(context, pending_category=None, pending_streak=0)


def _enter_death_result(context: StateContext, now_ns: int) -> tuple[StateContext, tuple[Effect, ...]]:
    """``DEATH_RESULT`` へ入る。

    やさしい説明: 「死んだ/結果画面になった」瞬間の入口です。combat reset は
    ``_reduce_death_result`` が次 tick で exactly once 出します
    (``death_result_reset_emitted=False`` にして「まだ出していない」ことを
    記録するだけ)。30:00 evidence を既に観測していたか(``success_latched``)
    はここでは変更せず、そのまま引き継ぎます。
    """
    new_context = _reset_pending(
        replace(
            context,
            state=ControllerState.DEATH_RESULT,
            state_entered_ns=now_ns,
            death_result_reset_emitted=False,
            ui_attempt=None,
        )
    )
    return new_context, ()


def _land_gameplay(context: StateContext, now_ns: int) -> tuple[StateContext, tuple[Effect, ...]]:
    """``RUN_SETUP`` から ``GAMEPLAY`` へ着地する(新規 attempt 扱い)。

    やさしい説明: formal_single_attempt では「予約した gameplay attempt は
    1 回しか activate できない」という約束をここで一括して守ります。
    ``RECOVER`` からの復帰(同じ attempt の一時停止からの再開)はここを
    通らず、``gameplay_entries`` を増やしません(別 attempt 扱いにしない)。
    """
    if context.campaign_run_mode is CampaignRunMode.FORMAL_SINGLE_ATTEMPT and context.gameplay_entries >= 1:
        return _emergency_stop(context, now_ns=now_ns, reason="second_gameplay_entry_rejected")
    new_context = _reset_pending(
        replace(
            context,
            state=ControllerState.GAMEPLAY,
            state_entered_ns=now_ns,
            gameplay_entries=context.gameplay_entries + 1,
            ui_attempt=None,
        )
    )
    return new_context, ()


def _dispatch_ui_click(
    context: StateContext,
    snapshot: "PerceptionSnapshot",
    decision: "AgentDecision",
    *,
    now_ns: int,
    profile: NavigationProfile,
) -> tuple[StateContext, tuple[Effect, ...]]:
    """現在の UI 画面のまま: decision を click/retry effect へ変換する共通ロジック。

    やさしい説明: LEVEL_UP/CHEST/TARGET_REACHED_PENDING_TRANSITION のどれでも
    同じ規則で動きます。「まだクリックしていなければ初回送信」「送信済みなら
    最大1回だけ retry」「意図が変わったら fail-closed」の3つがコアです。
    """
    if decision.kind != "ui" or decision.ui_intent is None:
        return context, ()  # move/no_op は UI 状態中は無視する(movement と競合させない)。
    intent = decision.ui_intent
    if intent.kind is UiIntentKind.STOP:
        return _emergency_stop(context, now_ns=now_ns, reason="ui_intent_stop_signal")
    if intent.kind is UiIntentKind.NO_OP:
        return context, ()

    identity = _intent_identity(intent)

    if context.ui_attempt is None:
        target = resolve_ui_target(intent, snapshot, mode="initial")
        if target is None:
            return context, ()  # 0件/複数件/invalid/confidence未達: effect を出さない。
        attempt = UiAttempt(intent_identity=identity, original_snapshot=snapshot, sent_ns=now_ns, retried=False)
        new_context = replace(context, ui_attempt=attempt)
        return new_context, _send_effects(target, intent=intent, mode="initial")

    attempt = context.ui_attempt
    if identity != attempt.intent_identity:
        # 最初の click 後に意図(候補/inventory/state key)が変わった -> ack 欠落でも再送しない。
        return _emergency_stop(context, now_ns=now_ns, reason="ui_intent_changed_mid_attempt")
    if attempt.retried:
        # retry は 1 回まで。timeout に任せてこれ以上は何もしない。
        return context, ()
    if snapshot.snapshot_id == attempt.original_snapshot.snapshot_id:
        # 同じ snapshot の再利用(ack frame 欠落等)は retry にならない。
        return context, ()
    if profile.retry_budget < 1:
        return context, ()

    target = resolve_ui_target(intent, snapshot, mode="retry", original_snapshot=attempt.original_snapshot)
    if target is None:
        # newer snapshot だが precondition(同一 semantic/ui_state_key/IoU/中心許容量)を
        # 満たさない: 再送せず fail-closed する。
        return _emergency_stop(context, now_ns=now_ns, reason="ui_retry_precondition_failed")
    new_context = replace(context, ui_attempt=replace(attempt, retried=True, sent_ns=now_ns))
    return new_context, _send_effects(target, intent=intent, mode="retry")


def _reduce_gameplay(
    context: StateContext,
    snapshot: "PerceptionSnapshot",
    decision: "AgentDecision",
    category: ControllerState,
    confirmed: bool,
    *,
    now_ns: int,
    profile: NavigationProfile,
) -> tuple[StateContext, tuple[Effect, ...]]:
    """GAMEPLAY 中の画面分類と decision から、次の状態と effect を決める。"""
    if category is ControllerState.DEATH_RESULT:
        if not confirmed:
            return context, ()
        return _enter_death_result(context, now_ns)

    if category is ControllerState.TARGET_REACHED_PENDING_TRANSITION:
        if not confirmed:
            return context, ()
        new_context = _reset_pending(
            replace(
                context,
                state=ControllerState.TARGET_REACHED_PENDING_TRANSITION,
                state_entered_ns=now_ns,
                success_latched=True,
                ui_attempt=None,
            )
        )
        return new_context, ()

    if category in (ControllerState.LEVEL_UP, ControllerState.CHEST):
        if not confirmed:
            return context, ()
        new_context = _reset_pending(
            replace(context, state=category, state_entered_ns=now_ns, ui_attempt=None)
        )
        return new_context, ()

    if category in (ControllerState.PAUSED, ControllerState.UNKNOWN):
        if not confirmed:
            return context, ()
        new_context = _reset_pending(
            replace(context, state=category, state_entered_ns=now_ns, ui_attempt=None)
        )
        return new_context, ()

    # category は GAMEPLAY のまま: movement を 1 tick 1 effect で転送する。
    if decision.kind == "move":
        return context, (Effect(kind="move", action_index=decision.action_index, reason="gameplay_move"),)
    return context, ()


def _reduce_modal_ui_screen(
    context: StateContext,
    snapshot: "PerceptionSnapshot",
    decision: "AgentDecision",
    category: ControllerState,
    confirmed: bool,
    *,
    now_ns: int,
    profile: NavigationProfile,
) -> tuple[StateContext, tuple[Effect, ...]]:
    """LEVEL_UP/CHEST 共通: ack(category 変化)/timeout/click-retry の3分岐。"""
    if confirmed and category is not context.state:
        if category is ControllerState.DEATH_RESULT:
            return _enter_death_result(context, now_ns)
        if category is ControllerState.GAMEPLAY:
            new_context = _reset_pending(
                replace(context, state=ControllerState.GAMEPLAY, state_entered_ns=now_ns, ui_attempt=None)
            )
            return new_context, ()
        # LEVEL_UP <-> CHEST のような想定外の直接遷移は fail-closed。
        return _emergency_stop(context, now_ns=now_ns, reason=f"unexpected_ui_transition:{category.value}")

    timeout_ns = profile.timeout_ns_for(_TIMEOUT_KEY[context.state])
    if now_ns - context.state_entered_ns >= timeout_ns:
        return _emergency_stop(context, now_ns=now_ns, reason=f"{context.state.value}_timeout")

    return _dispatch_ui_click(context, snapshot, decision, now_ns=now_ns, profile=profile)


def _reduce_target_reached(
    context: StateContext,
    snapshot: "PerceptionSnapshot",
    decision: "AgentDecision",
    category: ControllerState,
    confirmed: bool,
    *,
    now_ns: int,
    profile: NavigationProfile,
) -> tuple[StateContext, tuple[Effect, ...]]:
    """``TARGET_REACHED_PENDING_TRANSITION``: 30:00 evidence 後は何が起きても成功優先で ``COMPLETE`` に収束する。

    やさしい説明: この状態へ入った時点で既に「30分到達」の証拠は観測済みです。
    以後、画面が(post-30 の確認イベントで)gameplay へ戻ろうと、直後に
    death/result になろうと、timeout してしまおうと、結果は必ず成功
    (``COMPLETE``)になります。plan 本文の「success を優先する」という
    precedence ルールをここで実装します。
    """
    if confirmed and category is not ControllerState.TARGET_REACHED_PENDING_TRANSITION:
        if category is ControllerState.DEATH_RESULT:
            return _enter_death_result(context, now_ns)
        # target_reached 以外へ移った(= post-30 screen event を確認できた)ので成功確定。
        return _finalize_terminal(context, ControllerState.COMPLETE, now_ns=now_ns, reason="post_30_confirm_success")

    timeout_ns = profile.timeout_ns_for("target_reached")
    if now_ns - context.state_entered_ns >= timeout_ns:
        return _finalize_terminal(
            context, ControllerState.COMPLETE, now_ns=now_ns, reason="target_reached_timeout_default_success"
        )

    return _dispatch_ui_click(context, snapshot, decision, now_ns=now_ns, profile=profile)


def _reduce_paused(
    context: StateContext, category: ControllerState, confirmed: bool, *, now_ns: int, profile: NavigationProfile
) -> tuple[StateContext, tuple[Effect, ...]]:
    """``PAUSED``: timeout なし(indefinite、input 0)。gameplay を確認したら ``RECOVER`` で再確認する。"""
    if category is ControllerState.DEATH_RESULT:
        if not confirmed:
            return context, ()
        return _enter_death_result(context, now_ns)
    if category is ControllerState.PAUSED:
        return _reset_pending(context), ()
    if not confirmed:
        return context, ()
    if category is ControllerState.GAMEPLAY:
        new_context = _reset_pending(replace(context, state=ControllerState.RECOVER, state_entered_ns=now_ns))
        return new_context, ()
    # unknown 等、他カテゴリへ確定的に移った場合はそちらの timeout 管理へ委ねる。
    new_context = _reset_pending(replace(context, state=ControllerState.UNKNOWN, state_entered_ns=now_ns))
    return new_context, ()


def _reduce_unknown(
    context: StateContext, category: ControllerState, confirmed: bool, *, now_ns: int, profile: NavigationProfile
) -> tuple[StateContext, tuple[Effect, ...]]:
    """``UNKNOWN``: 1s(profile 由来)timeout で fail-closed。gameplay 復帰は ``RECOVER`` を経由する。"""
    if category is ControllerState.DEATH_RESULT:
        if not confirmed:
            return context, ()
        return _enter_death_result(context, now_ns)
    if category is ControllerState.GAMEPLAY:
        if not confirmed:
            return context, ()
        new_context = _reset_pending(replace(context, state=ControllerState.RECOVER, state_entered_ns=now_ns))
        return new_context, ()
    if category is ControllerState.PAUSED:
        if not confirmed:
            return context, ()
        new_context = _reset_pending(replace(context, state=ControllerState.PAUSED, state_entered_ns=now_ns))
        return new_context, ()
    if category in (ControllerState.LEVEL_UP, ControllerState.CHEST, ControllerState.TARGET_REACHED_PENDING_TRANSITION):
        if not confirmed:
            return context, ()
        new_context = _reset_pending(replace(context, state=category, state_entered_ns=now_ns))
        return new_context, ()
    # category は UNKNOWN のまま: timeout で fail-closed する。
    context = _reset_pending(context)
    if now_ns - context.state_entered_ns >= profile.timeout_ns_for("unknown"):
        return _emergency_stop(context, now_ns=now_ns, reason="unknown_timeout")
    return context, ()


def _reduce_recover(
    context: StateContext, category: ControllerState, confirmed: bool, *, now_ns: int, profile: NavigationProfile
) -> tuple[StateContext, tuple[Effect, ...]]:
    """``RECOVER``: gameplay の再確定バッファ。確定しなければ PAUSED/UNKNOWN の timeout 管理へ戻す。

    やさしい説明: ここでは絶対に入力を送りません(ESCAPE 等を自己判断で
    打つこともしません)。ただ「もう一度 gameplay 画面が安定して見えるか」を
    確認するだけの、純粋な待ち時間バッファです。
    """
    if category is ControllerState.DEATH_RESULT:
        if not confirmed:
            return context, ()
        return _enter_death_result(context, now_ns)
    if category is ControllerState.GAMEPLAY:
        if confirmed:
            new_context = _reset_pending(
                replace(context, state=ControllerState.GAMEPLAY, state_entered_ns=now_ns)
            )
            return new_context, ()
        return context, ()
    if category is ControllerState.PAUSED:
        new_context = _reset_pending(replace(context, state=ControllerState.PAUSED, state_entered_ns=now_ns))
        return new_context, ()
    # unknown を含む他カテゴリへ流れたら unknown の timeout 管理へ戻す。
    new_context = _reset_pending(replace(context, state=ControllerState.UNKNOWN, state_entered_ns=now_ns))
    return new_context, ()


def _reduce_death_result(
    context: StateContext, *, now_ns: int
) -> tuple[StateContext, tuple[Effect, ...]]:
    """DEATH_RESULT: combat LSTM reset を一度だけ出し、success 優先で terminal を確定する。"""
    if not context.death_result_reset_emitted:
        new_context = replace(context, death_result_reset_emitted=True)
        return new_context, (Effect(kind="combat_reset", reason="death_result_entered"),)

    if context.success_latched:
        return _finalize_terminal(
            context, ControllerState.COMPLETE, now_ns=now_ns, reason="death_after_target_reached_success"
        )

    if context.campaign_run_mode is CampaignRunMode.FORMAL_SINGLE_ATTEMPT:
        return _finalize_terminal(
            context,
            ControllerState.FORMAL_RUN_TERMINAL_FAILURE,
            now_ns=now_ns,
            reason="pre_30_death_result_failure",
        )

    # OPERATOR_DEBUG_RESTART だけが、terminal 化せず RUN_SETUP へ戻ってよい。
    new_context = _reset_pending(
        replace(
            context,
            state=ControllerState.RUN_SETUP,
            state_entered_ns=now_ns,
            ui_attempt=None,
        )
    )
    return new_context, ()


def reduce(
    context: StateContext,
    snapshot: "PerceptionSnapshot",
    decision: "AgentDecision",
    *,
    now_ns: int,
    profile: NavigationProfile | None = None,
    window_focused: bool = True,
    manual_abort_requested: bool = False,
) -> tuple[StateContext, tuple[Effect, ...]]:
    """1 tick 分の純粋な reduce: (state, snapshot, decision) -> (new_state, effects)。

    やさしい説明: I/O を一切しない、ただの計算です。同じ入力なら必ず同じ
    出力になるので、テストで安心して繰り返し呼べます。実際に
    ``InputLeaseController`` を呼ぶのは ``ui_navigation.execute_effect`` の役目です。
    手動中断(``manual_abort_requested``)は他の何よりも優先して emergency
    stop します。``window_focused=False``(フォーカス喪失)は screen 分類を
    強制的に ``UNKNOWN`` にします。
    """
    profile = profile or NavigationProfile.default_v1()

    if context.terminal_locked:
        return context, ()

    if manual_abort_requested:
        return _emergency_stop(context, now_ns=now_ns, reason="manual_abort")
    if decision.kind == "stop":
        return _emergency_stop(context, now_ns=now_ns, reason="agent_runtime_stop")

    if context.state is ControllerState.DISARMED:
        # arm() されるまで画面入力を一切見ない。
        return context, ()

    category = classify_screen_state(snapshot.screen_state, window_focused=window_focused)
    context, confirmed = _advance_debounce(context, category, profile.debounce_frames)

    if context.state is ControllerState.ACQUIRE_TARGET:
        if confirmed and category is not ControllerState.UNKNOWN:
            new_context = _reset_pending(
                replace(context, state=ControllerState.RUN_SETUP, state_entered_ns=now_ns)
            )
            return new_context, ()
        if now_ns - context.state_entered_ns >= profile.timeout_ns_for("run_setup"):
            return _emergency_stop(context, now_ns=now_ns, reason="acquire_target_timeout")
        return context, ()

    if context.state is ControllerState.RUN_SETUP:
        if confirmed and category is ControllerState.GAMEPLAY:
            return _land_gameplay(context, now_ns)
        if now_ns - context.state_entered_ns >= profile.timeout_ns_for("run_setup"):
            return _emergency_stop(context, now_ns=now_ns, reason="run_setup_timeout")
        return context, ()

    if context.state is ControllerState.GAMEPLAY:
        return _reduce_gameplay(context, snapshot, decision, category, confirmed, now_ns=now_ns, profile=profile)

    if context.state in (ControllerState.LEVEL_UP, ControllerState.CHEST):
        return _reduce_modal_ui_screen(context, snapshot, decision, category, confirmed, now_ns=now_ns, profile=profile)

    if context.state is ControllerState.TARGET_REACHED_PENDING_TRANSITION:
        return _reduce_target_reached(context, snapshot, decision, category, confirmed, now_ns=now_ns, profile=profile)

    if context.state is ControllerState.PAUSED:
        return _reduce_paused(context, category, confirmed, now_ns=now_ns, profile=profile)

    if context.state is ControllerState.UNKNOWN:
        return _reduce_unknown(context, category, confirmed, now_ns=now_ns, profile=profile)

    if context.state is ControllerState.RECOVER:
        return _reduce_recover(context, category, confirmed, now_ns=now_ns, profile=profile)

    if context.state is ControllerState.DEATH_RESULT:
        return _reduce_death_result(context, now_ns=now_ns)

    return _emergency_stop(context, now_ns=now_ns, reason=f"unhandled_state:{context.state.value}")


@dataclass
class StateMachine:
    """``reduce()`` を薄く包む、呼び出し側向けの状態保持オブジェクト。

    やさしい説明: テストや launcher からは、この ``StateMachine`` を1つ作って
    毎フレーム ``step()`` を呼ぶだけで済みます。実際の判定ロジックはすべて
    純粋関数 ``reduce()`` に委譲しているので、``StateMachine`` 自体は
    「今の状態を覚えておくだけの薄い箱」です。``DISARMED`` からは必ず
    ``arm()`` を明示的に呼ばないと動き出しません(画面認識だけで自己武装しない)。
    """

    profile: NavigationProfile = field(default_factory=NavigationProfile.default_v1)
    context: StateContext = field(default_factory=StateContext)

    def arm(
        self,
        *,
        campaign_run_mode: CampaignRunMode,
        run_id: str,
        gameplay_attempt_id: str,
        now_ns: int,
    ) -> None:
        """DISARMED から ACQUIRE_TARGET へ明示的に遷移させる。

        やさしい説明: 「この run/attempt をこれから見張る」と宣言する操作です。
        画面認識では起動できず、必ず呼び出し側が明示的に呼びます。
        """
        if self.context.state is not ControllerState.DISARMED:
            raise ValueError("arm() requires the state machine to be DISARMED")
        self.context = replace(
            self.context,
            state=ControllerState.ACQUIRE_TARGET,
            campaign_run_mode=campaign_run_mode,
            run_identity=RunIdentity(run_id=run_id, gameplay_attempt_id=gameplay_attempt_id),
            state_entered_ns=now_ns,
            pending_category=None,
            pending_streak=0,
        )

    def step(
        self, snapshot: "PerceptionSnapshot", decision: "AgentDecision", *, now_ns: int, **kwargs
    ) -> tuple[Effect, ...]:
        """1 tick 進め、このtickで発行すべき effect 列を返す。"""
        self.context, effects = reduce(self.context, snapshot, decision, now_ns=now_ns, profile=self.profile, **kwargs)
        return effects

    def emergency_stop(self, *, now_ns: int, reason: str) -> tuple[Effect, ...]:
        """呼び出し側(effect executor)が検知した致命的な入出力失敗を fail-closed に処理する。"""
        self.context, effects = _emergency_stop(self.context, now_ns=now_ns, reason=reason)
        return effects
