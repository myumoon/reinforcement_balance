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
  「key release が choice click より先」順序)。GAMEPLAY から他の状態へ
  確定的に抜けるtick自体でも、同様に ``release_all`` を1回発行する。
- レベルアップ/宝箱/確認画面のクリックは、送信直後の ack 待ち window
  (``NavigationProfile.retry_after_ns``)を経過するまで retry せず、
  その後1回だけ再送(retry)を許す。送信済み snapshot から見て
  ``ui_state_key``/``candidate_set_hash``/``inventory_hash`` が変わっても、
  同じ新しい組が ``debounce_frames`` 回連続で安定し、かつ ack 待ち window も
  経過して初めて「game 側の apply ack」と確定し、以後の intent を新規
  initial resolve として扱う(reroll/banish 成功後の次選択や連続 level-up を
  誤って emergency stop しない一方、フェード中の1フレームだけの揺れでは
  無制限に再クリックしない、M15 fix)。それでも変化がなければ input を止めて
  安全側(emergency stop)へ倒す(fail-closed)。
- unknown は 1s、paused は無期限(input 0 のまま画面変化を待つ)。どちらから
  復帰する場合も RECOVER で gameplay を再確認してから初めて移動を再開する。
  RECOVER/PAUSED/UNKNOWN は、対応する UiIntentV1 を持たない入力(例えば
  自己判断の ESCAPE 連打)を一切発行しない。
- run の成功/失敗/緊急停止は compare-and-set で一度だけ確定し、確定した
  直後に「入力解放 → controller 停止 → プロセス終了要求」の順で必ず1回だけ
  effect を出す。30:00 evidence(``TARGET_REACHED_PENDING_TRANSITION``)を
  観測した経路(GAMEPLAY直行/UNKNOWN経由のどちらでも)は共通の入口を通り、
  一貫して success を予約する。ただし成功が確定するのは、その後
  ``GAMEPLAY`` への復帰または ``death``/``result`` という post-30 の
  確定的な画面変化を確認できたときだけであり、未確認のままの timeout や
  他カテゴリへの遷移は成功にせず fail-closed する。
- ``StateMachine.arm()`` は ``profile`` 以外の実行文脈を毎回まっさらな
  ``StateContext`` として作り直す(前 run の success_latched/gameplay_entries
  等を新しい run に持ち越さない)。

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
    # apply ack 安定確認用のバッファ(M15 fix)。
    #
    # やさしい説明: ui_state_key/candidate_set_hash/inventory_hash が
    # original_snapshot と違う値へ「今 tick 何に変わったか」を覚えておく付箋です。
    # 同じ新しい値が ``debounce_frames`` 回連続で観測されるまでは apply ack と
    # 認定しないための一時カウンタで、値が安定しない限り増え続けません。
    pending_ack_signature: tuple[str, str, str] | None = None
    pending_ack_streak: int = 0


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


def _enter_target_reached(context: StateContext, now_ns: int) -> tuple[StateContext, tuple[Effect, ...]]:
    """``TARGET_REACHED_PENDING_TRANSITION`` へ入る(30:00 evidence 観測)。

    やさしい説明: GAMEPLAY から直接見えた場合も、一瞬 UNKNOWN(画面切り替え中の
    ノイズ等)を経由してから見えた場合も、必ずこの共通入口を通します
    (M7 fix: 経路によって ``success_latched`` を立てたり立てなかったりする
    抜け道をなくす)。ここに入った時点で「30:00 evidence は確かに観測した」と
    いう事実を ``success_latched=True`` として1つの真実にし、以後
    ``_reduce_target_reached``/``_reduce_death_result`` がこれを見て
    success を優先します。
    """
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


def _ui_signature(snapshot: "PerceptionSnapshot") -> tuple[str, str, str]:
    """apply ack 判定に使う (ui_state_key, candidate_set_hash, inventory_hash) の組を返す。

    やさしい説明: この3値のうちどれか1つでも前回送信時の snapshot と違えば
    「ゲーム側が実際にクリックを処理したらしい」候補ですが、それだけでは
    UI フェード中の1フレームだけの揺れと区別できません。``_dispatch_ui_click``
    側でこの組が ``debounce_frames`` 回連続で安定するまで待たせることで、
    apply ack の誤検出(と、それによる無制限 re-click)を防ぎます(M15 fix)。
    """
    return (
        snapshot.ui_state_key,
        snapshot.ui_presentation.candidate_set_hash,
        snapshot.ui_presentation.inventory_hash,
    )


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
    同じ規則で動きます。優先順位は次のとおりです。

    1. 送信済み click の対象から ``ui_state_key``/``candidate_set_hash``/
       ``inventory_hash`` のいずれかが変わっていたら、apply ack の *候補* と
       みなす。ただし同じ新しい組が ``debounce_frames`` 回連続で観測され、
       かつ ack 待ち window(``retry_after_ns``)も経過して初めて確定させ、
       ``ui_attempt`` をクリアして今 tick の intent を *新規* initial resolve
       として扱う(M15 fix: 安定条件を満たすまでは再送も新規clickもしない。
       1 フレームだけの揺れで無制限 re-click しないようにするため)。
    2. 安定条件を満たしていない(＝まだ確定していない)間は、identity 判定より
       先にここで待つ。intent identity 自体が変わっていない限り誤検出しない。
    3. apply ack 候補が無いのに intent identity が変わっていたら fail-closed。
    4. 既に retry 済みなら何もしない。
    5. ack 待ち window(``retry_after_ns``)を経過するまでは retry しない
       (M4 fix: capture/perception 遅延中の二重click防止)。
    6. 同一 snapshot の再利用は retry にならない。
    7. precondition を満たす newer snapshot だけ、最大1回 retry する。
    """
    if decision.kind != "ui" or decision.ui_intent is None:
        return context, ()  # move/no_op は UI 状態中は無視する(movement と競合させない)。
    intent = decision.ui_intent
    if intent.kind is UiIntentKind.STOP:
        return _emergency_stop(context, now_ns=now_ns, reason="ui_intent_stop_signal")
    if intent.kind is UiIntentKind.NO_OP:
        return context, ()

    if context.ui_attempt is not None:
        attempt = context.ui_attempt
        current_signature = _ui_signature(snapshot)
        if current_signature != _ui_signature(attempt.original_snapshot):
            # apply ack の候補: 前と同じ新しい組が続けて観測されているかを数える。
            if current_signature == attempt.pending_ack_signature:
                ack_streak = attempt.pending_ack_streak + 1
            else:
                ack_streak = 1
            ack_window_elapsed = now_ns - attempt.sent_ns >= profile.retry_after_ns
            if ack_streak >= profile.debounce_frames and ack_window_elapsed:
                # 安定条件を満たした: 本当に apply ack と確定し、今 tick の intent を
                # 新規 initial resolve として扱う(旧 intent が遅れて届いても
                # source snapshot binding(mode="initial")が一致しないため再送されない)。
                return _dispatch_ui_click(
                    replace(context, ui_attempt=None), snapshot, decision, now_ns=now_ns, profile=profile
                )
            # まだ確定していない揺れ: 再送も新規clickもせず、次tickの判定用に
            # カウンタだけ更新して待つ(fail-closed にもしない)。
            new_attempt = replace(attempt, pending_ack_signature=current_signature, pending_ack_streak=ack_streak)
            return replace(context, ui_attempt=new_attempt), ()

    identity = _intent_identity(intent)

    if context.ui_attempt is None:
        target = resolve_ui_target(intent, snapshot, mode="initial", min_confidence=profile.min_target_confidence)
        if target is None:
            return context, ()  # 0件/複数件/invalid/confidence未達: effect を出さない。
        attempt = UiAttempt(intent_identity=identity, original_snapshot=snapshot, sent_ns=now_ns, retried=False)
        new_context = replace(context, ui_attempt=attempt)
        return new_context, _send_effects(target, intent=intent, mode="initial")

    attempt = context.ui_attempt
    if identity != attempt.intent_identity:
        # apply ack ではないのに意図(候補/inventory/state key)が変わった
        # -> ack 欠落でも再送しない(fail-closed)。
        return _emergency_stop(context, now_ns=now_ns, reason="ui_intent_changed_mid_attempt")
    if attempt.retried:
        # retry は 1 回まで。timeout に任せてこれ以上は何もしない。
        return context, ()
    if now_ns - attempt.sent_ns < profile.retry_after_ns:
        # ack 待ち window 内: ゲームがまだ初回 click を処理し切っていない
        # 可能性があるので、newer snapshot が来ていても retry しない。
        return context, ()
    if snapshot.snapshot_id == attempt.original_snapshot.snapshot_id:
        # 同じ snapshot の再利用(ack frame 欠落等)は retry にならない。
        return context, ()
    if profile.retry_budget < 1:
        return context, ()

    target = resolve_ui_target(
        intent,
        snapshot,
        mode="retry",
        original_snapshot=attempt.original_snapshot,
        min_confidence=profile.min_target_confidence,
    )
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
    """GAMEPLAY 中の画面分類と decision から、次の状態と effect を決める。

    やさしい説明: GAMEPLAY から別の状態へ確定的に抜けるときは、必ず
    ``release_all`` を1回発行してから抜けます(overall_review fix: unknown/
    paused/focus loss/error は movement release、という plan18行目の要求を
    75ms movement lease の自然失効任せにせず明示的に保証する)。
    """
    _GAMEPLAY_EXIT_EFFECTS = (Effect(kind="release_all", reason="gameplay_exit"),)

    if category is ControllerState.DEATH_RESULT:
        if not confirmed:
            return context, ()
        new_context, _ = _enter_death_result(context, now_ns)
        return new_context, _GAMEPLAY_EXIT_EFFECTS

    if category is ControllerState.TARGET_REACHED_PENDING_TRANSITION:
        if not confirmed:
            return context, ()
        new_context, _ = _enter_target_reached(context, now_ns)
        return new_context, _GAMEPLAY_EXIT_EFFECTS

    if category in (ControllerState.LEVEL_UP, ControllerState.CHEST):
        if not confirmed:
            return context, ()
        new_context = _reset_pending(
            replace(context, state=category, state_entered_ns=now_ns, ui_attempt=None)
        )
        return new_context, _GAMEPLAY_EXIT_EFFECTS

    if category in (ControllerState.PAUSED, ControllerState.UNKNOWN):
        if not confirmed:
            return context, ()
        new_context = _reset_pending(
            replace(context, state=category, state_entered_ns=now_ns, ui_attempt=None)
        )
        return new_context, _GAMEPLAY_EXIT_EFFECTS

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
    """``TARGET_REACHED_PENDING_TRANSITION``: post-30 の確定的な画面変化を確認して初めて成功にする。

    やさしい説明: この状態へ入った時点で既に「30分到達」の証拠は観測済み
    (``success_latched=True``)ですが、それだけではまだ成功確定させません。
    plan50行目が要求する「profileで定義した post-30 screen event をtimeout内に
    確認したら成功」を、具体的には次の2つの確定的な画面変化に限定します
    (それ以外に target_reached 画面から遷移する現実的な行き先は無いため)。

    - ``GAMEPLAY`` へ戻ったことが確定した(post-30 confirm を押せてゲームが
      再開した) -> 即座に成功(``COMPLETE``)。
    - ``death``/``result`` へ直接移った(confirm 前に Reaper に倒された等) ->
      success_latched を優先して成功(``COMPLETE``、``_reduce_death_result``
      が最終確定する)。

    それ以外の確定的な画面変化(``UNKNOWN`` 等)や、post-30 event を確認
    できないままの timeout は、成功を確定させず fail-closed(campaign mode に
    応じて formal terminal failure か disarmed)に倒します(M7 fix: 以前は
    timeoutも未確認のconfirmed UNKNOWNも無条件でCOMPLETEにしていた)。

    confirmed LEVEL_UP/CHEST は modal screen 側の ``unexpected_ui_transition``
    と同じ「想定外の遷移」として即座に fail-closed にし、UNKNOWN/PAUSED
    (``window_focused=False`` による強制 UNKNOWN を含む)の間は confirm 待ちで
    あっても ``_dispatch_ui_click`` を一切呼ばず入力 0 のまま timeout を待ちます
    (M15 fix: 以前は confirmed UNKNOWN/PAUSED でも timeout の 5 秒間 Enter/click
    を送り続けており、plan18/96行目「unknown/paused/focus loss 中は入力 0」と
    受け入れ条件102行目「unknown はfail-closed」に反していた)。
    """
    if confirmed and category is ControllerState.DEATH_RESULT:
        return _enter_death_result(context, now_ns)
    if confirmed and category is ControllerState.GAMEPLAY:
        # post-30 confirm を確認できた(gameplay 再開が確定) -> 成功確定。
        return _finalize_terminal(context, ControllerState.COMPLETE, now_ns=now_ns, reason="post_30_confirm_success")
    if confirmed and category in (ControllerState.LEVEL_UP, ControllerState.CHEST):
        # modal screen(_reduce_modal_ui_screen)の unexpected_ui_transition と
        # 同じ「想定外の遷移」として即座に fail-closed にする(状態間の一貫性)。
        return _emergency_stop(context, now_ns=now_ns, reason=f"unexpected_ui_transition:{category.value}")

    timeout_ns = profile.timeout_ns_for("target_reached")
    if now_ns - context.state_entered_ns >= timeout_ns:
        # post-30 event を確認できないまま timeout: 成功を確定させず fail-closed。
        return _emergency_stop(context, now_ns=now_ns, reason="target_reached_timeout_unconfirmed")

    if category in (ControllerState.UNKNOWN, ControllerState.PAUSED):
        # UNKNOWN(focus loss含む)/PAUSED中はUI入力を出さず、timeoutまで入力0で待つ。
        return context, ()

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
    if category is ControllerState.TARGET_REACHED_PENDING_TRANSITION:
        if not confirmed:
            return context, ()
        # M7 fix: GAMEPLAY 直行の経路と同じ共通入口を通し、success_latched を
        # 一貫して立てる(UNKNOWN 経由でも 30:00 evidence の事実は変わらない)。
        return _enter_target_reached(context, now_ns)
    if category in (ControllerState.LEVEL_UP, ControllerState.CHEST):
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

        M9 fix: 以前は ``state``/``campaign_run_mode``/``run_identity``/
        ``pending_category``/``pending_streak`` しか初期化せず、
        ``success_latched``/``gameplay_entries``/``death_result_reset_emitted``/
        ``terminal_reason``/``ui_attempt`` が前 run から残ってしまっていました
        (非formalモードで emergency stop → 再arm した際、前 run の
        30:00到達成功や gameplay entry 済みという事実が新しい run に
        紛れ込む危険がありました)。ここでは ``profile`` 以外の文脈を
        完全に新しい ``StateContext``(既定値のみ)として作り直します。
        """
        if self.context.state is not ControllerState.DISARMED:
            raise ValueError("arm() requires the state machine to be DISARMED")
        self.context = StateContext(
            state=ControllerState.ACQUIRE_TARGET,
            campaign_run_mode=campaign_run_mode,
            run_identity=RunIdentity(run_id=run_id, gameplay_attempt_id=gameplay_attempt_id),
            state_entered_ns=now_ns,
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
