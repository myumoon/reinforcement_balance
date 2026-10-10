"""``state_machine.py`` の transition table・retry/timeout・terminal 確定を検証する。

やさしい説明:
このファイルは Survivors 実機操作の「交通整理役」が正しく振る舞うかを確認します。
- 画面が読み違えられても(1フレームの誤認識)暴走しないか(debounce)。
- レベルアップ中に移動入力が混ざらないか、クリックの順序は正しいか。
- クリックしたのに反応がないとき、諦めて安全側に倒れるか(fail-closed)。
- run の成功/失敗が二重に確定してしまわないか(compare-and-set)。
- 30:00 に到達した証拠を見たら、その後 何が起きても成功として扱われるか。
"""
from __future__ import annotations

import inspect
import random

import pytest
from reinbalance_survivors_contracts.ui_intent import UiIntentKind

from survivors.controller import state_machine
from survivors.controller.state_machine import (
    CampaignRunMode,
    ControllerState,
    StateContext,
    StateMachine,
    classify_screen_state,
    reduce,
)
from survivors.controller.ui_navigation import NavigationProfile
from survivors.perception_snapshot import NormalizedRoi

from . import _state_machine_fixtures as fx

_TICK_NS = 16_000_000  # 約 60fps 相当の1フレーム。


def _arm_to_gameplay(
    sm: StateMachine, now_ns: int, *, mode: CampaignRunMode = CampaignRunMode.OPERATOR_DEBUG_RESTART
) -> tuple[int, object]:
    """状態機械を arm して GAMEPLAY まで進める。

    通常プレイから始まる各テストに、同じ起動済み状態と時刻を用意します。
    """
    sm.arm(campaign_run_mode=mode, run_id="run-1", gameplay_attempt_id="attempt-1", now_ns=now_ns)
    gp = fx.make_perception_snapshot(screen_state="gameplay", snapshot_id="gp-boot", frame_id="fr-boot")
    for _ in range(6):
        now_ns += _TICK_NS
        sm.step(gp, fx.make_move_decision(gp), now_ns=now_ns)
    assert sm.context.state is ControllerState.GAMEPLAY
    return now_ns, gp


def _drive(sm: StateMachine, snapshot, decision, n: int, now_ns: int):
    """同じ観測と判断を指定 tick 数だけ流す。

    時刻を進め、最後に発生した effect と終了時刻を返します。
    """
    effects: tuple = ()
    for _ in range(n):
        now_ns += _TICK_NS
        effects = sm.step(snapshot, decision, now_ns=now_ns)
    return now_ns, effects


class TestImportBoundaries:
    """状態機械の依存境界を検証する。

    画素解析・選択 policy・OS 入力の内部へ制御ロジックが依存しないことを調べます。
    """

    def test_does_not_import_hud_or_vision_internals(self) -> None:
        """状態機械が vision 内部を import しない。

        観測は公開 snapshot を受け取り、HUD の構造へ直接アクセスしません。
        """
        source = inspect.getsource(state_machine)
        for forbidden in ("HudState", "hud_parser", "survivors.vision", "real_obs_assembler"):
            assert forbidden not in source

    def test_does_not_import_non_model_ui_policy_or_item_selector(self) -> None:
        """状態機械が item 選択の内部を import しない。

        選ぶ判断と画面遷移の制御を独立させます。
        """
        source = inspect.getsource(state_machine)
        for forbidden in ("NonModelUiPolicy", "ItemSelector", "decide_non_model_ui_intent"):
            assert forbidden not in source

    def test_does_not_reference_os_input_apis(self) -> None:
        # input driver は effect executor(ui_navigation.execute_effect)からしか
        # 呼ばれない: state_machine.py 自体は OS 入力 API を一切知らない。
        """状態機械が OS 入力 API を直接使わない。

        実入力の送信を effect の実行側に任せ、純粋な遷移処理を保ちます。
        """
        source = inspect.getsource(state_machine)
        for forbidden in ("SendInput", "pyautogui", "win32api", "ctypes", "keyboard"):
            assert forbidden not in source


class TestScreenClassification:
    """raw screen state を制御状態へ分類する。

    既知の画面、未知文字列、focus の喪失を対応する状態へ分けます。
    """

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("gameplay", ControllerState.GAMEPLAY),
            ("level_up_items", ControllerState.LEVEL_UP),
            ("level_up_fallback", ControllerState.LEVEL_UP),
            ("chest", ControllerState.CHEST),
            ("target_reached_transition", ControllerState.TARGET_REACHED_PENDING_TRANSITION),
            ("paused", ControllerState.PAUSED),
            ("death", ControllerState.DEATH_RESULT),
            ("result", ControllerState.DEATH_RESULT),
            ("unknown", ControllerState.UNKNOWN),
        ],
    )
    def test_known_screen_states(self, raw: str, expected: ControllerState) -> None:
        """既知画面が対応する controller state になる。

        カード・宝箱・終端などを遷移表で使う分類へ変換できます。
        """
        assert classify_screen_state(raw) is expected

    def test_unrecognized_string_is_unknown(self) -> None:
        """未知の画面名は UNKNOWN へ落とす。

        綴り違いを通常プレイとして扱わないことを確認します。
        """
        assert classify_screen_state("some_future_screen_name") is ControllerState.UNKNOWN

    def test_focus_loss_forces_unknown_even_for_gameplay(self) -> None:
        """focus が失われたら gameplay でも UNKNOWN にする。

        別ウィンドウへ移った状態でゲーム入力を続けません。
        """
        assert classify_screen_state("gameplay", window_focused=False) is ControllerState.UNKNOWN


class TestArmAndAcquireTarget:
    """明示 arm から通常プレイまでの起動を検証する。

    DISARMED の入力禁止と、観測を連続確認して開始する順序を調べます。
    """

    def test_disarmed_ignores_all_input_until_armed(self) -> None:
        """arm 前の DISARMED は入力を無視する。

        画面が認識できたことだけでは自動的に操作を始めません。
        """
        sm = StateMachine()
        gp = fx.make_perception_snapshot(screen_state="gameplay")
        effects = sm.step(gp, fx.make_move_decision(gp), now_ns=1_000_000_000)
        assert effects == ()
        assert sm.context.state is ControllerState.DISARMED

    def test_arm_requires_disarmed_state(self) -> None:
        """arm は DISARMED からだけ受け付ける。

        実行中の文脈を不意に上書きする再 arm を拒否します。
        """
        sm = StateMachine()
        sm.arm(campaign_run_mode=CampaignRunMode.OPERATOR_DEBUG_RESTART, run_id="r", gameplay_attempt_id="a", now_ns=0)
        with pytest.raises(ValueError):
            sm.arm(campaign_run_mode=CampaignRunMode.OPERATOR_DEBUG_RESTART, run_id="r", gameplay_attempt_id="a", now_ns=1)

    def test_three_frame_debounce_required_before_run_setup(self) -> None:
        """開始準備への遷移には三枚連続を要求する。

        一度だけ見えた画面で run を開始しないことを確認します。
        """
        sm = StateMachine()
        sm.arm(campaign_run_mode=CampaignRunMode.OPERATOR_DEBUG_RESTART, run_id="r", gameplay_attempt_id="a", now_ns=0)
        gp = fx.make_perception_snapshot(screen_state="gameplay")
        now = 0
        for i in range(2):
            now += _TICK_NS
            sm.step(gp, fx.make_move_decision(gp), now_ns=now)
            assert sm.context.state is ControllerState.ACQUIRE_TARGET
        now += _TICK_NS
        sm.step(gp, fx.make_move_decision(gp), now_ns=now)
        assert sm.context.state is ControllerState.RUN_SETUP

    def test_single_frame_flicker_does_not_confirm(self) -> None:
        """一枚の画面揺れを確定しない。

        次の frame が戻ったときは連続確認をやり直します。
        """
        sm = StateMachine()
        sm.arm(campaign_run_mode=CampaignRunMode.OPERATOR_DEBUG_RESTART, run_id="r", gameplay_attempt_id="a", now_ns=0)
        gp = fx.make_perception_snapshot(screen_state="gameplay")
        unk = fx.make_perception_snapshot(screen_state="unknown", snapshot_id="unk-flicker")
        now = 0
        # flicker(unk)の直後は streak がリセットされるため、その後2フレームでは
        # まだ3連続に届かず RUN_SETUP へ確定しない。
        for snap in (gp, gp, unk, gp, gp):
            now += _TICK_NS
            sm.step(snap, fx.make_move_decision(snap), now_ns=now)
        assert sm.context.state is ControllerState.ACQUIRE_TARGET
        # 3フレーム目でようやく確定する。
        now += _TICK_NS
        sm.step(gp, fx.make_move_decision(gp), now_ns=now)
        assert sm.context.state is ControllerState.RUN_SETUP

    def test_run_setup_timeout_fails_closed(self) -> None:
        """開始準備が進まなければ timeout で停止する。

        不明画面を待ち続ける正式 run を失敗として一度だけ確定します。
        """
        sm = StateMachine()
        sm.arm(campaign_run_mode=CampaignRunMode.FORMAL_SINGLE_ATTEMPT, run_id="r", gameplay_attempt_id="a", now_ns=0)
        stuck = fx.make_perception_snapshot(screen_state="unknown")
        now = 0
        for _ in range(2000):
            now += _TICK_NS
            sm.step(stuck, fx.make_no_op_decision(stuck), now_ns=now)
            if sm.context.terminal_locked:
                break
        assert sm.context.state is ControllerState.FORMAL_RUN_TERMINAL_FAILURE
        assert sm.context.terminal_locked


class TestGameplayMovement:
    """GAMEPLAY の移動判断を検証する。

    move は action を渡し、操作不要の判断からは入力を作りません。
    """

    def test_move_decision_forwards_action_index(self) -> None:
        """移動判断の action index を effect へ渡す。

        選ばれた方向を状態機械が別の番号に変換しないことを確認します。
        """
        sm = StateMachine()
        now, gp = _arm_to_gameplay(sm, 0)
        now += _TICK_NS
        effects = sm.step(gp, fx.make_move_decision(gp, action_index=5), now_ns=now)
        assert len(effects) == 1
        assert effects[0].kind == "move"
        assert effects[0].action_index == 5

    def test_no_op_decision_produces_no_effect(self) -> None:
        """no-op 判断では effect を作らない。

        入力不要の観測に移動や UI 操作が混ざりません。
        """
        sm = StateMachine()
        now, gp = _arm_to_gameplay(sm, 0)
        now += _TICK_NS
        effects = sm.step(gp, fx.make_no_op_decision(gp), now_ns=now)
        assert effects == ()


class TestLevelUpOrderingAndRetry:
    """レベルアップの解除・クリック・再送順序を検証する。

    選択画面では移動を止め、確認待ちと一回の retry を守ります。
    """

    def _enter_level_up(self, sm: StateMachine, now: int, *, choice_index: int = 0):
        """カード候補を持つ LEVEL_UP へ進める。

        初回クリック前の状態を揃え、各テストで順序や再送だけを変えます。
        """
        lu = fx.make_perception_snapshot(
            screen_state="level_up_items",
            snapshot_id="lu-0",
            frame_id="lu-f0",
            candidates=(
                fx.make_candidate_target(choice_id="card-0", choice_index=choice_index, confidence=0.995),
            ),
        )
        intent = fx.make_choose_card_intent(lu, target_index=choice_index)
        decision = fx.make_ui_decision(lu, intent)
        for _ in range(3):
            now += _TICK_NS
            effects = sm.step(lu, decision, now_ns=now)
        return now, lu, intent, decision, effects

    def test_movement_withheld_during_level_up(self) -> None:
        """LEVEL_UP 中は移動入力を送らない。

        UI の選択待ちに combat の action が混ざらないことを確認します。
        """
        sm = StateMachine()
        now, _gp = _arm_to_gameplay(sm, 0)
        now, lu, intent, decision, entry_effects = self._enter_level_up(sm, now)
        assert sm.context.state is ControllerState.LEVEL_UP
        # overall_review fix: GAMEPLAY を抜ける確定 tick 自体で release_all を
        # 1回発行する(movement lease の自然失効任せにしない)。まだ click は出ない。
        assert [e.kind for e in entry_effects] == ["release_all"]
        # 次の tick で click effect が出るが、move は一度も混ざらない。
        now += _TICK_NS
        effects = sm.step(lu, decision, now_ns=now)
        kinds = [effect.kind for effect in effects]
        assert "move" not in kinds

    def test_click_ordering_release_before_click_and_resume_after_ack(self) -> None:
        """解除を先に送り、適用確認後に移動を再開する。

        選択前の held input を外し、通常プレイが確定するまで再移動しません。
        """
        sm = StateMachine()
        now, gp = _arm_to_gameplay(sm, 0)
        now, lu, intent, decision, _ = self._enter_level_up(sm, now)
        now += _TICK_NS
        effects = sm.step(lu, decision, now_ns=now)
        # M3: release_all が choice click より先に、同じ tick でペアで出る。
        assert [effect.kind for effect in effects] == ["release_all", "ui_click"]
        assert effects[1].target.choice_id == "card-0"

        # ack が来るまで movement は一切出ない。
        for _ in range(5):
            now += _TICK_NS
            stale_effects = sm.step(lu, decision, now_ns=now)
            assert all(effect.kind not in ("move",) for effect in stale_effects)

        # ack: 画面が gameplay へ確定して初めて movement を再開してよい。
        gp2 = fx.make_perception_snapshot(screen_state="gameplay", snapshot_id="gp-ack", frame_id="fr-ack")
        for _ in range(2):
            now += _TICK_NS
            sm.step(gp2, fx.make_move_decision(gp2), now_ns=now)
        assert sm.context.state is ControllerState.LEVEL_UP  # まだ debounce 中。
        now += _TICK_NS
        effects = sm.step(gp2, fx.make_move_decision(gp2), now_ns=now)
        assert sm.context.state is ControllerState.GAMEPLAY
        now += _TICK_NS
        effects = sm.step(gp2, fx.make_move_decision(gp2, action_index=2), now_ns=now)
        assert [effect.kind for effect in effects] == ["move"]

    def test_duplicate_frame_and_decision_does_not_double_click(self) -> None:
        """同一観測と判断の重複で二重クリックしない。

        新しい frame を得ずに何度 step しても追加送信はありません。
        """
        sm = StateMachine()
        now, gp = _arm_to_gameplay(sm, 0)
        now, lu, intent, decision, _ = self._enter_level_up(sm, now)
        now += _TICK_NS
        first = sm.step(lu, decision, now_ns=now)
        assert first[1].kind == "ui_click"
        # 全く同じ snapshot/decision を何度repeatedly送っても再送しない。
        for _ in range(10):
            now += _TICK_NS
            effects = sm.step(lu, decision, now_ns=now)
            assert effects == ()

    def test_retry_withheld_within_ack_wait_window(self) -> None:
        # M4 fix: 初回 click 直後、ack 待ち window(既定 200ms)が経過するまでは
        # newer snapshot で precondition を満たしていても retry してはならない
        # (capture/perception 遅延中の二重click防止)。
        """確認待ち時間内は retry を送らない。

        撮影や反映の遅延がある間に二重クリックすることを防ぎます。
        """
        sm = StateMachine()
        now, gp = _arm_to_gameplay(sm, 0)
        now, lu, intent, decision, _ = self._enter_level_up(sm, now)
        now += _TICK_NS
        sm.step(lu, decision, now_ns=now)
        sent_ns = now

        jittered = fx.make_perception_snapshot(
            screen_state="level_up_items",
            snapshot_id="lu-1",
            frame_id="lu-f1",
            captured_ns=lu.captured_ns + 1,
            ui_state_key=lu.ui_state_key,
            candidate_set_hash=lu.ui_presentation.candidate_set_hash,
            candidates=(
                fx.make_candidate_target(
                    choice_id="card-0",
                    choice_index=0,
                    roi=NormalizedRoi(0.101, 0.101, 0.201, 0.201),
                    confidence=0.995,
                ),
            ),
        )
        retry_intent = fx.make_choose_card_intent(
            jittered, target_index=0, candidate_set_hash=lu.ui_presentation.candidate_set_hash
        )
        # 16ms後(1フレーム後)、ack待ちwindow(200ms)にはまだ全く届かない。
        now = sent_ns + _TICK_NS
        effects = sm.step(jittered, fx.make_ui_decision(jittered, retry_intent), now_ns=now)
        assert effects == ()
        assert sm.context.ui_attempt.retried is False

    def test_retry_allowed_once_after_ack_wait_window_with_small_roi_jitter(self) -> None:
        """確認待ち後の小さい矩形揺れには一回だけ再送する。

        同じカードと認められる新観測でも、三回目のクリックは許しません。
        """
        sm = StateMachine()
        now, gp = _arm_to_gameplay(sm, 0)
        now, lu, intent, decision, _ = self._enter_level_up(sm, now)
        now += _TICK_NS
        sm.step(lu, decision, now_ns=now)
        sent_ns = now

        jittered = fx.make_perception_snapshot(
            screen_state="level_up_items",
            snapshot_id="lu-1",
            frame_id="lu-f1",
            captured_ns=lu.captured_ns + 1,
            ui_state_key=lu.ui_state_key,
            candidate_set_hash=lu.ui_presentation.candidate_set_hash,
            candidates=(
                fx.make_candidate_target(
                    choice_id="card-0",
                    choice_index=0,
                    roi=NormalizedRoi(0.101, 0.101, 0.201, 0.201),
                    confidence=0.995,
                ),
            ),
        )
        retry_intent = fx.make_choose_card_intent(
            jittered, target_index=0, candidate_set_hash=lu.ui_presentation.candidate_set_hash
        )
        # ack待ちwindow(既定200ms)を確実に超えてから retry を送る。
        now = sent_ns + 250_000_000
        effects = sm.step(jittered, fx.make_ui_decision(jittered, retry_intent), now_ns=now)
        assert [effect.kind for effect in effects] == ["release_all", "ui_click"]
        assert effects[1].mode == "retry"
        assert sm.context.ui_attempt.retried is True

        # 2回目の retry は許されない(budget=1)。
        jittered2 = fx.make_perception_snapshot(
            screen_state="level_up_items",
            snapshot_id="lu-2",
            frame_id="lu-f2",
            captured_ns=jittered.captured_ns + 1,
            ui_state_key=lu.ui_state_key,
            candidate_set_hash=lu.ui_presentation.candidate_set_hash,
            candidates=(
                fx.make_candidate_target(
                    choice_id="card-0", choice_index=0,
                    roi=NormalizedRoi(0.102, 0.102, 0.202, 0.202), confidence=0.995,
                ),
            ),
        )
        retry_intent2 = fx.make_choose_card_intent(
            jittered2, target_index=0, candidate_set_hash=lu.ui_presentation.candidate_set_hash
        )
        now += 250_000_000
        effects = sm.step(jittered2, fx.make_ui_decision(jittered2, retry_intent2), now_ns=now)
        assert effects == ()

    def test_choice_count_change_mid_attempt_fails_closed(self) -> None:
        """試行中に候補枚数が変わったら停止する。

        古い選択番号を入れ替わったカード集合へ適用しません。
        """
        sm = StateMachine()
        now, gp = _arm_to_gameplay(sm, 0)
        now, lu, intent, decision, _ = self._enter_level_up(sm, now)
        now += _TICK_NS
        sm.step(lu, decision, now_ns=now)

        changed = fx.make_perception_snapshot(
            screen_state="level_up_items",
            snapshot_id="lu-changed",
            frame_id="lu-fc",
            ui_state_key=lu.ui_state_key,
            candidate_set_hash=lu.ui_presentation.candidate_set_hash,
            candidates=(fx.make_candidate_target(choice_id="card-9", choice_index=1),),
        )
        changed_intent = fx.make_choose_card_intent(
            changed, target_index=1, candidate_set_hash=lu.ui_presentation.candidate_set_hash
        )
        now += _TICK_NS
        effects = sm.step(changed, fx.make_ui_decision(changed, changed_intent), now_ns=now)
        assert sm.context.state is ControllerState.DISARMED
        assert effects and effects[0].kind == "release_all"

    def test_click_miss_zero_candidates_eventually_fails_closed(self) -> None:
        """クリック後に候補が読めなくなると最終的に停止する。

        有効な target を推測で補わず、待ち時間の上限を守ります。
        """
        sm = StateMachine()
        now, gp = _arm_to_gameplay(sm, 0, mode=CampaignRunMode.FORMAL_SINGLE_ATTEMPT)
        lu_empty = fx.make_perception_snapshot(screen_state="level_up_items", snapshot_id="lu-empty", candidates=())
        intent = fx.make_choose_card_intent(lu_empty, target_index=0)
        decision = fx.make_ui_decision(lu_empty, intent)
        for _ in range(3):
            now += _TICK_NS
            sm.step(lu_empty, decision, now_ns=now)
        assert sm.context.state is ControllerState.LEVEL_UP
        for _ in range(1000):
            now += _TICK_NS
            sm.step(lu_empty, decision, now_ns=now)
            if sm.context.terminal_locked:
                break
        assert sm.context.state is ControllerState.FORMAL_RUN_TERMINAL_FAILURE

    def test_stuck_level_up_after_timeout_fails_closed_non_formal(self) -> None:
        """進まない level-up は debug mode でも停止する。

        候補なしで待つ画面に移動を送り続けず、DISARMED へ戻します。
        """
        sm = StateMachine()
        now, gp = _arm_to_gameplay(sm, 0, mode=CampaignRunMode.OPERATOR_DEBUG_RESTART)
        lu_empty = fx.make_perception_snapshot(screen_state="level_up_items", snapshot_id="lu-stuck", candidates=())
        decision = fx.make_no_op_decision(lu_empty)
        for _ in range(3):
            now += _TICK_NS
            sm.step(lu_empty, decision, now_ns=now)
        for _ in range(1000):
            now += _TICK_NS
            effects = sm.step(lu_empty, decision, now_ns=now)
            if sm.context.state is ControllerState.DISARMED:
                break
        assert sm.context.state is ControllerState.DISARMED
        assert effects == (effects[0],) and effects[0].kind == "release_all"

    def test_unexpected_direct_transition_between_modal_screens_fails_closed(self) -> None:
        """カードから宝箱への不意の modal 遷移を停止する。

        予定外の UI を同じクリック試行の続きとして扱いません。
        """
        sm = StateMachine()
        now, gp = _arm_to_gameplay(sm, 0)
        now, lu, intent, decision, _ = self._enter_level_up(sm, now)
        chest = fx.make_perception_snapshot(screen_state="chest", snapshot_id="chest-surprise")
        for _ in range(3):
            now += _TICK_NS
            sm.step(chest, fx.make_no_op_decision(chest), now_ns=now)
        assert sm.context.state is ControllerState.DISARMED


class TestApplyAckRecognition:
    """M12/M15: apply ack(ui_state_key/candidate_set_hash/inventory_hash の変化)を

    intent identity の変化と誤認しない。ただし debounce_frames 連続で安定し、
    かつ retry_after_ns 経過するまでは apply ack と確定しない(M15 fix: 1フレーム
    だけの揺れで無制限に re-click しないように安定条件を必須にした)。
    """

    @staticmethod
    def _drive_until_ack_confirmed(sm: StateMachine, snapshot, decision, now: int) -> tuple[int, tuple]:
        """同じ (snapshot, decision) を、apply ack が確定するまで送り続ける。

        やさしい説明: debounce_frames 連続の安定観測と retry_after_ns 経過の
        両方を満たすまで、テスト側で機械的に tick を送り進めるヘルパーです。
        """
        effects: tuple = ()
        for _ in range(200):
            now += _TICK_NS
            effects = sm.step(snapshot, decision, now_ns=now)
            if effects or sm.context.ui_attempt is None:
                return now, effects
        pytest.fail("apply ack (stabilized + ack window elapsed) was never confirmed")
        return now, effects

    def _enter_level_up_with_candidate(
        self, sm: StateMachine, now: int, *, snapshot_id: str, choice_id: str, choice_index: int = 0
    ):
        """指定カードで LEVEL_UP の試行を始める。

        候補集合の変化を調べるテストに、初回操作の観測を用意します。
        """
        lu = fx.make_perception_snapshot(
            screen_state="level_up_items",
            snapshot_id=snapshot_id,
            frame_id=f"{snapshot_id}-f",
            candidates=(
                fx.make_candidate_target(choice_id=choice_id, choice_index=choice_index, confidence=0.995),
            ),
        )
        intent = fx.make_choose_card_intent(lu, target_index=choice_index)
        decision = fx.make_ui_decision(lu, intent)
        return lu, intent, decision

    def test_ui_state_key_jitter_alone_does_not_resend(self) -> None:
        """M15: candidate_set_hash/inventory_hash は不変のまま ui_state_key だけが

        A/B と揺れても、intent identity は変わらないので通常の retry 規則
        (initial 1 + retry 最大1 = 合計2)しか送らない(無制限 re-click しない)。
        """
        sm = StateMachine()
        now, gp = _arm_to_gameplay(sm, 0, mode=CampaignRunMode.FORMAL_SINGLE_ATTEMPT)
        lu, intent, decision = self._enter_level_up_with_candidate(sm, now, snapshot_id="lu-a", choice_id="card-a")
        for _ in range(3):
            now += _TICK_NS
            sm.step(lu, decision, now_ns=now)
        assert sm.context.state is ControllerState.LEVEL_UP
        now += _TICK_NS
        first_click = sm.step(lu, decision, now_ns=now)
        assert [e.kind for e in first_click] == ["release_all", "ui_click"]

        # candidate_set_hash/inventory_hash を固定したまま ui_state_key だけを
        # A/B と交互に変えた snapshot を 16ms 間隔で 20 tick 与える
        # (blocking-findings-2.json の再現手順そのもの)。
        lu_a = lu
        lu_b = fx.make_perception_snapshot(
            screen_state="level_up_items",
            snapshot_id=lu.snapshot_id,
            frame_id=lu.frame_id,
            ui_state_key=fx._hash_of("ui-state-key-B"),
            candidates=lu.ui_presentation.candidates,
            candidate_set_hash=lu.ui_presentation.candidate_set_hash,
            inventory_hash=lu.ui_presentation.inventory_hash,
        )
        click_count = 0
        retry_seen = False
        for tick in range(20):
            wobbling = lu_a if tick % 2 == 0 else lu_b
            now += _TICK_NS
            effects = sm.step(wobbling, decision, now_ns=now)
            if effects:
                assert [e.kind for e in effects] == ["release_all", "ui_click"]
                click_count += 1
                if effects[1].mode == "retry":
                    retry_seen = True
        # 揺れている間、initial 1 + retry 最大1 = 合計2件までしか click しない。
        assert click_count <= 1

    def test_ui_state_key_oscillation_with_fresh_snapshots_never_exceeds_visit_budget(self) -> None:
        """M4 再発防止: 毎tick新しいsnapshot_id・単調増加するcaptured_ns(実機の

        実フレームと同じ)を持つui_state_keyのA/B交互揺れを120tick(約2秒、
        level_up timeout未満)与えても、candidate_set_hash/target_indexが不変な
        限りintent identityは変わらないため、``ui_visit_click_count``が2を
        超えない。iteration2で見つかった『飛び飛びの観測でstreakが誤って
        積み上がり、約208msごとに再クリックし続ける』回帰の直接の再現テスト
        (同じsnapshot_idを使い回す既存のjitterテストでは、この経路は
        retry自体が別の理由でブロックされるため再現できなかった)。
        """
        sm = StateMachine()
        now, gp = _arm_to_gameplay(sm, 0, mode=CampaignRunMode.FORMAL_SINGLE_ATTEMPT)
        lu, intent, decision = self._enter_level_up_with_candidate(sm, now, snapshot_id="lu-osc-0", choice_id="card-a")
        for _ in range(3):
            now += _TICK_NS
            sm.step(lu, decision, now_ns=now)
        assert sm.context.state is ControllerState.LEVEL_UP

        key_a = lu.ui_state_key
        key_b = fx._hash_of("ui-state-key-B")

        def _snapshot(tick: int, key: str, captured_ns: int):
            """揺れる UI key の新しい snapshot を作る。

            frame ごとに identity と撮影時刻を進め、重複観測とは違う揺れを再現します。
            """
            return fx.make_perception_snapshot(
                screen_state="level_up_items",
                snapshot_id=f"lu-osc-{tick}",
                frame_id=f"lu-osc-{tick}-f",
                captured_ns=captured_ns,
                ui_state_key=key,
                candidates=lu.ui_presentation.candidates,
                candidate_set_hash=lu.ui_presentation.candidate_set_hash,
                inventory_hash=lu.ui_presentation.inventory_hash,
            )

        for tick in range(1, 121):
            key = key_a if tick % 2 == 0 else key_b
            now += _TICK_NS
            snapshot = _snapshot(tick, key, now)
            tick_intent = fx.make_choose_card_intent(snapshot, target_index=0)
            sm.step(snapshot, fx.make_ui_decision(snapshot, tick_intent), now_ns=now)
            assert sm.context.ui_visit_click_count <= 2, (
                f"tick={tick}: ui_visit_click_count={sm.context.ui_visit_click_count} が2を超えた"
            )
        assert sm.context.state is ControllerState.LEVEL_UP
        assert sm.context.state is ControllerState.LEVEL_UP
        assert not sm.context.terminal_locked

    def test_reroll_then_choose_new_candidate_is_not_emergency_stop(self) -> None:
        """リロール後の新候補選択は停止せず進める。

        確定した候補集合の変更を新しい選択試行として受け付けます。
        """
        sm = StateMachine()
        now, gp = _arm_to_gameplay(sm, 0, mode=CampaignRunMode.FORMAL_SINGLE_ATTEMPT)
        lu, intent, decision = self._enter_level_up_with_candidate(sm, now, snapshot_id="lu-a", choice_id="card-a")
        for _ in range(3):
            now += _TICK_NS
            sm.step(lu, decision, now_ns=now)
        assert sm.context.state is ControllerState.LEVEL_UP
        now += _TICK_NS
        first_click = sm.step(lu, decision, now_ns=now)
        assert [e.kind for e in first_click] == ["release_all", "ui_click"]

        # reroll成功でゲーム側が候補集合を入れ替えた(candidate_set_hash変化)。
        rerolled, _reroll_intent, reroll_decision = self._enter_level_up_with_candidate(
            sm, now, snapshot_id="lu-rerolled", choice_id="card-z"
        )
        # 変化した直後の1tickだけでは apply ack と確定しない
        # (debounce_frames 連続 + retry_after_ns 経過が必要、M15 fix)。
        now += _TICK_NS
        immediate_effects = sm.step(rerolled, reroll_decision, now_ns=now)
        assert immediate_effects == ()
        assert sm.context.state is ControllerState.LEVEL_UP
        assert sm.context.ui_attempt is not None

        now, effects = self._drive_until_ack_confirmed(sm, rerolled, reroll_decision, now)
        # 安定条件を満たして apply ack が確定し、intent identity 変化とみなして
        # emergency stop しない。
        assert sm.context.state is ControllerState.LEVEL_UP
        assert not sm.context.terminal_locked
        assert [e.kind for e in effects] == ["release_all", "ui_click"]
        assert effects[1].mode == "initial"
        assert effects[1].target.choice_id == "card-z"

    def test_consecutive_level_up_after_apply_ack_is_new_initial_click(self) -> None:
        """適用確認後の次の level-up は新しい初回操作になる。

        連続した訪問でも前の retry と取り違えません。
        """
        sm = StateMachine()
        now, gp = _arm_to_gameplay(sm, 0, mode=CampaignRunMode.OPERATOR_DEBUG_RESTART)
        lu, _intent, decision = self._enter_level_up_with_candidate(sm, now, snapshot_id="lu-1st", choice_id="knife")
        for _ in range(3):
            now += _TICK_NS
            sm.step(lu, decision, now_ns=now)
        now += _TICK_NS
        sm.step(lu, decision, now_ns=now)  # 1回目の選択を送信。

        # 選択が適用され、続けて2回目の level-up が提示された(連続 level-up)。
        lu2, _intent2, decision2 = self._enter_level_up_with_candidate(
            sm, now, snapshot_id="lu-2nd", choice_id="shield", choice_index=2
        )
        now, effects = self._drive_until_ack_confirmed(sm, lu2, decision2, now)
        assert sm.context.state is ControllerState.LEVEL_UP
        assert sm.context.state is not ControllerState.DISARMED
        assert [e.kind for e in effects] == ["release_all", "ui_click"]
        assert effects[1].target.choice_id == "shield"

    def test_apply_ack_with_stale_intent_does_not_resend(self) -> None:
        # 今tickのintentが古いsnapshot(lu基準)に束縛されたまま(遅延で旧intentが
        # 届いた等)で、現在のsnapshotは既にreroll済み(rerolled)の場合、intent
        # identityは元のattemptと同じ(どちらもlu/card-b基準)なのでノイズ扱いと
        # なり、apply ack候補にもretryにもならず、resendが一度も起きない
        # (M4 fix: identityが変わらない限りfall throughせず待つ)。
        # `_drive_until_ack_confirmed`は「effectsが空でなくなるまで回す」設計
        # なので、いずれ level_up timeout の fail-closed effect を拾って
        # しまい「停滞し続ける」ことを検証できない。有限tick数で
        # ui_click/ui_keyが一度も出ないことを直接確認する。
        """適用確認後に届いた古い intent を再送しない。

        source binding の違う判断を新しい画面に使わないことを確認します。
        """
        sm = StateMachine()
        now, gp = _arm_to_gameplay(sm, 0)
        lu, _intent, decision = self._enter_level_up_with_candidate(sm, now, snapshot_id="lu-b", choice_id="card-b")
        for _ in range(3):
            now += _TICK_NS
            sm.step(lu, decision, now_ns=now)
        now += _TICK_NS
        sm.step(lu, decision, now_ns=now)

        rerolled, _r, _rd = self._enter_level_up_with_candidate(sm, now, snapshot_id="lu-b-rerolled", choice_id="card-c")
        stale_decision = decision  # 古い intent(lu 基準)のまま。
        for _ in range(20):
            now += _TICK_NS
            effects = sm.step(rerolled, stale_decision, now_ns=now)
            kinds = [e.kind for e in effects]
            assert "ui_click" not in kinds and "ui_key" not in kinds
        assert sm.context.ui_attempt is not None
        assert sm.context.state is ControllerState.LEVEL_UP
        assert not sm.context.terminal_locked


class TestChestAndConfirmButtons:
    """宝箱終了と目標到達の確認操作を検証する。

    ack_chest はボタン矩形、confirm は Enter に解決されます。
    """

    def test_chest_click_uses_button_target(self) -> None:
        """宝箱クリックは観測されたボタン target を使う。

        入力解除の後に ack_chest の矩形へ初回操作を送ります。
        """
        sm = StateMachine()
        now, gp = _arm_to_gameplay(sm, 0)
        chest = fx.make_perception_snapshot(
            screen_state="chest", snapshot_id="chest-0",
            buttons=(fx.make_button_target(semantic_action="ack_chest"),),
        )
        intent = fx.make_button_intent(chest, kind=UiIntentKind.ACK_CHEST, semantic_action="ack_chest")
        decision = fx.make_ui_decision(chest, intent)
        for _ in range(3):
            now += _TICK_NS
            sm.step(chest, decision, now_ns=now)
        now += _TICK_NS
        effects = sm.step(chest, decision, now_ns=now)
        assert [e.kind for e in effects] == ["release_all", "ui_click"]

    def test_confirm_uses_enter_key_not_click(self) -> None:
        """目標確認は Enter key を送る。

        confirm の矩形があっても座標クリックとして実行しません。
        """
        sm = StateMachine()
        now, gp = _arm_to_gameplay(sm, 0)
        tr = fx.make_perception_snapshot(
            screen_state="target_reached_transition", snapshot_id="tr-0",
            buttons=(fx.make_button_target(semantic_action="confirm"),),
        )
        intent = fx.make_button_intent(tr, kind=UiIntentKind.CONFIRM, semantic_action="confirm")
        decision = fx.make_ui_decision(tr, intent)
        for _ in range(3):
            now += _TICK_NS
            sm.step(tr, decision, now_ns=now)
        now += _TICK_NS
        effects = sm.step(tr, decision, now_ns=now)
        assert [e.kind for e in effects] == ["release_all", "ui_key"]
        assert effects[1].key == "ENTER"


class TestTargetReachedSuccessPriority:
    """三十分到達後の成功判定を優先する。

    その後に死亡や結果へ進んでも到達証拠を失わないことを確認します。
    """

    def _reach_target(self, sm: StateMachine, now: int):
        """目標到達画面へ状態機械を進める。

        post-30 の証拠を揃え、終端判定を比較できる状態にします。
        """
        gp = fx.make_perception_snapshot(screen_state="gameplay", snapshot_id="gp-reach")
        for _ in range(6):
            now += _TICK_NS
            sm.step(gp, fx.make_move_decision(gp), now_ns=now)
        tr = fx.make_perception_snapshot(screen_state="target_reached_transition", snapshot_id="tr-reach", buttons=())
        for _ in range(3):
            now += _TICK_NS
            sm.step(tr, fx.make_no_op_decision(tr), now_ns=now)
        assert sm.context.state is ControllerState.TARGET_REACHED_PENDING_TRANSITION
        assert sm.context.success_latched is True
        return now, tr

    def test_death_after_target_reached_still_completes(self) -> None:
        """目標到達後の死亡は成功へ収束する。

        到達済みの run を後から失敗へ上書きしません。
        """
        sm = StateMachine()
        sm.arm(campaign_run_mode=CampaignRunMode.FORMAL_SINGLE_ATTEMPT, run_id="r", gameplay_attempt_id="a", now_ns=0)
        now, _tr = self._reach_target(sm, 0)
        death = fx.make_perception_snapshot(screen_state="death", snapshot_id="death-after-reach")
        for _ in range(1000):
            now += _TICK_NS
            sm.step(death, fx.make_no_op_decision(death), now_ns=now)
            if sm.context.terminal_locked:
                break
        assert sm.context.state is ControllerState.COMPLETE

    def test_result_after_target_reached_completes(self) -> None:
        """目標到達後の結果画面で成功を確定する。

        終端の表示に変わっても成功の証拠を保持します。
        """
        sm = StateMachine()
        sm.arm(campaign_run_mode=CampaignRunMode.OPERATOR_DEBUG_RESTART, run_id="r", gameplay_attempt_id="a", now_ns=0)
        now, _tr = self._reach_target(sm, 0)
        result = fx.make_perception_snapshot(screen_state="result", snapshot_id="result-after-reach")
        for _ in range(1000):
            now += _TICK_NS
            sm.step(result, fx.make_no_op_decision(result), now_ns=now)
            if sm.context.terminal_locked:
                break
        assert sm.context.state is ControllerState.COMPLETE

    def test_target_reached_timeout_without_confirmation_fails_closed_debug(self) -> None:
        # M7(b) fix: post-30 event(gameplay 再開 or death/result)を確認できない
        # まま timeout した場合は、以前のように無条件で COMPLETE にはしない。
        """確認なしの目標画面 timeout は debug mode で停止する。

        時間切れを成功の代用にせず、入力を解放して DISARMED へ戻します。
        """
        sm = StateMachine()
        sm.arm(campaign_run_mode=CampaignRunMode.OPERATOR_DEBUG_RESTART, run_id="r", gameplay_attempt_id="a", now_ns=0)
        now, tr = self._reach_target(sm, 0)
        for _ in range(2000):
            now += _TICK_NS
            sm.step(tr, fx.make_no_op_decision(tr), now_ns=now)
            if sm.context.state is not ControllerState.TARGET_REACHED_PENDING_TRANSITION:
                break
        assert sm.context.state is ControllerState.DISARMED
        assert not sm.context.terminal_locked

    def test_target_reached_timeout_without_confirmation_fails_closed_formal(self) -> None:
        """確認なしの目標画面 timeout は正式失敗になる。

        正式 run の成功を不確かな画面継続だけで決めません。
        """
        sm = StateMachine()
        sm.arm(campaign_run_mode=CampaignRunMode.FORMAL_SINGLE_ATTEMPT, run_id="r", gameplay_attempt_id="a", now_ns=0)
        now, tr = self._reach_target(sm, 0)
        for _ in range(2000):
            now += _TICK_NS
            sm.step(tr, fx.make_no_op_decision(tr), now_ns=now)
            if sm.context.terminal_locked:
                break
        assert sm.context.state is ControllerState.FORMAL_RUN_TERMINAL_FAILURE

    def test_target_reached_confirmed_unknown_does_not_auto_complete(self) -> None:
        # M7(b) fix: TARGET_REACHED -> UNKNOWN は illegal transition であり、
        # 以前のように「target_reached 以外へ確定的に移った」というだけで
        # 無条件に COMPLETE にしてはならない。
        """目標後に UNKNOWN が確定しても成功と断定しない。

        未認識の画面変化は post-30 の適用確認になりません。
        """
        sm = StateMachine()
        sm.arm(campaign_run_mode=CampaignRunMode.OPERATOR_DEBUG_RESTART, run_id="r", gameplay_attempt_id="a", now_ns=0)
        now, tr = self._reach_target(sm, 0)
        unk = fx.make_perception_snapshot(screen_state="totally_unrecognized_noise", snapshot_id="tr-to-unk")
        for _ in range(3):
            now += _TICK_NS
            sm.step(unk, fx.make_no_op_decision(unk), now_ns=now)
        assert sm.context.state is ControllerState.TARGET_REACHED_PENDING_TRANSITION
        assert not sm.context.terminal_locked

    def test_target_reached_latch_set_when_entered_via_unknown(self) -> None:
        # M7(a) fix: GAMEPLAY -> 一瞬 UNKNOWN(画面切替中のノイズ)-> TARGET_REACHED
        # という経路でも success_latched が一貫して立つ(以前は
        # _reduce_unknown 経由だけ latch が立たず、直後の death が誤って
        # pre-30 failure 扱いになっていた)。
        """UNKNOWN 経由の目標到達でも成功記録を立てる。

        一時的な読み違いを挟んでも到達の証拠が反映されます。
        """
        sm = StateMachine()
        sm.arm(campaign_run_mode=CampaignRunMode.FORMAL_SINGLE_ATTEMPT, run_id="r", gameplay_attempt_id="a", now_ns=0)
        now = 0
        gp = fx.make_perception_snapshot(screen_state="gameplay", snapshot_id="gp-tr-via-unknown")
        for _ in range(6):
            now += _TICK_NS
            sm.step(gp, fx.make_move_decision(gp), now_ns=now)
        noise = fx.make_perception_snapshot(screen_state="totally_unrecognized_noise", snapshot_id="noise-before-tr")
        for _ in range(3):
            now += _TICK_NS
            sm.step(noise, fx.make_no_op_decision(noise), now_ns=now)
        assert sm.context.state is ControllerState.UNKNOWN
        tr = fx.make_perception_snapshot(
            screen_state="target_reached_transition", snapshot_id="tr-via-unknown", buttons=()
        )
        for _ in range(3):
            now += _TICK_NS
            sm.step(tr, fx.make_no_op_decision(tr), now_ns=now)
        assert sm.context.state is ControllerState.TARGET_REACHED_PENDING_TRANSITION
        assert sm.context.success_latched is True

        death = fx.make_perception_snapshot(screen_state="death", snapshot_id="death-after-tr-via-unknown")
        for _ in range(1000):
            now += _TICK_NS
            sm.step(death, fx.make_no_op_decision(death), now_ns=now)
            if sm.context.terminal_locked:
                break
        assert sm.context.state is ControllerState.COMPLETE

    def test_pre_target_death_is_failure_in_formal_mode(self) -> None:
        """目標前の死亡は正式 run の失敗になる。

        三十分の証拠がない死亡を成功として確定しません。
        """
        sm = StateMachine()
        now, gp = _arm_to_gameplay(sm, 0, mode=CampaignRunMode.FORMAL_SINGLE_ATTEMPT)
        death = fx.make_perception_snapshot(screen_state="death", snapshot_id="death-pre-30")
        for _ in range(1000):
            now += _TICK_NS
            sm.step(death, fx.make_no_op_decision(death), now_ns=now)
            if sm.context.terminal_locked:
                break
        assert sm.context.state is ControllerState.FORMAL_RUN_TERMINAL_FAILURE

    def test_pre_target_death_restarts_in_operator_debug_mode(self) -> None:
        """目標前の死亡は debug mode の再開始経路へ進む。

        正式失敗の終了処理と操作者向けの再準備を区別します。
        """
        sm = StateMachine()
        now, gp = _arm_to_gameplay(sm, 0, mode=CampaignRunMode.OPERATOR_DEBUG_RESTART)
        death = fx.make_perception_snapshot(screen_state="death", snapshot_id="death-pre-30-restart")
        for _ in range(1000):
            now += _TICK_NS
            sm.step(death, fx.make_no_op_decision(death), now_ns=now)
            if sm.context.state is ControllerState.RUN_SETUP:
                break
        assert sm.context.state is ControllerState.RUN_SETUP
        assert not sm.context.terminal_locked
        gp2 = fx.make_perception_snapshot(screen_state="gameplay", snapshot_id="gp-2nd")
        for _ in range(6):
            now += _TICK_NS
            sm.step(gp2, fx.make_move_decision(gp2), now_ns=now)
        assert sm.context.state is ControllerState.GAMEPLAY
        assert sm.context.gameplay_entries == 2


class TestCombatResetExactlyOnce:
    """死亡・結果へ入るたび combat reset を一回だけ出す。

    同じ終端画面の連続観測で状態リセットが繰り返されないことを確認します。
    """

    def test_combat_reset_emitted_once_per_death_result_entry(self) -> None:
        """一つの終端訪問につき reset を一回だけ送る。

        別の訪問では再び reset でき、同じ訪問内では重複しません。
        """
        sm = StateMachine()
        now, gp = _arm_to_gameplay(sm, 0, mode=CampaignRunMode.OPERATOR_DEBUG_RESTART)
        death = fx.make_perception_snapshot(screen_state="death", snapshot_id="death-reset")
        reset_count = 0
        for _ in range(20):
            now += _TICK_NS
            effects = sm.step(death, fx.make_no_op_decision(death), now_ns=now)
            reset_count += sum(1 for e in effects if e.kind == "combat_reset")
            if sm.context.state is ControllerState.RUN_SETUP:
                break
        assert reset_count == 1


class TestFormalTerminalOrderingAndCas:
    """正式終端の順序と一度だけの確定を検証する。

    入力解除・停止・プロセス終了を順に出し、二回目の gameplay 開始を拒否します。
    """

    def test_terminal_effects_are_release_then_stop_then_terminate(self) -> None:
        """正式終了は解除・停止・終了の順になる。

        キーが押されたままプロセスだけを終了させません。
        """
        sm = StateMachine()
        now, gp = _arm_to_gameplay(sm, 0, mode=CampaignRunMode.FORMAL_SINGLE_ATTEMPT)
        death = fx.make_perception_snapshot(screen_state="death", snapshot_id="death-order")
        effects: tuple = ()
        for _ in range(1000):
            now += _TICK_NS
            effects = sm.step(death, fx.make_no_op_decision(death), now_ns=now)
            if sm.context.terminal_locked:
                break
        assert [e.kind for e in effects] == ["release_all", "controller_stop", "process_terminate"]

    def test_terminal_is_compare_and_set_exactly_once(self) -> None:
        """正式終端は一度しか確定しない。

        後の frame や別の終端理由が確定済みの結果を上書きできません。
        """
        sm = StateMachine()
        now, gp = _arm_to_gameplay(sm, 0, mode=CampaignRunMode.FORMAL_SINGLE_ATTEMPT)
        death = fx.make_perception_snapshot(screen_state="death", snapshot_id="death-cas")
        for _ in range(1000):
            now += _TICK_NS
            sm.step(death, fx.make_no_op_decision(death), now_ns=now)
            if sm.context.terminal_locked:
                break
        locked_state = sm.context.state
        # terminal 確定後、同じ process 内で後続 frame/decision を送っても状態も
        # effect も変化しない(restart effect 0、overwrite 拒否)。
        for _ in range(5):
            now += _TICK_NS
            effects = sm.step(death, fx.make_no_op_decision(death), now_ns=now)
            assert effects == ()
            assert sm.context.state is locked_state

    def test_formal_single_attempt_rejects_second_gameplay_entry(self) -> None:
        # 自然な画面遷移では formal mode は死亡後 RUN_SETUP へ戻らないため、
        # reducer の guard 自体をピンポイントに(直接 context を用意して)検証する。
        """単一試行の正式 mode は二回目の gameplay を拒否する。

        終了後の画面変化で別 run の実行を始めません。
        """
        ctx = StateContext(
            state=ControllerState.RUN_SETUP,
            campaign_run_mode=CampaignRunMode.FORMAL_SINGLE_ATTEMPT,
            gameplay_entries=1,
            state_entered_ns=0,
            pending_category=ControllerState.GAMEPLAY,
            pending_streak=2,
        )
        gp = fx.make_perception_snapshot(screen_state="gameplay", snapshot_id="gp-2nd-attempt")
        new_ctx, effects = reduce(ctx, gp, fx.make_move_decision(gp), now_ns=100_000_000)
        assert new_ctx.state is ControllerState.FORMAL_RUN_TERMINAL_FAILURE
        assert [e.kind for e in effects] == ["release_all", "controller_stop", "process_terminate"]


class TestArmResetsRunContext:
    """再 arm は前 run の文脈を消す。

    成功記録や訪問数、未完了操作を次の run へ持ち越しません。
    """

    def test_rearm_after_debug_target_reached_timeout_resets_stale_success_latch(self) -> None:
        # 前 run が TARGET_REACHED まで到達した(success_latched=True)まま
        # debug モードで fail-closed(DISARMED)し、再arm した場合、新しい run の
        # pre-30 death が誤って COMPLETE になってはならない。
        """debug timeout 後の再 arm で成功記録を消す。

        次の run の早い死亡が前の目標到達により成功扱いになりません。
        """
        sm = StateMachine()
        now, gp = _arm_to_gameplay(sm, 0, mode=CampaignRunMode.OPERATOR_DEBUG_RESTART)
        tr = fx.make_perception_snapshot(screen_state="target_reached_transition", snapshot_id="tr-stale", buttons=())
        for _ in range(3):
            now += _TICK_NS
            sm.step(tr, fx.make_no_op_decision(tr), now_ns=now)
        assert sm.context.state is ControllerState.TARGET_REACHED_PENDING_TRANSITION
        assert sm.context.success_latched is True

        for _ in range(2000):
            now += _TICK_NS
            sm.step(tr, fx.make_no_op_decision(tr), now_ns=now)
            if sm.context.state is ControllerState.DISARMED:
                break
        assert sm.context.state is ControllerState.DISARMED

        # 新しい run を再arm する(profile 以外の文脈は完全に作り直されるはず)。
        sm.arm(campaign_run_mode=CampaignRunMode.OPERATOR_DEBUG_RESTART, run_id="run-2", gameplay_attempt_id="attempt-2", now_ns=now)
        assert sm.context.success_latched is False
        assert sm.context.gameplay_entries == 0

        gp2 = fx.make_perception_snapshot(screen_state="gameplay", snapshot_id="gp-run2")
        for _ in range(6):
            now += _TICK_NS
            sm.step(gp2, fx.make_move_decision(gp2), now_ns=now)
        assert sm.context.state is ControllerState.GAMEPLAY
        assert sm.context.success_latched is False

        death = fx.make_perception_snapshot(screen_state="death", snapshot_id="death-run2-pre30")
        for _ in range(1000):
            now += _TICK_NS
            sm.step(death, fx.make_no_op_decision(death), now_ns=now)
            if sm.context.state is ControllerState.RUN_SETUP:
                break
        assert sm.context.state is ControllerState.RUN_SETUP  # COMPLETE ではない。

    def test_rearm_from_debug_to_formal_resets_gameplay_entries(self) -> None:
        # 前 run(debug restart)で既に gameplay_entries=1 のまま manual abort で
        # DISARMED へ戻り、次の process/run を formal_single_attempt で
        # re-arm した場合、最初の GAMEPLAY entry を拒否してはならない。
        """debug から正式 mode への再 arm で開始回数を消す。

        次の正式 run の初回 gameplay が二回目として拒否されないことを確認します。
        """
        sm = StateMachine()
        now, gp = _arm_to_gameplay(sm, 0, mode=CampaignRunMode.OPERATOR_DEBUG_RESTART)
        assert sm.context.gameplay_entries == 1
        now += _TICK_NS
        sm.step(gp, fx.make_move_decision(gp), now_ns=now, manual_abort_requested=True)
        assert sm.context.state is ControllerState.DISARMED

        now, gp2 = _arm_to_gameplay(sm, now, mode=CampaignRunMode.FORMAL_SINGLE_ATTEMPT)
        assert sm.context.state is ControllerState.GAMEPLAY
        assert sm.context.gameplay_entries == 1


class TestPausedUnknownRecover:
    """一時停止・不明・復帰の入力規則を検証する。

    paused は無期限に待ち、unknown は一秒で停止し、復帰は再確認を経ます。
    """

    def test_paused_waits_indefinitely_without_any_effect(self) -> None:
        """一時停止は時間制限なしで入力を送らない。

        長く待っても timeout やクリックを作りません。
        """
        sm = StateMachine()
        now, gp = _arm_to_gameplay(sm, 0)
        paused = fx.make_perception_snapshot(screen_state="paused", snapshot_id="paused-0")
        for _ in range(3):
            now += _TICK_NS
            sm.step(paused, fx.make_no_op_decision(paused), now_ns=now)
        assert sm.context.state is ControllerState.PAUSED
        for _ in range(50):
            now += 1_000_000_000
            effects = sm.step(paused, fx.make_no_op_decision(paused), now_ns=now)
            assert effects == ()
        assert sm.context.state is ControllerState.PAUSED

    def test_paused_recovers_to_gameplay_via_recover_buffer(self) -> None:
        """一時停止後は復帰バッファを通って通常プレイへ戻る。

        再開画面を一枚見ただけで移動を再開しません。
        """
        sm = StateMachine()
        now, gp = _arm_to_gameplay(sm, 0)
        paused = fx.make_perception_snapshot(screen_state="paused", snapshot_id="paused-1")
        for _ in range(3):
            now += _TICK_NS
            sm.step(paused, fx.make_no_op_decision(paused), now_ns=now)
        gp2 = fx.make_perception_snapshot(screen_state="gameplay", snapshot_id="gp-recover")
        for _ in range(3):
            now += _TICK_NS
            sm.step(gp2, fx.make_move_decision(gp2), now_ns=now)
        assert sm.context.state is ControllerState.RECOVER
        for _ in range(3):
            now += _TICK_NS
            effects = sm.step(gp2, fx.make_move_decision(gp2), now_ns=now)
            assert effects == ()
        assert sm.context.state is ControllerState.GAMEPLAY

    def test_unknown_never_emits_any_input(self) -> None:
        """UNKNOWN 中は入力を送らない。

        移動や UI の判断が届いても実行 effect に変換しません。
        """
        sm = StateMachine()
        now, gp = _arm_to_gameplay(sm, 0)
        unk = fx.make_perception_snapshot(screen_state="totally_unrecognized", snapshot_id="unk-0")
        for _ in range(3):
            now += _TICK_NS
            sm.step(unk, fx.make_no_op_decision(unk), now_ns=now)
        assert sm.context.state is ControllerState.UNKNOWN
        for _ in range(3):
            now += _TICK_NS
            effects = sm.step(unk, fx.make_no_op_decision(unk), now_ns=now)
            assert effects == ()

    def test_unknown_timeout_fails_closed_after_one_second(self) -> None:
        """不明画面が一秒続くと停止する。

        認識できない状態を無期限に操作待ちとして扱いません。
        """
        sm = StateMachine()
        now, gp = _arm_to_gameplay(sm, 0, mode=CampaignRunMode.OPERATOR_DEBUG_RESTART)
        unk = fx.make_perception_snapshot(screen_state="totally_unrecognized", snapshot_id="unk-timeout")
        for _ in range(3):
            now += _TICK_NS
            sm.step(unk, fx.make_no_op_decision(unk), now_ns=now)
        for _ in range(200):
            now += _TICK_NS
            effects = sm.step(unk, fx.make_no_op_decision(unk), now_ns=now)
            if sm.context.state is not ControllerState.UNKNOWN:
                break
        assert sm.context.state is ControllerState.DISARMED
        assert effects == (effects[0],) and effects[0].kind == "release_all"

    def test_focus_loss_during_gameplay_routes_through_unknown(self) -> None:
        """プレイ中の focus 喪失は UNKNOWN を経由する。

        ゲーム以外のウィンドウへ held input を送り続けません。
        """
        sm = StateMachine()
        now, gp = _arm_to_gameplay(sm, 0)
        for _ in range(3):
            now += _TICK_NS
            sm.step(gp, fx.make_no_op_decision(gp), now_ns=now, window_focused=False)
        assert sm.context.state is ControllerState.UNKNOWN


class TestGlobalSafetySignals:
    """手動停止と stop 判断を全状態より優先する。

    UI の確認待ちがあっても入力を即時に解放できます。
    """

    def test_manual_abort_overrides_everything(self) -> None:
        """手動 abort は通常遷移に優先する。

        新しい判断や成功候補があっても停止操作を後回しにしません。
        """
        sm = StateMachine()
        now, gp = _arm_to_gameplay(sm, 0, mode=CampaignRunMode.FORMAL_SINGLE_ATTEMPT)
        now += _TICK_NS
        effects = sm.step(gp, fx.make_move_decision(gp), now_ns=now, manual_abort_requested=True)
        assert sm.context.state is ControllerState.FORMAL_RUN_TERMINAL_FAILURE
        assert [e.kind for e in effects] == ["release_all", "controller_stop", "process_terminate"]

    def test_stop_decision_triggers_immediate_emergency_stop_from_gameplay(self) -> None:
        """GAMEPLAY の stop 判断で直ちに入力を解除する。

        次の画面確認を待たずに DISARMED へ戻ります。
        """
        sm = StateMachine()
        now, gp = _arm_to_gameplay(sm, 0, mode=CampaignRunMode.OPERATOR_DEBUG_RESTART)
        now += _TICK_NS
        effects = sm.step(gp, fx.make_stop_decision(gp), now_ns=now)
        assert sm.context.state is ControllerState.DISARMED
        assert effects and effects[0].kind == "release_all"

    def test_stop_decision_triggers_immediate_emergency_stop_from_level_up(self) -> None:
        """LEVEL_UP の stop 判断でも直ちに停止する。

        カードのクリック試行を続けず入力を解放します。
        """
        sm = StateMachine()
        now, gp = _arm_to_gameplay(sm, 0, mode=CampaignRunMode.OPERATOR_DEBUG_RESTART)
        lu = fx.make_perception_snapshot(
            screen_state="level_up_items", snapshot_id="lu-stop",
            candidates=(fx.make_candidate_target(choice_id="c0", choice_index=0),),
        )
        for _ in range(3):
            now += _TICK_NS
            sm.step(lu, fx.make_no_op_decision(lu), now_ns=now)
        assert sm.context.state is ControllerState.LEVEL_UP
        now += _TICK_NS
        effects = sm.step(lu, fx.make_stop_decision(lu), now_ns=now)
        assert sm.context.state is ControllerState.DISARMED
        assert effects[0].kind == "release_all"


class TestTransitionTable:
    """M2: (from_state, confirmed_category, campaign_run_mode) -> (to_state, effect_kinds) を
    table-driven に検証する。

    やさしい説明: これまで parametrize されていたのは ``classify_screen_state``
    だけでした。ここでは実際に state を1つ移す「遷移」そのものを表として
    列挙し、plan本文の legal transition と、代表的な illegal transition
    (LEVEL_UP/CHEST 相互、TARGET_REACHED→UNKNOWN、RECOVER→他)を網羅します。
    ``StateContext`` を直接組み立て、debounce が既に確定する寸前
    (``pending_streak = debounce_frames - 1``)にしておくことで、1回の
    ``reduce()`` 呼び出しで「確定 tick」だけを取り出してテストします。
    """

    _CATEGORY_RAW: dict[ControllerState, str] = {
        ControllerState.GAMEPLAY: "gameplay",
        ControllerState.LEVEL_UP: "level_up_items",
        ControllerState.CHEST: "chest",
        ControllerState.PAUSED: "paused",
        ControllerState.UNKNOWN: "totally_unrecognized_table_probe",
        ControllerState.DEATH_RESULT: "death",
        ControllerState.TARGET_REACHED_PENDING_TRANSITION: "target_reached_transition",
    }

    def _confirm(
        self,
        from_state: ControllerState,
        category: ControllerState,
        mode: CampaignRunMode,
        *,
        extra: dict | None = None,
        move_action: int | None = None,
        now_ns: int = 500_000_000,
    ):
        """画面分類を連続して確定させる。

        遷移表の各行に同じ debounce 条件を与えて結果を比べます。
        """
        profile = NavigationProfile.default_v1()
        raw = self._CATEGORY_RAW[category]
        snapshot = fx.make_perception_snapshot(
            screen_state=raw, snapshot_id=f"tt-{from_state.value}-{category.value}-{mode.value}"
        )
        ctx = StateContext(
            state=from_state,
            campaign_run_mode=mode,
            pending_category=category,
            pending_streak=profile.debounce_frames - 1,
            state_entered_ns=0,
            **(extra or {}),
        )
        decision = (
            fx.make_move_decision(snapshot, action_index=move_action)
            if move_action is not None
            else fx.make_no_op_decision(snapshot)
        )
        return reduce(ctx, snapshot, decision, now_ns=now_ns, profile=profile)

    @pytest.mark.parametrize(
        "from_state,category,mode,expected_to_state,expected_kinds",
        [
            # ACQUIRE_TARGET -> RUN_SETUP(legal) / confirmed UNKNOWN は素通りしない。
            pytest.param(
                ControllerState.ACQUIRE_TARGET, ControllerState.GAMEPLAY, CampaignRunMode.OPERATOR_DEBUG_RESTART,
                ControllerState.RUN_SETUP, (), id="acquire_target-gameplay-run_setup",
            ),
            pytest.param(
                ControllerState.ACQUIRE_TARGET, ControllerState.UNKNOWN, CampaignRunMode.OPERATOR_DEBUG_RESTART,
                ControllerState.ACQUIRE_TARGET, (), id="acquire_target-unknown-stays",
            ),
            # RUN_SETUP -> GAMEPLAY(legal) / confirmed UNKNOWN は素通りしない。
            pytest.param(
                ControllerState.RUN_SETUP, ControllerState.GAMEPLAY, CampaignRunMode.OPERATOR_DEBUG_RESTART,
                ControllerState.GAMEPLAY, (), id="run_setup-gameplay-lands",
            ),
            pytest.param(
                ControllerState.RUN_SETUP, ControllerState.UNKNOWN, CampaignRunMode.FORMAL_SINGLE_ATTEMPT,
                ControllerState.RUN_SETUP, (), id="run_setup-unknown-stays",
            ),
            # GAMEPLAY からの legal 遷移: 抜けるtickで release_all を1回出す。
            pytest.param(
                ControllerState.GAMEPLAY, ControllerState.LEVEL_UP, CampaignRunMode.OPERATOR_DEBUG_RESTART,
                ControllerState.LEVEL_UP, ("release_all",), id="gameplay-level_up",
            ),
            pytest.param(
                ControllerState.GAMEPLAY, ControllerState.CHEST, CampaignRunMode.OPERATOR_DEBUG_RESTART,
                ControllerState.CHEST, ("release_all",), id="gameplay-chest",
            ),
            pytest.param(
                ControllerState.GAMEPLAY, ControllerState.PAUSED, CampaignRunMode.OPERATOR_DEBUG_RESTART,
                ControllerState.PAUSED, ("release_all",), id="gameplay-paused",
            ),
            pytest.param(
                ControllerState.GAMEPLAY, ControllerState.UNKNOWN, CampaignRunMode.OPERATOR_DEBUG_RESTART,
                ControllerState.UNKNOWN, ("release_all",), id="gameplay-unknown",
            ),
            pytest.param(
                ControllerState.GAMEPLAY, ControllerState.DEATH_RESULT, CampaignRunMode.OPERATOR_DEBUG_RESTART,
                ControllerState.DEATH_RESULT, ("release_all",), id="gameplay-death_result",
            ),
            pytest.param(
                ControllerState.GAMEPLAY, ControllerState.TARGET_REACHED_PENDING_TRANSITION,
                CampaignRunMode.OPERATOR_DEBUG_RESTART, ControllerState.TARGET_REACHED_PENDING_TRANSITION,
                ("release_all",), id="gameplay-target_reached",
            ),
            # LEVEL_UP: illegal cross-modal(CHEST) / legal ack(GAMEPLAY/DEATH_RESULT)。
            pytest.param(
                ControllerState.LEVEL_UP, ControllerState.CHEST, CampaignRunMode.FORMAL_SINGLE_ATTEMPT,
                ControllerState.FORMAL_RUN_TERMINAL_FAILURE,
                ("release_all", "controller_stop", "process_terminate"), id="level_up-chest-illegal-formal",
            ),
            pytest.param(
                ControllerState.LEVEL_UP, ControllerState.CHEST, CampaignRunMode.OPERATOR_DEBUG_RESTART,
                ControllerState.DISARMED, ("release_all",), id="level_up-chest-illegal-debug",
            ),
            pytest.param(
                ControllerState.LEVEL_UP, ControllerState.GAMEPLAY, CampaignRunMode.OPERATOR_DEBUG_RESTART,
                ControllerState.GAMEPLAY, (), id="level_up-gameplay-ack",
            ),
            pytest.param(
                ControllerState.LEVEL_UP, ControllerState.DEATH_RESULT, CampaignRunMode.OPERATOR_DEBUG_RESTART,
                ControllerState.DEATH_RESULT, (), id="level_up-death_result",
            ),
            # CHEST: illegal cross-modal(LEVEL_UP) / legal ack(GAMEPLAY/DEATH_RESULT)。
            pytest.param(
                ControllerState.CHEST, ControllerState.LEVEL_UP, CampaignRunMode.FORMAL_SINGLE_ATTEMPT,
                ControllerState.FORMAL_RUN_TERMINAL_FAILURE,
                ("release_all", "controller_stop", "process_terminate"), id="chest-level_up-illegal-formal",
            ),
            pytest.param(
                ControllerState.CHEST, ControllerState.LEVEL_UP, CampaignRunMode.OPERATOR_DEBUG_RESTART,
                ControllerState.DISARMED, ("release_all",), id="chest-level_up-illegal-debug",
            ),
            pytest.param(
                ControllerState.CHEST, ControllerState.GAMEPLAY, CampaignRunMode.OPERATOR_DEBUG_RESTART,
                ControllerState.GAMEPLAY, (), id="chest-gameplay-ack",
            ),
            pytest.param(
                ControllerState.CHEST, ControllerState.DEATH_RESULT, CampaignRunMode.OPERATOR_DEBUG_RESTART,
                ControllerState.DEATH_RESULT, (), id="chest-death_result",
            ),
            # PAUSED: indefinite / GAMEPLAY は RECOVER 経由 / LEVEL_UP直行はillegal(UNKNOWNへ)。
            pytest.param(
                ControllerState.PAUSED, ControllerState.PAUSED, CampaignRunMode.OPERATOR_DEBUG_RESTART,
                ControllerState.PAUSED, (), id="paused-paused-stays",
            ),
            pytest.param(
                ControllerState.PAUSED, ControllerState.GAMEPLAY, CampaignRunMode.OPERATOR_DEBUG_RESTART,
                ControllerState.RECOVER, (), id="paused-gameplay-recover",
            ),
            pytest.param(
                ControllerState.PAUSED, ControllerState.DEATH_RESULT, CampaignRunMode.OPERATOR_DEBUG_RESTART,
                ControllerState.DEATH_RESULT, (), id="paused-death_result",
            ),
            pytest.param(
                ControllerState.PAUSED, ControllerState.UNKNOWN, CampaignRunMode.OPERATOR_DEBUG_RESTART,
                ControllerState.UNKNOWN, (), id="paused-unknown",
            ),
            pytest.param(
                ControllerState.PAUSED, ControllerState.LEVEL_UP, CampaignRunMode.OPERATOR_DEBUG_RESTART,
                ControllerState.UNKNOWN, (), id="paused-level_up-illegal-direct",
            ),
            # UNKNOWN: 1s timeout 管理下からの legal 遷移(TARGET_REACHED含む)。
            pytest.param(
                ControllerState.UNKNOWN, ControllerState.DEATH_RESULT, CampaignRunMode.OPERATOR_DEBUG_RESTART,
                ControllerState.DEATH_RESULT, (), id="unknown-death_result",
            ),
            pytest.param(
                ControllerState.UNKNOWN, ControllerState.GAMEPLAY, CampaignRunMode.OPERATOR_DEBUG_RESTART,
                ControllerState.RECOVER, (), id="unknown-gameplay-recover",
            ),
            pytest.param(
                ControllerState.UNKNOWN, ControllerState.PAUSED, CampaignRunMode.OPERATOR_DEBUG_RESTART,
                ControllerState.PAUSED, (), id="unknown-paused",
            ),
            pytest.param(
                ControllerState.UNKNOWN, ControllerState.LEVEL_UP, CampaignRunMode.OPERATOR_DEBUG_RESTART,
                ControllerState.LEVEL_UP, (), id="unknown-level_up",
            ),
            pytest.param(
                ControllerState.UNKNOWN, ControllerState.CHEST, CampaignRunMode.OPERATOR_DEBUG_RESTART,
                ControllerState.CHEST, (), id="unknown-chest",
            ),
            pytest.param(
                ControllerState.UNKNOWN, ControllerState.TARGET_REACHED_PENDING_TRANSITION,
                CampaignRunMode.OPERATOR_DEBUG_RESTART, ControllerState.TARGET_REACHED_PENDING_TRANSITION,
                (), id="unknown-target_reached",
            ),
            # RECOVER: GAMEPLAY/PAUSED/DEATH_RESULT は legal、その他は illegal(UNKNOWNへ戻す)。
            pytest.param(
                ControllerState.RECOVER, ControllerState.DEATH_RESULT, CampaignRunMode.OPERATOR_DEBUG_RESTART,
                ControllerState.DEATH_RESULT, (), id="recover-death_result",
            ),
            pytest.param(
                ControllerState.RECOVER, ControllerState.GAMEPLAY, CampaignRunMode.OPERATOR_DEBUG_RESTART,
                ControllerState.GAMEPLAY, (), id="recover-gameplay",
            ),
            pytest.param(
                ControllerState.RECOVER, ControllerState.PAUSED, CampaignRunMode.OPERATOR_DEBUG_RESTART,
                ControllerState.PAUSED, (), id="recover-paused",
            ),
            pytest.param(
                ControllerState.RECOVER, ControllerState.LEVEL_UP, CampaignRunMode.OPERATOR_DEBUG_RESTART,
                ControllerState.UNKNOWN, (), id="recover-level_up-illegal-direct",
            ),
            # TARGET_REACHED_PENDING_TRANSITION: DEATH_RESULT は legal(success優先)、
            # UNKNOWN への直接遷移は illegal(状態はそのまま)。
            pytest.param(
                ControllerState.TARGET_REACHED_PENDING_TRANSITION, ControllerState.DEATH_RESULT,
                CampaignRunMode.OPERATOR_DEBUG_RESTART, ControllerState.DEATH_RESULT, (), id="target_reached-death_result",
            ),
            pytest.param(
                ControllerState.TARGET_REACHED_PENDING_TRANSITION, ControllerState.UNKNOWN,
                CampaignRunMode.OPERATOR_DEBUG_RESTART, ControllerState.TARGET_REACHED_PENDING_TRANSITION, (),
                id="target_reached-unknown-illegal",
            ),
        ],
    )
    def test_confirmed_transition_table(
        self,
        from_state: ControllerState,
        category: ControllerState,
        mode: CampaignRunMode,
        expected_to_state: ControllerState,
        expected_kinds: tuple[str, ...],
    ) -> None:
        """確定画面の遷移表と effect を照合する。

        開始状態・次の画面・実行 mode の組ごとに合法な遷移と停止を検証します。
        """
        extra = {"success_latched": True} if from_state is ControllerState.TARGET_REACHED_PENDING_TRANSITION else {}
        new_ctx, effects = self._confirm(from_state, category, mode, extra=extra)
        assert new_ctx.state is expected_to_state
        assert tuple(effect.kind for effect in effects) == expected_kinds

    def _target_reached_confirm_button_step(
        self,
        *,
        category_raw: str,
        pending_category: ControllerState | None,
        pending_streak: int,
        window_focused: bool = True,
    ):
        """TARGET_REACHED 中に confirm button の ``ui`` decision を与えて1tick進める。

        やさしい説明: TR 中は本来 confirm button を Enter で押し続けるはずですが、
        M15 fix の対象は「UNKNOWN/PAUSED/focus loss の間はそれをしてはいけない」
        ことなので、修正前なら ``ui_key`` effect が出てしまうシナリオをそのまま
        与えて 0 件であることを確認するためのヘルパーです。
        """
        profile = NavigationProfile.default_v1()
        snapshot = fx.make_perception_snapshot(
            screen_state=category_raw,
            snapshot_id=f"tt-tr-zero-input-{category_raw}",
            buttons=(fx.make_button_target(semantic_action="confirm"),),
        )
        intent = fx.make_button_intent(snapshot, kind=UiIntentKind.CONFIRM, semantic_action="confirm")
        decision = fx.make_ui_decision(snapshot, intent)
        ctx = StateContext(
            state=ControllerState.TARGET_REACHED_PENDING_TRANSITION,
            campaign_run_mode=CampaignRunMode.OPERATOR_DEBUG_RESTART,
            pending_category=pending_category,
            pending_streak=pending_streak,
            state_entered_ns=0,
            success_latched=True,
        )
        return reduce(
            ctx, snapshot, decision, now_ns=500_000_000, profile=profile, window_focused=window_focused
        )

    def test_target_reached_confirmed_unknown_sends_zero_ui_input(self) -> None:
        # M15 fix: confirmed UNKNOWN の間、confirm button の ui_key を送り続けない
        # (以前は timeout の5秒間 Enter を送り続けていた)。
        """目標後の確定 UNKNOWN は UI 入力を送らない。

        読めない終了画面へ Enter やクリックを繰り返しません。
        """
        profile = NavigationProfile.default_v1()
        new_ctx, effects = self._target_reached_confirm_button_step(
            category_raw="totally_unrecognized_table_probe",
            pending_category=ControllerState.UNKNOWN,
            pending_streak=profile.debounce_frames - 1,
        )
        assert new_ctx.state is ControllerState.TARGET_REACHED_PENDING_TRANSITION
        assert effects == ()

    def test_target_reached_confirmed_paused_sends_zero_ui_input(self) -> None:
        # M15 fix: confirmed PAUSED の間も同様に入力0を維持する
        # (PAUSED は他の状態では無期限待機・入力0が原則であり、それと矛盾しない)。
        """目標後の確定 PAUSED でも UI 入力を送らない。

        操作者の一時停止を確認操作の継続として扱いません。
        """
        profile = NavigationProfile.default_v1()
        new_ctx, effects = self._target_reached_confirm_button_step(
            category_raw="paused",
            pending_category=ControllerState.PAUSED,
            pending_streak=profile.debounce_frames - 1,
        )
        assert new_ctx.state is ControllerState.TARGET_REACHED_PENDING_TRANSITION
        assert effects == ()

    def test_target_reached_focus_loss_sends_zero_ui_input(self) -> None:
        # M15 fix: window_focused=False は強制的に UNKNOWN 分類になり、
        # confirm 済みでなくても(debounce 未確定でも)入力0のまま待つ。
        """目標後の focus 喪失では確認入力を止める。

        別ウィンドウへ Enter を送り続けないことを確認します。
        """
        new_ctx, effects = self._target_reached_confirm_button_step(
            category_raw="target_reached_transition",
            pending_category=None,
            pending_streak=0,
            window_focused=False,
        )
        assert new_ctx.state is ControllerState.TARGET_REACHED_PENDING_TRANSITION
        assert effects == ()

    def test_target_reached_confirmed_level_up_fails_closed(self) -> None:
        # modal screen の unexpected_ui_transition と挙動を揃える(状態間の一貫性)。
        """目標後にカード画面が確定したら停止する。

        予定外の選択画面を成功確認として通しません。
        """
        profile = NavigationProfile.default_v1()
        new_ctx, effects = self._target_reached_confirm_button_step(
            category_raw="level_up_items",
            pending_category=ControllerState.LEVEL_UP,
            pending_streak=profile.debounce_frames - 1,
        )
        assert new_ctx.state is ControllerState.DISARMED
        assert tuple(effect.kind for effect in effects) == ("release_all",)

    @pytest.mark.parametrize("mode", [CampaignRunMode.OPERATOR_DEBUG_RESTART, CampaignRunMode.FORMAL_SINGLE_ATTEMPT])
    def test_target_reached_gameplay_confirm_completes_regardless_of_mode(self, mode: CampaignRunMode) -> None:
        """目標後の gameplay 確認は mode によらず成功になる。

        正式と debug の両方で同じ post-30 確定証拠を使います。
        """
        new_ctx, effects = self._confirm(
            ControllerState.TARGET_REACHED_PENDING_TRANSITION,
            ControllerState.GAMEPLAY,
            mode,
            extra={"success_latched": True},
        )
        assert new_ctx.state is ControllerState.COMPLETE
        assert new_ctx.terminal_locked is True
        assert [effect.kind for effect in effects] == ["release_all", "controller_stop", "process_terminate"]

    def test_gameplay_stay_forwards_move_effect(self) -> None:
        """通常プレイ継続中は移動 effect を渡す。

        画面状態が変わらない tick では選択された action を実行できます。
        """
        new_ctx, effects = self._confirm(
            ControllerState.GAMEPLAY, ControllerState.GAMEPLAY, CampaignRunMode.OPERATOR_DEBUG_RESTART, move_action=4
        )
        assert new_ctx.state is ControllerState.GAMEPLAY
        assert [effect.kind for effect in effects] == ["move"]
        assert effects[0].action_index == 4

    def test_target_reached_entry_from_gameplay_sets_success_latch(self) -> None:
        """gameplay から目標画面へ入ると成功記録を立てる。

        終端確認まで到達済みの事実を保持します。
        """
        new_ctx, _effects = self._confirm(
            ControllerState.GAMEPLAY,
            ControllerState.TARGET_REACHED_PENDING_TRANSITION,
            CampaignRunMode.OPERATOR_DEBUG_RESTART,
        )
        assert new_ctx.success_latched is True

    def test_target_reached_entry_from_unknown_sets_success_latch(self) -> None:
        # M7(a) の table-driven 側の固定: UNKNOWN 経由でも success_latched が立つ。
        """UNKNOWN から目標画面へ入っても成功記録を立てる。

        観測の空白があっても新しい到達証拠を採用します。
        """
        new_ctx, _effects = self._confirm(
            ControllerState.UNKNOWN,
            ControllerState.TARGET_REACHED_PENDING_TRANSITION,
            CampaignRunMode.OPERATOR_DEBUG_RESTART,
        )
        assert new_ctx.success_latched is True


class TestGoldenEndToEnd:
    """04-09/05-01 契約に沿った parser -> assembler -> AgentDecision -> resolver 相当の end-to-end 再生。

    やさしい説明: 04-09/05-01 の実物パーサ/アセンブラは import しませんが、
    それらが最終的に生成する形(PerceptionSnapshot + AgentDecision +
    UiIntentV1)を fixture で再現し、resolver が「選択 semantic だけ」を
    対応 ROI へ解決して安全な入力列になることを end-to-end で確認します。
    """

    def test_full_run_reaches_complete_with_expected_effect_sequence(self) -> None:
        """一 run の全体列が予定した順序で成功へ進む。

        起動・移動・UI 操作・目標確認を通して effect の順番を照合します。
        """
        sm = StateMachine()
        now, gp = _arm_to_gameplay(sm, 0, mode=CampaignRunMode.FORMAL_SINGLE_ATTEMPT)

        # レベルアップを1回、選択して gameplay へ戻る。
        lu = fx.make_perception_snapshot(
            screen_state="level_up_items", snapshot_id="golden-lu",
            candidates=(fx.make_candidate_target(choice_id="knife", choice_index=0),),
        )
        intent = fx.make_choose_card_intent(lu, target_index=0)
        decision = fx.make_ui_decision(lu, intent)
        click_effects: tuple = ()
        for _ in range(4):
            now += _TICK_NS
            click_effects = sm.step(lu, decision, now_ns=now)
        assert [e.kind for e in click_effects] == ["release_all", "ui_click"]

        gp2 = fx.make_perception_snapshot(screen_state="gameplay", snapshot_id="golden-gp2")
        for _ in range(3):
            now += _TICK_NS
            sm.step(gp2, fx.make_move_decision(gp2), now_ns=now)
        assert sm.context.state is ControllerState.GAMEPLAY

        # 30:00 到達 -> confirm -> result -> COMPLETE。
        tr = fx.make_perception_snapshot(
            screen_state="target_reached_transition", snapshot_id="golden-tr",
            buttons=(fx.make_button_target(semantic_action="confirm"),),
        )
        confirm_intent = fx.make_button_intent(tr, kind=UiIntentKind.CONFIRM, semantic_action="confirm")
        confirm_decision = fx.make_ui_decision(tr, confirm_intent)
        for _ in range(4):
            now += _TICK_NS
            sm.step(tr, confirm_decision, now_ns=now)
        assert sm.context.success_latched is True

        result = fx.make_perception_snapshot(screen_state="result", snapshot_id="golden-result")
        for _ in range(10):
            now += _TICK_NS
            sm.step(result, fx.make_no_op_decision(result), now_ns=now)
            if sm.context.terminal_locked:
                break
        assert sm.context.state is ControllerState.COMPLETE


class TestNoiseInvariantProperty:
    """M15: random/noisy state sequence でも安全不変条件が破れない。

    やさしい説明: 以前は毎tick完全に一様乱数で画面を選んでいたため、
    3-frame debounce がほとんど成立せず、LEVEL_UP/CHEST/PAUSED/UNKNOWN/
    RECOVER/TARGET_REACHED に一度も入らず ui_click effect も0件のまま
    「空振り」していました(実測済み)。ここでは同じ画面を1〜6フレーム
    連続させる run-length 付きノイズに変え、全状態への訪問と
    ui_click/ui_key の発生を最低件数で保証します。また fake な
    movement lease(75ms)モデルで「held」を追跡し、UI/UNKNOWN/PAUSED/
    RECOVER中や emergency stop 直後に held が残っていないことも検証します
    (以前は effect 種別を見ておらず、この検証は素通りしていました)。
    """

    _SCREEN_STATE_POOL = (
        "gameplay",
        "level_up_items",
        "level_up_fallback",
        "chest",
        "target_reached_transition",
        "paused",
        "death",
        "result",
        "unknown",
        "totally_unrecognized_noise",
    )

    _REQUIRED_VISITED_STATES = frozenset(
        {
            ControllerState.LEVEL_UP,
            ControllerState.CHEST,
            ControllerState.PAUSED,
            ControllerState.UNKNOWN,
            ControllerState.RECOVER,
            ControllerState.TARGET_REACHED_PENDING_TRANSITION,
            ControllerState.DEATH_RESULT,
        }
    )

    _LEASE_NS = 75_000_000  # InputLeaseController の movement lease(75ms)を模した閾値。

    def _generate_schedule(self, rng: random.Random, total_ticks: int) -> list[str]:
        """同じ画面分類を 1〜6 フレーム連続させる run-length 付きノイズ列を作る。

        やさしい説明: プール全体を毎 lap シャッフルしてから run-length を
        付けて並べるので、debounce(既定3フレーム)を上回る run が高確率で
        毎 lap 発生し、全画面状態への訪問を(運任せではなく)実質的に保証します。
        """
        schedule: list[str] = []
        pool = list(self._SCREEN_STATE_POOL)
        while len(schedule) < total_ticks:
            rng.shuffle(pool)
            for state in pool:
                schedule.extend([state] * rng.randint(1, 6))
        return schedule[:total_ticks]

    @staticmethod
    def _run_ids_for_schedule(schedule: list[str]) -> list[int]:
        """同じ raw_state が連続する区間(1つの UI 訪問)に共通の run id を振る。

        やさしい説明: M15 fix で追加した回帰テストです。以前はこの property test の
        fixture が tick ごとに全く別の ``ui_state_key``/``candidate_set_hash`` を
        生成していたため、`apply ack` バグ(1フレームの揺れで無制限 re-click)が
        毎tick発生していても検出できませんでした。同じ raw_state が連続する区間では
        同じ run id(＝同じ UI 訪問)を割り当てることで、fixture 側の
        ``ui_state_key``/``candidate_set_hash`` を安定させ、retry の正常系
        (initial 1 + retry 最大1)も実際に運動させて検証できるようにします。
        """
        run_ids: list[int] = []
        current_run_id = -1
        prev_raw: str | None = None
        for raw in schedule:
            if raw != prev_raw:
                current_run_id += 1
            run_ids.append(current_run_id)
            prev_raw = raw
        return run_ids

    def test_random_noisy_sequences_never_violate_safety_invariants(self) -> None:
        """ノイズを含む多数の画面列でも入力の不変条件を守る。

        訪問ごとの送信上限、停止時の解除、移動禁止状態を乱数列で確認します。
        """
        rng = random.Random(20260925)
        visited_states: set[ControllerState] = set()
        ui_click_count = 0
        ui_key_count = 0

        for trial in range(40):
            sm = StateMachine()
            mode = CampaignRunMode.FORMAL_SINGLE_ATTEMPT if trial % 2 else CampaignRunMode.OPERATOR_DEBUG_RESTART
            sm.arm(campaign_run_mode=mode, run_id=f"noise-{trial}", gameplay_attempt_id="a", now_ns=0)
            now = 0
            last_move_ns: int | None = None  # fake movement lease(75ms)モデル。
            # RECOVER は「UNKNOWN/PAUSED 中に3連続 gameplay を観測する」という
            # 具体的な前提が要るため、trial/tick数を十分に確保して(運任せに
            # せず)実質的に毎回訪問されるようにする(元は20trial*240tickで
            # RECOVER だけ未訪問になることがあった)。
            schedule = self._generate_schedule(random.Random(rng.random()), 360)
            run_ids = self._run_ids_for_schedule(schedule)
            # focus loss 注入専用の独立 RNG(共有 rng を消費すると schedule/move
            # action の乱数列がずれ、RECOVER 訪問保証などが崩れるため分離する)。
            focus_rng = random.Random(f"focus-loss-{trial}")
            # M4/M15 fix: 同一 UI 訪問(apply ack で attempt が入れ替わっても
            # 訪問をまたいで累積する StateContext.ui_visit_click_count)あたりの
            # click(ui_click/ui_key)が2件を超えないことを、実際の生産コードが
            # 持つカウンタ自身で検証する(attempt単位のヒューリスティックだと
            # apply ack が繰り返し確定して attempt が入れ替わり続けるケースを
            # 見逃す、というレビュー指摘の再発防止)。

            for tick, raw_state in enumerate(schedule):
                now += _TICK_NS
                pre_state = sm.context.state
                snapshot = fx.make_perception_snapshot(
                    screen_state=raw_state,
                    snapshot_id=f"noise-{trial}-{tick}",
                    frame_id=f"noise-frame-{trial}-{tick}",
                    captured_ns=now,
                    # 同じ raw_state の連続区間(run_ids[tick])では ui_state_key/
                    # candidate_set_hash を安定させる(fixtureのui_state_keyが
                    # tickごとに揺れる問題を修正、M15 fix)。
                    ui_state_key=fx._hash_of(f"noise-uikey-{trial}-{run_ids[tick]}"),
                    candidate_set_hash=fx._hash_of(f"noise-cand-{trial}-{run_ids[tick]}"),
                    candidates=(
                        (fx.make_candidate_target(choice_id="c0", choice_index=0),)
                        if raw_state == "level_up_items"
                        else ()
                    ),
                    buttons=(
                        (fx.make_button_target(semantic_action="ack_chest"),)
                        if raw_state == "chest"
                        else (fx.make_button_target(semantic_action="confirm"),)
                        if raw_state == "target_reached_transition"
                        else ()
                    ),
                )
                if pre_state is ControllerState.GAMEPLAY and raw_state == "gameplay":
                    decision = fx.make_move_decision(snapshot, action_index=rng.randrange(9))
                elif raw_state == "level_up_items" and pre_state is ControllerState.LEVEL_UP:
                    intent = fx.make_choose_card_intent(snapshot, target_index=0)
                    decision = fx.make_ui_decision(snapshot, intent)
                elif raw_state == "chest" and pre_state is ControllerState.CHEST:
                    intent = fx.make_button_intent(
                        snapshot, kind=UiIntentKind.ACK_CHEST, semantic_action="ack_chest"
                    )
                    decision = fx.make_ui_decision(snapshot, intent)
                elif raw_state == "target_reached_transition" and pre_state is ControllerState.TARGET_REACHED_PENDING_TRANSITION:
                    intent = fx.make_button_intent(snapshot, kind=UiIntentKind.CONFIRM, semantic_action="confirm")
                    decision = fx.make_ui_decision(snapshot, intent)
                else:
                    decision = fx.make_no_op_decision(snapshot)

                # M15 fix: TARGET_REACHED 中の一部tickでfocus lossを混ぜ、
                # window_focused=False(強制UNKNOWN分類)でも confirm の
                # ui_key を送らないことを property test でも検証する
                # (blocking-findings-2.json 2件目の再現条件そのもの)。
                window_focused = not (
                    raw_state == "target_reached_transition"
                    and pre_state is ControllerState.TARGET_REACHED_PENDING_TRANSITION
                    and focus_rng.random() < 0.3
                )
                category_this_tick = classify_screen_state(raw_state, window_focused=window_focused)

                if sm.context.terminal_locked:
                    effects = sm.step(snapshot, decision, now_ns=now, window_focused=window_focused)
                    assert effects == (), "terminal 確定後に effect が出てはならない"
                    continue

                effects = sm.step(snapshot, decision, now_ns=now, window_focused=window_focused)
                kinds = [effect.kind for effect in effects]
                visited_states.add(sm.context.state)
                ui_click_count += kinds.count("ui_click")
                ui_key_count += kinds.count("ui_key")

                # M15 fix: TARGET_REACHED 中に category が UNKNOWN/PAUSED
                # (focus loss を含む)なら、confirm 待ちであっても ui_click/ui_key
                # effect が絶対に0件であること。
                if pre_state is ControllerState.TARGET_REACHED_PENDING_TRANSITION and category_this_tick in (
                    ControllerState.UNKNOWN,
                    ControllerState.PAUSED,
                ):
                    assert "ui_click" not in kinds and "ui_key" not in kinds

                # M4/M15 fix: 同一 UI 訪問(apply ack で attempt が入れ替わっても
                # 訪問をまたいで累積する StateContext.ui_visit_click_count)あたりの
                # click は2件まで。attempt 単位のヒューリスティックではなく、
                # 生産コードが実際に持つカウンタ自身を直接検証する。
                assert sm.context.ui_visit_click_count <= 2, (
                    "同一UI訪問(ui_visit_click_count)あたりのclickはinitial+retryの最大2件まで"
                )

                for kind in kinds:
                    if kind == "move":
                        last_move_ns = now
                    elif kind == "release_all":
                        last_move_ns = None  # emergency_release は lease を即時に無効化する。

                # I4: 同一 tick で move と ui_click/ui_key を同時に出さない。
                assert not ("move" in kinds and ("ui_click" in kinds or "ui_key" in kinds))
                # UI 中(LEVEL_UP/CHEST/TARGET_REACHED)は move を出さない。
                if sm.context.state in (
                    ControllerState.LEVEL_UP,
                    ControllerState.CHEST,
                    ControllerState.TARGET_REACHED_PENDING_TRANSITION,
                ):
                    assert "move" not in kinds
                # PAUSED/UNKNOWN/RECOVER 中は入力を一切出さない(release_all も含め0件)。
                if sm.context.state in (
                    ControllerState.PAUSED,
                    ControllerState.UNKNOWN,
                    ControllerState.RECOVER,
                ) and pre_state == sm.context.state:
                    assert effects == ()
                # emergency stop(DISARMED への fail-closed 遷移)直後は held input が無い。
                if sm.context.state is ControllerState.DISARMED and pre_state is not ControllerState.DISARMED:
                    assert kinds in ([], ["release_all"])

                # fake movement lease モデル: UI/UNKNOWN/PAUSED/RECOVER 中や
                # emergency stop 直後は held(75ms 以内の move)が残っていない。
                held = last_move_ns is not None and (now - last_move_ns) < self._LEASE_NS
                if sm.context.state in (
                    ControllerState.LEVEL_UP,
                    ControllerState.CHEST,
                    ControllerState.TARGET_REACHED_PENDING_TRANSITION,
                    ControllerState.PAUSED,
                    ControllerState.UNKNOWN,
                    ControllerState.RECOVER,
                ):
                    assert not held, f"{sm.context.state} 中に movement lease が held のまま"
                if sm.context.state is ControllerState.DISARMED and pre_state is not ControllerState.DISARMED:
                    assert not held, "emergency stop 直後に movement lease が held のまま"

                # debug モードで fail-closed(DISARMED)した場合、trial 全体を
                # そこで無駄にせず即座に再arm する(1trialに1回の illegal
                # transitionでUNKNOWN/PAUSED経由のRECOVERまで辿り着く前に
                # 探索が終わってしまうのを防ぐ)。arm() は毎回まっさらな
                # StateContext を作るので(M9 fix)、run を跨いでも安全。
                if sm.context.state is ControllerState.DISARMED:
                    sm.arm(campaign_run_mode=mode, run_id=f"noise-{trial}-{tick}", gameplay_attempt_id="a", now_ns=now)

        # M15 fix: 空振り(全訪問状態が ACQUIRE_TARGET/RUN_SETUP/GAMEPLAY/
        # DEATH_RESULT/FORMAL_RUN_TERMINAL_FAILURE だけ)ではないことを最低件数で保証する。
        missing = self._REQUIRED_VISITED_STATES - visited_states
        assert not missing, f"訪問できなかった状態がある: {missing}"
        assert ui_click_count > 0, "ui_click effect が一度も出なかった"
        assert ui_key_count > 0, "ui_key effect が一度も出なかった"


class TestMeasuredModalReplay:
    """録画に近い三十 fps の長い宝箱とカード過渡を再生する。

    開く段階では候補なしで待ち、終了だけをクリックし、読めないカードは timeout で停止します。
    """

    def _tick(self, sm, state, tick, now, *, buttons=(), candidates=(), stable_key="modal"):
        """新しい撮影時刻と identity で一 tick を進める。

        終了ボタンが残る間は UI key を固定し、confidence のフェードだけでは別操作にしません。
        """
        now += 33_333_334
        snapshot = fx.make_perception_snapshot(screen_state=state, snapshot_id=f"replay-{tick}",
                                              frame_id=f"frame-{tick}", captured_ns=now,
                                              ui_state_key=fx._hash_of(stable_key), buttons=buttons, candidates=candidates,
                                              candidate_set_hash=fx._hash_of("replay-candidates"))
        if state == "chest":
            intent = fx.make_button_intent(snapshot, kind=UiIntentKind.ACK_CHEST, semantic_action="ack_chest")
            decision = fx.make_ui_decision(snapshot, intent)
        elif state == "level_up_items":
            intent = fx.make_choose_card_intent(snapshot)
            decision = fx.make_ui_decision(snapshot, intent)
        else:
            decision = fx.make_move_decision(snapshot)
        return now, sm.step(snapshot, decision, now_ns=now)

    def _chest_buttons(self, *, fading=False):
        """白文字入りの終了画面を本番 parser と UI 変換へ通す。

        retry の信頼度を手入力せず、フェード中に候補が消えることも同じ入口で確認します。
        """
        import numpy as np
        from survivors.vision.hud_parser import HudParser
        from survivors.perception_snapshot import build_ui_presentation_from_hud
        frame = np.zeros((1080, 1920, 4), dtype=np.uint8)
        frame[111:965, 642:1278, :3] = (102, 203, 255)
        frame[117:959, 648:1272, :3] = (116, 79, 75)
        frame[835:902, 822:1098, :3] = (205, 64, 39)
        frame[858:874, 920:1000, :3] = 255
        if fading:
            frame[835:841, 822:1098, :3] = 0
        frame[914:919, 660:1260, :3] = (102, 203, 255)
        hud = HudParser(parser_artifact_hash=fx._hash_of("parser_artifact")).parse(
            frame, session_id="chest-replay", frame_index=0, captured_monotonic_ns=0)
        return build_ui_presentation_from_hud(hud, (1920, 1080)).buttons

    @pytest.mark.parametrize("retry", [False, True])
    def test_chest_waits_twenty_one_seconds_then_closes(self, retry):
        """宝箱を六百四十 frame 待って終了を一回または retry 付きで押す。

        白文字入りの本番解析結果で初回を送り、二百 ms 後に同じボタンへ一度だけ retry します。
        """
        sm = StateMachine()
        now, _ = _arm_to_gameplay(sm, 0)
        for tick in range(640):
            now, effects = self._tick(sm, "chest", tick, now)
            assert all(e.kind not in {"ui_click", "move"} for e in effects)
            assert sm.context.state in {ControllerState.GAMEPLAY, ControllerState.CHEST}
        assert sm.context.state is ControllerState.CHEST
        buttons = self._chest_buttons()
        assert len(buttons) == 1 and buttons[0].confidence >= .99
        modes = []
        for tick in range(640, 650 if retry else 641):
            now, effects = self._tick(sm, "chest", tick, now, buttons=buttons, stable_key="chest-close")
            modes.extend(e.mode for e in effects if e.kind == "ui_click")
            assert sm.context.state is ControllerState.CHEST
        assert modes == (["initial", "retry"] if retry else ["initial"])
        assert sm.context.ui_visit_click_count == (2 if retry else 1)
        for tick in range(650, 653):
            now, _ = self._tick(sm, "gameplay", tick, now, stable_key="gameplay-return")
        assert sm.context.state is ControllerState.GAMEPLAY

    def test_fading_chest_waits_for_stable_button_then_retries(self):
        """フェードを待ち、定常の終了ボタンで initial と retry を送る。

        本番 parser が候補を出すまでクリックせず、その後画面が残っても再送で停止しません。
        """
        sm = StateMachine()
        now, _ = _arm_to_gameplay(sm, 0)
        for tick in range(3):
            now, _ = self._tick(sm, "chest", tick, now)
        fading = self._chest_buttons(fading=True)
        assert fading == ()
        for tick in range(3, 6):
            now, effects = self._tick(sm, "chest", tick, now, buttons=fading, stable_key="chest-fade")
            assert all(e.kind != "ui_click" for e in effects)
            assert sm.context.state is ControllerState.CHEST and sm.context.ui_visit_click_count == 0
        buttons = self._chest_buttons()
        modes = []
        for tick in range(6, 16):
            now, effects = self._tick(sm, "chest", tick, now, buttons=buttons, stable_key="chest-close")
            modes.extend(e.mode for e in effects if e.kind == "ui_click")
            assert sm.context.state is ControllerState.CHEST
        assert modes == ["initial", "retry"] and sm.context.ui_visit_click_count == 2

    def test_old_five_second_chest_timeout_stops_before_close(self):
        """旧五秒 profile では演出終了前に停止する。

        既定を六十秒へ上げる理由を、同じ frame 列との比較で固定します。
        """
        sm = StateMachine(profile=NavigationProfile(chest_timeout_ns=5_000_000_000))
        now, _ = _arm_to_gameplay(sm, 0)
        for tick in range(640):
            now, _ = self._tick(sm, "chest", tick, now)
            if sm.context.state is ControllerState.DISARMED:
                break
        assert sm.context.state is ControllerState.DISARMED
        assert sm.context.terminal_reason == "chest_timeout"
        assert tick < 640 and sm.context.ui_visit_click_count == 0

    def test_invalid_card_candidates_fail_closed_after_two_seconds(self):
        """atlas なしの低信頼候補ではクリックせず二秒で停止する。

        状態を正しく level-up と認めても、未知のカードに推測で入力を送りません。
        """
        sm = StateMachine()
        now, _ = _arm_to_gameplay(sm, 0)
        candidates = (fx.make_candidate_target(validity=False, confidence=0.),)
        for tick in range(80):
            now, effects = self._tick(sm, "level_up_items", tick, now, candidates=candidates)
            assert all(e.kind != "ui_click" for e in effects)
            if sm.context.state is ControllerState.DISARMED:
                break
        assert sm.context.state is ControllerState.DISARMED
        assert sm.context.terminal_reason == "level_up_timeout"

    def test_card_transient_then_rows_enters_level_up_normally(self):
        """過渡一枚と定常カードの連続で LEVEL_UP へ入る。

        controller は confidence を見ないため、両方の raw state が同じなら通常の debounce になります。
        """
        sm = StateMachine()
        now, _ = _arm_to_gameplay(sm, 0)
        for tick in range(4):
            candidate = fx.make_candidate_target(validity=False, confidence=0. if tick == 0 else .9)
            now, _ = self._tick(sm, "level_up_items", tick, now, candidates=(candidate,),
                                stable_key="card-transient" if tick == 0 else "card-rows:3")
        assert sm.context.state is ControllerState.LEVEL_UP
        assert sm.context.ui_visit_click_count == 0
