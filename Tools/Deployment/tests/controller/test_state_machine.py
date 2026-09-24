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
from survivors.perception_snapshot import NormalizedRoi

from . import _state_machine_fixtures as fx

_TICK_NS = 16_000_000  # 約 60fps 相当の1フレーム。


def _arm_to_gameplay(
    sm: StateMachine, now_ns: int, *, mode: CampaignRunMode = CampaignRunMode.OPERATOR_DEBUG_RESTART
) -> tuple[int, object]:
    """``StateMachine`` を arm し、GAMEPLAY へ確定させるところまで進める。"""
    sm.arm(campaign_run_mode=mode, run_id="run-1", gameplay_attempt_id="attempt-1", now_ns=now_ns)
    gp = fx.make_perception_snapshot(screen_state="gameplay", snapshot_id="gp-boot", frame_id="fr-boot")
    for _ in range(6):
        now_ns += _TICK_NS
        sm.step(gp, fx.make_move_decision(gp), now_ns=now_ns)
    assert sm.context.state is ControllerState.GAMEPLAY
    return now_ns, gp


def _drive(sm: StateMachine, snapshot, decision, n: int, now_ns: int):
    """同じ (snapshot, decision) を n tick 分送り、最後の effect を返す。"""
    effects: tuple = ()
    for _ in range(n):
        now_ns += _TICK_NS
        effects = sm.step(snapshot, decision, now_ns=now_ns)
    return now_ns, effects


class TestImportBoundaries:
    """I1/I2: 所有権境界を越える import をソース文字列レベルで禁止する。"""

    def test_does_not_import_hud_or_vision_internals(self) -> None:
        source = inspect.getsource(state_machine)
        for forbidden in ("HudState", "hud_parser", "survivors.vision", "real_obs_assembler"):
            assert forbidden not in source

    def test_does_not_import_non_model_ui_policy_or_item_selector(self) -> None:
        source = inspect.getsource(state_machine)
        for forbidden in ("NonModelUiPolicy", "ItemSelector", "decide_non_model_ui_intent"):
            assert forbidden not in source

    def test_does_not_reference_os_input_apis(self) -> None:
        # input driver は effect executor(ui_navigation.execute_effect)からしか
        # 呼ばれない: state_machine.py 自体は OS 入力 API を一切知らない。
        source = inspect.getsource(state_machine)
        for forbidden in ("SendInput", "pyautogui", "win32api", "ctypes", "keyboard"):
            assert forbidden not in source


class TestScreenClassification:
    """``classify_screen_state`` の raw screen_state -> ControllerState 分類。"""

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
        assert classify_screen_state(raw) is expected

    def test_unrecognized_string_is_unknown(self) -> None:
        assert classify_screen_state("some_future_screen_name") is ControllerState.UNKNOWN

    def test_focus_loss_forces_unknown_even_for_gameplay(self) -> None:
        assert classify_screen_state("gameplay", window_focused=False) is ControllerState.UNKNOWN


class TestArmAndAcquireTarget:
    """DISARMED -> ACQUIRE_TARGET -> RUN_SETUP -> GAMEPLAY の起動シーケンス。"""

    def test_disarmed_ignores_all_input_until_armed(self) -> None:
        sm = StateMachine()
        gp = fx.make_perception_snapshot(screen_state="gameplay")
        effects = sm.step(gp, fx.make_move_decision(gp), now_ns=1_000_000_000)
        assert effects == ()
        assert sm.context.state is ControllerState.DISARMED

    def test_arm_requires_disarmed_state(self) -> None:
        sm = StateMachine()
        sm.arm(campaign_run_mode=CampaignRunMode.OPERATOR_DEBUG_RESTART, run_id="r", gameplay_attempt_id="a", now_ns=0)
        with pytest.raises(ValueError):
            sm.arm(campaign_run_mode=CampaignRunMode.OPERATOR_DEBUG_RESTART, run_id="r", gameplay_attempt_id="a", now_ns=1)

    def test_three_frame_debounce_required_before_run_setup(self) -> None:
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
    """GAMEPLAY 中の movement effect と、それ以外の kind の扱い。"""

    def test_move_decision_forwards_action_index(self) -> None:
        sm = StateMachine()
        now, gp = _arm_to_gameplay(sm, 0)
        now += _TICK_NS
        effects = sm.step(gp, fx.make_move_decision(gp, action_index=5), now_ns=now)
        assert len(effects) == 1
        assert effects[0].kind == "move"
        assert effects[0].action_index == 5

    def test_no_op_decision_produces_no_effect(self) -> None:
        sm = StateMachine()
        now, gp = _arm_to_gameplay(sm, 0)
        now += _TICK_NS
        effects = sm.step(gp, fx.make_no_op_decision(gp), now_ns=now)
        assert effects == ()


class TestLevelUpOrderingAndRetry:
    """M3/M4: gameplay -> level_up の順序、retry/stuck 挙動。"""

    def _enter_level_up(self, sm: StateMachine, now: int, *, choice_index: int = 0):
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
        sm = StateMachine()
        now, _gp = _arm_to_gameplay(sm, 0)
        now, lu, intent, decision, entry_effects = self._enter_level_up(sm, now)
        assert sm.context.state is ControllerState.LEVEL_UP
        assert entry_effects == ()  # 遷移確定 tick そのものはまだ click しない。
        # 次の tick で click effect が出るが、move は一度も混ざらない。
        now += _TICK_NS
        effects = sm.step(lu, decision, now_ns=now)
        kinds = [effect.kind for effect in effects]
        assert "move" not in kinds

    def test_click_ordering_release_before_click_and_resume_after_ack(self) -> None:
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

    def test_retry_allowed_once_with_small_roi_jitter(self) -> None:
        sm = StateMachine()
        now, gp = _arm_to_gameplay(sm, 0)
        now, lu, intent, decision, _ = self._enter_level_up(sm, now)
        now += _TICK_NS
        sm.step(lu, decision, now_ns=now)

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
        now += _TICK_NS
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
        now += _TICK_NS
        effects = sm.step(jittered2, fx.make_ui_decision(jittered2, retry_intent2), now_ns=now)
        assert effects == ()

    def test_choice_count_change_mid_attempt_fails_closed(self) -> None:
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
        sm = StateMachine()
        now, gp = _arm_to_gameplay(sm, 0)
        now, lu, intent, decision, _ = self._enter_level_up(sm, now)
        chest = fx.make_perception_snapshot(screen_state="chest", snapshot_id="chest-surprise")
        for _ in range(3):
            now += _TICK_NS
            sm.step(chest, fx.make_no_op_decision(chest), now_ns=now)
        assert sm.context.state is ControllerState.DISARMED


class TestChestAndConfirmButtons:
    """chest(ack_chest)/target_reached(confirm)の button click。"""

    def test_chest_click_uses_button_target(self) -> None:
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
    """M7: 30:00 evidence 以降は success を最優先で COMPLETE へ収束する。"""

    def _reach_target(self, sm: StateMachine, now: int):
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

    def test_target_reached_timeout_still_completes(self) -> None:
        sm = StateMachine()
        sm.arm(campaign_run_mode=CampaignRunMode.OPERATOR_DEBUG_RESTART, run_id="r", gameplay_attempt_id="a", now_ns=0)
        now, tr = self._reach_target(sm, 0)
        for _ in range(2000):
            now += _TICK_NS
            sm.step(tr, fx.make_no_op_decision(tr), now_ns=now)
            if sm.context.terminal_locked:
                break
        assert sm.context.state is ControllerState.COMPLETE

    def test_pre_target_death_is_failure_in_formal_mode(self) -> None:
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
    """M10: death/result 侵入ごとに combat_reset がちょうど1回出る。"""

    def test_combat_reset_emitted_once_per_death_result_entry(self) -> None:
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
    """M8/M9: terminal effect の順序・exactly-once、2回目 GAMEPLAY entry の拒否。"""

    def test_terminal_effects_are_release_then_stop_then_terminate(self) -> None:
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


class TestPausedUnknownRecover:
    """PAUSED(indefinite)/UNKNOWN(1s timeout)/RECOVER(再確定バッファ、input 0)。"""

    def test_paused_waits_indefinitely_without_any_effect(self) -> None:
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
        sm = StateMachine()
        now, gp = _arm_to_gameplay(sm, 0)
        for _ in range(3):
            now += _TICK_NS
            sm.step(gp, fx.make_no_op_decision(gp), now_ns=now, window_focused=False)
        assert sm.context.state is ControllerState.UNKNOWN


class TestGlobalSafetySignals:
    """manual abort / ``kind="stop"`` は、どの状態からでも最優先で emergency stop する。"""

    def test_manual_abort_overrides_everything(self) -> None:
        sm = StateMachine()
        now, gp = _arm_to_gameplay(sm, 0, mode=CampaignRunMode.FORMAL_SINGLE_ATTEMPT)
        now += _TICK_NS
        effects = sm.step(gp, fx.make_move_decision(gp), now_ns=now, manual_abort_requested=True)
        assert sm.context.state is ControllerState.FORMAL_RUN_TERMINAL_FAILURE
        assert [e.kind for e in effects] == ["release_all", "controller_stop", "process_terminate"]

    def test_stop_decision_triggers_immediate_emergency_stop_from_gameplay(self) -> None:
        sm = StateMachine()
        now, gp = _arm_to_gameplay(sm, 0, mode=CampaignRunMode.OPERATOR_DEBUG_RESTART)
        now += _TICK_NS
        effects = sm.step(gp, fx.make_stop_decision(gp), now_ns=now)
        assert sm.context.state is ControllerState.DISARMED
        assert effects and effects[0].kind == "release_all"

    def test_stop_decision_triggers_immediate_emergency_stop_from_level_up(self) -> None:
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


class TestGoldenEndToEnd:
    """04-09/05-01 契約に沿った parser -> assembler -> AgentDecision -> resolver 相当の end-to-end 再生。

    やさしい説明: 04-09/05-01 の実物パーサ/アセンブラは import しませんが、
    それらが最終的に生成する形(PerceptionSnapshot + AgentDecision +
    UiIntentV1)を fixture で再現し、resolver が「選択 semantic だけ」を
    対応 ROI へ解決して安全な入力列になることを end-to-end で確認します。
    """

    def test_full_run_reaches_complete_with_expected_effect_sequence(self) -> None:
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
    """M15: random/noisy state sequence でも安全不変条件が破れない。"""

    _SCREEN_STATES = (
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

    def test_random_noisy_sequences_never_violate_safety_invariants(self) -> None:
        rng = random.Random(20260925)
        for trial in range(20):
            sm = StateMachine()
            mode = CampaignRunMode.FORMAL_SINGLE_ATTEMPT if trial % 2 else CampaignRunMode.OPERATOR_DEBUG_RESTART
            sm.arm(campaign_run_mode=mode, run_id=f"noise-{trial}", gameplay_attempt_id="a", now_ns=0)
            now = 0
            for tick in range(150):
                now += _TICK_NS
                raw_state = rng.choice(self._SCREEN_STATES)
                snapshot = fx.make_perception_snapshot(
                    screen_state=raw_state,
                    snapshot_id=f"noise-{trial}-{tick}",
                    frame_id=f"noise-frame-{trial}-{tick}",
                    captured_ns=now,
                    candidates=(
                        (fx.make_candidate_target(choice_id="c0", choice_index=0),)
                        if raw_state == "level_up_items"
                        else ()
                    ),
                    buttons=(
                        (fx.make_button_target(semantic_action="ack_chest"),)
                        if raw_state == "chest"
                        else ()
                    ),
                )
                pre_state = sm.context.state
                if pre_state is ControllerState.GAMEPLAY and raw_state == "gameplay":
                    decision = fx.make_move_decision(snapshot, action_index=rng.randrange(9))
                elif raw_state in ("level_up_items",) and pre_state is ControllerState.LEVEL_UP:
                    intent = fx.make_choose_card_intent(snapshot, target_index=0)
                    decision = fx.make_ui_decision(snapshot, intent)
                elif raw_state == "chest" and pre_state is ControllerState.CHEST:
                    intent = fx.make_button_intent(
                        snapshot, kind=UiIntentKind.ACK_CHEST, semantic_action="ack_chest"
                    )
                    decision = fx.make_ui_decision(snapshot, intent)
                else:
                    decision = fx.make_no_op_decision(snapshot)

                if sm.context.terminal_locked:
                    effects = sm.step(snapshot, decision, now_ns=now)
                    assert effects == (), "terminal 確定後に effect が出てはならない"
                    continue

                effects = sm.step(snapshot, decision, now_ns=now)
                kinds = [effect.kind for effect in effects]

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
