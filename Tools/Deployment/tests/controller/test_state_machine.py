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
        # overall_review fix: GAMEPLAY を抜ける確定 tick 自体で release_all を
        # 1回発行する(movement lease の自然失効任せにしない)。まだ click は出ない。
        assert [e.kind for e in entry_effects] == ["release_all"]
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

    def test_retry_withheld_within_ack_wait_window(self) -> None:
        # M4 fix: 初回 click 直後、ack 待ち window(既定 200ms)が経過するまでは
        # newer snapshot で precondition を満たしていても retry してはならない
        # (capture/perception 遅延中の二重click防止)。
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
        assert sm.context.state is ControllerState.LEVEL_UP
        assert not sm.context.terminal_locked

    def test_reroll_then_choose_new_candidate_is_not_emergency_stop(self) -> None:
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
        # apply ack を認識してもなお、今tickのintentが古いsnapshotに束縛された
        # ままなら(遅延で旧intentが届いた等)、initial resolve のsource binding
        # チェックに落ちて再送されない。
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
        now, effects = self._drive_until_ack_confirmed(sm, rerolled, stale_decision, now)
        assert effects == ()
        assert sm.context.ui_attempt is None
        assert sm.context.state is ControllerState.LEVEL_UP
        assert not sm.context.terminal_locked


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

    def test_target_reached_timeout_without_confirmation_fails_closed_debug(self) -> None:
        # M7(b) fix: post-30 event(gameplay 再開 or death/result)を確認できない
        # まま timeout した場合は、以前のように無条件で COMPLETE にはしない。
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


class TestArmResetsRunContext:
    """M9: ``arm()`` は profile 以外の実行文脈を毎回まっさらに作り直す。"""

    def test_rearm_after_debug_target_reached_timeout_resets_stale_success_latch(self) -> None:
        # 前 run が TARGET_REACHED まで到達した(success_latched=True)まま
        # debug モードで fail-closed(DISARMED)し、再arm した場合、新しい run の
        # pre-30 death が誤って COMPLETE になってはならない。
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
        new_ctx, effects = self._confirm(
            ControllerState.GAMEPLAY, ControllerState.GAMEPLAY, CampaignRunMode.OPERATOR_DEBUG_RESTART, move_action=4
        )
        assert new_ctx.state is ControllerState.GAMEPLAY
        assert [effect.kind for effect in effects] == ["move"]
        assert effects[0].action_index == 4

    def test_target_reached_entry_from_gameplay_sets_success_latch(self) -> None:
        new_ctx, _effects = self._confirm(
            ControllerState.GAMEPLAY,
            ControllerState.TARGET_REACHED_PENDING_TRANSITION,
            CampaignRunMode.OPERATOR_DEBUG_RESTART,
        )
        assert new_ctx.success_latched is True

    def test_target_reached_entry_from_unknown_sets_success_latch(self) -> None:
        # M7(a) の table-driven 側の固定: UNKNOWN 経由でも success_latched が立つ。
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
            # M15 fix: 同一 UiAttempt(同じ original_snapshot に束縛された試行)
            # あたりの click(ui_click/ui_key)が initial+retry の最大2件を
            # 超えないことを、attempt の identity が変わるたびに数え直して検証する。
            prev_attempt_key: tuple[str, tuple] | None = None
            attempt_click_count = 0

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

                # M15 fix: 同一 UiAttempt(同じ original_snapshot に束縛された試行)
                # あたりの click は initial+retry の最大2件まで
                # (ui_state_key/candidate_set_hash が揺れても無制限 re-click しない)。
                attempt = sm.context.ui_attempt
                attempt_key = (
                    (attempt.original_snapshot.snapshot_id, attempt.intent_identity) if attempt is not None else None
                )
                if attempt_key != prev_attempt_key:
                    attempt_click_count = 0
                prev_attempt_key = attempt_key
                attempt_click_count += kinds.count("ui_click") + kinds.count("ui_key")
                assert attempt_click_count <= 2, "同一UI状態訪問(1 attempt)あたりのclickはinitial+retryの最大2件まで"

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
