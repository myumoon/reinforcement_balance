"""``ui_navigation.py`` の resolve/choose/execute/profile を検証する。

やさしい説明: 「意図(intent)から正しい1個のROIだけを見つけられるか」
「0件/複数件/無効/信頼度不足は本当に諦めるか」「retryの条件を厳密に守るか」
「実際にcontrollerへ渡す呼び出しが正しいか」を1つずつ確認します。
"""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest
from reinbalance_survivors_contracts.ui_intent import UiIntentKind

from survivors.controller import ui_navigation
from survivors.controller.ui_navigation import (
    Effect,
    NavigationProfile,
    build_ui_action_telemetry,
    choose_button,
    choose_card,
    choose_fallback,
    execute_effect,
    load_navigation_profile,
    resolve_ui_target,
)
from survivors.perception_snapshot import NormalizedRoi

from ._state_machine_fixtures import (
    make_button_intent,
    make_button_target,
    make_candidate_target,
    make_choose_card_intent,
    make_choose_fallback_intent,
    make_perception_snapshot,
    make_stop_intent,
)

_CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "Deployment" / "configs" / "ui_navigation_1080p_ja_v1.yaml"
)


class _FakeController:
    """`InputLeaseController` の公開 API だけを真似るテスト double。

    やさしい説明: 本物は別プロセスを起動して OS へ入力を送りますが、テストでは
    「何を呼ばれたか」を記録するだけの偽物で十分です。
    """

    def __init__(self, *, click_ack: bool = True) -> None:
        self.calls: list[tuple[str, tuple]] = []
        self._click_ack = click_ack

    def send_action(self, action_index: int) -> bool:
        self.calls.append(("send_action", (action_index,)))
        return True

    def send_ui_click(self, normalized_x: float, normalized_y: float) -> bool:
        self.calls.append(("send_ui_click", (normalized_x, normalized_y)))
        return self._click_ack

    def send_ui_key(self, key: str) -> bool:
        self.calls.append(("send_ui_key", (key,)))
        return True

    def emergency_release(self) -> bool:
        self.calls.append(("emergency_release", ()))
        return True


class TestImportBoundaries:
    """I1/I2: 所有権境界を越える import をソース文字列レベルで禁止する。"""

    def test_does_not_import_hud_or_vision_internals(self) -> None:
        source = inspect.getsource(ui_navigation)
        for forbidden in ("HudState", "hud_parser", "survivors.vision", "real_obs_assembler"):
            assert forbidden not in source

    def test_does_not_import_non_model_ui_policy_or_item_selector(self) -> None:
        source = inspect.getsource(ui_navigation)
        for forbidden in ("NonModelUiPolicy", "ItemSelector", "decide_non_model_ui_intent"):
            assert forbidden not in source


class TestResolveUiTargetInitial:
    """`resolve_ui_target(mode="initial")` の照合規則。"""

    def test_matches_single_valid_candidate(self) -> None:
        snapshot = make_perception_snapshot(
            screen_state="level_up_items",
            candidates=(make_candidate_target(choice_id="c0", choice_index=0),),
        )
        intent = make_choose_card_intent(snapshot, target_index=0)
        target = resolve_ui_target(intent, snapshot, mode="initial")
        assert target is not None
        assert target.choice_id == "c0"

    def test_rejects_zero_matches(self) -> None:
        snapshot = make_perception_snapshot(screen_state="level_up_items", candidates=())
        intent = make_choose_card_intent(snapshot, target_index=0)
        assert resolve_ui_target(intent, snapshot, mode="initial") is None

    def test_rejects_duplicate_choice_index_construction(self) -> None:
        # `UiPresentationSnapshotV1.__post_init__` 自体が choice_index の重複を
        # 拒否するため、「複数件マッチ」は resolver 到達前に上流契約で防がれている
        # ことを確認する(0件/複数件どちらも fail-closed という PR 受け入れ条件の
        # 「複数件」側は、この上流の一意性保証によって実現される)。
        with pytest.raises(ValueError):
            make_perception_snapshot(
                screen_state="level_up_items",
                candidates=(
                    make_candidate_target(choice_id="c0", choice_index=0),
                    make_candidate_target(choice_id="c1", choice_index=0),
                ),
            )

    def test_rejects_invalid_target(self) -> None:
        snapshot = make_perception_snapshot(
            screen_state="level_up_items",
            candidates=(make_candidate_target(choice_id="c0", choice_index=0, validity=False),),
        )
        intent = make_choose_card_intent(snapshot, target_index=0)
        assert resolve_ui_target(intent, snapshot, mode="initial") is None

    def test_rejects_low_confidence(self) -> None:
        snapshot = make_perception_snapshot(
            screen_state="level_up_items",
            candidates=(make_candidate_target(choice_id="c0", choice_index=0, confidence=0.1),),
        )
        intent = make_choose_card_intent(snapshot, target_index=0)
        assert resolve_ui_target(intent, snapshot, mode="initial") is None

    def test_rejects_button_without_capability(self) -> None:
        snapshot = make_perception_snapshot(
            screen_state="chest",
            buttons=(make_button_target(semantic_action="ack_chest", capability=False),),
        )
        intent = make_button_intent(snapshot, kind=UiIntentKind.ACK_CHEST, semantic_action="ack_chest")
        assert resolve_ui_target(intent, snapshot, mode="initial") is None

    def test_rejects_source_hash_mismatch(self) -> None:
        snapshot_a = make_perception_snapshot(
            screen_state="level_up_items",
            snapshot_id="snap-a",
            candidates=(make_candidate_target(choice_id="c0", choice_index=0),),
        )
        snapshot_b = make_perception_snapshot(
            screen_state="level_up_items",
            snapshot_id="snap-b",
            candidates=(make_candidate_target(choice_id="c0", choice_index=0),),
        )
        intent = make_choose_card_intent(snapshot_a, target_index=0)
        # 別 snapshot(source binding が一致しない)へ initial 解決させようとすると拒否される。
        assert resolve_ui_target(intent, snapshot_b, mode="initial") is None

    def test_rejects_stop_and_no_op(self) -> None:
        snapshot = make_perception_snapshot(screen_state="level_up_items")
        stop_intent = make_stop_intent(snapshot)
        assert resolve_ui_target(stop_intent, snapshot, mode="initial") is None

    def test_choose_card_rejects_candidate_set_hash_mismatch(self) -> None:
        snapshot = make_perception_snapshot(
            screen_state="level_up_items",
            candidates=(make_candidate_target(choice_id="c0", choice_index=0),),
        )
        intent = make_choose_card_intent(snapshot, target_index=0, candidate_set_hash="x" * 64)
        assert resolve_ui_target(intent, snapshot, mode="initial") is None

    def test_choose_fallback_matches_by_id_and_index(self) -> None:
        snapshot = make_perception_snapshot(
            screen_state="level_up_fallback",
            candidates=(
                make_candidate_target(choice_id="gold", choice_index=0, semantic_kind="fallback_reward"),
            ),
        )
        intent = make_choose_fallback_intent(snapshot, target_id="gold", target_index=0, semantic="gold")
        target = resolve_ui_target(intent, snapshot, mode="initial")
        assert target is not None and target.choice_id == "gold"


class TestResolveUiTargetRetry:
    """`resolve_ui_target(mode="retry")` の同値判定。"""

    def _initial_and_retry_snapshots(self, *, jitter: float = 0.0, confidence: float = 0.995):
        original = make_perception_snapshot(
            screen_state="chest",
            snapshot_id="chest-0",
            frame_id="f0",
            captured_ns=10,
            buttons=(make_button_target(semantic_action="ack_chest", confidence=0.99),),
        )
        newer = make_perception_snapshot(
            screen_state="chest",
            snapshot_id="chest-1",
            frame_id="f1",
            captured_ns=11,
            ui_state_key=original.ui_state_key,
            buttons=(
                make_button_target(
                    semantic_action="ack_chest",
                    roi=NormalizedRoi(0.40 + jitter, 0.40 + jitter, 0.50 + jitter, 0.50 + jitter),
                    confidence=confidence,
                ),
            ),
        )
        return original, newer

    def test_allows_retry_with_small_jitter(self) -> None:
        original, newer = self._initial_and_retry_snapshots(jitter=0.001)
        intent = make_button_intent(newer, kind=UiIntentKind.ACK_CHEST, semantic_action="ack_chest")
        target = resolve_ui_target(intent, newer, mode="retry", original_snapshot=original)
        assert target is not None

    def test_rejects_retry_without_original_snapshot(self) -> None:
        original, newer = self._initial_and_retry_snapshots()
        intent = make_button_intent(newer, kind=UiIntentKind.ACK_CHEST, semantic_action="ack_chest")
        assert resolve_ui_target(intent, newer, mode="retry") is None

    def test_rejects_same_snapshot_reuse(self) -> None:
        original, _ = self._initial_and_retry_snapshots()
        intent = make_button_intent(original, kind=UiIntentKind.ACK_CHEST, semantic_action="ack_chest")
        assert resolve_ui_target(intent, original, mode="retry", original_snapshot=original) is None

    def test_rejects_stale_snapshot_not_newer(self) -> None:
        original, newer = self._initial_and_retry_snapshots()
        # captured_ns が original 以下(同時刻)になるよう、同じ captured_ns を使う。
        stale = make_perception_snapshot(
            screen_state="chest",
            snapshot_id="chest-stale",
            frame_id="f-stale",
            captured_ns=original.captured_ns,
            ui_state_key=original.ui_state_key,
            buttons=(make_button_target(semantic_action="ack_chest", confidence=0.99),),
        )
        intent = make_button_intent(stale, kind=UiIntentKind.ACK_CHEST, semantic_action="ack_chest")
        assert resolve_ui_target(intent, stale, mode="retry", original_snapshot=original) is None

    def test_rejects_ui_state_key_change(self) -> None:
        original, _ = self._initial_and_retry_snapshots()
        changed = make_perception_snapshot(
            screen_state="chest",
            snapshot_id="chest-2",
            frame_id="f2",
            captured_ns=11,
            buttons=(make_button_target(semantic_action="ack_chest", confidence=0.99),),
        )
        intent = make_button_intent(changed, kind=UiIntentKind.ACK_CHEST, semantic_action="ack_chest")
        assert resolve_ui_target(intent, changed, mode="retry", original_snapshot=original) is None

    def test_rejects_large_displacement(self) -> None:
        original, newer = self._initial_and_retry_snapshots(jitter=0.5)
        intent = make_button_intent(newer, kind=UiIntentKind.ACK_CHEST, semantic_action="ack_chest")
        assert resolve_ui_target(intent, newer, mode="retry", original_snapshot=original) is None

    def test_rejects_low_confidence_new_target(self) -> None:
        original, newer = self._initial_and_retry_snapshots(confidence=0.5)
        intent = make_button_intent(newer, kind=UiIntentKind.ACK_CHEST, semantic_action="ack_chest")
        assert resolve_ui_target(intent, newer, mode="retry", original_snapshot=original) is None

    def test_invalid_mode_raises(self) -> None:
        original, _ = self._initial_and_retry_snapshots()
        intent = make_button_intent(original, kind=UiIntentKind.ACK_CHEST, semantic_action="ack_chest")
        with pytest.raises(ValueError):
            resolve_ui_target(intent, original, mode="bogus")  # type: ignore[arg-type]


class TestChooseHelpers:
    """``choose_card``/``choose_fallback``/``choose_button`` の kind ガード。"""

    def test_choose_card_returns_none_for_other_kind(self) -> None:
        snapshot = make_perception_snapshot(
            screen_state="level_up_fallback",
            candidates=(make_candidate_target(choice_id="gold", choice_index=0, semantic_kind="fallback_reward"),),
        )
        intent = make_choose_fallback_intent(snapshot, target_id="gold", target_index=0, semantic="gold")
        assert choose_card(intent, snapshot.ui_presentation) is None
        assert choose_fallback(intent, snapshot.ui_presentation) is not None
        assert choose_button(intent, snapshot.ui_presentation) is None

    def test_choose_button_matches_semantic_and_capability(self) -> None:
        snapshot = make_perception_snapshot(
            screen_state="chest", buttons=(make_button_target(semantic_action="ack_chest"),)
        )
        intent = make_button_intent(snapshot, kind=UiIntentKind.ACK_CHEST, semantic_action="ack_chest")
        target = choose_button(intent, snapshot.ui_presentation)
        assert target is not None and target.semantic_action == "ack_chest"


class TestExecuteEffect:
    """`execute_effect` が `InputLeaseController` の公開 API だけを正しい引数で呼ぶ。"""

    def test_move_calls_send_action(self) -> None:
        controller = _FakeController()
        outcome = execute_effect(Effect(kind="move", action_index=3), controller)
        assert controller.calls == [("send_action", (3,))]
        assert outcome.ack is True

    def test_ui_click_uses_roi_center(self) -> None:
        controller = _FakeController()
        target = make_candidate_target(roi=NormalizedRoi(0.2, 0.4, 0.4, 0.6))
        outcome = execute_effect(Effect(kind="ui_click", target=target), controller)
        assert len(controller.calls) == 1
        name, args = controller.calls[0]
        assert name == "send_ui_click"
        assert args == pytest.approx((0.3, 0.5))
        assert outcome.ack is True

    def test_ui_click_ack_reflects_gate_result_only(self) -> None:
        # M12: ack は OS 受理の意味だけであり、ここでは常に True/False をそのまま反映する
        # (ゲーム側の適用成否は別途 snapshot で確認する)。
        controller = _FakeController(click_ack=False)
        target = make_candidate_target()
        outcome = execute_effect(Effect(kind="ui_click", target=target), controller)
        assert outcome.ack is False

    def test_ui_key_calls_send_ui_key(self) -> None:
        controller = _FakeController()
        execute_effect(Effect(kind="ui_key", key="ESCAPE"), controller)
        assert controller.calls == [("send_ui_key", ("ESCAPE",))]

    def test_release_all_calls_emergency_release(self) -> None:
        controller = _FakeController()
        execute_effect(Effect(kind="release_all"), controller)
        assert controller.calls == [("emergency_release", ())]

    @pytest.mark.parametrize("kind", ["combat_reset", "controller_stop", "process_terminate"])
    def test_control_effects_do_not_touch_controller(self, kind: str) -> None:
        controller = _FakeController()
        outcome = execute_effect(Effect(kind=kind), controller)
        assert controller.calls == []
        assert outcome.ack is True

    def test_unknown_kind_raises(self) -> None:
        with pytest.raises(ValueError):
            execute_effect(Effect(kind="bogus"), _FakeController())  # type: ignore[arg-type]


class TestNavigationProfile:
    """timeout/retry/confidence プロファイルの読み込みと検証。"""

    def test_default_v1_has_positive_timeouts(self) -> None:
        profile = NavigationProfile.default_v1()
        for key in ("level_up", "chest", "target_reached", "unknown", "run_setup"):
            assert profile.timeout_ns_for(key) > 0

    def test_rejects_bad_schema_version(self) -> None:
        with pytest.raises(ValueError):
            NavigationProfile(schema_version="bogus")

    def test_load_shipped_yaml_matches_documented_defaults(self) -> None:
        profile = load_navigation_profile(_CONFIG_PATH)
        assert profile.debounce_frames == 3
        assert profile.retry_budget == 1
        assert profile.timeout_ns_for("level_up") == 2_000_000_000
        assert profile.timeout_ns_for("chest") == 5_000_000_000
        assert profile.timeout_ns_for("target_reached") == 5_000_000_000
        assert profile.timeout_ns_for("unknown") == 1_000_000_000
        assert profile.timeout_ns_for("run_setup") == 15_000_000_000

    def test_shipped_yaml_has_no_pixel_or_normalized_coordinates(self) -> None:
        # I3 相当: 設定ファイル自体が固定座標を持たないことを保証する。
        text = _CONFIG_PATH.read_text(encoding="utf-8")
        for forbidden in ("roi", "normalized_x", "normalized_y", "pixel", "coordinate"):
            assert forbidden not in text.lower()

    def test_from_wire_rejects_unknown_field(self) -> None:
        with pytest.raises(ValueError):
            NavigationProfile.from_wire({"schema_version": NavigationProfile.default_v1().schema_version, "bogus": 1})


class TestBuildUiActionTelemetry:
    """M12: ack(OS受理)と telemetry(適用確認用の別記録)を区別できる。"""

    def test_telemetry_records_identity_without_raw_roi(self) -> None:
        snapshot = make_perception_snapshot(
            screen_state="chest", buttons=(make_button_target(semantic_action="ack_chest"),)
        )
        intent = make_button_intent(snapshot, kind=UiIntentKind.ACK_CHEST, semantic_action="ack_chest")
        target = choose_button(intent, snapshot.ui_presentation)
        telemetry = build_ui_action_telemetry(intent, target, "initial", snapshot)
        assert telemetry["semantic_action"] == "ack_chest"
        assert telemetry["mode"] == "initial"
        assert "roi" not in telemetry
        assert telemetry["ui_state_key"] == snapshot.ui_state_key
