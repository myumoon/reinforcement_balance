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
        """送信結果を記録する controller の代役を用意する。

        クリックの受理可否を指定し、入力要求の一覧を空から始めます。
        """
        self.calls: list[tuple[str, tuple]] = []
        self._click_ack = click_ack

    def send_action(self, action_index: int) -> bool:
        """移動 action の要求を記録する。

        実入力を送らず、呼び出された action index をテストから参照できるようにします。
        """
        self.calls.append(("send_action", (action_index,)))
        return True

    def send_ui_click(self, normalized_x: float, normalized_y: float) -> bool:
        """クリック座標と受理結果を記録する。

        ROI 解決で求めた位置が公開 API にそのまま渡ることを調べます。
        """
        self.calls.append(("send_ui_click", (normalized_x, normalized_y)))
        return self._click_ack

    def send_ui_key(self, key: str) -> bool:
        """UI key の要求を記録する。

        Enter などのキー操作がクリックと区別されることを確認できます。
        """
        self.calls.append(("send_ui_key", (key,)))
        return True

    def emergency_release(self) -> bool:
        """全入力の解除要求を記録する。

        停止時に controller の公開解除 API が呼ばれたことを確認します。
        """
        self.calls.append(("emergency_release", ()))
        return True


class TestImportBoundaries:
    """UI navigation の依存境界を検証する。

    画素解析や item 選択の内部処理を navigation に取り込まないことを確認します。
    """

    def test_does_not_import_hud_or_vision_internals(self) -> None:
        """navigation が vision 内部を import しない。

        座標解決は perception の公開 target を使い、HUD の解析結果を直接参照しません。
        """
        source = inspect.getsource(ui_navigation)
        for forbidden in ("HudState", "hud_parser", "survivors.vision", "real_obs_assembler"):
            assert forbidden not in source

    def test_does_not_import_non_model_ui_policy_or_item_selector(self) -> None:
        """navigation が選択 policy を import しない。

        何を選ぶかの判断を、クリック先の解決から独立させます。
        """
        source = inspect.getsource(ui_navigation)
        for forbidden in ("NonModelUiPolicy", "ItemSelector", "decide_non_model_ui_intent"):
            assert forbidden not in source


class TestResolveUiTargetInitial:
    """初回の target 解決を検証する。

    source binding と候補の有効性を揃えた場合だけ送信先が得られるかを確認します。
    """

    def test_matches_single_valid_candidate(self) -> None:
        """有効な一候補を初回 target として返す。

        intent と snapshot が対応していれば、指定番号の矩形を使えます。
        """
        snapshot = make_perception_snapshot(
            screen_state="level_up_items",
            candidates=(make_candidate_target(choice_id="c0", choice_index=0),),
        )
        intent = make_choose_card_intent(snapshot, target_index=0)
        target = resolve_ui_target(intent, snapshot, mode="initial")
        assert target is not None
        assert target.choice_id == "c0"

    def test_rejects_zero_matches(self) -> None:
        """該当候補がなければクリック先を返さない。

        空の選択肢を与えたとき、別の target を推測しません。
        """
        snapshot = make_perception_snapshot(screen_state="level_up_items", candidates=())
        intent = make_choose_card_intent(snapshot, target_index=0)
        assert resolve_ui_target(intent, snapshot, mode="initial") is None

    def test_rejects_duplicate_choice_index_construction(self) -> None:
        # `UiPresentationSnapshotV1.__post_init__` 自体が choice_index の重複を
        # 拒否するため、「複数件マッチ」は resolver 到達前に上流契約で防がれている
        # ことを確認する(0件/複数件どちらも fail-closed という PR 受け入れ条件の
        # 「複数件」側は、この上流の一意性保証によって実現される)。
        """候補番号の重複を構築時に拒否する。

        一つの番号が複数のカードへ解決される曖昧な snapshot を作れません。
        """
        with pytest.raises(ValueError):
            make_perception_snapshot(
                screen_state="level_up_items",
                candidates=(
                    make_candidate_target(choice_id="c0", choice_index=0),
                    make_candidate_target(choice_id="c1", choice_index=0),
                ),
            )

    def test_rejects_invalid_target(self) -> None:
        """invalid な候補を送信先にしない。

        名前と番号が合っても、観測の有効性が不足すれば解決を拒否します。
        """
        snapshot = make_perception_snapshot(
            screen_state="level_up_items",
            candidates=(make_candidate_target(choice_id="c0", choice_index=0, validity=False),),
        )
        intent = make_choose_card_intent(snapshot, target_index=0)
        assert resolve_ui_target(intent, snapshot, mode="initial") is None

    def test_rejects_low_confidence(self) -> None:
        """最低信頼度に届かない target を拒否する。

        不確かな矩形へクリックする初回経路を止めます。
        """
        snapshot = make_perception_snapshot(
            screen_state="level_up_items",
            candidates=(make_candidate_target(choice_id="c0", choice_index=0, confidence=0.1),),
        )
        intent = make_choose_card_intent(snapshot, target_index=0)
        assert resolve_ui_target(intent, snapshot, mode="initial") is None

    def test_rejects_button_without_capability(self) -> None:
        """能力が無効なボタンへ解決しない。

        ボタンの矩形が見えても、その操作を使える証拠が必要です。
        """
        snapshot = make_perception_snapshot(
            screen_state="chest",
            buttons=(make_button_target(semantic_action="ack_chest", capability=False),),
        )
        intent = make_button_intent(snapshot, kind=UiIntentKind.ACK_CHEST, semantic_action="ack_chest")
        assert resolve_ui_target(intent, snapshot, mode="initial") is None

    def test_rejects_source_hash_mismatch(self) -> None:
        """元の観測 hash が違う intent を拒否する。

        別 frame や別 content の判断が今の画面へ混ざらないことを確認します。
        """
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
        """stop と no-op は UI target を持たない。

        入力しない判断を誤ってクリック操作へ変換しません。
        """
        snapshot = make_perception_snapshot(screen_state="level_up_items")
        stop_intent = make_stop_intent(snapshot)
        assert resolve_ui_target(stop_intent, snapshot, mode="initial") is None

    def test_choose_card_rejects_candidate_set_hash_mismatch(self) -> None:
        """カード集合 hash の違いを拒否する。

        同じ番号でも選択肢が入れ替わった画面には古い intent を使いません。
        """
        snapshot = make_perception_snapshot(
            screen_state="level_up_items",
            candidates=(make_candidate_target(choice_id="c0", choice_index=0),),
        )
        intent = make_choose_card_intent(snapshot, target_index=0, candidate_set_hash="x" * 64)
        assert resolve_ui_target(intent, snapshot, mode="initial") is None

    def test_choose_fallback_matches_by_id_and_index(self) -> None:
        """fallback の ID と番号を同時に照合する。

        gold と chicken を位置だけで取り違えないことを確認します。
        """
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
    """再送 target の同等性を検証する。

    時系列、UI key、矩形の揺れと信頼度の全条件を満たす場合だけ retry を認めます。
    """

    def _initial_and_retry_snapshots(self, *, jitter: float = 0.0, confidence: float = 0.995):
        """初回と少し動いた再観測を作る。

        UI key は揃えて撮影時刻を進め、変位と confidence を個別に変更できます。
        """
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
        """小さい矩形の揺れなら retry を認める。

        同じボタンの新しい観測を、許容変位の範囲で再利用できます。
        """
        original, newer = self._initial_and_retry_snapshots(jitter=0.001)
        intent = make_button_intent(newer, kind=UiIntentKind.ACK_CHEST, semantic_action="ack_chest")
        target = resolve_ui_target(intent, newer, mode="retry", original_snapshot=original)
        assert target is not None

    def test_rejects_retry_without_original_snapshot(self) -> None:
        """初回 snapshot のない retry を拒否する。

        比較元がなければ同じ target かどうかを推測しません。
        """
        original, newer = self._initial_and_retry_snapshots()
        intent = make_button_intent(newer, kind=UiIntentKind.ACK_CHEST, semantic_action="ack_chest")
        assert resolve_ui_target(intent, newer, mode="retry") is None

    def test_rejects_same_snapshot_reuse(self) -> None:
        """同じ snapshot の再利用を retry にしない。

        新しい観測を得ていない重複送信を止めます。
        """
        original, _ = self._initial_and_retry_snapshots()
        intent = make_button_intent(original, kind=UiIntentKind.ACK_CHEST, semantic_action="ack_chest")
        assert resolve_ui_target(intent, original, mode="retry", original_snapshot=original) is None

    def test_rejects_stale_snapshot_not_newer(self) -> None:
        """撮影時刻が進んでいない再観測を拒否する。

        snapshot ID が違っても古い frame なら送信しません。
        """
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
        """UI 状態 key が変わった retry を拒否する。

        別の画面へ移った後に同じ操作を再送しません。
        """
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
        """大きく動いた矩形の retry を拒否する。

        同じ操作名でも別の位置にある target は同等と扱いません。
        """
        original, newer = self._initial_and_retry_snapshots(jitter=0.5)
        intent = make_button_intent(newer, kind=UiIntentKind.ACK_CHEST, semantic_action="ack_chest")
        assert resolve_ui_target(intent, newer, mode="retry", original_snapshot=original) is None

    def test_rejects_low_confidence_new_target(self) -> None:
        """再観測の低信頼 target を拒否する。

        初回の名前が明確でも、今の画面が不確かなら retry しません。
        """
        original, newer = self._initial_and_retry_snapshots(confidence=0.5)
        intent = make_button_intent(newer, kind=UiIntentKind.ACK_CHEST, semantic_action="ack_chest")
        assert resolve_ui_target(intent, newer, mode="retry", original_snapshot=original) is None

    def test_invalid_mode_raises(self) -> None:
        """未定義の解決 mode を拒否する。

        initial と retry 以外の文字列を黙って初回操作として使いません。
        """
        original, _ = self._initial_and_retry_snapshots()
        intent = make_button_intent(original, kind=UiIntentKind.ACK_CHEST, semantic_action="ack_chest")
        with pytest.raises(ValueError):
            resolve_ui_target(intent, original, mode="bogus")  # type: ignore[arg-type]


class TestResolveUiTargetMinConfidence:
    """profile の最低信頼度が解決へ反映される。

    既定値と厳しい設定で同じ候補の受け付け方が変わることを確認します。
    """

    def test_default_threshold_accepts_moderate_confidence(self) -> None:
        """既定の閾値を満たす中程度の信頼度を受け付ける。

        完全一致に限らず、profile が許す初回 target を解決できます。
        """
        snapshot = make_perception_snapshot(
            screen_state="level_up_items",
            candidates=(make_candidate_target(choice_id="c0", choice_index=0, confidence=0.8),),
        )
        intent = make_choose_card_intent(snapshot, target_index=0)
        assert resolve_ui_target(intent, snapshot, mode="initial") is not None

    def test_profile_min_confidence_rejects_below_threshold(self) -> None:
        # min_target_confidence を運用者が 0.9 に上げると、0.8 の target は拒否される
        # (以前は _lookup_target がモジュール定数 0.5 を hardcode していて無視されていた)。
        """profile の閾値未満の候補を拒否する。

        設定を厳しくすると低信頼カードへ入力が送られません。
        """
        snapshot = make_perception_snapshot(
            screen_state="level_up_items",
            candidates=(make_candidate_target(choice_id="c0", choice_index=0, confidence=0.8),),
        )
        intent = make_choose_card_intent(snapshot, target_index=0)
        assert resolve_ui_target(intent, snapshot, mode="initial", min_confidence=0.9) is None


class TestChooseHelpers:
    """選択 helper の intent 種別ガードを検証する。

    カード・fallback・ボタンの入口を取り違えた場合は解決しません。
    """

    def test_choose_card_returns_none_for_other_kind(self) -> None:
        """カード helper は他の intent 種別を拒否する。

        ボタン操作をカード番号へ変換しないことを確認します。
        """
        snapshot = make_perception_snapshot(
            screen_state="level_up_fallback",
            candidates=(make_candidate_target(choice_id="gold", choice_index=0, semantic_kind="fallback_reward"),),
        )
        intent = make_choose_fallback_intent(snapshot, target_id="gold", target_index=0, semantic="gold")
        assert choose_card(intent, snapshot.ui_presentation) is None
        assert choose_fallback(intent, snapshot.ui_presentation) is not None
        assert choose_button(intent, snapshot.ui_presentation) is None

    def test_choose_button_matches_semantic_and_capability(self) -> None:
        """ボタン helper は操作名と能力を照合する。

        使用可能な ack_chest だけが送信 target になります。
        """
        snapshot = make_perception_snapshot(
            screen_state="chest", buttons=(make_button_target(semantic_action="ack_chest"),)
        )
        intent = make_button_intent(snapshot, kind=UiIntentKind.ACK_CHEST, semantic_action="ack_chest")
        target = choose_button(intent, snapshot.ui_presentation)
        assert target is not None and target.semantic_action == "ack_chest"


class TestExecuteEffect:
    """effect から controller の公開 API への変換を検証する。

    移動・クリック・キー・解除を対応する送信先へ分けます。
    """

    def test_move_calls_send_action(self) -> None:
        """move effect は action index を送る。

        移動の番号を変更せず controller の send_action へ渡します。
        """
        controller = _FakeController()
        outcome = execute_effect(Effect(kind="move", action_index=3), controller)
        assert controller.calls == [("send_action", (3,))]
        assert outcome.ack is True

    def test_ui_click_uses_roi_center(self) -> None:
        """クリックは ROI の中心座標を使う。

        矩形の端ではなく中央が送信されることを確認します。
        """
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
        """クリック ack は送信 gate の受理結果を表す。

        OS に受け付けられたこととゲーム内の適用確認を混同しません。
        """
        controller = _FakeController(click_ack=False)
        target = make_candidate_target()
        outcome = execute_effect(Effect(kind="ui_click", target=target), controller)
        assert outcome.ack is False

    def test_ui_key_calls_send_ui_key(self) -> None:
        """key effect は UI key の公開 API を呼ぶ。

        Enter の送信が座標クリックへ変換されないことを確認します。
        """
        controller = _FakeController()
        execute_effect(Effect(kind="ui_key", key="ESCAPE"), controller)
        assert controller.calls == [("send_ui_key", ("ESCAPE",))]

    def test_release_all_calls_emergency_release(self) -> None:
        """release effect は全入力を解除する。

        停止時に held input を残さない公開経路を呼びます。
        """
        controller = _FakeController()
        execute_effect(Effect(kind="release_all"), controller)
        assert controller.calls == [("emergency_release", ())]

    @pytest.mark.parametrize("kind", ["combat_reset", "controller_stop", "process_terminate"])
    def test_control_effects_do_not_touch_controller(self, kind: str) -> None:
        """制御用 effect は入力 controller に触れない。

        状態更新や終了通知を実キー操作として送信しません。
        """
        controller = _FakeController()
        outcome = execute_effect(Effect(kind=kind), controller)
        assert controller.calls == []
        assert outcome.ack is True

    def test_unknown_kind_raises(self) -> None:
        """未知の effect 種別を拒否する。

        綴り違いを黙って入力なしの成功にしません。
        """
        with pytest.raises(ValueError):
            execute_effect(Effect(kind="bogus"), _FakeController())  # type: ignore[arg-type]


class TestNavigationProfile:
    """timeout と retry の設定読込を検証する。

    出荷 YAML とコード既定の値、未知項目や無効な範囲を確認します。
    """

    def test_default_v1_has_positive_timeouts(self) -> None:
        """既定 timeout は正の値で宝箱は六十秒になる。

        約二十一秒の演出を待てる時間を確保し、他の状態も有効な待ち時間を持ちます。
        """
        profile = NavigationProfile.default_v1()
        assert profile.timeout_ns_for("chest") == 60_000_000_000
        for key in ("level_up", "chest", "target_reached", "unknown", "run_setup"):
            assert profile.timeout_ns_for(key) > 0

    def test_rejects_bad_schema_version(self) -> None:
        """別 schema の navigation profile を拒否する。

        想定外の設定形式で timeout を読み違えないことを確認します。
        """
        with pytest.raises(ValueError):
            NavigationProfile(schema_version="bogus")

    def test_load_shipped_yaml_matches_documented_defaults(self) -> None:
        """出荷 YAML が文書の既定値と一致する。

        宝箱六十秒とカード二秒、debounce と retry の値を数値で比較します。
        """
        profile = load_navigation_profile(_CONFIG_PATH)
        assert profile.debounce_frames == 3
        assert profile.retry_budget == 1
        assert profile.retry_after_ns == 200_000_000
        assert profile.timeout_ns_for("level_up") == 2_000_000_000
        assert profile.timeout_ns_for("chest") == 60_000_000_000
        assert profile.timeout_ns_for("target_reached") == 5_000_000_000
        assert profile.timeout_ns_for("unknown") == 1_000_000_000
        assert profile.timeout_ns_for("run_setup") == 15_000_000_000

    def test_retry_after_ns_rejects_negative(self) -> None:
        """負の retry 待ち時間を拒否する。

        初回送信より前に再送可能になる設定を作れません。
        """
        with pytest.raises(ValueError):
            NavigationProfile(retry_after_ns=-1)

    def test_shipped_yaml_has_no_pixel_or_normalized_coordinates(self) -> None:
        # I3 相当: 設定ファイル自体が固定座標を持たないことを保証する。
        """出荷 profile にクリック座標を埋め込まない。

        画素位置は perception の矩形に任せ、navigation 設定の責務を保ちます。
        """
        text = _CONFIG_PATH.read_text(encoding="utf-8")
        for forbidden in ("roi", "normalized_x", "normalized_y", "pixel", "coordinate"):
            assert forbidden not in text.lower()

    def test_from_wire_rejects_unknown_field(self) -> None:
        """profile の未知 key を拒否する。

        誤った設定名が黙って無視されることを防ぎます。
        """
        with pytest.raises(ValueError):
            NavigationProfile.from_wire({"schema_version": NavigationProfile.default_v1().schema_version, "bogus": 1})


class TestBuildUiActionTelemetry:
    """送信記録と適用確認の情報を区別する。

    操作の identity は残し、生のクリック矩形は telemetry に書きません。
    """

    def test_telemetry_records_identity_without_raw_roi(self) -> None:
        """telemetry は identity を記録し生 ROI を含まない。

        どの観測から何を送ったかを追えて、内部の target データは露出しません。
        """
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
