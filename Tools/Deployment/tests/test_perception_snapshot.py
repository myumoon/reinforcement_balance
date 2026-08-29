"""UI presentation 契約とハッシュ境界を検証する。

連続する画面座標を操作用情報に閉じ込め、モデル入力へ漏らさないことを確認します。
"""
from dataclasses import FrozenInstanceError, fields, replace
import pytest
from survivors.perception_snapshot import (
    NormalizedRoi,
    PerceptionSnapshot,
    UiButtonTargetV1,
    UiCandidateTargetV1,
    UiPresentationSnapshotV1,
    build_ui_presentation_from_hud,
    is_equivalent_ui_target,
)
from survivors.vision.hud_parser import HudStateV1, ParsedButton, ParsedCard

def _hud(*, cards=(), buttons=(), captured_ns=1_000_000_000) -> HudStateV1:
    """UI テスト用の最小 HUD state を作る。

    parser 本体を動かさず、対象表現とハッシュ規則だけを独立して検証します。
    """
    return HudStateV1(
        schema_version="hud_state.v1", session_id="session", frame_index=7,
        captured_monotonic_ns=captured_ns, parser_artifact_hash="a" * 64,
        screen_state="level_up_items", screen_state_confidence=.9,
        screen_state_reason="fixture", timer_seconds=12., timer_confidence=.9,
        timer_reason="fixture", post_30_evidence=False, hp_ratio=.75,
        hp_confidence=.9, hp_reason="fixture", xp_ratio=.5, xp_confidence=.9,
        xp_reason="fixture", level=4, level_confidence=.9, level_reason="fixture",
        inventory=("whip",) + (None,) * 11, inventory_confidence=.8,
        inventory_hash="b" * 64, cards=tuple(cards), candidate_set_hash="c" * 64,
        buttons=tuple(buttons), reroll_available=True, skip_available=False,
        banish_available=True, capability_confidence=.9, capability_reason="fixture",
    )

def test_contract_dataclasses_are_exact_and_frozen() -> None:
    """公開 dataclass のフィールド順と凍結を固定する。

    後続 runtime が別の side channel を期待しないよう、指定された形だけを許します。
    """
    expected = {
        NormalizedRoi: ("left", "top", "right", "bottom"),
        UiCandidateTargetV1: ("choice_id", "choice_index", "semantic_kind", "roi", "validity", "confidence"),
        UiButtonTargetV1: ("semantic_action", "roi", "validity", "capability", "confidence"),
        UiPresentationSnapshotV1: ("schema_hash", "snapshot_id", "frame_id", "parser_artifact_hash", "screen_state", "candidate_set_hash", "inventory_hash", "source_content_hash", "ui_state_key", "candidates", "buttons"),
        PerceptionSnapshot: ("snapshot_id", "frame_id", "captured_ns", "parser_artifact_hash", "source_content_hash", "ui_state_key", "screen_state", "deploy_obs", "item_context", "choices", "ui_presentation", "diagnostics", "ui_policy_input"),
    }
    for cls, names in expected.items():
        assert tuple(field.name for field in fields(cls)) == names
    roi = NormalizedRoi(0., 0., 1., 1.)
    with pytest.raises(FrozenInstanceError):
        roi.left = .1

def test_source_hash_quantizes_but_ui_key_ignores_continuous_values() -> None:
    """source hash と cross-frame key の責務を分ける。

    微小な検出揺れは量子化し、時刻や ROI が変わっても意味状態の key は維持します。
    """
    first = _hud(cards=(ParsedCard(0, "whip", "weapon", 2, .99991, "ok", (100, 100, 500, 500)),))
    under_quantum = _hud(cards=(ParsedCard(0, "whip", "weapon", 2, .99994, "ok", (100, 100, 500, 500)),), captured_ns=1_100_000_000)
    moved = _hud(cards=(ParsedCard(0, "whip", "weapon", 2, .8, "ok", (110, 100, 510, 500)),), captured_ns=1_200_000_000)
    a = build_ui_presentation_from_hud(first, (1000, 1000), snapshot_id="s", frame_id="f")
    b = build_ui_presentation_from_hud(under_quantum, (1000, 1000), snapshot_id="s", frame_id="f")
    c = build_ui_presentation_from_hud(moved, (1000, 1000), snapshot_id="s2", frame_id="f2")
    assert a.source_content_hash != b.source_content_hash  # timestamp は source identity に含む
    same_time = build_ui_presentation_from_hud(_hud(cards=(ParsedCard(0, "whip", "weapon", 2, .99994, "ok", (100, 100, 500, 500)),)), (1000, 1000), snapshot_id="s", frame_id="f")
    assert a.source_content_hash == same_time.source_content_hash
    assert a.ui_state_key == b.ui_state_key == c.ui_state_key

def test_target_equivalence_requires_every_geometric_and_confidence_gate() -> None:
    """UI target の再利用条件を AND 条件で検証する。

    一つでも安全閾値を外れた対象は、前フレームと同じクリック先とみなしません。
    """
    _ns = dict(old_captured_ns=1, new_captured_ns=2, old_ui_state_key="k", new_ui_state_key="k")
    old = UiCandidateTargetV1("whip", 0, "item_card", NormalizedRoi(.1, .1, .4, .4), True, 1.)
    good = UiCandidateTargetV1("whip", 0, "item_card", NormalizedRoi(.101, .101, .401, .401), True, .99)
    assert is_equivalent_ui_target(old, good, **_ns)
    assert not is_equivalent_ui_target(old, good, old_captured_ns=2, new_captured_ns=1, old_ui_state_key="k", new_ui_state_key="k")
    assert not is_equivalent_ui_target(old, UiCandidateTargetV1("whip", 0, "item_card", good.roi, False, 1.), **_ns)
    assert not is_equivalent_ui_target(old, UiCandidateTargetV1("whip", 0, "item_card", good.roi, True, .989), **_ns)
    assert not is_equivalent_ui_target(old, UiCandidateTargetV1("whip", 0, "item_card", NormalizedRoi(.111, .1, .411, .4), True, 1.), **_ns)
    assert not is_equivalent_ui_target(old, UiCandidateTargetV1("whip", 0, "item_card", NormalizedRoi(.2, .1, .5, .4), True, 1.), **_ns)
    assert not is_equivalent_ui_target(old, UiCandidateTargetV1("axe", 0, "item_card", good.roi, True, 1.), **_ns)
    assert not is_equivalent_ui_target(old, good, old_captured_ns=1, new_captured_ns=2, old_ui_state_key="k1", new_ui_state_key="k2")

def test_fallback_cards_and_buttons_have_unique_semantics() -> None:
    """fallback reward と操作ボタンを重複なく分離する。

    gold/chicken は item card にせず、同じボタンの二重検出は一件へ畳みます。
    """
    cards = (
        ParsedCard(0, "gold_bag", "unknown", None, .9, "ok", (0, 0, 100, 100)),
        ParsedCard(1, "floor_chicken", "fallback", None, .9, "ok", (100, 0, 200, 100)),
        ParsedCard(2, "whip", "weapon", 2, .9, "ok", (200, 0, 300, 100)),
    )
    buttons = (
        ParsedButton("reroll", .7, "old", (0, 900, 100, 1000)),
        ParsedButton("reroll", .9, "best", (5, 900, 105, 1000)),
        ParsedButton("skip", .8, "ok", (100, 900, 200, 1000)),
        ParsedButton("banish", .8, "ok", (150, 900, 250, 1000)),
        ParsedButton("ack_chest", .8, "ok", (200, 900, 300, 1000)),
        ParsedButton("confirm", .8, "ok", (300, 900, 400, 1000)),
    )
    snapshot = build_ui_presentation_from_hud(_hud(cards=cards, buttons=buttons), (1000, 1000))
    assert [target.semantic_kind for target in snapshot.candidates] == ["fallback_reward", "fallback_reward", "item_card"]
    assert [target.semantic_action for target in snapshot.buttons] == ["reroll", "skip", "banish", "ack_chest", "confirm"]
    assert snapshot.buttons[0].confidence == .9


def test_ui_presentation_is_not_mapping_so_cannot_leak_into_tensor_builder() -> None:
    """UiPresentationSnapshotV1 が Mapping でないことを静的に確認する。

    deploy_obs_adapter の型チェックが緩んでも ROI が tensor に混入しない
    ことをクラス階層レベルで保証します。
    """
    from collections.abc import Mapping
    snapshot = build_ui_presentation_from_hud(_hud(), (1000, 1000))
    assert not isinstance(snapshot, Mapping)
    assert not isinstance(snapshot, dict)


def test_source_hash_absorbs_sub_quantum_confidence_at_same_timestamp() -> None:
    """量子化が 1e-4 未満の confidence 揺れを吸収することを明示的に確認する。

    同一タイムスタンプで confidence が 5e-5 しか違わない 2 枚の HUD が
    同じ source_content_hash を持つことで parser ノイズが identity を壊しません。
    """
    hud_a = _hud(cards=(ParsedCard(0, "whip", "weapon", 2, 0.80000, "ok", (100, 100, 500, 500)),))
    hud_b = _hud(cards=(ParsedCard(0, "whip", "weapon", 2, 0.80003, "ok", (100, 100, 500, 500)),))
    a = build_ui_presentation_from_hud(hud_a, (1000, 1000), snapshot_id="s", frame_id="f")
    b = build_ui_presentation_from_hud(hud_b, (1000, 1000), snapshot_id="s", frame_id="f")
    assert a.source_content_hash == b.source_content_hash  # both round to 0.8000
    assert a.ui_state_key == b.ui_state_key


def test_low_screen_confidence_invalidates_ui_targets() -> None:
    """screen_state_confidence が閾値未満のとき候補とボタンの validity が False になる。

    画面認識精度が低い状態で操作対象を valid=True にするとクリックが誤分類画面に飛びます。
    """
    hud = _hud(
        cards=(ParsedCard(0, "whip", "weapon", 2, .99, "ok", (100, 100, 400, 500)),),
        buttons=(ParsedButton("reroll", .9, "ok", (0, 900, 100, 1000)),),
    )
    low_conf_hud = replace(hud, screen_state_confidence=.3)
    snap = build_ui_presentation_from_hud(low_conf_hud, (1000, 1000), snapshot_id="s", frame_id="f")
    assert all(not t.validity for t in snap.candidates)
    assert all(not t.validity for t in snap.buttons)


def test_non_item_screen_state_invalidates_ui_targets() -> None:
    """item 以外の screen_state (gameplay 等) では UI target validity が False になる。

    gameplay/paused/death 画面に残留したカードを操作対象として公開しません。
    """
    hud = _hud(
        cards=(ParsedCard(0, "whip", "weapon", 2, .99, "ok", (100, 100, 400, 500)),),
        buttons=(ParsedButton("reroll", .9, "ok", (0, 900, 100, 1000)),),
    )
    gameplay_hud = replace(hud, screen_state="gameplay")
    snap = build_ui_presentation_from_hud(gameplay_hud, (1000, 1000), snapshot_id="s", frame_id="f")
    assert all(not t.validity for t in snap.candidates)
    assert all(not t.validity for t in snap.buttons)


def test_roi_outside_viewport_is_invalid() -> None:
    """viewport をはみ出した ROI は clip せず invalid として扱う。

    切り取り後の中心でクリックすると隣接カードや別 UI を誤操作するため、
    範囲外矩形はクリック対象として公開しません。
    """
    viewport = (1000, 1000)
    # 左端が viewport の外にはみ出している (-100, 100, 400, 500)
    hud = _hud(cards=(ParsedCard(0, "whip", "weapon", 2, .99, "ok", (-100, 100, 400, 500)),))
    snap = build_ui_presentation_from_hud(hud, viewport, snapshot_id="s", frame_id="f")
    assert not snap.candidates[0].validity
    assert snap.candidates[0].roi == NormalizedRoi(0., 0., 0., 0.)
    # 右端が viewport をはみ出している (100, 100, 1200, 500)
    hud2 = _hud(cards=(ParsedCard(0, "whip", "weapon", 2, .99, "ok", (100, 100, 1200, 500)),))
    snap2 = build_ui_presentation_from_hud(hud2, viewport, snapshot_id="s", frame_id="f")
    assert not snap2.candidates[0].validity
    # 完全に viewport 内に収まっている正常ケース
    hud3 = _hud(cards=(ParsedCard(0, "whip", "weapon", 2, .99, "ok", (100, 100, 400, 500)),))
    snap3 = build_ui_presentation_from_hud(hud3, viewport, snapshot_id="s", frame_id="f")
    assert snap3.candidates[0].validity


def test_duplicate_candidate_choice_id_or_index_raises() -> None:
    """重複する choice_id または choice_index を持つカードは ValueError を上げる。

    同一 identity で異なる ROI が共存すると UI 操作先の解決が曖昧になります。
    """
    dup_id = (
        ParsedCard(0, "whip", "weapon", 2, .99, "ok", (100, 100, 400, 500)),
        ParsedCard(1, "whip", "weapon", 2, .99, "ok", (500, 100, 800, 500)),  # same item_id
    )
    with pytest.raises(ValueError, match="choice_id"):
        build_ui_presentation_from_hud(_hud(cards=dup_id), (1000, 1000), snapshot_id="s", frame_id="f")
    dup_idx = (
        ParsedCard(0, "whip", "weapon", 2, .99, "ok", (100, 100, 400, 500)),
        ParsedCard(0, "knife", "weapon", 2, .99, "ok", (500, 100, 800, 500)),  # same slot_index
    )
    with pytest.raises(ValueError, match="choice_index"):
        build_ui_presentation_from_hud(_hud(cards=dup_idx), (1000, 1000), snapshot_id="s", frame_id="f")
