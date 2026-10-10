"""スロットレベル追跡器の規則 (a)(p)(r)(d)(e)(f) を検証する。

パネルの選択前の段階値を基準にし、選択結果が一意に決まるときだけ復帰時に更新します。
"""
from __future__ import annotations

import pytest

from survivors.slot_level_tracker import SlotLevelTracker
from survivors.vision.hud_parser import HudStateV1, ParsedCard

EMPTY = "empty_slot"


def _hud(state: str = "gameplay", *, inventory=("whip",), level: int | None = 1, cards=(), skip=False,
          session="s", confidence=.9, levels=None, levels_confidence=.9,
          capability_confidence=.9) -> HudStateV1:
    """追跡器に必要な項目だけを変えた HudStateV1 を作る。

    在庫は先頭から詰め、残りは空スロット確定（empty_slot）にします。None を渡すと読めなかった枠です。
    """
    inv = tuple(inventory) + (EMPTY,) * (12 - len(inventory))
    return HudStateV1(
        "hud_state.v1", session, 1, 1, "a" * 64, state, confidence, "ok",
        20., .9, "ok", False, .75, .9, "ok", .5, .9, "ok", level, .9, "ok",
        inv, .9, "b" * 64, tuple(cards), "c" * 64, (),
        False, skip, False, capability_confidence, "ok",
        inventory_levels=(None,) * 12 if levels is None else tuple(levels) + (None,) * (12 - len(levels)),
        inventory_levels_confidence=0.0 if levels is None else levels_confidence,
    )


def _card(item_id, level, *, index=0, kind="weapon", confidence=.99) -> ParsedCard:
    """レベルアップカード1枚を作る。

    item_id と新レベル以外は固定値です。
    """
    return ParsedCard(index, item_id, kind, level, confidence, "ok", None)


def _level_up(tracker, cards, *, before=1, inventory=("whip",), skip=False, after_inventory=None,
              levels=(1,), levels_confidence=.9, capability_confidence=.9) -> None:
    """gameplay → レベルアップ画面 → gameplay の1回分を追跡器へ流す。

    プレイヤーレベルは before から1上がった状態で戻ります。
    """
    tracker.observe(_hud(inventory=inventory, level=before))
    tracker.observe(_hud("level_up_items", inventory=inventory, level=before + 1, cards=cards, skip=skip,
                         levels=levels, levels_confidence=levels_confidence,
                         capability_confidence=capability_confidence))
    tracker.observe(_hud(inventory=after_inventory or inventory, level=before + 1))


def test_a_new_identity_starts_at_level_one():
    """(a) 在庫に初めて現れた identity は Lv1。

    ランの開始時の初期武器や、新しく取ったアイテムはレベル1から数えます。
    """
    tracker = SlotLevelTracker()
    tracker.observe(_hud())
    assert tracker.level("whip") == 1
    assert tracker.level("garlic") is None


def test_r_single_owned_card_increments_panel_baseline():
    """(r) 全枠を読めて skip 不可なら所持カード一枚の基準値を一つ増やす。

    カードの数字やプレイヤーレベルではなく、パネルの三段階を基準にします。
    """
    tracker = SlotLevelTracker()
    _level_up(tracker, [_card("whip", 8), _card("garlic", 1, index=1)], levels=(3,))
    assert tracker.level("whip") == 4


def test_r_fresh_one_restores_baseline_and_adds_level_one():
    """(r) 基準で空だった枠に一つだけ増えたら新アイテムを Lv1 にする。

    他の所持品は、過去の追跡値ではなくパネルの基準値へ復帰します。
    """
    tracker = SlotLevelTracker()
    _level_up(tracker, [_card("whip", 2), _card("garlic", 1, index=1)], levels=(5,),
              after_inventory=("whip", "garlic"))
    assert tracker.level("whip") == 5
    assert tracker.level("garlic") == 1


def test_a_evolved_identity_starts_at_level_one():
    """(a) 進化武器のカードを選んで現れた identity も Lv1。

    進化先は新しい identity として在庫に現れるので、規則 (a) で数えます。
    """
    tracker = SlotLevelTracker()
    tracker.observe(_hud())
    tracker.observe(_hud(inventory=("bloody_tear",)))
    assert tracker.level("bloody_tear") == 1


def test_r_multiple_owned_cards_make_them_unknown():
    """(r) 所持カードが二枚以上なら候補だけを不明にする。

    カードに無かった所持スロットは変わりません。
    """
    tracker = SlotLevelTracker()
    inventory = ("whip", "garlic", "knife")
    tracker.observe(_hud(inventory=inventory))
    _level_up(tracker, [_card("whip", 2), _card("garlic", 2, index=1)], inventory=inventory, levels=(3, 4, 5))
    assert tracker.level("whip") is None and tracker.level("garlic") is None
    assert tracker.level("knife") == 5


def test_r_skip_available_makes_single_owned_card_unknown():
    """(r) skip できる画面では所持カードを不明にする。

    選択か skip かを画面から区別できないため推測しません。
    """
    tracker = SlotLevelTracker()
    _level_up(tracker, [_card("whip", 2)], skip=True)
    assert tracker.level("whip") is None


def test_p_later_baseline_and_read_cards_survive_unread_frames():
    """(p) 訪問中の最後の基準値と読めたカードを保持する。

    gameplay を挟まない連続 level-up と後半の不読でも、最後の確定基準を使います。
    """
    tracker = SlotLevelTracker()
    tracker.observe(_hud(level=1))
    tracker.observe(_hud("level_up_items", level=3, levels=(3,), cards=[_card("whip", 4)]))
    tracker.observe(_hud("level_up_fallback", level=8, levels=(6,), cards=[_card("whip", 7)]))
    tracker.observe(_hud("level_up_items", level=8, capability_confidence=.1))
    tracker.observe(_hud(level=8))
    assert tracker.level("whip") == 7


def test_r_unreadable_cards_make_owned_unknown():
    """(r) 一枚もカードを読めなければ所持全枠を不明にする。

    読めなかったカードを選んだ可能性を否定できないためです。
    """
    for cards in ([_card(None, 2)], [_card("whip", 2, confidence=.1)], []):
        tracker = SlotLevelTracker()
        _level_up(tracker, cards)
        assert tracker.level("whip") is None, cards


def test_d_chest_makes_owned_and_following_new_identity_unknown():
    """(d) 宝箱画面の後は所持全スロットと、直後に現れた identity を None。

    宝箱の結果（どの武器が何段上がったか）は画面から読めないためです。
    """
    tracker = SlotLevelTracker()
    tracker.observe(_hud(level=5))
    tracker.observe(_hud("chest", level=5))
    tracker.observe(_hud(inventory=("whip", "garlic"), level=5))
    assert tracker.level("whip") is None and tracker.level("garlic") is None


def test_e_unknown_level_stays_unknown_until_new_identity():
    """(e) None になったスロットは、その後のレベルアップでも推測で埋まらない。

    一意な強化カードを選んだ場合は新レベルが分かるので、そこで初めて確定します。
    """
    tracker = SlotLevelTracker()
    _level_up(tracker, [_card("whip", 2)], skip=True)
    tracker.observe(_hud(level=2))
    tracker.observe(_hud(level=2))
    assert tracker.level("whip") is None
    _level_up(tracker, [_card("garlic", 1)], before=2, levels=(None,), after_inventory=("whip", "garlic"))
    assert tracker.level("whip") is None and tracker.level("garlic") == 1


def test_f_session_change_and_reset_clear_levels():
    """(f) session が変わる・reset するとレベルを全て消す。

    前のランの値を次のランへ持ち越しません。
    """
    tracker = SlotLevelTracker()
    _level_up(tracker, [_card("whip", 3)], levels=(2,))
    assert tracker.level("whip") == 3
    tracker.observe(_hud("paused", session="other"))
    assert tracker.level("whip") is None
    tracker.observe(_hud(session="other"))
    assert tracker.level("whip") == 1
    tracker.reset()
    assert tracker.level("whip") is None


def test_low_confidence_screen_changes_nothing():
    """画面状態の信頼度が低いフレームでは何も更新しない。

    誤認識した画面遷移でレベルを変えないためです。
    """
    tracker = SlotLevelTracker()
    tracker.observe(_hud())
    tracker.observe(_hud("chest", confidence=.2))
    tracker.observe(_hud(inventory=("whip", "garlic"), confidence=.2))
    assert tracker.level("whip") == 1 and tracker.level("garlic") is None


def test_r_unread_new_slot_after_level_up_does_not_infer_owned_pick():
    """読めない枠（None）が残ったまま戻ったら、所持カードが1枚でもレベルを推測しない。

    garlic を取ったがアイコンが low_margin で None のままのケースです。新アイテムを取ったのか
    whip を強化したのか決められないので whip は None（不明）になります。
    """
    tracker = SlotLevelTracker()
    unread_after = ("whip", None)
    _level_up(tracker, [_card("whip", 2), _card("garlic", 1, index=1)], after_inventory=unread_after)
    assert tracker.level("whip") is None
    # 後で garlic が読めても、直前に読めていなかった枠から出たので Lv1 とは断定しない
    tracker.observe(_hud(inventory=("whip", "garlic"), level=2))
    assert tracker.level("garlic") is None


def test_r_all_slots_read_still_sets_level():
    """全枠が identity か空確定として読めていれば、従来どおり所持カード1枚でレベルが確定する。

    読めない枠の規則が、読めている場合の判定を壊していないことの確認です。
    """
    tracker = SlotLevelTracker()
    _level_up(tracker, [_card("whip", 2), _card("garlic", 1, index=1)], after_inventory=("whip", EMPTY))
    assert tracker.level("whip") == 2


def test_r_new_identity_from_unread_baseline_is_not_treated_as_pick():
    """直前に読めなかった枠から現れた identity は、今回選んだ新アイテムとは断定しない。

    前から持っていた garlic が読めるようになっただけの可能性があるので、garlic も、
    カードにある所持 whip も None（不明）にします。
    """
    tracker = SlotLevelTracker()
    before = ("whip", None)
    _level_up(tracker, [_card("whip", 2), _card("garlic", 2, index=1)], inventory=before, after_inventory=("whip", "garlic"))
    assert tracker.level("whip") is None and tracker.level("garlic") is None


def test_a_first_gameplay_frame_counts_new_identities_even_with_unread_slots():
    """session 最初の gameplay では、読めない枠があっても見えた identity を Lv1 とする（ラン開始の初期武器）。

    読めなかった枠の identity は、その後に読めても出所が不確かなので不明のままです。
    """
    tracker = SlotLevelTracker()
    tracker.observe(_hud(inventory=("whip", None)))
    assert tracker.level("whip") == 1
    tracker.observe(_hud(inventory=("whip", "garlic")))
    assert tracker.level("garlic") is None


def test_r_two_fresh_identities_fail_closed():
    """(r) 空枠に二つ以上増えた場合は新アイテムを確定しない。

    所持カードの候補も不明にし、候補でない所持品だけを基準値で保ちます。
    """
    tracker = SlotLevelTracker()
    _level_up(tracker, [_card("whip", 5)], inventory=("whip", "axe"), levels=(4, 6),
              after_inventory=("whip", "axe", "garlic", "knife"))
    assert tracker.level("whip") is None
    assert tracker.level("axe") == 6
    assert tracker.level("garlic") is None and tracker.level("knife") is None


@pytest.mark.parametrize("baseline", [None, (3,)])
def test_r_missing_or_low_confidence_baseline_clears_owned(baseline):
    """(p)(r) 基準が無い・低信頼の場合は全所持を不明にする。

    前回の追跡値があっても、今回の選択前の値を読めなければ引き継ぎません。
    """
    tracker = SlotLevelTracker()
    _level_up(tracker, [_card("whip", 4)], levels=baseline, levels_confidence=.49)
    assert tracker.level("whip") is None


def test_r_unread_baseline_slot_blocks_increment():
    """(r) 基準側に一枠でも不読があれば強化を推測しない。

    戻った画面で全枠を読めても、基準の空枠を確認できないと新アイテムを否定できません。
    """
    tracker = SlotLevelTracker()
    _level_up(tracker, [_card("whip", 4)], inventory=("whip", None), levels=(3,),
              after_inventory=("whip", EMPTY))
    assert tracker.level("whip") is None


def test_r_unknown_baseline_level_is_not_incremented():
    """(r) identity だけ読めた基準値には一を足さない。

    枠の段階値が不明なら、全在庫と所持カードが読めてもレベルは不明のままです。
    """
    tracker = SlotLevelTracker()
    _level_up(tracker, [_card("whip", 4)], levels=(None,))
    assert tracker.level("whip") is None


def test_p_low_confidence_skip_is_not_trusted():
    """(p)(r) 低信頼の skip 不可を強化の根拠にしない。

    False に見えても capability の信頼度が足りなければ選択結果は不明です。
    """
    tracker = SlotLevelTracker()
    _level_up(tracker, [_card("whip", 4)], levels=(3,), capability_confidence=.49)
    assert tracker.level("whip") is None


def test_p_gameplay_level_fields_are_ignored():
    """(p) gameplay の段階値を基準にしない。

    パネル画面以外の値を注入しても、初期在庫は規則 (a) の Lv1 のままです。
    """
    tracker = SlotLevelTracker()
    tracker.observe(_hud(levels=(7,)))
    assert tracker.level("whip") == 1


def test_r_fresh_in_base_empty_position_does_not_require_new_identity():
    """(r) fresh は過去の辞書ではなく基準の空枠位置で決める。

    過去に観測した identity が別の空枠へ現れても、今回の新アイテムとして Lv1 にします。
    """
    tracker = SlotLevelTracker()
    tracker.observe(_hud(inventory=("whip", "garlic")))
    _level_up(tracker, [], levels=(4,), after_inventory=("whip", "garlic"))
    assert tracker.level("whip") == 4 and tracker.level("garlic") == 1


def test_r_without_fresh_and_owned_cards_keeps_other_baseline_levels():
    """(r) 読めたカードが新アイテムだけなら、基準の所持値を保つ。

    全カード不読とは区別し、候補にない所持品を不明化しません。
    """
    tracker = SlotLevelTracker()
    _level_up(tracker, [_card("garlic", 1)], levels=(6,))
    assert tracker.level("whip") == 6
