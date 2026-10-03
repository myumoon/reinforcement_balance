"""スロットレベル追跡器（SlotLevelTracker）の規則 (a)〜(f) を検証する。

HUD にスロットごとのレベルが出ない本家画面で、レベルアップ画面の選択と在庫の変化から
レベルを数え、一意に決められないときは None（不明）にすることを確かめます。
"""
from __future__ import annotations

from survivors.slot_level_tracker import SlotLevelTracker
from survivors.vision.hud_parser import HudStateV1, ParsedCard

EMPTY = "empty_slot"


def _hud(state: str = "gameplay", *, inventory=("whip",), level: int | None = 1, cards=(), skip=False,
         session="s", confidence=.9) -> HudStateV1:
    """追跡器に必要な項目だけを変えた HudStateV1 を作る。

    在庫は先頭から詰め、残りは空スロット確定（empty_slot）にします。None を渡すと読めなかった枠です。
    """
    inv = tuple(inventory) + (EMPTY,) * (12 - len(inventory))
    return HudStateV1(
        "hud_state.v1", session, 1, 1, "a" * 64, state, confidence, "ok",
        20., .9, "ok", False, .75, .9, "ok", .5, .9, "ok", level, .9, "ok",
        inv, .9, "b" * 64, tuple(cards), "c" * 64, (),
        False, skip, False, .9, "ok",
    )


def _card(item_id, level, *, index=0, kind="weapon", confidence=.99) -> ParsedCard:
    """レベルアップカード1枚を作る。

    item_id と新レベル以外は固定値です。
    """
    return ParsedCard(index, item_id, kind, level, confidence, "ok", None)


def _level_up(tracker, cards, *, before=1, inventory=("whip",), skip=False, after_inventory=None) -> None:
    """gameplay → レベルアップ画面 → gameplay の1回分を追跡器へ流す。

    プレイヤーレベルは before から1上がった状態で戻ります。
    """
    tracker.observe(_hud(inventory=inventory, level=before))
    tracker.observe(_hud("level_up_items", inventory=inventory, level=before + 1, cards=cards, skip=skip))
    tracker.observe(_hud(inventory=after_inventory or inventory, level=before + 1))


def test_a_new_identity_starts_at_level_one():
    """(a) 在庫に初めて現れた identity は Lv1。

    ランの開始時の初期武器や、新しく取ったアイテムはレベル1から数えます。
    """
    tracker = SlotLevelTracker()
    tracker.observe(_hud())
    assert tracker.level("whip") == 1
    assert tracker.level("garlic") is None


def test_b_single_owned_card_sets_new_level():
    """(b) 所持 identity のカードがちょうど1枚なら、戻ったときにそのカードの新レベルになる。

    新しい identity が増えていないので、所持品の強化を選んだと一意に決まります。
    """
    tracker = SlotLevelTracker()
    _level_up(tracker, [_card("whip", 2), _card("garlic", 1, index=1)])
    assert tracker.level("whip") == 2


def test_b_new_item_choice_keeps_owned_level_and_adds_level_one():
    """(a)(b) 新アイテムを選んだ場合は所持スロットは変わらず、新アイテムが Lv1。

    在庫に新しい identity が現れたことで、選んだのが新アイテムだと分かります。
    """
    tracker = SlotLevelTracker()
    _level_up(tracker, [_card("whip", 2), _card("garlic", 1, index=1)], after_inventory=("whip", "garlic"))
    assert tracker.level("whip") == 1
    assert tracker.level("garlic") == 1


def test_b_evolved_card_identity_starts_at_level_one():
    """(a) 進化武器のカードを選んで現れた identity も Lv1。

    進化先は新しい identity として在庫に現れるので、規則 (a) で数えます。
    """
    tracker = SlotLevelTracker()
    _level_up(tracker, [_card("bloody_tear", 1, kind="evolved")], after_inventory=("bloody_tear",))
    assert tracker.level("bloody_tear") == 1


def test_b_multiple_owned_cards_make_them_unknown():
    """(b) 所持 identity のカードが複数あると、どれを選んだか決められないのでそれらを None。

    カードに無かった所持スロットは変わりません。
    """
    tracker = SlotLevelTracker()
    inventory = ("whip", "garlic", "knife")
    tracker.observe(_hud(inventory=inventory))
    _level_up(tracker, [_card("whip", 2), _card("garlic", 2, index=1)], inventory=inventory)
    assert tracker.level("whip") is None and tracker.level("garlic") is None
    assert tracker.level("knife") == 1


def test_b_skip_available_makes_single_owned_card_unknown():
    """(b) skip できる画面では、カードを選ばなかった可能性があるので該当スロットを None。

    選択か skip かを画面から区別できないため推測しません。
    """
    tracker = SlotLevelTracker()
    _level_up(tracker, [_card("whip", 2)], skip=True)
    assert tracker.level("whip") is None


def test_b_level_jump_other_than_one_makes_all_unknown():
    """(b) 連続レベルアップ等でプレイヤーレベルが1以外の差で戻ったら所持全スロットを None。

    見えたカードは最後の1回分だけなので、途中の選択を数えられません。
    """
    tracker = SlotLevelTracker()
    tracker.observe(_hud(level=1))
    tracker.observe(_hud("level_up_items", level=3, cards=[_card("whip", 2)]))
    tracker.observe(_hud(level=3))
    assert tracker.level("whip") is None


def test_c_unreadable_cards_make_owned_unknown():
    """(c) item_id が読めない・低信頼のカードがあれば所持全スロットを None。

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
    _level_up(tracker, [_card("garlic", 1)], before=2, after_inventory=("whip", "garlic"))
    assert tracker.level("whip") is None and tracker.level("garlic") == 1


def test_f_session_change_and_reset_clear_levels():
    """(f) session が変わる・reset するとレベルを全て消す。

    前のランの値を次のランへ持ち越しません。
    """
    tracker = SlotLevelTracker()
    _level_up(tracker, [_card("whip", 3)])
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


def test_b_unread_new_slot_after_level_up_does_not_infer_owned_pick():
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


def test_b_all_slots_read_still_sets_level():
    """全枠が identity か空確定として読めていれば、従来どおり所持カード1枚でレベルが確定する。

    読めない枠の規則が、読めている場合の判定を壊していないことの確認です。
    """
    tracker = SlotLevelTracker()
    _level_up(tracker, [_card("whip", 2), _card("garlic", 1, index=1)], after_inventory=("whip", EMPTY))
    assert tracker.level("whip") == 2


def test_b_new_identity_from_previously_unread_slot_is_not_treated_as_pick():
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
