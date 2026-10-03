"""HUD の画面遷移から武器・パッシブのスロットレベルを数える最小の時系列追跡器。

本家の HUD はスロットごとのレベルを表示しないため、レベルアップ画面で選んだカードと
在庫の変化からレベルを数えます。選択を一意に決められないとき・カードを読めなかったとき・
宝箱の結果を読めないときは、そのスロットを None（不明）にし、推測で埋めません。
None になったスロットは、その identity が新しく在庫に現れるまで（実質そのランの間）不明のままです。
在庫の None は「読めなかった枠」で空枠とは区別できないので、空と断定するのは対応表の empty_slot だけです。
"""
from __future__ import annotations

from .hud_identity_vocabulary import load_hud_identity_vocabulary
from .vision.hud_parser import HudStateV1, ParsedCard

_SCREEN_CONFIDENCE = .5  # temporal_state の画面状態閾値と同じ
_CARD_CONFIDENCE = .35   # hud_parser の低信頼閾値と同じ
_LEVEL_UP_STATES = frozenset({"level_up_items", "level_up_fallback"})


class SlotLevelTracker:
    """identity ごとのレベルを gameplay への復帰時にまとめて更新する追跡器。

    規則: (a) 在庫に新しい identity が現れたら Lv1（進化武器も Lv1）。ただし、その枠が直前の gameplay で
    読めていなかった（None）なら、前から居た可能性があるので None（不明）にする（session 最初の gameplay は除く）。
    (b) レベルアップ画面から戻ったとき、プレイヤーレベルがちょうど1上がり、新しい identity が無く、
    在庫に読めない枠（None）が無く、所持 identity のカードがちょうど1枚で skip できなかったなら、
    そのスロットをカードの新レベルにする。読めない枠が残る・新 identity の出所が不確か・所持カードが複数・
    skip 可能なら該当スロットを None、レベル差が1でない（連続レベルアップ等）なら全て None。
    (c) カードを読めなかった（item_id 無し・低信頼・カード無し）なら所持全スロット None。
    (d) 宝箱画面の後は所持全スロットと、その直後に現れた identity を None。
    (e) None は次に (a) が起きるまで不明のまま。(f) session 変更・reset で全消去。
    """

    def __init__(self) -> None:
        """空の追跡状態を作る。

        reset と同じ初期化を行います。
        """
        self._session_id: str | None = None
        self.reset()

    def reset(self) -> None:
        """全スロットのレベルと保留中の画面イベントを消す（規則 f）。

        新しいランや別 session の値を持ち越さないために使います。
        """
        self._levels: dict[str, int | None] = {}
        self._cards: tuple[ParsedCard, ...] = ()
        self._skip_available = False
        self._saw_level_up = False
        self._saw_chest = False
        self._level_before: int | None = None
        self._last_inventory: tuple[str | None, ...] | None = None

    def level(self, identity: str) -> int | None:
        """identity の現在のレベルを返す。

        まだ数えていない、または不明になった identity は None です。
        """
        return self._levels.get(identity)

    def observe(self, hud: HudStateV1) -> None:
        """1フレームの HUD を取り込み、gameplay へ戻ったときにレベルを更新する。

        画面状態の信頼度が低いフレームは何も変えません。レベルアップ画面では最後に見えた
        カードを覚え、宝箱画面では所持全スロットを不明にします。
        """
        if not isinstance(hud, HudStateV1):
            raise TypeError("hud must be HudStateV1")
        if hud.session_id != self._session_id:
            self.reset()
            self._session_id = hud.session_id
        if hud.screen_state_confidence < _SCREEN_CONFIDENCE:
            return
        if hud.screen_state in _LEVEL_UP_STATES:
            self._saw_level_up = True
            self._cards = hud.cards
            self._skip_available = hud.skip_available
            return
        if hud.screen_state == "chest":
            self._saw_chest = True
            self._mark(self._levels, None)
            return
        if hud.screen_state != "gameplay":
            return
        empty = load_hud_identity_vocabulary().empty_slot
        inventory = hud.inventory
        new = {}
        for position, identity in enumerate(inventory):
            if identity is not None and identity != empty and identity not in self._levels:
                new.setdefault(identity, position)
        previous = self._last_inventory
        # 直前の gameplay で読めていた枠（空確定または別 identity）に現れた identity だけが「今現れた」と断定できる
        fresh = {identity for identity, position in new.items() if previous is None or previous[position] is not None}
        unread = any(identity is None for identity in inventory)
        new_unknown = False
        if self._saw_chest:
            self._mark(self._levels, None)
            new_unknown = True
        elif self._saw_level_up:
            new_unknown = self._resolve_level_up(list(new), fresh, unread, hud.level)
        for identity in new:
            self._levels[identity] = 1 if identity in fresh and not new_unknown else None
        self._cards, self._skip_available = (), False
        self._saw_level_up = self._saw_chest = False
        self._level_before = hud.level
        self._last_inventory = inventory

    def _resolve_level_up(self, new: list[str], fresh: set[str], unread: bool, level: int | None) -> bool:
        """レベルアップ画面1回分の選択結果を所持スロットへ反映する（規則 b・c）。

        新しく現れた identity を不明にすべきなら True を返します。読めない枠が残る・新 identity が
        読めなかった枠から出てきたときは、新アイテムを取ったのか所持品を強化したのか決められないので、
        カードにある所持 identity を None にします。
        """
        owned = list(self._levels)
        single = self._level_before is not None and level is not None and level - self._level_before == 1
        if not single or len(new) > 1:
            self._mark(owned, None)
            return True
        if new and new[0] in fresh:
            return False  # 1回の選択で新アイテムを取ったと断定できる: 所持スロットは変わらない
        cards = self._cards
        if not cards or any(card.item_id is None or card.confidence < _CARD_CONFIDENCE for card in cards):
            self._mark(owned, None)
            return False
        matches = [card for card in cards if card.item_id in self._levels]
        if len(matches) == 1 and not self._skip_available and not new and not unread:
            self._levels[matches[0].item_id] = matches[0].level
        else:
            self._mark([card.item_id for card in matches], None)
        return False

    def _mark(self, identities, value: int | None) -> None:
        """指定 identity のレベルをまとめて同じ値にする。

        不明化（None）にだけ使います。
        """
        for identity in list(identities):
            self._levels[identity] = value
