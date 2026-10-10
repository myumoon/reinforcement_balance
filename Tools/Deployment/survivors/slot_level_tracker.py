"""パネルの段階値を基準に武器・パッシブのレベルを追跡する。

level-up パネルは選択前のレベルです。gameplay へ戻るときにその値へ復帰し、
選択結果を一意に決められる場合だけ更新します。不確かな値を推測で埋めません。
"""
from __future__ import annotations

from .hud_identity_vocabulary import load_hud_identity_vocabulary
from .vision.hud_parser import HudStateV1, ParsedCard
from .vision.roi_layout import SLOT_LEVEL_VISIBLE_STATES

_SCREEN_CONFIDENCE = .5  # temporal_state の画面状態閾値と同じ
_CARD_CONFIDENCE = .35   # hud_parser の低信頼閾値と同じ


class SlotLevelTracker:
    """identity ごとのレベルを gameplay への復帰時にまとめて更新する追跡器。

    (a) gameplay で新しい identity が現れたら Lv1（進化武器も Lv1）。直前の同じ枠が None なら None
    （session 最初の gameplay は除く）。
    (p) 画面信頼度と段階値信頼度が .5 以上のパネルを、訪問中の選択前の基準値として保持する。
    後の確定フレームで上書きし、読めたカード（信頼度 .35 以上）と skip（信頼度 .5 以上）も保持する。
    (r) gameplay 復帰時、基準が無ければ全所持 None。基準があれば identity と level が両方読めた値で辞書を作り直す。
    基準の空枠に identity が一つだけ現れたら fresh を Lv1、他は基準のまま。二つ以上なら fresh と所持カードを None。
    fresh が無く、基準・復帰在庫の両方に None が無く、所持カードが一枚で確定した skip が False なら基準値 +1。
    それ以外は所持カードを None。カードを一枚も読めていなければ全所持 None。
    (d) 宝箱画面の後は所持全スロットと、その直後に現れた identity を None。
    (e) None は次に (a) または (p→r) が起きるまで不明のまま。(f) session 変更・reset で全消去。
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
        self._skip_available: bool | None = None
        self._baseline: HudStateV1 | None = None
        self._saw_level_up = False
        self._saw_chest = False
        self._last_inventory: tuple[str | None, ...] | None = None

    def level(self, identity: str) -> int | None:
        """identity の現在のレベルを返す。

        まだ数えていない、または不明になった identity は None です。
        """
        return self._levels.get(identity)

    def observe(self, hud: HudStateV1) -> None:
        """1フレームの HUD を取り込み、gameplay へ戻ったときにレベルを更新する。

        画面状態の信頼度が低いフレームは何も変えません。レベルアップ画面では最後に見えた
        基準値と読めたカードを覚え、宝箱画面では所持全スロットを不明にします。
        """
        if not isinstance(hud, HudStateV1):
            raise TypeError("hud must be HudStateV1")
        if hud.session_id != self._session_id:
            self.reset()
            self._session_id = hud.session_id
        if hud.screen_state_confidence < _SCREEN_CONFIDENCE:
            return
        if hud.screen_state in SLOT_LEVEL_VISIBLE_STATES:
            self._saw_level_up = True
            if hud.inventory_levels_confidence >= _SCREEN_CONFIDENCE:
                self._baseline = hud
            read_cards = tuple(card for card in hud.cards
                               if card.item_id is not None and card.confidence >= _CARD_CONFIDENCE)
            if read_cards:
                self._cards = read_cards
            if hud.capability_confidence >= _SCREEN_CONFIDENCE:
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
        if self._saw_chest:
            self._mark(self._levels, None)
            self._mark([identity for identity in inventory if identity not in (None, empty)], None)
        elif self._saw_level_up:
            self._resolve_level_up(inventory, empty)
        else:
            previous = self._last_inventory
            for position, identity in enumerate(inventory):
                if identity not in (None, empty) and identity not in self._levels:
                    self._levels[identity] = 1 if previous is None or previous[position] is not None else None
        self._cards, self._skip_available, self._baseline = (), None, None
        self._saw_level_up = self._saw_chest = False
        self._last_inventory = inventory

    def _resolve_level_up(self, inventory: tuple[str | None, ...], empty: str) -> None:
        """パネル基準へ復帰し、確定できる選択差分だけを反映する（規則 r）。

        空枠への新規取得と、全枠を読めた強化だけを確定します。
        不明になった identity は辞書にも残し、次の gameplay で Lv1 に戻ることを防ぎます。
        """
        owned = {identity for identity in inventory if identity not in (None, empty)}
        baseline = self._baseline
        if baseline is None:
            self._mark(self._levels, None)
            self._mark(owned, None)
            return
        owned.update(identity for identity in baseline.inventory if identity not in (None, empty))
        self._levels = {identity: level for identity, level in zip(baseline.inventory, baseline.inventory_levels)
                        if identity not in (None, empty) and level is not None}
        for identity in owned:
            self._levels.setdefault(identity, None)
        fresh = [identity for before, identity in zip(baseline.inventory, inventory)
                 if before == empty and identity not in (None, empty)]
        matches = [card.item_id for card in self._cards if card.item_id in owned]
        if len(fresh) == 1:
            self._levels[fresh[0]] = 1
        elif len(fresh) > 1:
            self._mark(fresh + matches, None)
        elif (None not in baseline.inventory and None not in inventory and len(matches) == 1
              and self._skip_available is False and self._levels.get(matches[0]) is not None):
            self._levels[matches[0]] += 1
        else:
            self._mark(matches if self._cards else owned, None)

    def _mark(self, identities, value: int | None) -> None:
        """指定 identity のレベルをまとめて同じ値にする。

        不明化（None）にだけ使います。
        """
        for identity in list(identities):
            self._levels[identity] = value
