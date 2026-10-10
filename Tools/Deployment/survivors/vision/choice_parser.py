"""レベルアップカード・ボタン・fallback の choice parser。

レベルアップオーバーレイ画面からカード・ボタン・fallback アイテムを解析し、
typed UI action として構造化します。
chest/fallback/button を item card へ誤変換せず、unknown/low-confidence は
unknown item/invalid field として返します。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from . import roi_layout as layout
from .icon_matcher import IconMatcher
from .hud_types import ParsedCard, ParsedButton, _compute_candidate_set_hash
from .screen_layout import (
    _card_count, _detect_card_rows, classify_blue, classify_red,
    horizontal_border_present, pixel_fraction, window_present, yellow_fraction,
)

# choice parser 固有の低信頼しきい値
_CARD_LOW_CONF: Final[float] = 0.30

# fallback アイテム ID (closed taxonomy: target_profile の fallbacks)
_FALLBACK_IDS: Final[frozenset[str]] = frozenset({"gold", "chicken"})

# taxonomy が許す能力名。未計測の skip/banish は解析結果には出さない。
_CAPABILITY_BUTTONS: Final[tuple[str, ...]] = ("reroll", "skip", "banish")

@dataclass(frozen=True, slots=True)
class ChoiceParseResult:
    """choice parser の解析結果。

    カードとボタンに能力の信頼度を添え、fallback の多い画面では状態も精緻化します。
    """

    cards: tuple[ParsedCard, ...]
    buttons: tuple[ParsedButton, ...]
    reroll_available: bool
    skip_available: bool
    banish_available: bool
    capability_confidence: float
    capability_reason: str
    candidate_set_hash: str
    screen_state: str


def _color_button(frame, button_type, roi, inner_roi, classifier, color):
    """ボタン内側の色の割合から一つの観測を作る。

    半分以上が対象色なら矩形を返し、金枠や白文字の形は条件にしません。
    """
    fraction = pixel_fraction(inner_roi.crop(frame), classifier)
    if fraction < .50:
        return ()
    return (ParsedButton(button_type, min(1., fraction), f"{color}:{fraction:.2f}", roi.as_xyxy()),)


class ChoiceParser:
    """レベルアップ選択肢とボタンを解析するパーサー。

    icon_matcher を使ってカードアイコンをアイテム ID に変換します。
    chest/fallback/button は typed UI action として構造化します。
    """

    def __init__(
        self,
        *,
        icon_matcher: IconMatcher | None = None,
        width: int = 1920,
        height: int = 1080,
    ) -> None:
        """照合器と対応画面サイズを保存する。

        時間的な保持は HudParser が担当し、この parser は一枚だけで判断します。
        """
        self._matcher = icon_matcher
        self._width = width
        self._height = height

    def parse(
        self,
        frame_bgra: NDArray[np.uint8],
        *,
        screen_state: str,
    ) -> ChoiceParseResult:
        """フレームとヒントとなる screen_state から choice 情報を解析する。

        カード、宝箱終了、終端確認を分けて読み、paused と gameplay の候補は空にします。
        """
        # schema の状態集合を使い、module 読込時の循環を避ける。
        from .hud_parser import SCREEN_STATES

        if screen_state not in SCREEN_STATES:
            raise ValueError(f"unknown screen_state: {screen_state!r}")

        w, h = self._width, self._height
        if frame_bgra.size == 0 or (w, h) != layout.REFERENCE_SIZE or frame_bgra.shape != (1080, 1920, 4):
            return ChoiceParseResult((), (), False, False, False, 0., "unsupported_resolution_or_empty_frame",
                                     _compute_candidate_set_hash("unknown", ()), "unknown")

        if screen_state in ("level_up_items", "level_up_fallback"):
            cards, inferred_state = self._parse_cards(frame_bgra, w, h, screen_state)
            buttons = self._parse_capability_buttons(frame_bgra, w, h)
        elif screen_state == "chest":
            cards = ()
            inferred_state = "chest"
            buttons = self._parse_chest_buttons(frame_bgra, w, h)
        elif screen_state in {"death", "result"}:
            cards = ()
            inferred_state = screen_state
            buttons = self._parse_confirm_button(frame_bgra, screen_state)
        else:
            cards = ()
            inferred_state = screen_state
            buttons = ()

        # capability ボタンの存否
        btn_types = {b.button_type for b in buttons}
        reroll = "reroll" in btn_types
        cap_reason = "skip_banish_roi_undefined" if screen_state in {"level_up_items", "level_up_fallback"} else "not_applicable"

        csh = _compute_candidate_set_hash(inferred_state, cards)

        return ChoiceParseResult(
            cards=cards,
            buttons=buttons,
            reroll_available=reroll,
            skip_available=False,
            banish_available=False,
            capability_confidence=0.0,
            capability_reason=cap_reason,
            candidate_set_hash=csh,
            screen_state=inferred_state,
        )

    def _parse_cards(
        self,
        frame_bgra: NDArray[np.uint8],
        w: int,
        h: int,
        screen_state: str,
    ) -> tuple[tuple[ParsedCard, ...], str]:
        """枠が揃った一枚から四枚のカードを上から読む。

        右枠が立たない過渡と、黄色い光が残る宝箱では選択肢を作りません。
        """
        count = _card_count(_detect_card_rows(frame_bgra))
        if not all(window_present(frame_bgra)) or count == 0 or yellow_fraction(frame_bgra) >= .05:
            return (), screen_state
        best_cards = tuple(
            self._parse_single_card(layout.card_icon_roi(k, w, h).crop(frame_bgra),
                                    slot_index=k, roi_xyxy=layout.card_roi(k, w, h).as_xyxy())
            for k in range(count)
        )

        # fallback アイテム比率でスクリーン状態を精緻化
        fallback_count = sum(
            1 for c in best_cards
            if c.item_id in _FALLBACK_IDS
        )
        if fallback_count > len(best_cards) // 2:
            inferred_state = "level_up_fallback"
        else:
            inferred_state = "level_up_items"

        return best_cards, inferred_state

    def _parse_single_card(
        self,
        crop: NDArray[np.uint8],
        *,
        slot_index: int,
        roi_xyxy: tuple[int, int, int, int],
    ) -> ParsedCard:
        """カード左端の icon を card surface へ照合する。

        文字のレベルは読まず、テンプレートの level もカードへ採用しません。
        """
        if self._matcher is None or crop.size == 0:
            return ParsedCard(
                slot_index=slot_index,
                item_id=None,
                kind="unknown",
                level=None,
                confidence=0.0,
                reason="no_matcher" if self._matcher is None else "empty_crop",
                roi_xyxy=roi_xyxy,
            )

        match = self._matcher.match(crop, surface="card")

        if match.item_id is None or match.confidence < _CARD_LOW_CONF:
            return ParsedCard(
                slot_index=slot_index,
                item_id=None,
                kind="unknown",
                level=None,
                confidence=match.confidence,
                reason=match.reason,
                roi_xyxy=roi_xyxy,
            )

        # fallback アイテムなら kind=fallback
        kind = "fallback" if match.item_id in _FALLBACK_IDS else match.kind or "unknown"

        return ParsedCard(
            slot_index=slot_index,
            item_id=match.item_id,
            kind=kind,
            level=None,
            confidence=match.confidence,
            reason=match.reason,
            roi_xyxy=roi_xyxy,
        )

    def _parse_capability_buttons(
        self,
        frame_bgra: NDArray[np.uint8],
        w: int,
        h: int,
    ) -> tuple[ParsedButton, ...]:
        """実測済みのリロールボタンだけを読む。

        skip と banish の位置は未観測なので、有無も信頼できるとは扱いません。
        """
        return _color_button(frame_bgra, "reroll", layout.REROLL_BUTTON_ROI, layout.REROLL_BUTTON_INNER_ROI, classify_blue, "blue")

    def _parse_chest_buttons(
        self,
        frame_bgra: NDArray[np.uint8],
        w: int,
        h: int,
    ) -> tuple[ParsedButton, ...]:
        """下枠が縮んだ終了段階だけを ack_chest にする。

        開くボタンを押すと既存契約では終了を別操作にできないため、長いパネルでは返しません。
        """
        if not horizontal_border_present(frame_bgra, (905, 925), layout.WINDOW_BORDER_X):
            return ()
        return _color_button(frame_bgra, "ack_chest", layout.CHEST_ACK_BUTTON_ROI, layout.CHEST_ACK_INNER_ROI, classify_blue, "blue")

    def _parse_confirm_button(self, frame, screen_state):
        """死亡と結果の終了ボタンを confirm として観測する。

        状態に応じて赤と青の内側を読み、controller の終端動作は変更しません。
        """
        if screen_state == "death":
            return _color_button(frame, "confirm", layout.DEATH_CONFIRM_ROI, layout.DEATH_CONFIRM_INNER_ROI, classify_red, "red")
        return _color_button(frame, "confirm", layout.RESULT_CONFIRM_ROI, layout.RESULT_CONFIRM_INNER_ROI, classify_blue, "blue")
