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

from reinbalance_survivors_contracts.canonical_json import canonical_hash

from .roi_layout import (
    BUTTON_ROIS,
    CHEST_ACK_ROI,
    CARD_ROIS,
    norm_to_pixels,
)
from .icon_matcher import AtlasManifest, IconMatcher, MatchResult
from .hud_parser import (
    ParsedCard,
    ParsedButton,
    _compute_candidate_set_hash,
    SCREEN_STATES,
)

# choice parser 固有の低信頼しきい値
_CARD_LOW_CONF: Final[float] = 0.30
_BUTTON_LOW_CONF: Final[float] = 0.30

# fallback アイテム ID (closed taxonomy: target_profile の fallbacks)
_FALLBACK_IDS: Final[frozenset[str]] = frozenset({"gold", "chicken"})

# capability ボタン型
_CAPABILITY_BUTTONS: Final[tuple[str, ...]] = ("reroll", "skip", "banish")


@dataclass(frozen=True, slots=True)
class ChoiceParseResult:
    """choice parser の解析結果。"""

    cards: tuple[ParsedCard, ...]
    buttons: tuple[ParsedButton, ...]
    reroll_available: bool
    skip_available: bool
    banish_available: bool
    capability_confidence: float
    capability_reason: str
    candidate_set_hash: str
    screen_state: str   # "level_up_items", "level_up_fallback", "chest", "unknown"


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

        screen_state が "level_up_items" または "level_up_fallback" のとき
        カードを、"chest" のとき chest ack ボタンを、それ以外は空を返します。
        """
        if screen_state not in SCREEN_STATES:
            raise ValueError(f"unknown screen_state: {screen_state!r}")

        w, h = self._width, self._height

        if screen_state in ("level_up_items", "level_up_fallback"):
            cards, inferred_state = self._parse_cards(frame_bgra, w, h, screen_state)
            buttons = self._parse_capability_buttons(frame_bgra, w, h)
        elif screen_state == "chest":
            cards = ()
            inferred_state = "chest"
            buttons = self._parse_chest_buttons(frame_bgra, w, h)
        else:
            cards = ()
            inferred_state = screen_state
            buttons = ()

        # capability ボタンの存否
        btn_types = {b.button_type for b in buttons}
        reroll = "reroll" in btn_types
        skip = "skip" in btn_types
        banish = "banish" in btn_types

        cap_confidences = [b.confidence for b in buttons if b.button_type in _CAPABILITY_BUTTONS]
        cap_conf = float(np.mean(cap_confidences)) if cap_confidences else 0.0
        cap_reason = "ok" if cap_confidences else "no_capability_buttons"

        csh = _compute_candidate_set_hash(inferred_state, cards)

        return ChoiceParseResult(
            cards=cards,
            buttons=buttons,
            reroll_available=reroll,
            skip_available=skip,
            banish_available=banish,
            capability_confidence=cap_conf,
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
        """レベルアップカードを解析する (3 枚または 4 枚)。

        カード枚数は ROI 内の前景量から推定します。
        """
        best_cards: tuple[ParsedCard, ...] = ()
        best_count = 0
        best_avg_fg = 0.0
        inferred_state = screen_state

        # 3 枚・4 枚の両方を試し、前景ピクセル率が高い方を採用
        for count in (3, 4):
            norms = CARD_ROIS.get(count, ())
            cards_for_count: list[ParsedCard] = []
            total_fg = 0.0

            for i, norm in enumerate(norms):
                roi = norm_to_pixels(norm, w, h)
                crop = roi.crop(frame_bgra)
                fg_ratio = float(np.mean(crop[..., :3] >= 32)) if crop.size > 0 else 0.0
                total_fg += fg_ratio

                card = self._parse_single_card(crop, slot_index=i, roi_xyxy=roi.as_xyxy())
                cards_for_count.append(card)

            avg_fg = total_fg / count if count > 0 else 0.0

            if avg_fg > 0.20 and avg_fg > best_avg_fg:
                best_cards = tuple(cards_for_count)
                best_count = count
                best_avg_fg = avg_fg

        if not best_cards:
            return (), screen_state

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
        """カードクロップを 1 枚解析して ParsedCard を返す。"""
        if self._matcher is None or crop.size == 0:
            return ParsedCard(
                slot_index=slot_index,
                item_id=None,
                kind="unknown",
                level=None,
                confidence=0.0,
                reason="no_matcher_or_empty_crop",
                roi_xyxy=roi_xyxy,
            )

        # カード全体ではなくアイコン部分 (上部 40%) をマッチングに使う
        icon_height = max(1, int(crop.shape[0] * 0.40))
        icon_crop = crop[:icon_height, :]
        match = self._matcher.match(icon_crop)

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
            level=match.level,
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
        """reroll/skip/banish ボタンの存在を前景量で判定する。

        ボタン領域に前景ピクセルが存在すれば available とみなします。
        """
        buttons: list[ParsedButton] = []
        for btn_type, norm in BUTTON_ROIS.items():
            roi = norm_to_pixels(norm, w, h)
            crop = roi.crop(frame_bgra)
            if crop.size == 0:
                continue
            fg_ratio = float(np.mean(crop[..., :3] >= 32))
            if fg_ratio > 0.15:
                buttons.append(ParsedButton(
                    button_type=btn_type,
                    confidence=min(1.0, fg_ratio * 2.0),
                    reason=f"fg_ratio:{fg_ratio:.2f}",
                    roi_xyxy=roi.as_xyxy(),
                ))
        return tuple(buttons)

    def _parse_chest_buttons(
        self,
        frame_bgra: NDArray[np.uint8],
        w: int,
        h: int,
    ) -> tuple[ParsedButton, ...]:
        """chest 確認ボタン (ack_chest) の存在を判定する。"""
        roi = norm_to_pixels(CHEST_ACK_ROI, w, h)
        crop = roi.crop(frame_bgra)
        if crop.size == 0:
            return ()
        fg_ratio = float(np.mean(crop[..., :3] >= 32))
        if fg_ratio > 0.15:
            return (ParsedButton(
                button_type="ack_chest",
                confidence=min(1.0, fg_ratio * 2.0),
                reason=f"fg_ratio:{fg_ratio:.2f}",
                roi_xyxy=roi.as_xyxy(),
            ),)
        return ()
