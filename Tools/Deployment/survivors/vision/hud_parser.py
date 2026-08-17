"""HUD parser コアと HudStateV1 データ契約。

CapturedFrame から HUD 値(タイマー・HP/XP・レベル・インベントリ・UI 状態)を
抽出し、versioned HudStateV1 として返します。
HudStateV1 は 04-09 assembler の入力であり、05-03 が直接参照することはありません。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Final

import numpy as np
from numpy.typing import NDArray

from reinbalance_survivors_contracts.canonical_json import canonical_hash

from .roi_layout import (
    HP_BAR_ROI,
    XP_BAR_ROI,
    TIMER_ROI,
    LEVEL_ROI,
    INV_SLOT_ROIS,
    SCREEN_CENTER_ROI,
    CARD_ROIS,
    norm_to_pixels,
    layout_validity_score,
)
from .digit_parser import (
    TimerResult,
    LevelResult,
    parse_timer,
    parse_level,
    apply_temporal_timer,
    apply_temporal_level,
)
from .bar_parser import BarResult, parse_hp_bar, parse_xp_bar
from .icon_matcher import AtlasManifest, IconMatcher, MatchResult

# HudStateV1 のスキーマバージョン
HUD_STATE_SCHEMA_VERSION: Final[str] = "hud_state.v1"

# 有効な画面状態
SCREEN_STATES: Final[frozenset[str]] = frozenset({
    "gameplay",
    "level_up_items",
    "level_up_fallback",
    "chest",
    "paused",
    "target_reached_transition",
    "death",
    "result",
    "unknown",
})

# インベントリスロット数 (6 weapon + 6 passive)
INV_SLOT_COUNT: Final[int] = 12

# 状態判定の最低信頼度
_STATE_LOW_CONF: Final[float] = 0.35

# 画面中央領域の支配色 HSV 範囲 (UI オーバーレイ検出用)
# レベルアップ画面: 暗い背景に明るいカード
_LEVELUP_OVERLAY_BRIGHTNESS_THRESHOLD: Final[float] = 0.25


@dataclass(frozen=True, slots=True)
class ParsedCard:
    """レベルアップカードスロットの解析結果。"""

    slot_index: int
    item_id: str | None    # 語彙アイテム ID; unknown なら None
    kind: str              # "weapon", "passive", "evolved", "fallback", "unknown"
    level: int | None      # アイテムレベル; unknown なら None
    confidence: float      # 0.0..1.0
    reason: str
    roi_xyxy: tuple[int, int, int, int] | None  # ピクセル ROI (x0,y0,x1,y1)


@dataclass(frozen=True, slots=True)
class ParsedButton:
    """UI ボタンの解析結果 (reroll/skip/banish/ack_chest/confirm)。"""

    button_type: str       # "reroll", "skip", "banish", "ack_chest", "confirm"
    confidence: float
    reason: str
    roi_xyxy: tuple[int, int, int, int] | None


@dataclass(frozen=True)
class HudStateV1:
    """HUD 解析結果の versioned データ契約。

    04-09 assembler が PerceptionSnapshot を構築するための中間表現です。
    フィールドが揃っていない・スキーマ不一致の場合は __post_init__ で拒否します。
    """

    # ── スキーマ・フレーム識別 ──────────────────────────────────
    schema_version: str
    session_id: str
    frame_index: int
    captured_monotonic_ns: int
    parser_artifact_hash: str   # parser 設定+atlas manifest の canonical hash

    # ── 画面状態 ─────────────────────────────────────────────────
    screen_state: str           # SCREEN_STATES のいずれか
    screen_state_confidence: float
    screen_state_reason: str

    # ── タイマー ─────────────────────────────────────────────────
    timer_seconds: float | None
    timer_confidence: float
    timer_reason: str
    post_30_evidence: bool      # 30:00 超遷移を観測した場合 True

    # ── HP バー ──────────────────────────────────────────────────
    hp_ratio: float | None
    hp_confidence: float
    hp_reason: str

    # ── XP バー ──────────────────────────────────────────────────
    xp_ratio: float | None
    xp_confidence: float
    xp_reason: str

    # ── レベル ───────────────────────────────────────────────────
    level: int | None
    level_confidence: float
    level_reason: str

    # ── インベントリ (12 スロット) ────────────────────────────────
    inventory: tuple[str | None, ...]  # len == INV_SLOT_COUNT
    inventory_confidence: float
    inventory_hash: str                # canonical hash of inventory tuple

    # ── レベルアップカード ───────────────────────────────────────
    cards: tuple[ParsedCard, ...]
    candidate_set_hash: str            # hash of (screen_state, sorted card item_ids)

    # ── ボタン ───────────────────────────────────────────────────
    buttons: tuple[ParsedButton, ...]

    # ── capability ───────────────────────────────────────────────
    reroll_available: bool
    skip_available: bool
    banish_available: bool
    capability_confidence: float
    capability_reason: str

    def __post_init__(self) -> None:
        if self.schema_version != HUD_STATE_SCHEMA_VERSION:
            raise ValueError(f"unsupported hud_state schema: {self.schema_version!r}")
        if self.screen_state not in SCREEN_STATES:
            raise ValueError(f"unknown screen_state: {self.screen_state!r}")
        if len(self.inventory) != INV_SLOT_COUNT:
            raise ValueError(f"inventory must have {INV_SLOT_COUNT} slots, got {len(self.inventory)}")
        if not (0.0 <= self.screen_state_confidence <= 1.0):
            raise ValueError("screen_state_confidence out of range")
        if self.timer_seconds is not None and not (0.0 <= self.timer_seconds <= 99 * 60.0):
            raise ValueError(f"timer_seconds out of range: {self.timer_seconds}")
        if self.hp_ratio is not None and not (0.0 <= self.hp_ratio <= 1.0):
            raise ValueError(f"hp_ratio out of range: {self.hp_ratio}")
        if self.xp_ratio is not None and not (0.0 <= self.xp_ratio <= 1.0):
            raise ValueError(f"xp_ratio out of range: {self.xp_ratio}")
        if self.level is not None and not (1 <= self.level <= 99):
            raise ValueError(f"level out of range: {self.level}")
        for slot in self.inventory:
            if slot is not None and not isinstance(slot, str):
                raise ValueError("inventory slots must be str or None")
        for card in self.cards:
            if card.slot_index < 0:
                raise ValueError(f"card slot_index negative: {card.slot_index}")
            if card.kind not in {"weapon", "passive", "evolved", "fallback", "unknown"}:
                raise ValueError(f"unknown card kind: {card.kind!r}")
        for btn in self.buttons:
            if btn.button_type not in {"reroll", "skip", "banish", "ack_chest", "confirm"}:
                raise ValueError(f"unknown button_type: {btn.button_type!r}")

    def to_wire(self) -> dict:
        """JSON シリアライズ可能な dict に変換する (golden fixture 保存用)。"""
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "frame_index": self.frame_index,
            "captured_monotonic_ns": self.captured_monotonic_ns,
            "parser_artifact_hash": self.parser_artifact_hash,
            "screen_state": self.screen_state,
            "screen_state_confidence": self.screen_state_confidence,
            "screen_state_reason": self.screen_state_reason,
            "timer_seconds": self.timer_seconds,
            "timer_confidence": self.timer_confidence,
            "timer_reason": self.timer_reason,
            "post_30_evidence": self.post_30_evidence,
            "hp_ratio": self.hp_ratio,
            "hp_confidence": self.hp_confidence,
            "hp_reason": self.hp_reason,
            "xp_ratio": self.xp_ratio,
            "xp_confidence": self.xp_confidence,
            "xp_reason": self.xp_reason,
            "level": self.level,
            "level_confidence": self.level_confidence,
            "level_reason": self.level_reason,
            "inventory": list(self.inventory),
            "inventory_confidence": self.inventory_confidence,
            "inventory_hash": self.inventory_hash,
            "cards": [
                {
                    "slot_index": c.slot_index,
                    "item_id": c.item_id,
                    "kind": c.kind,
                    "level": c.level,
                    "confidence": c.confidence,
                    "reason": c.reason,
                    "roi_xyxy": list(c.roi_xyxy) if c.roi_xyxy is not None else None,
                }
                for c in self.cards
            ],
            "candidate_set_hash": self.candidate_set_hash,
            "buttons": [
                {
                    "button_type": b.button_type,
                    "confidence": b.confidence,
                    "reason": b.reason,
                    "roi_xyxy": list(b.roi_xyxy) if b.roi_xyxy is not None else None,
                }
                for b in self.buttons
            ],
            "reroll_available": self.reroll_available,
            "skip_available": self.skip_available,
            "banish_available": self.banish_available,
            "capability_confidence": self.capability_confidence,
            "capability_reason": self.capability_reason,
        }

    @classmethod
    def from_wire(cls, wire: dict) -> "HudStateV1":
        """JSON dict から HudStateV1 を復元する (golden fixture 検証用)。"""
        expected_keys = {
            "schema_version", "session_id", "frame_index", "captured_monotonic_ns",
            "parser_artifact_hash", "screen_state", "screen_state_confidence",
            "screen_state_reason", "timer_seconds", "timer_confidence", "timer_reason",
            "post_30_evidence", "hp_ratio", "hp_confidence", "hp_reason",
            "xp_ratio", "xp_confidence", "xp_reason", "level", "level_confidence",
            "level_reason", "inventory", "inventory_confidence", "inventory_hash",
            "cards", "candidate_set_hash", "buttons", "reroll_available",
            "skip_available", "banish_available", "capability_confidence", "capability_reason",
        }
        if set(wire) != expected_keys:
            raise ValueError(
                f"HudStateV1 wire fields mismatch. "
                f"Extra: {set(wire)-expected_keys}, Missing: {expected_keys-set(wire)}"
            )
        cards = tuple(
            ParsedCard(
                slot_index=c["slot_index"],
                item_id=c["item_id"],
                kind=c["kind"],
                level=c["level"],
                confidence=c["confidence"],
                reason=c["reason"],
                roi_xyxy=tuple(c["roi_xyxy"]) if c["roi_xyxy"] is not None else None,
            )
            for c in wire["cards"]
        )
        buttons = tuple(
            ParsedButton(
                button_type=b["button_type"],
                confidence=b["confidence"],
                reason=b["reason"],
                roi_xyxy=tuple(b["roi_xyxy"]) if b["roi_xyxy"] is not None else None,
            )
            for b in wire["buttons"]
        )
        return cls(
            schema_version=wire["schema_version"],
            session_id=wire["session_id"],
            frame_index=wire["frame_index"],
            captured_monotonic_ns=wire["captured_monotonic_ns"],
            parser_artifact_hash=wire["parser_artifact_hash"],
            screen_state=wire["screen_state"],
            screen_state_confidence=wire["screen_state_confidence"],
            screen_state_reason=wire["screen_state_reason"],
            timer_seconds=wire["timer_seconds"],
            timer_confidence=wire["timer_confidence"],
            timer_reason=wire["timer_reason"],
            post_30_evidence=wire["post_30_evidence"],
            hp_ratio=wire["hp_ratio"],
            hp_confidence=wire["hp_confidence"],
            hp_reason=wire["hp_reason"],
            xp_ratio=wire["xp_ratio"],
            xp_confidence=wire["xp_confidence"],
            xp_reason=wire["xp_reason"],
            level=wire["level"],
            level_confidence=wire["level_confidence"],
            level_reason=wire["level_reason"],
            inventory=tuple(wire["inventory"]),
            inventory_confidence=wire["inventory_confidence"],
            inventory_hash=wire["inventory_hash"],
            cards=cards,
            candidate_set_hash=wire["candidate_set_hash"],
            buttons=buttons,
            reroll_available=wire["reroll_available"],
            skip_available=wire["skip_available"],
            banish_available=wire["banish_available"],
            capability_confidence=wire["capability_confidence"],
            capability_reason=wire["capability_reason"],
        )


def _compute_inventory_hash(inventory: tuple[str | None, ...]) -> str:
    """インベントリ tuple の canonical hash を計算する。"""
    return canonical_hash({"slots": list(inventory)})


def _compute_candidate_set_hash(screen_state: str, cards: tuple[ParsedCard, ...]) -> str:
    """画面状態とカード ID セットの canonical hash を計算する。"""
    card_ids = sorted(c.item_id or "unknown" for c in cards)
    return canonical_hash({"screen_state": screen_state, "card_ids": card_ids})


def _detect_screen_state(
    frame_bgra: NDArray[np.uint8],
    *,
    width: int = 1920,
    height: int = 1080,
) -> tuple[str, float, str]:
    """画面全体の特徴から UI 状態を分類して返す。

    Returns: (state, confidence, reason)
    """
    if frame_bgra.size == 0:
        return ("unknown", 0.0, "empty_frame")

    center_roi = norm_to_pixels(SCREEN_CENTER_ROI, width, height)
    center = center_roi.crop(frame_bgra)

    if center.size == 0:
        return ("unknown", 0.0, "empty_center_roi")

    # グレースケール輝度
    bgr = center[..., :3].astype(np.float32)
    brightness = float(np.mean(bgr)) / 255.0

    # HP/XP バーの存在チェックでゲームプレイ中かを判定
    layout_score = layout_validity_score(frame_bgra, width=width, height=height)

    # 画面全体の平均輝度
    full_brightness = float(np.mean(frame_bgra[..., :3])) / 255.0

    if layout_score > 0.3:
        # 3枚・4枚の各レイアウトで全スロットに前景があれば level_up_items
        for cnt in (3, 4):
            norms = CARD_ROIS[cnt]
            if all(
                (crop := norm_to_pixels(norm, width, height).crop(frame_bgra)).size > 0
                and float(np.mean(crop[..., :3] >= 32)) > 0.20
                for norm in norms
            ):
                return ("level_up_items", 0.55, f"hud_card_layout_{cnt}")
        return ("gameplay", 0.60, f"hud_present:{layout_score:.2f}")

    # HUD なし
    if full_brightness > 0.70:
        return ("result", 0.50, "bright_full_screen")
    # ponytail: 死亡固有 ROI なし; 暗さだけでは death 確定不可なので unknown を返す
    if brightness < 0.15 and full_brightness < 0.20:
        return ("unknown", 0.35, "dark_no_hud")
    return ("unknown", 0.30, f"layout_score_low:{layout_score:.2f}")


class HudParser:
    """HUD parser コア。CapturedFrame から HudStateV1 を生成する。

    atlas (IconMatcher) はオプションです。None の場合インベントリは全スロット None になります。
    """

    def __init__(
        self,
        *,
        parser_artifact_hash: str,
        icon_matcher: IconMatcher | None = None,
        width: int = 1920,
        height: int = 1080,
    ) -> None:
        if not isinstance(parser_artifact_hash, str) or not parser_artifact_hash:
            raise ValueError("parser_artifact_hash must be a non-empty string")
        self._artifact_hash = parser_artifact_hash
        self._matcher = icon_matcher
        self._width = width
        self._height = height

        # 時間的状態 (単調性チェック用)
        self._prev_timer_seconds: float | None = None
        self._prev_level: int | None = None

    def reset_temporal_state(self) -> None:
        """セッション境界でタイマー・レベルの時間的状態をリセットする。"""
        self._prev_timer_seconds = None
        self._prev_level = None

    def parse(
        self,
        frame_bgra: NDArray[np.uint8],
        *,
        session_id: str,
        frame_index: int,
        captured_monotonic_ns: int,
    ) -> HudStateV1:
        """1 フレームを解析して HudStateV1 を返す。"""
        w, h = self._width, self._height

        # ── 画面状態検出 ───────────────────────────────────────────
        state, state_conf, state_reason = _detect_screen_state(
            frame_bgra, width=w, height=h
        )

        # ── タイマー ───────────────────────────────────────────────
        timer_roi = norm_to_pixels(TIMER_ROI, w, h)
        timer_crop = timer_roi.crop(frame_bgra)
        timer_raw = parse_timer(timer_crop)
        timer_result = apply_temporal_timer(timer_raw, self._prev_timer_seconds)
        if timer_result.seconds is not None:
            self._prev_timer_seconds = timer_result.seconds
        post_30 = (
            timer_result.seconds is not None and timer_result.seconds >= 1800.0
        )

        # ── HP バー ────────────────────────────────────────────────
        hp_roi = norm_to_pixels(HP_BAR_ROI, w, h)
        hp_crop = hp_roi.crop(frame_bgra)
        hp_result = parse_hp_bar(hp_crop)

        # ── XP バー ────────────────────────────────────────────────
        xp_roi = norm_to_pixels(XP_BAR_ROI, w, h)
        xp_crop = xp_roi.crop(frame_bgra)
        xp_result = parse_xp_bar(xp_crop)

        # ── レベル ─────────────────────────────────────────────────
        level_roi = norm_to_pixels(LEVEL_ROI, w, h)
        level_crop = level_roi.crop(frame_bgra)
        level_raw = parse_level(level_crop)
        level_result = apply_temporal_level(level_raw, self._prev_level)
        if level_result.level is not None:
            self._prev_level = level_result.level

        # ── インベントリ ───────────────────────────────────────────
        inventory, inv_conf = self._parse_inventory(frame_bgra, w, h)
        inv_hash = _compute_inventory_hash(inventory)

        # ── カード・ボタン (choice_parser が担当; ここでは空) ────────
        cards: tuple[ParsedCard, ...] = ()
        buttons: tuple[ParsedButton, ...] = ()
        candidate_set_hash = _compute_candidate_set_hash(state, cards)

        return HudStateV1(
            schema_version=HUD_STATE_SCHEMA_VERSION,
            session_id=session_id,
            frame_index=frame_index,
            captured_monotonic_ns=captured_monotonic_ns,
            parser_artifact_hash=self._artifact_hash,
            screen_state=state,
            screen_state_confidence=state_conf,
            screen_state_reason=state_reason,
            timer_seconds=timer_result.seconds,
            timer_confidence=timer_result.confidence,
            timer_reason=timer_result.reason,
            post_30_evidence=post_30,
            hp_ratio=hp_result.ratio,
            hp_confidence=hp_result.confidence,
            hp_reason=hp_result.reason,
            xp_ratio=xp_result.ratio,
            xp_confidence=xp_result.confidence,
            xp_reason=xp_result.reason,
            level=level_result.level,
            level_confidence=level_result.confidence,
            level_reason=level_result.reason,
            inventory=inventory,
            inventory_confidence=inv_conf,
            inventory_hash=inv_hash,
            cards=cards,
            candidate_set_hash=candidate_set_hash,
            buttons=buttons,
            reroll_available=False,
            skip_available=False,
            banish_available=False,
            capability_confidence=0.0,
            capability_reason="not_parsed_by_hud_parser",
        )

    def _parse_inventory(
        self,
        frame_bgra: NDArray[np.uint8],
        w: int,
        h: int,
    ) -> tuple[tuple[str | None, ...], float]:
        """インベントリスロットを解析してアイテム ID タプルと平均信頼度を返す。"""
        slots: list[str | None] = []
        confidences: list[float] = []

        for slot_norm in INV_SLOT_ROIS:
            slot_roi = norm_to_pixels(slot_norm, w, h)
            crop = slot_roi.crop(frame_bgra)

            if self._matcher is None or crop.size == 0:
                slots.append(None)
                confidences.append(0.0)
                continue

            result = self._matcher.match(crop)
            slots.append(result.item_id)
            confidences.append(result.confidence)

        avg_conf = float(np.mean(confidences)) if confidences else 0.0
        return tuple(slots), avg_conf
