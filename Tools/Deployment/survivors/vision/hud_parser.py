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
    CARD_GAP_ROIS,
    EMPTY_SLOT_ID,
    SLOT_LEVEL_VISIBLE_STATES,
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
from .slot_level_parser import SlotLevelResult, parse_slot_levels, has_panel_evidence

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
_PANEL_HOLD_FRAMES = 3
_GAMEPLAY_NONE_RESET_FRAMES = 30
_CARD_CONTRAST_RESET_FRAMES = 3
_SLOT_LEVEL_CONFIDENCE = 0.5

# 画面中央領域の支配色 HSV 範囲 (UI オーバーレイ検出用)
# レベルアップ画面: 暗い背景に明るいカード
_LEVELUP_OVERLAY_BRIGHTNESS_THRESHOLD: Final[float] = 0.25


@dataclass(frozen=True, slots=True)
class ParsedCard:
    """レベルアップカードスロットの解析結果。

    カードの位置、アイテム名、種別、レベルと読み取りの信頼度をまとめます。
    アイテム名やレベルが読めなかった場合は None のまま残します。
    """

    slot_index: int
    item_id: str | None    # 語彙アイテム ID; unknown なら None
    kind: str              # "weapon", "passive", "evolved", "fallback", "unknown"
    level: int | None      # アイテムレベル; unknown なら None
    confidence: float      # 0.0..1.0
    reason: str
    roi_xyxy: tuple[int, int, int, int] | None  # ピクセル ROI (x0,y0,x1,y1)


@dataclass(frozen=True, slots=True)
class ParsedButton:
    """UI ボタンの解析結果 (reroll/skip/banish/ack_chest/confirm)。

    ボタンの種類と画面上の位置に、検出の信頼度と判定理由を添えます。
    """

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

    inventory_levels: tuple[int | None, ...] = (None,) * INV_SLOT_COUNT
    inventory_levels_confidence: float = 0.0

    def __post_init__(self) -> None:
        """HUD のスキーマと各観測値の範囲を検証する。

        段階値は整数か不明だけを受け取り、bool や小数を level に変換しません。
        """
        if self.schema_version != HUD_STATE_SCHEMA_VERSION:
            raise ValueError(f"unsupported hud_state schema: {self.schema_version!r}")
        if self.screen_state not in SCREEN_STATES:
            raise ValueError(f"unknown screen_state: {self.screen_state!r}")
        if len(self.inventory) != INV_SLOT_COUNT:
            raise ValueError(f"inventory must have {INV_SLOT_COUNT} slots, got {len(self.inventory)}")
        if len(self.inventory_levels) != INV_SLOT_COUNT:
            raise ValueError(f"inventory_levels must have {INV_SLOT_COUNT} slots")
        if any(value is not None and (type(value) is not int or not 1 <= value <= 9)
               for value in self.inventory_levels):
            raise ValueError("inventory_levels must contain int 1..9 or None")
        if not (0.0 <= self.inventory_levels_confidence <= 1.0):
            raise ValueError("inventory_levels_confidence out of range")
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
        """JSON シリアライズ可能な dict に変換する (golden fixture 保存用)。

        在庫・カード・ボタンを JSON に保存できるリストへ変え、HUD の全フィールドを書き出します。
        読めなかった値は None のまま保存します。
        """
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
            "inventory_levels": list(self.inventory_levels),
            "inventory_levels_confidence": self.inventory_levels_confidence,
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
        """JSON dict から HudStateV1 を復元する (golden fixture 検証用)。

        キーの不足や余分なキーを拒否してから、在庫のタプルとカード・ボタンの解析結果を復元します。
        復元した値は HudStateV1 の値域検証も通します。
        """
        expected_keys = {
            "schema_version", "session_id", "frame_index", "captured_monotonic_ns",
            "parser_artifact_hash", "screen_state", "screen_state_confidence",
            "screen_state_reason", "timer_seconds", "timer_confidence", "timer_reason",
            "post_30_evidence", "hp_ratio", "hp_confidence", "hp_reason",
            "xp_ratio", "xp_confidence", "xp_reason", "level", "level_confidence",
            "level_reason", "inventory", "inventory_confidence", "inventory_hash",
            "cards", "candidate_set_hash", "buttons", "reroll_available",
            "skip_available", "banish_available", "capability_confidence", "capability_reason",
            "inventory_levels", "inventory_levels_confidence",
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
            inventory_levels=tuple(wire["inventory_levels"]),
            inventory_levels_confidence=wire["inventory_levels_confidence"],
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
    """インベントリ tuple の canonical hash を計算する。

    スロット順のアイテム名だけをハッシュ化します。段階値や読み取りの信頼度は対象に含めません。
    """
    return canonical_hash({"slots": list(inventory)})


def _compute_candidate_set_hash(screen_state: str, cards: tuple[ParsedCard, ...]) -> str:
    """画面状態とカード ID セットの canonical hash を計算する。

    カード名を並べ替えて画面状態と組み合わせるので、同じ候補なら表示順に左右されません。
    読めないカード名は unknown として区別します。
    """
    card_ids = sorted(c.item_id or "unknown" for c in cards)
    return canonical_hash({"screen_state": screen_state, "card_ids": card_ids})


def _mean_roi_brightness(
    frame_bgra: NDArray[np.uint8],
    norms: tuple,
    width: int,
    height: int,
) -> float:
    """指定 ROI 群の平均輝度 (0.0..1.0) を返す。

    各領域の色成分の平均を0〜1へ直してから、領域同士の平均を求めます。
    空の切り出しは除き、対象が一つもなければ0を返します。
    """
    vals: list[float] = []
    for norm in norms:
        crop = norm_to_pixels(norm, width, height).crop(frame_bgra)
        if crop.size > 0:
            vals.append(float(np.mean(crop[..., :3])) / 255.0)
    return sum(vals) / len(vals) if vals else 0.0


def _min_roi_brightness(
    frame_bgra: NDArray[np.uint8],
    norms: tuple,
    width: int,
    height: int,
) -> float:
    """指定 ROI 群の最小スロット輝度 (0.0..1.0) を返す。

    各領域の平均輝度を0〜1で求め、最も暗い領域の値を返します。
    空の切り出しは除き、対象が一つもなければ0を返します。
    """
    vals: list[float] = []
    for norm in norms:
        crop = norm_to_pixels(norm, width, height).crop(frame_bgra)
        if crop.size > 0:
            vals.append(float(np.mean(crop[..., :3])) / 255.0)
    return min(vals) if vals else 0.0


def _detect_screen_state(
    frame_bgra: NDArray[np.uint8],
    *,
    width: int = 1920,
    height: int = 1080,
    panel_evidence: bool = False,
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
        if panel_evidence:
            return ("level_up_items", 0.60, "slot_panel")
        # カード固有の構造証拠:
        #   1) 全スロットが一定輝度以上 (min > 0.08) → 未描画スロットを除外
        #   2) カード平均輝度がギャップより有意に高い (contrast > 0.10) → 均一背景・帯を除外
        for cnt in (3, 4):
            min_card = _min_roi_brightness(frame_bgra, CARD_ROIS[cnt], width, height)
            mean_card = _mean_roi_brightness(frame_bgra, CARD_ROIS[cnt], width, height)
            mean_gap = _mean_roi_brightness(frame_bgra, CARD_GAP_ROIS[cnt], width, height)
            if min_card > 0.08 and mean_card - mean_gap > 0.10:
                return ("level_up_items", 0.55, f"hud_card_contrast_{cnt}")
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
        """解析に使う初期状態を準備する。

        解析器を識別するハッシュ、アイコン照合器、画面サイズを設定します。
        タイマーや在庫などのフレーム間の記録は空の状態から始めます。
        """
        if not isinstance(parser_artifact_hash, str) or not parser_artifact_hash:
            raise ValueError("parser_artifact_hash must be a non-empty string")
        self._artifact_hash = parser_artifact_hash
        self._matcher = icon_matcher
        self._width = width
        self._height = height

        self.reset_temporal_state()

    def reset_temporal_state(self) -> None:
        """セッション境界で全観測の時間的状態をリセットする。

        タイマー・レベル・パネル・訪問前在庫を次のランへ持ち越しません。
        """
        self._prev_timer_seconds = None
        self._prev_level = None
        self._panel_hold = 0
        self._slot_prev: list[tuple[int | None, bool] | None] = [None] * INV_SLOT_COUNT
        self._slot_adopted: list[tuple[int | None, bool] | None] = [None] * INV_SLOT_COUNT
        self._slot_cell_counts = [0] * INV_SLOT_COUNT
        self._slot_prev_cell_counts = [0] * INV_SLOT_COUNT
        self._reset_gameplay_inventory()

    def _reset_gameplay_inventory(self) -> None:
        """保存した gameplay 在庫と採用候補を全消去する。

        宝箱・終端・reset の各経路で、古い一致候補が再採用されることを防ぎます。
        """
        self._gameplay_inventory: list[str | None] = [None] * INV_SLOT_COUNT
        self._gameplay_prev: list[tuple[str | None, ...]] = []
        self._gameplay_none_count = [0] * INV_SLOT_COUNT
        self._card_contrast_count = 0

    def _observe_gameplay_inventory(self, inventory: tuple[str | None, ...]) -> None:
        """gameplay 在庫を枠ごとに三枚一致で保存する。

        短い遮蔽では上書きせず、三十枚連続の不読だけで保存値を消します。
        """
        for slot, identity in enumerate(inventory):
            if identity is None:
                self._gameplay_none_count[slot] += 1
                if self._gameplay_none_count[slot] >= _GAMEPLAY_NONE_RESET_FRAMES:
                    self._gameplay_inventory[slot] = None
            else:
                self._gameplay_none_count[slot] = 0
                if len(self._gameplay_prev) == 2 and all(prev[slot] == identity for prev in self._gameplay_prev):
                    self._gameplay_inventory[slot] = identity
        self._gameplay_prev = (self._gameplay_prev + [inventory])[-2:]

    def _join_slot_levels(self, grid: SlotLevelResult, evidence: bool):
        """採用した段階値と訪問前の在庫を位置で結合する。

        hold は採用に使いません。空枠や進化種別が食い違う場合は両方を不明にします。
        """
        if evidence:
            for slot, level in enumerate(grid.levels):
                candidate = (level, slot in grid.empty_slots)
                if (candidate == self._slot_prev[slot] and (level is not None or candidate[1])
                        and grid.cell_counts[slot] == self._slot_prev_cell_counts[slot]):
                    self._slot_adopted[slot] = candidate
                    self._slot_cell_counts[slot] = grid.cell_counts[slot]
                self._slot_prev[slot] = candidate
                self._slot_prev_cell_counts[slot] = grid.cell_counts[slot]
        else:
            # 証拠のないフレームを挟んだ候補は「二枚連続」とみなさない
            self._slot_prev = [None] * INV_SLOT_COUNT
            self._slot_prev_cell_counts = [0] * INV_SLOT_COUNT
        kinds: dict[str, set[str]] = {}
        if self._matcher is not None:
            for entry in self._matcher.manifest.entries:
                kinds.setdefault(entry.item_id, set()).add(entry.kind)
        inventory: list[str | None] = []
        levels: list[int | None] = []
        reasons: list[str] = []
        for slot, adopted in enumerate(self._slot_adopted):
            identity = self._gameplay_inventory[slot]
            level = joined = None
            if adopted is not None:
                level, empty = adopted
                if empty:
                    if identity in (None, EMPTY_SLOT_ID):
                        joined = EMPTY_SLOT_ID
                    else:
                        reasons.append(f"slot{slot}:empty_mismatch")
                elif identity == EMPTY_SLOT_ID:
                    level = None
                    reasons.append(f"slot{slot}:identity_mismatch")
                elif identity is not None:
                    expected_kind = "evolved" if self._slot_cell_counts[slot] == 1 else (
                        "weapon" if slot < INV_SLOT_COUNT // 2 else "passive")
                    if kinds.get(identity) == {expected_kind}:
                        joined = identity
                    else:
                        level = None
                        reasons.append(f"slot{slot}:kind_mismatch")
            inventory.append(joined)
            levels.append(level)
        level_conf = sum(value is not None for value in self._slot_adopted) / INV_SLOT_COUNT
        inv_conf = sum(value is not None for value in inventory) / INV_SLOT_COUNT
        return tuple(inventory), inv_conf, tuple(levels), level_conf, reasons

    def parse(
        self,
        frame_bgra: NDArray[np.uint8],
        *,
        session_id: str,
        frame_index: int,
        captured_monotonic_ns: int,
    ) -> HudStateV1:
        """1 フレームを解析して HudStateV1 を返す。

        段階格子から画面状態を判定し、バー・数字・在庫・カードを読み取ります。
        段階値は連続フレームでの安定化と在庫の位置照合を通して採用します。
        """
        w, h = self._width, self._height

        # ── 画面状態検出 ───────────────────────────────────────────
        grid = parse_slot_levels(frame_bgra, w, h)
        evidence = has_panel_evidence(grid)
        if evidence:
            self._panel_hold = _PANEL_HOLD_FRAMES
        panel_active = evidence or self._panel_hold > 0
        state, state_conf, state_reason = _detect_screen_state(
            frame_bgra, width=w, height=h, panel_evidence=panel_active
        )
        if not evidence:
            if state_reason == "slot_panel":
                state_reason = "slot_panel_hold"
            self._panel_hold = max(0, self._panel_hold - 1)
            self._slot_prev = [None] * INV_SLOT_COUNT
            self._slot_prev_cell_counts = [0] * INV_SLOT_COUNT
        if not panel_active:
            self._slot_adopted = [None] * INV_SLOT_COUNT
            self._slot_cell_counts = [0] * INV_SLOT_COUNT

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
        if state == "gameplay":
            self._card_contrast_count = 0
            self._observe_gameplay_inventory(inventory)
        elif state == "level_up_items" and state_reason.startswith("hud_card_contrast"):
            self._card_contrast_count += 1
            if self._card_contrast_count >= _CARD_CONTRAST_RESET_FRAMES:
                self._reset_gameplay_inventory()
        elif state in {"chest", "death", "result", "unknown"}:
            self._reset_gameplay_inventory()
        else:
            self._card_contrast_count = 0
        inventory_levels: tuple[int | None, ...] = (None,) * INV_SLOT_COUNT
        levels_conf = 0.0
        if state in SLOT_LEVEL_VISIBLE_STATES and state_conf >= _SLOT_LEVEL_CONFIDENCE:
            inventory, inv_conf, inventory_levels, levels_conf, join_reasons = self._join_slot_levels(grid, evidence)
            if join_reasons:
                state_reason += ";" + ";".join(join_reasons)
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
            inventory_levels=inventory_levels,
            inventory_levels_confidence=levels_conf,
        )

    def _parse_inventory(
        self,
        frame_bgra: NDArray[np.uint8],
        w: int,
        h: int,
    ) -> tuple[tuple[str | None, ...], float]:
        """インベントリスロットを解析してアイテム ID タプルと平均信頼度を返す。

        十二枠の画像を順にアイコン照合器へ渡し、結果のアイテム名と信頼度を集めます。
        照合器が無い枠や切り出せない枠は、アイテム名を None、信頼度を0にします。
        """
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
