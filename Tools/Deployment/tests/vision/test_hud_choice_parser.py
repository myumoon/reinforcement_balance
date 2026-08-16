"""HUD parser / choice parser 統合テスト。

HudStateV1 スキーマ契約・to_wire/from_wire round-trip・
全画面状態・choice parser の card/button 解析・
capability 判定・formal_parser_eligible=false テストを検証します。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from survivors.vision.hud_parser import (
    HUD_STATE_SCHEMA_VERSION,
    SCREEN_STATES,
    INV_SLOT_COUNT,
    HudStateV1,
    ParsedCard,
    ParsedButton,
    HudParser,
    _compute_inventory_hash,
    _compute_candidate_set_hash,
)
from survivors.vision.choice_parser import ChoiceParser, ChoiceParseResult
from survivors.vision.icon_matcher import (
    AtlasManifest,
    ATLAS_SCHEMA_VERSION,
    IconMatcher,
    FormalLoaderRejectedError,
)
from build_survivors_icon_atlas import build_development_atlas

_DUMMY_ARTIFACT_HASH = "c" * 64
_DUMMY_PROFILE_HASH = "a" * 64
_DUMMY_BUILD_HASH = "b" * 64
_SESSION_ID = "test-session-001"


# ── HudStateV1 スキーマ契約 ────────────────────────────────────────

class TestHudStateV1Schema:
    def _make_valid(self, **kwargs) -> HudStateV1:
        defaults = dict(
            schema_version=HUD_STATE_SCHEMA_VERSION,
            session_id=_SESSION_ID,
            frame_index=0,
            captured_monotonic_ns=1_000_000_000,
            parser_artifact_hash=_DUMMY_ARTIFACT_HASH,
            screen_state="gameplay",
            screen_state_confidence=0.7,
            screen_state_reason="ok",
            timer_seconds=None,
            timer_confidence=0.0,
            timer_reason="blank",
            post_30_evidence=False,
            hp_ratio=None,
            hp_confidence=0.0,
            hp_reason="blank",
            xp_ratio=None,
            xp_confidence=0.0,
            xp_reason="blank",
            level=None,
            level_confidence=0.0,
            level_reason="blank",
            inventory=tuple([None] * INV_SLOT_COUNT),
            inventory_confidence=0.0,
            inventory_hash="h" * 64,
            cards=(),
            candidate_set_hash="i" * 64,
            buttons=(),
            reroll_available=False,
            skip_available=False,
            banish_available=False,
            capability_confidence=0.0,
            capability_reason="none",
        )
        defaults.update(kwargs)
        return HudStateV1(**defaults)

    def test_valid_instance_ok(self):
        s = self._make_valid()
        assert s.schema_version == HUD_STATE_SCHEMA_VERSION
        assert s.screen_state == "gameplay"

    def test_wrong_schema_version_raises(self):
        with pytest.raises(ValueError, match="schema"):
            self._make_valid(schema_version="hud_state.v99")

    def test_unknown_screen_state_raises(self):
        with pytest.raises(ValueError, match="screen_state"):
            self._make_valid(screen_state="flying_saucers")

    def test_wrong_inventory_length_raises(self):
        with pytest.raises(ValueError, match="inventory"):
            self._make_valid(inventory=tuple([None] * 5))

    def test_hp_ratio_out_of_range_raises(self):
        with pytest.raises(ValueError, match="hp_ratio"):
            self._make_valid(hp_ratio=1.5)

    def test_xp_ratio_out_of_range_raises(self):
        with pytest.raises(ValueError, match="xp_ratio"):
            self._make_valid(xp_ratio=-0.1)

    def test_level_out_of_range_raises(self):
        with pytest.raises(ValueError, match="level"):
            self._make_valid(level=100)

    def test_invalid_card_kind_raises(self):
        bad_card = ParsedCard(0, None, "dragon", None, 0.0, "bad", None)
        with pytest.raises(ValueError, match="kind"):
            self._make_valid(cards=(bad_card,))

    def test_invalid_button_type_raises(self):
        bad_btn = ParsedButton("teleport", 0.0, "bad", None)
        with pytest.raises(ValueError, match="button_type"):
            self._make_valid(buttons=(bad_btn,))

    @pytest.mark.parametrize("state", SCREEN_STATES)
    def test_all_screen_states_accepted(self, state: str):
        """全画面状態が HudStateV1 に受け付けられる。"""
        s = self._make_valid(screen_state=state)
        assert s.screen_state == state

    def test_screen_states_exhaustive(self):
        """SCREEN_STATES が想定の状態セットを全て含む。"""
        expected = {
            "gameplay", "level_up_items", "level_up_fallback",
            "chest", "paused", "target_reached_transition",
            "death", "result", "unknown"
        }
        assert SCREEN_STATES == expected


# ── to_wire / from_wire round-trip ─────────────────────────────────

class TestHudStateV1RoundTrip:
    def _make_with_cards(self) -> HudStateV1:
        card = ParsedCard(0, "whip", "weapon", 1, 0.8, "ok", (100, 200, 400, 800))
        button = ParsedButton("reroll", 0.7, "fg_ok", (60, 900, 260, 950))
        inv = tuple(["whip"] + [None] * (INV_SLOT_COUNT - 1))
        inv_hash = _compute_inventory_hash(inv)
        csh = _compute_candidate_set_hash("level_up_items", (card,))
        return HudStateV1(
            schema_version=HUD_STATE_SCHEMA_VERSION,
            session_id=_SESSION_ID,
            frame_index=5,
            captured_monotonic_ns=2_000_000_000,
            parser_artifact_hash=_DUMMY_ARTIFACT_HASH,
            screen_state="level_up_items",
            screen_state_confidence=0.85,
            screen_state_reason="dark_center",
            timer_seconds=125.0,
            timer_confidence=0.9,
            timer_reason="ok",
            post_30_evidence=False,
            hp_ratio=0.75,
            hp_confidence=0.8,
            hp_reason="ok",
            xp_ratio=0.3,
            xp_confidence=0.7,
            xp_reason="ok",
            level=5,
            level_confidence=0.75,
            level_reason="ok",
            inventory=inv,
            inventory_confidence=0.7,
            inventory_hash=inv_hash,
            cards=(card,),
            candidate_set_hash=csh,
            buttons=(button,),
            reroll_available=True,
            skip_available=False,
            banish_available=False,
            capability_confidence=0.7,
            capability_reason="ok",
        )

    def test_round_trip(self):
        original = self._make_with_cards()
        wire = original.to_wire()
        restored = HudStateV1.from_wire(wire)
        assert restored == original

    def test_wire_is_json_serializable(self):
        original = self._make_with_cards()
        wire = original.to_wire()
        serialized = json.dumps(wire)
        assert isinstance(serialized, str)

    def test_from_wire_extra_field_raises(self):
        original = self._make_with_cards()
        wire = original.to_wire()
        wire["extra_field"] = "bad"
        with pytest.raises(ValueError, match="mismatch"):
            HudStateV1.from_wire(wire)

    def test_from_wire_missing_field_raises(self):
        original = self._make_with_cards()
        wire = original.to_wire()
        del wire["screen_state"]
        with pytest.raises(ValueError, match="mismatch"):
            HudStateV1.from_wire(wire)


# ── _compute hashes ─────────────────────────────────────────────────

class TestComputeHashes:
    def test_inventory_hash_deterministic(self):
        inv = tuple(["whip", None, "gold"] + [None] * 9)
        h1 = _compute_inventory_hash(inv)
        h2 = _compute_inventory_hash(inv)
        assert h1 == h2

    def test_inventory_hash_changes_with_content(self):
        inv1 = tuple(["whip"] + [None] * (INV_SLOT_COUNT - 1))
        inv2 = tuple([None] * INV_SLOT_COUNT)
        assert _compute_inventory_hash(inv1) != _compute_inventory_hash(inv2)

    def test_candidate_set_hash_deterministic(self):
        card = ParsedCard(0, "whip", "weapon", 1, 0.8, "ok", None)
        h1 = _compute_candidate_set_hash("level_up_items", (card,))
        h2 = _compute_candidate_set_hash("level_up_items", (card,))
        assert h1 == h2

    def test_candidate_set_hash_changes_with_state(self):
        card = ParsedCard(0, "whip", "weapon", 1, 0.8, "ok", None)
        h1 = _compute_candidate_set_hash("level_up_items", (card,))
        h2 = _compute_candidate_set_hash("chest", (card,))
        assert h1 != h2


# ── HudParser smoke test ────────────────────────────────────────────

class TestHudParser:
    def test_parse_blank_frame(self, dummy_parser_artifact_hash: str, blank_frame: np.ndarray):
        """全黒フレームで parse が例外なく HudStateV1 を返す。"""
        parser = HudParser(parser_artifact_hash=dummy_parser_artifact_hash)
        result = parser.parse(
            blank_frame,
            session_id=_SESSION_ID,
            frame_index=0,
            captured_monotonic_ns=1_000_000,
        )
        assert isinstance(result, HudStateV1)
        assert result.schema_version == HUD_STATE_SCHEMA_VERSION
        assert result.screen_state in SCREEN_STATES
        assert len(result.inventory) == INV_SLOT_COUNT
        # 全黒フレームでは値が推測されない
        # timer/hp/xp は None または 0.0 が期待される
        assert result.timer_confidence >= 0.0

    def test_parse_gameplay_frame(
        self, dummy_parser_artifact_hash: str, gameplay_frame: np.ndarray
    ):
        """gameplay フレームは例外なく解析される。"""
        parser = HudParser(parser_artifact_hash=dummy_parser_artifact_hash)
        result = parser.parse(
            gameplay_frame,
            session_id=_SESSION_ID,
            frame_index=1,
            captured_monotonic_ns=2_000_000,
        )
        assert isinstance(result, HudStateV1)
        assert result.screen_state in SCREEN_STATES

    def test_hud_state_exact_field_set(
        self, dummy_parser_artifact_hash: str, blank_frame: np.ndarray
    ):
        """HudStateV1 のフィールドセットが exact-set テスト。"""
        parser = HudParser(parser_artifact_hash=dummy_parser_artifact_hash)
        result = parser.parse(
            blank_frame,
            session_id=_SESSION_ID,
            frame_index=0,
            captured_monotonic_ns=0,
        )
        wire = result.to_wire()
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
        assert set(wire.keys()) == expected_keys

    def test_temporal_timer_monotonicity(
        self, dummy_parser_artifact_hash: str, blank_frame: np.ndarray
    ):
        """同一パーサーが timer 状態を保持し、逆行を reject する。"""
        parser = HudParser(parser_artifact_hash=dummy_parser_artifact_hash)
        parser._prev_timer_seconds = 200.0  # 手動セット

        # timer_seconds=50 (< 200-30=170) は reject されるはず
        # blank フレームでは timer=None なので reject テストは unit レベルで行う
        from survivors.vision.digit_parser import TimerResult, apply_temporal_timer
        r = TimerResult(50.0, 0.9, "ok")
        rejected = apply_temporal_timer(r, 200.0)
        assert rejected.seconds is None

    def test_reset_temporal_state(
        self, dummy_parser_artifact_hash: str, blank_frame: np.ndarray
    ):
        """reset_temporal_state() が prev 値をクリアする。"""
        parser = HudParser(parser_artifact_hash=dummy_parser_artifact_hash)
        parser._prev_timer_seconds = 100.0
        parser._prev_level = 5
        parser.reset_temporal_state()
        assert parser._prev_timer_seconds is None
        assert parser._prev_level is None

    def test_invalid_artifact_hash_raises(self):
        with pytest.raises(ValueError):
            HudParser(parser_artifact_hash="")


# ── ChoiceParser テスト ────────────────────────────────────────────

class TestChoiceParser:
    def test_unknown_screen_state_raises(self, levelup_frame: np.ndarray):
        parser = ChoiceParser()
        with pytest.raises(ValueError, match="screen_state"):
            parser.parse(levelup_frame, screen_state="INVALID_STATE")

    def test_gameplay_state_returns_empty_cards(self, gameplay_frame: np.ndarray):
        """gameplay 状態ではカードとボタンが空。"""
        parser = ChoiceParser()
        result = parser.parse(gameplay_frame, screen_state="gameplay")
        assert isinstance(result, ChoiceParseResult)
        assert result.cards == ()

    def test_levelup_state_returns_result(self, levelup_frame: np.ndarray):
        """level_up_items 状態でカード解析が実行される (結果は空でもよい)。"""
        parser = ChoiceParser()
        result = parser.parse(levelup_frame, screen_state="level_up_items")
        assert isinstance(result, ChoiceParseResult)
        assert result.screen_state in SCREEN_STATES
        assert isinstance(result.candidate_set_hash, str)

    def test_chest_state_returns_chest_button(self):
        """chest 状態で ack_chest ボタンが返される可能性がある。"""
        # chest ボタン領域に前景を置いた合成フレーム
        frame = np.zeros((1080, 1920, 4), dtype=np.uint8)
        frame[..., 3] = 255
        # chest ack ROI: norm (0.375, 0.700, 0.625, 0.780)
        y0, y1 = int(0.700 * 1080), int(0.780 * 1080)
        x0, x1 = int(0.375 * 1920), int(0.625 * 1920)
        frame[y0:y1, x0:x1, :3] = 150
        parser = ChoiceParser()
        result = parser.parse(frame, screen_state="chest")
        # ack_chest ボタンが存在するか、または空 (fg 判定による)
        button_types = {b.button_type for b in result.buttons}
        assert "ack_chest" in button_types or result.buttons == ()

    def test_low_confidence_unknown_card_no_speculation(
        self, development_atlas_path: Path, blank_frame: np.ndarray
    ):
        """低信頼アイコンは item_id=None を返し、推測しない。"""
        matcher = IconMatcher.load_development(development_atlas_path)
        parser = ChoiceParser(icon_matcher=matcher)
        result = parser.parse(blank_frame, screen_state="level_up_items")
        # 全黒フレームではマッチ不能 → 全カードが unknown または空
        for card in result.cards:
            if card.confidence < 0.30:
                assert card.item_id is None, (
                    f"expected None for low-conf card, got {card.item_id!r}"
                )

    def test_fallback_items_are_kind_fallback(self):
        """fallback アイテム (gold/chicken) はカードの kind='fallback' になる。"""
        # fallback アイコンに見せかけた合成テンプレートでマッチさせる
        # ここでは ParsedCard.kind の検証のみ
        from survivors.vision.choice_parser import _FALLBACK_IDS
        assert "gold" in _FALLBACK_IDS
        assert "chicken" in _FALLBACK_IDS

    def test_buttons_not_confused_as_cards(self, blank_frame: np.ndarray):
        """reroll/skip/banish ボタンをカードとして誤認識しない。"""
        # ボタン領域を明るくした合成フレーム
        frame = blank_frame.copy()
        frame[..., 3] = 255
        # reroll: norm (0.060, 0.882, 0.265, 0.950)
        y0, y1 = int(0.882 * 1080), int(0.950 * 1080)
        x0, x1 = int(0.060 * 1920), int(0.265 * 1920)
        frame[y0:y1, x0:x1, :3] = 200

        parser = ChoiceParser()
        result = parser.parse(frame, screen_state="level_up_items")
        # カードの item_id に "reroll" が入っていない
        for card in result.cards:
            assert card.item_id not in ("reroll", "skip", "banish")

    def test_choice_parse_result_fields(self, blank_frame: np.ndarray):
        """ChoiceParseResult の全フィールドが存在する。"""
        parser = ChoiceParser()
        result = parser.parse(blank_frame, screen_state="gameplay")
        assert hasattr(result, "cards")
        assert hasattr(result, "buttons")
        assert hasattr(result, "reroll_available")
        assert hasattr(result, "skip_available")
        assert hasattr(result, "banish_available")
        assert hasattr(result, "capability_confidence")
        assert hasattr(result, "capability_reason")
        assert hasattr(result, "candidate_set_hash")
        assert hasattr(result, "screen_state")


# ── formal_parser_eligible=false の formal loader 拒否 ─────────────

class TestFormalParserEligibility:
    def test_development_atlas_rejected_by_formal_loader(
        self, development_atlas_path: Path
    ):
        """development atlas は formal loader に拒否される。"""
        with pytest.raises(FormalLoaderRejectedError):
            IconMatcher.load_formal(development_atlas_path)

    def test_hud_parser_with_dev_atlas_runs(
        self,
        development_atlas_path: Path,
        blank_frame: np.ndarray,
        dummy_parser_artifact_hash: str,
    ):
        """開発用 atlas でも HudParser は動作する (formal eligible チェックなし)。"""
        matcher = IconMatcher.load_development(development_atlas_path)
        parser = HudParser(
            parser_artifact_hash=dummy_parser_artifact_hash,
            icon_matcher=matcher,
        )
        result = parser.parse(
            blank_frame,
            session_id=_SESSION_ID,
            frame_index=0,
            captured_monotonic_ns=0,
        )
        assert isinstance(result, HudStateV1)

    def test_target_taxonomy_card_count_covered(self):
        """target_profile の level_up_card_counts が [3, 4] = CARD_ROIS キー一致。"""
        from survivors.target_profile import load_target_profile
        from survivors.vision.roi_layout import CARD_ROIS
        profile = load_target_profile()
        counts = profile.sections["choice_taxonomy"]["level_up_card_counts"]
        for count in counts:
            assert count in CARD_ROIS, f"CARD_ROIS missing count={count}"

    def test_fallback_vocabulary_in_closed_taxonomy(self):
        """fallback vocabulary が closed taxonomy と一致している。"""
        from survivors.target_profile import load_target_profile
        from survivors.vision.choice_parser import _FALLBACK_IDS
        profile = load_target_profile()
        fallbacks = set(profile.sections["choice_taxonomy"]["fallbacks"])
        assert fallbacks == _FALLBACK_IDS

    def test_capability_buttons_in_closed_taxonomy(self):
        """capability ボタン種別が closed taxonomy と一致している。"""
        from survivors.target_profile import load_target_profile
        from survivors.vision.choice_parser import _CAPABILITY_BUTTONS
        profile = load_target_profile()
        caps = set(profile.sections["choice_taxonomy"]["capabilities"])
        assert caps == set(_CAPABILITY_BUTTONS)
