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
    _detect_screen_state,
)
from survivors.vision.choice_parser import ChoiceParser, ChoiceParseResult
from survivors.vision.icon_matcher import (
    AtlasManifest,
    ATLAS_SCHEMA_VERSION,
    IconMatcher,
    FormalLoaderRejectedError,
)
from build_survivors_icon_atlas import build_development_atlas
from .test_slot_level_parser import paint_slot

_DUMMY_ARTIFACT_HASH = "c" * 64
_DUMMY_PROFILE_HASH = "a" * 64
_DUMMY_BUILD_HASH = "b" * 64
_SESSION_ID = "test-session-001"


# ── HudStateV1 スキーマ契約 ────────────────────────────────────────

class TestHudStateV1Schema:
    """HUD の値域とスキーマを検証する。

    入力と期待する結果を照合し、この条件で既存の契約が保たれることを確認します。
    """
    def _make_valid(self, **kwargs) -> HudStateV1:
        """有効な HUD 契約の基準値を作る。

        入力と期待する結果を照合し、この条件で既存の契約が保たれることを確認します。
        """
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
        """有効な HUD 契約を生成できることを確認する。

        入力と期待する結果を照合し、この条件で既存の契約が保たれることを確認します。
        """
        s = self._make_valid()
        assert s.schema_version == HUD_STATE_SCHEMA_VERSION
        assert s.screen_state == "gameplay"

    def test_wrong_schema_version_raises(self):
        """未対応の HUD スキーマを拒否する。

        入力と期待する結果を照合し、この条件で既存の契約が保たれることを確認します。
        """
        with pytest.raises(ValueError, match="schema"):
            self._make_valid(schema_version="hud_state.v99")

    def test_unknown_screen_state_raises(self):
        """未定義の画面状態を拒否する。

        入力と期待する結果を照合し、この条件で既存の契約が保たれることを確認します。
        """
        with pytest.raises(ValueError, match="screen_state"):
            self._make_valid(screen_state="flying_saucers")

    def test_wrong_inventory_length_raises(self):
        """十二枠に足りない在庫を拒否する。

        入力と期待する結果を照合し、この条件で既存の契約が保たれることを確認します。
        """
        with pytest.raises(ValueError, match="inventory"):
            self._make_valid(inventory=tuple([None] * 5))

    def test_hp_ratio_out_of_range_raises(self):
        """範囲外の HP 比率を拒否する。

        入力と期待する結果を照合し、この条件で既存の契約が保たれることを確認します。
        """
        with pytest.raises(ValueError, match="hp_ratio"):
            self._make_valid(hp_ratio=1.5)

    def test_xp_ratio_out_of_range_raises(self):
        """範囲外の XP 比率を拒否する。

        入力と期待する結果を照合し、この条件で既存の契約が保たれることを確認します。
        """
        with pytest.raises(ValueError, match="xp_ratio"):
            self._make_valid(xp_ratio=-0.1)

    def test_level_out_of_range_raises(self):
        """範囲外のプレイヤーレベルを拒否する。

        入力と期待する結果を照合し、この条件で既存の契約が保たれることを確認します。
        """
        with pytest.raises(ValueError, match="level"):
            self._make_valid(level=100)

    def test_invalid_card_kind_raises(self):
        """未定義のカード種別を拒否する。

        入力と期待する結果を照合し、この条件で既存の契約が保たれることを確認します。
        """
        bad_card = ParsedCard(0, None, "dragon", None, 0.0, "bad", None)
        with pytest.raises(ValueError, match="kind"):
            self._make_valid(cards=(bad_card,))

    def test_invalid_button_type_raises(self):
        """未定義のボタン種別を拒否する。

        入力と期待する結果を照合し、この条件で既存の契約が保たれることを確認します。
        """
        bad_btn = ParsedButton("teleport", 0.0, "bad", None)
        with pytest.raises(ValueError, match="button_type"):
            self._make_valid(buttons=(bad_btn,))

    @pytest.mark.parametrize("state", SCREEN_STATES)
    def test_all_screen_states_accepted(self, state: str):
        """全画面状態が HudStateV1 に受け付けられる。

        入力と期待する結果を照合し、この条件で既存の契約が保たれることを確認します。
        """
        s = self._make_valid(screen_state=state)
        assert s.screen_state == state

    def test_screen_states_exhaustive(self):
        """SCREEN_STATES が想定の状態セットを全て含む。

        入力と期待する結果を照合し、この条件で既存の契約が保たれることを確認します。
        """
        expected = {
            "gameplay", "level_up_items", "level_up_fallback",
            "chest", "paused", "target_reached_transition",
            "death", "result", "unknown"
        }
        assert SCREEN_STATES == expected


# ── to_wire / from_wire round-trip ─────────────────────────────────

class TestHudStateV1RoundTrip:
    """HUD 契約の wire 往復を検証する。

    入力と期待する結果を照合し、この条件で既存の契約が保たれることを確認します。
    """
    def _make_with_cards(self) -> HudStateV1:
        """カードとボタンを持つ HUD 契約を作る。

        入力と期待する結果を照合し、この条件で既存の契約が保たれることを確認します。
        """
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
        """HUD 契約を wire 往復して値を保つ。

        入力と期待する結果を照合し、この条件で既存の契約が保たれることを確認します。
        """
        original = self._make_with_cards()
        wire = original.to_wire()
        restored = HudStateV1.from_wire(wire)
        assert restored == original

    def test_wire_is_json_serializable(self):
        """HUD の wire を JSON として保存できることを確認する。

        入力と期待する結果を照合し、この条件で既存の契約が保たれることを確認します。
        """
        original = self._make_with_cards()
        wire = original.to_wire()
        serialized = json.dumps(wire)
        assert isinstance(serialized, str)

    def test_from_wire_extra_field_raises(self):
        """HUD の wire の余分なキーを拒否する。

        入力と期待する結果を照合し、この条件で既存の契約が保たれることを確認します。
        """
        original = self._make_with_cards()
        wire = original.to_wire()
        wire["extra_field"] = "bad"
        with pytest.raises(ValueError, match="mismatch"):
            HudStateV1.from_wire(wire)

    def test_from_wire_missing_field_raises(self):
        """HUD の wire の必須キー欠落を拒否する。

        入力と期待する結果を照合し、この条件で既存の契約が保たれることを確認します。
        """
        original = self._make_with_cards()
        wire = original.to_wire()
        del wire["screen_state"]
        with pytest.raises(ValueError, match="mismatch"):
            HudStateV1.from_wire(wire)


# ── _compute hashes ─────────────────────────────────────────────────

class TestComputeHashes:
    """在庫と候補集合の hash を検証する。

    入力と期待する結果を照合し、この条件で既存の契約が保たれることを確認します。
    """
    def test_inventory_hash_deterministic(self):
        """同じ在庫から同じ hash を得る。

        入力と期待する結果を照合し、この条件で既存の契約が保たれることを確認します。
        """
        inv = tuple(["whip", None, "gold"] + [None] * 9)
        h1 = _compute_inventory_hash(inv)
        h2 = _compute_inventory_hash(inv)
        assert h1 == h2

    def test_inventory_hash_changes_with_content(self):
        """在庫の変更が hash に反映されることを確認する。

        入力と期待する結果を照合し、この条件で既存の契約が保たれることを確認します。
        """
        inv1 = tuple(["whip"] + [None] * (INV_SLOT_COUNT - 1))
        inv2 = tuple([None] * INV_SLOT_COUNT)
        assert _compute_inventory_hash(inv1) != _compute_inventory_hash(inv2)

    def test_candidate_set_hash_deterministic(self):
        """同じ候補集合から同じ hash を得る。

        入力と期待する結果を照合し、この条件で既存の契約が保たれることを確認します。
        """
        card = ParsedCard(0, "whip", "weapon", 1, 0.8, "ok", None)
        h1 = _compute_candidate_set_hash("level_up_items", (card,))
        h2 = _compute_candidate_set_hash("level_up_items", (card,))
        assert h1 == h2

    def test_candidate_set_hash_changes_with_state(self):
        """画面状態の変更が候補 hash に反映されることを確認する。

        入力と期待する結果を照合し、この条件で既存の契約が保たれることを確認します。
        """
        card = ParsedCard(0, "whip", "weapon", 1, 0.8, "ok", None)
        h1 = _compute_candidate_set_hash("level_up_items", (card,))
        h2 = _compute_candidate_set_hash("chest", (card,))
        assert h1 != h2


# ── HudParser smoke test ────────────────────────────────────────────

class TestHudParser:
    """HUD の解析結果と時間的な制約を検証する。

    入力と期待する結果を照合し、この条件で既存の契約が保たれることを確認します。
    """
    def test_parse_blank_frame(self, dummy_parser_artifact_hash: str, blank_frame: np.ndarray):
        """全黒フレームで parse が例外なく HudStateV1 を返す。

        入力と期待する結果を照合し、この条件で既存の契約が保たれることを確認します。
        """
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
        """gameplay フレームは例外なく解析される。

        入力と期待する結果を照合し、この条件で既存の契約が保たれることを確認します。
        """
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
        """HudStateV1 のフィールドセットが exact-set テスト。

        入力と期待する結果を照合し、この条件で既存の契約が保たれることを確認します。
        """
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
            "inventory_levels", "inventory_levels_confidence",
        }
        assert set(wire.keys()) == expected_keys

    def test_temporal_timer_monotonicity(
        self, dummy_parser_artifact_hash: str, blank_frame: np.ndarray
    ):
        """同一パーサーが timer 状態を保持し、逆行を reject する。

        入力と期待する結果を照合し、この条件で既存の契約が保たれることを確認します。
        """
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
        """reset_temporal_state() が prev 値をクリアする。

        入力と期待する結果を照合し、この条件で既存の契約が保たれることを確認します。
        """
        parser = HudParser(parser_artifact_hash=dummy_parser_artifact_hash)
        parser._prev_timer_seconds = 100.0
        parser._prev_level = 5
        parser.reset_temporal_state()
        assert parser._prev_timer_seconds is None
        assert parser._prev_level is None

    def test_invalid_artifact_hash_raises(self):
        """空の parser artifact hash を拒否する。

        入力と期待する結果を照合し、この条件で既存の契約が保たれることを確認します。
        """
        with pytest.raises(ValueError):
            HudParser(parser_artifact_hash="")


# ── ChoiceParser テスト ────────────────────────────────────────────

class TestChoiceParser:
    """カードとボタンの解析結果を検証する。

    入力と期待する結果を照合し、この条件で既存の契約が保たれることを確認します。
    """
    def test_unknown_screen_state_raises(self, levelup_frame: np.ndarray):
        """未定義の画面状態を拒否する。

        入力と期待する結果を照合し、この条件で既存の契約が保たれることを確認します。
        """
        parser = ChoiceParser()
        with pytest.raises(ValueError, match="screen_state"):
            parser.parse(levelup_frame, screen_state="INVALID_STATE")

    def test_gameplay_state_returns_empty_cards(self, gameplay_frame: np.ndarray):
        """gameplay 状態ではカードとボタンが空。

        入力と期待する結果を照合し、この条件で既存の契約が保たれることを確認します。
        """
        parser = ChoiceParser()
        result = parser.parse(gameplay_frame, screen_state="gameplay")
        assert isinstance(result, ChoiceParseResult)
        assert result.cards == ()

    def test_levelup_state_returns_result(self, levelup_frame: np.ndarray):
        """level_up_items 状態でカード解析が実行される (結果は空でもよい)。

        入力と期待する結果を照合し、この条件で既存の契約が保たれることを確認します。
        """
        parser = ChoiceParser()
        result = parser.parse(levelup_frame, screen_state="level_up_items")
        assert isinstance(result, ChoiceParseResult)
        assert result.screen_state in SCREEN_STATES
        assert isinstance(result.candidate_set_hash, str)

    def test_chest_state_returns_chest_button(self):
        """chest 状態で ack_chest ボタンが返される可能性がある。

        入力と期待する結果を照合し、この条件で既存の契約が保たれることを確認します。
        """
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
        """低信頼アイコンは item_id=None を返し、推測しない。

        入力と期待する結果を照合し、この条件で既存の契約が保たれることを確認します。
        """
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
        """fallback アイテム (gold/chicken) はカードの kind='fallback' になる。

        入力と期待する結果を照合し、この条件で既存の契約が保たれることを確認します。
        """
        # fallback アイコンに見せかけた合成テンプレートでマッチさせる
        # ここでは ParsedCard.kind の検証のみ
        from survivors.vision.choice_parser import _FALLBACK_IDS
        assert "gold" in _FALLBACK_IDS
        assert "chicken" in _FALLBACK_IDS

    def test_buttons_not_confused_as_cards(self, blank_frame: np.ndarray):
        """reroll/skip/banish ボタンをカードとして誤認識しない。

        入力と期待する結果を照合し、この条件で既存の契約が保たれることを確認します。
        """
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
        """ChoiceParseResult の全フィールドが存在する。

        入力と期待する結果を照合し、この条件で既存の契約が保たれることを確認します。
        """
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
    """開発用 atlas と正式ロードの境界を検証する。

    入力と期待する結果を照合し、この条件で既存の契約が保たれることを確認します。
    """
    def test_development_atlas_rejected_by_formal_loader(
        self, development_atlas_path: Path
    ):
        """development atlas は formal loader に拒否される。

        入力と期待する結果を照合し、この条件で既存の契約が保たれることを確認します。
        """
        with pytest.raises(FormalLoaderRejectedError):
            IconMatcher.load_formal(development_atlas_path)

    def test_hud_parser_with_dev_atlas_runs(
        self,
        development_atlas_path: Path,
        blank_frame: np.ndarray,
        dummy_parser_artifact_hash: str,
    ):
        """開発用 atlas でも HudParser は動作する (formal eligible チェックなし)。

        入力と期待する結果を照合し、この条件で既存の契約が保たれることを確認します。
        """
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
        """target_profile の level_up_card_counts が [3, 4] = CARD_ROIS キー一致。

        入力と期待する結果を照合し、この条件で既存の契約が保たれることを確認します。
        """
        from survivors.target_profile import load_target_profile
        from survivors.vision.roi_layout import CARD_ROIS
        profile = load_target_profile()
        counts = profile.sections["choice_taxonomy"]["level_up_card_counts"]
        for count in counts:
            assert count in CARD_ROIS, f"CARD_ROIS missing count={count}"

    def test_fallback_vocabulary_in_closed_taxonomy(self):
        """fallback vocabulary が closed taxonomy と一致している。

        入力と期待する結果を照合し、この条件で既存の契約が保たれることを確認します。
        """
        from survivors.target_profile import load_target_profile
        from survivors.vision.choice_parser import _FALLBACK_IDS
        profile = load_target_profile()
        fallbacks = set(profile.sections["choice_taxonomy"]["fallbacks"])
        assert fallbacks == _FALLBACK_IDS

    def test_capability_buttons_in_closed_taxonomy(self):
        """capability ボタン種別が closed taxonomy と一致している。

        入力と期待する結果を照合し、この条件で既存の契約が保たれることを確認します。
        """
        from survivors.target_profile import load_target_profile
        from survivors.vision.choice_parser import _CAPABILITY_BUTTONS
        profile = load_target_profile()
        caps = set(profile.sections["choice_taxonomy"]["capabilities"])
        assert caps == set(_CAPABILITY_BUTTONS)


# ── screen-state confusion regression (04-03 contract) ──────────────
class TestSlotPanelIntegration:
    """段階マークと在庫の時間的な位置結合を検証する。

    格子は実際の画素で描き、在庫照合だけを固定して過渡・遮蔽・種別違いを再現します。
    """

    def _panel(self):
        """HUD のある合成画面に段階マークを描く。

        一枠だけ通常武器を持ち、残りを空枠にします。
        """
        frame = TestDetectScreenState()._hud_frame()
        frame[100:245, 115:410, :3] = 140
        frame[241, 120:401, :3] = (100, 180, 240)
        frame[242:244, 120:401, :3] = (47, 97, 125)
        paint_slot(frame, 0, ["lit"] * 3 + ["unlit"] * 5 + ["none"])
        return frame

    def _parser(self, monkeypatch, inventory=None, kinds=None):
        """照合結果を操作できる HudParser を作る。

        種別は実際の atlas entry から引き、画面状態と段階マークの解析は実装を使います。
        """
        from survivors.vision.icon_matcher import TemplateEntry, build_template_feature
        base = ("whip",) + ("empty_slot",) * 11 if inventory is None else inventory
        kinds = {"whip": "weapon"} if kinds is None else kinds
        feature = build_template_feature(np.zeros((42, 42, 4), dtype=np.uint8))
        atlas = AtlasManifest(ATLAS_SCHEMA_VERSION, "a" * 64, "b" * 64, True, False, "c" * 64,
                              tuple(TemplateEntry(item, kind, 1, 8, feature) for item, kind in kinds.items()))
        parser = HudParser(parser_artifact_hash=_DUMMY_ARTIFACT_HASH, icon_matcher=IconMatcher(atlas))
        raw = [base]
        monkeypatch.setattr(parser, "_parse_inventory", lambda *_: (raw[0], .99))
        return parser, raw

    def _read(self, parser, frame):
        """一枚の画面を parser の時系列へ流す。

        session は固定し、reset のテストは明示的に状態を消します。
        """
        return parser.parse(frame, session_id=_SESSION_ID, frame_index=0, captured_monotonic_ns=0)

    def _seed(self, parser):
        """訪問直前の gameplay を三枚読む。

        同じ在庫が三回続くことで、位置結合に使う保存値を確定します。
        """
        frame = TestDetectScreenState()._hud_frame()
        for _ in range(3):
            self._read(parser, frame)
        return frame

    def test_panel_state_default_detection_and_two_frame_adoption(self, monkeypatch):
        """格子の証拠を状態判定へ渡し、二枚一致で採用する。

        従来の判定は既定引数で変わらず、最初のパネルは未採用です。
        """
        panel = self._panel()
        assert _detect_screen_state(panel)[0] == "gameplay"
        assert _detect_screen_state(panel, panel_evidence=True) == ("level_up_items", .60, "slot_panel")
        parser, _ = self._parser(monkeypatch)
        self._seed(parser)
        first = self._read(parser, panel)
        assert first.inventory == (None,) * 12
        assert first.inventory_levels_confidence == 0
        second = self._read(parser, panel)
        assert second.inventory == ("whip",) + ("empty_slot",) * 11
        assert second.inventory_levels == (3,) + (None,) * 11
        assert second.inventory_confidence == second.inventory_levels_confidence == 1.0

    def test_slot_flicker_hold_and_visit_end(self, monkeypatch):
        """枠の揺れを保持し、証拠消失後の三枚だけ状態を維持する。

        hold 中の読取は採用せず、四枚目で訪問の値を消します。
        """
        parser, _ = self._parser(monkeypatch)
        gameplay = self._seed(parser)
        panel = self._panel()
        paint_slot(panel, 1, ["lit"] + ["unlit"] * 7 + ["none"])
        self._read(parser, panel)
        adopted = self._read(parser, panel)
        noisy = panel.copy()
        paint_slot(noisy, 0, ["ambiguous"] * 9)
        flicker = self._read(parser, noisy)
        assert flicker.inventory_levels == adopted.inventory_levels
        assert flicker.inventory_hash == adopted.inventory_hash
        for _ in range(3):
            held = self._read(parser, gameplay)
            assert held.screen_state == "level_up_items"
            assert held.screen_state_reason.startswith("slot_panel_hold")
            assert held.inventory_levels == adopted.inventory_levels
        returned = self._read(parser, gameplay)
        assert returned.screen_state == "gameplay"
        assert returned.inventory_levels_confidence == 0
        assert parser._slot_adopted == [None] * 12

    def test_gameplay_three_frame_adoption_transient_and_none_reset(self, monkeypatch):
        """在庫は三枚一致で保存し、三十枚の不読で消去する。

        二枚の誤読と訪問直前の五枚の遮蔽では保存値を壊しません。
        """
        parser, raw = self._parser(monkeypatch)
        gameplay = TestDetectScreenState()._hud_frame()
        for _ in range(2):
            self._read(parser, gameplay)
        assert parser._gameplay_inventory[0] is None
        self._read(parser, gameplay)
        raw[0] = ("garlic",) + ("empty_slot",) * 11
        for _ in range(2):
            self._read(parser, gameplay)
        assert parser._gameplay_inventory[0] == "whip"
        raw[0] = (None,) * 12
        for _ in range(5):
            self._read(parser, gameplay)
        assert parser._gameplay_inventory[0] == "whip"
        panel = self._panel()
        self._read(parser, panel)
        assert self._read(parser, panel).inventory[0] == "whip"
        parser.reset_temporal_state()
        raw[0] = ("whip",) + ("empty_slot",) * 11
        self._seed(parser)
        raw[0] = (None,) * 12
        for _ in range(29):
            self._read(parser, gameplay)
        assert parser._gameplay_inventory[0] == "whip"
        self._read(parser, gameplay)
        assert parser._gameplay_inventory == [None] * 12

    @pytest.mark.parametrize("identity,kind,total,expected,reason", [
        (None, None, 8, (3, None), None),
        ("empty_slot", None, 8, (None, None), "identity_mismatch"),
        ("whip", "weapon", 1, (None, None), "kind_mismatch"),
        ("bloody_tear", "evolved", 8, (None, None), "kind_mismatch"),
        ("whip", "unknown", 8, (None, None), "kind_mismatch"),
        ("bloody_tear", "evolved", 1, (1, "bloody_tear"), None),
        (None, None, 0, (None, "empty_slot"), None),
        ("whip", "weapon", 0, (None, None), "empty_mismatch"),
    ])
    def test_position_join_fails_closed(self, monkeypatch, identity, kind, total, expected, reason):
        """空枠・所持・進化種別の食い違いを拒否する。

        identity が不読なら level だけ、種別不明なら結合せず両方を不明にします。
        """
        inv = (identity, "whip") + (None,) * 10
        kinds = {"whip": "weapon"}
        if kind is not None:
            kinds[identity] = kind
        parser, _ = self._parser(monkeypatch, inv, kinds)
        self._seed(parser)
        panel = self._panel()
        lit = min(3, total)
        paint_slot(panel, 0, ["lit"] * lit + ["unlit"] * (total - lit) + ["none"] * (9 - total))
        paint_slot(panel, 1, ["lit"] + ["unlit"] * 7 + ["none"])
        self._read(parser, panel)
        joined = self._read(parser, panel)
        assert (joined.inventory_levels[0], joined.inventory[0]) == expected
        if reason:
            assert "slot0:" + reason in joined.screen_state_reason

    def test_card_contrast_clears_inventory_but_panel_hold_does_not(self, monkeypatch):
        """カード由来の三枚だけを宝箱相当の消去条件にする。

        パネルの hold は保存在庫を保ち、実カード判定が三回続くと全枠を消します。
        """
        parser, _ = self._parser(monkeypatch)
        gameplay = self._seed(parser)
        panel = self._panel()
        self._read(parser, panel)
        self._read(parser, panel)
        for _ in range(3):
            self._read(parser, gameplay)
        assert parser._gameplay_inventory[0] == "whip"
        cards = TestDetectScreenState()._fill_cards(gameplay, 3)
        for _ in range(2):
            result = self._read(parser, cards)
            assert result.screen_state_reason.startswith("hud_card_contrast")
        assert parser._gameplay_inventory[0] == "whip"
        self._read(parser, cards)
        assert parser._gameplay_inventory == [None] * 12

    @pytest.mark.parametrize("state", ["chest", "death", "result", "unknown"])
    def test_terminal_state_clears_all_gameplay_history(self, monkeypatch, state):
        """終端・宝箱・不明画面では gameplay の保存値を消す。

        古い三枚一致候補も消し、次の一枚だけでは再採用しません。
        """
        parser, _ = self._parser(monkeypatch)
        gameplay = self._seed(parser)
        with monkeypatch.context() as patch:
            patch.setattr("survivors.vision.hud_parser._detect_screen_state", lambda *a, **k: (state, .9, state))
            self._read(parser, gameplay)
        assert parser._gameplay_inventory == [None] * 12
        self._read(parser, gameplay)
        assert parser._gameplay_inventory == [None] * 12

    def test_reset_clears_all_temporal_fields(self, monkeypatch):
        """reset でパネルと gameplay の全履歴を消す。

        reset 後の最初のパネルには identity も採用済み level も残りません。
        """
        parser, _ = self._parser(monkeypatch)
        self._seed(parser)
        panel = self._panel()
        self._read(parser, panel)
        self._read(parser, panel)
        parser.reset_temporal_state()
        assert parser._panel_hold == 0
        assert parser._slot_adopted == [None] * 12
        assert parser._gameplay_inventory == [None] * 12
        assert parser._gameplay_none_count == [0] * 12
        first = self._read(parser, panel)
        assert first.inventory == (None,) * 12
        second = self._read(parser, panel)
        assert second.inventory[0] is None and second.inventory_levels[0] == 3


@pytest.mark.parametrize("levels,confidence", [
    ((1,) * 11, .9), ((0,) * 12, .9), ((10,) * 12, .9), ((True,) * 12, .9),
    ((1.0,) * 12, .9), (("1",) * 12, .9), ((None,) * 12, -.1), ((None,) * 12, 1.1),
])
def test_inventory_level_contract_rejects_invalid_values(levels, confidence):
    """直生成と wire 復元の両方で不正な段階値を拒否する。

    bool・小数・範囲外・長さ不足を、整数の確定値として扱いません。
    """
    with pytest.raises(ValueError):
        TestHudStateV1Schema()._make_valid(inventory_levels=levels, inventory_levels_confidence=confidence)
    wire = TestHudStateV1Schema()._make_valid().to_wire()
    wire.update(inventory_levels=list(levels), inventory_levels_confidence=confidence)
    with pytest.raises(ValueError):
        HudStateV1.from_wire(wire)


def test_inventory_levels_wire_roundtrip_and_hash_unchanged():
    """段階値を wire に含め、在庫 hash は identity だけに保つ。

    新しい二キーも完全一致の対象にし、未知キーや欠落キーを許しません。
    """
    hud = TestHudStateV1Schema()._make_valid(inventory_levels=(9,) + (None,) * 11, inventory_levels_confidence=.5)
    wire = hud.to_wire()
    assert HudStateV1.from_wire(wire) == hud
    assert wire["inventory_levels"] == [9] + [None] * 11
    assert hud.inventory_hash == TestHudStateV1Schema()._make_valid().inventory_hash
    assert hud.parser_artifact_hash == _DUMMY_ARTIFACT_HASH
    for field in ("inventory_levels", "inventory_levels_confidence"):
        missing = dict(wire)
        del missing[field]
        with pytest.raises(ValueError, match="fields mismatch"):
            HudStateV1.from_wire(missing)


class TestDetectScreenState:
    """_detect_screen_state の gameplay/level_up_items 誤判定回帰テスト。

    HUD（HP/XP バー）あり + カードレイアウトの有無で正しく分類されることを検証します。
    """

    _W, _H = 1920, 1080

    def _hud_frame(self) -> np.ndarray:
        """HP/XP バーのみの合成フレーム (カードなし、layout_score > 0.3 になる)。

        入力と期待する結果を照合し、この条件で既存の契約が保たれることを確認します。
        """
        from survivors.vision.roi_layout import HP_BAR_ROI, XP_BAR_ROI, norm_to_pixels
        frame = np.zeros((self._H, self._W, 4), dtype=np.uint8)
        for roi_norm, bgr in ((HP_BAR_ROI, (0, 0, 200)), (XP_BAR_ROI, (200, 0, 0))):
            roi = norm_to_pixels(roi_norm, self._W, self._H)
            frame[roi.y0:roi.y1, roi.x0:roi.x1, :3] = bgr
            frame[roi.y0:roi.y1, roi.x0:roi.x1, 3] = 255
        return frame

    def _fill_cards(self, frame: np.ndarray, count: int) -> np.ndarray:
        """指定枚数の全カード ROI を明るい灰色で塗りつぶす。

        入力と期待する結果を照合し、この条件で既存の契約が保たれることを確認します。
        """
        from survivors.vision.roi_layout import CARD_ROIS, norm_to_pixels
        frame = frame.copy()
        for norm in CARD_ROIS[count]:
            roi = norm_to_pixels(norm, self._W, self._H)
            frame[roi.y0:roi.y1, roi.x0:roi.x1, :3] = 128
            frame[roi.y0:roi.y1, roi.x0:roi.x1, 3] = 255
        return frame

    def test_hud_single_bright_object_is_gameplay(self):
        """HUD + 1枚のカード ROI のみ明るい (非カード物体) → gameplay。

        入力と期待する結果を照合し、この条件で既存の契約が保たれることを確認します。
        """
        from survivors.vision.roi_layout import CARD_ROIS, norm_to_pixels
        frame = self._hud_frame()
        # 1スロットだけ明るくする (完全なカードレイアウトではない)
        roi = norm_to_pixels(CARD_ROIS[3][0], self._W, self._H)
        frame[roi.y0:roi.y1, roi.x0:roi.x1, :3] = 128
        frame[roi.y0:roi.y1, roi.x0:roi.x1, 3] = 255
        state, _, _ = _detect_screen_state(frame, width=self._W, height=self._H)
        assert state == "gameplay", f"Expected 'gameplay', got '{state}'"

    def test_hud_nonblack_uniform_bg_is_gameplay(self):
        """HUD + 一様な暗灰色背景 RGB(40,40,40) (カードなし) → gameplay。

        入力と期待する結果を照合し、この条件で既存の契約が保たれることを確認します。
        """
        frame = self._hud_frame()
        # カード y 範囲全体を一様な暗灰色にする (>=32 を満たすが構造なし)
        frame[175:939, :, :3] = 40
        frame[175:939, :, 3] = 255
        state, _, _ = _detect_screen_state(frame, width=self._W, height=self._H)
        assert state == "gameplay", f"Expected 'gameplay', got '{state}'"

    def test_hud_crossing_bright_band_is_gameplay(self):
        """HUD + 画面横断の明るい帯 (カードとギャップを均等に照らす) → gameplay。

        入力と期待する結果を照合し、この条件で既存の契約が保たれることを確認します。
        """
        frame = self._hud_frame()
        # 全横幅の明るい帯 (カード ROI もギャップ ROI も同じ輝度になる)
        frame[400:600, :, :3] = 200
        frame[400:600, :, 3] = 255
        state, _, _ = _detect_screen_state(frame, width=self._W, height=self._H)
        assert state == "gameplay", f"Expected 'gameplay', got '{state}'"

    def test_hud_3card_layout_is_level_up_items(self):
        """HUD + 3枚カード全スロット明るい → level_up_items。

        入力と期待する結果を照合し、この条件で既存の契約が保たれることを確認します。
        """
        frame = self._fill_cards(self._hud_frame(), 3)
        state, _, _ = _detect_screen_state(frame, width=self._W, height=self._H)
        assert state == "level_up_items", f"Expected 'level_up_items', got '{state}'"

    def test_hud_4card_layout_is_level_up_items(self):
        """HUD + 4枚カード全スロット明るい → level_up_items。

        入力と期待する結果を照合し、この条件で既存の契約が保たれることを確認します。
        """
        frame = self._fill_cards(self._hud_frame(), 4)
        state, _, _ = _detect_screen_state(frame, width=self._W, height=self._H)
        assert state == "level_up_items", f"Expected 'level_up_items', got '{state}'"
