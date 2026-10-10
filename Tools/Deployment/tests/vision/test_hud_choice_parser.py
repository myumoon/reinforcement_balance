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
from .conftest import _make_gameplay_frame, _make_levelup_frame

_DUMMY_ARTIFACT_HASH = "c" * 64
_DUMMY_PROFILE_HASH = "a" * 64
_DUMMY_BUILD_HASH = "b" * 64
_SESSION_ID = "test-session-001"


# ── HudStateV1 スキーマ契約 ────────────────────────────────────────

class TestHudStateV1Schema:
    """HUD の値域とスキーマを検証する。

    HUD・カード・ボタンを作るときに、未対応の種類や範囲外の値を受け付けないことを調べます。
    """
    def _make_valid(self, **kwargs) -> HudStateV1:
        """有効な HUD 契約の基準値を作る。

        正常な gameplay の値を用意し、指定されたフィールドだけを上書きして異常な入力を作れるようにします。
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

        基準値から HudStateV1 を作り、スキーマ名と gameplay の画面状態がそのまま保存されるかを調べます。
        """
        s = self._make_valid()
        assert s.schema_version == HUD_STATE_SCHEMA_VERSION
        assert s.screen_state == "gameplay"

    def test_wrong_schema_version_raises(self):
        """未対応の HUD スキーマを拒否する。

        schema_version を未対応の値へ変えた場合に、生成時点で ValueError になることを確かめます。
        """
        with pytest.raises(ValueError, match="schema"):
            self._make_valid(schema_version="hud_state.v99")

    def test_unknown_screen_state_raises(self):
        """未定義の画面状態を拒否する。

        HUD の screen_state に定義外の名前を渡すと、生成時点で ValueError になることを確かめます。
        """
        with pytest.raises(ValueError, match="screen_state"):
            self._make_valid(screen_state="flying_saucers")

    def test_wrong_inventory_length_raises(self):
        """十二枠に足りない在庫を拒否する。

        在庫の要素数を十二枠より少なくし、スロット構成が不完全な HUD を生成できないことを確かめます。
        """
        with pytest.raises(ValueError, match="inventory"):
            self._make_valid(inventory=tuple([None] * 5))

    def test_hp_ratio_out_of_range_raises(self):
        """範囲外の HP 比率を拒否する。

        hp_ratio に0〜1の範囲外の値を渡すと、HUD の値域検証で ValueError になることを確かめます。
        """
        with pytest.raises(ValueError, match="hp_ratio"):
            self._make_valid(hp_ratio=1.5)

    def test_xp_ratio_out_of_range_raises(self):
        """範囲外の XP 比率を拒否する。

        xp_ratio に0〜1の範囲外の値を渡すと、HUD の値域検証で ValueError になることを確かめます。
        """
        with pytest.raises(ValueError, match="xp_ratio"):
            self._make_valid(xp_ratio=-0.1)

    def test_level_out_of_range_raises(self):
        """範囲外のプレイヤーレベルを拒否する。

        対応するレベル範囲を外れた値で HUD を作り、正常なレベルとして保存されないことを確かめます。
        """
        with pytest.raises(ValueError, match="level"):
            self._make_valid(level=100)

    def test_invalid_card_kind_raises(self):
        """未定義のカード種別を拒否する。

        カードの kind に定義外の名前を渡し、HUD のカード検証で ValueError になることを確かめます。
        """
        bad_card = ParsedCard(0, None, "dragon", None, 0.0, "bad", None)
        with pytest.raises(ValueError, match="kind"):
            self._make_valid(cards=(bad_card,))

    def test_invalid_button_type_raises(self):
        """未定義のボタン種別を拒否する。

        ボタンの button_type に定義外の名前を渡し、HUD のボタン検証で ValueError になることを確かめます。
        """
        bad_btn = ParsedButton("teleport", 0.0, "bad", None)
        with pytest.raises(ValueError, match="button_type"):
            self._make_valid(buttons=(bad_btn,))

    @pytest.mark.parametrize("state", SCREEN_STATES)
    def test_all_screen_states_accepted(self, state: str):
        """全画面状態が HudStateV1 に受け付けられる。

        SCREEN_STATES の各状態で HUD を生成し、定義済みの名前が検証で拒否されないことを確かめます。
        """
        s = self._make_valid(screen_state=state)
        assert s.screen_state == state

    def test_screen_states_exhaustive(self):
        """SCREEN_STATES が想定の状態セットを全て含む。

        gameplay・選択・停止などの状態名を明示した集合と比べ、画面状態の追加漏れや意図しない変更を見つけます。
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

    カード・ボタンを含む HUD を辞書へ変換して戻し、値の欠落や余分なキーの受け入れを調べます。
    """
    def _make_with_cards(self) -> HudStateV1:
        """カードとボタンを持つ HUD 契約を作る。

        wire の往復でカードやボタンも比較できるよう、空でない解析結果を持つ HUD を用意します。
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

        to_wire の辞書を from_wire で戻し、カードやボタンも含めて元の HUD と等しくなるかを調べます。
        """
        original = self._make_with_cards()
        wire = original.to_wire()
        restored = HudStateV1.from_wire(wire)
        assert restored == original

    def test_wire_is_json_serializable(self):
        """HUD の wire を JSON として保存できることを確認する。

        to_wire の辞書を JSON 文字列へ変換し、タプルや解析結果が保存を妨げないことを確かめます。
        """
        original = self._make_with_cards()
        wire = original.to_wire()
        serialized = json.dumps(wire)
        assert isinstance(serialized, str)

    def test_from_wire_extra_field_raises(self):
        """HUD の wire の余分なキーを拒否する。

        正しい辞書に未定義のキーを足し、from_wire が黙って読み飛ばさず ValueError にすることを確かめます。
        """
        original = self._make_with_cards()
        wire = original.to_wire()
        wire["extra_field"] = "bad"
        with pytest.raises(ValueError, match="mismatch"):
            HudStateV1.from_wire(wire)

    def test_from_wire_missing_field_raises(self):
        """HUD の wire の必須キー欠落を拒否する。

        正しい辞書からキーを一つ取り除き、from_wire が既定値で埋めず ValueError にすることを確かめます。
        """
        original = self._make_with_cards()
        wire = original.to_wire()
        del wire["screen_state"]
        with pytest.raises(ValueError, match="mismatch"):
            HudStateV1.from_wire(wire)


# ── _compute hashes ─────────────────────────────────────────────────

class TestComputeHashes:
    """在庫と候補集合の hash を検証する。

    同じ入力のハッシュが安定し、アイテム名や画面状態を変えるとハッシュも変わることを調べます。
    """
    def test_inventory_hash_deterministic(self):
        """同じ在庫から同じ hash を得る。

        同じスロット順の在庫を二度ハッシュ化し、実行ごとに識別値が変わらないことを確かめます。
        """
        inv = tuple(["whip", None, "gold"] + [None] * 9)
        h1 = _compute_inventory_hash(inv)
        h2 = _compute_inventory_hash(inv)
        assert h1 == h2

    def test_inventory_hash_changes_with_content(self):
        """在庫の変更が hash に反映されることを確認する。

        スロットのアイテム名が異なる二つの在庫を比べ、同一の在庫として扱われないことを確かめます。
        """
        inv1 = tuple(["whip"] + [None] * (INV_SLOT_COUNT - 1))
        inv2 = tuple([None] * INV_SLOT_COUNT)
        assert _compute_inventory_hash(inv1) != _compute_inventory_hash(inv2)

    def test_candidate_set_hash_deterministic(self):
        """同じ候補集合から同じ hash を得る。

        同じ画面状態とカード候補からハッシュを二度求め、候補集合の識別値が安定していることを確かめます。
        """
        card = ParsedCard(0, "whip", "weapon", 1, 0.8, "ok", None)
        h1 = _compute_candidate_set_hash("level_up_items", (card,))
        h2 = _compute_candidate_set_hash("level_up_items", (card,))
        assert h1 == h2

    def test_candidate_set_hash_changes_with_state(self):
        """画面状態の変更が候補 hash に反映されることを確認する。

        カード候補が同じでも画面状態を変えると、候補集合のハッシュが変わることを確かめます。
        """
        card = ParsedCard(0, "whip", "weapon", 1, 0.8, "ok", None)
        h1 = _compute_candidate_set_hash("level_up_items", (card,))
        h2 = _compute_candidate_set_hash("chest", (card,))
        assert h1 != h2


# ── HudParser smoke test ────────────────────────────────────────────

class TestHudParser:
    """HUD の解析結果と時間的な制約を検証する。

    合成画像から HUD を生成し、画面状態・wire のフィールド・タイマーの逆行防止・リセットを調べます。
    """
    def test_parse_blank_frame(self, dummy_parser_artifact_hash: str, blank_frame: np.ndarray):
        """全黒フレームで parse が例外なく HudStateV1 を返す。

        何も読めない画像でも、定義済みの画面状態と十二枠の在庫を持つ HUD が返ることを確かめます。
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

        合成した gameplay 画像を渡し、HudStateV1 として受け取れて画面状態が定義内に収まることを確かめます。
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

        wire のキーを全列挙した集合と比べ、必須フィールドの欠落や意図しない追加を見つけます。
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

        前に読んだ時刻より小さいタイマー値を次に読み、過去へ戻る値が None になることを確かめます。
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

        タイマーとプレイヤーレベルの過去値を設定してリセットし、次のセッションへ持ち越さないことを確かめます。
        """
        parser = HudParser(parser_artifact_hash=dummy_parser_artifact_hash)
        parser._prev_timer_seconds = 100.0
        parser._prev_level = 5
        parser.reset_temporal_state()
        assert parser._prev_timer_seconds is None
        assert parser._prev_level is None

    def test_invalid_artifact_hash_raises(self):
        """空の parser artifact hash を拒否する。

        解析器を識別する値を空文字にして初期化し、出所を識別できない解析器が作られないことを確かめます。
        """
        with pytest.raises(ValueError):
            HudParser(parser_artifact_hash="")


# ── ChoiceParser テスト ────────────────────────────────────────────

class TestChoiceParser:
    """カードとボタンの解析結果を検証する。

    画面状態ごとの候補・ボタンの出し分けと、読めないアイコンを推測しない動作を調べます。
    """
    def test_unknown_screen_state_raises(self, levelup_frame: np.ndarray):
        """未定義の画面状態を拒否する。

        カード解析へ定義外の screen_state を渡すと、解析を続けず ValueError になることを確かめます。
        """
        parser = ChoiceParser()
        with pytest.raises(ValueError, match="screen_state"):
            parser.parse(levelup_frame, screen_state="INVALID_STATE")

    def test_gameplay_state_returns_empty_cards(self, gameplay_frame: np.ndarray):
        """gameplay 状態ではカードとボタンが空。

        通常プレイ中の画像をカード解析へ渡し、選択候補として扱うカードが追加されないことを確かめます。
        """
        parser = ChoiceParser()
        result = parser.parse(gameplay_frame, screen_state="gameplay")
        assert isinstance(result, ChoiceParseResult)
        assert result.cards == ()

    def test_levelup_state_returns_result(self, levelup_frame: np.ndarray):
        """level_up_items 状態でカード解析が実行される (結果は空でもよい)。

        アイテム選択画面として解析し、候補が読めなくても解析結果と候補集合のハッシュが返ることを確かめます。
        """
        parser = ChoiceParser()
        result = parser.parse(levelup_frame, screen_state="level_up_items")
        assert isinstance(result, ChoiceParseResult)
        assert result.screen_state in SCREEN_STATES
        assert isinstance(result.candidate_set_hash, str)

    def test_chest_state_returns_chest_button(self):
        """chest 状態で ack_chest ボタンが返される可能性がある。

        宝箱画面のボタンが空か、宝箱を閉じる ack_chest を含む結果になることを確かめます。
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

        カード領域へ雑音画像を入れ、照合が不確かなカードにアイテム名を割り当てないことを確かめます。
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

        fallback 用の語彙に gold と chicken が含まれ、通常アイテムとは別に扱う対象が揃っていることを確かめます。
        """
        # fallback アイコンに見せかけた合成テンプレートでマッチさせる
        # ここでは ParsedCard.kind の検証のみ
        from survivors.vision.choice_parser import _FALLBACK_IDS
        assert "gold" in _FALLBACK_IDS
        assert "chicken" in _FALLBACK_IDS

    def test_buttons_not_confused_as_cards(self, blank_frame: np.ndarray):
        """reroll/skip/banish ボタンをカードとして誤認識しない。

        ボタン周辺を含む画像を解析し、出力カードのアイテム名にボタンの名前が紛れ込まないことを確かめます。
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

        カード・ボタン・各操作の可否など、呼び出し側が使う属性を解析結果から取り出せることを確かめます。
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

    試作用のアイコン画像集を解析で使えることと、正式成果物としては読み込めないことを分けて調べます。
    """
    def test_development_atlas_rejected_by_formal_loader(
        self, development_atlas_path: Path
    ):
        """development atlas は formal loader に拒否される。

        開発用 atlas を正式ロード処理へ渡し、FormalLoaderRejectedError で止まることを確かめます。
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

        開発用 atlas の照合器で HUD を解析し、正式ロードの資格が無くても試作時の解析は実行できることを確かめます。
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
        """対象プロファイルのカード枚数を実測配置で扱える。

        最後のカードまで矩形があり、画面からはみ出さないことを確かめます。
        """
        from survivors.target_profile import load_target_profile
        from survivors.vision.roi_layout import card_roi
        profile = load_target_profile()
        counts = profile.sections["choice_taxonomy"]["level_up_card_counts"]
        for count in counts:
            assert card_roi(count - 1).y1 <= 1080

    def test_fallback_vocabulary_in_closed_taxonomy(self):
        """fallback vocabulary が closed taxonomy と一致している。

    対象プロファイルの fallback 名と解析器の語彙を比べ、片側だけにあるアイテムが無いことを確かめます。
        """
        from survivors.target_profile import load_target_profile
        from survivors.vision.choice_parser import _FALLBACK_IDS
        profile = load_target_profile()
        fallbacks = set(profile.sections["choice_taxonomy"]["fallbacks"])
        assert fallbacks == _FALLBACK_IDS

    def test_capability_buttons_in_closed_taxonomy(self):
        """capability ボタン種別が closed taxonomy と一致している。

    対象プロファイルが許す操作名と解析器のボタン一覧を比べ、名前や種類のずれを見つけます。
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

    def test_card_transient_preserves_inventory_and_panel_hold(self, monkeypatch):
        """滑り込みのカードは訪問前在庫を壊さない。

        右枠がまだ立たない低信頼フレームを挟んでも、次の格子と位置結合できます。
        """
        parser, _ = self._parser(monkeypatch)
        gameplay = self._seed(parser)
        panel = self._panel()
        self._read(parser, panel)
        self._read(parser, panel)
        for _ in range(3):
            self._read(parser, gameplay)
        assert parser._gameplay_inventory[0] == "whip"
        cards = _make_levelup_frame()
        cards[300:900, 1268:1282, :3] = 0
        for _ in range(3):
            result = self._read(parser, cards)
            assert result.screen_state_reason == "card_transient"
            assert result.screen_state_confidence == .45
            assert result.inventory_levels == (None,) * 12
        assert parser._gameplay_inventory[0] == "whip"
        self._read(parser, panel)
        assert self._read(parser, panel).inventory[0] == "whip"

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
        """上端 XP バーの金枠二本を持つ画面を作る。

        格子のテストも同じ通常プレイ画面を入口にします。
        """
        return _make_gameplay_frame()

    def _fill_cards(self, frame: np.ndarray, count: int) -> np.ndarray:
        """指定枚数の実配置カードと中央ウィンドウを描く。

        枠と灰色面を持つ縦並びを共通 fixture から作ります。
        """
        return _make_levelup_frame(count)

    def test_hud_single_bright_object_is_gameplay(self):
        """HUD + 1枚のカード ROI のみ明るい (非カード物体) → gameplay。

    一つの明るい物体だけではカードの並びが揃わず、アイテム選択画面へ誤判定しないことを確かめます。
        """
        frame = self._hud_frame()
        # 1スロットだけ明るくする (完全なカードレイアウトではない)
        frame[267:271, 656:1265, :3] = (102, 203, 255)
        state, _, _ = _detect_screen_state(frame, width=self._W, height=self._H)
        assert state == "gameplay", f"Expected 'gameplay', got '{state}'"

    def test_hud_nonblack_uniform_bg_is_gameplay(self):
        """HUD + 一様な暗灰色背景 RGB(40,40,40) (カードなし) → gameplay。

    真っ黒ではない一様な背景でも、カードと隙間の明るさの差が無ければ通常プレイと判定することを確かめます。
        """
        frame = self._hud_frame()
        # カード y 範囲全体を一様な暗灰色にする (>=32 を満たすが構造なし)
        frame[175:939, :, :3] = 40
        frame[175:939, :, 3] = 255
        state, _, _ = _detect_screen_state(frame, width=self._W, height=self._H)
        assert state == "gameplay", f"Expected 'gameplay', got '{state}'"

    def test_hud_crossing_bright_band_is_gameplay(self):
        """HUD + 画面横断の明るい帯 (カードとギャップを均等に照らす) → gameplay。

    領域と隙間が一緒に明るくなる帯を描き、カードの並びとして誤認識しないことを確かめます。
        """
        frame = self._hud_frame()
        # 全横幅の明るい帯 (カード ROI もギャップ ROI も同じ輝度になる)
        frame[400:600, :, :3] = 200
        frame[400:600, :, 3] = 255
        state, _, _ = _detect_screen_state(frame, width=self._W, height=self._H)
        assert state == "gameplay", f"Expected 'gameplay', got '{state}'"

    def test_hud_3card_layout_is_level_up_items(self):
        """HUD + 3枚カード全スロット明るい → level_up_items。

    三枚のカード領域だけを明るくし、暗い隙間との組み合わせからアイテム選択画面になることを確かめます。
        """
        frame = self._fill_cards(self._hud_frame(), 3)
        state, _, _ = _detect_screen_state(frame, width=self._W, height=self._H)
        assert state == "level_up_items", f"Expected 'level_up_items', got '{state}'"

    def test_hud_4card_layout_is_level_up_items(self):
        """HUD + 4枚カード全スロット明るい → level_up_items。

    四枚のカード領域だけを明るくし、三枚の配置と同様にアイテム選択画面へ判定できることを確かめます。
        """
        frame = self._fill_cards(self._hud_frame(), 4)
        state, _, _ = _detect_screen_state(frame, width=self._W, height=self._H)
        assert state == "level_up_items", f"Expected 'level_up_items', got '{state}'"


class TestMeasuredLayout:
    """実測配置の画面判定と本番 choice 配線を検証する。

    未計測の能力や宝箱を開く操作を候補へ出さず、短い遮蔽だけを保持します。
    """

    @pytest.mark.parametrize("count", [1, 2, 3, 4])
    def test_card_counts_and_unknown_identities(self, count):
        """一枚から四枚の選択肢を上から順に返す。

        atlas がなければアイテム名は未知でも、枚数とクリック矩形は実測値になります。
        """
        from survivors.vision.roi_layout import card_roi
        frame = _make_levelup_frame(count)
        assert _detect_screen_state(frame) == ("level_up_items", .70, f"card_rows:{count}")
        result = ChoiceParser().parse(frame, screen_state="level_up_items")
        assert len(result.cards) == count
        for k, card in enumerate(result.cards):
            assert card.roi_xyxy == card_roi(k).as_xyxy()
            assert card.item_id is None and card.kind == "unknown" and card.level is None
            assert card.confidence == 0. and card.reason == "no_matcher"

    @pytest.mark.parametrize("bgr", [(70, 60, 100), (255, 255, 255)])
    def test_colored_and_bright_no_hud_frames_are_unknown(self, bgr):
        """明るさだけでタイトルや白画面を結果にしない。

        旧 layout score が高くなる色でも、実測の枠がない画像は不明のままです。
        """
        frame = np.full((1080, 1920, 4), 255, dtype=np.uint8)
        frame[..., :3] = bgr
        assert _detect_screen_state(frame) == ("unknown", .30, "no_hud")

    @pytest.mark.parametrize("hud", [False, True])
    def test_yellow_flash_is_chest(self, hud):
        """黄色い光は HUD の有無によらず宝箱になる。

        白飛びでバーが消えても、結果画面には分類しません。
        """
        frame = _make_gameplay_frame() if hud else np.zeros((1080, 1920, 4), dtype=np.uint8)
        frame[100:, :, :3] = (20, 220, 240)
        assert _detect_screen_state(frame) == ("chest", .60, "chest_flash")

    @pytest.mark.parametrize("variant", ["panel", "white_border", "gold_only", "decay"])
    def test_chest_panel_not_card_transient(self, variant):
        """宝箱の光や金色の行だけではカード過渡にしない。

        上枠の白飛びと黄色い光の減衰も、灰色面なしの宝箱パネルへ落とします。
        """
        frame = _make_levelup_frame(0)
        if variant == "white_border":
            frame[111:117, 642:1278, :3] = (255, 248, 255)
        if variant in {"gold_only", "decay"}:
            for top in (267, 424, 581):
                frame[top + 1:top + 4, 700:900 if variant == "gold_only" else 1200, :3] = (102, 203, 255)
        if variant == "decay":
            frame[100:220, :, :3] = (20, 220, 240)
            frame[111:117, 660:1260, :3] = (102, 203, 255)
        assert _detect_screen_state(frame) == ("chest", .65, "chest_panel")

    @pytest.mark.parametrize("count", [2, 3])
    def test_missing_right_border_is_card_transient(self, count):
        """右枠が立つ前は低信頼の過渡に留める。

        灰色面が見えていても、滑り込み中はカード候補を返しません。
        """
        frame = _make_levelup_frame(count)
        frame[300:900, 1268:1282, :3] = 0
        assert _detect_screen_state(frame) == ("level_up_items", .45, "card_transient")
        assert ChoiceParser().parse(frame, screen_state="level_up_items").cards == ()

    @pytest.mark.parametrize("dim", [False, True])
    def test_pause_needs_dim_hud_and_resume(self, dim):
        """半暗の HUD と再開ボタンで一時停止を判定する。

        通常 HUD に青い物体があるだけでは paused にしません。
        """
        frame = _make_gameplay_frame()
        if dim:
            frame[(2, 32), 300:1600, :3] = (51, 102, 127)
        frame[952:1033, 1448:1732, :3] = (198, 61, 38)
        assert _detect_screen_state(frame) == (("paused", .70, "pause_menu") if dim else ("gameplay", .60, "hud_present"))
        assert ChoiceParser().parse(frame, screen_state="paused").buttons == ()

    def test_death_with_red_tinted_hud(self):
        """赤く染まったバーと GAME OVER・赤ボタンで死亡を認める。

        枠の青成分が変わっても warm クラスなら HUD の証拠にできます。
        """
        frame = _make_gameplay_frame()
        frame[(2, 32), 300:1600, :3] = (87, 128, 249)
        frame[292:352, 709:1202, :3] = (102, 203, 255)
        frame[708:780, 819:1101, :3] = (12, 43, 211)
        assert _detect_screen_state(frame) == ("death", .70, "game_over")

    @pytest.mark.parametrize("hud", [False, True])
    def test_result_panel_independent_of_hud(self, hud):
        """結果パネルは HUD の有無に依存せず認める。

        大きな枠と背景色を揃え、通常プレイへの誤分類を防ぎます。
        """
        frame = _make_gameplay_frame() if hud else np.zeros((1080, 1920, 4), dtype=np.uint8)
        frame[66:922, 277:1644, :3] = (102, 203, 255)
        frame[72:912, 283:1638, :3] = (116, 79, 75)
        assert _detect_screen_state(frame) == ("result", .70, "result_panel")

    def test_unsupported_resolution_and_empty(self):
        """未計測の解像度と空画像を拒否する。

        実画像と設定値が食い違う場合もカードやボタンを推測しません。
        """
        frame = np.zeros((720, 1280, 4), dtype=np.uint8)
        for kwargs in ({}, {"width": 1280, "height": 720}):
            assert _detect_screen_state(frame, **kwargs) == ("unknown", 0., "unsupported_resolution")
        assert _detect_screen_state(np.zeros((0, 0, 4), dtype=np.uint8)) == ("unknown", 0., "empty_frame")
        for parser in (ChoiceParser(), ChoiceParser(width=1280, height=720)):
            result = parser.parse(frame, screen_state="level_up_items")
            assert result.screen_state == "unknown" and result.cards == () and result.buttons == ()

    def test_reroll_inner_color_and_fail_closed_capability(self):
        """リロールは金枠なしでも青い内側で検出する。

        未計測の skip と banish は返さず、能力信頼度も零に保ちます。
        """
        frame = _make_levelup_frame()
        frame[253:317, 1413:1693, :3] = (205, 64, 39)
        result = ChoiceParser().parse(frame, screen_state="level_up_items")
        assert result.reroll_available and [b.button_type for b in result.buttons] == ["reroll"]
        assert not result.skip_available and not result.banish_available
        assert result.capability_confidence == 0. and result.capability_reason == "skip_banish_roi_undefined"

    @pytest.mark.parametrize("button_type", ["ack_chest", "reroll"])
    @pytest.mark.parametrize("white_rows", [5, 11])
    def test_action_buttons_require_stable_color_and_allow_white_text(self, button_type, white_rows):
        """操作ボタンは白文字を許し、フェード中の面は出力しない。

        定常の青色割合を信頼度へ直結させず、既存 retry gate を満たす観測だけを返します。
        """
        from survivors.vision import roi_layout as layout
        frame = _make_levelup_frame(0 if button_type == "ack_chest" else 3)
        inner = layout.CHEST_ACK_INNER_ROI if button_type == "ack_chest" else layout.REROLL_BUTTON_INNER_ROI
        inner.crop(frame)[..., :3] = (205, 64, 39)
        inner.crop(frame)[:white_rows, :, :3] = 255
        if button_type == "ack_chest":
            frame[914:919, 660:1260, :3] = (102, 203, 255)
        result = ChoiceParser().parse(frame, screen_state="chest" if button_type == "ack_chest" else "level_up_items")
        if white_rows == 5:
            assert len(result.buttons) == 1
            assert result.buttons[0].button_type == button_type
            assert result.buttons[0].confidence >= .99
        else:
            assert result.buttons == ()
            assert not result.reroll_available

    @pytest.mark.parametrize("after_panel", [False, True])
    def test_paused_grid_cannot_charge_panel_hold_or_adopt_levels(self, after_panel):
        """一時停止の格子を保持せず、再開の一枚目から gameplay を返す。

        本物の格子証拠を二枚続けても、段階値の一致履歴と採用値へ混ぜません。
        """
        from survivors.vision.slot_level_parser import has_panel_evidence, parse_slot_levels
        paused = TestSlotPanelIntegration()._panel()
        paused[(2, 32), 300:1600, :3] = (51, 102, 127)
        paused[952:1033, 1448:1732, :3] = (198, 61, 38)
        assert has_panel_evidence(parse_slot_levels(paused, 1920, 1080))
        parser = HudParser(parser_artifact_hash=_DUMMY_ARTIFACT_HASH)
        if after_panel:
            for _ in range(2):
                self._read(parser, TestSlotPanelIntegration()._panel())
            assert parser._panel_hold > 0 and parser._slot_adopted[0] is not None
        for _ in range(2):
            assert self._read(parser, paused).screen_state == "paused"
            assert parser._panel_hold == 0
            assert parser._slot_prev == parser._slot_adopted == [None] * 12
            assert parser._slot_prev_cell_counts == parser._slot_cell_counts == [0] * 12
        resumed = self._read(parser, _make_gameplay_frame())
        assert (resumed.screen_state, resumed.screen_state_reason) == ("gameplay", "hud_present")
        assert resumed.cards == () and resumed.inventory_levels == (None,) * 12

    @pytest.mark.parametrize("phase", ["open", "close", "no_button"])
    def test_ack_chest_requires_shortened_panel(self, phase):
        """縮んだパネルの終了ボタンだけを ack_chest にする。

        開くボタン自身の短い金枠も描き、閾値に届かないことを確かめます。
        """
        frame = _make_levelup_frame(0)
        if phase != "no_button":
            frame[835:902, 822:1098, :3] = (255, 96, 64)
        frame[910:913, 810:1109, :3] = (102, 203, 255)
        if phase == "close":
            frame[914:919, 660:1260, :3] = (102, 203, 255)
        result = ChoiceParser().parse(frame, screen_state="chest")
        assert [b.button_type for b in result.buttons] == (["ack_chest"] if phase == "close" else [])

    @pytest.mark.parametrize("state", ["death", "result"])
    def test_confirm_uses_only_inner_color(self, state):
        """終端の終了ボタンは対応する内側の色で観測する。

        金枠を省いても死亡は赤、結果は青で confirm が返ります。
        """
        frame = np.zeros((1080, 1920, 4), dtype=np.uint8)
        if state == "death":
            frame[708:780, 819:1101, :3] = (12, 43, 211)
        else:
            frame[974:1040, 821:1099, :3] = (255, 96, 64)
        assert [b.button_type for b in ChoiceParser().parse(frame, screen_state=state).buttons] == ["confirm"]

    def test_hud_parser_wires_cards_buttons_and_hash(self):
        """本番入口に cards・buttons・候補 hash を配線する。

        通常プレイへ戻ると選択情報は空になり、能力は適用外になります。
        """
        parser = HudParser(parser_artifact_hash=_DUMMY_ARTIFACT_HASH)
        frame = _make_levelup_frame()
        frame[253:317, 1413:1693, :3] = (205, 64, 39)
        result = self._read(parser, frame)
        assert len(result.cards) == 3 and result.reroll_available
        assert result.candidate_set_hash == _compute_candidate_set_hash(result.screen_state, result.cards)
        assert result.candidate_set_hash != _compute_candidate_set_hash(result.screen_state, ())
        gameplay = self._read(parser, _make_gameplay_frame())
        assert gameplay.cards == () and gameplay.buttons == () and gameplay.capability_reason == "not_applicable"

    def _read(self, parser, frame):
        """同じ session の一枚を本番 parser に流す。

        保持の回数だけを比較できるよう、数字や在庫は fixture のままにします。
        """
        return parser.parse(frame, session_id=_SESSION_ID, frame_index=0, captured_monotonic_ns=0)

    def test_chest_hold_expires_after_fourteen_unknown_frames(self):
        """不明十四枚を保持し十五枚目で chest の保持を切る。

        保持の結果が新しい宝箱証拠として再計上されることを防ぎます。
        """
        parser = HudParser(parser_artifact_hash=_DUMMY_ARTIFACT_HASH)
        assert self._read(parser, _make_levelup_frame(0)).screen_state == "chest"
        black = np.zeros((1080, 1920, 4), dtype=np.uint8)
        for _ in range(14):
            held = self._read(parser, black)
            assert (held.screen_state, held.screen_state_confidence, held.screen_state_reason) == ("chest", .50, "chest_hold")
        assert self._read(parser, black).screen_state == "unknown"

    def test_chest_hold_does_not_delay_gameplay_and_resets(self):
        """通常プレイを保持せず reset で宝箱記憶を消す。

        新しい session の黒画面が直前の宝箱に分類されないことを確かめます。
        """
        parser = HudParser(parser_artifact_hash=_DUMMY_ARTIFACT_HASH)
        self._read(parser, _make_levelup_frame(0))
        assert self._read(parser, _make_gameplay_frame()).screen_state == "gameplay"
        self._read(parser, _make_levelup_frame(0))
        parser.reset_temporal_state()
        assert parser._chest_hold == 0
        assert self._read(parser, np.zeros((1080, 1920, 4), dtype=np.uint8)).screen_state == "unknown"

    def test_shared_hud_types_keep_reexports(self):
        """移動したカード型とボタン型の旧 import を維持する。

        hud_parser からも新 module からも同一の型を読み込めます。
        """
        from survivors.vision import hud_types
        assert hud_types.ParsedCard is ParsedCard and hud_types.ParsedButton is ParsedButton

    @pytest.mark.parametrize("items", [("whip",), ("whip", "gold"), ("whip", "gold", "chicken"),
                                       ("whip", "gold", "chicken", "whip"), ("gold", "chicken", "gold")])
    def test_card_surface_identities_from_development_atlas(self, dev_atlas_manifest, items):
        """開発 atlas のカード色を実測位置の icon から照合する。

        各 item の一 level だけを使い、カードの画素サイズで feature を作って位置と surface を検証します。
        """
        from dataclasses import replace
        from build_survivors_icon_atlas import _make_synth_template, _ITEM_COLORS_BGR
        from survivors.vision.icon_matcher import build_template_feature
        entries = tuple(e for e in dev_atlas_manifest.entries if e.item_id in items and e.level == e.max_level)
        icons = tuple(_make_synth_template(item, next(e.max_level for e in entries if e.item_id == item),
                                           _ITEM_COLORS_BGR[item]) for item in items)
        # 合成64px円を縮めた際の edge 差を、実ゲームの同一 crop 照合と混同しない。
        ys = np.linspace(0, 63, 55).astype(int)
        xs = np.linspace(0, 63, 51).astype(int)
        icons = tuple(icon[ys[:, None], xs[None, :]] for icon in icons)
        features = {item: build_template_feature(icon) for item, icon in zip(items, icons)}
        entries = tuple(replace(e, feature=features[e.item_id]) if e.surface == "card" else e for e in entries)
        matcher = IconMatcher(replace(dev_atlas_manifest, entries=entries))
        result = ChoiceParser(icon_matcher=matcher).parse(_make_levelup_frame(len(items), icons), screen_state="level_up_items")
        assert tuple(c.item_id for c in result.cards) == items
        assert all(c.level is None and c.confidence >= .5 for c in result.cards)
        fallback = sum(item in {"gold", "chicken"} for item in items) > len(items) // 2
        assert result.screen_state == ("level_up_fallback" if fallback else "level_up_items")
        if fallback:
            assert all(c.kind == "fallback" for c in result.cards if c.item_id in {"gold", "chicken"})

    def test_black_card_icons_stay_unknown(self, dev_atlas_manifest):
        """黒い icon は名前を推測しない。

        カード面と枚数が明確でも、atlas と区別できない icon は未知のまま返します。
        """
        result = ChoiceParser(icon_matcher=IconMatcher(dev_atlas_manifest)).parse(_make_levelup_frame(), screen_state="level_up_items")
        assert len(result.cards) == 3 and all(c.item_id is None for c in result.cards)

    def test_pixel_rois_are_bounded_disjoint_and_exact(self):
        """全実測 ROI が画面内にありカード同士が重ならない。

        icon は各カードの内側に収まり、未計測の解像度で拡大縮小しません。
        """
        from survivors.vision import roi_layout as layout
        width, height = layout.REFERENCE_SIZE
        for value in vars(layout).values():
            if isinstance(value, layout.PixelROI):
                assert 0 <= value.x0 < value.x1 <= width and 0 <= value.y0 < value.y1 <= height
        cards = [layout.card_roi(k) for k in range(4)]
        assert [c.as_xyxy() for c in cards] == [(656, y, 1265, y + 154) for y in (267, 424, 581, 738)]
        assert all(a.y1 <= b.y0 for a, b in zip(cards, cards[1:]))
        for k, card in enumerate(cards):
            icon = layout.card_icon_roi(k)
            assert icon.as_xyxy() == (669, card.y0 + 13, 720, card.y0 + 68)
            assert card.x0 < icon.x0 < icon.x1 < card.x1 and card.y0 < icon.y0 < icon.y1 < card.y1
        for function in (layout.card_roi, layout.card_icon_roi):
            for args in ((-1,), (4,), (0, 1280, 720)):
                with pytest.raises(ValueError):
                    function(*args)

    @pytest.mark.parametrize("state", ["gameplay", "level_up_items", "death", "result", "paused"])
    def test_chest_hold_never_replaces_other_known_states(self, monkeypatch, state):
        """宝箱の保持は他の既知状態を上書きしない。

        死亡や一時停止へ切り替わった時点で、その状態を即座に観測へ渡します。
        """
        parser = HudParser(parser_artifact_hash=_DUMMY_ARTIFACT_HASH)
        self._read(parser, _make_levelup_frame(0))
        monkeypatch.setattr("survivors.vision.hud_parser._detect_screen_state", lambda *a, **k: (state, .70, "known_state"))
        assert self._read(parser, _make_gameplay_frame()).screen_state == state

    @pytest.mark.parametrize("shape, reason", [((720, 1280, 4), "unsupported_resolution"),
                                              ((0, 0, 4), "empty_frame")])
    def test_invalid_frame_cannot_inherit_modal_hold(self, shape, reason):
        """解像度違いと空画像を直前の画面保持で隠さない。

        宝箱と段階格子のどちらを見た後でも、元の理由と信頼度零を維持します。
        """
        for prior in (_make_levelup_frame(0), TestSlotPanelIntegration()._panel()):
            parser = HudParser(parser_artifact_hash=_DUMMY_ARTIFACT_HASH)
            self._read(parser, prior)
            result = self._read(parser, np.zeros(shape, dtype=np.uint8))
            assert (result.screen_state, result.screen_state_confidence, result.screen_state_reason,
                    result.cards, result.buttons) == ("unknown", 0., reason, (), ())
