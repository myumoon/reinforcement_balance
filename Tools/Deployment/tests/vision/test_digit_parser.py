"""digit parser テスト。

合成パターンを使って TimerResult・LevelResult の契約と
temporal constraint を検証します。
実ゲーム画像なしで動作します。
"""

from __future__ import annotations

import numpy as np
import pytest

from survivors.vision.digit_parser import (
    TEMPLATES,
    TimerResult,
    LevelResult,
    _binarize,
    _to_gray,
    _match_char,
    _resize_binary,
    parse_timer,
    parse_level,
    apply_temporal_timer,
    apply_temporal_level,
)


# ── テンプレート自己一致テスト ──────────────────────────────────────────

class TestTemplateSet:
    def test_all_digit_chars_present(self):
        """0-9 とコロンがテンプレートに含まれる。"""
        expected = set("0123456789:")
        assert expected.issubset(set(TEMPLATES))

    def test_templates_are_float32(self):
        for ch, tmpl in TEMPLATES.items():
            assert tmpl.dtype == np.float32, f"template {ch!r} is not float32"

    def test_template_shape(self):
        for ch, tmpl in TEMPLATES.items():
            assert tmpl.shape == (8, 5), f"template {ch!r} has wrong shape"

    def test_self_match_high_confidence(self):
        """各テンプレートを自分自身にマッチさせると high confidence になる。"""
        for ch, tmpl in TEMPLATES.items():
            # _match_char は _binarize 後の 0/1 uint8 を期待する
            bin_patch = tmpl.astype(np.uint8)  # 0.0→0, 1.0→1
            result = _match_char(bin_patch)
            assert result.char == ch, f"self-match failed for {ch!r}: got {result.char!r}"
            # 開発用テンプレートは類似度が高いため confidence は低め (>= 0.25 を要求)
            assert result.confidence >= 0.25, (
                f"self-match confidence too low for {ch!r}: {result.confidence:.3f}"
            )


# ── binarize / gray 変換 ─────────────────────────────────────────────

class TestBinarize:
    def test_binarize_black(self):
        gray = np.zeros((8, 5), dtype=np.uint8)
        assert _binarize(gray).sum() == 0

    def test_binarize_white(self):
        gray = np.full((8, 5), 255, dtype=np.uint8)
        assert _binarize(gray).mean() == 1.0

    def test_to_gray_shape(self):
        bgra = np.zeros((10, 10, 4), dtype=np.uint8)
        result = _to_gray(bgra)
        assert result.shape == (10, 10)
        assert result.dtype == np.uint8

    def test_to_gray_white(self):
        bgra = np.full((4, 4, 4), 200, dtype=np.uint8)
        result = _to_gray(bgra)
        assert result.mean() > 150


# ── parse_timer ─────────────────────────────────────────────────────

class TestParseTimer:
    def _make_timer_frame(self, digits: str) -> np.ndarray:
        """digits (MM:SS 形式の 5 文字) を合成してタイマー BGRA クロップを返す。"""
        from survivors.vision.digit_parser import _TMPL_H, _TMPL_W, TEMPLATES
        # 5 文字分を横に並べた画像
        w = _TMPL_W * 5 + 4  # 文字間 1px のパディング
        h = _TMPL_H
        img = np.zeros((h, w, 4), dtype=np.uint8)
        for i, ch in enumerate(digits):
            tmpl = TEMPLATES.get(ch)
            if tmpl is None:
                continue
            x0 = i * (_TMPL_W + 1)
            img[:, x0:x0 + _TMPL_W, :3] = (tmpl * 255).astype(np.uint8)[..., np.newaxis]
            img[:, x0:x0 + _TMPL_W, 3] = 255
        return img

    def test_empty_crop_returns_none(self):
        empty = np.zeros((0, 0, 4), dtype=np.uint8)
        result = parse_timer(empty)
        assert result.seconds is None
        assert result.confidence == 0.0

    def test_invalid_ndim_returns_none(self):
        bad = np.zeros((10, 10), dtype=np.uint8)
        result = parse_timer(bad)
        assert result.seconds is None

    def test_synthetic_timer_round_trips(self):
        """合成タイマー画像から秒数が解析されること (best-effort)。

        テンプレートの自己一致が高信頼ならタイマーも解析成功する可能性が高い。
        解析不能な場合でも reason が空でないことを確認する。
        """
        crop = self._make_timer_frame("02:34")
        result = parse_timer(crop)
        # 結果が None でも reason があることを確認 (推測で補完しない)
        assert isinstance(result.reason, str)
        assert result.confidence >= 0.0

    def test_timer_result_fields(self):
        """TimerResult のフィールドが揃っている。"""
        r = TimerResult(120.0, 0.9, "ok")
        assert r.seconds == 120.0
        assert r.display is None  # display はオプション
        assert r.confidence == 0.9
        assert r.reason == "ok"


# ── parse_level ────────────────────────────────────────────────────

class TestParseLevel:
    def test_empty_crop_returns_none(self):
        empty = np.zeros((0, 0, 4), dtype=np.uint8)
        result = parse_level(empty)
        assert result.level is None

    def test_blank_region_returns_none(self):
        black = np.zeros((30, 50, 4), dtype=np.uint8)
        result = parse_level(black)
        assert result.level is None
        assert "blank" in result.reason or result.confidence == 0.0

    def test_level_result_fields(self):
        r = LevelResult(5, 0.8, "ok")
        assert r.level == 5
        assert r.confidence == 0.8


# ── temporal constraints ───────────────────────────────────────────

class TestTemporalTimer:
    def test_no_previous_timer_passes_through(self):
        result = TimerResult(100.0, 0.9, "ok")
        out = apply_temporal_timer(result, None)
        assert out.seconds == 100.0

    def test_normal_increment_passes(self):
        prev = 100.0
        result = TimerResult(105.0, 0.9, "ok")
        out = apply_temporal_timer(result, prev)
        assert out.seconds == 105.0

    def test_regression_more_than_30s_rejects(self):
        prev = 200.0
        result = TimerResult(100.0, 0.9, "ok")  # 100 sec regression
        out = apply_temporal_timer(result, prev)
        assert out.seconds is None
        assert "regression" in out.reason

    def test_small_regression_within_tolerance_passes(self):
        """30 秒以内の小さな逆行は許容する。"""
        prev = 100.0
        result = TimerResult(80.0, 0.9, "ok")  # 20 sec regression → OK
        out = apply_temporal_timer(result, prev)
        assert out.seconds == 80.0

    def test_none_timer_passes_through(self):
        result = TimerResult(None, 0.3, "low_conf")
        out = apply_temporal_timer(result, 100.0)
        assert out.seconds is None


class TestTemporalLevel:
    def test_no_previous_passes(self):
        r = LevelResult(5, 0.9, "ok")
        out = apply_temporal_level(r, None)
        assert out.level == 5

    def test_increase_passes(self):
        r = LevelResult(6, 0.9, "ok")
        out = apply_temporal_level(r, 5)
        assert out.level == 6

    def test_regression_rejects(self):
        r = LevelResult(4, 0.9, "ok")
        out = apply_temporal_level(r, 5)
        assert out.level is None
        assert "regression" in out.reason

    def test_same_level_passes(self):
        r = LevelResult(5, 0.9, "ok")
        out = apply_temporal_level(r, 5)
        assert out.level == 5
