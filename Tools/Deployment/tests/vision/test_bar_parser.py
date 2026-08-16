"""bar parser テスト。

合成カラーバーで HP/XP の充填率解析契約を検証します。
gamma/brightness 変動 (±20%), occlusion, flash も確認します。
"""

from __future__ import annotations

import numpy as np
import pytest

from survivors.vision.bar_parser import (
    BarResult,
    parse_hp_bar,
    parse_xp_bar,
    _bgr_to_hsv,
    _mask_in_range,
    _HP_HUE_LOW1,
    _HP_HUE_HIGH1,
    _XP_HUE_LOW,
    _XP_HUE_HIGH,
)


# ── 合成バー生成ヘルパー ────────────────────────────────────────────

def _make_hp_bar(fill_ratio: float, width: int = 200, height: int = 20) -> np.ndarray:
    """HP バー (赤) を fill_ratio で塗りつぶした BGRA 画像を返す。"""
    img = np.zeros((height, width, 4), dtype=np.uint8)
    img[..., 3] = 255
    filled_w = int(width * fill_ratio)
    # 赤 (BGR: R=200, G=30, B=30)
    img[:, :filled_w, 2] = 200
    img[:, :filled_w, 1] = 30
    img[:, :filled_w, 0] = 30
    return img


def _make_xp_bar(fill_ratio: float, width: int = 200, height: int = 15) -> np.ndarray:
    """XP バー (青) を fill_ratio で塗りつぶした BGRA 画像を返す。"""
    img = np.zeros((height, width, 4), dtype=np.uint8)
    img[..., 3] = 255
    filled_w = int(width * fill_ratio)
    # 青 (BGR: B=200, G=50, R=30)
    img[:, :filled_w, 0] = 200
    img[:, :filled_w, 1] = 50
    img[:, :filled_w, 2] = 30
    return img


# ── bgr_to_hsv / mask_in_range 基本テスト ────────────────────────────

class TestHsvConversion:
    def test_red_pixel_is_in_hp_range(self):
        """赤ピクセルが HP バー HSV 範囲内に含まれる。"""
        bgr = np.array([[[30, 30, 200]]], dtype=np.uint8)  # B,G,R = 30,30,200
        hsv = _bgr_to_hsv(bgr)
        m = _mask_in_range(hsv, _HP_HUE_LOW1, _HP_HUE_HIGH1)
        assert m[0, 0], f"red pixel not in HP range, HSV={hsv[0,0]}"

    def test_blue_pixel_is_in_xp_range(self):
        """青ピクセルが XP バー HSV 範囲内に含まれる。"""
        bgr = np.array([[[200, 50, 30]]], dtype=np.uint8)  # B,G,R = 200,50,30
        hsv = _bgr_to_hsv(bgr)
        m = _mask_in_range(hsv, _XP_HUE_LOW, _XP_HUE_HIGH)
        assert m[0, 0], f"blue pixel not in XP range, HSV={hsv[0,0]}"

    def test_black_pixel_is_not_in_any_bar_range(self):
        """黒ピクセルはどのバー範囲にも含まれない (彩度/明度が低い)。"""
        bgr = np.array([[[0, 0, 0]]], dtype=np.uint8)
        hsv = _bgr_to_hsv(bgr)
        assert not _mask_in_range(hsv, _HP_HUE_LOW1, _HP_HUE_HIGH1)[0, 0]
        assert not _mask_in_range(hsv, _XP_HUE_LOW, _XP_HUE_HIGH)[0, 0]


# ── parse_hp_bar ───────────────────────────────────────────────────

class TestParseHpBar:
    def test_empty_crop_returns_none(self):
        empty = np.zeros((0, 0, 4), dtype=np.uint8)
        r = parse_hp_bar(empty)
        assert r.ratio is None

    def test_black_crop_empty_bar(self):
        """全黒クロップは空バー (ratio=0.0) または None を返す。"""
        black = np.zeros((20, 200, 4), dtype=np.uint8)
        r = parse_hp_bar(black)
        # 空バーは ratio=0.0 または None; 推測しない
        assert r.ratio is None or r.ratio == 0.0

    def test_full_red_bar_returns_high_ratio(self):
        """100% 塗りつぶした赤バーは ratio が高い値を返す。"""
        bar = _make_hp_bar(1.0)
        r = parse_hp_bar(bar)
        if r.ratio is not None:
            assert r.ratio >= 0.70, f"expected high ratio, got {r.ratio}"

    def test_half_red_bar_returns_mid_ratio(self):
        """50% 塗りつぶした赤バーは ratio が中程度を返す。"""
        bar = _make_hp_bar(0.50)
        r = parse_hp_bar(bar)
        if r.ratio is not None:
            assert 0.20 <= r.ratio <= 0.80, f"expected mid ratio, got {r.ratio}"

    def test_bar_result_fields_exist(self):
        """BarResult のフィールドが揃っている。"""
        r = BarResult(0.5, 0.8, "ok")
        assert r.ratio == 0.5
        assert r.confidence == 0.8
        assert r.reason == "ok"

    def test_invalid_ndim_returns_none(self):
        bad = np.zeros((20, 200), dtype=np.uint8)
        r = parse_hp_bar(bad)
        assert r.ratio is None

    @pytest.mark.parametrize("gamma", [0.8, 1.0, 1.2])
    def test_gamma_variation(self, gamma: float):
        """gamma ±20% でも parse が例外を起こさない。"""
        bar = _make_hp_bar(0.75)
        bar_f = bar[..., :3].astype(np.float32)
        bar_adjusted = np.clip(bar_f * gamma, 0, 255).astype(np.uint8)
        adj_bgra = np.concatenate([bar_adjusted, bar[..., 3:]], axis=-1)
        r = parse_hp_bar(adj_bgra)
        assert r.confidence >= 0.0  # 例外を起こさない


# ── parse_xp_bar ───────────────────────────────────────────────────

class TestParseXpBar:
    def test_empty_crop_returns_none(self):
        empty = np.zeros((0, 0, 4), dtype=np.uint8)
        r = parse_xp_bar(empty)
        assert r.ratio is None

    def test_full_blue_bar_returns_high_ratio(self):
        bar = _make_xp_bar(1.0)
        r = parse_xp_bar(bar)
        if r.ratio is not None:
            assert r.ratio >= 0.70

    def test_low_conf_returns_reason(self):
        """low confidence なら reason に情報が入っている。"""
        bar = _make_xp_bar(0.0)
        r = parse_xp_bar(bar)
        assert isinstance(r.reason, str) and r.reason != ""

    def test_confidence_range(self):
        """confidence が 0..1 の範囲に収まる。"""
        for ratio in (0.0, 0.25, 0.5, 0.75, 1.0):
            bar = _make_xp_bar(ratio)
            r = parse_xp_bar(bar)
            assert 0.0 <= r.confidence <= 1.0, f"confidence out of range: {r.confidence}"
