"""HP バー・XP バーの色域セグメンテーションによる充填率解析。

バー領域クロップの前景色ピクセル割合を充填率として返します。
flash / occlusion / gamma 変動を考慮し、低信頼時は推測せず理由を返します。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

# 信頼度しきい値
_LOW_CONF: Final[float] = 0.35

# HSV 範囲定義 (OpenCV 形式: H 0..179, S 0..255, V 0..255)
# 前景判定は BGR → HSV 変換後に範囲内ピクセルを数える
# HP バー: 赤〜橙 (H: 0..15 or 160..179)
_HP_HUE_LOW1: Final[tuple[int, int, int]] = (0, 80, 80)
_HP_HUE_HIGH1: Final[tuple[int, int, int]] = (15, 255, 255)
_HP_HUE_LOW2: Final[tuple[int, int, int]] = (160, 80, 80)
_HP_HUE_HIGH2: Final[tuple[int, int, int]] = (179, 255, 255)

# XP バー: 青〜青紫 (H: 100..140)
_XP_HUE_LOW: Final[tuple[int, int, int]] = (100, 60, 60)
_XP_HUE_HIGH: Final[tuple[int, int, int]] = (140, 255, 255)

# 最小前景ピクセル率: これ未満は「空バー」または「オクルージョン」とみなす
_MIN_FOREGROUND_RATIO: Final[float] = 0.01


@dataclass(frozen=True, slots=True)
class BarResult:
    """バー充填率の解析結果。"""

    ratio: float | None  # 0.0..1.0; 不明なら None
    confidence: float    # 0.0..1.0
    reason: str          # "ok", "flash", "occlusion", "low_foreground", 等


def _bgr_to_hsv(bgr: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """BGR (uint8) を HSV (uint8, H 0..179, S/V 0..255) に変換する。

    OpenCV 非依存の純 NumPy 実装。H の精度は整数レベルで十分。
    """
    b = bgr[..., 0].astype(np.float32) / 255.0
    g = bgr[..., 1].astype(np.float32) / 255.0
    r = bgr[..., 2].astype(np.float32) / 255.0

    vmax = np.maximum(np.maximum(r, g), b)
    vmin = np.minimum(np.minimum(r, g), b)
    delta = vmax - vmin

    # Saturation
    s = np.where(vmax > 0, delta / (vmax + 1e-7), 0.0)

    # Hue (0..360)
    h = np.zeros_like(r)
    mask_r = (vmax == r) & (delta > 0)
    mask_g = (vmax == g) & (delta > 0)
    mask_b = (vmax == b) & (delta > 0)
    h[mask_r] = (60.0 * ((g[mask_r] - b[mask_r]) / (delta[mask_r] + 1e-7))) % 360.0
    h[mask_g] = 60.0 * ((b[mask_g] - r[mask_g]) / (delta[mask_g] + 1e-7)) + 120.0
    h[mask_b] = 60.0 * ((r[mask_b] - g[mask_b]) / (delta[mask_b] + 1e-7)) + 240.0

    # OpenCV scale: H 0..179, S/V 0..255
    h_cv = np.clip(h / 2.0, 0, 179).astype(np.uint8)
    s_cv = np.clip(s * 255.0, 0, 255).astype(np.uint8)
    v_cv = np.clip(vmax * 255.0, 0, 255).astype(np.uint8)

    return np.stack([h_cv, s_cv, v_cv], axis=-1)


def _mask_in_range(
    hsv: NDArray[np.uint8],
    low: tuple[int, int, int],
    high: tuple[int, int, int],
) -> NDArray[np.bool_]:
    """HSV 画像内の各ピクセルが [low, high] 範囲内かを返す。"""
    low_arr = np.array(low, dtype=np.uint8)
    high_arr = np.array(high, dtype=np.uint8)
    return np.all((hsv >= low_arr) & (hsv <= high_arr), axis=-1)


def _foreground_ratio_hp(hsv: NDArray[np.uint8]) -> float:
    """HP バーの前景ピクセル比率を返す (赤〜橙、2 範囲の OR)。"""
    mask1 = _mask_in_range(hsv, _HP_HUE_LOW1, _HP_HUE_HIGH1)
    mask2 = _mask_in_range(hsv, _HP_HUE_LOW2, _HP_HUE_HIGH2)
    return float(np.mean(mask1 | mask2))


def _foreground_ratio_xp(hsv: NDArray[np.uint8]) -> float:
    """XP バーの前景ピクセル比率を返す (青〜青紫)。"""
    mask = _mask_in_range(hsv, _XP_HUE_LOW, _XP_HUE_HIGH)
    return float(np.mean(mask))


def _estimate_fill_ratio(
    crop_bgra: NDArray[np.uint8],
    foreground_fn,
) -> tuple[float, float, str]:
    """前景ピクセルの横方向分布からバー充填率を推定する。

    左端から右端に向かって前景ピクセルが連続している部分の割合を充填率とします。
    バーが left-to-right で満ちる想定です。
    """
    if crop_bgra.size == 0:
        return (0.0, 0.0, "empty_crop")

    bgr = crop_bgra[..., :3]
    hsv = _bgr_to_hsv(bgr)
    fg_ratio = foreground_fn(hsv)

    if fg_ratio < _MIN_FOREGROUND_RATIO:
        return (0.0, 0.3, "very_low_foreground")

    # 列ごとの前景率 (左から右)
    h, w = hsv.shape[:2]
    col_fg = np.mean(foreground_fn(hsv).reshape(1, -1) if False else  # noqa
                     np.column_stack([
                         _mask_in_range(hsv[:, j:j+1], *(_HP_HUE_LOW1, _HP_HUE_HIGH1))
                         if foreground_fn == _foreground_ratio_hp else
                         _mask_in_range(hsv[:, j:j+1], *(_XP_HUE_LOW, _XP_HUE_HIGH))
                         for j in range(w)
                     ]), axis=0)

    # 右端から最初に前景がある列を見つける
    filled_cols = (col_fg >= 0.2).astype(int)
    if filled_cols.sum() == 0:
        return (0.0, 0.5, "no_filled_columns")

    # 最初と最後の前景列から充填率を推定
    first_fg = int(np.argmax(filled_cols))
    last_fg = int(len(filled_cols) - 1 - np.argmax(filled_cols[::-1]))
    fill_ratio = float(last_fg + 1) / float(w)

    # 信頼度: 前景ピクセル率が高いほど高い、ただし flash (全体が明るい) は低め
    confidence = min(1.0, fg_ratio * 3.0)
    return (fill_ratio, confidence, "ok")


def _compute_column_fg(
    hsv: NDArray[np.uint8],
    foreground_fn,
) -> NDArray[np.float32]:
    """各列の前景ピクセル比率を配列で返す。"""
    w = hsv.shape[1]
    result = np.empty(w, dtype=np.float32)
    for j in range(w):
        col = hsv[:, j:j+1]
        if foreground_fn == _foreground_ratio_hp:
            m1 = _mask_in_range(col, _HP_HUE_LOW1, _HP_HUE_HIGH1)
            m2 = _mask_in_range(col, _HP_HUE_LOW2, _HP_HUE_HIGH2)
            result[j] = float(np.mean(m1 | m2))
        else:
            result[j] = float(np.mean(_mask_in_range(col, _XP_HUE_LOW, _XP_HUE_HIGH)))
    return result


def _parse_bar(
    crop_bgra: NDArray[np.uint8],
    foreground_fn,
) -> BarResult:
    """バー領域から充填率を解析して BarResult を返す。"""
    if crop_bgra.ndim != 3 or crop_bgra.shape[2] != 4:
        return BarResult(None, 0.0, "invalid_crop")
    if crop_bgra.size == 0:
        return BarResult(None, 0.0, "empty_crop")

    bgr = crop_bgra[..., :3]
    hsv = _bgr_to_hsv(bgr)
    fg_ratio = foreground_fn(hsv)

    if fg_ratio < _MIN_FOREGROUND_RATIO:
        return BarResult(None, 0.4, "empty_or_occluded")

    col_fg = _compute_column_fg(hsv, foreground_fn)

    # 充填列 (前景率 >= 20%)
    filled = (col_fg >= 0.20)
    if not filled.any():
        return BarResult(0.0, 0.35, "no_filled_columns")

    # 最後の充填列位置 = 充填率
    w = len(col_fg)
    last_fg = int(np.max(np.where(filled)))
    fill_ratio = float(last_fg + 1) / float(w)
    fill_ratio = max(0.0, min(1.0, fill_ratio))

    # 信頼度: 前景ピクセル率と充填パターンの単調性から計算
    # 充填部分が左から連続していれば高信頼
    gap_penalty = 0.0
    if last_fg > 0:
        expected_filled = filled[:last_fg + 1]
        gap_count = int(np.sum(~expected_filled))
        gap_penalty = gap_count / max(1, last_fg + 1)

    confidence = min(1.0, fg_ratio * 2.5) * (1.0 - gap_penalty * 0.5)

    if confidence < _LOW_CONF:
        return BarResult(None, confidence, f"low_conf:fg={fg_ratio:.2f},gap={gap_penalty:.2f}")

    # flash 検出: バー全体が非常に明るい場合
    v_channel = hsv[..., 2]
    if float(np.mean(v_channel)) > 240:
        return BarResult(fill_ratio, confidence * 0.7, "flash_detected")

    return BarResult(fill_ratio, confidence, "ok")


def parse_hp_bar(crop_bgra: NDArray[np.uint8]) -> BarResult:
    """HP バー領域のクロップ (BGRA) から充填率を解析する。

    赤〜橙の前景色を検出し、左から右への充填率を返します。
    """
    return _parse_bar(crop_bgra, _foreground_ratio_hp)


def parse_xp_bar(crop_bgra: NDArray[np.uint8]) -> BarResult:
    """XP バー領域のクロップ (BGRA) から充填率を解析する。

    青〜青紫の前景色を検出し、左から右への充填率を返します。
    """
    return _parse_bar(crop_bgra, _foreground_ratio_xp)
