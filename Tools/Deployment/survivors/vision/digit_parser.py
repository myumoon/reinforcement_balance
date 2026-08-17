"""数字グリフ解析: binarize + connected components + 正規化テンプレート距離。

timer (MM:SS) とレベル (1..99) を画像クロップから読み取ります。
OCR ライブラリには依存せず、小さな二値テンプレートとのマッチングで解析します。
low-confidence の場合は推測せず validity=0.0 と理由を返します。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

# 二値化しきい値（Otsu に近い固定値; キャリブレーション前の开発用）
_BIN_THRESHOLD: Final[int] = 128

# テンプレートの正規化サイズ (高さ × 幅)
_TMPL_H: Final[int] = 8
_TMPL_W: Final[int] = 5

# 最小 margin しきい値: これ未満は即時 None を返す
_MIN_MARGIN: Final[float] = 0.05
# confidence 正規化基準 margin (0.25 の margin で confidence≈1.0 になる)
_NOMINAL_MARGIN: Final[float] = 0.25
# 信頼度しきい値: これ未満なら value=None を返す
# ponytail: 開発用合成テンプレートは類似度が高いため 0.20 に設定; 実データでは 04-04 で再キャリブレーション
_LOW_CONF: Final[float] = 0.20

# --------------------------------------------------------------------------
# 開発用合成グリフテンプレート (8x5 の二値パターン)
# 各行: 0=黒, 1=白 (前景)
# --------------------------------------------------------------------------
# フォーマット: 行×列 の numpy uint8 (0 or 255)
_RAW_GLYPHS: dict[str, list[list[int]]] = {
    "0": [
        [0, 1, 1, 1, 0],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 1, 1],
        [1, 0, 1, 0, 1],
        [1, 1, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ],
    "1": [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ],
    "2": [
        [0, 1, 1, 1, 0],
        [1, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 1, 1, 0],
        [0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0],
    ],
    "3": [
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ],
    "4": [
        [0, 0, 0, 1, 0],
        [0, 0, 1, 1, 0],
        [0, 1, 0, 1, 0],
        [1, 0, 0, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0],
    ],
    "5": [
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 1, 1, 1, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
        [1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ],
    "6": [
        [0, 1, 1, 1, 0],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 1, 1, 1, 0],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ],
    "7": [
        [1, 1, 1, 1, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
    ],
    "8": [
        [0, 1, 1, 1, 0],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [0, 1, 1, 1, 0],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ],
    "9": [
        [0, 1, 1, 1, 0],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [0, 1, 1, 1, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ],
    ":": [
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ],
}

# テンプレートを numpy 配列に変換 (float32, 0 or 1)
TEMPLATES: dict[str, NDArray[np.float32]] = {
    ch: np.array(rows, dtype=np.float32)
    for ch, rows in _RAW_GLYPHS.items()
}


@dataclass(frozen=True, slots=True)
class DigitResult:
    """1 文字の解析結果。"""

    char: str | None       # 解析された文字。不明なら None
    confidence: float      # 0.0..1.0; _LOW_CONF 未満で char=None
    reason: str            # confidence が低い場合の理由


@dataclass(frozen=True, slots=True)
class TimerResult:
    """タイマー (MM:SS) の解析結果。"""

    seconds: float | None  # 合計秒数; 不明なら None
    confidence: float
    reason: str
    display: str | None = None  # "MM:SS" 形式; オプション


@dataclass(frozen=True, slots=True)
class LevelResult:
    """レベル (1..99) の解析結果。"""

    level: int | None
    confidence: float
    reason: str


def _binarize(crop_gray: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """グレースケール画像を二値化する。"""
    return (crop_gray >= _BIN_THRESHOLD).astype(np.uint8)


def _to_gray(crop_bgra: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """BGRA 画像をグレースケールに変換する。"""
    # BT.601 coefficients
    gray = (
        crop_bgra[..., 0].astype(np.float32) * 0.114
        + crop_bgra[..., 1].astype(np.float32) * 0.587
        + crop_bgra[..., 2].astype(np.float32) * 0.299
    )
    return np.clip(gray, 0, 255).astype(np.uint8)


def _resize_binary(patch: NDArray[np.uint8], h: int, w: int) -> NDArray[np.float32]:
    """バイナリパッチを (h, w) にリサイズする（nearest neighbor）。"""
    if patch.size == 0:
        return np.zeros((h, w), dtype=np.float32)
    src_h, src_w = patch.shape
    row_idx = (np.arange(h) * src_h // h).clip(0, src_h - 1)
    col_idx = (np.arange(w) * src_w // w).clip(0, src_w - 1)
    return patch[np.ix_(row_idx, col_idx)].astype(np.float32)


def _match_char(patch: NDArray[np.uint8]) -> DigitResult:
    """バイナリパッチを全テンプレートと比較して最良一致を返す。"""
    if patch.size == 0 or patch.sum() == 0:
        return DigitResult(None, 0.0, "empty_patch")

    resized = _resize_binary(patch, _TMPL_H, _TMPL_W)
    scores: dict[str, float] = {}
    for ch, tmpl in TEMPLATES.items():
        diff = float(np.mean(np.abs(resized - tmpl)))
        # diff は 0..1; 1-diff がスコア
        scores[ch] = 1.0 - diff

    sorted_chars = sorted(scores, key=lambda c: scores[c], reverse=True)
    best_ch = sorted_chars[0]
    best_score = scores[best_ch]
    second_score = scores[sorted_chars[1]] if len(sorted_chars) > 1 else 0.0

    # margin = best - second; _NOMINAL_MARGIN で正規化して confidence を計算
    margin = best_score - second_score
    if margin < _MIN_MARGIN:
        return DigitResult(None, margin / _NOMINAL_MARGIN, f"low_margin:{margin:.3f}")
    confidence = min(1.0, margin / _NOMINAL_MARGIN)

    if confidence < _LOW_CONF:
        return DigitResult(None, confidence, f"low_confidence:{confidence:.3f}")
    return DigitResult(best_ch, confidence, "ok")


def _split_char_columns(
    bin_image: NDArray[np.uint8],
    n_chars: int,
) -> list[NDArray[np.uint8]]:
    """二値画像を n_chars 個の列スロットに均等分割して返す。"""
    w = bin_image.shape[1]
    step = max(1, w // n_chars)
    cols: list[NDArray[np.uint8]] = []
    for i in range(n_chars):
        x0 = i * step
        x1 = x0 + step if i < n_chars - 1 else w
        cols.append(bin_image[:, x0:x1])
    return cols


def parse_timer(crop_bgra: NDArray[np.uint8]) -> TimerResult:
    """タイマー領域のクロップ (BGRA) から MM:SS を解析する。

    二値化 → 5 スロット分割 → 各スロットをテンプレートマッチング で
    MM:SS 形式のタイマー値を返します。low-confidence の場合は
    seconds=None と reason を返し、前回値を再利用しません。
    """
    if crop_bgra.ndim != 3 or crop_bgra.shape[2] != 4 or crop_bgra.size == 0:
        return TimerResult(None, 0.0, "invalid_crop")

    gray = _to_gray(crop_bgra)
    binary = _binarize(gray)

    # タイマー: M1 M2 : S1 S2 → 5 文字
    slots = _split_char_columns(binary, 5)
    chars: list[DigitResult] = [_match_char(s) for s in slots]

    confidences = [c.confidence for c in chars]
    min_conf = min(confidences)
    avg_conf = float(np.mean(confidences))

    if min_conf < _LOW_CONF:
        reasons = [c.reason for c in chars if c.confidence < _LOW_CONF]
        return TimerResult(None, avg_conf, f"low_char_confidence:{reasons}")

    # chars[2] should be ':'; if not, still try to parse MM:SS
    parsed: list[str] = []
    for i, ch_result in enumerate(chars):
        if ch_result.char is None:
            return TimerResult(None, avg_conf, f"unreadable_char_at_{i}")
        parsed.append(ch_result.char)

    # colon at position 2
    if parsed[2] != ":":
        # treat as best effort – might be off by 1
        return TimerResult(None, avg_conf, f"colon_not_found:got_{parsed[2]}")

    try:
        minutes = int(parsed[0]) * 10 + int(parsed[1])
        seconds_part = int(parsed[3]) * 10 + int(parsed[4])
    except (ValueError, TypeError):
        return TimerResult(None, avg_conf, "digit_parse_error")

    if not (0 <= minutes <= 99 and 0 <= seconds_part <= 59):
        return TimerResult(None, avg_conf, f"out_of_range:{minutes}:{seconds_part}")

    total_seconds = float(minutes * 60 + seconds_part)
    display = f"{minutes:02d}:{seconds_part:02d}"
    return TimerResult(total_seconds, avg_conf, "ok", )


def parse_level(crop_bgra: NDArray[np.uint8]) -> LevelResult:
    """レベル領域のクロップ (BGRA) からレベル値 (1..99) を解析する。

    1〜2 桁の数字を解析します。low-confidence の場合は level=None を返します。
    """
    if crop_bgra.ndim != 3 or crop_bgra.shape[2] != 4 or crop_bgra.size == 0:
        return LevelResult(None, 0.0, "invalid_crop")

    gray = _to_gray(crop_bgra)
    binary = _binarize(gray)

    # 前景ピクセルの割合で桁数を推定
    foreground_ratio = float(binary.mean())
    if foreground_ratio < 0.02:
        return LevelResult(None, 0.0, "blank_region")

    # 最大 2 桁試行 (2 桁 → 1 桁 の順)
    for n_chars in (2, 1):
        slots = _split_char_columns(binary, n_chars)
        results = [_match_char(s) for s in slots]
        avg_conf = float(np.mean([r.confidence for r in results]))

        if all(r.confidence >= _LOW_CONF for r in results):
            digits_str = "".join(r.char or "?" for r in results)
            try:
                value = int(digits_str)
            except ValueError:
                continue
            if 1 <= value <= 99:
                return LevelResult(value, avg_conf, "ok")

    # フォールバック: 1 桁で再試行した最大 confidence を報告
    single_result = _match_char(binary)
    return LevelResult(None, single_result.confidence, f"level_out_of_range_or_low_conf")


def apply_temporal_timer(
    current: TimerResult,
    previous_seconds: float | None,
) -> TimerResult:
    """タイマーの時間的単調性制約を適用する。

    タイマーは常に増加するはずです (逆行は破棄されます)。
    30:00 を超える逆行も同様に reject します。
    """
    if current.seconds is None or previous_seconds is None:
        return current
    # タイマーが前回より大幅に減少した場合は reject (30 秒の許容範囲)
    if current.seconds < previous_seconds - 30.0:
        return TimerResult(
            None,
            current.confidence * 0.5,
            f"timer_regression:{previous_seconds:.1f}→{current.seconds:.1f}",
        )
    return current


def apply_temporal_level(
    current: LevelResult,
    previous_level: int | None,
) -> LevelResult:
    """レベルの単調増加制約を適用する。

    レベルは増加のみ許容します。逆行は reject します。
    """
    if current.level is None or previous_level is None:
        return current
    if current.level < previous_level:
        return LevelResult(
            None,
            current.confidence * 0.5,
            f"level_regression:{previous_level}→{current.level}",
        )
    return current
