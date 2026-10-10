"""viewport-relative ROI anchor の定義と画素座標への変換。

画面要素（タイマー・HP バー・XP バー・インベントリ・カード等）の位置を
正規化座標 (0..1) で定義し、解像度に合わせてピクセル座標へ変換します。
レイアウト妥当性を anchor mismatch score で判定する機能も持ちます。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray


# 正規化座標 (x0, y0, x1, y1)  ─  解像度非依存で定義する
# 1920x1080 を基準に Vampire Survivors の HUD 配置から導いた値
# 04-04/04-05 のキャリブレーションで上書きされる予定
class _N(NamedTuple):
    """画面内の矩形を正規化座標で表す。

    各辺を画面の幅と高さに対する割合で保存します。
    """

    x0: float
    y0: float
    x1: float
    y1: float


# タイマー (MM:SS) – 画面上部中央
TIMER_ROI = _N(0.406, 0.005, 0.594, 0.043)

# HP バー – 画面最上部の赤いバー
HP_BAR_ROI = _N(0.000, 0.030, 1.000, 0.052)

# XP バー – HP バーの下の青いバー
XP_BAR_ROI = _N(0.000, 0.052, 1.000, 0.070)

# レベル数字 – タイマー左下付近
LEVEL_ROI = _N(0.020, 0.005, 0.100, 0.042)

# インベントリスロット (6 weapon + 6 passive = 12 スロット)
# スロット0〜5: 武器、スロット6〜11: パッシブ
# 左上の二段。1920x1080 の gameplay アイコン枠を正規化する
INV_SLOT_ROIS: tuple[_N, ...] = tuple(
    _N((100 + 46 * (i % 6)) / 1920, (40 if i < 6 else 86) / 1080,
       (142 + 46 * (i % 6)) / 1920, (82 if i < 6 else 128) / 1080)
    for i in range(12)
)

EMPTY_SLOT_ID = "empty_slot"


class SlotLevelPipGrid(NamedTuple):
    """段階マークの格子とパネル下端の測定座標。

    横方向は画面幅、縦方向は画面高さに対する割合です。
    内部サンプルとセル寸法は縦横の割合を持ち、丸めは画素へ戻すときだけ行います。
    """

    cell_x0: float = 124 / 1920
    slot_dx: float = 45.75 / 1920
    col_dx: float = 13.5 / 1920
    cols: int = 3
    weapon_row_y0: tuple[float, float, float] = (110 / 1080, 124 / 1080, 137 / 1080)
    passive_row_y0: tuple[float, float] = (213 / 1080, 226 / 1080)
    inner_offset: tuple[float, float] = (3 / 1920, 3 / 1080)
    inner_size: tuple[float, float] = (5 / 1920, 5 / 1080)
    cell_w: float = 10 / 1920
    cell_h: float = 9 / 1080
    row_span_x: tuple[float, float] = (120 / 1920, 400 / 1920)
    bottom_scan_y: tuple[float, float] = (140 / 1080, 300 / 1080)
    bottom_gold_ratio: float = 0.9


SLOT_LEVEL_PIP_GRID = SlotLevelPipGrid()

# 段階マーク（アイコン下の小さな四角の列）が見える画面状態の集合
# level-up 画面の左上パネルにだけ出る。gameplay・chest では出ないので、
# HUD truth の expected_slot_levels はこの集合の画面でだけ非 null にできる。
SLOT_LEVEL_VISIBLE_STATES: frozenset[str] = frozenset({"level_up_items", "level_up_fallback"})

# レベルアップカード (最大 4 枚) – 3 枚と 4 枚で位置が変わる
# 3 枚レイアウト: 等間隔 3 分割
_CARD3_ROIS: tuple[_N, ...] = (
    _N(0.055, 0.162, 0.345, 0.870),
    _N(0.375, 0.162, 0.625, 0.870),
    _N(0.655, 0.162, 0.945, 0.870),
)
# 4 枚レイアウト: 等間隔 4 分割
_CARD4_ROIS: tuple[_N, ...] = (
    _N(0.030, 0.162, 0.265, 0.870),
    _N(0.285, 0.162, 0.490, 0.870),
    _N(0.510, 0.162, 0.715, 0.870),
    _N(0.735, 0.162, 0.970, 0.870),
)
CARD_ROIS: dict[int, tuple[_N, ...]] = {3: _CARD3_ROIS, 4: _CARD4_ROIS}

# カード間ギャップ – レベルアップオーバーレイ背景確認用 (カードより暗いはず)
CARD_GAP_ROIS: dict[int, tuple[_N, ...]] = {
    3: (
        _N(0.345, 0.162, 0.375, 0.870),  # card1-card2 間
        _N(0.625, 0.162, 0.655, 0.870),  # card2-card3 間
    ),
    4: (
        _N(0.265, 0.162, 0.285, 0.870),  # card1-card2 間
        _N(0.490, 0.162, 0.510, 0.870),  # card2-card3 間
        _N(0.715, 0.162, 0.735, 0.870),  # card3-card4 間
    ),
}

# ボタン (reroll / skip / banish) – カード下部
BUTTON_ROIS: dict[str, _N] = {
    "reroll": _N(0.060, 0.882, 0.265, 0.950),
    "skip":   _N(0.375, 0.882, 0.625, 0.950),
    "banish": _N(0.735, 0.882, 0.940, 0.950),
}

# chest 確認ボタン
CHEST_ACK_ROI = _N(0.375, 0.700, 0.625, 0.780)

# スクリーン全体領域（状態判定用アンカーサンプル点）
# 状態判定には中央領域の支配色を使う
SCREEN_CENTER_ROI = _N(0.380, 0.380, 0.620, 0.620)

# layout validity チェック用アンカー点（HP バー前景色が存在するか等）
_ANCHOR_SAMPLE_POINTS = (
    (0.5, 0.041),   # HP バー中央
    (0.5, 0.061),   # XP バー中央
    (0.5, 0.025),   # タイマー中央
)


@dataclass(frozen=True, slots=True)
class PixelROI:
    """ピクセル座標で表した矩形 ROI。

    画面内の位置を呼出し元へ渡し、同じ範囲で切り出しや座標変換を行えます。
    """

    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        """矩形の横幅を画素数で返す。

        画面内の位置を呼出し元へ渡し、同じ範囲で切り出しや座標変換を行えます。
        """
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        """矩形の高さを画素数で返す。

        画面内の位置を呼出し元へ渡し、同じ範囲で切り出しや座標変換を行えます。
        """
        return self.y1 - self.y0

    def crop(self, frame_bgra: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """フレーム配列から ROI を切り出す。

        画面内の位置を呼出し元へ渡し、同じ範囲で切り出しや座標変換を行えます。
        """
        return frame_bgra[self.y0:self.y1, self.x0:self.x1]

    def as_xyxy(self) -> tuple[int, int, int, int]:
        """矩形の四辺を座標の組で返す。

        画面内の位置を呼出し元へ渡し、同じ範囲で切り出しや座標変換を行えます。
        """
        return (self.x0, self.y0, self.x1, self.y1)


def norm_to_pixels(
    norm: _N,
    width: int = 1920,
    height: int = 1080,
) -> PixelROI:
    """正規化 ROI をピクセル ROI に変換する。

    画面内の位置を呼出し元へ渡し、同じ範囲で切り出しや座標変換を行えます。
    """
    x0 = max(0, min(int(norm.x0 * width), width - 1))
    y0 = max(0, min(int(norm.y0 * height), height - 1))
    x1 = max(x0 + 1, min(int(norm.x1 * width), width))
    y1 = max(y0 + 1, min(int(norm.y1 * height), height))
    return PixelROI(x0, y0, x1, y1)


def card_rois_for_count(count: int, width: int = 1920, height: int = 1080) -> tuple[PixelROI, ...]:
    """カード枚数に応じた ROI リストを返す。不明枚数は空を返す。

    画面内の位置を呼出し元へ渡し、同じ範囲で切り出しや座標変換を行えます。
    """
    norms = CARD_ROIS.get(count)
    if norms is None:
        return ()
    return tuple(norm_to_pixels(n, width, height) for n in norms)


def layout_validity_score(
    frame_bgra: NDArray[np.uint8],
    *,
    width: int = 1920,
    height: int = 1080,
) -> float:
    """HUD レイアウトの妥当性スコア (0.0..1.0) を返す。

    HP バーと XP バーのアンカー領域に前景色ピクセルが存在するかを確認し、
    存在割合を妥当性スコアとして返します。0.5 未満はレイアウト不正とみなします。
    """
    if frame_bgra.ndim != 3 or frame_bgra.shape[2] != 4:
        return 0.0

    votes: list[float] = []
    hp_roi = norm_to_pixels(HP_BAR_ROI, width, height)
    xp_roi = norm_to_pixels(XP_BAR_ROI, width, height)

    for roi in (hp_roi, xp_roi):
        crop = roi.crop(frame_bgra)
        if crop.size == 0:
            votes.append(0.0)
            continue
        # 前景ピクセル: いずれかのチャンネルが 32 以上あれば non-black
        non_black = np.any(crop[..., :3] >= 32, axis=-1)
        votes.append(float(np.mean(non_black)))

    return float(np.mean(votes)) if votes else 0.0
