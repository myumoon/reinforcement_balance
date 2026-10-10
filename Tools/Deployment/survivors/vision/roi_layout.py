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

# layout validity チェック用アンカー点（HP バー前景色が存在するか等）
_ANCHOR_SAMPLE_POINTS = (
    (0.5, 0.041),   # HP バー中央
    (0.5, 0.061),   # XP バー中央
    (0.5, 0.025),   # タイマー中央
)


@dataclass(frozen=True, slots=True)
class PixelROI:
    """ピクセル座標で表した矩形 ROI。

    左上と右下の座標で、画像から切り出す領域を表します。
    右端と下端の画素は切り出しに含みません。
    """

    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        """矩形の横幅を画素数で返す。

        右端の x 座標から左端の x 座標を引き、切り出し画像の列数を求めます。
        """
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        """矩形の高さを画素数で返す。

        下端の y 座標から上端の y 座標を引き、切り出し画像の行数を求めます。
        """
        return self.y1 - self.y0

    def crop(self, frame_bgra: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """フレーム配列から ROI を切り出す。

        y0 から y1 の直前までの行と、x0 から x1 の直前までの列を取り出します。
        BGRA の色成分はそのまま残します。
        """
        return frame_bgra[self.y0:self.y1, self.x0:self.x1]

    def as_xyxy(self) -> tuple[int, int, int, int]:
        """矩形の四辺を座標の組で返す。

        左・上・右・下の順に整数座標を並べ、照合結果などの矩形情報として使えるようにします。
        """
        return (self.x0, self.y0, self.x1, self.y1)


def norm_to_pixels(
    norm: _N,
    width: int = 1920,
    height: int = 1080,
) -> PixelROI:
    """正規化 ROI をピクセル ROI に変換する。

    画面サイズを掛けて整数座標へ変え、画面の外にはみ出す部分を収めます。
    切り出し領域は縦横とも最低一画素を確保します。
    """
    x0 = max(0, min(int(norm.x0 * width), width - 1))
    y0 = max(0, min(int(norm.y0 * height), height - 1))
    x1 = max(x0 + 1, min(int(norm.x1 * width), width))
    y1 = max(y0 + 1, min(int(norm.y1 * height), height))
    return PixelROI(x0, y0, x1, y1)


# 日本語 UI の実測値。正規化の往復による一画素のずれを避ける。
REFERENCE_SIZE = (1920, 1080)
HUD_BAR_ROWS = (2, 32)
HUD_BAR_X = (300, 1600)
LEVEL_UP_WINDOW_ROI = PixelROI(642, 111, 1278, 965)
WINDOW_TOP_BAND = (108, 124)
WINDOW_BOTTOM_BAND = (905, 970)
WINDOW_BORDER_X = (660, 1260)
WINDOW_RIGHT_BAND = (1268, 1282)
WINDOW_RIGHT_Y = (300, 900)
CARD_TOP_Y = (267, 424, 581, 738)
CARD_HEIGHT = 154
CARD_X = (656, 1265)
CARD_ROW_BAND_X = (700, 1200)
CARD_ICON_OFFSET = (669, 13, 720, 68)
CARD_ZONE_ROI = PixelROI(660, 265, 1260, 740)
PAUSE_RESUME_INNER_ROI = PixelROI(1448, 952, 1732, 1033)
REROLL_BUTTON_ROI = PixelROI(1404, 247, 1702, 329)
REROLL_BUTTON_INNER_ROI = PixelROI(1413, 253, 1693, 317)
CHEST_ACK_BUTTON_ROI = PixelROI(810, 827, 1107, 913)
CHEST_ACK_INNER_ROI = PixelROI(822, 835, 1098, 902)
DEATH_TEXT_ROI = PixelROI(700, 280, 1220, 360)
DEATH_CONFIRM_ROI = PixelROI(810, 702, 1110, 789)
DEATH_CONFIRM_INNER_ROI = PixelROI(819, 708, 1101, 780)
RESULT_WINDOW_ROI = PixelROI(277, 66, 1644, 922)
RESULT_TOP_BAND = (62, 72)
RESULT_BOTTOM_BAND = (912, 926)
RESULT_BORDER_X = (300, 1620)
RESULT_BG_ROI = PixelROI(300, 80, 1620, 900)
RESULT_CONFIRM_ROI = PixelROI(812, 968, 1108, 1049)
RESULT_CONFIRM_INNER_ROI = PixelROI(821, 974, 1099, 1040)


def card_roi(k: int, width: int = 1920, height: int = 1080) -> PixelROI:
    """上から k 番目のカード矩形を実測座標で返す。

    一枚目を零として数え、未計測の解像度や五枚目以降は受け付けません。
    """
    if (width, height) != REFERENCE_SIZE or not 0 <= k < len(CARD_TOP_Y):
        raise ValueError("unsupported card layout")
    return PixelROI(CARD_X[0], CARD_TOP_Y[k], CARD_X[1], CARD_TOP_Y[k] + CARD_HEIGHT)


def card_icon_roi(k: int, width: int = 1920, height: int = 1080) -> PixelROI:
    """カード左端のアイコン内側だけを返す。

    横位置は画面の絶対座標、縦位置だけは各カード上端からの差です。
    """
    top = card_roi(k, width, height).y0
    x0, y0, x1, y1 = CARD_ICON_OFFSET
    return PixelROI(x0, top + y0, x1, top + y1)


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
