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
# 画面左下に横並び
_INV_Y0, _INV_Y1 = 0.866, 0.972
_INV_SLOT_W = 0.047
INV_SLOT_ROIS: tuple[_N, ...] = tuple(
    _N(i * _INV_SLOT_W, _INV_Y0, (i + 1) * _INV_SLOT_W, _INV_Y1)
    for i in range(12)
)

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
    """ピクセル座標で表した矩形 ROI。"""

    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    def crop(self, frame_bgra: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """フレーム配列から ROI を切り出す。"""
        return frame_bgra[self.y0:self.y1, self.x0:self.x1]

    def as_xyxy(self) -> tuple[int, int, int, int]:
        return (self.x0, self.y0, self.x1, self.y1)


def norm_to_pixels(
    norm: _N,
    width: int = 1920,
    height: int = 1080,
) -> PixelROI:
    """正規化 ROI をピクセル ROI に変換する。"""
    x0 = max(0, min(int(norm.x0 * width), width - 1))
    y0 = max(0, min(int(norm.y0 * height), height - 1))
    x1 = max(x0 + 1, min(int(norm.x1 * width), width))
    y1 = max(y0 + 1, min(int(norm.y1 * height), height))
    return PixelROI(x0, y0, x1, y1)


def card_rois_for_count(count: int, width: int = 1920, height: int = 1080) -> tuple[PixelROI, ...]:
    """カード枚数に応じた ROI リストを返す。不明枚数は空を返す。"""
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
