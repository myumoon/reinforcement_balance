"""level-up パネルの段階マークを画素の割合から読む。

点灯した宝石の数が現在レベルです。格子の並びや色が不確かな枠は不明にし、
パネル下端の外側を空枠として扱います。identity の照合は HudParser が担当します。
"""

from __future__ import annotations

from typing import Literal, NamedTuple

import numpy as np
from numpy.typing import NDArray

from .roi_layout import SLOT_LEVEL_PIP_GRID, SlotLevelPipGrid

_CELL_RATIO = 0.6
_GOLD_R_MIN, _GOLD_G_MIN, _GOLD_B_MAX = 200, 150, 180
_DARK_R_MAX, _DARK_G_MAX, _DARK_B_MAX = 90, 60, 80
_GRAY_DELTA_MAX, _GRAY_R_MIN, _GRAY_R_MAX = 30, 95, 190
_BROWN_R_MIN, _BROWN_R_MAX, _BROWN_G_MIN, _BROWN_G_MAX, _BROWN_B_MAX = 100, 200, 60, 150, 100
_BROWN_BORDER_ROWS = 2
_SLOTS_PER_KIND = 6
_SLOT_COUNT = 12
_MAX_SLOT_LEVELS = (8, 5)

CellKind = Literal["lit", "unlit", "none", "ambiguous"]


class SlotLevelResult(NamedTuple):
    """十二枠の段階マークの読取結果。

    空枠と不明枠を分け、下端を確認できた二セル以上の格子だけをパネルの証拠に数えます。
    表示セル総数は、保存した identity と進化武器の種別を照合するために使います。
    """

    levels: tuple[int | None, ...]
    confidence: float
    evidence_slots: int
    empty_slots: tuple[int, ...]
    panel_bottom_y: int | None
    reasons: tuple[str, ...]
    cell_counts: tuple[int, ...]


def _pixel(value: float, extent: int) -> int:
    """正規化座標を最後に一度だけ丸める。

    slot 間隔などの小数を先に丸めず、各セルの実位置を求めます。
    """
    # 半画素の計測値に生じる浮動小数の誤差だけを落とし、整数への丸めを揃える
    return round(round(value * extent, 9))


def _gold_pixels(block_bgra: NDArray[np.uint8]) -> NDArray[np.bool_]:
    """金色の画素を三色の閾値で判別する。

    セルと下端線に同じ閾値を使い、色判定の違いを防ぎます。
    """
    b, g, r = (block_bgra[..., i] for i in range(3))
    return (r > _GOLD_R_MIN) & (g > _GOLD_G_MIN) & (b < _GOLD_B_MAX)


def classify_pixels(block_bgra: NDArray[np.uint8]) -> CellKind:
    """セル内部の六割を占める色から状態を決める。

    点灯セルは二色の宝石模様なので、平均色ではなく各色の画素数を使います。
    """
    if block_bgra.size == 0:
        return "ambiguous"
    b, g, r = (block_bgra[..., i].astype(np.int16) for i in range(3))
    masks = (
        ("lit", _gold_pixels(block_bgra)),
        ("unlit", (r < _DARK_R_MAX) & (g < _DARK_G_MAX) & (b < _DARK_B_MAX)),
        ("none", (np.abs(r - g) < _GRAY_DELTA_MAX) & (np.abs(g - b) < _GRAY_DELTA_MAX)
         & (r > _GRAY_R_MIN) & (r < _GRAY_R_MAX)),
    )
    for kind, mask in masks:
        if float(np.mean(mask)) >= _CELL_RATIO:
            return kind
    return "ambiguous"


def panel_bottom(frame_bgra: NDArray[np.uint8], grid: SlotLevelPipGrid) -> int | None:
    """全幅の九割が金色で、直下二行が茶色の最初の行をパネル下端にする。

    縁は金色一画素と茶色二画素です。茶色を確かめ、宝箱の光を下端と誤認しません。
    """
    height, width = frame_bgra.shape[:2]
    x0, x1 = (_pixel(v, width) for v in grid.row_span_x)
    y0, y1 = (_pixel(v, height) for v in grid.bottom_scan_y)
    block = frame_bgra[y0:y1 + 1 + _BROWN_BORDER_ROWS, x0:x1 + 1]
    gold = _gold_pixels(block[:y1 - y0 + 1])
    if gold.size == 0:
        return None
    rows = np.flatnonzero(np.mean(gold, axis=1) >= grid.bottom_gold_ratio)
    b, g, r = (block[..., i] for i in range(3))
    brown = ((r > _BROWN_R_MIN) & (r < _BROWN_R_MAX) & (g > _BROWN_G_MIN)
             & (g < _BROWN_G_MAX) & (b < _BROWN_B_MAX) & (r > g) & (g > b))
    for row in rows:
        border = brown[row + 1:row + 1 + _BROWN_BORDER_ROWS]
        if len(border) == _BROWN_BORDER_ROWS and np.all(np.mean(border, axis=1) >= grid.bottom_gold_ratio):
            return y0 + int(row)
    return None


def parse_slot_levels(
    frame_bgra: NDArray[np.uint8], width: int, height: int,
    *, grid: SlotLevelPipGrid = SLOT_LEVEL_PIP_GRID,
) -> SlotLevelResult:
    """段階マークの連続 prefix から各枠の現在レベルを読む。

    点灯・消灯・セル無しの順番と種別ごとの上限を守る枠だけを採用します。
    行がパネル下端を越える場合は、その行をセル無しにします。
    """
    if frame_bgra.size == 0:
        return SlotLevelResult((None,) * _SLOT_COUNT, 0.0, 0, (), None,
                               ("empty_frame",), (0,) * _SLOT_COUNT)
    bottom = panel_bottom(frame_bgra, grid)
    levels: list[int | None] = []
    empty_slots: list[int] = []
    reasons: list[str] = []
    cell_counts: list[int] = []
    evidence_slots = valid_slots = 0
    dx, dy = (_pixel(v, size) for v, size in zip(grid.inner_offset, (width, height)))
    sw, sh = (_pixel(v, size) for v, size in zip(grid.inner_size, (width, height)))
    cell_h = _pixel(grid.cell_h, height)
    for slot in range(_SLOT_COUNT):
        kind_index = slot // _SLOTS_PER_KIND
        rows = grid.weapon_row_y0 if kind_index == 0 else grid.passive_row_y0
        cells: list[CellKind] = []
        for row in rows:
            y = _pixel(row, height)
            for col in range(grid.cols):
                x = _pixel(grid.cell_x0 + (slot % _SLOTS_PER_KIND) * grid.slot_dx
                           + col * grid.col_dx, width)
                cells.append("none" if bottom is not None and y + cell_h > bottom else
                             classify_pixels(frame_bgra[y + dy:y + dy + sh, x + dx:x + dx + sw]))
        lit = cells.count("lit")
        total = lit + cells.count("unlit")
        cell_counts.append(total)
        level = None
        if all(cell == "none" for cell in cells):
            empty_slots.append(slot)
            valid_slots += 1
        else:
            reason = None
            if "ambiguous" in cells:
                reason = "ambiguous"
            elif total > _MAX_SLOT_LEVELS[kind_index]:
                reason = "over_max"
            elif cells != ["lit"] * lit + ["unlit"] * (total - lit) + ["none"] * (len(cells) - total):
                reason = "non_prefix"
            elif lit == 0:
                reason = "no_lit"
            if reason is not None:
                reasons.append(f"slot{slot}:{reason}")
            else:
                level = lit
                valid_slots += 1
                evidence_slots += int(total >= 2 and bottom is not None)
        levels.append(level)
    return SlotLevelResult(tuple(levels), valid_slots / _SLOT_COUNT, evidence_slots,
                           tuple(empty_slots), bottom, tuple(reasons), tuple(cell_counts))


def has_panel_evidence(result: SlotLevelResult) -> bool:
    """下端を確認した二セル以上の valid slot をパネルの証拠にする。

    進化武器の一セルや、下端のない背景の偶然の prefix は証拠として数えません。
    """
    return result.evidence_slots >= 1
