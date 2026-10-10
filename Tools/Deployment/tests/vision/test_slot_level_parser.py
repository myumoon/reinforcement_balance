"""段階マーク格子の分類と prefix 規則の回帰テスト。

実測座標に二色の宝石と一画素の下端線を描き、読めないセルを確定しないことを確認します。
"""

from pathlib import Path

import numpy as np
import pytest
import yaml


def panel_frame(*, bottom=241):
    """実測位置に灰色の空パネルを作る。

下端は金色一画素と茶色二画素で描き、存在しない行の除外を再現します。
    """
    frame = np.full((1080, 1920, 4), 140, dtype=np.uint8)
    frame[..., 3] = 255
    if bottom is not None:
        frame[bottom, 120:401, :3] = (100, 180, 240)
        frame[bottom + 1:bottom + 3, 120:401, :3] = (47, 97, 125)
    return frame


def paint_slot(frame, slot, cells):
    """測定したセル内部に指定した分類色を描く。

点灯セルは金色十五画素と暗色十画素にし、一色の塗りつぶしに頼らず分類を試します。
    """
    rows = (110, 124, 137) if slot < 6 else (213, 226)
    for index, kind in enumerate(cells):
        x = round(124 + 45.75 * (slot % 6) + 13.5 * (index % 3))
        y = rows[index // 3]
        block = frame[y + 3:y + 8, x + 3:x + 8, :3]
        block[:] = {"lit": (20, 20, 30), "unlit": (20, 20, 30),
                    "none": (140, 140, 140), "ambiguous": (200, 100, 20)}[kind]
        if kind == "lit":
            block[:3] = (100, 180, 240)


def read_panel(frame):
    """本物のパーサを合成パネルへ適用する。

新規モジュールの未実装も各テストの失敗として記録できるよう、呼出時に読み込みます。
    """
    from survivors.vision.slot_level_parser import parse_slot_levels
    return parse_slot_levels(frame, 1920, 1080)


@pytest.mark.parametrize("kind", ["lit", "unlit", "none", "ambiguous"])
def test_pixel_classes(kind):
    """内部画素の割合から四種類を区別する。

金色が六割の二色模様も点灯として読めます。
    """
    from survivors.vision.slot_level_parser import classify_pixels
    frame = panel_frame()
    paint_slot(frame, 0, [kind])
    assert classify_pixels(frame[113:118, 127:132]) == kind


def test_pixel_ratio_below_sixty_percent_is_ambiguous():
    """六割に届かない色は確定しない。

金色十四画素と暗色十一画素では、どちらにも分類できません。
    """
    from survivors.vision.slot_level_parser import classify_pixels
    block = np.full((5, 5, 4), (20, 20, 30, 255), dtype=np.uint8)
    block.reshape(-1, 4)[:14] = (100, 180, 240, 255)
    assert classify_pixels(block) == "ambiguous"
    assert classify_pixels(block[:0]) == "ambiguous"


def test_levels_empty_and_evolved_evidence():
    """武器・パッシブ・進化武器の最大セル数を区別する。

八段階、五段階、二段階、一段階を読み、進化武器だけではパネルの証拠にしません。
    """
    from survivors.vision.slot_level_parser import has_panel_evidence
    frame = panel_frame()
    paint_slot(frame, 0, ["lit"] * 3 + ["unlit"] * 5 + ["none"])
    paint_slot(frame, 1, ["lit"] + ["none"] * 8)
    paint_slot(frame, 6, ["lit"] * 5 + ["none"])
    paint_slot(frame, 7, ["lit"] * 2 + ["none"] * 4)
    result = read_panel(frame)
    assert result.levels == (3, 1, None, None, None, None, 5, 2, None, None, None, None)
    assert result.empty_slots == (2, 3, 4, 5, 8, 9, 10, 11)
    assert result.cell_counts == (8, 1, 0, 0, 0, 0, 5, 2, 0, 0, 0, 0)
    assert result.evidence_slots == 3
    assert result.confidence == 1.0
    assert has_panel_evidence(result)
    evolved = panel_frame()
    paint_slot(evolved, 0, ["lit"] + ["none"] * 8)
    assert read_panel(evolved).levels[0] == 1
    assert not has_panel_evidence(read_panel(evolved))


@pytest.mark.parametrize("cells,reason", [
    (["unlit", "lit"] + ["none"] * 7, "non_prefix"),
    (["lit", "none", "unlit"] + ["none"] * 6, "non_prefix"),
    (["unlit"] * 8 + ["none"], "no_lit"),
    (["lit", "ambiguous"] + ["none"] * 7, "ambiguous"),
    (["lit"] * 9, "over_max"),
])
def test_invalid_slots_fail_closed(cells, reason):
    """順序崩れ・不明・上限超過を採用しない。

不正な枠を空枠ともみなさず、位置と原因を結果に残します。
    """
    frame = panel_frame()
    paint_slot(frame, 3, cells)
    result = read_panel(frame)
    assert result.levels[3] is None
    assert 3 not in result.empty_slots
    assert result.evidence_slots == 0
    assert result.confidence == pytest.approx(11 / 12)
    assert f"slot3:{reason}" in result.reasons


def test_all_lit_chest_is_not_panel_evidence():
    """全面黄色の宝箱画面を格子の証拠にしない。

全セルが点灯すると両種別の上限を超えるため、十二枠とも不明になります。
    """
    from survivors.vision.slot_level_parser import has_panel_evidence
    frame = panel_frame(bottom=274)
    for slot in range(12):
        paint_slot(frame, slot, ["lit"] * (9 if slot < 6 else 6))
    result = read_panel(frame)
    assert result.levels == (None,) * 12
    assert result.empty_slots == ()
    assert result.confidence == 0.0
    assert not has_panel_evidence(result)
    assert result.reasons == tuple(f"slot{i}:over_max" for i in range(12))


def test_gold_flash_without_brown_border_is_not_panel_bottom():
    """宝箱の金色帯で格子を切り詰めない。

    実フレーム12201のような二百三十行目の金色帯には茶色二行がなく、
    全六セルのパッシブは上限超過のままで証拠になりません。
    """
    from survivors.vision.slot_level_parser import has_panel_evidence
    frame = panel_frame(bottom=None)
    for slot in range(12):
        paint_slot(frame, slot, ["lit"] * (9 if slot < 6 else 6))
    frame[230, 120:401, :3] = (100, 180, 240)
    result = read_panel(frame)
    assert result.panel_bottom_y is None
    assert result.levels == (None,) * 12
    assert not has_panel_evidence(result)
    frame[..., :3] = (100, 180, 240)
    assert not has_panel_evidence(read_panel(frame))


def test_shrunken_panel_excludes_second_passive_row():
    """縮んだパネルの外側の画素をセルとして数えない。

二百二十八行目の線より下を黄色にしても、パッシブは三セルの一段だけになります。
    """
    frame = panel_frame(bottom=228)
    paint_slot(frame, 6, ["lit"] + ["unlit"] * 2 + ["lit"] * 3)
    result = read_panel(frame)
    assert result.panel_bottom_y == 228
    assert result.levels[6] == 1
    assert result.cell_counts[6] == 3


def test_no_passive_panel_and_missing_bottom():
    """パッシブ行がない場合と下端未検出を扱う。

短いパネルでは六枠を空枠にし、線がない場合は全行を解析します。
    """
    frame = panel_frame(bottom=170)
    for slot in range(6, 12):
        paint_slot(frame, slot, ["lit"] * 6)
    assert set(range(6, 12)) <= set(read_panel(frame).empty_slots)
    full = panel_frame(bottom=None)
    paint_slot(full, 0, ["lit"] + ["unlit"] * 7 + ["none"])
    assert read_panel(full).panel_bottom_y is None
    assert read_panel(full).levels[0] == 1


def test_prefix_without_panel_border_is_not_screen_evidence():
    """下端が無い偶然の prefix を画面判定に使わない。

    実フレーム22334のように背景の一枠だけが成立しても、パネルの縁を確認できなければ証拠にしません。
    """
    from survivors.vision.slot_level_parser import has_panel_evidence
    frame = panel_frame(bottom=None)
    paint_slot(frame, 10, ["lit"] * 4 + ["none"] * 2)
    result = read_panel(frame)
    assert result.levels[10] == 4
    assert result.panel_bottom_y is None
    assert result.evidence_slots == 0
    assert not has_panel_evidence(result)


def test_inventory_rois_and_empty_identity_agree():
    """左上在庫座標と空枠の三者契約を確認する。

測定した四十二画素の枠を使い、語彙 YAML と開発用 atlas の空枠 identity を照合します。
    """
    from survivors.vision.roi_layout import EMPTY_SLOT_ID, INV_SLOT_ROIS, norm_to_pixels
    from survivors.vision.icon_matcher import AtlasManifest, TemplateEntry, build_template_feature
    config = Path(__file__).resolve().parents[2] / "configs/hud_identity_vocabulary_v1.yaml"
    empty = yaml.safe_load(config.read_text(encoding="utf-8"))["empty_slot"]
    atlas = AtlasManifest("icon_atlas.v1", "a" * 64, "b" * 64, True, False, "c" * 64,
                          (TemplateEntry("empty_slot", "unknown", 1, 1,
                                         build_template_feature(panel_frame()[:42, :42])),))
    assert EMPTY_SLOT_ID == empty == atlas.entries[0].item_id
    for slot, roi in enumerate(INV_SLOT_ROIS):
        box = norm_to_pixels(roi, 1920, 1080)
        x, y = 100 + 46 * (slot % 6), 40 if slot < 6 else 86
        assert (box.x0, box.y0, box.x1, box.y1) == (x, y, x + 42, y + 42)
