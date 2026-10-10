"""1920×1080 日本語 UI の枠・カード行を画素から判定する。

画面状態と choice の両方が同じ実測位置・色の条件を使います。
他の解像度を拒否する責任は、各 parser の入口にあります。
"""

from __future__ import annotations

import numpy as np

from . import roi_layout as layout


def classify_gold(rgb):
    """枠や文字の金色を抽出する。

    赤と緑が明るく青が少ない画素を拾います。
    """
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return (r > 190) & (g > 140) & (b < 130)


def classify_yellow(rgb):
    """宝箱の黄色い光を抽出する。

    金枠より緑が強い広い発光領域を数えます。
    """
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return (r > 190) & (g > 180) & (b < 180)


def classify_blue(rgb):
    """ボタン内側の青色を抽出する。

    白い文字を除き、青成分だけが強い背景を拾います。
    """
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return (b > 180) & (r < 120) & (g < 140)


def classify_red(rgb):
    """死亡画面の赤い終了ボタンを抽出する。

    背景全体の暗赤ではなく、赤成分の強いボタン面を拾います。
    """
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return (r > 180) & (g < 80) & (b < 80)


def classify_card_grey(rgb):
    """カードの中間灰色の面を抽出する。

    符号付きの差で三色の近さを調べ、黄色い光を過渡カードと区別します。
    """
    rgb = rgb.astype(np.int16)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return (np.abs(r - g) < 6) & (np.abs(g - b) < 6) & (r > 120) & (r < 150)


def classify_border(rgb):
    """金色または白飛びしたウィンドウ枠を抽出する。

    宝箱の発光で上枠が白くなってもパネルの輪郭を認めます。
    """
    return classify_gold(rgb) | np.all(rgb > 230, axis=-1)


def classify_dim(rgb):
    """一時停止で半暗になった HUD 枠を抽出する。

    通常の金色枠とは明るさを分け、再開ボタンと組み合わせて使います。
    """
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return (r > 100) & (r < 150) & (g > 80) & (g < 120) & (b > 30) & (b < 70)


def classify_warm(rgb):
    """通常と死亡時の上端 HUD 枠を抽出する。

    赤く染まった金枠も含めるため、金色より緑の下限を低くします。
    """
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return (r > 190) & (g > 90) & (b < 130)


def pixel_fraction(crop_bgra, classifier) -> float:
    """切り出しに含まれる指定色の割合を返す。

    BGRA を RGB の並びへ直し、空領域では証拠なしの零を返します。
    """
    return float(np.mean(classifier(crop_bgra[..., 2::-1]))) if crop_bgra.size else 0.0


def horizontal_border_present(frame, band, x_span, *, classifier=classify_border) -> bool:
    """帯の中に八割以上が枠色の横一行があるか調べる。

    太さや上下位置が数画素揺れても、最も連続した一行を使います。
    """
    crop = frame[band[0]:band[1], x_span[0]:x_span[1], 2::-1]
    return bool(crop.size and np.max(np.mean(classifier(crop), axis=1)) >= .80)


def _hud_rows_present(frame, classifier) -> bool:
    """上端バーを挟む二行がどちらも指定色か調べる。

    バーの中身や残量に依存せず、横に連続する枠だけを証拠にします。
    """
    x0, x1 = layout.HUD_BAR_X
    return all(pixel_fraction(frame[y:y + 1, x0:x1], classifier) >= .80 for y in layout.HUD_BAR_ROWS)


def hud_present(frame) -> bool:
    """通常または死亡時の HUD 枠二本を認める。

    HP や XP の領域に単に明るい色があるだけでは HUD としません。
    """
    return _hud_rows_present(frame, classify_warm)


def pause_menu_present(frame) -> bool:
    """半暗の HUD 枠と青い再開ボタンを認める。

    操作者が一時停止した画面では移動やカード選択を止めます。
    """
    return (_hud_rows_present(frame, classify_dim)
            and pixel_fraction(layout.PAUSE_RESUME_INNER_ROI.crop(frame), classify_blue) >= .50)


def window_present(frame) -> tuple[bool, bool, bool]:
    """中央パネルの上枠・下枠・右枠の存在を返す。

    右枠が立つ前の滑り込みを、安定したカード配置と区別します。
    """
    top = horizontal_border_present(frame, layout.WINDOW_TOP_BAND, layout.WINDOW_BORDER_X)
    bottom = horizontal_border_present(frame, layout.WINDOW_BOTTOM_BAND, layout.WINDOW_BORDER_X)
    x0, x1 = layout.WINDOW_RIGHT_BAND
    y0, y1 = layout.WINDOW_RIGHT_Y
    right = bool(np.max(np.mean(classify_border(frame[y0:y1, x0:x1, 2::-1]), axis=0)) >= .80)
    return top, bottom, right


def _detect_card_rows(frame) -> list[float]:
    """四つのカード上枠帯に含まれる金色の割合を返す。

    各上端の次の三行を読み、カード面や説明文字は数えません。
    """
    x0, x1 = layout.CARD_ROW_BAND_X
    return [pixel_fraction(frame[top + 1:top + 4, x0:x1], classify_gold) for top in layout.CARD_TOP_Y]


def _card_count(rows) -> int:
    """先頭から連続して六割以上が金色の行数を数える。

    間に空の行があればそこで止め、離れた金色の物体をカード枚数に加えません。
    """
    count = 0
    for fraction in rows:
        if fraction < .60:
            break
        count += 1
    return count


def yellow_fraction(frame) -> float:
    """画面全体に占める宝箱の黄色い光の割合を返す。

    発光の強さを、パネルやカードの色より先に判定するために使います。
    """
    return pixel_fraction(frame, classify_yellow)


def card_grey_fraction(frame) -> float:
    """カードが滑り込む領域の灰色面の割合を返す。

    金色だけが乗る宝箱の減衰を、カード過渡から除外します。
    """
    return pixel_fraction(layout.CARD_ZONE_ROI.crop(frame), classify_card_grey)


def result_panel_present(frame) -> bool:
    """結果パネルの上下金枠と紫がかった背景を認める。

    HUD の有無とは独立に、大きな結果ウィンドウの構造を調べます。
    """
    if not (horizontal_border_present(frame, layout.RESULT_TOP_BAND, layout.RESULT_BORDER_X, classifier=classify_gold)
            and horizontal_border_present(frame, layout.RESULT_BOTTOM_BAND, layout.RESULT_BORDER_X, classifier=classify_gold)):
        return False
    rgb = layout.RESULT_BG_ROI.crop(frame)[..., 2::-1].astype(np.int16)
    background = (np.abs(rgb[..., 0] - 75) < 10) & (np.abs(rgb[..., 1] - 79) < 10) & (np.abs(rgb[..., 2] - 116) < 12)
    return float(np.mean(background)) >= .50


def game_over_present(frame) -> bool:
    """GAME OVER の金文字と赤い終了ボタンを認める。

    暗さや画面全体の赤みだけでは死亡を決めません。
    """
    return (pixel_fraction(layout.DEATH_TEXT_ROI.crop(frame), classify_gold) >= .06
            and pixel_fraction(frame[715:780, 830:1090], classify_red) >= .50)
