"""vision テスト共通フィクスチャ (synthetic frame / atlas 生成)。

実ゲーム映像を使わず、numpy で生成した合成フレームで全テストを動かします。
全フィクスチャは development_only=true として明示し、formal eligibility テストで拒否を確認します。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from survivors.vision.icon_matcher import (
    ATLAS_SCHEMA_VERSION,
    AtlasManifest,
    TemplateEntry,
    build_template_feature,
    serialize_manifest,
)
from build_survivors_icon_atlas import build_development_atlas

# 合成フレームサイズ
FRAME_H, FRAME_W = 1080, 1920

# ダミー SHA-256
_DUMMY_PROFILE_HASH = "a" * 64
_DUMMY_BUILD_HASH = "b" * 64


def _make_blank_frame() -> np.ndarray:
    """全黒 BGRA フレームを返す。

    UI の枠やバーが一つもない不明画面の土台にします。
    """
    return np.zeros((FRAME_H, FRAME_W, 4), dtype=np.uint8)


def _make_gameplay_frame() -> np.ndarray:
    """gameplay 状態を模した合成フレームを返す。

    バーの中身に加え、実画面の上端 XP バーを挟む金枠二本を描きます。
    """
    frame = _make_blank_frame()
    # HP バー: y=33..56, 赤 (B=0, G=0, R=200)
    hp_y0, hp_y1 = 33, 56
    frame[hp_y0:hp_y1, :960, 0] = 0
    frame[hp_y0:hp_y1, :960, 1] = 0
    frame[hp_y0:hp_y1, :960, 2] = 200
    frame[hp_y0:hp_y1, :, 3] = 255
    # XP バー: y=56..76, 青 (B=200, G=0, R=0)
    xp_y0, xp_y1 = 56, 76
    frame[xp_y0:xp_y1, :480, 0] = 200
    frame[xp_y0:xp_y1, :480, 1] = 0
    frame[xp_y0:xp_y1, :480, 2] = 0
    frame[xp_y0:xp_y1, :, 3] = 255
    frame[1:4, 96:1824, :3] = (102, 203, 255)
    frame[31:34, 96:1824, :3] = (102, 204, 255)
    return frame


def _make_levelup_frame(count: int = 3, icons: tuple = ()) -> np.ndarray:
    """実測位置に中央ウィンドウと縦積みカードを描く。

    一枚から四枚までを同じ間隔で置き、指定した色や画像を左のアイコンへ入れます。
    count が零の画像は、カードを持たない宝箱パネルにも使えます。
    """
    frame = _make_gameplay_frame()
    gold = (102, 203, 255)
    frame[111:965, 642:1278, :3] = gold
    frame[117:959, 648:1272, :3] = (116, 79, 75)
    for k, top in enumerate((267, 424, 581, 738)[:count]):
        frame[top:top + 154, 656:1265, :3] = gold
        frame[top + 6:top + 148, 662:1259, :3] = 134
        icon = icons[k] if k < len(icons) else (0, 0, 0)
        if isinstance(icon, np.ndarray):
            ys = np.linspace(0, icon.shape[0] - 1, 55).astype(int)
            xs = np.linspace(0, icon.shape[1] - 1, 51).astype(int)
            frame[top + 13:top + 68, 669:720] = icon[ys[:, None], xs[None, :]]
        else:
            frame[top + 13:top + 68, 669:720, :3] = icon
    return frame


@pytest.fixture
def blank_frame() -> np.ndarray:
    """全黒フレームを供給する。

    検出できない場面で値を推測しないテストに使います。
    """
    return _make_blank_frame()


@pytest.fixture
def gameplay_frame() -> np.ndarray:
    """通常プレイの合成フレームを供給する。

    HUD はあり、カードウィンドウはありません。
    """
    return _make_gameplay_frame()


@pytest.fixture
def levelup_frame() -> np.ndarray:
    """三枚の選択肢がある合成フレームを供給する。

    中央の金枠に、実画面と同じ縦並びの灰色カードを描きます。
    """
    return _make_levelup_frame()


@pytest.fixture
def development_atlas_path(tmp_path: Path) -> Path:
    """開発用合成 atlas を一時ディレクトリに生成する。

    本番用として使えない印を持つ JSON を fixture ごとに作ります。
    """
    atlas_path = tmp_path / "dev_atlas.json"
    build_development_atlas(
        atlas_path,
        profile_hash=_DUMMY_PROFILE_HASH,
        build_hash=_DUMMY_BUILD_HASH,
    )
    return atlas_path


@pytest.fixture
def dev_atlas_manifest(development_atlas_path: Path) -> AtlasManifest:
    """生成した開発 atlas の読込結果を返す。

    JSON の内容を照合器と同じ入口から読みます。
    """
    from survivors.vision.icon_matcher import _load_manifest
    return _load_manifest(development_atlas_path)


@pytest.fixture
def dummy_parser_artifact_hash() -> str:
    """テスト用の parser 識別子を返す。

    実ファイルのハッシュとは区別できる固定値です。
    """
    return "c" * 64
