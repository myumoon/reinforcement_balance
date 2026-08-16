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
    """全黒 BGRA フレームを返す。"""
    return np.zeros((FRAME_H, FRAME_W, 4), dtype=np.uint8)


def _make_gameplay_frame() -> np.ndarray:
    """gameplay 状態を模した合成フレームを返す。

    HP バー領域 (赤) と XP バー領域 (青) を塗りつぶして HUD を再現します。
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
    return frame


def _make_levelup_frame() -> np.ndarray:
    """level_up 状態を模した合成フレームを返す (中央が暗いオーバーレイ)。"""
    frame = _make_blank_frame()
    # 全画面を暗い背景にする (HUD なし)
    frame[..., :3] = 20
    frame[..., 3] = 255
    # カード領域に明るいパッチを配置 (3 枚カードレイアウト)
    for card_x0, card_x1 in [(106, 662), (720, 1200), (1258, 1814)]:
        frame[175:940, card_x0:card_x1, :3] = 80
        frame[175:940, card_x0:card_x1, 3] = 255
    return frame


@pytest.fixture
def blank_frame() -> np.ndarray:
    """全黒フレーム。"""
    return _make_blank_frame()


@pytest.fixture
def gameplay_frame() -> np.ndarray:
    """gameplay 合成フレーム。"""
    return _make_gameplay_frame()


@pytest.fixture
def levelup_frame() -> np.ndarray:
    """level_up 合成フレーム。"""
    return _make_levelup_frame()


@pytest.fixture
def development_atlas_path(tmp_path: Path) -> Path:
    """開発用合成 atlas を tmp_path に生成してパスを返す。"""
    atlas_path = tmp_path / "dev_atlas.json"
    build_development_atlas(
        atlas_path,
        profile_hash=_DUMMY_PROFILE_HASH,
        build_hash=_DUMMY_BUILD_HASH,
    )
    return atlas_path


@pytest.fixture
def dev_atlas_manifest(development_atlas_path: Path) -> AtlasManifest:
    """開発用 AtlasManifest を返す。"""
    from survivors.vision.icon_matcher import _load_manifest
    return _load_manifest(development_atlas_path)


@pytest.fixture
def dummy_parser_artifact_hash() -> str:
    return "c" * 64
