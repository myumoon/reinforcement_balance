"""開発用合成アイコン atlas ビルダー。

target profile に登録されたアイテム語彙から合成テンプレート画像を生成し、
JSON atlas manifest を出力します。出力は常に development_only=true / formal_parser_eligible=false です。
正式 atlas は 04-05 で実ゲーム画像から生成します。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

# sys.path 操作なしで Deployment パッケージを解決するため絶対インポートを使用
_THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(_THIS_DIR))

from survivors.vision.icon_matcher import (
    ATLAS_SCHEMA_VERSION,
    TemplateEntry,
    AtlasManifest,
    build_template_feature,
    serialize_manifest,
)
from survivors.target_profile import load_target_profile
from reinbalance_survivors_contracts.canonical_json import canonical_hash

# 合成テンプレートのサイズ (H × W × BGRA)
_SYNTH_H: Final[int] = 64
_SYNTH_W: Final[int] = 64

# アイテムごとの固定色マップ (BGR) – 実ゲーム色とは無関係な開発用値
_ITEM_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "whip":    (60, 120, 200),   # 青みがかり
    "gold":    (40, 180, 220),   # 黄色っぽい (B=40, G=180, R=220 → 黄)
    "chicken": (80,  80, 160),   # 赤みがかり
}

# デフォルト色 (未定義アイテム)
_DEFAULT_COLOR_BGR: tuple[int, int, int] = (128, 128, 128)

# アイテムの種別マップ
_ITEM_KIND: dict[str, str] = {
    "whip":    "weapon",
    "gold":    "fallback",
    "chicken": "fallback",
}

# アイテムの最大レベル (仮値; 実ゲームと異なる場合がある)
_ITEM_MAX_LEVEL: dict[str, int] = {
    "whip":    8,
    "gold":    1,
    "chicken": 1,
}


def _make_synth_template(
    item_id: str,
    level: int,
    color_bgr: tuple[int, int, int],
) -> NDArray[np.uint8]:
    """合成テンプレート画像 (BGRA, 64×64) を生成する。

    中央に塗りつぶした円を配置し、レベルに応じて若干色を変化させます。
    これは開発用の仮テンプレートであり、実ゲーム画像とは一致しません。
    """
    img = np.zeros((_SYNTH_H, _SYNTH_W, 4), dtype=np.uint8)

    # 背景色 (暗いグレー)
    img[..., :3] = 30
    img[..., 3] = 255

    # レベルで若干明度変化 (level/max_level の比率で輝度調整)
    max_level = _ITEM_MAX_LEVEL.get(item_id, 1)
    level_ratio = level / max(1, max_level)
    factor = 0.7 + 0.3 * level_ratio

    b = int(min(255, color_bgr[0] * factor))
    g = int(min(255, color_bgr[1] * factor))
    r = int(min(255, color_bgr[2] * factor))

    # 中央円を塗りつぶす
    cy, cx = _SYNTH_H // 2, _SYNTH_W // 2
    radius = _SYNTH_H // 3
    ys, xs = np.ogrid[:_SYNTH_H, :_SYNTH_W]
    mask = (ys - cy) ** 2 + (xs - cx) ** 2 <= radius ** 2
    img[mask, 0] = b
    img[mask, 1] = g
    img[mask, 2] = r
    img[mask, 3] = 255

    # レベル表示用の小さい正方形 (左上)
    sq = min(8, level)
    img[:sq, :sq, 0] = 255
    img[:sq, :sq, 1] = 255
    img[:sq, :sq, 2] = 255
    img[:sq, :sq, 3] = 255

    return img


def build_development_atlas(
    output_path: Path,
    *,
    profile_hash: str,
    build_hash: str,
) -> AtlasManifest:
    """開発用合成 atlas を生成して JSON ファイルに保存する。

    development_only=True、formal_parser_eligible=False で出力します。
    """
    profile = load_target_profile()
    vocabulary: list[str] = profile.sections["choice_taxonomy"]["candidate_vocabulary"]

    entries: list[TemplateEntry] = []

    for item_id in sorted(vocabulary):
        color_bgr = _ITEM_COLORS_BGR.get(item_id, _DEFAULT_COLOR_BGR)
        kind = _ITEM_KIND.get(item_id, "unknown")
        max_level = _ITEM_MAX_LEVEL.get(item_id, 1)

        for level in range(1, max_level + 1):
            template_bgra = _make_synth_template(item_id, level, color_bgr)
            feature = build_template_feature(template_bgra)
            entries.append(TemplateEntry(
                item_id=item_id,
                kind=kind,
                level=level,
                max_level=max_level,
                feature=feature,
            ))

    # atlas_content_hash: 全エントリの feature を結合した SHA-256
    all_features = np.concatenate([e.feature for e in entries]).tobytes()
    atlas_content_hash = hashlib.sha256(all_features).hexdigest()

    manifest = AtlasManifest(
        schema_version=ATLAS_SCHEMA_VERSION,
        profile_hash=profile_hash,
        build_hash=build_hash,
        development_only=True,       # ponytail: 合成 atlas は常に開発専用
        formal_parser_eligible=False,
        atlas_content_hash=atlas_content_hash,
        entries=tuple(entries),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(serialize_manifest(manifest))
    return manifest


def main(argv: list[str] | None = None) -> None:
    """CLI エントリーポイント。"""
    parser = argparse.ArgumentParser(
        description="開発用合成 Survivors アイコン atlas を生成する"
    )
    parser.add_argument(
        "output",
        type=Path,
        help="出力 JSON ファイルパス",
    )
    parser.add_argument(
        "--profile-hash",
        default="a" * 64,
        help="target profile hash (プレースホルダー; 実運用では正確な値を指定する)",
    )
    parser.add_argument(
        "--build-hash",
        default="b" * 64,
        help="build hash (プレースホルダー)",
    )
    args = parser.parse_args(argv)

    manifest = build_development_atlas(
        args.output,
        profile_hash=args.profile_hash,
        build_hash=args.build_hash,
    )
    print(f"development atlas written: {args.output}")
    print(f"  entries={len(manifest.entries)}")
    print(f"  development_only={manifest.development_only}")
    print(f"  formal_parser_eligible={manifest.formal_parser_eligible}")
    print(f"  atlas_content_hash={manifest.atlas_content_hash}")


if __name__ == "__main__":
    main()
