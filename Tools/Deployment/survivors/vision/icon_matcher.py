"""アイコンテンプレートマッチング: crop normalization + color/edge 特徴距離。

atlas に登録された各アイテムのテンプレート画像と比較し、
top-1/top-2 距離マージンを confidence として返します。
低マージン・レイアウト不正・エフェクト遮蔽は unknown とします。
formal_parser_eligible=false の development atlas を正式ロードしようとすると拒否します。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from reinbalance_survivors_contracts.canonical_json import canonical_hash

# 正規化サイズ (H × W)
_NORM_H: Final[int] = 48
_NORM_W: Final[int] = 48

# 特徴: 色ヒストグラム (8 bins × 3 ch) + エッジマップ (8×8 bins)
_COLOR_BINS: Final[int] = 8
_EDGE_BINS: Final[int] = 8
FEATURE_SIZE: Final[int] = _COLOR_BINS * 3 + _EDGE_BINS * _EDGE_BINS  # = 88

# confidence しきい値
_LOW_MARGIN: Final[float] = 0.15   # top-1/top-2 margin がこれ未満 → unknown
_LOW_CONF: Final[float] = 0.30

# atlas schema version
ATLAS_SCHEMA_VERSION: Final[str] = "icon_atlas.v1"


class FormalLoaderRejectedError(ValueError):
    """development_only=true または formal_parser_eligible=false の atlas を正式ロードした際に送出。"""


@dataclass(frozen=True, slots=True)
class TemplateEntry:
    """atlas 内の 1 テンプレートエントリ。"""

    item_id: str           # vocabulary 語彙の ID ("whip", "gold", "chicken", …)
    kind: str              # "weapon", "passive", "evolved", "fallback", "unknown"
    level: int             # このテンプレートが表すアイテムレベル (1 〜 max_level)
    max_level: int         # アイテムの最大レベル
    feature: NDArray[np.float32]  # 正規化済み特徴ベクトル


@dataclass(frozen=True)
class AtlasManifest:
    """アイコン atlas のメタデータと全テンプレート。"""

    schema_version: str
    profile_hash: str          # 対応 target_profile の hash
    build_hash: str            # 対応 build hash
    development_only: bool     # 合成/開発用 atlas は必ず True
    formal_parser_eligible: bool  # 正式 atlas のみ True
    atlas_content_hash: str    # テンプレート全体の SHA-256
    entries: tuple[TemplateEntry, ...]

    def __post_init__(self) -> None:
        if self.schema_version != ATLAS_SCHEMA_VERSION:
            raise ValueError(f"unsupported atlas schema: {self.schema_version!r}")
        if not isinstance(self.development_only, bool) or not isinstance(self.formal_parser_eligible, bool):
            raise ValueError("development_only and formal_parser_eligible must be bool")
        if self.development_only and self.formal_parser_eligible:
            raise ValueError("development_only atlas cannot be formal_parser_eligible")


@dataclass(frozen=True, slots=True)
class MatchResult:
    """アイコンマッチング結果。"""

    item_id: str | None    # 最良一致アイテム ID; unknown なら None
    kind: str              # "weapon", "passive", "evolved", "fallback", "unknown"
    level: int | None      # 推定レベル; unknown なら None
    confidence: float      # 0.0..1.0
    reason: str            # 理由文字列


def _extract_color_hist(rgb: NDArray[np.uint8]) -> NDArray[np.float32]:
    """RGB 画像から色ヒストグラム特徴を抽出する (各チャンネル 8 ビン)。"""
    hist = np.zeros(_COLOR_BINS * 3, dtype=np.float32)
    for ch in range(3):
        for b in range(_COLOR_BINS):
            lo = b * 256 // _COLOR_BINS
            hi = (b + 1) * 256 // _COLOR_BINS
            hist[ch * _COLOR_BINS + b] = float(np.sum((rgb[..., ch] >= lo) & (rgb[..., ch] < hi)))
    total = rgb.shape[0] * rgb.shape[1]
    if total > 0:
        hist /= total
    return hist


def _extract_edge_map(gray: NDArray[np.uint8]) -> NDArray[np.float32]:
    """グレースケール画像からエッジ強度マップを抽出し 8×8 ブロック平均を返す。"""
    gray_f = gray.astype(np.float32)
    # 簡易 Sobel
    dx = np.abs(np.diff(gray_f, axis=1, append=gray_f[:, -1:]))
    dy = np.abs(np.diff(gray_f, axis=0, append=gray_f[-1:, :]))
    edge = dx + dy
    # 8×8 グリッドに分割して各ブロックの平均を返す
    h, w = edge.shape
    bh, bw = h // _EDGE_BINS, w // _EDGE_BINS
    result = np.zeros(_EDGE_BINS * _EDGE_BINS, dtype=np.float32)
    for i in range(_EDGE_BINS):
        for j in range(_EDGE_BINS):
            block = edge[i * bh:(i + 1) * bh, j * bw:(j + 1) * bw]
            result[i * _EDGE_BINS + j] = float(block.mean()) if block.size > 0 else 0.0
    # 正規化
    norm = float(np.linalg.norm(result))
    if norm > 0:
        result /= norm
    return result


def _extract_feature(crop_bgra: NDArray[np.uint8]) -> NDArray[np.float32]:
    """BGRA クロップから特徴ベクトル (色ヒストグラム + エッジマップ) を抽出する。"""
    if crop_bgra.size == 0:
        feat_len = _COLOR_BINS * 3 + _EDGE_BINS * _EDGE_BINS
        return np.zeros(feat_len, dtype=np.float32)

    # nearest-neighbor リサイズ
    h, w = crop_bgra.shape[:2]
    row_idx = (np.arange(_NORM_H) * h // _NORM_H).clip(0, h - 1)
    col_idx = (np.arange(_NORM_W) * w // _NORM_W).clip(0, w - 1)
    resized = crop_bgra[np.ix_(row_idx, col_idx)]

    rgb = resized[..., [2, 1, 0]]  # BGR → RGB
    gray = (
        resized[..., 0].astype(np.float32) * 0.114
        + resized[..., 1].astype(np.float32) * 0.587
        + resized[..., 2].astype(np.float32) * 0.299
    ).astype(np.uint8)

    color_feat = _extract_color_hist(rgb)
    edge_feat = _extract_edge_map(gray)
    return np.concatenate([color_feat, edge_feat])


def build_template_feature(crop_bgra: NDArray[np.uint8]) -> NDArray[np.float32]:
    """テンプレート登録用特徴ベクトルを生成して返す (atlas builder 用)。"""
    return _extract_feature(crop_bgra)


def _l2_distance(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    """2 特徴ベクトルの L2 距離を返す。"""
    return float(np.linalg.norm(a - b))


class IconMatcher:
    """アイコンマッチングエンジン。

    atlas からロードしたテンプレートに対して特徴距離マッチングを行います。
    """

    def __init__(self, manifest: AtlasManifest) -> None:
        if not isinstance(manifest, AtlasManifest):
            raise TypeError("manifest must be an AtlasManifest")
        self._manifest = manifest

    @classmethod
    def load_formal(cls, atlas_path: Path) -> "IconMatcher":
        """正式 atlas を JSON ファイルからロードする。

        development_only=true または formal_parser_eligible=false の atlas は拒否します。
        """
        manifest = _load_manifest(atlas_path)
        if manifest.development_only or not manifest.formal_parser_eligible:
            raise FormalLoaderRejectedError(
                "atlas is not formal_parser_eligible "
                f"(development_only={manifest.development_only}, "
                f"formal_parser_eligible={manifest.formal_parser_eligible})"
            )
        return cls(manifest)

    @classmethod
    def load_development(cls, atlas_path: Path) -> "IconMatcher":
        """開発用 atlas をロードする (formal_parser_eligible チェックを行わない)。"""
        manifest = _load_manifest(atlas_path)
        return cls(manifest)

    @property
    def manifest(self) -> AtlasManifest:
        return self._manifest

    def match(self, crop_bgra: NDArray[np.uint8]) -> MatchResult:
        """クロップ画像に対してアイコンマッチングを行い結果を返す。

        top-1/top-2 distance margin が _LOW_MARGIN 未満の場合は unknown を返します。
        """
        if crop_bgra.ndim != 3 or crop_bgra.shape[2] != 4:
            return MatchResult(None, "unknown", None, 0.0, "invalid_crop")
        if crop_bgra.size == 0:
            return MatchResult(None, "unknown", None, 0.0, "empty_crop")
        if not self._manifest.entries:
            return MatchResult(None, "unknown", None, 0.0, "empty_atlas")

        query_feat = _extract_feature(crop_bgra)

        distances: list[tuple[float, TemplateEntry]] = []
        for entry in self._manifest.entries:
            dist = _l2_distance(query_feat, entry.feature)
            distances.append((dist, entry))

        distances.sort(key=lambda x: x[0])
        best_dist, best_entry = distances[0]

        if len(distances) < 2:
            # atlas に 1 テンプレートしかない → margin = max_dist - best_dist
            margin = 1.0
            second_dist = best_dist + 1.0
        else:
            second_dist, _ = distances[1]
            margin = max(0.0, second_dist - best_dist)

        # confidence: margin を [0, LOW_MARGIN*2] → [0, 1] に正規化
        confidence = min(1.0, margin / max(_LOW_MARGIN * 2, 1e-6))

        if margin < _LOW_MARGIN:
            return MatchResult(
                None, "unknown", None, confidence,
                f"low_margin:{margin:.3f}<{_LOW_MARGIN}"
            )

        if confidence < _LOW_CONF:
            return MatchResult(
                None, "unknown", None, confidence,
                f"low_confidence:{confidence:.3f}"
            )

        return MatchResult(
            best_entry.item_id,
            best_entry.kind,
            best_entry.level,
            confidence,
            "ok",
        )


def _load_manifest(atlas_path: Path) -> AtlasManifest:
    """JSON atlas ファイルを AtlasManifest としてロードする。"""
    if not atlas_path.is_file():
        raise FileNotFoundError(f"atlas not found: {atlas_path}")

    raw = atlas_path.read_bytes()
    try:
        wire = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid atlas JSON") from exc

    required_keys = {
        "schema_version", "profile_hash", "build_hash",
        "development_only", "formal_parser_eligible",
        "atlas_content_hash", "entries",
    }
    if not isinstance(wire, dict) or set(wire) != required_keys:
        raise ValueError(f"atlas manifest fields do not match schema: {set(wire)}")

    entries: list[TemplateEntry] = []
    for e in wire["entries"]:
        feat = np.array(e["feature"], dtype=np.float32)
        entries.append(TemplateEntry(
            item_id=e["item_id"],
            kind=e["kind"],
            level=int(e["level"]),
            max_level=int(e["max_level"]),
            feature=feat,
        ))

    return AtlasManifest(
        schema_version=wire["schema_version"],
        profile_hash=wire["profile_hash"],
        build_hash=wire["build_hash"],
        development_only=bool(wire["development_only"]),
        formal_parser_eligible=bool(wire["formal_parser_eligible"]),
        atlas_content_hash=wire["atlas_content_hash"],
        entries=tuple(entries),
    )


def serialize_manifest(manifest: AtlasManifest) -> bytes:
    """AtlasManifest を JSON bytes にシリアライズする。"""
    wire: dict[str, Any] = {
        "schema_version": manifest.schema_version,
        "profile_hash": manifest.profile_hash,
        "build_hash": manifest.build_hash,
        "development_only": manifest.development_only,
        "formal_parser_eligible": manifest.formal_parser_eligible,
        "atlas_content_hash": manifest.atlas_content_hash,
        "entries": [
            {
                "item_id": e.item_id,
                "kind": e.kind,
                "level": e.level,
                "max_level": e.max_level,
                "feature": e.feature.tolist(),
            }
            for e in manifest.entries
        ],
    }
    return (json.dumps(wire, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
