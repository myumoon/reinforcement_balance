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
    """開発専用 atlas の正式利用を拒否する例外。

    正式な解析器として使えない画像集を誤って本番へ読み込まないために送出します。
    """


@dataclass(frozen=True, slots=True)
class TemplateEntry:
    """atlas 内の一テンプレートエントリ。

    所持欄とカードの描画を surface で分け、旧 entry は inventory として扱います。
    """

    item_id: str           # vocabulary 語彙の ID ("whip", "gold", "chicken", …)
    kind: str              # "weapon", "passive", "evolved", "fallback", "unknown"
    level: int             # このテンプレートが表すアイテムレベル (1 〜 max_level)
    max_level: int         # アイテムの最大レベル
    feature: NDArray[np.float32]  # 正規化済み特徴ベクトル
    surface: str = "inventory"


@dataclass(frozen=True)
class AtlasManifest:
    """atlas のメタデータと全テンプレート。

    対象プロファイル、画像集の識別子、正式利用の可否と各 icon の特徴を一緒に保持します。
    """

    schema_version: str
    profile_hash: str          # 対応 target_profile の hash
    build_hash: str            # 対応 build hash
    development_only: bool     # 合成/開発用 atlas は必ず True
    formal_parser_eligible: bool  # 正式 atlas のみ True
    atlas_content_hash: str    # テンプレート全体の SHA-256
    entries: tuple[TemplateEntry, ...]

    def __post_init__(self) -> None:
        """atlas の schema と正式利用フラグを検証する。

        開発専用と正式利用可能が同時に設定された矛盾を、構築時に拒否します。
        """
        if self.schema_version != ATLAS_SCHEMA_VERSION:
            raise ValueError(f"unsupported atlas schema: {self.schema_version!r}")
        if not isinstance(self.development_only, bool) or not isinstance(self.formal_parser_eligible, bool):
            raise ValueError("development_only and formal_parser_eligible must be bool")
        if self.development_only and self.formal_parser_eligible:
            raise ValueError("development_only atlas cannot be formal_parser_eligible")


@dataclass(frozen=True, slots=True)
class MatchResult:
    """icon 照合の結果。

    読めた item ID とレベルに信頼度・理由を添え、不確かな名前は None のまま返します。
    """

    item_id: str | None    # 最良一致アイテム ID; unknown なら None
    kind: str              # "weapon", "passive", "evolved", "fallback", "unknown"
    level: int | None      # 推定レベル; unknown なら None
    confidence: float      # 0.0..1.0
    reason: str            # 理由文字列


def _extract_color_hist(rgb: NDArray[np.uint8]) -> NDArray[np.float32]:
    """RGB の各成分を八区間の histogram にする。

    色の多さを数えて正規化し、画像の大きさが違っても比較できる特徴にします。
    """
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
    """輪郭の強さを八行八列の平均にまとめる。

    明暗の変化から形の違いを拾い、色だけが似た icon を区別する材料にします。
    """
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
    """BGRA crop から色と輪郭の特徴を作る。

    画像を共通サイズへ揃え、色の分布と輪郭を一つの比較用ベクトルに結合します。
    """
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
    """登録用の template feature を生成する。

    照合時と同じ処理を使い、atlas に保存する画像特徴を作ります。
    """
    return _extract_feature(crop_bgra)


def _l2_distance(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    """二つの feature の L2 距離を求める。

    対応する成分の差をまとめ、小さいほど画像の特徴が近い値にします。
    """
    return float(np.linalg.norm(a - b))


class IconMatcher:
    """アイコンマッチングエンジン。

    atlas からロードしたテンプレートに対して特徴距離マッチングを行います。
    """

    def __init__(self, manifest: AtlasManifest) -> None:
        """検証済みの atlas を照合器へ保存する。

        画像集以外のオブジェクトは受け付けず、以後の照合はこのテンプレート群を使います。
        """
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
        """開発用 JSON atlas を読み込む。

        正式利用の可否で拒否せず、合成画像や scratch atlas をテストへ使えるようにします。
        """
        manifest = _load_manifest(atlas_path)
        return cls(manifest)

    @property
    def manifest(self) -> AtlasManifest:
        """照合器が使用する atlas を返す。

        呼び出し側が template の種別や出自を参照するための読み取り口です。
        """
        return self._manifest

    def match(self, crop_bgra: NDArray[np.uint8], *, surface: str = "inventory") -> MatchResult:
        """クロップ画像に対してアイコンマッチングを行い結果を返す。

        指定 surface だけで距離差を比較し、候補がない場合や差が小さい場合は unknown を返します。
        """
        if crop_bgra.ndim != 3 or crop_bgra.shape[2] != 4:
            return MatchResult(None, "unknown", None, 0.0, "invalid_crop")
        if crop_bgra.size == 0:
            return MatchResult(None, "unknown", None, 0.0, "empty_crop")
        entries = [entry for entry in self._manifest.entries if entry.surface == surface]
        if not entries:
            return MatchResult(None, "unknown", None, 0.0, "no_surface_entries")

        query_feat = _extract_feature(crop_bgra)

        distances: list[tuple[float, TemplateEntry]] = []
        for entry in entries:
            dist = _l2_distance(query_feat, entry.feature)
            distances.append((dist, entry))

        distances.sort(key=lambda x: x[0])
        best_dist, best_entry = distances[0]

        # shortcut: card は実測 feature の完全一致だけ採用し、描画差を許す時は実フレームで距離を較正する。
        if surface == "card" and best_dist != 0.0:
            return MatchResult(None, "unknown", None, 0.0, "card_template_mismatch")

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
    """JSON atlas をデータ型へ復元する。

    必須項目と surface を確認し、未知の描画面を黙って所持欄へ戻しません。
    """
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
        surface = e.get("surface", "inventory")
        if surface not in ("inventory", "card"):
            raise ValueError(f"unsupported atlas surface: {surface!r}")
        feat = np.array(e["feature"], dtype=np.float32)
        entries.append(TemplateEntry(
            item_id=e["item_id"],
            kind=e["kind"],
            level=int(e["level"]),
            max_level=int(e["max_level"]),
            feature=feat,
            surface=surface,
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
    """atlas を安定した JSON bytes にする。

    全 entry の surface を明記し、key 順序と末尾改行を揃えて保存します。
    """
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
                "surface": e.surface,
            }
            for e in manifest.entries
        ],
    }
    return (json.dumps(wire, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
