"""icon matcher テスト。

合成 atlas を使ってマッチング契約・formal loader 拒否・
development_only フラグを検証します。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from survivors.vision.icon_matcher import (
    ATLAS_SCHEMA_VERSION,
    FEATURE_SIZE,
    AtlasManifest,
    FormalLoaderRejectedError,
    IconMatcher,
    MatchResult,
    TemplateEntry,
    _extract_feature,
    build_template_feature,
    serialize_manifest,
    _load_manifest,
)
from build_survivors_icon_atlas import build_development_atlas

_DUMMY_PROFILE_HASH = "a" * 64
_DUMMY_BUILD_HASH = "b" * 64


# ── AtlasManifest 契約テスト ─────────────────────────────────────────

class TestAtlasManifest:
    def _minimal_manifest(self, **kwargs) -> AtlasManifest:
        defaults = dict(
            schema_version=ATLAS_SCHEMA_VERSION,
            profile_hash=_DUMMY_PROFILE_HASH,
            build_hash=_DUMMY_BUILD_HASH,
            development_only=True,
            formal_parser_eligible=False,
            atlas_content_hash="c" * 64,
            entries=(),
        )
        defaults.update(kwargs)
        return AtlasManifest(**defaults)

    def test_valid_dev_manifest_ok(self):
        m = self._minimal_manifest()
        assert m.development_only is True
        assert m.formal_parser_eligible is False

    def test_wrong_schema_version_raises(self):
        with pytest.raises(ValueError, match="schema"):
            self._minimal_manifest(schema_version="icon_atlas.v99")

    def test_dev_only_cannot_be_formal_eligible(self):
        with pytest.raises(ValueError):
            self._minimal_manifest(development_only=True, formal_parser_eligible=True)

    def test_formal_eligible_non_dev_is_ok(self):
        m = self._minimal_manifest(development_only=False, formal_parser_eligible=True)
        assert m.formal_parser_eligible is True


# ── formal loader 拒否テスト ──────────────────────────────────────────

class TestFormalLoader:
    def test_formal_load_rejects_dev_atlas(self, development_atlas_path: Path):
        """development_only=true の atlas を formal load しようとすると拒否される。"""
        with pytest.raises(FormalLoaderRejectedError):
            IconMatcher.load_formal(development_atlas_path)

    def test_dev_load_accepts_dev_atlas(self, development_atlas_path: Path):
        """development atlas は load_development() では受け付ける。"""
        matcher = IconMatcher.load_development(development_atlas_path)
        assert matcher.manifest.development_only is True

    def test_formal_load_accepts_formal_atlas(self, tmp_path: Path):
        """formal_parser_eligible=true / development_only=false は load_formal() で受け付ける。"""
        feat = np.zeros(FEATURE_SIZE, dtype=np.float32)
        entry = TemplateEntry("whip", "weapon", 1, 8, feat)
        manifest = AtlasManifest(
            schema_version=ATLAS_SCHEMA_VERSION,
            profile_hash=_DUMMY_PROFILE_HASH,
            build_hash=_DUMMY_BUILD_HASH,
            development_only=False,
            formal_parser_eligible=True,
            atlas_content_hash="d" * 64,
            entries=(entry,),
        )
        path = tmp_path / "formal_atlas.json"
        path.write_bytes(serialize_manifest(manifest))
        matcher = IconMatcher.load_formal(path)
        assert matcher.manifest.formal_parser_eligible is True


# ── feature extraction ────────────────────────────────────────────────

class TestFeatureExtraction:
    def test_feature_shape(self):
        img = np.zeros((64, 64, 4), dtype=np.uint8)
        feat = _extract_feature(img)
        expected_len = FEATURE_SIZE
        assert feat.shape == (expected_len,), f"expected ({expected_len},), got {feat.shape}"
        assert feat.dtype == np.float32

    def test_empty_returns_zeros(self):
        empty = np.zeros((0, 0, 4), dtype=np.uint8)
        feat = _extract_feature(empty)
        assert (feat == 0).all()

    def test_different_colors_produce_different_features(self):
        """異なる色の画像は異なる特徴ベクトルになる。"""
        red_img = np.zeros((64, 64, 4), dtype=np.uint8)
        red_img[..., 2] = 200  # R チャンネル
        red_img[..., 3] = 255

        blue_img = np.zeros((64, 64, 4), dtype=np.uint8)
        blue_img[..., 0] = 200  # B チャンネル
        blue_img[..., 3] = 255

        f_red = _extract_feature(red_img)
        f_blue = _extract_feature(blue_img)
        assert not np.allclose(f_red, f_blue), "red and blue produce identical features"


# ── IconMatcher マッチング ────────────────────────────────────────────

class TestIconMatcher:
    def test_empty_atlas_returns_unknown(self, dev_atlas_manifest):
        """エントリが 0 の atlas では unknown を返す。"""
        empty_manifest = AtlasManifest(
            schema_version=ATLAS_SCHEMA_VERSION,
            profile_hash=_DUMMY_PROFILE_HASH,
            build_hash=_DUMMY_BUILD_HASH,
            development_only=True,
            formal_parser_eligible=False,
            atlas_content_hash="e" * 64,
            entries=(),
        )
        matcher = IconMatcher(empty_manifest)
        crop = np.zeros((64, 64, 4), dtype=np.uint8)
        result = matcher.match(crop)
        assert result.item_id is None
        assert result.kind == "unknown"

    def test_empty_crop_returns_unknown(self, dev_atlas_manifest):
        matcher = IconMatcher(dev_atlas_manifest)
        empty = np.zeros((0, 0, 4), dtype=np.uint8)
        result = matcher.match(empty)
        assert result.item_id is None

    def test_self_match_returns_item(self, dev_atlas_manifest):
        """atlas に登録されたアイテムのテンプレート画像はそのアイテムとマッチする。

        合成 atlas では完全一致するため、top-1/top-2 margin が十分に高いはず。
        ただし atlas に複数アイテムが存在するため、同色に近いアイテムには注意が必要。
        """
        matcher = IconMatcher(dev_atlas_manifest)

        # atlas から最初のエントリの特徴で逆合成テンプレートを作成
        entry = dev_atlas_manifest.entries[0]
        # entry.feature を持つような入力画像は存在しないが、
        # 同じ build_template_feature が返すような入力を作れる
        # ここでは atlas builder と同じ合成画像を使う
        from build_survivors_icon_atlas import _make_synth_template, _ITEM_COLORS_BGR
        template = _make_synth_template(
            entry.item_id,
            entry.level,
            _ITEM_COLORS_BGR.get(entry.item_id, (128, 128, 128)),
        )
        result = matcher.match(template)
        # 同じ画像から生成した特徴なので item_id が一致するはず
        assert result.item_id == entry.item_id, (
            f"expected {entry.item_id!r}, got {result.item_id!r} (conf={result.confidence:.3f})"
        )

    def test_match_result_fields(self, dev_atlas_manifest):
        """MatchResult のフィールドが揃っている。"""
        matcher = IconMatcher(dev_atlas_manifest)
        crop = np.zeros((64, 64, 4), dtype=np.uint8)
        result = matcher.match(crop)
        assert hasattr(result, "item_id")
        assert hasattr(result, "kind")
        assert hasattr(result, "level")
        assert hasattr(result, "confidence")
        assert hasattr(result, "reason")
        assert 0.0 <= result.confidence <= 1.0

    def test_low_margin_returns_unknown(self, tmp_path: Path):
        """top-1/top-2 margin が小さいときは unknown を返す。"""
        # 全く同じ色の 2 テンプレートを作成 → margin ≒ 0
        feat = np.ones(FEATURE_SIZE, dtype=np.float32) * 0.5
        entries = (
            TemplateEntry("whip", "weapon", 1, 8, feat.copy()),
            TemplateEntry("gold", "fallback", 1, 1, feat.copy()),
        )
        manifest = AtlasManifest(
            schema_version=ATLAS_SCHEMA_VERSION,
            profile_hash=_DUMMY_PROFILE_HASH,
            build_hash=_DUMMY_BUILD_HASH,
            development_only=True,
            formal_parser_eligible=False,
            atlas_content_hash="f" * 64,
            entries=entries,
        )
        matcher = IconMatcher(manifest)
        # 全ゼロ画像 → 全テンプレートと同距離
        result = matcher.match(np.zeros((64, 64, 4), dtype=np.uint8))
        # margin が低いので unknown になるはず
        assert result.item_id is None or result.confidence < 0.5

    def test_confidence_in_range(self, dev_atlas_manifest):
        matcher = IconMatcher(dev_atlas_manifest)
        crop = np.random.randint(0, 255, (64, 64, 4), dtype=np.uint8)
        result = matcher.match(crop)
        assert 0.0 <= result.confidence <= 1.0


# ── atlas serialization round-trip ─────────────────────────────────

class TestAtlasSerialization:
    def test_serialize_deserialize(self, tmp_path: Path):
        """atlas を JSON に保存して再ロードできる。"""
        feat = np.arange(FEATURE_SIZE, dtype=np.float32)
        entry = TemplateEntry("whip", "weapon", 1, 8, feat)
        manifest = AtlasManifest(
            schema_version=ATLAS_SCHEMA_VERSION,
            profile_hash=_DUMMY_PROFILE_HASH,
            build_hash=_DUMMY_BUILD_HASH,
            development_only=True,
            formal_parser_eligible=False,
            atlas_content_hash="g" * 64,
            entries=(entry,),
        )
        path = tmp_path / "test_atlas.json"
        path.write_bytes(serialize_manifest(manifest))

        loaded = _load_manifest(path)
        assert loaded.schema_version == manifest.schema_version
        assert len(loaded.entries) == 1
        assert loaded.entries[0].item_id == "whip"
        assert np.allclose(loaded.entries[0].feature, feat)

    def test_development_atlas_flags(self, development_atlas_path: Path):
        """開発用 atlas は development_only=true / formal_parser_eligible=false。"""
        manifest = _load_manifest(development_atlas_path)
        assert manifest.development_only is True
        assert manifest.formal_parser_eligible is False

    def test_development_atlas_has_vocabulary_entries(self, development_atlas_path: Path):
        """開発用 atlas が vocabulary の全アイテムを含む。"""
        from survivors.target_profile import load_target_profile
        profile = load_target_profile()
        vocab = set(profile.sections["choice_taxonomy"]["candidate_vocabulary"])
        manifest = _load_manifest(development_atlas_path)
        item_ids = {e.item_id for e in manifest.entries}
        assert vocab.issubset(item_ids), f"missing items: {vocab - item_ids}"
