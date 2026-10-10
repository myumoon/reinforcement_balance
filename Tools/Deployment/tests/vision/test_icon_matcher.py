"""icon matcher テスト。

合成 atlas を使ってマッチング契約・formal loader 拒否・
development_only フラグを検証します。
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import replace
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
    """atlas manifest の構築条件を検証する。

    schema の違いと開発・正式フラグの矛盾を区別して確認します。
    """
    def _minimal_manifest(self, **kwargs) -> AtlasManifest:
        """必要最小限の manifest を作る。

        個々の検証で変える値だけを上書きし、共通の既定条件を揃えます。
        """
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
        """正しい開発 manifest を構築できる。

        開発専用かつ正式利用不可の組み合わせを受け付けます。
        """
        m = self._minimal_manifest()
        assert m.development_only is True
        assert m.formal_parser_eligible is False

    def test_wrong_schema_version_raises(self):
        """未知の atlas schema を拒否する。

        別形式の画像集を同じデータ構造として読み違えないことを確認します。
        """
        with pytest.raises(ValueError, match="schema"):
            self._minimal_manifest(schema_version="icon_atlas.v99")

    def test_dev_only_cannot_be_formal_eligible(self):
        """開発専用の正式利用フラグを拒否する。

        合成 atlas が本番用として扱われる矛盾した設定を作れません。
        """
        with pytest.raises(ValueError):
            self._minimal_manifest(development_only=True, formal_parser_eligible=True)

    def test_formal_eligible_non_dev_is_ok(self):
        """正式用のフラグ組み合わせを受け付ける。

        開発専用でない画像集なら正式利用可能として構築できます。
        """
        m = self._minimal_manifest(development_only=False, formal_parser_eligible=True)
        assert m.formal_parser_eligible is True


# ── formal loader 拒否テスト ──────────────────────────────────────────

class TestFormalLoader:
    """正式 loader と開発 loader の境界を検証する。

    同じ JSON でも利用目的によって受け付け方が違うことを確認します。
    """
    def test_formal_load_rejects_dev_atlas(self, development_atlas_path: Path):
        """開発 atlas の正式ロードを拒否する。

        実画像の検証を経ていない合成 template を正式入口へ渡すと例外になります。
        """
        with pytest.raises(FormalLoaderRejectedError):
            IconMatcher.load_formal(development_atlas_path)

    def test_dev_load_accepts_dev_atlas(self, development_atlas_path: Path):
        """開発 loader は開発 atlas を受け付ける。

        正式利用不可の印を保持したまま合成 template のテストを行えます。
        """
        matcher = IconMatcher.load_development(development_atlas_path)
        assert matcher.manifest.development_only is True

    def test_formal_load_accepts_formal_atlas(self, tmp_path: Path):
        """正式 loader は正式利用可能な atlas を受け付ける。

        開発専用でない template を JSON から読み、照合器を作れることを確認します。
        """
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
    """icon の特徴抽出を検証する。

    固定次元、空画像、色の違いが比較用データへ正しく反映されるかを調べます。
    """
    def test_feature_shape(self):
        """feature の次元が契約値に一致する。

        画像の大きさによって保存するベクトル長が変わらないことを確認します。
        """
        img = np.zeros((64, 64, 4), dtype=np.uint8)
        feat = _extract_feature(img)
        expected_len = FEATURE_SIZE
        assert feat.shape == (expected_len,), f"expected ({expected_len},), got {feat.shape}"
        assert feat.dtype == np.float32

    def test_empty_returns_zeros(self):
        """空画像の feature は零ベクトルになる。

        画素のない crop から有効な icon の特徴を作らないことを確認します。
        """
        empty = np.zeros((0, 0, 4), dtype=np.uint8)
        feat = _extract_feature(empty)
        assert (feat == 0).all()

    def test_different_colors_produce_different_features(self):
        """異なる色は異なる feature になる。

        色の分布が変わると照合距離に反映されることを確認します。
        """
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
    """照合結果と不確かな入力の扱いを検証する。

    名前を決められない場面で unknown と理由を返し、信頼度を範囲内に保ちます。
    """
    def test_empty_atlas_returns_unknown(self, dev_atlas_manifest):
        """空 atlas では item ID を推測しない。

        登録 template がない照合でも安全に unknown を返します。
        """
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
        """空 crop を未知の icon として返す。

        画像なしの入力が template の名前へ誤って結び付かないことを確認します。
        """
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
        """MatchResult の値が揃っている。

        名前・種別・レベル・信頼度・理由を利用側が参照できることを確認します。
        """
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
        """距離差の小さい二候補では名前を確定しない。

        同じ feature を持つ別 item の template を与え、曖昧さを unknown として残します。
        """
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
        """照合 confidence が零から一に収まる。

        画像と template の距離によって上限や下限を超えないことを確認します。
        """
        matcher = IconMatcher(dev_atlas_manifest)
        crop = np.random.randint(0, 255, (64, 64, 4), dtype=np.uint8)
        result = matcher.match(crop)
        assert 0.0 <= result.confidence <= 1.0


# ── atlas serialization round-trip ─────────────────────────────────

class TestAtlasSerialization:
    """atlas の JSON 保存と読込を検証する。

    feature と利用可能性の情報を往復しても失わず、全語彙を登録します。
    """
    def test_serialize_deserialize(self, tmp_path: Path):
        """JSON 往復で atlas の値を保つ。

        item 名と feature 配列を保存し、同じ内容で再び読み込めます。
        """
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
        """合成 atlas の開発専用フラグを確認する。

        builder が正式利用可能と誤って宣言しないことを確認します。
        """
        manifest = _load_manifest(development_atlas_path)
        assert manifest.development_only is True
        assert manifest.formal_parser_eligible is False

    def test_development_atlas_has_vocabulary_entries(self, development_atlas_path: Path):
        """開発 atlas が全候補語彙を含む。

        対象プロファイルの item ID が一つも登録漏れにならないことを確認します。
        """
        from survivors.target_profile import load_target_profile
        profile = load_target_profile()
        vocab = set(profile.sections["choice_taxonomy"]["candidate_vocabulary"])
        manifest = _load_manifest(development_atlas_path)
        item_ids = {e.item_id for e in manifest.entries}
        assert vocab.issubset(item_ids), f"missing items: {vocab - item_ids}"


class TestAtlasSurfaces:
    """所持欄とカードのテンプレートを別々に照合する。

    古い JSON は所持欄として読み、未知 surface とカード template の不足を隠しません。
    """

    def _manifest(self):
        """同じ画像に異なる surface と ID を割り当てる。

        色の距離ではなく、surface の絞り込みで結果が分かれるようにします。
        """
        crop = np.full((55, 51, 4), 255, dtype=np.uint8)
        crop[..., :3] = (20, 40, 220)
        feature = build_template_feature(crop)
        entries = (TemplateEntry("whip", "weapon", 1, 8, feature),
                   TemplateEntry("gold", "fallback", 1, 1, feature, surface="card"))
        manifest = AtlasManifest(ATLAS_SCHEMA_VERSION, _DUMMY_PROFILE_HASH, _DUMMY_BUILD_HASH,
                                 True, False, hashlib.sha256(np.concatenate([e.feature for e in entries]).tobytes()).hexdigest(), entries)
        return manifest, crop

    def test_matching_filters_surface(self):
        """同じ feature でも指定 surface の ID だけを返す。

        引数を省略した照合では従来の inventory entry を使います。
        """
        manifest, crop = self._manifest()
        matcher = IconMatcher(manifest)
        assert matcher.match(crop).item_id == "whip"
        assert matcher.match(crop, surface="card").item_id == "gold"

    @pytest.mark.parametrize("second_template", [False, True])
    def test_unseen_card_is_unknown_despite_large_margin(self, second_template):
        """距離が大きい未知カードを候補間の差だけで採用しない。

        一候補で margin が固定される場合と、遠い二候補で差が大きい場合を試します。
        """
        manifest, crop = self._manifest()
        unknown = crop.copy()
        unknown[..., :3] = (20, 40, 130)
        if second_template:
            far = crop.copy()
            far[..., :3] = (20, 220, 40)
            manifest = replace(manifest, entries=manifest.entries + (
                TemplateEntry("chicken", "fallback", 1, 1, build_template_feature(far), surface="card"),
            ))
        result = IconMatcher(manifest).match(unknown, surface="card")
        assert result == MatchResult(None, "unknown", None, 0.0, "card_template_mismatch")

    def test_inventory_keeps_distance_tolerant_matching(self):
        """カードの完全一致条件を所持欄へ広げない。

        既存 inventory は従来どおり候補間の margin で照合します。
        """
        manifest, crop = self._manifest()
        changed = crop.copy()
        changed[..., :3] = (20, 40, 130)
        result = IconMatcher(manifest).match(changed)
        assert result.item_id == "whip"
        assert result.confidence == 1.0

    def test_no_surface_entries_is_unknown(self):
        """カード template がない atlas は照合を推測しない。

        所持欄の画像があっても card に流用しません。
        """
        manifest, crop = self._manifest()
        matcher = IconMatcher(replace(manifest, entries=manifest.entries[:1]))
        assert matcher.match(crop, surface="card") == MatchResult(None, "unknown", None, 0., "no_surface_entries")

    def test_legacy_json_defaults_to_inventory(self, tmp_path):
        """surface が省略された既存 JSON を inventory として読む。

        atlas の必須トップレベル key は増やしません。
        """
        manifest, _ = self._manifest()
        wire = json.loads(serialize_manifest(manifest))
        for entry in wire["entries"]:
            entry.pop("surface", None)
        path = tmp_path / "legacy.json"
        path.write_text(json.dumps(wire), encoding="utf-8")
        assert all(e.surface == "inventory" for e in _load_manifest(path).entries)

    @pytest.mark.parametrize("surface", ["popup", None, ["card"]])
    def test_unknown_surface_rejected_on_load(self, tmp_path, surface):
        """未知値や誤った型の surface を JSON 読込で拒否する。

        無効なカード template が黙って inventory に戻ることを防ぎます。
        """
        manifest, _ = self._manifest()
        wire = json.loads(serialize_manifest(manifest))
        wire["entries"][0]["surface"] = surface
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(wire), encoding="utf-8")
        with pytest.raises(ValueError, match="surface"):
            _load_manifest(path)

    def test_roundtrip_preserves_surface_and_feature_hash(self, tmp_path):
        """surface の往復と feature だけの hash 定義を維持する。

        surface を交換しても同じ feature 列なら atlas_content_hash は変わりません。
        """
        manifest, _ = self._manifest()
        path = tmp_path / "roundtrip.json"
        path.write_bytes(serialize_manifest(manifest))
        loaded = _load_manifest(path)
        assert [e.surface for e in loaded.entries] == ["inventory", "card"]
        assert loaded.atlas_content_hash == hashlib.sha256(np.concatenate([e.feature for e in loaded.entries]).tobytes()).hexdigest()
        swapped = replace(manifest, entries=tuple(replace(e, surface="card" if e.surface == "inventory" else "inventory") for e in manifest.entries))
        assert swapped.atlas_content_hash == manifest.atlas_content_hash

    def test_development_builder_has_both_surfaces_in_stable_order(self, dev_atlas_manifest):
        """開発 atlas は全 item・level に二つの surface を持つ。

        順序は item ID、surface、level に固定し、同じ組の feature は共有します。
        """
        entries = dev_atlas_manifest.entries
        keys = [(e.item_id, e.surface, e.level) for e in entries]
        assert keys == sorted(keys)
        features = {(e.item_id, e.surface, e.level): e.feature for e in entries}
        for item, surface, level in keys:
            assert np.array_equal(features[item, "card", level], features[item, "inventory", level])
