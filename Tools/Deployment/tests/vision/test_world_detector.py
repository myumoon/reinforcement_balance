"""WorldDetector のテスト。

feasibility config の architecture dispatch、ssdlite320 head の num_classes=12 置換、
未実装 architecture の silent fallback 禁止、checkpoint manifest hash 保存、
development checkpoint の formal loader 拒否を検証する。
実 GPU・実画像・実 weight は不要。
"""
from __future__ import annotations

import pathlib
import hashlib
import json

import numpy as np
import pytest

from survivors.vision.world_detector import (
    WorldDetector,
    DetectionResult,
    CheckpointManifest,
    FormalDetectorRejectedError,
    UnknownArchitectureError,
    load_detector_config,
)


CONFIGS_DIR = pathlib.Path(__file__).parents[2] / "configs"
DETECTOR_CONFIG_PATH = CONFIGS_DIR / "world_detector_v1.yaml"
CLASS_MAP_PATH = CONFIGS_DIR / "world_class_map_v1.yaml"


# ---- config loading ----

class TestDetectorConfig:
    def test_load_config_schema_version(self):
        cfg = load_detector_config(DETECTOR_CONFIG_PATH)
        assert cfg["schema_version"] == "world_detector.v1"

    def test_formal_detector_eligible_false(self):
        cfg = load_detector_config(DETECTOR_CONFIG_PATH)
        assert cfg["formal_detector_eligible"] is False

    def test_num_classes_is_12(self):
        cfg = load_detector_config(DETECTOR_CONFIG_PATH)
        assert cfg["model"]["num_classes"] == 12

    def test_architecture_is_ssdlite320(self):
        cfg = load_detector_config(DETECTOR_CONFIG_PATH)
        assert cfg["model"]["architecture"] == "ssdlite320"


# ---- architecture dispatch ----

class TestArchitectureDispatch:
    """未実装 architecture に silent fallback しないことを検証する。"""

    def test_unknown_architecture_raises(self, tmp_path):
        cfg = load_detector_config(DETECTOR_CONFIG_PATH)
        cfg["model"]["architecture"] = "unknown_future_arch"
        with pytest.raises(UnknownArchitectureError):
            WorldDetector.from_config(cfg, CLASS_MAP_PATH)

    def test_ssdlite320_creates_detector(self):
        """ssdlite320 は正常に detector を組み立てられる（weight ロードなし）。"""
        cfg = load_detector_config(DETECTOR_CONFIG_PATH)
        det = WorldDetector.from_config(cfg, CLASS_MAP_PATH)
        assert det.num_classes == 12

    def test_head_output_count_matches_num_classes(self):
        """モデルヘッドの分類次元が num_classes と一致する。"""
        cfg = load_detector_config(DETECTOR_CONFIG_PATH)
        det = WorldDetector.from_config(cfg, CLASS_MAP_PATH)
        # head_num_classes は model 内部の出力クラス数を返す
        assert det.head_num_classes == 12


# ---- detection result contract ----

class TestDetectionResult:
    """DetectionResult のフィールド・正規化・viewport 座標を検証する。"""

    def test_detection_result_fields(self):
        dr = DetectionResult(
            boxes_xyxy=np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32),
            scores=np.array([0.9], dtype=np.float32),
            class_ids=np.array([1], dtype=np.int32),
            image_width=1920,
            image_height=1080,
        )
        assert dr.boxes_xyxy.shape == (1, 4)
        assert dr.scores.shape == (1,)
        assert dr.class_ids.shape == (1,)

    def test_normalized_centers(self):
        dr = DetectionResult(
            boxes_xyxy=np.array([[192, 108, 384, 216]], dtype=np.float32),
            scores=np.array([0.8], dtype=np.float32),
            class_ids=np.array([2], dtype=np.int32),
            image_width=1920,
            image_height=1080,
        )
        cx, cy = dr.normalized_centers[0]
        assert abs(cx - 0.15) < 1e-4
        assert abs(cy - 0.15) < 1e-4

    def test_empty_detection_result(self):
        dr = DetectionResult(
            boxes_xyxy=np.zeros((0, 4), dtype=np.float32),
            scores=np.zeros(0, dtype=np.float32),
            class_ids=np.zeros(0, dtype=np.int32),
            image_width=1920,
            image_height=1080,
        )
        assert len(dr) == 0
        assert dr.normalized_centers.shape == (0, 2)


# ---- inference (synthetic) ----

class TestDetectorInference:
    """synthetic 画像で推論パスが通ることを確認する（weight なし）。"""

    def test_infer_returns_detection_result(self):
        cfg = load_detector_config(DETECTOR_CONFIG_PATH)
        det = WorldDetector.from_config(cfg, CLASS_MAP_PATH)
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = det.infer(frame, score_threshold=0.5)
        assert isinstance(result, DetectionResult)

    def test_infer_bgr_input_accepted(self):
        """BGR 入力（OpenCV デフォルト）を受け付ける。"""
        cfg = load_detector_config(DETECTOR_CONFIG_PATH)
        det = WorldDetector.from_config(cfg, CLASS_MAP_PATH)
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = det.infer(frame, score_threshold=0.5)
        assert result.image_width == 1920
        assert result.image_height == 1080

    def test_infer_wrong_shape_raises(self):
        """不正な入力形状はエラーになる。"""
        cfg = load_detector_config(DETECTOR_CONFIG_PATH)
        det = WorldDetector.from_config(cfg, CLASS_MAP_PATH)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        # 小さいフレームでもエラーにならないが形状がおかしい場合は拒否
        # 実装で (H,W,3) uint8 のみ受け付ける
        bad = np.zeros((1080, 1920), dtype=np.uint8)  # no channel dim
        with pytest.raises(ValueError, match="shape"):
            det.infer(bad)


# ---- checkpoint manifest ----

class TestCheckpointManifest:
    """checkpoint manifest に必要なハッシュが含まれ、development は formal loader に拒否されることを確認する。"""

    def test_manifest_contains_required_hashes(self, tmp_path):
        manifest = CheckpointManifest(
            model_hash="a" * 64,
            data_hash="b" * 64,
            config_hash="c" * 64,
            build_hash="d" * 64,
            class_map_hash="e" * 64,
            formal_detector_eligible=False,
        )
        assert len(manifest.model_hash) == 64
        assert manifest.formal_detector_eligible is False

    def test_development_manifest_rejected_by_formal_loader(self):
        manifest = CheckpointManifest(
            model_hash="a" * 64,
            data_hash="b" * 64,
            config_hash="c" * 64,
            build_hash="d" * 64,
            class_map_hash="e" * 64,
            formal_detector_eligible=False,
        )
        with pytest.raises(FormalDetectorRejectedError):
            manifest.assert_formal_eligible()

    def test_manifest_round_trip_json(self, tmp_path):
        manifest = CheckpointManifest(
            model_hash="a" * 64,
            data_hash="b" * 64,
            config_hash="c" * 64,
            build_hash="d" * 64,
            class_map_hash="e" * 64,
            formal_detector_eligible=False,
        )
        path = tmp_path / "manifest.json"
        manifest.save(path)
        loaded = CheckpointManifest.load(path)
        assert loaded.model_hash == manifest.model_hash
        assert loaded.formal_detector_eligible is False

    def test_manifest_development_only_and_training_mode_round_trip(self, tmp_path):
        """development_only と training_mode が save/load で保持される。"""
        for mode in ("smoke", "development"):
            manifest = CheckpointManifest(
                model_hash="a" * 64,
                data_hash="b" * 64,
                config_hash="c" * 64,
                build_hash="d" * 64,
                class_map_hash="e" * 64,
                formal_detector_eligible=False,
                development_only=True,
                training_mode=mode,
            )
            path = tmp_path / f"manifest_{mode}.json"
            manifest.save(path)
            loaded = CheckpointManifest.load(path)
            assert loaded.development_only is True
            assert loaded.training_mode == mode

    def test_load_rejects_invalid_hash_format(self, tmp_path):
        """SHA-256 形式でないハッシュを持つ manifest の load を拒否する。"""
        bad = {
            "model_hash": "bad_hash",  # 64 hex chars でない
            "data_hash": "b" * 64,
            "config_hash": "c" * 64,
            "build_hash": "d" * 64,
            "class_map_hash": "e" * 64,
            "formal_detector_eligible": False,
        }
        path = tmp_path / "bad_manifest.json"
        path.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(ValueError, match="model_hash"):
            CheckpointManifest.load(path)

    def test_load_rejects_string_formal_flag(self, tmp_path):
        """formal_detector_eligible が文字列 'false' の manifest は load で拒否する。"""
        bad = {
            "model_hash": "a" * 64,
            "data_hash": "b" * 64,
            "config_hash": "c" * 64,
            "build_hash": "d" * 64,
            "class_map_hash": "e" * 64,
            "formal_detector_eligible": "false",  # bool でない
        }
        path = tmp_path / "bad_formal.json"
        path.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(ValueError, match="bool"):
            CheckpointManifest.load(path)

    def test_assert_formal_eligible_rejects_non_bool_true(self):
        """formal_detector_eligible=True (bool) でないと assert_formal_eligible() は拒否する。"""
        manifest = CheckpointManifest(
            model_hash="a" * 64,
            data_hash="b" * 64,
            config_hash="c" * 64,
            build_hash="d" * 64,
            class_map_hash="e" * 64,
            formal_detector_eligible=False,
        )
        with pytest.raises(FormalDetectorRejectedError):
            manifest.assert_formal_eligible()

    def test_verify_identity_passes_on_match(self):
        """verify_identity() は期待 hash と一致すれば例外を送出しない。"""
        manifest = CheckpointManifest(
            model_hash="a" * 64,
            data_hash="b" * 64,
            config_hash="c" * 64,
            build_hash="d" * 64,
            class_map_hash="e" * 64,
            formal_detector_eligible=False,
        )
        manifest.verify_identity({"model_hash": "a" * 64, "data_hash": "b" * 64})

    def test_verify_identity_fails_on_mismatch(self):
        """verify_identity() は期待 hash と不一致なら ValueError を送出する。"""
        manifest = CheckpointManifest(
            model_hash="a" * 64,
            data_hash="b" * 64,
            config_hash="c" * 64,
            build_hash="d" * 64,
            class_map_hash="e" * 64,
            formal_detector_eligible=False,
        )
        with pytest.raises(ValueError, match="identity 不一致"):
            manifest.verify_identity({"model_hash": "0" * 64})


class TestArchitectureDispatchP1:
    """P1 指摘: feasibility 既知でも未実装 architecture は拒否される。"""

    def test_tile_2x2_raises_not_fallback_to_ssdlite320(self):
        """tile_2x2 は feasibility 既知だが未実装 → silent fallback せず拒否する。"""
        cfg = load_detector_config(DETECTOR_CONFIG_PATH)
        cfg["model"]["architecture"] = "tile_2x2"
        with pytest.raises(UnknownArchitectureError):
            WorldDetector.from_config(cfg, CLASS_MAP_PATH)

    def test_ssdlite640_multiscale_raises(self):
        cfg = load_detector_config(DETECTOR_CONFIG_PATH)
        cfg["model"]["architecture"] = "ssdlite640_multiscale"
        with pytest.raises(UnknownArchitectureError):
            WorldDetector.from_config(cfg, CLASS_MAP_PATH)

    def test_num_classes_mismatch_raises(self, tmp_path):
        """class map の num_classes と config が不一致なら拒否する。"""
        cfg = load_detector_config(DETECTOR_CONFIG_PATH)
        cfg["model"]["num_classes"] = 5  # class map は 12
        with pytest.raises(ValueError, match="num_classes"):
            WorldDetector.from_config(cfg, CLASS_MAP_PATH)
