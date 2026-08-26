"""WorldDataset のテスト。

COCO bbox bounds/class/session split/empty frame/crowded frame、class map の
整合性、preflight チェックを synthetic fixture だけで検証する。
実画像・実アノテーションは不要。
"""
from __future__ import annotations

import json
import math
import pathlib
import hashlib
import pytest

from survivors.vision.world_dataset import (
    COCOAnnotation,
    WorldDataset,
    WorldClassMap,
    DatasetPreflightError,
    SessionSplit,
    load_class_map,
)


# ---- fixtures ----

CONFIGS_DIR = pathlib.Path(__file__).parents[2] / "configs"
CLASS_MAP_PATH = CONFIGS_DIR / "world_class_map_v1.yaml"


def _coco_record(
    image_id: int,
    bbox: tuple[float, float, float, float],
    category_id: int,
    session_id: str = "session-001",
) -> dict:
    """テスト用 COCO アノテーションレコードを生成する。"""
    return {
        "image_id": image_id,
        "category_id": category_id,
        "bbox": list(bbox),  # [x, y, w, h]
        "session_id": session_id,
    }


def _make_coco_json(
    records: list[dict],
    num_images: int,
    categories: list[dict] | None = None,
    *,
    add_images: bool = True,
) -> dict:
    """最小限の COCO JSON 構造を組み立てる。

    annotation の session_id を images[].session_id へ伝播し、
    DatasetSample.session_id（画像レベル正本）が正しく設定されるようにする。
    """
    if categories is None:
        categories = [
            {"id": i, "name": name}
            for i, name in enumerate(
                [
                    "__background__",
                    "player_anchor",
                    "enemy_normal",
                    "enemy_elite",
                    "enemy_boss",
                    "gem_blue",
                    "gem_green",
                    "gem_red",
                    "pickup_heal",
                    "pickup_special",
                    "hazard_projectile",
                    "hazard_area",
                ]
            )
        ]
    # annotation の session_id を image レベルへ伝播する（最初に出現した値を採用）
    image_session: dict[int, str] = {}
    for r in records:
        sid = r.get("session_id", "")
        if sid and r["image_id"] not in image_session:
            image_session[r["image_id"]] = sid
    images = [
        {
            "id": i,
            "file_name": f"frame_{i:06d}.png",
            "width": 1920,
            "height": 1080,
            **( {"session_id": image_session[i]} if i in image_session else {} ),
        }
        for i in range(num_images)
    ] if add_images else []
    annotations = [
        {
            "id": idx,
            "image_id": r["image_id"],
            "category_id": r["category_id"],
            "bbox": r["bbox"],
            "area": r["bbox"][2] * r["bbox"][3],
            "iscrowd": 0,
            "session_id": r.get("session_id", "session-001"),
        }
        for idx, r in enumerate(records)
    ]
    return {"images": images, "annotations": annotations, "categories": categories}


def _write_coco(tmp_path: pathlib.Path, data: dict, name: str = "annotations.json") -> pathlib.Path:
    p = tmp_path / name
    p.write_text(json.dumps(data, ensure_ascii=False))
    return p


# ---- class map ----

class TestWorldClassMap:
    """WorldClassMap の正確な ID・background 予約・num_classes を検証する。"""

    def test_load_class_map_file(self):
        cm = load_class_map(CLASS_MAP_PATH)
        assert cm.num_classes == 12
        assert cm.background_label == 0
        assert cm.background_name == "__background__"
        assert cm.schema_version == "world_class_map.v1"

    def test_foreground_ids_are_1_to_11(self):
        cm = load_class_map(CLASS_MAP_PATH)
        ids = [fc["id"] for fc in cm.foreground_classes]
        assert ids == list(range(1, 12))

    def test_id_to_name_roundtrip(self):
        cm = load_class_map(CLASS_MAP_PATH)
        assert cm.id_to_name(0) == "__background__"
        assert cm.id_to_name(1) == "player_anchor"
        assert cm.id_to_name(11) == "hazard_area"

    def test_unknown_label_rejected(self):
        cm = load_class_map(CLASS_MAP_PATH)
        with pytest.raises(KeyError):
            cm.id_to_name(12)

    def test_name_to_id_roundtrip(self):
        cm = load_class_map(CLASS_MAP_PATH)
        assert cm.name_to_id("player_anchor") == 1
        assert cm.name_to_id("hazard_area") == 11

    def test_unknown_name_rejected(self):
        cm = load_class_map(CLASS_MAP_PATH)
        with pytest.raises(KeyError):
            cm.name_to_id("nonexistent_class")

    def test_head_output_count_equals_num_classes(self):
        """モデルヘッドの出力次元 = num_classes であることを保証する。"""
        cm = load_class_map(CLASS_MAP_PATH)
        assert len(cm.all_class_names) == cm.num_classes


# ---- dataset loading ----

class TestWorldDatasetLoading:
    """COCO アノテーション読み込みの正常系・異常系。"""

    def test_load_valid_annotations(self, tmp_path):
        records = [
            _coco_record(0, (100, 200, 50, 60), 1),
            _coco_record(0, (300, 400, 30, 40), 2),
            _coco_record(1, (500, 600, 20, 30), 3),
        ]
        path = _write_coco(tmp_path, _make_coco_json(records, num_images=2))
        ds = WorldDataset(path, CLASS_MAP_PATH)
        assert len(ds) == 2  # 2 images
        sample0 = ds[0]
        assert len(sample0.annotations) == 2
        assert len(ds[1].annotations) == 1

    def test_bbox_bounds_validated(self, tmp_path):
        """bbox が画像外へはみ出したアノテーションを拒否する。"""
        records = [_coco_record(0, (-10, 0, 50, 60), 1)]
        path = _write_coco(tmp_path, _make_coco_json(records, num_images=1))
        with pytest.raises(ValueError, match="bbox"):
            WorldDataset(path, CLASS_MAP_PATH, validate_bounds=True)

    def test_empty_frame_allowed(self, tmp_path):
        """アノテーションなし画像（空フレーム）は許容される。"""
        path = _write_coco(tmp_path, _make_coco_json([], num_images=1))
        ds = WorldDataset(path, CLASS_MAP_PATH)
        assert len(ds) == 1
        assert len(ds[0].annotations) == 0

    def test_crowded_frame_allowed(self, tmp_path):
        """密集フレーム（30 エンティティ）を正常ロードできる。"""
        records = [_coco_record(0, (i * 30, 0, 25, 25), (i % 11) + 1) for i in range(30)]
        path = _write_coco(tmp_path, _make_coco_json(records, num_images=1))
        ds = WorldDataset(path, CLASS_MAP_PATH)
        assert len(ds[0].annotations) == 30

    def test_unknown_category_id_rejected(self, tmp_path):
        """class map に存在しない category_id は拒否する。"""
        records = [_coco_record(0, (0, 0, 10, 10), 99)]
        path = _write_coco(tmp_path, _make_coco_json(records, num_images=1))
        with pytest.raises(ValueError, match="category_id"):
            WorldDataset(path, CLASS_MAP_PATH)

    def test_background_category_rejected(self, tmp_path):
        """background (id=0) をアノテーションとして使おうとすると拒否する。"""
        records = [_coco_record(0, (0, 0, 10, 10), 0)]
        path = _write_coco(tmp_path, _make_coco_json(records, num_images=1))
        with pytest.raises(ValueError, match="background"):
            WorldDataset(path, CLASS_MAP_PATH)


# ---- session split ----

class TestSessionSplit:
    """session 単位 split の凍結・整合性チェック。"""

    def test_split_assigns_sessions_to_purposes(self, tmp_path):
        sessions = ["s1", "s2", "s3", "s4"]
        split = SessionSplit(
            train=["s1", "s2"],
            validation=["s3"],
            error_calibration=["s4"],
            final_e2e_test=[],
        )
        assert set(split.train) == {"s1", "s2"}
        assert set(split.validation) == {"s3"}

    def test_split_overlap_rejected(self):
        """同一 session を複数用途に割り当てると拒否する。"""
        with pytest.raises(ValueError, match="overlap"):
            SessionSplit(
                train=["s1"],
                validation=["s1"],
                error_calibration=[],
                final_e2e_test=[],
            )

    def test_split_canonical_hash_stable(self):
        split = SessionSplit(
            train=["s1", "s2"],
            validation=["s3"],
            error_calibration=[],
            final_e2e_test=[],
        )
        h1 = split.canonical_hash()
        h2 = split.canonical_hash()
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_split_hash_changes_on_mutation(self):
        s1 = SessionSplit(train=["s1"], validation=["s2"], error_calibration=[], final_e2e_test=[])
        s2 = SessionSplit(train=["s1", "x"], validation=["s2"], error_calibration=[], final_e2e_test=[])
        assert s1.canonical_hash() != s2.canonical_hash()


# ---- preflight ----

class TestDatasetPreflight:
    """minimum frame/entity/class/time-band 数チェック。"""

    def _make_preflight_records(
        self, n_frames=300, n_classes=6, bands=4, n_entities_per_frame=2
    ) -> list[dict]:
        records = []
        frames_per_band = n_frames // bands
        for band in range(bands):
            for fi in range(frames_per_band):
                fid = band * frames_per_band + fi
                for ei in range(max(n_entities_per_frame, n_classes)):
                    cid = (ei % n_classes) + 1
                    records.append(_coco_record(fid, (ei * 40, 0, 30, 30), cid, f"session-{band:02d}"))
        return records

    def test_preflight_passes_with_sufficient_data(self, tmp_path):
        records = self._make_preflight_records()
        n_images = 300
        path = _write_coco(tmp_path, _make_coco_json(records, n_images))
        # must not raise
        WorldDataset(path, CLASS_MAP_PATH).run_preflight(
            min_frames=300, min_entities=500, min_classes=6, min_time_bands=4
        )

    def test_preflight_fails_on_insufficient_frames(self, tmp_path):
        records = self._make_preflight_records(n_frames=100)
        path = _write_coco(tmp_path, _make_coco_json(records, 100))
        ds = WorldDataset(path, CLASS_MAP_PATH)
        with pytest.raises(DatasetPreflightError, match="min_frames"):
            ds.run_preflight(min_frames=300, min_entities=10, min_classes=1, min_time_bands=1)

    def test_preflight_fails_on_insufficient_classes(self, tmp_path):
        records = self._make_preflight_records(n_frames=300, n_classes=3)
        path = _write_coco(tmp_path, _make_coco_json(records, 300))
        ds = WorldDataset(path, CLASS_MAP_PATH)
        with pytest.raises(DatasetPreflightError, match="min_classes"):
            ds.run_preflight(min_frames=300, min_entities=10, min_classes=6, min_time_bands=1)

    def test_preflight_fails_on_insufficient_time_bands(self, tmp_path):
        records = self._make_preflight_records(n_frames=300, bands=2)
        path = _write_coco(tmp_path, _make_coco_json(records, 300))
        ds = WorldDataset(path, CLASS_MAP_PATH)
        with pytest.raises(DatasetPreflightError, match="min_time_bands"):
            ds.run_preflight(min_frames=300, min_entities=10, min_classes=1, min_time_bands=4)

    def test_min_time_bands_uses_image_session_not_annotation_session(self, tmp_path):
        """annotation.session_id が空でも images[].session_id が 4 種あれば min_time_bands=4 を通過する。

        P1回帰テスト: min_time_bands を DatasetSample.session_id（画像レベル）から計算することを確認する。
        """
        sessions = ["s0", "s1", "s2", "s3"]
        images = [
            {"id": i, "file_name": f"frame_{i:04d}.png", "width": 1920, "height": 1080,
             "session_id": sessions[i]}
            for i in range(4)
        ]
        # annotation.session_id を意図的に省略（空）
        annotations = [
            {"id": i, "image_id": i, "category_id": (i % 3) + 1, "bbox": [10, 10, 20, 20]}
            for i in range(4)
        ]
        categories = [{"id": k, "name": f"c{k}"} for k in range(1, 12)]
        data = {"images": images, "annotations": annotations, "categories": categories}
        path = tmp_path / "ann.json"
        path.write_text(__import__("json").dumps(data), encoding="utf-8")

        ds = WorldDataset(path, CLASS_MAP_PATH)
        # annotation session なし・画像 session 4 種 → min_time_bands=4 を通過
        ds.run_preflight(min_frames=4, min_entities=4, min_classes=1, min_time_bands=4)


# ---- 新規回帰テスト: 画像レベル session_id の rejected_sessions チェック ----

class TestImageLevelSessionRejection:
    """画像レベルの session_id による rejected_sessions チェックを確認する（annotation なし負例含む）。"""

    def _write_coco_image_session(
        self, tmp_path: pathlib.Path, image_session: str, with_annotation: bool = False
    ) -> pathlib.Path:
        """images[].session_id に image_session を付与した COCO JSON を作成する。"""
        img = {"id": 0, "file_name": "frame_0.png", "width": 1920, "height": 1080,
               "session_id": image_session}
        anns = []
        if with_annotation:
            anns.append({"id": 0, "image_id": 0, "category_id": 1,
                          "bbox": [0, 0, 10, 10], "session_id": image_session})
        data = {
            "images": [img],
            "annotations": anns,
            "categories": [{"id": k, "name": f"c{k}"} for k in range(1, 12)],
        }
        p = tmp_path / "ann.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        return p

    def test_train_session_no_annotation_passes(self, tmp_path):
        """train session の annotation なし負例は通過する。"""
        path = self._write_coco_image_session(tmp_path, "train_session", with_annotation=False)
        ds = WorldDataset(path, CLASS_MAP_PATH, rejected_sessions={"final_e2e_test"})
        assert len(ds) == 1

    def test_validation_session_no_annotation_passes(self, tmp_path):
        """validation session の annotation なし負例も通過する。"""
        path = self._write_coco_image_session(tmp_path, "validation_session", with_annotation=False)
        ds = WorldDataset(path, CLASS_MAP_PATH, rejected_sessions={"final_e2e_test"})
        assert len(ds) == 1

    def test_error_calibration_no_annotation_raises(self, tmp_path):
        """error_calibration session の annotation なし負例は DatasetPreflightError。"""
        path = self._write_coco_image_session(tmp_path, "error_calibration", with_annotation=False)
        with pytest.raises(DatasetPreflightError):
            WorldDataset(path, CLASS_MAP_PATH, rejected_sessions={"error_calibration"})

    def test_final_e2e_test_no_annotation_raises(self, tmp_path):
        """final_e2e_test session の annotation なし負例は DatasetPreflightError。"""
        path = self._write_coco_image_session(tmp_path, "final_e2e_test", with_annotation=False)
        with pytest.raises(DatasetPreflightError):
            WorldDataset(path, CLASS_MAP_PATH, rejected_sessions={"final_e2e_test"})

    def test_annotation_session_mismatch_raises(self, tmp_path):
        """annotation.session_id と image.session_id が不一致のとき DatasetPreflightError。"""
        data = {
            "images": [{"id": 0, "file_name": "f.png", "width": 1920, "height": 1080,
                         "session_id": "img_sess"}],
            "annotations": [{"id": 0, "image_id": 0, "category_id": 1,
                              "bbox": [0, 0, 10, 10], "session_id": "ann_sess"}],
            "categories": [{"id": k, "name": f"c{k}"} for k in range(1, 12)],
        }
        path = tmp_path / "ann.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(DatasetPreflightError, match="一致しません"):
            WorldDataset(path, CLASS_MAP_PATH)

    def test_matching_annotation_image_session_passes(self, tmp_path):
        """annotation.session_id と image.session_id が一致すれば通過する。"""
        data = {
            "images": [{"id": 0, "file_name": "f.png", "width": 1920, "height": 1080,
                         "session_id": "s1"}],
            "annotations": [{"id": 0, "image_id": 0, "category_id": 1,
                              "bbox": [0, 0, 10, 10], "session_id": "s1"}],
            "categories": [{"id": k, "name": f"c{k}"} for k in range(1, 12)],
        }
        path = tmp_path / "ann.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        ds = WorldDataset(path, CLASS_MAP_PATH, rejected_sessions={"final_e2e_test"})
        assert len(ds) == 1

    def test_rejected_session_with_annotation_also_raises(self, tmp_path):
        """annotation ありでも rejected session は DatasetPreflightError（重複チェック確認）。"""
        path = self._write_coco_image_session(tmp_path, "error_calibration", with_annotation=True)
        with pytest.raises(DatasetPreflightError):
            WorldDataset(path, CLASS_MAP_PATH, rejected_sessions={"error_calibration"})
