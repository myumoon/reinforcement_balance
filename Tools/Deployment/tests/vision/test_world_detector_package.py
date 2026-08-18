"""world_detector_package のテスト。

preflight 拒否（missing/stale feasibility verdict、split 混入）、
error_calibration/final_e2e_test session 拒否、
checkpoint selection、metric gate 境界値、
package publish / restore、schema 検証、formal_detector_eligible=false を検証する。
"""
from __future__ import annotations

import json
import pathlib
import shutil
from dataclasses import asdict

import numpy as np
import pytest

# ---- helpers ----

CONFIGS_DIR = pathlib.Path(__file__).parents[2] / "configs"
DETECTOR_CONFIG_PATH = CONFIGS_DIR / "world_detector_v1.yaml"
CLASS_MAP_PATH = CONFIGS_DIR / "world_class_map_v1.yaml"


def _make_coco(
    n_images: int = 5,
    sessions: list[str] | None = None,
    n_classes: int = 6,
) -> dict:
    """synthetic COCO JSON を生成する。

    n_images 枚の画像に対して 1 アノテーションずつ付与する。
    セッションは sessions リストから round-robin で割り当てる。
    """
    sessions = sessions or ["s1", "s2", "s3", "s4"]
    images = [
        {"id": i, "file_name": f"frame_{i:04d}.png", "width": 1920, "height": 1080}
        for i in range(n_images)
    ]
    annotations = []
    for i in range(n_images):
        cat_id = (i % n_classes) + 1  # 1..n_classes
        annotations.append({
            "id": i,
            "image_id": i,
            "category_id": cat_id,
            "bbox": [10, 10, 50, 50],
            "session_id": sessions[i % len(sessions)],
        })
    categories = [{"id": k, "name": f"class_{k}"} for k in range(1, 12)]
    return {"images": images, "annotations": annotations, "categories": categories}


def _write_coco(tmp_path: pathlib.Path, data: dict) -> pathlib.Path:
    p = tmp_path / "annotations.json"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return p


def _make_checkpoint_manifest(formal: bool = False):
    """ダミー CheckpointManifest を生成する。"""
    from survivors.vision.world_detector import CheckpointManifest
    return CheckpointManifest(
        model_hash="a" * 64,
        data_hash="b" * 64,
        config_hash=_sha256_file(DETECTOR_CONFIG_PATH),  # 実 config hash（package の検証に通る）
        build_hash="d" * 64,
        class_map_hash=_sha256_file(CLASS_MAP_PATH),
        formal_detector_eligible=formal,
    )


def _sha256_file(path: pathlib.Path) -> str:
    import hashlib
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


# ---- Task 1: formal preflight / session rejection ----

class TestRejectedSessions:
    """error_calibration / final_e2e_test session の拒否を検証する。"""

    def test_rejected_session_raises_on_load(self, tmp_path):
        """拒否セッション ID を含む COCO JSON を渡すと DatasetPreflightError。"""
        from survivors.vision.world_dataset import WorldDataset, DatasetPreflightError

        data = _make_coco(n_images=10, sessions=["train1", "error_cal"])
        ann_path = _write_coco(tmp_path, data)

        with pytest.raises(DatasetPreflightError, match="error_cal"):
            WorldDataset(
                ann_path, CLASS_MAP_PATH,
                rejected_sessions={"error_cal"},
            )

    def test_non_rejected_session_passes(self, tmp_path):
        """拒否セッションを含まない場合は正常にロードできる。"""
        from survivors.vision.world_dataset import WorldDataset

        data = _make_coco(n_images=10, sessions=["s1", "s2", "s3", "s4"])
        ann_path = _write_coco(tmp_path, data)

        ds = WorldDataset(ann_path, CLASS_MAP_PATH, rejected_sessions={"error_cal"})
        assert len(ds) > 0

    def test_empty_rejected_set_allows_all(self, tmp_path):
        """rejected_sessions=None のときは全セッションが通過する。"""
        from survivors.vision.world_dataset import WorldDataset

        data = _make_coco(n_images=5, sessions=["any_session"])
        ann_path = _write_coco(tmp_path, data)

        ds = WorldDataset(ann_path, CLASS_MAP_PATH, rejected_sessions=None)
        assert len(ds) > 0


class TestSplitPreflight:
    """SessionSplit の preflight チェックを検証する。"""

    def test_error_calibration_in_data_raises(self, tmp_path):
        """error_calibration session がデータに含まれていると run_split_preflight が失敗。"""
        from survivors.vision.world_dataset import WorldDataset, SessionSplit, DatasetPreflightError

        data = _make_coco(n_images=20, sessions=["s1", "s2", "s3", "s4"])
        ann_path = _write_coco(tmp_path, data)
        ds = WorldDataset(ann_path, CLASS_MAP_PATH)

        split = SessionSplit(
            train=["s1", "s2"],
            validation=["s3"],
            error_calibration=["s4"],  # s4 はデータに含まれている
            final_e2e_test=[],
        )
        with pytest.raises(DatasetPreflightError, match="s4"):
            ds.run_split_preflight(split, min_frames_per_split=1, min_entities_per_split=1, min_classes_per_split=1)

    def test_final_e2e_in_data_raises(self, tmp_path):
        """final_e2e_test session がデータに含まれていると失敗。"""
        from survivors.vision.world_dataset import WorldDataset, SessionSplit, DatasetPreflightError

        data = _make_coco(n_images=20, sessions=["s1", "s2", "s3", "s4"])
        ann_path = _write_coco(tmp_path, data)
        ds = WorldDataset(ann_path, CLASS_MAP_PATH)

        split = SessionSplit(
            train=["s1", "s2"],
            validation=["s3"],
            error_calibration=[],
            final_e2e_test=["s4"],  # s4 はデータに含まれている
        )
        with pytest.raises(DatasetPreflightError, match="s4"):
            ds.run_split_preflight(split, min_frames_per_split=1, min_entities_per_split=1, min_classes_per_split=1)

    def test_zero_session_split_raises(self, tmp_path):
        """train session が 0 件の split は失敗する。"""
        from survivors.vision.world_dataset import WorldDataset, SessionSplit, DatasetPreflightError

        data = _make_coco(n_images=20, sessions=["s1", "s2", "s3", "s4"])
        ann_path = _write_coco(tmp_path, data)
        ds = WorldDataset(ann_path, CLASS_MAP_PATH)

        split = SessionSplit(
            train=[],  # 0 件
            validation=["s1"],
            error_calibration=[],
            final_e2e_test=[],
        )
        with pytest.raises(DatasetPreflightError, match="train"):
            ds.run_split_preflight(split, min_frames_per_split=1, min_entities_per_split=1, min_classes_per_split=1)


# ---- Task 2: training smoke / checkpoint selection ----

class TestCheckpointSelector:
    """CheckpointSelector の選択規則を検証する。"""

    def test_best_is_highest_metric(self):
        """複数記録のうち val_map50_95 最大のものが best になる。"""
        from train_survivors_world_detector import CheckpointSelector, CheckpointRecord

        sel = CheckpointSelector(metric="val_map50_95", keep_top_k=3)
        sel.record(CheckpointRecord(1, 0.1, "e1.pt"))
        sel.record(CheckpointRecord(2, 0.3, "e2.pt"))
        sel.record(CheckpointRecord(3, 0.2, "e3.pt"))
        assert sel.best is not None
        assert sel.best.epoch == 2
        assert sel.best.val_map50_95 == pytest.approx(0.3)

    def test_keep_top_k_enforced(self):
        """keep_top_k を超えた候補は削除される。"""
        from train_survivors_world_detector import CheckpointSelector, CheckpointRecord

        sel = CheckpointSelector(metric="val_map50_95", keep_top_k=2)
        for ep in range(5):
            sel.record(CheckpointRecord(ep, float(ep) * 0.1, f"e{ep}.pt"))
        assert len(sel._records) <= 2

    def test_to_dict_contains_selection_rules(self):
        """to_dict は metric / keep_top_k / best_epoch / best_val を含む。"""
        from train_survivors_world_detector import CheckpointSelector, CheckpointRecord

        sel = CheckpointSelector(metric="val_map50_95", keep_top_k=3)
        sel.record(CheckpointRecord(5, 0.42, "e5.pt"))
        d = sel.to_dict()
        assert d["metric"] == "val_map50_95"
        assert d["keep_top_k"] == 3
        assert d["best_epoch"] == 5
        assert d["best_val_map50_95"] == pytest.approx(0.42)

    def test_none_best_when_no_records(self):
        """記録なしのとき best は None。"""
        from train_survivors_world_detector import CheckpointSelector

        sel = CheckpointSelector(metric="val_map50_95", keep_top_k=3)
        assert sel.best is None

    def test_from_dict_restores_records(self, tmp_path):
        """from_dict は records を正しく復元する（resume 用）。"""
        from train_survivors_world_detector import CheckpointSelector, CheckpointRecord

        sel = CheckpointSelector(metric="val_map50_95", keep_top_k=3)
        sel.record(CheckpointRecord(10, 0.5, "e10.pt"))
        d = sel.to_dict()

        restored = CheckpointSelector.from_dict(d)
        assert restored.best is not None
        assert restored.best.epoch == 10
        assert restored.best.val_map50_95 == pytest.approx(0.5)

    def test_resume_raises_on_missing_state_file(self, tmp_path):
        """training_state.json が存在しない resume_dir は ValueError。"""
        from train_survivors_world_detector import (
            CheckpointSelector, _load_resume_state, _RESUME_STATE_NAME
        )

        sel = CheckpointSelector(metric="val_map50_95", keep_top_k=3)
        with pytest.raises(ValueError, match="training_state.json"):
            _load_resume_state(tmp_path, sel, None, None)

    def test_resume_raises_on_invalid_last_epoch(self, tmp_path):
        """last_epoch が不正な training_state.json は ValueError。"""
        from train_survivors_world_detector import (
            CheckpointSelector, _load_resume_state, _RESUME_STATE_NAME
        )

        bad_state = {"last_epoch": "not_an_int", "selector": {}}
        (tmp_path / _RESUME_STATE_NAME).write_text(json.dumps(bad_state), encoding="utf-8")

        sel = CheckpointSelector(metric="val_map50_95", keep_top_k=3)
        with pytest.raises(ValueError, match="last_epoch"):
            _load_resume_state(tmp_path, sel, None, None)


class TestSessionBalancedStep:
    """_session_balanced_step が実 DatasetSample から target を構築することを確認する。"""

    def test_step_uses_dataset_targets(self, tmp_path):
        """DatasetSample の annotation から実 target が構築されることを確認する。

        SSDLite320（stub）は list を返すが、例外なしで完了することを検証する。
        torch が不在（_StubModel 環境）ではスキップする。
        """
        # torch が不在ならこのテストはスキップ
        torch = pytest.importorskip("torch")
        import torch.optim as optim

        from survivors.vision.world_dataset import WorldDataset
        from survivors.vision.world_detector import WorldDetector, load_detector_config, _TORCH_AVAILABLE
        from train_survivors_world_detector import _session_balanced_step

        if not _TORCH_AVAILABLE:
            pytest.skip("torch 不在環境では _StubModel なので skip")

        data = _make_coco(n_images=5, sessions=["s1", "s2"])
        ann_path = _write_coco(tmp_path, data)
        ds = WorldDataset(ann_path, CLASS_MAP_PATH)

        cfg = load_detector_config(DETECTOR_CONFIG_PATH)
        detector = WorldDetector.from_config(cfg, CLASS_MAP_PATH)
        optimizer = optim.SGD(detector._model.parameters(), lr=0.001)

        # torch と DatasetSample がある状態で step を実行し、例外がないことを確認
        _session_balanced_step(detector, ds, optimizer, cfg)


# ---- Task 3: performance gate boundary values ----

class TestPerformanceGateBoundary:
    """metric gate 境界値を synthetic metrics でテストする。"""

    def _gate_cfg(self) -> dict:
        """テスト用の gate config（実際の閾値）。"""
        return {
            "map50_95_min": 0.0,
            "class_recall_min": {
                "enemy_normal": 0.90,
                "enemy_boss": 0.98,
                "gem_blue": 0.92,
            },
            "density_correlation_min": 0.85,
            "nearest_distance_median_max": 0.04,
            "gpu_p95_latency_max_ms": 25.0,
            "slice_gate": {
                "heavy_effect": {"recall_min": 0.80, "min_instances": 4, "min_sessions": 4},
                "boss": {"recall_min": 0.98, "min_instances": 3, "min_sessions": 3},
            },
        }

    def _base_metrics(self):
        """base の合格 metrics（enemy_normal recall は 0.95 に設定）。"""
        from eval_survivors_world_detector import EvalMetrics
        return EvalMetrics(
            proxy_ap50_95=0.5,
            class_recall={2: 0.95, 4: 0.99, 5: 0.93},  # enemy_normal=2, boss=4, gem_blue=5
            density_error=0.10,  # 1 - 0.10 = 0.90 >= 0.85
            count_error=0.10,
            nearest_distance_error=0.02,  # <= 0.04
            mean_latency_ms=10.0,
        )

    def _class_name_by_id(self) -> dict:
        return {2: "enemy_normal", 4: "enemy_boss", 5: "gem_blue"}

    def test_enemy_recall_below_threshold_fails(self):
        """enemy_normal recall < 0.90 はゲート FAIL。"""
        from eval_survivors_world_detector import EvalMetrics, check_performance_gate

        m = self._base_metrics()
        m.class_recall[2] = 0.89  # 0.90 未満
        result = check_performance_gate(m, self._gate_cfg(), self._class_name_by_id())
        assert not result.class_recall_results["enemy_normal"]["passed"]
        assert not result.passed

    def test_enemy_recall_at_threshold_passes(self):
        """enemy_normal recall = 0.90 ちょうどで PASS。"""
        from eval_survivors_world_detector import EvalMetrics, check_performance_gate

        m = self._base_metrics()
        m.class_recall[2] = 0.90
        result = check_performance_gate(m, self._gate_cfg(), self._class_name_by_id())
        assert result.class_recall_results["enemy_normal"]["passed"]

    def test_boss_recall_below_threshold_fails(self):
        """enemy_boss recall < 0.98 はゲート FAIL。"""
        from eval_survivors_world_detector import EvalMetrics, check_performance_gate

        m = self._base_metrics()
        m.class_recall[4] = 0.97
        result = check_performance_gate(m, self._gate_cfg(), self._class_name_by_id())
        assert not result.class_recall_results["enemy_boss"]["passed"]

    def test_gem_recall_below_threshold_fails(self):
        """gem_blue recall < 0.92 はゲート FAIL。"""
        from eval_survivors_world_detector import EvalMetrics, check_performance_gate

        m = self._base_metrics()
        m.class_recall[5] = 0.91
        result = check_performance_gate(m, self._gate_cfg(), self._class_name_by_id())
        assert not result.class_recall_results["gem_blue"]["passed"]

    def test_density_correlation_below_threshold_fails(self):
        """density_error=0.20 → density_corr=0.80 < 0.85 は FAIL。"""
        from eval_survivors_world_detector import check_performance_gate

        m = self._base_metrics()
        m.density_error = 0.20
        result = check_performance_gate(m, self._gate_cfg(), self._class_name_by_id())
        assert not result.passed

    def test_nearest_distance_above_max_fails(self):
        """nearest_distance_error > 0.04 は FAIL。"""
        from eval_survivors_world_detector import check_performance_gate

        m = self._base_metrics()
        m.nearest_distance_error = 0.05
        result = check_performance_gate(m, self._gate_cfg(), self._class_name_by_id())
        assert not result.passed

    def test_slice_instance_below_min_fails(self):
        """heavy_effect slice の instance 数 < min_instances は FAIL。"""
        from eval_survivors_world_detector import check_performance_gate

        m = self._base_metrics()
        # heavy_effect slice に 3 件しかない（min_instances=4）
        slice_annotations = {
            "heavy_effect": [
                {"image_id": 0, "category_id": 2, "bbox": [10, 10, 50, 50], "session_id": "s1"},
                {"image_id": 1, "category_id": 2, "bbox": [10, 10, 50, 50], "session_id": "s2"},
                {"image_id": 2, "category_id": 2, "bbox": [10, 10, 50, 50], "session_id": "s3"},
            ],
        }
        result = check_performance_gate(
            m, self._gate_cfg(), self._class_name_by_id(),
            slice_annotations=slice_annotations,
        )
        heavy = next(r for r in result.slice_results if r.slice_name == "heavy_effect")
        assert not heavy.passed

    def test_slice_recall_computed_from_predictions(self):
        """slice recall は prediction との一対一 matching で計算される。"""
        from eval_survivors_world_detector import check_performance_gate

        m = self._base_metrics()
        gate_cfg = {
            "map50_95_min": 0.0,
            "class_recall_min": {},
            "density_correlation_min": 0.0,
            "nearest_distance_median_max": 1.0,
            "gpu_p95_latency_max_ms": 1000.0,
            "slice_gate": {
                "boss": {"recall_min": 0.98, "min_instances": 3, "min_sessions": 3},
            },
        }
        # GT: 3 件 / 3 session、pred: 完全一致する 3 件
        gt_anns = [
            {"image_id": i, "category_id": 4, "bbox": [10, 10, 50, 50], "session_id": f"s{i+1}"}
            for i in range(3)
        ]
        pred_anns = [
            {"image_id": i, "category_id": 4, "bbox": [10, 10, 50, 50], "score": 0.9}
            for i in range(3)
        ]
        result = check_performance_gate(
            m, gate_cfg, {4: "enemy_boss"},
            slice_annotations={"boss": gt_anns},
            slice_predictions={"boss": pred_anns},
        )
        boss = next(r for r in result.slice_results if r.slice_name == "boss")
        assert boss.recall == pytest.approx(1.0)
        assert boss.passed

    def test_slice_recall_zero_when_no_predictions(self):
        """prediction が空のとき recall=0.0、閾値 > 0 ならば FAIL。"""
        from eval_survivors_world_detector import check_performance_gate

        m = self._base_metrics()
        gate_cfg = {
            "map50_95_min": 0.0,
            "class_recall_min": {},
            "density_correlation_min": 0.0,
            "nearest_distance_median_max": 1.0,
            "gpu_p95_latency_max_ms": 1000.0,
            "slice_gate": {
                "heavy_effect": {"recall_min": 0.80, "min_instances": 4, "min_sessions": 4},
            },
        }
        gt_anns = [
            {"image_id": i, "category_id": 2, "bbox": [10, 10, 50, 50], "session_id": f"s{i+1}"}
            for i in range(4)
        ]
        result = check_performance_gate(
            m, gate_cfg, {},
            slice_annotations={"heavy_effect": gt_anns},
            slice_predictions={"heavy_effect": []},  # 空予測
        )
        heavy = next(r for r in result.slice_results if r.slice_name == "heavy_effect")
        assert heavy.recall == pytest.approx(0.0)
        assert not heavy.passed

    def test_latency_above_max_fails(self):
        """GPU p95 latency > 25ms は FAIL。"""
        from eval_survivors_world_detector import check_performance_gate

        m = self._base_metrics()
        result = check_performance_gate(
            m, self._gate_cfg(), self._class_name_by_id(),
            gpu_p95_latency_ms=26.0,
        )
        assert not result.passed

    def test_all_pass_returns_passed_true(self):
        """全 gate を通過したとき passed=True。"""
        from eval_survivors_world_detector import check_performance_gate

        m = self._base_metrics()
        gate_cfg = {
            "map50_95_min": 0.0,
            "class_recall_min": {},
            "density_correlation_min": 0.0,
            "nearest_distance_median_max": 1.0,
            "gpu_p95_latency_max_ms": 1000.0,
            "slice_gate": {},
        }
        result = check_performance_gate(m, gate_cfg, {})
        assert result.passed


# ---- Task 4: package writer ----

class TestPackagePublish:
    """package publish / restore / schema 検証を synthetic fixture で確認する。"""

    def test_publish_creates_manifest_json(self, tmp_path):
        """publish_development_package が manifest.json を作成する。"""
        from survivors.vision.world_detector_package import publish_development_package

        store = tmp_path / "store"
        store.mkdir()
        pkg_path = publish_development_package(
            _make_checkpoint_manifest(),
            metrics_dict={"proxy_ap50_95": 0.0},
            checkpoint_selection={"metric": "val_map50_95", "keep_top_k": 3},
            store_dir=store,
            cfg_path=DETECTOR_CONFIG_PATH,
            cm_path=CLASS_MAP_PATH,
        )
        assert pkg_path.exists()
        assert pkg_path.name == "manifest.json"

    def test_package_contains_config_and_class_map(self, tmp_path):
        """package ディレクトリに config / class_map がコピーされている。"""
        from survivors.vision.world_detector_package import publish_development_package

        store = tmp_path / "store"
        store.mkdir()
        pkg_path = publish_development_package(
            _make_checkpoint_manifest(),
            metrics_dict={},
            checkpoint_selection={},
            store_dir=store,
            cfg_path=DETECTOR_CONFIG_PATH,
            cm_path=CLASS_MAP_PATH,
        )
        pkg_dir = pkg_path.parent
        assert (pkg_dir / "world_detector_v1.yaml").exists()
        assert (pkg_dir / "world_class_map_v1.yaml").exists()

    def test_published_package_is_formal_ineligible(self, tmp_path):
        """開発 package の formal_detector_eligible は常に False。"""
        from survivors.vision.world_detector_package import publish_development_package, PackageManifest

        store = tmp_path / "store"
        store.mkdir()
        pkg_path = publish_development_package(
            _make_checkpoint_manifest(),
            metrics_dict={},
            checkpoint_selection={},
            store_dir=store,
            cfg_path=DETECTOR_CONFIG_PATH,
            cm_path=CLASS_MAP_PATH,
        )
        raw = json.loads(pkg_path.read_text(encoding="utf-8"))
        pm = PackageManifest.from_dict(raw)
        assert pm.formal_detector_eligible is False

    def test_assert_formal_eligible_raises_for_dev(self, tmp_path):
        """development package の assert_formal_eligible は例外を送出する。"""
        from survivors.vision.world_detector_package import (
            publish_development_package, PackageManifest, FormalPackageRejectedError
        )

        store = tmp_path / "store"
        store.mkdir()
        pkg_path = publish_development_package(
            _make_checkpoint_manifest(),
            metrics_dict={},
            checkpoint_selection={},
            store_dir=store,
            cfg_path=DETECTOR_CONFIG_PATH,
            cm_path=CLASS_MAP_PATH,
        )
        raw = json.loads(pkg_path.read_text(encoding="utf-8"))
        pm = PackageManifest.from_dict(raw)
        with pytest.raises(FormalPackageRejectedError):
            pm.assert_formal_eligible()

    def test_formal_publish_requires_validation_pass(self, tmp_path):
        """formal publish は validation_passed=False のとき ValueError。"""
        from survivors.vision.world_detector_package import publish_formal_package

        store = tmp_path / "store"
        store.mkdir()
        with pytest.raises(ValueError, match="validation PASS"):
            publish_formal_package(
                _make_checkpoint_manifest(formal=True),
                {}, {}, store,
                cfg_path=DETECTOR_CONFIG_PATH,
                cm_path=CLASS_MAP_PATH,
                validation_passed=False,
            )

    def test_formal_publish_calls_assert_formal_eligible(self, tmp_path):
        """formal publish は assert_formal_eligible() を呼び parent hash を検証する。

        formal_detector_eligible=False の checkpoint は FormalDetectorRejectedError で拒否される。
        CheckpointManifest.assert_formal_eligible() は FormalDetectorRejectedError を送出し、
        publish_formal_package はそれを伝播させる。
        """
        from survivors.vision.world_detector import FormalDetectorRejectedError
        from survivors.vision.world_detector_package import publish_formal_package

        store = tmp_path / "store"
        store.mkdir()
        # formal_detector_eligible=False の checkpoint を渡す
        with pytest.raises(FormalDetectorRejectedError):
            publish_formal_package(
                _make_checkpoint_manifest(formal=False),   # development checkpoint
                {}, {}, store,
                cfg_path=DETECTOR_CONFIG_PATH,
                cm_path=CLASS_MAP_PATH,
                validation_passed=True,
            )

    def test_formal_publish_requires_valid_hashes(self, tmp_path):
        """assert_formal_eligible は SHA-256 形式不正も拒否する。

        model_hash="" → _validate_hash_format() が ValueError を送出する。
        publish_formal_package はこれを伝播させる。
        """
        from survivors.vision.world_detector import CheckpointManifest, FormalDetectorRejectedError
        from survivors.vision.world_detector_package import publish_formal_package

        store = tmp_path / "store"
        store.mkdir()
        # model_hash が SHA-256 形式不正（64 hex 文字でない）の formal manifest
        bad_cm = CheckpointManifest(
            model_hash="",     # 不正
            data_hash="b" * 64,
            config_hash="c" * 64,
            build_hash="d" * 64,
            class_map_hash="e" * 64,
            formal_detector_eligible=True,
        )
        # assert_formal_eligible → _validate_hash_format → ValueError
        with pytest.raises(ValueError):
            publish_formal_package(
                bad_cm, {}, {}, store,
                cfg_path=DETECTOR_CONFIG_PATH,
                cm_path=CLASS_MAP_PATH,
                validation_passed=True,
            )

    def test_idempotent_publish(self, tmp_path):
        """同じ内容を 2 回 publish しても同じ path が返る。"""
        from survivors.vision.world_detector_package import publish_development_package

        store = tmp_path / "store"
        store.mkdir()
        kwargs = dict(
            metrics_dict={"proxy_ap50_95": 0.0},
            checkpoint_selection={},
            store_dir=store,
            cfg_path=DETECTOR_CONFIG_PATH,
            cm_path=CLASS_MAP_PATH,
        )
        p1 = publish_development_package(_make_checkpoint_manifest(), **kwargs)
        p2 = publish_development_package(_make_checkpoint_manifest(), **kwargs)
        assert p1 == p2

    def test_schema_version_mismatch_raises(self):
        """schema_version が一致しない manifest は from_dict で拒否される。"""
        from survivors.vision.world_detector_package import PackageManifest

        bad = {"schema_version": "unknown.v99", "formal_detector_eligible": False}
        with pytest.raises(ValueError, match="schema_version"):
            PackageManifest.from_dict(bad)


class TestPackageRestore:
    """restore_package が package 内 config を使って TrackedWorldStateV1 を返す。"""

    def _publish(self, tmp_path) -> pathlib.Path:
        """テスト用 development package を publish して manifest_path を返す。"""
        from survivors.vision.world_detector_package import publish_development_package

        store = tmp_path / "store"
        store.mkdir()
        return publish_development_package(
            _make_checkpoint_manifest(),
            metrics_dict={},
            checkpoint_selection={},
            store_dir=store,
            cfg_path=DETECTOR_CONFIG_PATH,
            cm_path=CLASS_MAP_PATH,
        )

    def test_restore_returns_tracked_world_state_v1(self, tmp_path):
        """restore_package は package 内 config を使って TrackedWorldStateV1 を返す。"""
        from survivors.vision.world_detector_package import restore_package
        from survivors.vision.entity_tracker import TrackedWorldStateV1

        pkg_path = self._publish(tmp_path)
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        state = restore_package(pkg_path, frame)

        assert isinstance(state, TrackedWorldStateV1)
        assert state.frame_index == 0
        assert isinstance(state.tracks, list)

    def test_restore_reject_formal_for_dev(self, tmp_path):
        """require_formal=True で development package をロードすると例外。"""
        from survivors.vision.world_detector_package import restore_package, FormalPackageRejectedError

        pkg_path = self._publish(tmp_path)
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        with pytest.raises(FormalPackageRejectedError):
            restore_package(pkg_path, frame, require_formal=True)

    def test_restore_detects_config_hash_mismatch(self, tmp_path):
        """package 内の config を改ざんすると PackageSchemaError。"""
        from survivors.vision.world_detector_package import restore_package, PackageSchemaError

        pkg_path = self._publish(tmp_path)
        # config を改ざん
        cfg_in_pkg = pkg_path.parent / "world_detector_v1.yaml"
        original = cfg_in_pkg.read_text(encoding="utf-8")
        cfg_in_pkg.write_text(original + "\n# tampered", encoding="utf-8")

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        with pytest.raises(PackageSchemaError, match="config_hash"):
            restore_package(pkg_path, frame)
