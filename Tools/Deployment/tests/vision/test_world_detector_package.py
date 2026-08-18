"""world_detector_package のテスト。

preflight 拒否（missing/stale feasibility verdict、split 混入）、
error_calibration/final_e2e_test session 拒否、
checkpoint selection、metric gate 境界値、
package publish / restore、schema 検証、formal_detector_eligible=false を検証する。
"""
from __future__ import annotations

import json
import pathlib
import tempfile
from dataclasses import dataclass

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
        sel.record(CheckpointRecord(1, 0.1, pathlib.Path("e1.pt")))
        sel.record(CheckpointRecord(2, 0.3, pathlib.Path("e2.pt")))
        sel.record(CheckpointRecord(3, 0.2, pathlib.Path("e3.pt")))
        assert sel.best is not None
        assert sel.best.epoch == 2
        assert sel.best.val_map50_95 == pytest.approx(0.3)

    def test_keep_top_k_enforced(self):
        """keep_top_k を超えた候補は削除される。"""
        from train_survivors_world_detector import CheckpointSelector, CheckpointRecord

        sel = CheckpointSelector(metric="val_map50_95", keep_top_k=2)
        for ep in range(5):
            sel.record(CheckpointRecord(ep, float(ep) * 0.1, pathlib.Path(f"e{ep}.pt")))
        assert len(sel._records) <= 2

    def test_to_dict_contains_selection_rules(self):
        """to_dict は metric / keep_top_k / best_epoch / best_val を含む。"""
        from train_survivors_world_detector import CheckpointSelector, CheckpointRecord

        sel = CheckpointSelector(metric="val_map50_95", keep_top_k=3)
        sel.record(CheckpointRecord(5, 0.42, pathlib.Path("e5.pt")))
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

    def test_resume_epoch_increments(self, tmp_path):
        """resume manifest から start_epoch を引き継ぐ。"""
        from train_survivors_world_detector import CheckpointSelector, CheckpointRecord

        manifest_data = {
            "checkpoint_selection": {"metric": "val_map50_95", "best_epoch": 10, "best_val_map50_95": 0.5}
        }
        (tmp_path / "manifest.json").write_text(json.dumps(manifest_data), encoding="utf-8")

        # resume 後は epoch 11 から開始するはずだが、CheckpointSelector はステートレス
        sel = CheckpointSelector(metric="val_map50_95", keep_top_k=3)
        sel.record(CheckpointRecord(11, 0.6, tmp_path / "e11.pt"))
        assert sel.best.epoch == 11


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
                {"session_id": "s1"}, {"session_id": "s2"},
                {"session_id": "s3"},  # 3 件、4 件必要
            ],
        }
        result = check_performance_gate(
            m, self._gate_cfg(), self._class_name_by_id(),
            slice_annotations=slice_annotations,
        )
        heavy = next(r for r in result.slice_results if r.slice_name == "heavy_effect")
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
        # slice に十分なインスタンス（recall は 0 だが min 設定が 0 のケース）
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

    def _dummy_checkpoint_manifest(self):
        """ダミー CheckpointManifest を生成する。"""
        from survivors.vision.world_detector import CheckpointManifest
        return CheckpointManifest(
            model_hash="a" * 64,
            data_hash="b" * 64,
            config_hash="c" * 64,
            build_hash="d" * 64,
            class_map_hash="e" * 64,
            formal_detector_eligible=False,
        )

    def test_publish_creates_manifest_json(self, tmp_path):
        """publish_development_package が manifest.json を作成する。"""
        from survivors.vision.world_detector_package import publish_development_package

        store = tmp_path / "store"
        store.mkdir()
        pkg_path = publish_development_package(
            self._dummy_checkpoint_manifest(),
            metrics_dict={"proxy_ap50_95": 0.0},
            checkpoint_selection={"metric": "val_map50_95", "keep_top_k": 3},
            store_dir=store,
        )
        assert pkg_path.exists()
        assert pkg_path.name == "manifest.json"

    def test_published_package_is_formal_ineligible(self, tmp_path):
        """開発 package の formal_detector_eligible は常に False。"""
        from survivors.vision.world_detector_package import publish_development_package, PackageManifest

        store = tmp_path / "store"
        store.mkdir()
        pkg_path = publish_development_package(
            self._dummy_checkpoint_manifest(),
            metrics_dict={},
            checkpoint_selection={},
            store_dir=store,
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
            self._dummy_checkpoint_manifest(),
            metrics_dict={},
            checkpoint_selection={},
            store_dir=store,
        )
        raw = json.loads(pkg_path.read_text(encoding="utf-8"))
        pm = PackageManifest.from_dict(raw)
        with pytest.raises(FormalPackageRejectedError):
            pm.assert_formal_eligible()

    def test_formal_publish_requires_validation_pass(self, tmp_path):
        """formal publish は validation_passed=False のとき ValueError。"""
        from survivors.vision.world_detector_package import publish_formal_package
        from survivors.vision.world_detector import CheckpointManifest

        store = tmp_path / "store"
        store.mkdir()
        cm = CheckpointManifest(
            model_hash="a" * 64,
            data_hash="b" * 64,
            config_hash="c" * 64,
            build_hash="d" * 64,
            class_map_hash="e" * 64,
            formal_detector_eligible=True,
        )
        with pytest.raises(ValueError, match="validation PASS"):
            publish_formal_package(
                cm, {}, {}, store,
                validation_passed=False,
            )

    def test_formal_publish_requires_formal_checkpoint(self, tmp_path):
        """formal publish には formal_detector_eligible=True の checkpoint が必要。"""
        from survivors.vision.world_detector_package import publish_formal_package

        store = tmp_path / "store"
        store.mkdir()
        with pytest.raises(ValueError, match="formal_detector_eligible"):
            publish_formal_package(
                self._dummy_checkpoint_manifest(),  # formal_detector_eligible=False
                {}, {}, store,
                validation_passed=True,
            )

    def test_idempotent_publish(self, tmp_path):
        """同じ内容を 2 回 publish しても同じ path が返る。"""
        from survivors.vision.world_detector_package import publish_development_package

        store = tmp_path / "store"
        store.mkdir()
        args = (
            self._dummy_checkpoint_manifest(),
            {"proxy_ap50_95": 0.0},
            {},
        )
        p1 = publish_development_package(*args, store_dir=store)
        p2 = publish_development_package(*args, store_dir=store)
        assert p1 == p2

    def test_schema_version_mismatch_raises(self, tmp_path):
        """schema_version が一致しない manifest は from_dict で拒否される。"""
        from survivors.vision.world_detector_package import PackageManifest

        bad = {"schema_version": "unknown.v99", "formal_detector_eligible": False}
        with pytest.raises(ValueError, match="schema_version"):
            PackageManifest.from_dict(bad)


class TestPackageRestore:
    """restore_package が TrackedWorldStateV1 を返すことを確認する。"""

    def _publish(self, tmp_path):
        """テスト用 development package を publish して path を返す。"""
        from survivors.vision.world_detector_package import publish_development_package
        from survivors.vision.world_detector import CheckpointManifest

        store = tmp_path / "store"
        store.mkdir()
        cm = CheckpointManifest(
            model_hash="a" * 64,
            data_hash="b" * 64,
            config_hash="c" * 64,
            build_hash="d" * 64,
            class_map_hash="e" * 64,
            formal_detector_eligible=False,
        )
        return publish_development_package(cm, {}, {}, store_dir=store)

    def test_restore_returns_tracked_world_state_v1(self, tmp_path):
        """restore_package は TrackedWorldStateV1 を返す。"""
        from survivors.vision.entity_tracker import TrackedWorldStateV1

        pkg_path = self._publish(tmp_path)
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        state = _restore_with_wt_configs(pkg_path, frame)

        assert isinstance(state, TrackedWorldStateV1)
        assert state.frame_index == 0
        assert isinstance(state.tracks, list)

    def test_restore_reject_formal_for_dev(self, tmp_path):
        """require_formal=True で development package をロードすると例外。"""
        from survivors.vision.world_detector_package import restore_package, FormalPackageRejectedError

        pkg_path = self._publish(tmp_path)
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        with pytest.raises(FormalPackageRejectedError):
            _restore_with_wt_configs(pkg_path, frame, require_formal=True)


def _restore_with_wt_configs(
    manifest_path: pathlib.Path,
    frame_bgr: np.ndarray,
    *,
    require_formal: bool = False,
) -> "TrackedWorldStateV1":
    """worktree の configs を使って restore_package を呼ぶテスト専用ヘルパー。

    world_detector_package.restore_package の cfg/cm パス解決を
    worktree configs ディレクトリに差し替える。
    """
    import yaml
    from survivors.vision.world_detector import WorldDetector
    from survivors.vision.entity_tracker import EntityTracker, TrackedWorldStateV1
    from survivors.vision.world_dataset import load_class_map
    from survivors.vision.world_detector_package import (
        PackageManifest, FormalPackageRejectedError, PackageSchemaError, _compute_contract_hash
    )

    wt_configs = pathlib.Path(__file__).parents[2] / "configs"
    cfg_path = wt_configs / "world_detector_v1.yaml"
    cm_path = wt_configs / "world_class_map_v1.yaml"

    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    pkg_manifest = PackageManifest.from_dict(raw)

    if require_formal:
        pkg_manifest.assert_formal_eligible()

    current_contract_hash = _compute_contract_hash()
    if pkg_manifest.contract_hash != current_contract_hash:
        raise PackageSchemaError(
            f"contract_hash 不一致: {pkg_manifest.contract_hash!r} vs {current_contract_hash!r}"
        )

    with cfg_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    detector = WorldDetector.from_config(cfg, cm_path)
    result = detector.infer(frame_bgr, score_threshold=0.5)

    tracker_cfg = cfg.get("tracker", {})
    cm = load_class_map(cm_path)
    max_age_raw: dict = tracker_cfg.get("max_age_by_class", {})
    max_age_by_class: dict[int, int] = {}
    for name, age in max_age_raw.items():
        try:
            max_age_by_class[cm.name_to_id(name)] = age
        except KeyError:
            pass

    tracker = EntityTracker(
        max_age_by_class=max_age_by_class,
        max_match_cost=tracker_cfg.get("max_match_cost", 0.7),
        velocity_ema_alpha=tracker_cfg.get("velocity_ema_alpha", 0.6),
        confidence_decay_per_frame=tracker_cfg.get("confidence_decay_per_frame", 0.9),
    )
    state = tracker.update(result, frame_index=0, timestamp_ns=0)
    return TrackedWorldStateV1.from_state(state, frame_index=0, timestamp_ns=0)
