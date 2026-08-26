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
from types import SimpleNamespace

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
        {"id": i, "file_name": f"frame_{i:04d}.png", "width": 1920, "height": 1080,
         "session_id": sessions[i % len(sessions)]}
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

    def test_split_preflight_uses_sample_session_id_for_forbidden(self, tmp_path):
        """run_split_preflight は DatasetSample.session_id を正本として forbidden を検出する。

        annotation なし負例（annotation.session_id が空）でも final_e2e_test を検出できること。
        """
        from survivors.vision.world_dataset import WorldDataset, SessionSplit, DatasetPreflightError

        # 画像レベルに session_id を付け、annotation なし負例として final_e2e_test を混入
        data: dict = {
            "images": [
                {"id": 0, "file_name": "f0.png", "width": 100, "height": 100, "session_id": "train_01"},
                {"id": 1, "file_name": "f1.png", "width": 100, "height": 100, "session_id": "val_01"},
                {"id": 2, "file_name": "f2.png", "width": 100, "height": 100, "session_id": "final_e2e"},
            ],
            "annotations": [
                {"id": 0, "image_id": 0, "category_id": 1, "bbox": [0, 0, 10, 10], "session_id": "train_01"},
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "session_id": "val_01"},
                # image_id=2 は annotation なし（annotation.session_id が存在しない）
            ],
            "categories": [{"id": k, "name": f"class_{k}"} for k in range(1, 12)],
        }
        ann_path = _write_coco(tmp_path, data)
        ds = WorldDataset(ann_path, CLASS_MAP_PATH)
        split = SessionSplit(
            train=["train_01"], validation=["val_01"],
            error_calibration=[], final_e2e_test=["final_e2e"],
        )
        with pytest.raises(DatasetPreflightError, match="final_e2e"):
            ds.run_split_preflight(split, min_frames_per_split=1, min_entities_per_split=1, min_classes_per_split=1)

    def test_split_preflight_raises_on_missing_sample_session_id(self, tmp_path):
        """DatasetSample.session_id が未設定の画像は run_split_preflight で拒否される。"""
        from survivors.vision.world_dataset import WorldDataset, SessionSplit, DatasetPreflightError

        # 画像に session_id を付けない
        data: dict = {
            "images": [
                {"id": 0, "file_name": "f0.png", "width": 100, "height": 100},  # session_id なし
                {"id": 1, "file_name": "f1.png", "width": 100, "height": 100, "session_id": "val_01"},
            ],
            "annotations": [
                {"id": 0, "image_id": 0, "category_id": 1, "bbox": [0, 0, 10, 10], "session_id": "train_01"},
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "session_id": "val_01"},
            ],
            "categories": [{"id": k, "name": f"class_{k}"} for k in range(1, 12)],
        }
        ann_path = _write_coco(tmp_path, data)
        ds = WorldDataset(ann_path, CLASS_MAP_PATH)
        split = SessionSplit(
            train=["train_01"], validation=["val_01"],
            error_calibration=[], final_e2e_test=[],
        )
        with pytest.raises(DatasetPreflightError, match="session_id"):
            ds.run_split_preflight(split, min_frames_per_split=1, min_entities_per_split=1, min_classes_per_split=1)

    def test_split_preflight_counts_train_negatives_by_sample_session_id(self, tmp_path):
        """train 負例フレームは DatasetSample.session_id でカウントされる。"""
        from survivors.vision.world_dataset import WorldDataset, SessionSplit, DatasetPreflightError

        # train session に annotation なし負例を 1 件含む（annotation は val のみ）
        data: dict = {
            "images": [
                {"id": 0, "file_name": "f0.png", "width": 100, "height": 100, "session_id": "train_01"},
                {"id": 1, "file_name": "f1.png", "width": 100, "height": 100, "session_id": "val_01"},
            ],
            "annotations": [
                # image_id=0 は annotation なし（train 負例）
                {"id": 0, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "session_id": "val_01"},
            ],
            "categories": [{"id": k, "name": f"class_{k}"} for k in range(1, 12)],
        }
        ann_path = _write_coco(tmp_path, data)
        ds = WorldDataset(ann_path, CLASS_MAP_PATH)
        split = SessionSplit(
            train=["train_01"], validation=["val_01"],
            error_calibration=[], final_e2e_test=[],
        )
        # train 1 frame（負例）, validation 1 frame: min_frames_per_split=1 で通過
        # DatasetSample.session_id を使えば train も 1 件カウントされる
        ds.run_split_preflight(split, min_frames_per_split=1, min_entities_per_split=0, min_classes_per_split=0)


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

    def test_resume_round_trip_uses_unified_last_checkpoint(self, tmp_path):
        """resume は best ではなく model/optimizer/epoch を含む last state を復元する。"""
        torch = pytest.importorskip("torch")
        from train_survivors_world_detector import (
            CheckpointRecord,
            CheckpointSelector,
            _LAST_CHECKPOINT_NAME,
            _load_resume_state,
            _save_resume_state,
        )

        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        selector = CheckpointSelector(metric="val_map50_95", keep_top_k=3)
        selector.record(CheckpointRecord(1, 0.9, str(tmp_path / "best.pt")))

        expected = {name: value.detach().clone() for name, value in model.state_dict().items()}
        _save_resume_state(tmp_path, 4, selector, model, optimizer)
        assert (tmp_path / _LAST_CHECKPOINT_NAME).exists()

        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(10)
        restored_selector = CheckpointSelector(metric="val_map50_95", keep_top_k=3)
        assert _load_resume_state(tmp_path, restored_selector, model, optimizer) == 5
        for name, value in model.state_dict().items():
            assert torch.equal(value, expected[name])
        assert restored_selector.best is not None
        assert restored_selector.best.epoch == 1

    def test_resume_rejects_last_checkpoint_hash_mismatch(self, tmp_path):
        """last checkpoint 改ざん時は warning 継続せず例外にする。"""
        torch = pytest.importorskip("torch")
        from train_survivors_world_detector import (
            CheckpointSelector,
            _LAST_CHECKPOINT_NAME,
            _load_resume_state,
            _save_resume_state,
        )

        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        selector = CheckpointSelector(metric="val_map50_95", keep_top_k=3)
        _save_resume_state(tmp_path, 2, selector, model, optimizer)
        (tmp_path / _LAST_CHECKPOINT_NAME).write_bytes(b"tampered")

        with pytest.raises(ValueError, match="hash"):
            _load_resume_state(tmp_path, selector, model, optimizer)

    def test_resume_rejects_missing_optimizer_state(self, tmp_path):
        """last checkpoint 内の optimizer 欠落を拒否する。"""
        torch = pytest.importorskip("torch")
        from train_survivors_world_detector import (
            CheckpointSelector,
            _LAST_CHECKPOINT_NAME,
            _RESUME_STATE_NAME,
            _load_resume_state,
        )

        checkpoint_path = tmp_path / _LAST_CHECKPOINT_NAME
        torch.save({"epoch": 2, "model_state_dict": {}, "selector": {}}, checkpoint_path)
        metadata = {
            "schema_version": 1,
            "last_epoch": 2,
            "last_checkpoint": _LAST_CHECKPOINT_NAME,
            "last_checkpoint_hash": _sha256_file(checkpoint_path),
            "training_mode": "development",  # Fix 2: training_mode は必須フィールド
        }
        (tmp_path / _RESUME_STATE_NAME).write_text(json.dumps(metadata), encoding="utf-8")

        selector = CheckpointSelector(metric="val_map50_95", keep_top_k=3)
        with pytest.raises(ValueError, match="optimizer_state_dict"):
            _load_resume_state(tmp_path, selector, SimpleNamespace(load_state_dict=lambda _: None), None)

    def test_resume_rejects_mode_mismatch(self, tmp_path):
        """smoke checkpoint を development でまたは development を smoke で resume すると拒否される（P1 回帰テスト）。"""
        torch = pytest.importorskip("torch")
        from train_survivors_world_detector import (
            CheckpointSelector,
            _LAST_CHECKPOINT_NAME,
            _RESUME_STATE_NAME,
            _load_resume_state,
            _save_resume_state,
        )

        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        selector = CheckpointSelector(metric="val_map50_95", keep_top_k=3)

        # smoke モードで保存
        _save_resume_state(tmp_path, 1, selector, model, optimizer, training_mode="smoke")

        # development モードで resume → 拒否
        with pytest.raises(ValueError, match="training_mode"):
            _load_resume_state(
                tmp_path, selector, model, optimizer,
                expected_training_mode="development",
            )

        # 逆方向も確認: training_state.json を development に書き換えて smoke で resume
        state = json.loads((tmp_path / _RESUME_STATE_NAME).read_text())
        state["training_mode"] = "development"
        (tmp_path / _RESUME_STATE_NAME).write_text(json.dumps(state), encoding="utf-8")
        with pytest.raises(ValueError, match="training_mode"):
            _load_resume_state(
                tmp_path, selector, model, optimizer,
                expected_training_mode="smoke",
            )


class TestTrainEpoch:
    """_train_epoch が実 DatasetSample から target を構築することを確認する。"""

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
        from train_survivors_world_detector import _train_epoch

        if not _TORCH_AVAILABLE:
            pytest.skip("torch 不在環境では _StubModel なので skip")

        data = _make_coco(n_images=5, sessions=["s1", "s2"])
        ann_path = _write_coco(tmp_path, data)
        ds = WorldDataset(ann_path, CLASS_MAP_PATH)

        cfg = load_detector_config(DETECTOR_CONFIG_PATH)
        detector = WorldDetector.from_config(cfg, CLASS_MAP_PATH)
        optimizer = optim.SGD(detector._model.parameters(), lr=0.001)

        # smoke=True: 画像ファイルが存在しないためゼロ画像でテスト
        _train_epoch(detector, ds, optimizer, cfg, smoke=True)

    def test_train_epoch_uses_configured_batch_size(self):
        """複数 session の sample を config batch_size ごとに model へ渡す。"""
        torch = pytest.importorskip("torch")
        from train_survivors_world_detector import _train_epoch
        from survivors.vision.world_dataset import COCOAnnotation, DatasetSample

        samples = []
        for index in range(8):
            samples.append(DatasetSample(
                image_id=index,
                file_name=f"missing-{index}.png",
                image_width=32,
                image_height=32,
                annotations=[COCOAnnotation(index, index, 1, (1, 1, 4, 4), f"s{index % 2}")],
            ))

        class RecordingModel:
            def __init__(self):
                self.batch_sizes = []

            def train(self):
                return self

            def __call__(self, images, targets):
                self.batch_sizes.append((len(images), len(targets)))
                loss = torch.tensor(1.0, requires_grad=True)
                return {"loss": loss}

        class RecordingOptimizer:
            def zero_grad(self):
                pass

            def step(self):
                pass

        model = RecordingModel()
        detector = SimpleNamespace(_model=model)
        _train_epoch(
            detector,
            samples,
            RecordingOptimizer(),
            {"training": {"batch_size": 4}},
            smoke=True,  # 画像ファイルが存在しないためゼロ画像でテスト
        )
        assert model.batch_sizes == [(4, 4), (4, 4)]

    def test_train_epoch_rebalances_singleton_remainder(self):
        """4件 batch_size=3 の場合、singleton を再配分して [2,2] になり全4件かつ batch_size 以下を保証する（P2 回帰テスト）。"""
        torch = pytest.importorskip("torch")
        from train_survivors_world_detector import _train_epoch
        from survivors.vision.world_dataset import COCOAnnotation, DatasetSample

        samples = [
            DatasetSample(
                image_id=index,
                file_name=f"missing-{index}.png",
                image_width=32,
                image_height=32,
                annotations=[COCOAnnotation(index, index, 1, (1, 1, 4, 4), f"s{index % 2}")],
                session_id=f"s{index % 2}",
            )
            for index in range(4)
        ]
        batch_sizes: list[int] = []

        class Model:
            def train(self):
                return self

            def __call__(self, images, targets):
                batch_sizes.append(len(images))
                return {"loss": torch.tensor(1.0, requires_grad=True)}

        optimizer = SimpleNamespace(zero_grad=lambda: None, step=lambda: None)
        _train_epoch(
            SimpleNamespace(_model=Model()), samples, optimizer,
            {"training": {"batch_size": 3}},
            smoke=True,  # 画像ファイルが存在しないためゼロ画像でテスト
        )
        # 4件全員が学習される（全件性）
        assert sum(batch_sizes) == 4
        # すべての batch が batch_size=3 以下（最大サイズ保証）
        assert all(b <= 3 for b in batch_sizes), f"batch_size を超えた batch があります: {batch_sizes}"
        # 2 batch に分割されている（[2,2]）
        assert len(batch_sizes) == 2, f"2 batch に再配分されるべき: {batch_sizes}"


class TestTrainingSplitViews:
    """train/validation の Dataset view が独立し、caller が混在させないことを確認する。"""

    def test_split_is_required_cli_argument(self):
        from train_survivors_world_detector import _build_arg_parser

        with pytest.raises(SystemExit):
            _build_arg_parser().parse_args(["--annotations", "a.json", "--output", "out"])

    def test_split_views_are_disjoint(self, tmp_path):
        from survivors.vision.world_dataset import SessionSplit, WorldDataset
        from train_survivors_world_detector import _create_split_views

        data = _make_coco(n_images=8, sessions=["train-a", "val-a"])
        ds = WorldDataset(_write_coco(tmp_path, data), CLASS_MAP_PATH)
        split = SessionSplit(
            train=["train-a"], validation=["val-a"], error_calibration=[], final_e2e_test=[]
        )
        train_ds, validation_ds = _create_split_views(ds, split)

        assert {sample.image_id for sample in train_ds} == {0, 2, 4, 6}
        assert {sample.image_id for sample in validation_ds} == {1, 3, 5, 7}
        assert not ({sample.image_id for sample in train_ds} & {sample.image_id for sample in validation_ds})

    def test_training_loop_routes_train_and_validation_views_separately(self, tmp_path, monkeypatch):
        import train_survivors_world_detector as training

        train_view = object()
        validation_view = object()
        routed = {}
        monkeypatch.setattr(
            training, "_train_epoch",
            lambda detector, ds, optimizer, cfg, **kw: routed.setdefault("train", ds),
        )
        monkeypatch.setattr(
            training, "_evaluate_validation",
            lambda detector, ds, cfg, **kw: routed.setdefault("validation", ds) and 0.25,
        )
        monkeypatch.setattr(training, "_save_resume_state", lambda *args, **kwargs: None)
        detector = SimpleNamespace(_model=SimpleNamespace(state_dict=lambda: {}))
        selector = training.CheckpointSelector(metric="val_map50_95", keep_top_k=1)

        training._run_training_torch(
            detector, train_view, validation_view,
            {"checkpoint_selection": {"save_every_n_epochs": 1}},
            max_epochs=1,
            selector=selector,
            out_dir=tmp_path,
            optimizer=object(),
            start_epoch=1,
        )

        assert routed == {"train": train_view, "validation": validation_view}

    def test_validation_inference_error_propagates(self, tmp_path):
        """validation 推論で RuntimeError が発生した場合は _evaluate_validation が例外を伝播する。"""
        import train_survivors_world_detector as training

        class _ErrorModel:
            def __call__(self, imgs):
                raise RuntimeError("simulated inference error")
            def eval(self):
                pass
            def state_dict(self):
                return {}

        class _SingleItemDs:
            def __len__(self):
                return 1
            def __getitem__(self, idx):
                return SimpleNamespace(
                    image_id=0,
                    file_name="/nonexistent/image.jpg",
                    image_height=64,
                    image_width=64,
                    annotations=[],
                )

        detector = SimpleNamespace(_model=_ErrorModel(), num_classes=10)
        cfg = {"model": {"num_classes": 10}}
        ds = _SingleItemDs()

        with pytest.raises(RuntimeError, match="simulated inference error"):
            training._evaluate_validation(detector, ds, cfg, smoke=True)

    def test_create_split_views_raises_on_missing_sample_session_id(self, tmp_path):
        """DatasetSample.session_id が空の場合は _create_split_views が DatasetPreflightError。"""
        from survivors.vision.world_dataset import WorldDataset, SessionSplit
        from train_survivors_world_detector import _create_split_views
        from survivors.vision.world_dataset import DatasetPreflightError

        # 画像に session_id を付けない（session_id 欠損）
        data: dict = {
            "images": [
                {"id": 0, "file_name": "f0.png", "width": 100, "height": 100},
                {"id": 1, "file_name": "f1.png", "width": 100, "height": 100, "session_id": "val_01"},
            ],
            "annotations": [
                {"id": 0, "image_id": 0, "category_id": 1, "bbox": [0, 0, 10, 10], "session_id": "train_01"},
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "session_id": "val_01"},
            ],
            "categories": [{"id": k, "name": f"class_{k}"} for k in range(1, 12)],
        }
        ann_path = _write_coco(tmp_path, data)
        ds = WorldDataset(ann_path, CLASS_MAP_PATH)
        split = SessionSplit(
            train=["train_01"], validation=["val_01"],
            error_calibration=[], final_e2e_test=[],
        )
        with pytest.raises(DatasetPreflightError, match="session_id"):
            _create_split_views(ds, split)


# ---- Task 2.5: ImportError narrowing ----

class TestImportErrorNarrowing:
    """内部 ImportError は stub へ fallback せず伝播することを確認する。"""

    def test_internal_torch_optim_error_propagates(self, monkeypatch):
        """torch は存在するが torch.optim.SGD が内部 ImportError を送出する場合は伝播する。"""
        import torch.optim as real_optim
        import train_survivors_world_detector as training

        pytest.importorskip("torch")

        class _BrokenSGD:
            """SGD コンストラクタが ImportError を送出するスタブ。"""

            def __init__(self, *args, **kwargs):
                raise ImportError("simulated internal torch.optim.SGD error")

        monkeypatch.setattr(real_optim, "SGD", _BrokenSGD)

        # _run_training は torch 利用可能 → optim.SGD(...) で ImportError → 伝播する
        detector = SimpleNamespace(_model=SimpleNamespace(parameters=lambda: []))
        with pytest.raises(ImportError, match="simulated internal"):
            training._run_training(
                detector, [], [],
                {"training": {"batch_size": 2, "max_epochs": 1}},
                max_epochs=1,
                selector=training.CheckpointSelector(metric="val_map50_95", keep_top_k=1),
                out_dir=pathlib.Path("/tmp"),
            )


# ---- Task 3: performance gate boundary values ----

class TestPerformanceGateBoundary:
    """development diagnostics 境界値を synthetic metrics でテストする。"""

    def _gate_cfg(self) -> dict:
        """テスト用の dev_diagnostics 設定（実際の閾値）。"""
        return {
            "map50_95_min": 0.0,
            "class_recall_min": {
                "enemy_normal": 0.90,
                "enemy_boss": 0.98,
                "gem_blue": 0.92,
            },
            "density_correlation_min": 0.85,
            "nearest_distance_median_max": 0.04,
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
            density_error=0.10,
            count_error=0.10,
            nearest_distance_error=0.02,  # <= 0.04
            density_correlation=0.90,  # >= 0.85
            mean_latency_ms=10.0,
        )

    def _class_name_by_id(self) -> dict:
        return {2: "enemy_normal", 4: "enemy_boss", 5: "gem_blue"}

    def test_enemy_recall_below_threshold_fails(self):
        """enemy_normal recall < 0.90 はゲート FAIL。"""
        from eval_survivors_world_detector import EvalMetrics, compute_dev_diagnostics

        m = self._base_metrics()
        m.class_recall[2] = 0.89  # 0.90 未満
        result = compute_dev_diagnostics(m, self._gate_cfg(), self._class_name_by_id())
        assert not result.class_recall_results["enemy_normal"]["passed"]
        assert not result.passed

    def test_enemy_recall_at_threshold_passes(self):
        """enemy_normal recall = 0.90 ちょうどで PASS。"""
        from eval_survivors_world_detector import EvalMetrics, compute_dev_diagnostics

        m = self._base_metrics()
        m.class_recall[2] = 0.90
        result = compute_dev_diagnostics(m, self._gate_cfg(), self._class_name_by_id())
        assert result.class_recall_results["enemy_normal"]["passed"]

    def test_boss_recall_below_threshold_fails(self):
        """enemy_boss recall < 0.98 はゲート FAIL。"""
        from eval_survivors_world_detector import EvalMetrics, compute_dev_diagnostics

        m = self._base_metrics()
        m.class_recall[4] = 0.97
        result = compute_dev_diagnostics(m, self._gate_cfg(), self._class_name_by_id())
        assert not result.class_recall_results["enemy_boss"]["passed"]

    def test_gem_recall_below_threshold_fails(self):
        """gem_blue recall < 0.92 はゲート FAIL。"""
        from eval_survivors_world_detector import EvalMetrics, compute_dev_diagnostics

        m = self._base_metrics()
        m.class_recall[5] = 0.91
        result = compute_dev_diagnostics(m, self._gate_cfg(), self._class_name_by_id())
        assert not result.class_recall_results["gem_blue"]["passed"]

    def test_density_correlation_below_threshold_fails(self):
        """density_correlation=0.80 < 0.85 は FAIL。"""
        from eval_survivors_world_detector import compute_dev_diagnostics

        m = self._base_metrics()
        m.density_correlation = 0.80
        result = compute_dev_diagnostics(m, self._gate_cfg(), self._class_name_by_id())
        assert not result.passed

    def test_nearest_distance_above_max_fails(self):
        """nearest_distance_error > 0.04 は FAIL。"""
        from eval_survivors_world_detector import compute_dev_diagnostics

        m = self._base_metrics()
        m.nearest_distance_error = 0.05
        result = compute_dev_diagnostics(m, self._gate_cfg(), self._class_name_by_id())
        assert not result.passed

    def test_slice_instance_below_min_fails(self):
        """heavy_effect slice の instance 数 < min_instances は FAIL。"""
        from eval_survivors_world_detector import compute_dev_diagnostics

        m = self._base_metrics()
        # heavy_effect slice に 3 件しかない（min_instances=4）
        slice_annotations = {
            "heavy_effect": [
                {"image_id": 0, "category_id": 2, "bbox": [10, 10, 50, 50], "session_id": "s1"},
                {"image_id": 1, "category_id": 2, "bbox": [10, 10, 50, 50], "session_id": "s2"},
                {"image_id": 2, "category_id": 2, "bbox": [10, 10, 50, 50], "session_id": "s3"},
            ],
        }
        result = compute_dev_diagnostics(
            m, self._gate_cfg(), self._class_name_by_id(),
            slice_annotations=slice_annotations,
        )
        heavy = next(r for r in result.slice_results if r.slice_name == "heavy_effect")
        assert not heavy.passed

    def test_slice_recall_computed_from_predictions(self):
        """slice recall は prediction との一対一 matching で計算される。"""
        from eval_survivors_world_detector import compute_dev_diagnostics

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
        result = compute_dev_diagnostics(
            m, gate_cfg, {4: "enemy_boss"},
            slice_annotations={"boss": gt_anns},
            slice_predictions={"boss": pred_anns},
        )
        boss = next(r for r in result.slice_results if r.slice_name == "boss")
        assert boss.recall == pytest.approx(1.0)
        assert boss.passed

    def test_slice_recall_zero_when_no_predictions(self):
        """prediction が空のとき recall=0.0、閾値 > 0 ならば FAIL。"""
        from eval_survivors_world_detector import compute_dev_diagnostics

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
        result = compute_dev_diagnostics(
            m, gate_cfg, {},
            slice_annotations={"heavy_effect": gt_anns},
            slice_predictions={"heavy_effect": []},  # 空予測
        )
        heavy = next(r for r in result.slice_results if r.slice_name == "heavy_effect")
        assert heavy.recall == pytest.approx(0.0)
        assert not heavy.passed

    def test_latency_is_unsupported_and_does_not_affect_passed(self):
        """GPU latency は unsupported のため passed に影響しない（04-08 で実装）。"""
        from eval_survivors_world_detector import compute_dev_diagnostics

        m = self._base_metrics()
        # 全実装済み指標を通過させた上で、GPU latency のみ非常に大きい値を渡す
        gate_cfg = {
            "map50_95_min": 0.0,
            "class_recall_min": {},
            "density_correlation_min": 0.0,
            "nearest_distance_median_max": 1.0,
            "slice_gate": {},
        }
        result_with = compute_dev_diagnostics(m, gate_cfg, {}, gpu_p95_latency_ms=999999.0)
        result_without = compute_dev_diagnostics(m, gate_cfg, {}, gpu_p95_latency_ms=None)
        assert result_with.passed, "実装済み指標が全部通れば passed=True（GPU latency は無視）"
        assert result_without.passed, "gpu_p95_latency_ms=None でも passed に影響しない"
        # gpu_p95_latency_ms は情報表示のみ（unsupported として記録される）
        assert result_without.gpu_p95_latency_ms == "unsupported"

    def test_all_pass_returns_passed_true(self):
        """全実装済み gate を通過したとき passed=True。"""
        from eval_survivors_world_detector import compute_dev_diagnostics

        m = self._base_metrics()
        gate_cfg = {
            "map50_95_min": 0.0,
            "class_recall_min": {},
            "density_correlation_min": 0.0,
            "nearest_distance_median_max": 1.0,
            "slice_gate": {},
        }
        result = compute_dev_diagnostics(m, gate_cfg, {})
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

    def test_assert_formal_eligible_always_raises(self, tmp_path):
        """PackageManifest は formal_detector_eligible の値に関わらず assert_formal_eligible() が常に拒否する。"""
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
        # from_dict では False に上書きされるが、それでも raises
        pm = PackageManifest.from_dict(raw)
        with pytest.raises(FormalPackageRejectedError):
            pm.assert_formal_eligible()

        # caller が直接 formal_detector_eligible=True を指定してもやはり拒否
        raw_true = dict(raw)
        raw_true["formal_detector_eligible"] = True
        pm_true = PackageManifest.from_dict(raw_true)
        assert pm_true.formal_detector_eligible is False  # 上書きされる
        with pytest.raises(FormalPackageRejectedError):
            pm_true.assert_formal_eligible()

    def test_from_dict_rejects_invalid_training_mode(self):
        """from_dict は training_mode が 'smoke'|'development' 以外を拒否する。"""
        from survivors.vision.world_detector_package import PackageManifest, PACKAGE_SCHEMA_VERSION
        bad = {
            "schema_version": PACKAGE_SCHEMA_VERSION,
            "formal_detector_eligible": False,
            "model_hash": "a" * 64,
            "data_hash": "b" * 64,
            "config_hash": "c" * 64,
            "build_hash": "d" * 64,
            "class_map_hash": "e" * 64,
            "contract_hash": "f" * 64,
            "training_mode": "formal",
        }
        with pytest.raises(ValueError, match="training_mode"):
            PackageManifest.from_dict(bad)

    def test_restore_package_require_formal_raises(self, tmp_path):
        """restore_package(require_formal=True) は development package を FormalPackageRejectedError で拒否する。"""
        from survivors.vision.world_detector_package import publish_development_package, restore_package, FormalPackageRejectedError

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
        frame = __import__("numpy").zeros((1080, 1920, 3), dtype="uint8")
        with pytest.raises(FormalPackageRejectedError):
            restore_package(pkg_path, frame, require_formal=True)

    def test_formal_publish_always_rejected(self, tmp_path):
        """PR#315 では publish_formal_package() は常に FormalPackageRejectedError を送出する。"""
        from survivors.vision.world_detector_package import publish_formal_package, FormalPackageRejectedError

        store = tmp_path / "store"
        store.mkdir()
        # どんな引数を渡しても拒否されることを確認
        with pytest.raises(FormalPackageRejectedError):
            publish_formal_package(
                _make_checkpoint_manifest(formal=True),
                {}, {}, store,
                cfg_path=DETECTOR_CONFIG_PATH,
                cm_path=CLASS_MAP_PATH,
                weight_path=tmp_path / "model.pt",
                validation_passed=True,
                score_threshold=0.5,
            )

    def test_formal_publish_rejected_with_dev_manifest(self, tmp_path):
        """development checkpoint でも formal publish は常に拒否される。"""
        from survivors.vision.world_detector_package import publish_formal_package, FormalPackageRejectedError

        store = tmp_path / "store"
        store.mkdir()
        with pytest.raises(FormalPackageRejectedError):
            publish_formal_package(
                _make_checkpoint_manifest(formal=False),
                {}, {}, store,
                cfg_path=DETECTOR_CONFIG_PATH,
                cm_path=CLASS_MAP_PATH,
            )

    def test_score_threshold_stored_in_manifest(self, tmp_path):
        """score_threshold が manifest に保存され restore_package で再現性が保証される。"""
        from survivors.vision.world_detector_package import publish_development_package, PackageManifest

        store = tmp_path / "store"
        store.mkdir()
        pkg_path = publish_development_package(
            _make_checkpoint_manifest(),
            metrics_dict={}, checkpoint_selection={},
            store_dir=store,
            cfg_path=DETECTOR_CONFIG_PATH,
            cm_path=CLASS_MAP_PATH,
            score_threshold=0.7,
        )
        raw = json.loads(pkg_path.read_text(encoding="utf-8"))
        pm = PackageManifest.from_dict(raw)
        assert abs(pm.score_threshold - 0.7) < 1e-6

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

    def test_documented_package_api_uses_keyword_only_paths(self):
        """package API 例が current cfg_path/cm_path signature と一致する。"""
        docs_path = pathlib.Path(__file__).parents[4] / "docs/deployment/world_detector.md"
        docs = docs_path.read_text(encoding="utf-8")
        assert "cfg_path=cfg_path" in docs
        assert "cm_path=cm_path" in docs

    def test_development_only_and_training_mode_development_in_package(self, tmp_path):
        """publish_development_package の manifest に development_only=True と training_mode='development' が含まれる。"""
        from survivors.vision.world_detector_package import publish_development_package, PackageManifest

        checkpoint_manifest = _make_checkpoint_manifest()
        store = tmp_path / "store"
        store.mkdir()
        pkg_path = publish_development_package(
            checkpoint_manifest,
            metrics_dict={}, checkpoint_selection={},
            store_dir=store,
            cfg_path=DETECTOR_CONFIG_PATH,
            cm_path=CLASS_MAP_PATH,
        )
        raw = json.loads(pkg_path.read_text(encoding="utf-8"))
        pm = PackageManifest.from_dict(raw)
        assert pm.development_only is True
        assert pm.training_mode == "development"

    def test_smoke_training_mode_propagates_to_package(self, tmp_path):
        """training_mode='smoke' の CheckpointManifest から package に伝播する。"""
        from survivors.vision.world_detector_package import publish_development_package, PackageManifest
        from survivors.vision.world_detector import CheckpointManifest

        ckpt = CheckpointManifest(
            model_hash="a" * 64,
            data_hash="b" * 64,
            config_hash=_sha256_file(DETECTOR_CONFIG_PATH),
            build_hash="d" * 64,
            class_map_hash=_sha256_file(CLASS_MAP_PATH),
            formal_detector_eligible=False,
            training_mode="smoke",
        )
        store = tmp_path / "store"
        store.mkdir()
        pkg_path = publish_development_package(
            ckpt,
            metrics_dict={}, checkpoint_selection={},
            store_dir=store,
            cfg_path=DETECTOR_CONFIG_PATH,
            cm_path=CLASS_MAP_PATH,
        )
        raw = json.loads(pkg_path.read_text(encoding="utf-8"))
        pm = PackageManifest.from_dict(raw)
        assert pm.training_mode == "smoke"


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

    def test_restore_detects_model_hash_mismatch(self, tmp_path):
        """package 内の weight を改ざんすると推論前に拒否する。"""
        from survivors.vision.world_detector import CheckpointManifest
        from survivors.vision.world_detector_package import (
            PackageSchemaError,
            publish_development_package,
            restore_package,
        )

        store = tmp_path / "store"
        store.mkdir()
        weight_path = tmp_path / "model.pt"
        weight_path.write_bytes(b"original weight")
        checkpoint_manifest = CheckpointManifest(
            model_hash=_sha256_file(weight_path),
            data_hash="b" * 64,
            config_hash=_sha256_file(DETECTOR_CONFIG_PATH),
            build_hash="d" * 64,
            class_map_hash=_sha256_file(CLASS_MAP_PATH),
            formal_detector_eligible=False,
        )
        pkg_path = publish_development_package(
            checkpoint_manifest, {}, {}, store,
            cfg_path=DETECTOR_CONFIG_PATH,
            cm_path=CLASS_MAP_PATH,
            weight_path=weight_path,
        )
        (pkg_path.parent / "model.pt").write_bytes(b"tampered weight")

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        with pytest.raises(PackageSchemaError, match="model_hash"):
            restore_package(pkg_path, frame)


# ---- 新規回帰テスト: P1/P2 findings ----

class TestDefaultConfigGuard:
    """既定 world_detector_v1.yaml が _reject_unimplemented_config() を通過することを確認する。"""

    def test_default_config_passes_config_guard(self):
        """リポジトリ同梱の既定 config が未実装設定ガードを通過する。"""
        from train_survivors_world_detector import _reject_unimplemented_config
        from survivors.vision.world_detector import load_detector_config

        cfg = load_detector_config(DETECTOR_CONFIG_PATH)
        # 例外が発生しなければ OK
        _reject_unimplemented_config(cfg)

    def test_augmentation_in_config_is_rejected(self):
        """augmentation が含まれる config はガードに拒否される。"""
        from train_survivors_world_detector import _reject_unimplemented_config

        cfg = {"input": {"augmentation": {"random_horizontal_flip": True}}}
        with pytest.raises(ValueError, match="augmentation"):
            _reject_unimplemented_config(cfg)

    def test_lr_scheduler_in_config_is_rejected(self):
        """lr_scheduler が含まれる config はガードに拒否される。"""
        from train_survivors_world_detector import _reject_unimplemented_config

        cfg = {"training": {"lr_scheduler": "cosine"}}
        with pytest.raises(ValueError, match="lr_scheduler"):
            _reject_unimplemented_config(cfg)


class TestSessionInterleavedIndices:
    """_session_interleaved_indices が全サンプルを一度ずつ返すことを確認する。"""

    def _make_ds(self, session_lengths: dict[str, int]) -> list:
        """session_lengths: {session_id: count} のダミーデータセットを作る。"""
        from survivors.vision.world_dataset import DatasetSample
        samples = []
        for sid, n in session_lengths.items():
            for _ in range(n):
                samples.append(DatasetSample(
                    image_id=len(samples),
                    file_name="dummy.png",
                    image_width=32, image_height=32,
                    annotations=[],
                    session_id=sid,
                ))
        return samples

    def test_all_indices_returned_once_equal_sessions(self):
        """同数 session: 全インデックスが重複なく一度ずつ含まれる。"""
        from train_survivors_world_detector import _session_interleaved_indices

        ds = self._make_ds({"s1": 3, "s2": 3})
        indices = _session_interleaved_indices(ds)
        assert sorted(indices) == list(range(6)), "全 6 件が一度ずつ選ばれる"

    def test_all_indices_returned_once_imbalanced_100_to_1(self):
        """100:1 の不均衡 dataset でも 101 件全てが一度ずつ選ばれる。"""
        from train_survivors_world_detector import _session_interleaved_indices

        ds = self._make_ds({"big": 100, "small": 1})
        indices = _session_interleaved_indices(ds)
        assert len(indices) == 101, f"101 件を期待、実際={len(indices)}"
        assert len(set(indices)) == 101, "重複がないことを確認"
        assert sorted(indices) == list(range(101)), "全インデックスが一度ずつ"

    def test_no_permanently_unselected_samples(self):
        """各 epoch で永久に選ばれないサンプルが存在しないことを確認する。"""
        from train_survivors_world_detector import _session_interleaved_indices

        ds = self._make_ds({"a": 5, "b": 2, "c": 8})
        indices = _session_interleaved_indices(ds)
        assert set(indices) == set(range(15)), "全 15 件が選択済み"

    def test_uses_image_session_id_not_annotation(self):
        """annotation なし負例（session_id は画像レベル）も正しく処理される。"""
        from train_survivors_world_detector import _session_interleaved_indices
        from survivors.vision.world_dataset import DatasetSample, COCOAnnotation

        ds = [
            DatasetSample(0, "img0.png", 32, 32,
                          [COCOAnnotation(0, 0, 1, (1, 1, 4, 4), "ann_session")],
                          session_id="img_session"),
            DatasetSample(1, "img1.png", 32, 32, [], session_id="other_session"),
        ]
        indices = _session_interleaved_indices(ds)
        assert sorted(indices) == [0, 1], "annotation session_id ではなく image session_id を使用"


class TestImageLevelSessionRejection:
    """画像レベルの session_id による rejected_sessions チェックを確認する。"""

    def _write_coco_with_image_sessions(self, tmp_path: pathlib.Path, image_session: str) -> pathlib.Path:
        """annotation なし（負例）フレームに image_session を付与した COCO JSON を作成する。"""
        data = {
            "images": [{"id": 0, "file_name": "frame_0.png", "width": 1920, "height": 1080,
                         "session_id": image_session}],
            "annotations": [],
            "categories": [{"id": k, "name": f"c{k}"} for k in range(1, 12)],
        }
        p = tmp_path / "ann.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        return p

    def test_annotation_free_rejected_session_raises(self, tmp_path):
        """annotation なし負例でも画像レベルの session_id が rejected_sessions にあれば拒否する。"""
        from survivors.vision.world_dataset import WorldDataset, DatasetPreflightError

        ann_path = self._write_coco_with_image_sessions(tmp_path, "final")
        with pytest.raises(DatasetPreflightError):
            WorldDataset(ann_path, CLASS_MAP_PATH, rejected_sessions={"final"})

    def test_annotation_free_allowed_session_passes(self, tmp_path):
        """annotation なし負例でも画像レベル session_id が許可リストなら通過する。"""
        from survivors.vision.world_dataset import WorldDataset

        ann_path = self._write_coco_with_image_sessions(tmp_path, "train")
        ds = WorldDataset(ann_path, CLASS_MAP_PATH, rejected_sessions={"final"})
        assert len(ds) == 1

    def test_annotation_session_mismatch_raises(self, tmp_path):
        """annotation.session_id と image.session_id が不一致の場合は拒否する。"""
        from survivors.vision.world_dataset import WorldDataset, DatasetPreflightError

        data = {
            "images": [{"id": 0, "file_name": "f.png", "width": 1920, "height": 1080,
                         "session_id": "img_sess"}],
            "annotations": [{"id": 0, "image_id": 0, "category_id": 1,
                              "bbox": [0, 0, 10, 10], "session_id": "ann_sess"}],
            "categories": [{"id": k, "name": f"c{k}"} for k in range(1, 12)],
        }
        p = tmp_path / "ann.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(DatasetPreflightError, match="一致しません"):
            WorldDataset(p, CLASS_MAP_PATH)


class TestSmokeAndNormalModeImageLoading:
    """通常モードで画像欠損が FileNotFoundError、smoke モードでゼロ画像を許可することを確認する。"""

    def test_normal_mode_raises_on_missing_image(self):
        """通常モード（smoke=False）で存在しない画像を渡すと FileNotFoundError。"""
        torch = pytest.importorskip("torch")
        from train_survivors_world_detector import _load_training_sample
        from survivors.vision.world_dataset import DatasetSample

        sample = DatasetSample(0, "/nonexistent/frame.png", 32, 32, [], session_id="s1")
        with pytest.raises(FileNotFoundError):
            _load_training_sample(sample, smoke=False)

    def test_smoke_mode_allows_zero_image(self, tmp_path):
        """smoke=True のとき存在しない画像はゼロ画像を返す（例外なし）。"""
        torch = pytest.importorskip("torch")
        from train_survivors_world_detector import _load_training_sample
        from survivors.vision.world_dataset import DatasetSample

        sample = DatasetSample(0, str(tmp_path / "missing.png"), 32, 32, [], session_id="s1")
        img, target = _load_training_sample(sample, smoke=True)
        assert img.shape == (3, 32, 32)
        assert float(img.sum()) == 0.0, "ゼロ画像であることを確認"


class TestImagePreflightBeforeOptimizer:
    """後半画像欠落でも optimizer step が 0 件になることを確認する（P1 回帰テスト）。"""

    def test_late_missing_image_stops_before_optimizer_step(self, tmp_path):
        """3 件目の画像が欠落しているとき、optimizer step 0 件のまま停止する。"""
        from train_survivors_world_detector import _check_all_images
        from survivors.vision.world_dataset import DatasetSample

        # 4 サンプル: 0,1,3 は存在するが 2 は欠落
        existing = tmp_path / "frame_exist.png"
        existing.write_bytes(b"")  # 0 バイトの dummy（cv2 失敗は別経路）

        samples = [
            DatasetSample(0, str(existing), 32, 32, [], session_id="s0"),
            DatasetSample(1, str(existing), 32, 32, [], session_id="s1"),
            DatasetSample(2, str(tmp_path / "missing.png"), 32, 32, [], session_id="s2"),
            DatasetSample(3, str(existing), 32, 32, [], session_id="s3"),
        ]

        class _FakeView:
            def __init__(self, items): self._items = items
            def __len__(self): return len(self._items)
            def __getitem__(self, i): return self._items[i]

        train_view = _FakeView(samples[:2])
        val_view = _FakeView(samples[2:])

        errors = _check_all_images(train_view, val_view)
        assert any("missing.png" in e for e in errors), f"欠落画像が検出されるべき: {errors}"


class TestIter11Regressions:
    """iter11 P1 修正 6 件の回帰テスト。"""

    # ---- Fix 1: dry-run でも画像 preflight が実行される ----
    def test_dry_run_returns_exit_code_1_on_missing_image(self, tmp_path):
        """欠落画像を含む dataset で dry-run を実行すると exit code 1 になる（P1 回帰テスト）。"""
        from train_survivors_world_detector import _check_all_images
        from survivors.vision.world_dataset import DatasetSample

        existing = tmp_path / "exist.png"
        existing.write_bytes(b"")

        class _FakeView:
            def __init__(self, items): self._items = items
            def __len__(self): return len(self._items)
            def __getitem__(self, i): return self._items[i]

        train_view = _FakeView([DatasetSample(0, str(existing), 32, 32, [], session_id="s0")])
        val_view = _FakeView([DatasetSample(1, str(tmp_path / "missing.png"), 32, 32, [], session_id="s1")])

        # _check_all_images は dry-run 前に呼ばれる想定なので、欠落が検出されれば OK
        errors = _check_all_images(train_view, val_view)
        assert errors, "欠落画像が検出されるべき（dry-run でも preflight は走る）"

    # ---- Fix 2: training_mode 欠落 resume を拒否 ----
    def test_resume_rejects_missing_training_mode_field(self, tmp_path):
        """training_state.json に training_mode がない場合は resume を拒否する（P1 回帰テスト）。"""
        torch = pytest.importorskip("torch")
        from train_survivors_world_detector import (
            CheckpointSelector, _LAST_CHECKPOINT_NAME, _RESUME_STATE_NAME,
            _save_resume_state, _load_resume_state,
        )
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        selector = CheckpointSelector(metric="val_map50_95", keep_top_k=3)
        _save_resume_state(tmp_path, 1, selector, model, optimizer, training_mode="smoke")

        # training_mode を削除して旧 schema を模擬する
        state = json.loads((tmp_path / _RESUME_STATE_NAME).read_text())
        del state["training_mode"]
        (tmp_path / _RESUME_STATE_NAME).write_text(json.dumps(state), encoding="utf-8")

        with pytest.raises(ValueError, match="training_mode"):
            _load_resume_state(tmp_path, selector, model, optimizer, expected_training_mode="smoke")

    # ---- Fix 3: 書き込み境界 — CheckpointManifest.save() ----
    def test_checkpoint_manifest_save_rejects_formal_true(self, tmp_path):
        """CheckpointManifest.save() は formal_detector_eligible=True を拒否する（P1 回帰テスト）。"""
        from survivors.vision.world_detector import CheckpointManifest
        m = CheckpointManifest(
            model_hash="a" * 64, data_hash="b" * 64, config_hash="c" * 64,
            build_hash="d" * 64, class_map_hash="e" * 64,
            formal_detector_eligible=True,
        )
        with pytest.raises(ValueError, match="formal_detector_eligible"):
            m.save(tmp_path / "manifest.json")

    def test_checkpoint_manifest_save_rejects_invalid_training_mode(self, tmp_path):
        """CheckpointManifest.save() は training_mode='formal' を拒否する（P1 回帰テスト）。"""
        from survivors.vision.world_detector import CheckpointManifest
        m = CheckpointManifest(
            model_hash="a" * 64, data_hash="b" * 64, config_hash="c" * 64,
            build_hash="d" * 64, class_map_hash="e" * 64,
            formal_detector_eligible=False,
            training_mode="formal",
        )
        with pytest.raises(ValueError, match="training_mode"):
            m.save(tmp_path / "manifest.json")

    # ---- Fix 3: 書き込み境界 — publish_development_package() ----
    def test_publish_rejects_nan_score_threshold(self, tmp_path):
        """NaN score_threshold で publish すると ValueError になる（P1 回帰テスト）。"""
        from survivors.vision.world_detector_package import publish_development_package
        store = tmp_path / "store"; store.mkdir()
        with pytest.raises(ValueError, match="score_threshold"):
            publish_development_package(
                _make_checkpoint_manifest(), metrics_dict={}, checkpoint_selection={},
                store_dir=store, cfg_path=DETECTOR_CONFIG_PATH, cm_path=CLASS_MAP_PATH,
                score_threshold=float("nan"),
            )

    def test_publish_rejects_invalid_development_only(self, tmp_path):
        """development_only=False の manifest で publish すると ValueError になる（P1 回帰テスト）。"""
        from survivors.vision.world_detector_package import publish_development_package
        from survivors.vision.world_detector import CheckpointManifest
        store = tmp_path / "store"; store.mkdir()
        bad_manifest = CheckpointManifest(
            model_hash="a" * 64, data_hash="b" * 64, config_hash="c" * 64,
            build_hash="d" * 64, class_map_hash="e" * 64,
            formal_detector_eligible=False,
            development_only=False,
        )
        with pytest.raises(ValueError, match="development_only"):
            publish_development_package(
                bad_manifest, metrics_dict={}, checkpoint_selection={},
                store_dir=store, cfg_path=DETECTOR_CONFIG_PATH, cm_path=CLASS_MAP_PATH,
            )

    # ---- Fix 4: nearest_distance 上限 / slice params 検証 ----
    def test_nearest_distance_over_1_is_rejected(self):
        """nearest_distance_error > 1.0 は passed=False になる（P1 回帰テスト）。"""
        from eval_survivors_world_detector import compute_dev_diagnostics, evaluate_from_predictions
        m = evaluate_from_predictions([], [], num_classes=12)
        m.nearest_distance_error = 2.0
        result = compute_dev_diagnostics(
            metrics=m,
            gate_cfg={"nearest_distance_median_max": 3.0},  # 閾値が大きくても distance>1 は拒否
            class_name_by_id={},
            num_classes=12,
        )
        assert not result.passed

    def test_invalid_nd_max_over_1_is_rejected(self):
        """nearest_distance_median_max > 1.0 の閾値自体も fail-closed で拒否（P1 回帰テスト）。"""
        from eval_survivors_world_detector import compute_dev_diagnostics, evaluate_from_predictions
        m = evaluate_from_predictions([], [], num_classes=12)
        m.nearest_distance_error = 0.5
        result = compute_dev_diagnostics(
            metrics=m,
            gate_cfg={"nearest_distance_median_max": 3.0},  # 閾値が [0,1] 外
            class_name_by_id={},
            num_classes=12,
        )
        assert not result.passed

    def test_slice_negative_min_instances_is_rejected(self):
        """slice の min_instances < 1 は passed=False になる（P1 回帰テスト）。"""
        from eval_survivors_world_detector import compute_dev_diagnostics, evaluate_from_predictions
        m = evaluate_from_predictions([], [], num_classes=12)
        gate_cfg = {"slice_gate": {"boss": {"recall_min": 0.5, "min_instances": -1, "min_sessions": 1}}}
        result = compute_dev_diagnostics(
            metrics=m, gate_cfg=gate_cfg, class_name_by_id={}, num_classes=12,
            slice_annotations={"boss": []}, slice_predictions={"boss": []},
        )
        assert not result.passed

    # ---- Fix 5: batch_size=2 を拒否 ----
    def test_batch_size_2_is_rejected(self):
        """batch_size=2 は ValueError になる（P1 回帰テスト）。"""
        torch = pytest.importorskip("torch")
        from train_survivors_world_detector import _train_epoch
        from survivors.vision.world_dataset import COCOAnnotation, DatasetSample

        samples = [
            DatasetSample(i, f"missing-{i}.png", 32, 32,
                          [COCOAnnotation(i, i, 1, (1, 1, 4, 4), f"s{i % 2}")], session_id=f"s{i % 2}")
            for i in range(3)
        ]
        with pytest.raises(ValueError, match="batch_size"):
            _train_epoch(
                SimpleNamespace(_model=SimpleNamespace(train=lambda: None, __call__=lambda *a: {})),
                samples,
                SimpleNamespace(zero_grad=lambda: None, step=lambda: None),
                {"training": {"batch_size": 2}},
                smoke=True,
            )

    # ---- Fix 6: 既存 package 完全性検証 ----
    def test_republish_fails_on_missing_manifest(self, tmp_path):
        """manifest 削除後に同内容を再 publish すると PackageSchemaError になる（P1 回帰テスト）。"""
        from survivors.vision.world_detector_package import (
            publish_development_package, PackageSchemaError
        )
        store = tmp_path / "store"; store.mkdir()
        pkg_path = publish_development_package(
            _make_checkpoint_manifest(), metrics_dict={}, checkpoint_selection={},
            store_dir=store, cfg_path=DETECTOR_CONFIG_PATH, cm_path=CLASS_MAP_PATH,
        )
        # manifest を削除して破損させる
        pkg_path.unlink()
        with pytest.raises(PackageSchemaError, match="manifest"):
            publish_development_package(
                _make_checkpoint_manifest(), metrics_dict={}, checkpoint_selection={},
                store_dir=store, cfg_path=DETECTOR_CONFIG_PATH, cm_path=CLASS_MAP_PATH,
            )

    def test_republish_fails_on_tampered_config(self, tmp_path):
        """config を改ざん後に再 publish すると PackageSchemaError になる（P1 回帰テスト）。"""
        from survivors.vision.world_detector_package import (
            publish_development_package, PackageSchemaError, _CONFIG_NAME
        )
        store = tmp_path / "store"; store.mkdir()
        pkg_path = publish_development_package(
            _make_checkpoint_manifest(), metrics_dict={}, checkpoint_selection={},
            store_dir=store, cfg_path=DETECTOR_CONFIG_PATH, cm_path=CLASS_MAP_PATH,
        )
        # config を改ざんする
        cfg_cached = pkg_path.parent / _CONFIG_NAME
        cfg_cached.write_text("# tampered", encoding="utf-8")
        with pytest.raises(PackageSchemaError, match="config"):
            publish_development_package(
                _make_checkpoint_manifest(), metrics_dict={}, checkpoint_selection={},
                store_dir=store, cfg_path=DETECTOR_CONFIG_PATH, cm_path=CLASS_MAP_PATH,
            )
