"""formal release の既知値と拒否条件を検証する。\n\npreflight、verdict、artifact release の境界で親 identity が保たれることを確認する。"""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import shutil
from dataclasses import replace
import numpy as np
import pytest

from eval_survivors_world_detector import compute_ap_coco101
from eval_survivors_world_detector import _build_arg_parser as build_evaluation_parser
from train_survivors_world_detector import _build_arg_parser as build_training_parser
from reinbalance_survivors_contracts.artifact_identity import (
    ArtifactDescriptor,
    ArtifactRef,
    RestoreTestVerdict,
    artifact_uri,
)
from reinbalance_survivors_contracts.canonical_json import canonical_hash
from reinbalance_survivors_contracts.artifact_store import ArtifactStore
from survivors.vision.world_detector import DetectionResult, drop_ignored_detections
from survivors.vision.world_formal_release import (
    FormalPreflightError,
    FormalPreflightReport,
    emit_resolved_detector_yaml,
    GpuLatencyMeasurement,
    ValidationVerdict,
    build_validation_verdict,
    latency_gate,
    run_formal_preflight,
    _frame_records,
    FormalPreflightReport,
    publish_formal_detector_artifact,
    read_preflight_report,
    verify_validation_annotations,
    select_formal_threshold,
    session_recall_gate,
)


def _dataset_descriptor() -> tuple[ArtifactDescriptor, ArtifactDescriptor]:
    """preflight の fail-closed test に使う source/restore descriptor を作る。

    実 artifact store を使わず、共有 wire contract の identity を再現する。
    """
    digest = "a" * 64
    manifest_ref = ArtifactRef(
        logical_id="dataset/split",
        sha256=digest,
        size_bytes=1,
        media_type="application/json",
        store_uri=artifact_uri(digest),
    )
    dataset = ArtifactDescriptor(
        logical_id="dataset/world",
        node_kind="source_descriptor",
        producer_id="synthetic-fixture",
        producer_version="world-dataset.v1",
        identity_metadata={"manifest_logical_id": manifest_ref.logical_id},
        files=(manifest_ref,),
    )
    restore = RestoreTestVerdict(
        logical_id="restore/world",
        subject=dataset.node_ref(),
        manifest_hash=digest,
        primary_root="primary",
        backup_root="backup",
        verify_mode="full",
        checked_object_count=1,
        passed=True,
        blocking_reasons=(),
    ).to_descriptor()
    return dataset, restore


def test_coco101_ap_uses_recall_point_interpolation_and_ignores_background() -> None:
    """既知の真陽性と背景予測から COCO 101 点 AP を得る。

    背景 class の高 score が正式 precision を変えないことも固定する。
    """
    ap = compute_ap_coco101(
        np.asarray([[0, 0, 10, 10]], dtype=np.float32),
        np.asarray([1], dtype=np.int64),
        np.asarray([[0, 0, 10, 10], [40, 40, 50, 50]], dtype=np.float32),
        np.asarray([0.9, 1.0], dtype=np.float32),
        np.asarray([1, 0], dtype=np.int64),
    )

    assert ap == 1.0


def test_ignore_region_drops_all_classes_by_center_only() -> None:
    """アイテム HUD 内の center を全 class から除外する。

    同じ box の edge が外へ出ていても center 判定を使う。
    """
    result = DetectionResult(
        boxes_xyxy=np.asarray([[90, 35, 400, 140], [500, 500, 600, 600]], dtype=np.float32),
        scores=np.asarray([0.9, 0.8], dtype=np.float32),
        class_ids=np.asarray([1, 15], dtype=np.int32),
        image_width=1920,
        image_height=1080,
    )

    filtered = drop_ignored_detections(result, [[0.05, 0.033, 0.203, 0.120]])

    np.testing.assert_array_equal(filtered.class_ids, [15])


def test_formal_eval_filters_ground_truth_and_predictions_by_box_center() -> None:
    """formal eval は ignore region 内の GT と予測を同じ center 規則で落とす。

    領域外の class は双方の配列に残し、FP/FN の母集団を一致させる。
    """
    frames = _frame_records(
        {"session": [{
            "image_id": 1,
            "image_width": 1920,
            "image_height": 1080,
            "boxes_xyxy": np.asarray([[180, 70, 220, 110], [500, 500, 540, 540]], dtype=np.float32),
            "scores": np.asarray([0.9, 0.8], dtype=np.float32),
            "class_ids": np.asarray([2, 5], dtype=np.int32),
        }]},
        {"session": [
            {"image_id": 1, "category_id": 2, "bbox": [180, 70, 40, 40], "image_width": 1920, "image_height": 1080},
            {"image_id": 1, "category_id": 5, "bbox": [500, 500, 40, 40], "image_width": 1920, "image_height": 1080},
        ]},
        [[0.05, 0.033, 0.203, 0.120]],
    )

    assert len(frames) == 1
    assert [item["class_id"] for item in frames[0]["gt"]] == [5]
    assert [item["class_id"] for item in frames[0]["pred"]] == [5]


def test_world_dataset_preserves_formal_track_and_time_slice_metadata(tmp_path) -> None:
    """COCO の正式評価 metadata を DatasetSample annotation に引き継ぐ。\n\ntrack switch と late-game gate が参照する元 field を loader が落とさないことを確認する。"""
    from survivors.vision.world_dataset import WorldDataset

    class_map = Path(__file__).parents[2] / "configs" / "world_class_map_v2.yaml"
    annotations_path = tmp_path / "annotations.json"
    annotations_path.write_text(json.dumps({
        "images": [{
            "id": 1, "file_name": "frame.png", "width": 100, "height": 100,
            "session_id": "validation-1", "time_band": "late",
        }],
        "annotations": [{
            "id": 4, "image_id": 1, "category_id": 2, "bbox": [30, 30, 5, 5],
            "session_id": "validation-1", "track_id": "enemy-4", "late_game": True,
            "time_band": "late",
        }],
        "categories": [{"id": 2, "name": "enemy_normal"}],
    }), encoding="utf-8")

    loaded = WorldDataset(annotations_path, class_map)[0].annotations[0]

    assert loaded.track_id == "enemy-4"
    assert loaded.late_game is True
    assert loaded.time_band == "late"


def test_formal_preflight_rejects_failed_feasibility_before_opening_dataset(tmp_path) -> None:
    """feasibility verdict の FAIL を formal preflight 入口で拒否する。

    missing capture/annotation paths を読みに行かない順序も固定する。
    """
    dataset, restore = _dataset_descriptor()
    feasibility = tmp_path / "feasibility.json"
    feasibility.write_text(
        '{"schema_version":"perception_feasibility_verdict.v1",'
        '"status":"FAIL","selected_architecture":"ssdlite320"}',
        encoding="utf-8",
    )

    with pytest.raises(FormalPreflightError, match="feasibility"):
        run_formal_preflight(
            feasibility_verdict_path=feasibility,
            capture_store_root=tmp_path / "missing-capture",
            dataset_descriptor=dataset,
            restore_verdict_descriptor=restore,
            annotations_path=tmp_path / "missing-coco.json",
            class_map_path=tmp_path / "missing-class-map.yaml",
            config_path=tmp_path / "missing-config.yaml",
            cli_overrides={},
            target_profile_hash="b" * 64,
            hardware_id="synthetic-gpu",
            annotation_agreement_path=tmp_path / "missing-agreement.json",
            minimum_counts={},
        )


def test_formal_preflight_rejects_development_or_synthetic_descriptor(tmp_path) -> None:
    """development または synthetic の dataset descriptor を preflight 入口で拒否する。\n\ncapture と annotation file を開く前に自己申告の正式 eligibility を信用しないことを確かめる。"""
    dataset, restore = _dataset_descriptor()
    dataset = ArtifactDescriptor(
        logical_id=dataset.logical_id,
        node_kind=dataset.node_kind,
        producer_id=dataset.producer_id,
        producer_version=dataset.producer_version,
        identity_metadata={**dataset.identity_metadata, "game_build_id": "build-v1", "synthetic": True},
        files=dataset.files,
    )
    feasibility = tmp_path / "feasibility.json"
    feasibility.write_text(
        '{"schema_version":"perception_feasibility_verdict.v1",'
        '"status":"PASS","selected_architecture":"ssdlite320"}',
        encoding="utf-8",
    )

    with pytest.raises(FormalPreflightError, match="synthetic|development"):
        run_formal_preflight(
            feasibility_verdict_path=feasibility,
            capture_store_root=tmp_path / "missing-capture",
            dataset_descriptor=dataset,
            restore_verdict_descriptor=restore,
            annotations_path=tmp_path / "missing-coco.json",
            class_map_path=Path(__file__).parents[2] / "configs" / "world_class_map_v2.yaml",
            config_path=Path(__file__).parents[2] / "configs" / "world_detector_v2.yaml",
            cli_overrides={},
            target_profile_hash="a" * 64,
            hardware_id="gpu-1",
            annotation_agreement_path=tmp_path / "missing-agreement.json",
            minimum_counts={},
        )


def test_preflight_annotation_hash_rejects_eval_dataset_substitution(tmp_path) -> None:
    """evaluation annotation file を preflight が束縛した COCO export と照合する。\n\n学習後に model_validation の入力を差し替えても verdict の親 identity が変わらないことを確認する。"""
    raw = b"sealed model validation annotations"
    annotations = tmp_path / "annotations.json"
    annotations.write_bytes(raw)
    _checkpoint, _verdict, _config, _class_map, _tracker, preflight = _formal_package_inputs(tmp_path)
    preflight = replace(preflight, annotations_sha256=hashlib.sha256(raw).hexdigest())
    preflight = replace(
        preflight,
        preflight_canonical_hash=canonical_hash(preflight.to_dict(include_hash=False)),
    )
    verify_validation_annotations(preflight, annotations)
    annotations.write_bytes(b"different annotations")
    with pytest.raises(FormalPreflightError, match="annotation hash"):
        verify_validation_annotations(preflight, annotations)


def test_formal_training_parser_uses_descriptor_without_split_argument() -> None:
    """formal CLI は descriptor を受け、開封済み split 引数を要求しない。

    formal preflight が descriptor 内の frozen split だけを読む境界を固定する。
    """
    parsed = build_training_parser().parse_args([
        "--annotations", "model-coco.json", "--output", "checkpoint",
        "--formal", "--feasibility-verdict", "feasibility.json",
        "--capture-store", "capture", "--target-profile-hash", "a" * 64,
        "--hardware-id", "target-gpu", "--dataset-descriptor", "dataset.json",
        "--restore-verdict", "restore.json", "--annotation-agreement", "agreement.json",
        "--minimum-counts", "minimums.json", "--preflight-report-out", "preflight.json",
    ])

    assert parsed.formal is True
    assert parsed.split is None


def test_session_recall_gate_bootstraps_ratios_and_ignores_zero_gt_sessions() -> None:
    """cluster CI は session recall の比率を再標本化する。

    GT 0 session は non-applicable とし、低い session を frame 数で薄めない。
    """
    lower, passed = session_recall_gate(
        {"a": (1, 1), "b": (0, 1), "c": (0, 1), "d": (0, 1), "empty": (0, 0)},
        floor=0.8,
        resamples=500,
    )

    assert lower < 0.8
    assert passed is False


def test_latency_gate_requires_target_hardware_and_one_hundred_samples() -> None:
    """target GPU 以外と 100 未満の計測は latency gate を落とす。

    hardware identity の食い違いも同じ入口で拒否する。
    """
    other = latency_gate(
        GpuLatencyMeasurement(12.0, 100, "gpu-a", "other", False),
        hardware_id="gpu-a",
    )
    short = latency_gate(
        GpuLatencyMeasurement(12.0, 99, "gpu-a", "target_gpu", True),
        hardware_id="gpu-a",
    )
    mismatch = latency_gate(
        GpuLatencyMeasurement(12.0, 100, "gpu-b", "target_gpu", True),
        hardware_id="gpu-a",
    )

    assert not other["passed"] and not short["passed"] and not mismatch["passed"]


def test_formal_threshold_selection_is_deterministic_and_respects_precision() -> None:
    """選択規則は precision floor を満たす候補の平均 recall 最大値を選ぶ。

    同じ candidate 表に対する実行は常に同じ threshold を返す。
    """
    candidates = {0.05: (0.95, 0.70), 0.10: (0.80, 0.85), 0.15: (0.80, 0.90)}

    first = select_formal_threshold(candidates, precision_floor=0.80)
    second = select_formal_threshold(candidates, precision_floor=0.80)

    assert first == second
    assert first[0] == 0.10


def test_synthetic_formal_verdict_binds_threshold_metrics_and_parent_hashes() -> None:
    """合成 4 session の perfect detection から identity-bound verdict を作る。

    verdict hash 改ざんを検出し、threshold と COCO metric が保存される。
    """
    from survivors.vision.world_dataset import load_class_map

    class_map = load_class_map(Path(__file__).resolve().parents[2] / "configs" / "world_class_map_v2.yaml")
    class_ids = {
        name: class_map.name_to_id(name)
        for name in ("enemy_normal", "enemy_boss", "gem_blue", "hazard_projectile", "weapon_projectile")
    }
    detections = {}
    annotations = {}
    for session_index in range(4):
        session_id = f"formal-{session_index}"
        gt = []
        boxes = []
        labels = []
        for class_index, class_id in enumerate(class_ids.values()):
            x, y = 5 + 17 * class_index, 5 + 15 * class_index
            box = [x, y, x + 8, y + 8]
            boxes.append(box)
            labels.append(class_id)
            gt.append({
                "image_id": session_index,
                "category_id": class_id,
                "bbox": [x, y, 8, 8],
                "image_width": 100,
                "image_height": 100,
                "late_game": True,
                "track_id": f"{class_id}-{session_index}",
            })
        annotations[session_id] = gt
        detections[session_id] = [{
            "image_id": session_index,
            "image_width": 100,
            "image_height": 100,
            "boxes_xyxy": np.asarray(boxes),
            "scores": np.full(len(boxes), 0.9),
            "class_ids": np.asarray(labels),
        }]
    identity = {name: value * 64 for name, value in zip(
        ("dataset_identity", "config_identity", "build_identity", "feasibility_verdict_hash", "target_profile_hash", "split_hash"),
        "abcdef",
    )}
    preflight = FormalPreflightReport(
        **identity,
        annotations_sha256="8" * 64,
        dataset_logical_id="dataset/world",
        restore_verdict_logical_id="restore/world",
        game_build_id="build-1",
        hardware_id="gpu-1",
        train_sessions=("train-only",),
        validation_sessions=tuple(annotations),
        slice_counts={},
        agreement={},
        code_hashes={"world_detector.py": "1" * 64},
        resolved_config={"input": {"ignore_regions": []}, "tracker": {}},
        preflight_canonical_hash="7" * 64,
    )
    checkpoint = {
        "training_mode": "formal",
        "development_only": False,
        "formal_detector_eligible": False,
        "data_hash": preflight.dataset_identity,
        "build_hash": preflight.build_identity,
        "resolved_config_hash": preflight.config_identity,
    }

    verdict = build_validation_verdict(
        detections_by_session=detections,
        annotations_by_session=annotations,
        class_map=class_map,
        gates={"ci_resamples": 100},
        latency=GpuLatencyMeasurement(1.0, 100, "gpu-1", "target_gpu", True),
        preflight=preflight,
        checkpoint_manifest=checkpoint,
        checkpoint_manifest_hash="8" * 64,
        weight_hash="9" * 64,
        threshold_rule={"kind": "max_mean_recall_subject_to_precision", "precision_floor": 0.8, "candidates": [0.5, 0.9, 0.95]},
    )

    assert verdict.status == "PASS"
    assert verdict.metrics["map50_95"] == 1.0
    assert verdict.metrics["density_error_ci_upper"] == 0.0
    assert verdict.metrics["directional_density_correlation_ci_lower"] == 1.0
    assert verdict.score_threshold == 0.5
    assert ValidationVerdict.from_dict(verdict.to_dict()).verdict_canonical_hash == verdict.verdict_canonical_hash
    tampered = verdict.to_dict()
    tampered["metrics"]["map50_95"] = 0.0
    with pytest.raises(ValueError, match="canonical hash"):
        ValidationVerdict.from_dict(tampered)


def test_formal_eval_parser_exposes_in_process_checkpoint_inputs() -> None:
    """formal eval は checkpoint/verdict 入力を受け predictions を要求しない。

    threshold 選択が raw score を受け取る正式経路を parser で固定する。
    """
    parsed = build_evaluation_parser().parse_args([
        "--annotations", "model-coco.json", "--formal",
        "--preflight-report", "preflight.json", "--checkpoint-dir", "checkpoint",
        "--latency", "latency.json", "--gates", "gates.yaml", "--verdict-out", "verdict.json",
    ])

    assert parsed.formal is True
    assert parsed.predictions is None

    with pytest.raises(SystemExit) as exc:
        from eval_survivors_world_detector import main as eval_main
        eval_main([
            "--annotations", "model-coco.json", "--formal", "--predictions", "external.json",
            "--preflight-report", "preflight.json", "--checkpoint-dir", "checkpoint",
            "--latency", "latency.json", "--gates", "gates.yaml", "--verdict-out", "verdict.json",
        ])
    assert exc.value.code == 2


def test_formal_eval_runs_checkpoint_in_process_and_writes_verdict(tmp_path, monkeypatch) -> None:
    """checkpoint weight を in-process 推論し raw score から verdict を作る。\n\n外部 prediction を使わず detector に score_threshold=0.0 を渡すことも確認する。"""
    import cv2
    import torch
    from types import SimpleNamespace
    from eval_survivors_world_detector import main as eval_main
    from survivors.vision.world_dataset import load_class_map
    from survivors.vision.world_detector import DetectionResult, WorldDetector

    checkpoint, _old_verdict, config, class_map, _tracker, preflight = _formal_package_inputs(tmp_path)
    annotations_path = tmp_path / "annotations.json"
    annotations_path.write_bytes(b"formal annotations")
    preflight.write(tmp_path / "preflight.json")
    gate_path = tmp_path / "gates.yaml"
    import yaml
    gate_path.write_text(yaml.safe_dump({
        "ci_resamples": 100,
        "boss_min_instances": 1,
        "boss_min_sessions": 3,
        "hazard_min_instances": 1,
        "hazard_min_sessions": 3,
    }), encoding="utf-8")
    class_map_value = load_class_map(class_map)
    samples = []
    image_by_path = {}
    boxes = []
    labels = []
    for class_index, category in enumerate(class_map_value.foreground_classes):
        x = 30 + (class_index % 5) * 12
        y = 30 + (class_index // 5) * 12
        boxes.append([x, y, x + 5, y + 5])
        labels.append(category["id"])
    for session_index, session_id in enumerate(preflight.validation_sessions):
        file_name = str(tmp_path / f"frame-{session_index}.png")
        image_by_path[file_name] = np.full((100, 100, 3), session_index, dtype=np.uint8)
        annotations = []
        for class_index, category in enumerate(class_map_value.foreground_classes):
            x = 30 + (class_index % 5) * 12
            y = 30 + (class_index // 5) * 12
            annotations.append(SimpleNamespace(
                category_id=category["id"], bbox_xywh=(x, y, 5, 5),
                track_id=None, late_game=True, time_band="late",
            ))
        samples.append(SimpleNamespace(
            session_id=session_id,
            image_id=session_index,
            image_width=100,
            image_height=100,
            file_name=file_name,
            annotations=annotations,
        ))

    class FakeDataset:
        """sealed session に結び付いた synthetic validation sample を返す。\n\nWorldDataset の eval constructor と同じ呼び方で使用できる。"""
        def __init__(self, *args, **kwargs):
            """test 側で frame を用意するため dataset path を保持しない。\n\n許可する session id は preflight report から固定されている。"""
            pass

        def __iter__(self):
            """model_validation session ごとに sample を一度ずつ返す。\n\n各 sample に全 foreground class を付けて gate の入力を安定させる。"""
            return iter(samples)

    class FakeDetector:
        """raw-score threshold を記録しながら synthetic detection を返す。\n\ntorchvision SSD と同じ推論 cap を持つ軽量 detector として使う。"""
        def __init__(self):
            """state dict の受け口と推論 cap を初期化する。\n\nこのテストでは runtime 用の torchvision weight を読み込まない。"""
            self._model = SimpleNamespace(
                score_thresh=0.0,
                topk_candidates=300,
                detections_per_img=100,
                load_state_dict=lambda state, **kwargs: None,
                eval=lambda: None,
            )
            self.thresholds = []

        def infer(self, frame, *, score_threshold):
            """synthetic validation frame の正解 box をそのまま返す。\n\n呼出元の threshold を記録し formal 経路が raw score を保つことを確認する。"""
            self.thresholds.append(score_threshold)
            return DetectionResult(
                np.asarray(boxes, dtype=np.float32),
                np.full(len(labels), 0.99, dtype=np.float32),
                np.asarray(labels, dtype=np.int32),
                100,
                100,
            )

    detector = FakeDetector()
    monkeypatch.setattr(WorldDetector, "from_config", staticmethod(lambda *args: detector))
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: {})
    monkeypatch.setattr("survivors.vision.world_dataset.WorldDataset", FakeDataset)
    monkeypatch.setattr(cv2, "imread", lambda path: image_by_path[path])
    latency_path = tmp_path / "latency.json"
    latency_path.write_text(json.dumps(GpuLatencyMeasurement(
        1.0, 100, preflight.hardware_id, "target_gpu", True,
    ).to_dict()), encoding="utf-8")
    verdict_path = tmp_path / "eval-verdict.json"
    result = eval_main([
        "--annotations", str(annotations_path),
        "--formal",
        "--preflight-report", str(tmp_path / "preflight.json"),
        "--checkpoint-dir", str(checkpoint),
        "--latency", str(latency_path),
        "--gates", str(gate_path),
        "--config", str(config),
        "--class-map", str(class_map),
        "--verdict-out", str(verdict_path),
    ])

    verdict = ValidationVerdict.from_dict(json.loads(verdict_path.read_text(encoding="utf-8")))
    assert result == 0
    assert detector.thresholds == [0.0] * len(preflight.validation_sessions)
    assert verdict.status == "PASS"
    assert verdict.weight_hash == hashlib.sha256((checkpoint / "model.pt").read_bytes()).hexdigest()
    assert verdict.inference_caps["topk_candidates"] == 300


def test_formal_publish_cli_exposes_restore_and_dataset_identity_inputs() -> None:
    """publisher CLI は artifact restore と dataset binding に必要な入力を要求する。

    formal package だけを作る経路へ迂回できないよう parser contract を固定する。
    """
    from publish_survivors_world_detector_formal import _build_arg_parser

    parsed = _build_arg_parser().parse_args([
        "--verdict", "verdict.json",
        "--preflight-report", "preflight.json",
        "--checkpoint-dir", "checkpoint",
        "--out-root", "packages",
        "--artifact-store", "artifact-store",
        "--backup-root", "backup",
        "--restore-root", "restore",
        "--dataset-descriptor", "dataset.json",
        "--logical-prefix", "release/world-detector",
        "--resolved-yaml-out", "resolved.yaml",
    ])

    assert parsed.restore_root == "restore"
    assert parsed.preflight_report == "preflight.json"


def _formal_package_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path, dict, FormalPreflightReport]:
    """PASS package publish 用の checkpoint/verdict pair を合成する。

    正式 store を使わず各 parent hash を checkpoint file に結び付ける。
    """
    from survivors.vision.world_detector import CheckpointManifest, load_detector_config
    from survivors.vision.world_formal_release import resolved_config_hash

    config = Path(__file__).parents[2] / "configs" / "world_detector_v2.yaml"
    class_map = Path(__file__).parents[2] / "configs" / "world_class_map_v2.yaml"
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir(parents=True)
    weight = checkpoint_dir / "model.pt"
    weight.write_bytes(b"synthetic formal weights")
    cfg = load_detector_config(config)
    data_hash = "a" * 64
    game_build_id = "synthetic-formal-build-v1"
    target_profile_hash = "f" * 64
    hardware_id = "synthetic-gpu"
    resolved_hash = resolved_config_hash(cfg)
    code_hashes = {"world_detector.py": "e" * 64}
    build_hash = canonical_hash({
        "target_profile_hash": target_profile_hash,
        "game_build_id": game_build_id,
        "hardware_id": hardware_id,
        "code_hashes": code_hashes,
    })
    preflight = FormalPreflightReport(
        dataset_identity=data_hash,
        annotations_sha256=hashlib.sha256(b"formal annotations").hexdigest(),
        config_identity=resolved_hash,
        build_identity=build_hash,
        feasibility_verdict_hash="9" * 64,
        dataset_logical_id="dataset/world-formal",
        restore_verdict_logical_id="restore/world-formal",
        target_profile_hash=target_profile_hash,
        game_build_id=game_build_id,
        hardware_id=hardware_id,
        split_hash="c" * 64,
        train_sessions=("train-synthetic",),
        validation_sessions=tuple(f"validation-{index}" for index in range(4)),
        slice_counts={},
        agreement={},
        code_hashes=code_hashes,
        resolved_config=cfg,
        preflight_canonical_hash="0" * 64,
    )
    preflight = replace(
        preflight,
        preflight_canonical_hash=canonical_hash(preflight.to_dict(include_hash=False)),
    )
    manifest = CheckpointManifest(
        model_hash=hashlib.sha256(weight.read_bytes()).hexdigest(),
        data_hash=data_hash,
        config_hash=hashlib.sha256(config.read_bytes()).hexdigest(),
        build_hash=build_hash,
        class_map_hash=hashlib.sha256(class_map.read_bytes()).hexdigest(),
        formal_detector_eligible=False,
        split_hash="c" * 64,
        resolved_config_hash=resolved_hash,
        development_only=False,
        training_mode="formal",
    )
    manifest_path = checkpoint_dir / "manifest.json"
    manifest.save(manifest_path)
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_payload["checkpoint_selection"] = {"metric": "val_map50_95", "keep_top_k": 1}
    manifest_path.write_text(json.dumps(manifest_payload, sort_keys=True), encoding="utf-8")
    checkpoint_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    (checkpoint_dir / "run_manifest.json").write_text(json.dumps({
        "schema_version": "world_detector_training_run.v1",
        "resolved_config": cfg,
        "selection_rule": manifest_payload["checkpoint_selection"],
        "preflight_canonical_hash": preflight.preflight_canonical_hash,
        "dataset_identity": data_hash,
        "build_identity": build_hash,
        "code_hashes": code_hashes,
    }), encoding="utf-8")
    from survivors.vision.world_dataset import load_class_map
    class_map_value = load_class_map(class_map)
    detections_by_session = {}
    annotations_by_session = {}
    for session_index in range(4):
        session_id = f"validation-{session_index}"
        boxes = []
        labels = []
        gt = []
        for class_index, category in enumerate(class_map_value.foreground_classes):
            x = 30 + (class_index % 5) * 12
            y = 30 + (class_index // 5) * 12
            boxes.append([x, y, x + 5, y + 5])
            labels.append(category["id"])
            gt.append({
                "image_id": session_index,
                "category_id": category["id"],
                "bbox": [x, y, 5, 5],
                "image_width": 100,
                "image_height": 100,
                "late_game": True,
            })
        detections_by_session[session_id] = [{
            "image_id": session_index,
            "image_width": 100,
            "image_height": 100,
            "boxes_xyxy": np.asarray(boxes, dtype=np.float32),
            "scores": np.full(len(labels), 0.99, dtype=np.float32),
            "class_ids": np.asarray(labels, dtype=np.int32),
        }]
        annotations_by_session[session_id] = gt
    verdict_value = build_validation_verdict(
        detections_by_session=detections_by_session,
        annotations_by_session=annotations_by_session,
        class_map=class_map_value,
        gates={
            "ci_resamples": 100,
            "boss_min_instances": 1,
            "boss_min_sessions": 3,
            "hazard_min_instances": 1,
            "hazard_min_sessions": 3,
        },
        latency=GpuLatencyMeasurement(1.0, 100, "synthetic-gpu", "target_gpu", True),
        preflight=preflight,
        checkpoint_manifest=manifest_payload,
        checkpoint_manifest_hash=checkpoint_hash,
        weight_hash=manifest.model_hash,
        threshold_rule={
            "kind": "max_mean_recall_subject_to_precision",
            "precision_floor": 0.8,
            "candidates": [0.5],
        },
        inference_caps={"score_thresh": 0.0, "topk_candidates": 300},
    )
    assert verdict_value.status == "PASS"
    verdict_path = tmp_path / "verdict.json"
    verdict_path.write_text(json.dumps(verdict_value.to_dict()), encoding="utf-8")
    return checkpoint_dir, verdict_path, config, class_map, cfg["tracker"], preflight


def _formal_package_with_verdict_change(tmp_path: Path, field: str, value: object) -> Path:
    """指定した verdict 内容を保った package fixture を用意する。

    verdict と manifest の file hash、content key を更新し、内容検証だけを試せる状態にする。
    """
    from survivors.vision.world_detector_package import PackageManifest, _content_key, publish_formal_package

    checkpoint, verdict_path, config, class_map, tracker, _preflight = _formal_package_inputs(tmp_path)
    manifest_path = publish_formal_package(
        verdict_path=verdict_path,
        checkpoint_dir=checkpoint,
        out_root=tmp_path / "packages",
        class_map_path=class_map,
        config_path=config,
        tracker_config=tracker,
    )
    bad_verdict = json.loads(verdict_path.read_bytes())
    bad_verdict[field] = value
    bad_verdict.pop("verdict_canonical_hash")
    bad_verdict["verdict_canonical_hash"] = canonical_hash(bad_verdict)
    verdict_bytes = json.dumps(bad_verdict, sort_keys=True).encode("utf-8")
    manifest = PackageManifest.from_dict(json.loads(manifest_path.read_bytes()))
    bad_manifest = replace(
        manifest,
        validation_verdict_hash=hashlib.sha256(verdict_bytes).hexdigest(),
    )
    package_dir = manifest_path.parent.parent / _content_key(bad_manifest)
    shutil.copytree(manifest_path.parent, package_dir)
    (package_dir / "validation_verdict.json").write_bytes(verdict_bytes)
    (package_dir / "manifest.json").write_text(
        json.dumps(bad_manifest.to_dict(), sort_keys=True), encoding="utf-8",
    )
    return package_dir


@pytest.mark.parametrize(("field", "value"), [("status", "FAIL"), ("score_threshold", 0.01)])
def test_formal_loaders_reject_verdict_content_mismatch(
    tmp_path: Path, field: str, value: object,
) -> None:
    """formal loader と runtime restore は verdict 内容の不一致を拒否する。

    file hash を manifest に結び直しても PASS と package identity の条件を維持させる。
    """
    from survivors.vision.world_detector_package import (
        PackageSchemaError,
        load_formal_detector,
        restore_package,
    )

    package_dir = _formal_package_with_verdict_change(tmp_path, field, value)
    with pytest.raises(PackageSchemaError):
        load_formal_detector(package_dir)
    with pytest.raises(PackageSchemaError):
        restore_package(
            package_dir / "manifest.json",
            np.zeros((1080, 1920, 3), dtype=np.uint8),
            require_formal=True,
        )


def test_formal_predictor_rejects_detector_verdict_content_mismatch(tmp_path: Path, monkeypatch) -> None:
    """formal predictor は detector verdict が FAIL の artifact を復元しない。

    parser の検証を固定して detector artifact を実際の formal loader まで渡す。
    """
    from types import SimpleNamespace
    import survivors.formal_perception_provider as provider
    from survivors.vision.world_detector_package import PackageSchemaError

    package_dir = _formal_package_with_verdict_change(tmp_path, "status", "FAIL")
    manifest_bytes = (package_dir / "manifest.json").read_bytes()
    detector_manifest = json.loads(manifest_bytes)
    atlas_hash = "a" * 64
    parser_bytes = b"{}"
    atlas_bytes = b'{"schema_version":"icon_atlas.v1"}'
    restored = {
        "parser_package": SimpleNamespace(
            files={"parser.json": parser_bytes, "atlas.json": atlas_bytes},
            manifest={"parser_artifact_hash": hashlib.sha256(parser_bytes).hexdigest()},
        ),
        "detector_package": SimpleNamespace(
            files={
                f"{package_dir.name}/{path.name}": path.read_bytes()
                for path in package_dir.iterdir()
            },
            manifest={
                "detector_artifact_hash": hashlib.sha256(manifest_bytes).hexdigest(),
                "model_hash": detector_manifest["model_hash"],
            },
        ),
    }
    monkeypatch.setattr(
        provider, "PackageLoader",
        SimpleNamespace(load_formal=lambda _path: SimpleNamespace(
            meta=SimpleNamespace(atlas_content_hash=atlas_hash),
        )),
    )
    monkeypatch.setattr(
        provider, "IconMatcher",
        SimpleNamespace(load_formal=lambda _path: SimpleNamespace(
            manifest=SimpleNamespace(atlas_content_hash=atlas_hash),
        )),
    )

    with pytest.raises(PackageSchemaError):
        provider.build_formal_predictor(restored)


def test_formal_package_publish_and_restore_bind_verdict_weight_and_ignore_region(tmp_path, monkeypatch) -> None:
    """formal publish と runtime restore が verdict identity と同じ ignore filter を使う。

    restore 前に weight hash と config identity が有効であることも確認する。
    """
    import torch
    from survivors.vision.entity_tracker import EntityTracker
    from survivors.vision.world_detector import DetectionResult, WorldDetector
    from survivors.vision.world_detector_package import PackageManifest, publish_formal_package, restore_package

    checkpoint_dir, verdict_path, config, class_map, tracker, _preflight = _formal_package_inputs(tmp_path)
    manifest_path = publish_formal_package(
        verdict_path=verdict_path,
        checkpoint_dir=checkpoint_dir,
        out_root=tmp_path / "packages",
        class_map_path=class_map,
        config_path=config,
        tracker_config=tracker,
    )
    manifest = PackageManifest.from_dict(json.loads(manifest_path.read_text(encoding="utf-8")))
    assert manifest.training_mode == "formal"
    assert manifest.validation_verdict_hash == hashlib.sha256(verdict_path.read_bytes()).hexdigest()

    detector = type("StubDetector", (), {
        "_model": type("StubModel", (), {"score_thresh": 0.5, "load_state_dict": lambda self, state, **kwargs: None})(),
        "infer": lambda self, frame, score_threshold: DetectionResult(
            np.asarray([[180, 70, 220, 110]], dtype=np.float32),
            np.asarray([0.9], dtype=np.float32), np.asarray([1], dtype=np.int32), 1920, 1080,
        ),
    })()
    monkeypatch.setattr(WorldDetector, "from_config", staticmethod(lambda *args: detector))
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: {})
    seen = {}
    original_update = EntityTracker.update

    def capture_update(self, result, **kwargs):
        """tracker に渡る detection count を記録する。

        元の update を呼び、restore の実経路も継続する。
        """
        seen["detections"] = len(result)
        return original_update(self, result, **kwargs)

    monkeypatch.setattr(EntityTracker, "update", capture_update)
    restored = restore_package(
        manifest_path, np.zeros((1080, 1920, 3), dtype=np.uint8), require_formal=True,
    )
    assert restored is not None
    assert seen["detections"] == 0
    assert detector._model.score_thresh == 0.0

    from survivors.vision.world_detector_package import PackageSchemaError
    original_manifest = manifest_path.read_bytes()
    raw_manifest = json.loads(original_manifest)
    raw_manifest["score_threshold"] = 0.56
    manifest_path.write_text(json.dumps(raw_manifest), encoding="utf-8")
    with pytest.raises(PackageSchemaError, match="content key"):
        restore_package(manifest_path, np.zeros((1080, 1920, 3), dtype=np.uint8), require_formal=True)
    manifest_path.write_bytes(original_manifest)
    verdict_file = manifest_path.parent / "validation_verdict.json"
    verdict_file.write_text("{}", encoding="utf-8")
    with pytest.raises(PackageSchemaError, match="validation verdict"):
        restore_package(manifest_path, np.zeros((1080, 1920, 3), dtype=np.uint8), require_formal=True)


def test_formal_publisher_rejects_failed_verdict_and_replaced_weight(tmp_path) -> None:
    """publisher は FAIL verdict と checkpoint weight 差し替えを保存前に拒否する。

    不正入力では package root を作らないことを確認する。
    """
    from survivors.vision.world_detector_package import FormalPackageRejectedError, publish_formal_package

    checkpoint, verdict_path, config, class_map, tracker, _preflight = _formal_package_inputs(tmp_path / "failed")
    failed = json.loads(verdict_path.read_text(encoding="utf-8"))
    failed["status"] = "FAIL"
    failed.pop("verdict_canonical_hash")
    failed["verdict_canonical_hash"] = canonical_hash(failed)
    verdict_path.write_text(json.dumps(failed), encoding="utf-8")
    out_root = tmp_path / "failed-packages"
    with pytest.raises(FormalPackageRejectedError, match="PASS"):
        publish_formal_package(
            verdict_path=verdict_path,
            checkpoint_dir=checkpoint,
            out_root=out_root,
            class_map_path=class_map,
            config_path=config,
            tracker_config=tracker,
        )
    assert not out_root.exists()

    checkpoint, verdict_path, config, class_map, tracker, _preflight = _formal_package_inputs(tmp_path / "run-identity")
    run_manifest_path = checkpoint / "run_manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    run_manifest["preflight_canonical_hash"] = "0" * 64
    run_manifest_path.write_text(json.dumps(run_manifest), encoding="utf-8")
    out_root = tmp_path / "run-identity-packages"
    with pytest.raises(FormalPackageRejectedError, match="identity"):
        publish_formal_package(
            verdict_path=verdict_path,
            checkpoint_dir=checkpoint,
            out_root=out_root,
            class_map_path=class_map,
            config_path=config,
            tracker_config=tracker,
        )
    assert not out_root.exists()

    checkpoint, verdict_path, config, class_map, tracker, _preflight = _formal_package_inputs(tmp_path / "weight")
    (checkpoint / "model.pt").write_bytes(b"replacement weight")
    out_root = tmp_path / "weight-packages"
    with pytest.raises(FormalPackageRejectedError, match="identity"):
        publish_formal_package(
            verdict_path=verdict_path,
            checkpoint_dir=checkpoint,
            out_root=out_root,
            class_map_path=class_map,
            config_path=config,
            tracker_config=tracker,
        )
    assert not out_root.exists()


def test_formal_artifact_envelope_restore_and_session_build_identity(tmp_path) -> None:
    """descriptor と envelope が 04-10 key contract と restore verdict に一致する。

    synthetic typed session の build hash が envelope の build object hash と同じになる。
    """
    from survivors.perception_session_split import SessionRecord
    from survivors.vision.world_detector_package import publish_formal_package
    from survivors.artifact_restore_test import run_restore_test

    checkpoint_dir, verdict_path, config, class_map, tracker, preflight = _formal_package_inputs(tmp_path)
    package_manifest_path = publish_formal_package(
        verdict_path=verdict_path,
        checkpoint_dir=checkpoint_dir,
        out_root=tmp_path / "packages",
        class_map_path=class_map,
        config_path=config,
        tracker_config=tracker,
    )
    descriptor_dataset = ArtifactDescriptor(
        logical_id=preflight.dataset_logical_id,
        node_kind="source_descriptor",
        producer_id="synthetic-dataset",
        producer_version="world-dataset.v1",
        identity_metadata={
            "dataset_identity_hash": preflight.dataset_identity,
            "game_build_id": preflight.game_build_id,
        },
    )
    store = ArtifactStore(tmp_path / "artifacts")
    verdict = ValidationVerdict.from_dict(json.loads(verdict_path.read_text(encoding="utf-8")))
    descriptor = publish_formal_detector_artifact(
        store,
        package_dir=package_manifest_path.parent,
        preflight=preflight,
        verdict=verdict,
        dataset_descriptor=descriptor_dataset,
        logical_prefix="formal/world-detector",
    )
    manifest_ref = next(ref for ref in descriptor.files if ref.logical_id == descriptor.identity_metadata["manifest_logical_id"])
    envelope = json.loads(store.object_path(manifest_ref.store_uri).read_text(encoding="utf-8"))
    import benchmark_survivors_perception as benchmark
    assert set(envelope) == benchmark._PACKAGE_REQUIRED_FIELDS["detector_package"]
    assert benchmark._manifest_ref(descriptor) == manifest_ref
    package_ref = next(ref for ref in descriptor.files if ref.logical_id == envelope["files"]["detector_artifact_hash"])
    model_ref = next(ref for ref in descriptor.files if ref.logical_id == envelope["files"]["model_hash"])
    build_ref = next(ref for ref in descriptor.files if ref.logical_id == envelope["files"]["build_hash"])
    assert envelope["detector_artifact_hash"] == package_ref.sha256
    assert envelope["model_hash"] == model_ref.sha256
    assert envelope["build_hash"] == build_ref.sha256

    session_build_hash = hashlib.sha256(preflight.game_build_id.encode("utf-8")).hexdigest()
    session = SessionRecord(
        session_id="formal-session-synthetic",
        session_hash="1" * 64,
        build_hash=session_build_hash,
        target_profile_hash=preflight.target_profile_hash,
        resolution_wh=(1920, 1080),
        duration_seconds=1800,
        kind="model_train",
        source_policy="lossless",
    )
    assert session.build_hash == envelope["build_hash"]

    _, restore_descriptor = run_restore_test(
        store,
        descriptor,
        backup_root=tmp_path / "backup",
        restore_root=tmp_path / "restore",
        verdict_logical_id="formal/world-detector/restore-verdict",
    )
    benchmark._validate_restore_verdict(descriptor, restore_descriptor)
    import yaml
    resolved_path = emit_resolved_detector_yaml(
        verdict, descriptor, restore_descriptor, preflight, tmp_path / "resolved.yaml",
    )
    resolved = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    assert resolved["schema_version"] == "world_detector_target_config.v1"
    assert resolved["score_threshold"] == verdict.score_threshold
    assert resolved["restore_verdict_logical_id"] == restore_descriptor.logical_id
    assert resolved["ignore_regions"] == preflight.resolved_config["input"]["ignore_regions"]
    assert resolved["verdict_canonical_hash"] == verdict.verdict_canonical_hash
    assert "validation_verdict_hash" not in resolved


def _formal_preflight_inputs(
    tmp_path, *, mutation: str | None = None, agreement_mode: str = "independent_pair",
    agreement_gap_hours: float = 24.0, agreement_blind: bool = True,
) -> dict:
    """capture store と全 descriptor を共有 API で組み立てる。\n\npreflight の成功入力と一項目だけを壊した反例を同じ経路で用意する。"""
    import yaml
    from survivors.capture.captured_frame import CapturedFrame
    from survivors.capture_dataset import DatasetWriter, SplitFreezer
    from survivors.artifact_restore_test import run_restore_test
    from survivors.vision.world_dataset import load_class_map

    profile_hash = "c" * 64
    build_id = "synthetic-formal-build"
    session_ids = ["train", "validation-0", "validation-1", "validation-2", "sealed"]
    capture_root = tmp_path / "capture"
    frame_paths = {}
    for session_index, session_id in enumerate(session_ids):
        writer = DatasetWriter(capture_root, session_id, profile_hash, build_id)
        pixels = np.zeros((1080, 1920, 4), dtype=np.uint8)
        pixels[..., 3] = 255
        pixels[..., 0] = session_index
        writer.write_frame(CapturedFrame(
            frame_bgra=pixels,
            captured_monotonic_ns=1,
            session_frame_index=0,
            client_rect_screen_px=(0, 0, 1920, 1080),
            foreground=True,
            target_profile_hash=profile_hash,
            game_build_id=build_id,
        ))
        writer.publish()
        session_manifest_path = capture_root / "capture_sessions" / session_id / "session_manifest.json"
        session_manifest = json.loads(session_manifest_path.read_text(encoding="utf-8"))
        session_manifest["formal_dataset_eligible"] = not (
            mutation == "development-session" and session_id == "validation-0"
        )
        # shortcut: DatasetWriter deliberately has no eligibility override; this fixture models the sealed formal exporter output.
        session_manifest_path.write_text(json.dumps(session_manifest, sort_keys=True, separators=(",", ":")), encoding="utf-8")
        frame_paths[session_id] = capture_root / "capture_sessions" / session_id / "frames" / "00000000.png"

    freezer = SplitFreezer(capture_root)
    freezer.assign("model_train", ["train"])
    freezer.assign("model_validation", ["validation-0", "validation-1", "validation-2"])
    freezer.assign("error_calibration", ["sealed"])
    freezer.freeze()
    split_path = freezer.manifest_path
    if mutation == "unfrozen-split":
        split_wire = json.loads(split_path.read_text(encoding="utf-8"))
        split_wire["frozen"] = False
        split_path.write_text(json.dumps(split_wire, sort_keys=True, separators=(",", ":")), encoding="utf-8")

    class_map_path = Path(__file__).parents[2] / "configs" / "world_class_map_v2.yaml"
    class_map = load_class_map(class_map_path)
    images = []
    annotations = []
    image_id = 1
    annotation_id = 1
    for session_id in session_ids[:-1]:
        image_path = frame_paths[session_id]
        if session_id == "train" and mutation == "sealed-png":
            image_path = frame_paths["sealed"]
        elif session_id == "train" and mutation == "external-png":
            image_path = tmp_path / "outside-session.png"
            image_path.write_bytes(b"not a captured session frame")
        image_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
        image = {
            "id": image_id,
            "file_name": str(image_path),
            "width": 1920,
            "height": 1080,
            "session_id": session_id,
            "frame_id": 0,
            "object_sha256": "0" * 64 if mutation == "bad-object-sha" and session_id == "train" else image_hash,
        }
        images.append(image)
        for class_index, category in enumerate(class_map.foreground_classes):
            x = 150 + (class_index % 5) * 300
            y = 250 + (class_index // 5) * 200
            annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category["id"],
                "bbox": [x, y, 40, 40],
                "session_id": session_id,
                "track_id": f"{session_id}-{category['id']}",
                "late_game": True,
                "time_band": "late",
            })
            annotation_id += 1
        image_id += 1
    coco_path = tmp_path / "world-coco.json"
    coco_path.write_text(json.dumps({
        "images": images,
        "annotations": annotations,
        "categories": [{"id": item["id"], "name": item["name"]} for item in class_map.foreground_classes],
    }, sort_keys=True), encoding="utf-8")
    agreement_value = 0.4 if mutation == "agreement-below-floor" else 1.0
    agreement_path = tmp_path / "agreement.json"
    agreement_data = {
        "schema_version": "annotation_agreement.v1",
        "pair_count": 1,
        "iou_mean": agreement_value,
        "class_agreement": agreement_value,
    }
    if agreement_mode == "self_repeat":
        agreement_data.update({
            "schema_version": "annotation_agreement.v2",
            "unmatched_first": 0, "unmatched_second": 0, "sessions": [], "per_class": {},
            "independence_mode": "self_repeat", "annotator_a": "alice", "annotator_b": "alice",
            "min_gap_hours": agreement_gap_hours, "required_min_gap_hours": 24.0,
            "blind": agreement_blind,
        })
    agreement_path.write_text(json.dumps(agreement_data, sort_keys=True), encoding="utf-8")

    store = ArtifactStore(tmp_path / "source-store")
    split_ref = store.put(logical_id="dataset/split-manifest", source_path=split_path, media_type="application/json")
    session_refs = {}
    for session_id in session_ids:
        manifest_path = capture_root / "capture_sessions" / session_id / "session_manifest.json"
        session_refs[session_id] = store.put(
            logical_id=f"dataset/session/{session_id}",
            source_path=manifest_path,
            media_type="application/json",
        )
    coco_ref = store.put(logical_id="dataset/world-coco-model", source_path=coco_path, media_type="application/json")
    agreement_ref = store.put(logical_id="dataset/annotation-agreement", source_path=agreement_path, media_type="application/json")
    bindings = {}
    for session_id in session_ids[:-1]:
        manifest_path = capture_root / "capture_sessions" / session_id / "session_manifest.json"
        bindings[session_id] = {
            "session_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "session_manifest_logical_id": session_refs[session_id].logical_id,
        }
    dataset_descriptor = ArtifactDescriptor(
        logical_id="dataset/world-formal-test",
        node_kind="source_descriptor",
        producer_id="synthetic-fixture",
        producer_version="world-dataset.v1",
        identity_metadata={
            "manifest_logical_id": split_ref.logical_id,
            "game_build_id": build_id,
            "build_hash": hashlib.sha256(build_id.encode("utf-8")).hexdigest(),
            "development_only": False,
            "model_session_bindings": bindings,
            "dataset_exports": {
                "world_coco_model": {"logical_id": coco_ref.logical_id, "sha256": coco_ref.sha256},
                "annotation_agreement": {"logical_id": agreement_ref.logical_id, "sha256": agreement_ref.sha256},
            },
        },
        files=(split_ref, *session_refs.values(), coco_ref, agreement_ref),
    )
    restore_verdict, restore_descriptor = run_restore_test(
        store,
        dataset_descriptor,
        backup_root=tmp_path / "source-backup",
        restore_root=tmp_path / "source-restore",
        verdict_logical_id="dataset/restore-verdict",
    )
    if mutation == "restore-manifest-mismatch":
        metadata = restore_descriptor.identity_metadata
        non_identity = restore_descriptor.non_identity_metadata
        restore_descriptor = RestoreTestVerdict(
            logical_id=restore_descriptor.logical_id,
            subject=dataset_descriptor.node_ref(),
            manifest_hash="0" * 64,
            primary_root=non_identity["primary_root"],
            backup_root=non_identity["backup_root"],
            verify_mode=metadata["verify_mode"],
            checked_object_count=metadata["checked_object_count"],
            passed=metadata["passed"],
            blocking_reasons=metadata["blocking_reasons"],
        ).to_descriptor()

    config_path = tmp_path / "world-detector.yaml"
    config = yaml.safe_load((Path(__file__).parents[2] / "configs" / "world_detector_v2.yaml").read_text(encoding="utf-8"))
    config["training"]["max_epochs"] = 1
    config["training"]["preflight"] = {
        "min_frames": 1,
        "min_entities": 1,
        "min_classes": 1,
        "min_time_bands": 1,
        "min_frames_per_split": 1,
        "min_entities_per_split": 1,
        "min_classes_per_split": 1,
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    feasibility_path = tmp_path / "feasibility.json"
    feasibility_path.write_text(json.dumps({
        "schema_version": "perception_feasibility_verdict.v1",
        "status": "PASS",
        "selected_architecture": "ssdlite320",
    }), encoding="utf-8")
    minimum_counts = {
        "min_frames": 1,
        "min_entities": 1,
        "min_classes": 1,
        "min_time_bands": 1,
        "min_frames_per_split": 1,
        "min_entities_per_split": 1,
        "min_classes_per_split": 1,
        "heavy_effect_slice": 1,
        "agreement": {"pair_count": 1, "iou_mean": 0.9, "class_agreement": 0.9},
    }
    minimum_counts_path = tmp_path / "minimum-counts.json"
    minimum_counts_path.write_text(json.dumps(minimum_counts), encoding="utf-8")
    dataset_descriptor_path = tmp_path / "dataset-descriptor.json"
    dataset_descriptor_path.write_text(json.dumps(dataset_descriptor.to_wire()), encoding="utf-8")
    restore_descriptor_path = tmp_path / "restore-descriptor.json"
    restore_descriptor_path.write_text(json.dumps(restore_descriptor.to_wire()), encoding="utf-8")
    return {
        "capture_root": capture_root,
        "dataset_descriptor": dataset_descriptor,
        "dataset_descriptor_path": dataset_descriptor_path,
        "restore_descriptor": restore_descriptor,
        "restore_descriptor_path": restore_descriptor_path,
        "annotations_path": coco_path,
        "agreement_path": agreement_path,
        "feasibility_path": feasibility_path,
        "config_path": config_path,
        "class_map_path": class_map_path,
        "minimum_counts": minimum_counts,
        "minimum_counts_path": minimum_counts_path,
        "store": store,
        "profile_hash": profile_hash,
        "hardware_id": "synthetic-gpu",
    }


def _run_preflight_fixture(inputs: dict) -> FormalPreflightReport:
    """fixture が束縛した capture、descriptor、COCO を formal preflight へ渡す。\n\n各 test が同一の入口を通って成功・失敗条件だけを比較する。"""
    return run_formal_preflight(
        feasibility_verdict_path=inputs["feasibility_path"],
        capture_store_root=inputs["capture_root"],
        dataset_descriptor=inputs["dataset_descriptor"],
        restore_verdict_descriptor=inputs["restore_descriptor"],
        annotations_path=inputs["annotations_path"],
        class_map_path=inputs["class_map_path"],
        config_path=inputs["config_path"],
        cli_overrides={},
        target_profile_hash=inputs["profile_hash"],
        hardware_id=inputs["hardware_id"],
        annotation_agreement_path=inputs["agreement_path"],
        minimum_counts=inputs["minimum_counts"],
    )


def test_formal_preflight_accepts_and_round_trips_a_bound_capture_dataset(tmp_path) -> None:
    """frozen split と全 formal descriptor を揃えた preflight 成功経路を通す。\n\nreport の split、dataset、config identity が実 artifact と一致して往復することを確認する。"""
    from survivors.vision.world_formal_release import read_preflight_report, resolved_config_hash

    inputs = _formal_preflight_inputs(tmp_path)
    report = _run_preflight_fixture(inputs)
    report_path = report.write(tmp_path / "preflight.json")
    restored = read_preflight_report(report_path)

    assert restored.split_hash == hashlib.sha256((inputs["capture_root"] / "capture_split_manifest.json").read_bytes()).hexdigest()
    assert restored.dataset_identity == report.dataset_identity
    assert restored.config_identity == resolved_config_hash(restored.resolved_config)
    assert restored.train_sessions == ("train",)
    assert restored.validation_sessions == ("validation-0", "validation-1", "validation-2")
    assert restored.restore_verdict_logical_id == inputs["restore_descriptor"].logical_id


def test_formal_preflight_accepts_self_repeat_and_records_independence_mode(tmp_path) -> None:
    """formal preflight は有効な v2 self_repeat を受理し report に mode を記録する。

    agreement 件数の閾値は independent_pair と同じ minimum_counts を通る。
    """
    report = _run_preflight_fixture(
        _formal_preflight_inputs(tmp_path, agreement_mode="self_repeat"))

    assert report.agreement["independence_mode"] == "self_repeat"


@pytest.mark.parametrize(("gap", "blind"), [(23.0, True), (24.0, False)])
def test_formal_preflight_rejects_self_repeat_without_gap_or_blind(tmp_path, gap, blind) -> None:
    """formal preflight は gap 不足または blind false の self_repeat を拒否する。

    下流の訓練入口が feasibility gate の検査を迂回しても PASS させない。
    """
    inputs = _formal_preflight_inputs(
        tmp_path, agreement_mode="self_repeat", agreement_gap_hours=gap, agreement_blind=blind)

    with pytest.raises(FormalPreflightError, match="self_repeat"):
        _run_preflight_fixture(inputs)


@pytest.mark.parametrize(("mutation", "message"), [
    ("unfrozen-split", "split manifest"),
    ("development-session", "formal eligible"),
    ("sealed-png", "session frames"),
    ("external-png", "session frames"),
    ("bad-object-sha", "session frames"),
    ("agreement-below-floor", "annotation agreement"),
    ("restore-manifest-mismatch", "restore verdict must"),
])
def test_formal_preflight_rejects_one_broken_bound_input(tmp_path, mutation, message) -> None:
    """positive fixture の一項目を壊した正式入力を fail-closed で拒否する。\n\nsealed split、session image、agreement、restore verdict の兄弟条件を一つずつ検証する。"""
    inputs = _formal_preflight_inputs(tmp_path, mutation=mutation)

    with pytest.raises(FormalPreflightError, match=message):
        _run_preflight_fixture(inputs)


def test_formal_error_ci_gates_fail_on_session_concentrated_false_positives(tmp_path) -> None:
    """default count-error CI ceiling が一 session の偏った誤検出を拒否する。\n\npooled precision が 0.8 を超えても session-cluster 上限で PASS にしない。"""
    from survivors.vision.world_dataset import load_class_map

    checkpoint, _verdict_path, _config, _class_map, _tracker, preflight = _formal_package_inputs(tmp_path)
    class_map = load_class_map(_class_map)
    detections_by_session = {}
    annotations_by_session = {}
    for session_index, session_id in enumerate(preflight.validation_sessions):
        boxes = []
        labels = []
        ground_truth = []
        for class_index, category in enumerate(class_map.foreground_classes):
            x = 30 + (class_index % 5) * 12
            y = 30 + (class_index // 5) * 12
            boxes.append([x, y, x + 5, y + 5])
            labels.append(category["id"])
            ground_truth.append({
                "image_id": session_index,
                "category_id": category["id"],
                "bbox": [x, y, 5, 5],
                "image_width": 100,
                "image_height": 100,
                "late_game": True,
            })
        if session_index == 0:
            boxes.extend([[90, 0, 95, 5], [90, 10, 95, 15], [90, 20, 95, 25], [90, 30, 95, 35]])
            labels.extend([class_map.name_to_id("enemy_normal")] * 4)
        detections_by_session[session_id] = [{
            "image_id": session_index,
            "image_width": 100,
            "image_height": 100,
            "boxes_xyxy": np.asarray(boxes, dtype=np.float32),
            "scores": np.full(len(labels), 0.99, dtype=np.float32),
            "class_ids": np.asarray(labels, dtype=np.int32),
        }]
        annotations_by_session[session_id] = ground_truth
    manifest_path = checkpoint / "manifest.json"
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    verdict = build_validation_verdict(
        detections_by_session=detections_by_session,
        annotations_by_session=annotations_by_session,
        class_map=class_map,
        gates={"ci_resamples": 100},
        latency=GpuLatencyMeasurement(1.0, 100, preflight.hardware_id, "target_gpu", True),
        preflight=preflight,
        checkpoint_manifest=manifest_payload,
        checkpoint_manifest_hash=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        weight_hash=manifest_payload["model_hash"],
        threshold_rule={"kind": "max_mean_recall_subject_to_precision", "precision_floor": 0.8, "candidates": [0.5]},
    )

    total_ground_truth = sum(len(items) for items in annotations_by_session.values())
    total_predictions = sum(
        len(frame["scores"])
        for frames in detections_by_session.values()
        for frame in frames
    )
    assert total_ground_truth / total_predictions >= 0.8
    assert verdict.gates["thresholds"]["density_error_ci_upper_max"] == 0.20
    assert verdict.gates["thresholds"]["nearest_distance_error_ci_upper_max"] == 0.04
    assert verdict.gates["results"]["density_error_ci"]["passed"] is False
    assert verdict.gates["results"]["nearest_distance_error_ci"]["pooled_median"] == 0.0
    assert verdict.status == "FAIL"


def test_formal_publisher_cli_retries_with_fresh_restore_roots(tmp_path, capsys) -> None:
    """同じ release を異なる空 restore root から二度 publish できる。\n\n各 run の restore verdict logical id と resolved YAML が独立して発行される。"""
    import yaml
    from publish_survivors_world_detector_formal import main as publish_main

    checkpoint, verdict_path, config_path, class_map_path, _tracker, preflight = _formal_package_inputs(tmp_path)
    preflight_path = preflight.write(tmp_path / "preflight.json")
    dataset = ArtifactDescriptor(
        logical_id=preflight.dataset_logical_id,
        node_kind="source_descriptor",
        producer_id="synthetic-dataset",
        producer_version="world-dataset.v1",
        identity_metadata={"dataset_identity_hash": preflight.dataset_identity, "game_build_id": preflight.game_build_id},
    )
    dataset_path = tmp_path / "dataset-descriptor.json"
    dataset_path.write_text(json.dumps(dataset.to_wire()), encoding="utf-8")
    restore_ids = []
    for run_index in (1, 2):
        resolved_path = tmp_path / f"resolved-{run_index}.yaml"
        result = publish_main([
            "--verdict", str(verdict_path),
            "--preflight-report", str(preflight_path),
            "--checkpoint-dir", str(checkpoint),
            "--out-root", str(tmp_path / "packages"),
            "--artifact-store", str(tmp_path / "artifact-store"),
            "--backup-root", str(tmp_path / f"backup-{run_index}"),
            "--restore-root", str(tmp_path / f"restore-{run_index}"),
            "--dataset-descriptor", str(dataset_path),
            "--logical-prefix", "formal/retry-test",
            "--resolved-yaml-out", str(resolved_path),
            "--config", str(config_path),
            "--class-map", str(class_map_path),
        ])
        output = json.loads(capsys.readouterr().out)
        assert result == 0
        assert resolved_path.is_file()
        assert yaml.safe_load(resolved_path.read_text(encoding="utf-8"))["ignore_regions"] == preflight.resolved_config["input"]["ignore_regions"]
        restore_ids.append(output["restore_verdict_logical_id"])

    assert restore_ids[0] != restore_ids[1]


def test_synthetic_formal_preflight_train_eval_publish_restore_envelope_e2e(tmp_path, monkeypatch) -> None:
    """synthetic capture から preflight、formal CLI、package restore まで一続きで実行する。\n\ntrainer、in-process verdict、publisher envelope、benchmark restore contract を同一 artifact identity で確かめる。"""
    import yaml
    import torch
    import train_survivors_world_detector as training
    from eval_survivors_world_detector import main as eval_main
    from publish_survivors_world_detector_formal import main as publish_main
    from survivors.vision import world_detector as world_detector_module
    from survivors.vision import world_formal_sampling
    from survivors.vision.world_dataset import load_class_map
    from survivors.vision.world_detector import DetectionResult, WorldDetector
    from survivors.vision.world_detector_package import PackageManifest, restore_package
    from survivors.vision.world_formal_release import (
        publish_formal_detector_artifact,
        read_preflight_report,
        run_formal_preflight,
    )
    from survivors.artifact_restore_test import run_restore_test

    inputs = _formal_preflight_inputs(tmp_path)
    preflight = _run_preflight_fixture(inputs)
    checkpoint_dir = tmp_path / "checkpoint"
    report_path = preflight.write(checkpoint_dir / "preflight.json")
    class_map = load_class_map(inputs["class_map_path"])
    train_detector = type("TrainDetector", (), {})()
    train_detector._model = torch.nn.Linear(1, 1)
    train_detector.num_classes = class_map.num_classes
    monkeypatch.setattr(world_detector_module, "_TORCH_AVAILABLE", True)
    monkeypatch.setattr(WorldDetector, "from_config", staticmethod(lambda *args: train_detector))
    monkeypatch.setattr(world_formal_sampling, "class_weights", lambda *args, **kwargs: {})
    monkeypatch.setattr(world_formal_sampling, "attach_weighted_loss", lambda model, weights: model)

    def run_stub(detector, train_ds, validation_ds, cfg, *, max_epochs, selector, out_dir, **kwargs):
        """formal CLI の出力形式を保ったまま checkpoint を一つ作る。\n\nmodel の forward 学習だけを省き、preflight と checkpoint manifest の実経路を通す。"""
        checkpoint = out_dir / "epoch_0001.pt"
        torch.save(detector._model.state_dict(), checkpoint)
        selector.record(training.CheckpointRecord(1, 1.0, str(checkpoint)))
        detector._formal_epoch_metrics = [{"epoch": 1, "val_map50_95": 1.0}]

    monkeypatch.setattr(training, "_run_training", run_stub)
    train_result = training.main([
        "--annotations", str(inputs["annotations_path"]),
        "--formal",
        "--feasibility-verdict", str(inputs["feasibility_path"]),
        "--capture-store", str(inputs["capture_root"]),
        "--target-profile-hash", inputs["profile_hash"],
        "--hardware-id", inputs["hardware_id"],
        "--dataset-descriptor", str(inputs["dataset_descriptor_path"]),
        "--restore-verdict", str(inputs["restore_descriptor_path"]),
        "--annotation-agreement", str(inputs["agreement_path"]),
        "--minimum-counts", str(inputs["minimum_counts_path"]),
        "--preflight-report-out", str(report_path),
        "--config", str(inputs["config_path"]),
        "--class-map", str(inputs["class_map_path"]),
        "--output", str(checkpoint_dir),
        "--epochs", "1",
    ])
    assert train_result == 0
    assert (checkpoint_dir / "run_manifest.json").is_file()
    trained_preflight = read_preflight_report(report_path)
    assert trained_preflight.dataset_identity == preflight.dataset_identity

    boxes = []
    labels = []
    for class_index, category in enumerate(class_map.foreground_classes):
        x = 150 + (class_index % 5) * 300
        y = 250 + (class_index // 5) * 200
        boxes.append([x, y, x + 40, y + 40])
        labels.append(category["id"])
    eval_model = type("EvalModel", (), {
        "score_thresh": 0.0,
        "topk_candidates": 300,
        "detections_per_img": 100,
        "load_state_dict": lambda self, state, **kwargs: None,
        "eval": lambda self: None,
    })()
    eval_detector = type("EvalDetector", (), {
        "_model": eval_model,
        "infer": lambda self, frame, *, score_threshold: DetectionResult(
            np.asarray(boxes, dtype=np.float32),
            np.full(len(labels), 0.99, dtype=np.float32),
            np.asarray(labels, dtype=np.int32),
            1920,
            1080,
        ),
    })()
    monkeypatch.setattr(WorldDetector, "from_config", staticmethod(lambda *args: eval_detector))
    latency_path = tmp_path / "latency.json"
    latency_path.write_text(json.dumps(GpuLatencyMeasurement(
        1.0, 100, inputs["hardware_id"], "target_gpu", True,
    ).to_dict()), encoding="utf-8")
    gates_path = tmp_path / "gates.yaml"
    gates_path.write_text(yaml.safe_dump({
        "ci_resamples": 100,
        "boss_min_instances": 1,
        "boss_min_sessions": 3,
        "hazard_min_instances": 1,
        "hazard_min_sessions": 3,
    }), encoding="utf-8")
    verdict_path = tmp_path / "verdict.json"
    eval_result = eval_main([
        "--annotations", str(inputs["annotations_path"]),
        "--formal",
        "--preflight-report", str(report_path),
        "--checkpoint-dir", str(checkpoint_dir),
        "--latency", str(latency_path),
        "--gates", str(gates_path),
        "--config", str(inputs["config_path"]),
        "--class-map", str(inputs["class_map_path"]),
        "--verdict-out", str(verdict_path),
    ])
    assert eval_result == 0
    verdict = ValidationVerdict.from_dict(json.loads(verdict_path.read_text(encoding="utf-8")))
    assert verdict.status == "PASS"
    assert verdict.preflight_canonical_hash == preflight.preflight_canonical_hash

    resolved_path = tmp_path / "resolved.yaml"
    publish_result = publish_main([
        "--verdict", str(verdict_path),
        "--preflight-report", str(report_path),
        "--checkpoint-dir", str(checkpoint_dir),
        "--out-root", str(tmp_path / "packages"),
        "--artifact-store", str(tmp_path / "artifact-store"),
        "--backup-root", str(tmp_path / "publish-backup"),
        "--restore-root", str(tmp_path / "publish-restore"),
        "--dataset-descriptor", str(inputs["dataset_descriptor_path"]),
        "--logical-prefix", "formal/world-e2e",
        "--resolved-yaml-out", str(resolved_path),
        "--config", str(inputs["config_path"]),
        "--class-map", str(inputs["class_map_path"]),
    ])
    assert publish_result == 0
    resolved = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    assert resolved["ignore_regions"] == preflight.resolved_config["input"]["ignore_regions"]
    assert resolved["verdict_canonical_hash"] == verdict.verdict_canonical_hash

    package_manifest_path = next((tmp_path / "packages").glob("*/manifest.json"))
    package_manifest = PackageManifest.from_dict(json.loads(package_manifest_path.read_text(encoding="utf-8")))
    assert package_manifest.validation_verdict_hash == hashlib.sha256(
        (package_manifest_path.parent / "validation_verdict.json").read_bytes()
    ).hexdigest()
    restored_package = restore_package(
        package_manifest_path,
        np.zeros((1080, 1920, 3), dtype=np.uint8),
        require_formal=True,
    )
    assert restored_package is not None

    store = ArtifactStore(tmp_path / "artifact-store")
    descriptor = publish_formal_detector_artifact(
        store,
        package_dir=package_manifest_path.parent,
        preflight=trained_preflight,
        verdict=verdict,
        dataset_descriptor=inputs["dataset_descriptor"],
        logical_prefix="formal/world-e2e",
    )
    import benchmark_survivors_perception as benchmark
    manifest_ref = benchmark._manifest_ref(descriptor)
    envelope = json.loads(store.object_path(manifest_ref.store_uri).read_text(encoding="utf-8"))
    assert set(envelope) == benchmark._PACKAGE_REQUIRED_FIELDS["detector_package"]
    assert benchmark._manifest_ref(descriptor) == manifest_ref
    _, restore_descriptor = run_restore_test(
        store,
        descriptor,
        backup_root=tmp_path / "verify-backup",
        restore_root=tmp_path / "verify-restore",
        verdict_logical_id="formal/world-e2e/test-restore-verdict",
    )
    benchmark._validate_restore_verdict(descriptor, restore_descriptor)
