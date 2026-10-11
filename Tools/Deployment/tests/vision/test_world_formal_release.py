"""formal release metric と threshold の小さな既知例を固定する。"""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
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
    """formal validation metadata survives COCO loading into DatasetSample annotations.

    Track switch and late-game gates must not silently lose source annotation fields.
    """
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
    """formal preflight rejects descriptor markers before opening capture or annotation files.

    A source descriptor cannot self-label a development or synthetic export as formal.
    """
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
    """evaluation annotation file must match the preflight-bound COCO export.

    Replacing model_validation input after training must not change formal verdict parents.
    """
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
    """formal eval reads the checkpoint weight, infers raw scores and binds its verdict.

    External predictions are absent and the detector receives score_threshold=0.0.
    """
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
        """synthetic validation view returning sealed session-bound samples.

        Constructor compatibility matches the WorldDataset eval call.
        """
        def __init__(self, *args, **kwargs):
            """store no paths because frame decoding is supplied by the test.

            The preflight report already fixes the allowed session identifiers.
            """
            pass

        def __iter__(self):
            """yield each model_validation session once.

            Every synthetic sample carries all foreground classes for stable gates.
            """
            return iter(samples)

    class FakeDetector:
        """return exact synthetic detections while recording the raw-score threshold.

        The fake model exposes the same inference caps as torchvision SSD.
        """
        def __init__(self):
            """initialize the state-dict sink and observable score threshold.

            Runtime inference does not require torchvision model weights in this test.
            """
            self._model = SimpleNamespace(
                score_thresh=0.0,
                topk_candidates=300,
                detections_per_img=100,
                load_state_dict=lambda state, **kwargs: None,
                eval=lambda: None,
            )
            self.thresholds = []

        def infer(self, frame, *, score_threshold):
            """return exact boxes for one synthetic validation frame.

            Record caller threshold to prove the formal path preserves raw scores.
            """
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
