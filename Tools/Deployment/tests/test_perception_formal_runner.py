"""formal runner と typed UiPresentation replay の synthetic store fixture。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from Tools.Artifacts.artifact_store import ArtifactStore
from benchmark_survivors_perception import FormalBenchmarkRequest, run_formal_pipeline
from reinbalance_survivors_contracts.deploy_obs import DeployObsSchema
from survivors.capture.captured_frame import CapturedFrame
from survivors.capture_dataset import DatasetWriter, SplitFreezer
from survivors.perception_benchmark import SnapshotReplayTick
from survivors.perception_error_fit import CalibrationResidual
from survivors.real_obs_assembler import RealObsAssembler
from survivors.vision.entity_tracker import (
    PlayerAnchorState,
    TrackedEntityV1,
    TrackedWorldStateV1,
)
from survivors.vision.hud_parser import (
    HudStateV1,
    ParsedCard,
    _compute_candidate_set_hash,
    _compute_inventory_hash,
)


PROFILE_HASH = "a" * 64
PACKAGE_HASHES = {
    "assembler_schema_hash": "b" * 64,
    "ui_presentation_schema_hash": "c" * 64,
    "atlas_vocabulary_hash": "d" * 64,
    "assembler_impl_hash": "e" * 64,
    "roi_resolver_input_hash": "f" * 64,
    "threshold_hash": "1" * 64,
}


def _captured_frame(session_index: int, frame_index: int, timestamp_ns: int) -> CapturedFrame:
    pixels = np.zeros((1080, 1920, 4), dtype=np.uint8)
    pixels[0, 0] = (session_index, frame_index, 7, 255)
    return CapturedFrame(
        pixels, timestamp_ns, frame_index, (0, 0, 1920, 1080), True,
        PROFILE_HASH, "synthetic-build",
    )


def _capture_fixture(tmp_path: Path) -> tuple[Path, object]:
    capture_root = tmp_path / "captures"
    calibration_ids = [f"cal-{index}" for index in range(3)]
    final_ids = [f"final-{index}" for index in range(3)]
    for session_index, session_id in enumerate((*calibration_ids, *final_ids)):
        with DatasetWriter(capture_root, session_id, PROFILE_HASH, "synthetic-build") as writer:
            writer.write_frame(_captured_frame(session_index, 0, 1_000_000_000))
            writer.write_frame(_captured_frame(session_index, 1, 1_801_000_000_000))
            published = writer.publish(operator_checkpoint="synthetic-formal-fixture")
        # fixture は実 formal session を開封せず、manifest gate だけを synthetic に再現する。
        manifest_path = published.session_path / "session_manifest.json"
        wire = json.loads(manifest_path.read_text(encoding="utf-8"))
        wire["formal_dataset_eligible"] = True
        manifest_path.write_text(
            json.dumps(wire, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
    freezer = SplitFreezer(capture_root)
    freezer.assign("error_calibration", calibration_ids)
    freezer.assign("final_e2e_test", final_ids)
    return capture_root, freezer.freeze()


def _package_ref(store: ArtifactStore, logical_id: str, payload: dict[str, object]):
    return store.put_bytes(
        logical_id=logical_id,
        data=(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode(),
        media_type="application/json",
    )


def _dependency_refs(store: ArtifactStore, split) -> dict[str, object]:
    refs = {
        "capture_dataset": store.put(
            logical_id="fixtures/capture_split.json",
            source_path=split.manifest_path,
            media_type="application/json",
        ),
        "parser_package": _package_ref(store, "fixtures/parser.json", {
            "schema_version": "hud_parser_package.v1", "development_only": False,
            "formal_eligible": True,
            # _hud_world の HudStateV1.parser_artifact_hash と一致させる。
            "artifact_hash": "9" * 64,
        }),
        "detector_package": _package_ref(store, "fixtures/detector.json", {
            "schema_version": "world_detector_package.v1", "development_only": False,
            "formal_eligible": True,
            "artifact_hash": "2" * 64,
        }),
        "assembler_config": _package_ref(store, "fixtures/assembler.json", {
            "schema_version": "assembler_config.v1", "development_only": False,
            "formal_eligible": True, **{key: PACKAGE_HASHES[key] for key in (
                "assembler_schema_hash", "ui_presentation_schema_hash",
                "atlas_vocabulary_hash", "assembler_impl_hash", "roi_resolver_input_hash",
            )},
        }),
        "target_config": _package_ref(store, "fixtures/target.json", {
            "schema_version": "perception_thresholds.v1", "development_only": False,
            "formal_eligible": True, "threshold_hash": PACKAGE_HASHES["threshold_hash"],
        }),
    }
    return refs


def _hud_world(session_id: str, frame_index: int, timestamp_ns: int, screen_state: str):
    card = ParsedCard(0, "whip", "weapon", 2, 1.0, "fixture", (100, 100, 400, 500))
    inventory = ("whip",) + (None,) * 11
    # canonical hash を実際の計算関数で生成し、_recomputed_snapshot_hashes の照合に通す。
    real_inventory_hash = _compute_inventory_hash(inventory)
    real_candidate_set_hash = _compute_candidate_set_hash(screen_state, (card,))
    hud = HudStateV1(
        "hud_state.v1", session_id, frame_index, timestamp_ns, "9" * 64,
        screen_state, 1.0, "fixture", 20.0, 1.0, "fixture", False,
        0.75, 1.0, "fixture", 0.5, 1.0, "fixture", 4, 1.0, "fixture",
        inventory, 1.0, real_inventory_hash, (card,), real_candidate_set_hash,
        (), False, False, False, 1.0, "fixture",
    )
    entity = TrackedEntityV1(
        1, 2, "enemy_normal", "enemy", 1.0, 1, frame_index,
        0.7, 0.5, 0.2, 0.0, 0.0, 0.0, True, False,
    )
    world = TrackedWorldStateV1(
        frame_index, timestamp_ns, [entity], PlayerAnchorState(0.5, 0.5, 1.0, False)
    )
    return hud, world


def _calibration_provider(manifests, _restored):
    return [
        CalibrationResidual(
            manifest.session_id, f"f{index}", "hp_ratio", residual,
            1.0, 0, 1.0,
        )
        for manifest in manifests
        for index, residual in enumerate((-0.01, 0.01))
    ]


def _replay_provider(manifests, _restored):
    schema = DeployObsSchema.default_v1()
    ticks = []
    for manifest in manifests:
        assembler = RealObsAssembler()
        for frame_index, (timestamp, state) in enumerate((
            (1_000_000_000, "gameplay"),
            (1_801_000_000_000, "level_up_items"),
        )):
            snapshot = assembler.assemble(
                *_hud_world(manifest.session_id, frame_index, timestamp, state),
                schema, (1920, 1080),
            )
            assert snapshot is not None
            ticks.append(SnapshotReplayTick(
                manifest.session_id, "final_e2e_test", "lossless",
                snapshot.frame_id, snapshot, snapshot, 1.0,
            ))
    return ticks


def _request(tmp_path: Path):
    capture_root, split = _capture_fixture(tmp_path)
    store = ArtifactStore(tmp_path / "artifacts")
    request = FormalBenchmarkRequest(
        store=store,
        dependency_refs=_dependency_refs(store, split),
        capture_store_root=capture_root,
        calibration_provider=_calibration_provider,
        final_replay_provider=_replay_provider,
    )
    return request


def test_formal_runner_calls_real_pipeline_and_atomic_writers(tmp_path: Path) -> None:
    result = run_formal_pipeline(_request(tmp_path))

    assert result.profile.field_sample_counts == {"hp_ratio": 6}
    assert result.verdict.development_only is False
    assert result.verdict.formal_perception_verdict_eligible is True
    assert result.verdict.passed is True
    assert result.calibration_ref.sha256 == result.descriptors[1].files[0].sha256
    assert result.verdict_ref.sha256 == result.descriptors[2].files[0].sha256
    assert result.descriptors[1].node_kind == "perception_calibration_profile"
    assert result.descriptors[2].node_kind == "perception_final_verdict"


def test_dependency_tamper_fails_before_any_runner_output(tmp_path: Path) -> None:
    request = _request(tmp_path)
    before_indexes = set(request.store.logical_root.glob("*.json"))
    parser_ref = request.dependency_refs["parser_package"]
    request.store.object_path(parser_ref.store_uri).write_bytes(b"tampered")

    with pytest.raises(ValueError, match="verification failed"):
        run_formal_pipeline(request)

    assert set(request.store.logical_root.glob("*.json")) == before_indexes


def test_typed_replay_recomputes_ui_hashes_and_rejects_tamper(tmp_path: Path) -> None:
    request = _request(tmp_path)
    manifests = tuple(
        DatasetWriter.restore(request.capture_store_root, f"final-{index}")
        for index in range(3)
    )
    ticks = list(_replay_provider(manifests, {}))
    from dataclasses import replace
    from survivors.perception_benchmark import ExpectedTick, run_benchmark

    target = ticks[1].predicted.ui_presentation
    bad_ui = replace(target, source_content_hash="0" * 64)
    bad_snapshot = replace(ticks[1].predicted, source_content_hash="0" * 64, ui_presentation=bad_ui)
    ticks[1] = replace(ticks[1], predicted=bad_snapshot)
    expected = [ExpectedTick(tick.session_id, tick.frame_id) for tick in ticks]
    with pytest.raises(ValueError, match="source/content hash"):
        run_benchmark(ticks, expected_ticks=expected)
