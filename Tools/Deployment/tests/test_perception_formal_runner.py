"""formal runner と typed UiPresentation replay の synthetic store fixture。"""

from __future__ import annotations

import json
from dataclasses import replace
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
            # detector_artifact_hash は PerceptionSnapshot 側（SnapshotReplayTick ではない）へ設定する。
            snapshot = replace(snapshot, detector_artifact_hash="2" * 64)
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


def test_formal_runner_rejects_precomputed_benchmark_records(tmp_path: Path) -> None:
    """formal runner が SnapshotReplayTick 以外を返す provider を拒否する。

    run_formal_pipeline は typed snapshot のみを受理し、事前計算値では formal verdict を
    生成できない構造である必要があります。
    """
    import dataclasses

    request = _request(tmp_path)

    class _FakeTick:
        """SnapshotReplayTick でない偽オブジェクト（事前計算値の代わり）。"""

    def _bad_provider(manifests, restored):
        return [_FakeTick()]

    bad_request = dataclasses.replace(request, final_replay_provider=_bad_provider)
    with pytest.raises(TypeError, match="SnapshotReplayTick"):
        run_formal_pipeline(bad_request)


def test_candidate_set_hash_tamper_caught_by_raw_card_ids(tmp_path: Path) -> None:
    """candidate_set_hash 単独改ざんを raw_card_ids 独立検証で検出する。

    outer hash を更新せずに candidate_set_hash だけ差し替えても、
    raw_card_ids から再計算したハッシュと照合することで改ざんを検出します。
    """
    request = _request(tmp_path)
    manifests = tuple(
        DatasetWriter.restore(request.capture_store_root, f"final-{index}")
        for index in range(3)
    )
    ticks = list(_replay_provider(manifests, {}))
    from dataclasses import replace
    from survivors.perception_benchmark import ExpectedTick, run_benchmark

    target = ticks[0].ground_truth.ui_presentation
    # raw_card_ids は実値のまま、candidate_set_hash だけ偽値に差し替える。
    # raw_card_ids チェックが先に発火して coherent 偽値でも検出する。
    bad_ui = replace(target, candidate_set_hash="f" * 64)
    bad_snapshot = replace(ticks[0].ground_truth, ui_presentation=bad_ui)
    ticks[0] = replace(ticks[0], ground_truth=bad_snapshot)
    expected = [ExpectedTick(tick.session_id, tick.frame_id) for tick in ticks]
    with pytest.raises(ValueError, match="candidate_set_hash does not match typed card IDs"):
        run_benchmark(ticks, expected_ticks=expected)


def test_inventory_hash_tamper_caught_by_raw_inventory(tmp_path: Path) -> None:
    """inventory_hash 単独改ざんを raw_inventory 独立検証で検出する。"""
    request = _request(tmp_path)
    manifests = tuple(
        DatasetWriter.restore(request.capture_store_root, f"final-{index}")
        for index in range(3)
    )
    ticks = list(_replay_provider(manifests, {}))
    from survivors.perception_benchmark import ExpectedTick, run_benchmark

    target = ticks[0].ground_truth.ui_presentation
    bad_ui = replace(target, inventory_hash="e" * 64)
    bad_snapshot = replace(ticks[0].ground_truth, ui_presentation=bad_ui)
    ticks[0] = replace(ticks[0], ground_truth=bad_snapshot)
    expected = [ExpectedTick(tick.session_id, tick.frame_id) for tick in ticks]
    with pytest.raises(ValueError, match="inventory_hash does not match typed inventory"):
        run_benchmark(ticks, expected_ticks=expected)


def test_detector_artifact_hash_mismatch_rejected_by_formal_runner(tmp_path: Path) -> None:
    """formal runner が誤 detector_artifact_hash のスナップショットを拒否する。

    SnapshotReplayTick ではなく PerceptionSnapshot 内の detector_artifact_hash を
    検証するため、provider が後から hash を書き換えても検出できます。
    """
    import dataclasses

    request = _request(tmp_path)
    manifests = tuple(
        DatasetWriter.restore(request.capture_store_root, f"final-{index}")
        for index in range(3)
    )

    def _bad_detector_provider(final_manifests, restored):
        for tick in _replay_provider(final_manifests, restored):
            # detector_artifact_hash を誤値に差し替えた snapshot へ置き換える。
            bad_snapshot = replace(tick.ground_truth, detector_artifact_hash="f" * 64)
            yield replace(tick, ground_truth=bad_snapshot, predicted=bad_snapshot)

    bad_request = dataclasses.replace(request, final_replay_provider=_bad_detector_provider)
    with pytest.raises(ValueError, match="detector_artifact_hash"):
        run_formal_pipeline(bad_request)


def test_canonical_logical_ids_not_set_when_batch_commit_fails(tmp_path: Path) -> None:
    """batch commit（commit marker）失敗時に canonical logical_id が設定されないことを確認する。

    staging → batch commit → seal → canonical の順序で publish するため、
    batch commit が失敗しても consumer から見える canonical logical_id が残りません。
    batch commit は単一の put_bytes であり、これが唯一の commit 点です。
    """
    import dataclasses
    from unittest.mock import patch

    from Tools.Artifacts.artifact_store import ArtifactStoreError

    request = _request(tmp_path)
    canonical_cal = request.calibration_logical_id
    canonical_ver = request.verdict_logical_id

    original_put_bytes = request.store.put_bytes

    def _fail_on_batch_commit(*, logical_id, data, media_type):
        if logical_id.startswith("perception/batch_commit/"):
            raise ArtifactStoreError("simulated batch commit failure")
        return original_put_bytes(logical_id=logical_id, data=data, media_type=media_type)

    with patch.object(request.store, "put_bytes", side_effect=_fail_on_batch_commit):
        with pytest.raises(ArtifactStoreError, match="simulated batch commit failure"):
            run_formal_pipeline(request)

    # canonical logical_id の index が存在しないことを確認する。
    # ArtifactStore は logical_id を logical_root 以下のパスへ記録する。
    cal_index = request.store._logical_index_path(canonical_cal)
    ver_index = request.store._logical_index_path(canonical_ver)
    assert not cal_index.exists(), "calibration canonical logical_id must not be set before batch commit"
    assert not ver_index.exists(), "verdict canonical logical_id must not be set before batch commit"
