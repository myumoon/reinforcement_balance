"""formal runner と typed UiPresentation replay の synthetic store fixture。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from Tools.Artifacts.artifact_store import ArtifactStore
from benchmark_survivors_perception import (
    FormalBenchmarkRequest,
    RestoredFormalArtifact,
    _annotation_evidence_index,
    _calibration_descriptor_chain,
    _calibration_provenance_bytes,
    _descriptor_chain,
    _derive_calibration_residuals,
    _ground_truth_semantic_hash,
    _put_descriptor_file,
    _replay_sessions,
    _recover_committed_result,
    _subject_hashes,
    _verified_annotation_payloads,
    run_formal_pipeline,
)
from reinbalance_survivors_contracts.artifact_identity import ArtifactDescriptor
from reinbalance_survivors_contracts.canonical_json import canonical_hash, canonical_json_bytes
from reinbalance_survivors_contracts.deploy_obs import DeployObsSchema
from survivors.capture.captured_frame import CapturedFrame
from survivors.capture_dataset import DatasetWriter, SplitFreezer
from survivors.perception_benchmark import (
    SnapshotReplayTick,
    _foreground_classification_values,
    recompute_gate_from_metrics,
    replay_snapshots,
    run_benchmark,
)
from survivors.perception_error_fit import (
    _FORMAL_FACTORY_TOKEN,
    _HASH_FIELDS,
    CalibrationResidual,
    FinalSessionAlreadyOpenedError,
    FittedPerceptionErrorProfile,
    PerceptionFinalVerdict,
    _create_formal_lineage_seal,
    _load_formal_verdict_from_verified_store,
)
from survivors.perception_session_split import SessionRecord, SessionSplit, validate_split
from survivors.perception_snapshot import FormalReplayEvidence
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
    pixels[0, 0] = (session_index, frame_index % 256, 7, 255)
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
            for frame_index, timestamp_ns in enumerate(_TIMESTAMPS):
                writer.write_frame(_captured_frame(session_index, frame_index, timestamp_ns))
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


def _hud_world(
    session_id: str, frame_index: int, timestamp_ns: int, screen_state: str,
    parser_artifact_hash: str = "9" * 64,
    card_roi: tuple[int, int, int, int] = (100, 100, 400, 500),
):
    card = ParsedCard(0, "whip", "weapon", 2, 1.0, "fixture", card_roi)
    inventory = ("whip",) + (None,) * 11
    # canonical hash を実際の計算関数で生成し、_recomputed_snapshot_hashes の照合に通す。
    real_inventory_hash = _compute_inventory_hash(inventory)
    real_candidate_set_hash = _compute_candidate_set_hash(screen_state, (card,))
    elapsed_time = _ELAPSED_TIMES[frame_index] if frame_index < len(_ELAPSED_TIMES) else 20.0
    hud = HudStateV1(
        "hud_state.v1", session_id, frame_index, timestamp_ns, parser_artifact_hash,
        screen_state, 1.0, "fixture", elapsed_time, 1.0, "fixture", False,
        0.75, 1.0, "fixture", 0.5, 1.0, "fixture", 4, 1.0, "fixture",
        inventory, 1.0, real_inventory_hash, (card,), real_candidate_set_hash,
        (), False, False, False, 1.0, "fixture",
    )
    boss = TrackedEntityV1(
        1, 2, "enemy_boss", "enemy", 1.0, 1, frame_index,
        0.7, 0.5, 0.2, 0.0, 0.0, 0.0, True, False,
    )
    hazard = TrackedEntityV1(
        2, 3, "hazard_area", "hazard", 1.0, 1, frame_index,
        0.6, 0.5, 0.1, 0.0, 0.0, 0.0, True, False,
    )
    world = TrackedWorldStateV1(
        frame_index, timestamp_ns, [boss, hazard], PlayerAnchorState(0.5, 0.5, 1.0, False)
    )
    return hud, world


_CALIBRATION_REQUIRED_FIELDS = frozenset({
    "coord_noise", "hp_ratio", "xp_ratio", "timer_seconds",
    "inventory_hash", "coord_quantization_px",
    "burst_enter", "burst_exit", "burst_dropout",
    "unknown_screen_collapse", "unknown_screen_collapse_duration",
    "item_category", "enemy_category",
})
_CALIBRATION_SAMPLES_PER_FIELD_PER_SESSION = 3  # formal 要件（3 samples × ≥3 sessions）を満たす最小値


def _calibration_provider(manifests, _restored):
    """formal=True で fit_error_profile が必須 field をすべて受理できる残差を返す。

    各 field で 3 サンプル × manifests（cal sessions）分の残差を生成します。
    item_category / enemy_category は confusion matrix 用に category を設定します。
    """
    rows = []
    for manifest in manifests:
        sid = manifest.session_id
        for field in _CALIBRATION_REQUIRED_FIELDS:
            for frame_idx in range(_CALIBRATION_SAMPLES_PER_FIELD_PER_SESSION):
                gt_cat = (frame_idx % 3) if "category" in field else None
                pred_cat = (frame_idx % 3) if "category" in field else None
                rows.append(CalibrationResidual(
                    sid, f"{field}-{frame_idx}", field, 0.01,
                    1.0, 0, 0.0, gt_cat, pred_cat,
                ))
    return rows


# 4 フレーム × 3 final sessions = ≥ 6 records/slice を達成するフィクスチャ
# （_NAMED_SLICE_RECORD_FLOORS のテスト用 floor 値に合わせた最小設定）
# frame_index: 0=gameplay/early, 1=level_up, 2=gameplay/late, 3=level_up → 各スライス 3×2=6 records
_TIMESTAMPS = [1_000_000_000, 2_000_000_000, 1_800_000_000_000, 1_801_000_000_000]
_STATES = ["gameplay", "level_up_items", "gameplay", "level_up_fallback"]
_ELAPSED_TIMES = [20.0, 20.0, 1300.0, 1300.0]


class _MockAssembler:
    """テスト用の FormalAssembler：CapturedFrame の session_frame_index を使って合成観測を生成する。

    セッション間でタイムスタンプが逆転するため、セッションごとに独立した RealObsAssembler を保持する。
    FormalReplayEvidence として raw_card_ids / raw_inventory を HUD から抽出して返します。
    """

    def __init__(self, parser_artifact_hash: str = "9" * 64) -> None:
        self._assemblers: dict[str, RealObsAssembler] = {}
        self._schema = DeployObsSchema.default_v1()
        self._parser_artifact_hash = parser_artifact_hash

    def assemble_frame(self, frame, session_id: str, frame_index: int):
        if frame_index >= len(_TIMESTAMPS):
            return None, None, 0.0, None
        if session_id not in self._assemblers:
            self._assemblers[session_id] = RealObsAssembler()
        hud, world = _hud_world(
            session_id, frame_index, _TIMESTAMPS[frame_index], _STATES[frame_index],
            self._parser_artifact_hash,
        )
        snapshot = self._assemblers[session_id].assemble(hud, world, self._schema, (1920, 1080))
        assert snapshot is not None
        evidence = FormalReplayEvidence(
            raw_card_ids=tuple(c.item_id for c in sorted(hud.cards, key=lambda c: c.slot_index)),
            raw_inventory=hud.inventory,
        )
        # synthetic fixture は正解と予測の内容を一致させる。
        # runner が別入力として扱う契約を守るため、両方を明示的に返す。
        return snapshot, snapshot, 1.0, evidence


def _assembler_factory(restored):
    """テスト用の FormalAssemblerFactory：runner に渡す mock assembler を返す。"""
    # fixture も自己申告値でなく、restored package object の content hash を使用する。
    # runner/verdict と同一 identity を snapshot の parser binding に設定する。
    parser_content_hash = hashlib.sha256(restored["parser_package"]).hexdigest()
    return _MockAssembler(parser_content_hash)


def _default_cli_factory(restored):
    """テスト用の _CLI_PROVIDER_FACTORY：default calibration provider と mock assembler を返す。"""
    return _calibration_provider, _assembler_factory


def _request(tmp_path: Path):
    capture_root, split = _capture_fixture(tmp_path)
    store = ArtifactStore(tmp_path / "artifacts")
    return FormalBenchmarkRequest(
        store=store,
        capture_store_root=capture_root,
    )


def test_formal_entry_rejects_ref_only_synthetic_fixture_before_provider(
    tmp_path: Path, monkeypatch
) -> None:
    """ArtifactDescriptor/restore verdict のない synthetic fixture は formal に入れない。"""
    import benchmark_survivors_perception as module

    called = False

    def forbidden_factory(_restored):
        nonlocal called
        called = True
        raise AssertionError("provider must not run")

    monkeypatch.setattr(module, "_CLI_PROVIDER_FACTORY", forbidden_factory)
    request = _request(tmp_path)
    before = set(request.store.logical_root.glob("*.json"))
    with pytest.raises(ValueError, match="descriptors/restore verdicts"):
        run_formal_pipeline(request)
    assert called is False
    assert set(request.store.logical_root.glob("*.json")) == before


def test_final_metadata_preflight_does_not_open_or_hash_png(tmp_path: Path) -> None:
    capture_root, _split = _capture_fixture(tmp_path)
    full = DatasetWriter.restore(capture_root, "final-0")
    png_path = full.session_path / full.frame_records[0].object_path
    png_path.unlink()

    metadata = DatasetWriter.restore(capture_root, "final-0", metadata_only=True)

    assert metadata.frame_records == full.frame_records
    assert metadata.frames == ()
    with pytest.raises(ValueError, match="missing object"):
        DatasetWriter.restore(capture_root, "final-0")


def test_formal_predictor_cannot_return_ground_truth_residual_or_evidence(
    tmp_path: Path,
) -> None:
    """旧4-tuple provider output は prediction-only 境界で拒否する。"""
    import benchmark_survivors_perception as module

    with pytest.raises(TypeError, match="prediction, latency_ms"):
        module._unpack_prediction_result((None, None, 1.0, FormalReplayEvidence((), ())))


@pytest.mark.skip(reason="formal release descriptors/sessions are intentionally unavailable")
def test_formal_runner_calls_real_pipeline_and_atomic_writers(tmp_path: Path, monkeypatch) -> None:
    import benchmark_survivors_perception as _bsp
    monkeypatch.setattr(_bsp, "_CLI_PROVIDER_FACTORY", _default_cli_factory)
    result = run_formal_pipeline(_request(tmp_path))

    # 3 cal sessions × 3 samples/session × 7 required fields
    expected_counts = {field: 9 for field in _CALIBRATION_REQUIRED_FIELDS}
    assert result.profile.field_sample_counts == expected_counts
    assert result.verdict.development_only is False
    assert result.verdict.formal_perception_verdict_eligible is True
    assert result.verdict.passed is True, result.verdict.blocking_reasons
    assert result.calibration_ref.sha256 == result.descriptors[1].files[0].sha256
    assert result.verdict_ref.sha256 == result.descriptors[2].files[0].sha256
    assert result.descriptors[1].node_kind == "perception_calibration_profile"
    assert result.descriptors[2].node_kind == "perception_final_verdict"
    assert result.verdict.metrics["slice_counts"]["time_band:early"] == 3
    assert result.verdict.metrics["slice_counts"]["event:hazard"] == 6

    # 必須 named slice の欠落と rare slice の最低件数未満を再計算 gate が拒否する。
    # persisted metrics 改変でも同じ blocking 判定になることを確認する。
    missing = dict(result.verdict.metrics)
    missing["slice_counts"] = dict(missing["slice_counts"])
    del missing["slice_counts"]["event:boss"]
    passed, reasons = recompute_gate_from_metrics(missing)
    assert passed is False
    assert any("event:boss" in reason and "0 records" in reason for reason in reasons)
    underpowered = dict(result.verdict.metrics)
    underpowered["slice_counts"] = dict(underpowered["slice_counts"])
    underpowered["slice_counts"]["event:boss"] = 1
    passed, reasons = recompute_gate_from_metrics(underpowered)
    assert passed is False
    assert any("event:boss" in reason and "1 records" in reason for reason in reasons)


@pytest.mark.skip(reason="formal release descriptors/sessions are intentionally unavailable")
def test_published_calibration_profile_is_consumer_compatible(tmp_path: Path, monkeypatch) -> None:
    """canonical logical_id に保存した profile が PerceptionErrorProfile.from_wire() で読み込める。

    Training consumer は to_artifact_wire() ラッパーではなく to_wire() を期待する（#8 回帰防止）。
    """
    from reinbalance_survivors_contracts.perception_error import PerceptionErrorProfile

    import benchmark_survivors_perception as _bsp
    monkeypatch.setattr(_bsp, "_CLI_PROVIDER_FACTORY", _default_cli_factory)
    request = _request(tmp_path)
    result = run_formal_pipeline(request)

    cal_index = request.store._logical_index_path(request.calibration_logical_id)
    from Tools.Artifacts.artifact_store import ArtifactRef
    import json
    cal_ref_wire = json.loads(cal_index.read_text("utf-8"))
    cal_bytes = request.store.object_path(cal_ref_wire["store_uri"]).read_bytes()
    wire = json.loads(cal_bytes)
    # PerceptionErrorProfile.from_wire() が例外を投げないことを確認する（schema互換性）
    profile = PerceptionErrorProfile.from_wire(wire)
    assert profile is not None


def test_committed_batch_second_recovery_uses_profile_artifact_envelope(
    tmp_path: Path,
) -> None:
    """producer descriptorの実payloadから2回復旧してもformal profileを再発行できる。"""
    from test_perception_formal_guard import _passing_formal_verdict_wire

    store = ArtifactStore(tmp_path / "artifacts")
    capture_ref = store.put_bytes(
        logical_id="fixtures/capture.json",
        data=b"{}",
        media_type="application/json",
    )
    target_ref = store.put_bytes(
        logical_id="fixtures/target.json",
        data=b"{}",
        media_type="application/json",
    )

    def source_descriptor(logical_id, ref):
        """fixture refを最小source descriptorへ束縛する。"""
        return ArtifactDescriptor(
            logical_id=logical_id,
            node_kind="source_descriptor",
            producer_id="pytest",
            producer_version="v1",
            identity_metadata={"content_hash": ref.sha256},
            files=(ref,),
        )

    request = FormalBenchmarkRequest(
        store=store,
        capture_store_root=tmp_path / "captures",
        dependency_descriptors={
            "capture_dataset": source_descriptor("capture", capture_ref),
            "target_config": source_descriptor("target", target_ref),
        },
    )
    payloads = {
        "parser_package": {"parser_artifact_hash": "1" * 64},
        "detector_package": {
            "detector_artifact_hash": "2" * 64,
            "model_hash": "3" * 64,
            "build_hash": "4" * 64,
        },
        "assembler_config": {
            "assembler_schema_hash": "5" * 64,
            "ui_presentation_schema_hash": "6" * 64,
            "ui_presentation_golden_fixture_hash": "7" * 64,
            "atlas_vocabulary_hash": "8" * 64,
            "assembler_impl_hash": "9" * 64,
            "roi_resolver_input_hash": "a" * 64,
        },
        "target_config": {"threshold_hash": "b" * 64},
    }
    profile = FittedPerceptionErrorProfile(
        calibration_session_ids=["cal-0"],
        final_e2e_session_ids=["final-0"],
        calibration_session_hashes={"cal-0": "c" * 64},
        field_sample_counts={"hp_ratio": 2},
        fit_code_hash="d" * 64,
        development_only=False,
        _factory_token=_FORMAL_FACTORY_TOKEN,
    )
    run_key = "e" * 64
    provisional = _subject_hashes(
        request,
        payloads,
        calibration_profile_hash="0" * 64,
        lineage_seal_hash="0" * 64,
    )
    provenance_subjects = {
        name: value
        for name, value in provisional.items()
        if name not in {"calibration_profile_hash", "lineage_seal_hash"}
    }
    calibration_descriptors = _calibration_descriptor_chain(
        request,
        profile,
        provenance_subjects,
        request.calibration_logical_id(run_key),
    )
    profile_node = calibration_descriptors[1]
    raw_file = next(ref for ref in profile_node.files if ref.logical_id.endswith("/profile.json"))
    artifact_file = next(
        ref for ref in profile_node.files if ref.logical_id.endswith("/profile.artifact.json")
    )
    provenance_file = next(
        ref for ref in profile_node.files if ref.logical_id.endswith("/provenance.json")
    )
    profile_refs = [
        _put_descriptor_file(store, raw_file, canonical_json_bytes(profile.to_wire())),
        _put_descriptor_file(
            store, artifact_file, canonical_json_bytes(profile.to_artifact_wire())
        ),
        _put_descriptor_file(
            store,
            provenance_file,
            _calibration_provenance_bytes(profile, provenance_subjects),
        ),
    ]

    final_hash = "f" * 64
    preseal_subjects = _subject_hashes(
        request,
        payloads,
        calibration_profile_hash=profile_node.identity_hash,
        lineage_seal_hash="0" * 64,
    )
    seal = _create_formal_lineage_seal(
        final_session_hashes={"final-0": final_hash},
        store=store,
        _publish=False,
        **{
            name: value
            for name, value in preseal_subjects.items()
            if name != "lineage_seal_hash"
        },
    )
    seal_ref = store.put_bytes(
        logical_id=f"perception/package/lineage/{seal.seal_id}/seal.json",
        data=canonical_json_bytes(seal.to_wire()),
        media_type="application/json",
    )
    subjects = _subject_hashes(
        request,
        payloads,
        calibration_profile_hash=profile_node.identity_hash,
        lineage_seal_hash=seal_ref.sha256,
    )
    verdict_wire = _passing_formal_verdict_wire(subjects)
    verdict_wire["seal_id"] = seal.seal_id
    verdict_wire["verdict_id"] = canonical_hash({
        "seal_id": seal.seal_id,
        "final_session_ids": verdict_wire["final_session_ids"],
        "subject_hashes": {name: verdict_wire[name] for name in _HASH_FIELDS},
        "metrics": verdict_wire["metrics"],
    })
    stored_verdict = store.put_bytes(
        logical_id="fixtures/verdict.json",
        data=canonical_json_bytes(verdict_wire),
        media_type="application/json",
    )
    verdict = _load_formal_verdict_from_verified_store(
        verdict_wire,
        store=store,
        ref=stored_verdict,
        current_subject_hashes=subjects,
    )
    assert isinstance(verdict, PerceptionFinalVerdict)
    descriptors = _descriptor_chain(
        calibration_descriptors,
        request,
        verdict,
        request.verdict_logical_id(run_key),
    )
    verdict_ref = _put_descriptor_file(
        store, descriptors[2].files[0], canonical_json_bytes(verdict_wire)
    )
    refs = [capture_ref, *profile_refs, seal_ref, verdict_ref]
    store.put_bytes(
        logical_id=f"perception/batch_commit/{run_key}",
        data=canonical_json_bytes({
            "schema_version": "perception_formal_batch_commit.v2",
            "run_key": run_key,
            "descriptors": [descriptor.to_wire() for descriptor in descriptors],
            "descriptor_hashes": [descriptor.identity_hash for descriptor in descriptors],
            "refs": [ref.to_wire() for ref in refs],
            "profile_artifact": profile.to_artifact_wire(),
            "verdict": verdict_wire,
        }),
        media_type="application/json",
    )
    split = SessionSplit((SessionRecord(
        session_id="final-0",
        session_hash=final_hash,
        build_hash="4" * 64,
        target_profile_hash="0" * 64,
        resolution_wh=(1920, 1080),
        duration_seconds=1800.0,
        kind="final_e2e_test",
        source_policy="lossless",
    ),))

    first = _recover_committed_result(request, split, run_key, payloads)
    second = _recover_committed_result(request, split, run_key, payloads)

    assert first is not None and second is not None
    assert second.profile.to_artifact_wire() == profile.to_artifact_wire()
    assert second.calibration_artifact_ref == first.calibration_artifact_ref
    assert store.resolve(request.calibration_logical_id(run_key)) == second.calibration_ref
    assert json.loads(store.object_path(second.calibration_ref.store_uri).read_bytes()) == (
        profile.to_wire()
    )
    assert json.loads(
        store.object_path(second.calibration_artifact_ref.store_uri).read_bytes()
    ) == profile.to_artifact_wire()


@pytest.mark.skip(reason="formal release descriptors/sessions are intentionally unavailable")
def test_formal_runner_blocks_when_prediction_differs_from_ground_truth(tmp_path: Path, monkeypatch) -> None:
    """予測 HP を大きく外すと自己比較せず formal gate が公開を拒否する。

    UI hash は変更しないため、同じ raw evidence による gt/pred 双方の独立検証と
    HP 精度 gate の失敗を同時に確認できます。
    """
    class _WrongHpAssembler:
        """正解 snapshot を維持し、予測側の HP だけを誤らせる wrapper。"""

        def __init__(self, parser_artifact_hash: str) -> None:
            self._inner = _MockAssembler(parser_artifact_hash)

        def assemble_frame(self, frame, session_id: str, frame_index: int):
            ground_truth, predicted, latency, evidence = self._inner.assemble_frame(
                frame, session_id, frame_index
            )
            if predicted is not None and predicted.item_context is not None:
                predicted = replace(
                    predicted,
                    item_context=replace(predicted.item_context, hp_ratio=0.0),
                )
            return ground_truth, predicted, latency, evidence

    def _wrong_hp_factory(restored):
        return _calibration_provider, lambda r: _WrongHpAssembler(
            hashlib.sha256(r["parser_package"]).hexdigest()
        )

    import benchmark_survivors_perception as _bsp
    monkeypatch.setattr(_bsp, "_CLI_PROVIDER_FACTORY", _wrong_hp_factory)
    request = _request(tmp_path)
    with pytest.raises(ValueError, match="HP MAE outside formal threshold"):
        run_formal_pipeline(request)

    # gate 不合格時は canonical verdict を公開しない。
    # 自己比較で pass していた旧 runner への直接的な回帰防止です。
    assert not request.store._logical_index_path(request.verdict_logical_id).exists()


def test_replay_excludes_zero_area_ground_rois_from_geometry_records() -> None:
    """幅ゼロ・高さゼロの各 ground ROI を実 replay record 経路で除外する。

    対応 predicted が usable でも geometry 成功値は生成せず、余剰予測として
    ui_false_positive に記録されることを確認します。
    """
    schema = DeployObsSchema.default_v1()
    ticks: list[SnapshotReplayTick] = []
    for index, ground_roi in enumerate(((200, 100, 200, 500), (100, 200, 400, 200))):
        session_id = f"zero-roi-{index}"
        ground_hud, world = _hud_world(
            session_id, 0, 1_000_000_000, "level_up_items", card_roi=ground_roi
        )
        predicted_hud, _ = _hud_world(
            session_id, 0, 1_000_000_000, "level_up_items"
        )
        ground_truth = RealObsAssembler().assemble(
            ground_hud, world, schema, (1920, 1080)
        )
        predicted = RealObsAssembler().assemble(
            predicted_hud, world, schema, (1920, 1080)
        )
        assert ground_truth is not None and predicted is not None
        ticks.append(SnapshotReplayTick(
            session_id, "final_e2e_test", "lossless",
            ground_truth.frame_id, ground_truth, predicted, 1.0,
        ))

    records = replay_snapshots(ticks)
    geometry = {
        record.field for record in records
        if record.field in {"ui_inside_region", "ui_roi_center_error"}
    }
    false_positives = [
        record for record in records if record.field == "ui_false_positive"
    ]
    assert geometry == set()
    assert len(false_positives) == 2
    assert all(record.predicted is True for record in false_positives)


@pytest.mark.skip(reason="formal release descriptors/sessions are intentionally unavailable")
def test_dependency_tamper_fails_before_any_runner_output(tmp_path: Path, monkeypatch) -> None:
    import benchmark_survivors_perception as _bsp
    monkeypatch.setattr(_bsp, "_CLI_PROVIDER_FACTORY", _default_cli_factory)
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
    ticks, evidence = _run_assembler_for_test(manifests)
    from survivors.perception_benchmark import ExpectedTick, run_benchmark

    target = ticks[1].predicted.ui_presentation
    bad_ui = replace(target, source_content_hash="0" * 64)
    bad_snapshot = replace(ticks[1].predicted, source_content_hash="0" * 64, ui_presentation=bad_ui)
    ticks[1] = replace(ticks[1], predicted=bad_snapshot)
    expected = [ExpectedTick(tick.session_id, tick.frame_id) for tick in ticks]
    with pytest.raises(ValueError, match="source/content hash"):
        run_benchmark(ticks, expected_ticks=expected)


def _run_assembler_for_test(
    manifests: "Sequence[Any]", detector_hash: str = "2" * 64
) -> "tuple[list[SnapshotReplayTick], dict[str, FormalReplayEvidence]]":
    """unit test 用：mock assembler を使って SnapshotReplayTick リストと evidence を生成する。

    run_formal_pipeline が内部で行うのと同等の tick 生成を再現します。
    """
    assembler = _MockAssembler()
    ticks = []
    evidence: dict[str, FormalReplayEvidence] = {}
    for manifest in manifests:
        for frame in manifest.frames:
            ground_truth, predicted, latency, ev = assembler.assemble_frame(
                frame, manifest.session_id, frame.session_frame_index
            )
            if ground_truth is None:
                continue
            ground_truth = replace(
                ground_truth, detector_artifact_hash=detector_hash
            )
            if predicted is not None:
                predicted = replace(predicted, detector_artifact_hash=detector_hash)
            if ev is not None:
                evidence[ground_truth.frame_id] = ev
            ticks.append(SnapshotReplayTick(
                manifest.session_id, "final_e2e_test", "lossless",
                ground_truth.frame_id, ground_truth, predicted, latency,
            ))
    return ticks, evidence


@pytest.mark.skip(reason="formal release descriptors/sessions are intentionally unavailable")
def test_formal_runner_assembler_cannot_override_detector_hash(tmp_path: Path, monkeypatch) -> None:
    """assembler が任意 detector hash を設定しても runner が restored package hash で上書きする。

    verdict の detector_artifact_hash は ArtifactStore 内の detector package ファイル自体の
    SHA-256（_subject_hashes 経由）であり、assembler の自己申告とは無関係です。
    """
    import benchmark_survivors_perception as _bsp
    monkeypatch.setattr(_bsp, "_CLI_PROVIDER_FACTORY", _default_cli_factory)
    request = _request(tmp_path)
    result = run_formal_pipeline(request)
    # verdict.detector_artifact_hash = ArtifactStore 内 detector_package JSON の sha256
    expected = request.dependency_refs["detector_package"].sha256
    assert result.verdict.detector_artifact_hash == expected


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
    ticks, evidence = _run_assembler_for_test(manifests)
    from dataclasses import replace
    from survivors.perception_benchmark import ExpectedTick, run_benchmark

    target = ticks[0].ground_truth.ui_presentation
    # evidence（raw_card_ids）は実値のまま、candidate_set_hash だけ偽値に差し替える。
    # formal_evidence を渡すと evidence check が先に発火して coherent 偽値でも検出する。
    bad_ui = replace(target, candidate_set_hash="f" * 64)
    bad_snapshot = replace(ticks[0].ground_truth, ui_presentation=bad_ui)
    ticks[0] = replace(ticks[0], ground_truth=bad_snapshot)
    expected = [ExpectedTick(tick.session_id, tick.frame_id) for tick in ticks]
    with pytest.raises(ValueError, match="candidate_set_hash does not match formal evidence card IDs"):
        run_benchmark(ticks, expected_ticks=expected, formal_evidence=evidence)


def test_inventory_hash_tamper_caught_by_raw_inventory(tmp_path: Path) -> None:
    """inventory_hash 単独改ざんを raw_inventory 独立検証で検出する。"""
    request = _request(tmp_path)
    manifests = tuple(
        DatasetWriter.restore(request.capture_store_root, f"final-{index}")
        for index in range(3)
    )
    ticks, evidence = _run_assembler_for_test(manifests)
    from survivors.perception_benchmark import ExpectedTick, run_benchmark

    target = ticks[0].ground_truth.ui_presentation
    bad_ui = replace(target, inventory_hash="e" * 64)
    bad_snapshot = replace(ticks[0].ground_truth, ui_presentation=bad_ui)
    ticks[0] = replace(ticks[0], ground_truth=bad_snapshot)
    expected = [ExpectedTick(tick.session_id, tick.frame_id) for tick in ticks]
    with pytest.raises(ValueError, match="inventory_hash does not match formal evidence inventory"):
        run_benchmark(ticks, expected_ticks=expected, formal_evidence=evidence)


def test_annotation_evidence_is_not_applied_to_prediction(
    tmp_path: Path, monkeypatch
) -> None:
    """annotation raw evidence は ground truth にだけ照合する。"""
    import survivors.perception_benchmark as module

    request = _request(tmp_path)
    manifests = (DatasetWriter.restore(request.capture_store_root, "final-0"),)
    ticks, evidence = _run_assembler_for_test(manifests)
    calls: list[str] = []
    monkeypatch.setattr(
        module,
        "_verify_formal_evidence",
        lambda _ui, _evidence, label: calls.append(label),
    )
    module.replay_snapshots(ticks, formal_evidence=evidence)
    assert calls == [f"ground/{tick.frame_id}" for tick in ticks]


def test_all_foreground_classes_use_entity_semantic_correctness(tmp_path: Path) -> None:
    request = _request(tmp_path)
    manifest = DatasetWriter.restore(request.capture_store_root, "final-0")
    frame = manifest.frames[0]
    ground, predicted, _latency, _evidence = _MockAssembler().assemble_frame(
        frame, manifest.session_id, frame.session_frame_index
    )
    assert ground is not None and predicted is not None
    classes = (
        "player_anchor", "enemy_normal", "enemy_elite", "enemy_boss",
        "gem_blue", "gem_green", "gem_red", "pickup_heal",
        "pickup_special", "hazard_projectile", "hazard_area",
    )
    for class_name in classes:
        wrong_class = "enemy_elite" if class_name == "enemy_normal" else "enemy_normal"
        annotated = replace(
            ground,
            diagnostics={
                "foreground_entity_counts": {class_name: 1},
                "foreground_entity_classes": {"entity-0": class_name},
            },
        )
        wrong = replace(
            predicted,
            diagnostics={"foreground_entity_classes": {"entity-0": wrong_class}},
        )
        values = _foreground_classification_values(annotated, wrong)
        assert values[f"foreground_class:{class_name}"] == [0.0]


def test_loader_cannot_self_authenticate_different_ground_truth(tmp_path: Path) -> None:
    """loader が evidence を返せず、immutable annotation の GT hash と不一致なら拒否する。"""
    request = _request(tmp_path)
    manifest = DatasetWriter.restore(request.capture_store_root, "final-0")
    frame = manifest.frames[0]
    original_ground, predicted, _latency, evidence = _MockAssembler().assemble_frame(
        frame, manifest.session_id, frame.session_frame_index
    )
    assert original_ground is not None and predicted is not None and evidence is not None
    bound_frame_id = f"{manifest.session_id}:{frame.session_frame_index}"
    original_ground = replace(
        original_ground,
        frame_id=bound_frame_id,
        ui_presentation=replace(
            original_ground.ui_presentation, frame_id=bound_frame_id
        ),
    )
    predicted = replace(
        predicted,
        frame_id=bound_frame_id,
        ui_presentation=replace(predicted.ui_presentation, frame_id=bound_frame_id),
    )
    forged_ui = replace(
        original_ground.ui_presentation, screen_state="forged-screen-state"
    )
    forged_ground = replace(
        original_ground,
        screen_state="forged-screen-state",
        ui_presentation=forged_ui,
    )

    class ForgedLoader:
        def load_frame(self, _frame, _session_id, _frame_index):
            return forged_ground

    class Predictor:
        def predict_frame(self, _frame, _session_id, _frame_index):
            return replace(predicted, detector_artifact_hash="2" * 64), 1.0

    annotation = {
        (manifest.session_id, frame.session_frame_index): (
            evidence,
            _ground_truth_semantic_hash(original_ground),
        )
    }
    with pytest.raises(ValueError, match="immutable annotation"):
        _replay_sessions(
            (replace(manifest, frames=(frame,)),),
            {
                manifest.session_id: SimpleNamespace(
                    kind="final_e2e_test", source_policy="lossless"
                )
            },
            Predictor(),
            ForgedLoader(),
            annotation,
            parser_hash="9" * 64,
            detector_hash="2" * 64,
        )


def test_annotation_payload_must_match_descriptor_and_session_manifest(
    tmp_path: Path,
) -> None:
    capture_root, split = _capture_fixture(tmp_path)
    store = ArtifactStore(tmp_path / "annotation-artifacts")
    split_ref = store.put(
        logical_id="capture/split.json",
        source_path=split.manifest_path,
        media_type="application/json",
    )
    refs = [split_ref]
    bindings = {}
    manifests = {}
    for units in split.splits.values():
        for unit in units:
            session_id = unit.session_id
            session_root = capture_root / "capture_sessions" / session_id
            annotation_path = session_root / "annotations.jsonl"
            annotation_path.write_text(
                json.dumps({
                    "schema_version": "perception_formal_annotation.v1",
                    "session_id": session_id,
                    "frame_index": 0,
                    "raw_card_ids": ["whip"],
                    "raw_inventory": ["whip"],
                    "ground_truth_semantic_hash": "1" * 64,
                }, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            annotation_ref = store.put(
                logical_id=f"capture/{session_id}/annotations.jsonl",
                source_path=annotation_path,
                media_type="application/x-ndjson",
            )
            manifest_ref = store.put(
                logical_id=f"capture/{session_id}/session_manifest.json",
                source_path=session_root / "session_manifest.json",
                media_type="application/json",
            )
            refs.extend((annotation_ref, manifest_ref))
            bindings[session_id] = {
                "annotation_logical_id": annotation_ref.logical_id,
                "annotation_sha256": annotation_ref.sha256,
                "session_manifest_logical_id": manifest_ref.logical_id,
                "session_manifest_sha256": manifest_ref.sha256,
            }
            manifests[session_id] = DatasetWriter.restore(
                capture_root, session_id, metadata_only=True
            )
    descriptor = ArtifactDescriptor(
        logical_id="capture/formal-source",
        node_kind="source_descriptor",
        producer_id="test",
        producer_version="v1",
        identity_metadata={
            "manifest_logical_id": split_ref.logical_id,
            "annotation_bindings": bindings,
        },
        parents=(),
        files=tuple(refs),
    )
    restored = {
        "capture_dataset": RestoredFormalArtifact(
            descriptor,
            descriptor,
            {
                ref.logical_id: store.object_path(ref.store_uri).read_bytes()
                for ref in refs
            },
            None,
        )
    }
    validated = validate_split(split, manifests)
    request = FormalBenchmarkRequest(
        store=store,
        capture_store_root=capture_root,
        dependency_descriptors={"capture_dataset": descriptor},
    )
    selected = {"cal-0"}
    payloads = _verified_annotation_payloads(
        request, restored, validated, selected
    )
    assert set(_annotation_evidence_index(payloads)) == {("cal-0", 0)}

    annotation_path = capture_root / "capture_sessions/cal-0/annotations.jsonl"
    annotation_path.write_bytes(annotation_path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="immutable descriptor"):
        _verified_annotation_payloads(request, restored, validated, selected)


def test_subject_hashes_use_current_fit_file_hash(monkeypatch, tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "subject-artifacts")
    capture_ref = _package_ref(store, "capture.json", {"capture": True})
    target_ref = _package_ref(store, "target.json", {"target": True})

    def descriptor(logical_id, ref):
        return ArtifactDescriptor(
            logical_id=logical_id,
            node_kind="source_descriptor",
            producer_id="test",
            producer_version="v1",
            identity_metadata={"manifest_logical_id": ref.logical_id},
            parents=(),
            files=(ref,),
        )

    request = FormalBenchmarkRequest(
        store=store,
        capture_store_root=tmp_path,
        dependency_descriptors={
            "capture_dataset": descriptor("capture", capture_ref),
            "target_config": descriptor("target", target_ref),
        },
    )
    payloads = {
        "parser_package": {"parser_artifact_hash": "1" * 64},
        "detector_package": {
            "detector_artifact_hash": "2" * 64,
            "model_hash": "3" * 64,
            "build_hash": "4" * 64,
        },
        "assembler_config": {
            "assembler_schema_hash": "5" * 64,
            "ui_presentation_schema_hash": "6" * 64,
            "ui_presentation_golden_fixture_hash": "7" * 64,
            "atlas_vocabulary_hash": "8" * 64,
            "assembler_impl_hash": "9" * 64,
            "roi_resolver_input_hash": "a" * 64,
        },
        "target_config": {"threshold_hash": "b" * 64},
    }
    monkeypatch.setattr(
        "benchmark_survivors_perception._current_fit_code_hash",
        lambda: "f" * 64,
    )
    subjects = _subject_hashes(
        request,
        payloads,
        calibration_profile_hash="c" * 64,
        lineage_seal_hash="d" * 64,
    )
    assert subjects["benchmark_fit_code_hash"] == "f" * 64
    assert set(subjects) == {
        "parser_artifact_hash", "detector_artifact_hash", "model_hash", "build_hash",
        "assembler_schema_hash", "ui_presentation_schema_hash",
        "ui_presentation_golden_fixture_hash", "config_hash", "capture_dataset_hash",
        "calibration_profile_hash", "threshold_hash", "atlas_vocabulary_hash",
        "assembler_impl_hash", "roi_resolver_input_hash", "benchmark_fit_code_hash",
        "lineage_seal_hash",
    }


@pytest.mark.skip(reason="formal release descriptors/sessions are intentionally unavailable")
def test_runner_pins_detector_artifact_hash_from_restored_package(tmp_path: Path, monkeypatch) -> None:
    """runner は assembler output に依らず detector_artifact_hash を restored package から固定する。

    外部 provider（assembler）が任意の detector hash を返しても、runner が
    restored package の content hash で上書きするため、verdict には常に
    ArtifactStore object と同一の detector hash が記録されます。
    """
    class _MaliciousAssembler:
        """悪意ある assembler：detector_artifact_hash を "b" * 64 に偽装しようとする。"""

        def __init__(self, parser_artifact_hash: str) -> None:
            self._inner = _MockAssembler(parser_artifact_hash)

        def assemble_frame(self, frame, session_id: str, frame_index: int):
            ground_truth, predicted, latency, evidence = self._inner.assemble_frame(
                frame, session_id, frame_index
            )
            if ground_truth is not None:
                ground_truth = replace(ground_truth, detector_artifact_hash="b" * 64)
            if predicted is not None:
                predicted = replace(predicted, detector_artifact_hash="b" * 64)
            return ground_truth, predicted, latency, evidence

    def _malicious_factory(restored):
        return _calibration_provider, lambda r: _MaliciousAssembler(
            hashlib.sha256(r["parser_package"]).hexdigest()
        )

    import benchmark_survivors_perception as _bsp
    monkeypatch.setattr(_bsp, "_CLI_PROVIDER_FACTORY", _malicious_factory)
    request = _request(tmp_path)
    result = run_formal_pipeline(request)
    # verdict.detector_artifact_hash は ArtifactStore 内 package の sha256（assembler 出力ではない）
    assert result.verdict.detector_artifact_hash == request.dependency_refs["detector_package"].sha256


@pytest.mark.skip(reason="formal release descriptors/sessions are intentionally unavailable")
def test_canonical_logical_ids_not_set_when_batch_commit_fails(tmp_path: Path, monkeypatch) -> None:
    """batch commit（commit marker）失敗時に canonical logical_id が設定されないことを確認する。

    staging → batch commit → seal → canonical の順序で publish するため、
    batch commit が失敗しても consumer から見える canonical logical_id が残りません。
    batch commit は単一の put_bytes であり、これが唯一の commit 点です。
    """
    from unittest.mock import patch

    from Tools.Artifacts.artifact_store import ArtifactStoreError

    import benchmark_survivors_perception as _bsp
    monkeypatch.setattr(_bsp, "_CLI_PROVIDER_FACTORY", _default_cli_factory)
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


@pytest.mark.skip(reason="formal release descriptors/sessions are intentionally unavailable")
def test_final_sessions_are_reserved_before_provider_and_remain_consumed(tmp_path: Path, monkeypatch) -> None:
    """final 集合を provider/publish 前に予約し、後続失敗でも消費済みにする。"""
    import benchmark_survivors_perception as _bsp

    request = _request(tmp_path)

    def _failing_provider(manifests, restored):
        for index in range(3):
            marker = request.store._logical_index_path(
                f"perception/lineage/opened/final-{index}.json"
            )
            assert marker.exists()
        assert not request.store._logical_index_path(request.verdict_logical_id).exists()
        raise RuntimeError("provider failed after reservation")

    def _failing_factory(restored):
        return _failing_provider, _assembler_factory

    monkeypatch.setattr(_bsp, "_CLI_PROVIDER_FACTORY", _failing_factory)
    with pytest.raises(RuntimeError, match="provider failed after reservation"):
        run_formal_pipeline(request)
    monkeypatch.setattr(_bsp, "_CLI_PROVIDER_FACTORY", _default_cli_factory)
    with pytest.raises(FinalSessionAlreadyOpenedError):
        run_formal_pipeline(request)


def _make_tick(session_id: str, frame_idx: int, screen_state: str) -> SnapshotReplayTick:
    """指定 screen_state の最小 SnapshotReplayTick を生成するヘルパー（回帰テスト用）。"""
    schema = DeployObsSchema.default_v1()
    assembler = RealObsAssembler()
    hud, world = _hud_world(session_id, frame_idx, frame_idx * 1_000_000_000, screen_state)
    snapshot = assembler.assemble(hud, world, schema, (1920, 1080))
    assert snapshot is not None
    return SnapshotReplayTick(
        session_id, "final_e2e_test", "lossless",
        snapshot.frame_id, snapshot, snapshot, 0.0,
    )


def test_event_rising_edge_counts_two_occurrences_not_four() -> None:
    """event:level_up は rising-edge (False→True) のみ計上し、継続 True は重複しない。

    gameplay→level_up→gameplay→level_up→gameplay の 2 occurrence 入力に対して
    count=4 でなく count=2 になることを確認する（P1-1 regression）。
    """
    states = ["gameplay", "level_up_items", "gameplay", "level_up_items", "gameplay"]
    ticks = [_make_tick("sess", i, s) for i, s in enumerate(states)]
    report = run_benchmark(ticks)
    assert report.slice_counts.get("event:level_up", 0) == 2


def test_coord_quantization_uses_half_dimension_for_normalized_domain() -> None:
    """coord_quantization_px は [-1,1] domain なので 1 unit = dimension/2 px で変換する。

    正規化差 0.02、幅 1920 の場合、19.2px（38.4px ではない）になることを確認する（P1-4 regression）。
    """
    schema = DeployObsSchema.default_v1()
    assembler = RealObsAssembler()
    width, height = 1920, 1080
    session_id = "coord-test"
    hud_g, world = _hud_world(session_id, 0, 1_000_000_000, "gameplay")
    ground = assembler.assemble(hud_g, world, schema, (width, height))
    assert ground is not None
    ps_off, ps_sz = schema.layout["player_screen_pos"]
    # x 方向に 0.02 の正規化差を注入する（[−1,1] domain → 1920/2 * 0.02 = 19.2 px 期待値）。
    pred_values = np.array(ground.deploy_obs.values, dtype=float)
    pred_values[ps_off] += 0.02
    pred_deploy = replace(ground.deploy_obs, values=pred_values)
    predicted = replace(ground, deploy_obs=pred_deploy)
    tick = SnapshotReplayTick(session_id, "error_calibration", "lossless", ground.frame_id,
                              ground, predicted, 0.0)
    residuals = _derive_calibration_residuals([tick], resolution_wh=(width, height))
    quant_residuals = [r for r in residuals if r.field == "coord_quantization_px"]
    assert quant_residuals, "coord_quantization_px residual が生成されていない"
    assert any(abs(r.residual - 19.2) < 0.5 for r in quant_residuals), (
        f"expected ~19.2 px but got {[r.residual for r in quant_residuals]}"
    )


def test_invalid_predicted_validity_excludes_coord_residual() -> None:
    """predicted validity=0 の座標は coord residual を生成しない（P1-5 regression）。

    invalid な predicted が noise sample として混入しないことを確認する。
    """
    schema = DeployObsSchema.default_v1()
    assembler = RealObsAssembler()
    width, height = 1920, 1080
    session_id = "validity-test"
    hud_g, world = _hud_world(session_id, 0, 1_000_000_000, "gameplay")
    ground = assembler.assemble(hud_g, world, schema, (width, height))
    assert ground is not None
    # predicted validity を全 coord segment で 0 にして invalid にする。
    pred_validity = np.zeros_like(ground.deploy_obs.validity)
    pred_deploy = replace(ground.deploy_obs, validity=pred_validity)
    predicted = replace(ground, deploy_obs=pred_deploy)
    tick = SnapshotReplayTick(session_id, "error_calibration", "lossless", ground.frame_id,
                              ground, predicted, 0.0)
    residuals = _derive_calibration_residuals([tick], resolution_wh=(width, height))
    coord_residuals = [
        r for r in residuals
        if r.field in {"coord_noise", "coord_quantization_px"}
        and r.session_id == session_id and r.frame_id == ground.frame_id
    ]
    assert coord_residuals == [], (
        f"invalid predicted validity でも {len(coord_residuals)} 件の coord residual が生成された"
    )


def test_predicted_none_skips_all_continuous_residuals() -> None:
    """predicted=None のフレームは coord/item_category/enemy_category/hp/xp/timer/inventory を生成しない。

    欠測フレームが ground truth を prediction として代入したゼロ残差で calibration を
    汚染しないことを確認する（P1 #2 regression）。
    """
    schema = DeployObsSchema.default_v1()
    assembler = RealObsAssembler()
    width, height = 1920, 1080
    session_id = "none-predicted-test"
    hud_g, world = _hud_world(session_id, 0, 1_000_000_000, "gameplay")
    ground = assembler.assemble(hud_g, world, schema, (width, height))
    assert ground is not None
    # predicted=None: detector/parser が出力できなかったフレームを模擬する。
    tick = SnapshotReplayTick(
        session_id, "error_calibration", "lossless",
        ground.frame_id, ground, None, 0.0,
    )
    residuals = _derive_calibration_residuals([tick], resolution_wh=(width, height))
    continuous_fields = {
        "coord_noise", "coord_quantization_px",
        "hp_ratio", "xp_ratio", "timer_seconds", "inventory_hash",
        "item_category", "enemy_category",
    }
    bad = [r for r in residuals if r.field in continuous_fields]
    assert bad == [], (
        f"predicted=None なのに {len(bad)} 件の連続 residual が生成された: "
        f"{[r.field for r in bad]}"
    )


def test_age_frames_propagated_from_ui_state_age() -> None:
    """hp_ratio/xp_ratio/timer の age_frames は predicted の ui_state_age から変換する。

    age_frames が常に 0 に固定されると fit_error_profile の confidence weighting が
    完全に無効になる（P1 #3 regression）。item_context を合成して直接注入する。
    """
    from reinbalance_survivors_contracts.item_decision import CandidateFeatures, ItemDecisionFeatures
    schema = DeployObsSchema.default_v1()
    assembler = RealObsAssembler()
    width, height = 1920, 1080
    session_id = "age-test"
    from benchmark_survivors_perception import _CONSUMER_FRAME_PERIOD_MS
    frame_period_ms = _CONSUMER_FRAME_PERIOD_MS
    age_ms = 4.0 * frame_period_ms  # 4 フレーム分の遅延
    hud_g, world = _hud_world(session_id, 0, 1_000_000_000, "gameplay")
    ground = assembler.assemble(hud_g, world, schema, (width, height))
    assert ground is not None
    # ItemDecisionFeatures を直接構築して ui_state_age を制御する。
    # CandidateFeatures は padding 1 枚で最小有効な choice_count=0 にする。
    real_cand = CandidateFeatures(
        kind="weapon", item_id="whip", new_level=1, owned=False,
        is_new=True, is_evolve=False, is_union=False, has_prerequisite=False, slot_capacity=0,
    )
    padding_cand = CandidateFeatures(
        kind="padding", item_id="__padding__", new_level=0, owned=False,
        is_new=False, is_evolve=False, is_union=False, has_prerequisite=False, slot_capacity=0,
    )
    synthetic_context = ItemDecisionFeatures(
        decision_id="age-test-decision",
        feature_schema="context_only_v1",
        elapsed_time=60.0, level=2, hp_ratio=1.0, xp_ratio=0.5,
        weapon_slots=(1, 0, 0, 0, 0, 0), passive_slots=(0, 0, 0, 0, 0, 0),
        empty_slot_count=11, evolution_readiness=0.0, choice_count=1, card_mask=(True, False),
        fallback_kind="none",
        ui_state_validity=0.7,
        ui_state_age=age_ms / 1000.0,
        candidates=(real_cand,), max_item_cards=2,
    )
    # ground も item_context を持つ必要がある（guard: predicted_context is not None and ground_context is not None）。
    # age は predicted の ui_state_age から取るので ground の値は 0 にする。
    ground_context = replace(synthetic_context, ui_state_age=0.0, ui_state_validity=1.0, decision_id="age-test-ground")
    ground_with_ctx = replace(ground, item_context=ground_context)
    predicted = replace(ground_with_ctx, item_context=synthetic_context)
    tick = SnapshotReplayTick(
        session_id, "error_calibration", "lossless",
        ground.frame_id, ground_with_ctx, predicted, 0.0,
    )
    residuals = _derive_calibration_residuals([tick], resolution_wh=(width, height))
    hp_residuals = [r for r in residuals if r.field == "hp_ratio"]
    expected_age = round(age_ms / frame_period_ms)
    assert hp_residuals, "hp_ratio residual が生成されていない"
    assert all(r.age_frames == expected_age for r in hp_residuals), (
        f"expected age_frames={expected_age}, got {[r.age_frames for r in hp_residuals]}"
    )
    assert all(r.confidence == pytest.approx(0.7) for r in hp_residuals), (
        f"expected confidence=0.7, got {[r.confidence for r in hp_residuals]}"
    )


def test_coord_age_comes_from_deploy_obs_age_not_item_context() -> None:
    """coord residual の age_frames は deploy_obs.age から取得する（P1-4 regression）。

    item_context が None（ゲームプレイ中の標準状態）でも age_frames は 0 にならず、
    deploy_obs.age 平面から正しく変換されることを確認する。
    """
    schema = DeployObsSchema.default_v1()
    assembler = RealObsAssembler()
    width, height = 1920, 1080
    session_id = "coord-age-test"
    from benchmark_survivors_perception import _CONSUMER_FRAME_PERIOD_MS
    frame_period_ms = _CONSUMER_FRAME_PERIOD_MS
    # player_screen_pos の max_age_ms = 500.0（schema.default_v1 による）
    ps_max_age_ms = 500.0
    # age = 0.5 → 0.5 * 500 / frame_period_ms フレーム分の遅延
    age_normalized = 0.5
    expected_age = round(age_normalized * ps_max_age_ms / frame_period_ms)
    hud_g, world = _hud_world(session_id, 0, 1_000_000_000, "gameplay")
    ground = assembler.assemble(hud_g, world, schema, (width, height))
    assert ground is not None
    ps_off, ps_sz = schema.layout["player_screen_pos"]
    ne_off, ne_sz = schema.layout["nearest_enemy_offset"]
    # predicted の coord segment age 平面を両方とも age_normalized に設定する。
    # 値は ground と同一（residual=0）にして coord_noise residual のみを取得する。
    pred_age = np.array(ground.deploy_obs.age, dtype=float)
    pred_age[ps_off:ps_off + ps_sz] = age_normalized
    pred_age[ne_off:ne_off + ne_sz] = age_normalized
    # validity は有効値（>0）にする。
    pred_validity = np.array(ground.deploy_obs.validity, dtype=float)
    pred_validity[ps_off:ps_off + ps_sz] = 1.0
    pred_validity[ne_off:ne_off + ne_sz] = 1.0
    pred_deploy = replace(ground.deploy_obs, age=pred_age, validity=pred_validity)
    # item_context は None のまま（ゲームプレイ中の標準状態）。
    predicted = replace(ground, deploy_obs=pred_deploy, item_context=None)
    # ground の validity も有効にする。
    gnd_validity = np.array(ground.deploy_obs.validity, dtype=float)
    gnd_validity[ps_off:ps_off + ps_sz] = 1.0
    gnd_validity[ne_off:ne_off + ne_sz] = 1.0
    gnd_deploy = replace(ground.deploy_obs, validity=gnd_validity)
    ground_with_valid = replace(ground, deploy_obs=gnd_deploy, item_context=None)
    tick = SnapshotReplayTick(
        session_id, "error_calibration", "lossless",
        ground.frame_id, ground_with_valid, predicted, 0.0,
    )
    residuals = _derive_calibration_residuals([tick], resolution_wh=(width, height))
    coord_residuals = [
        r for r in residuals
        if r.field in {"coord_noise", "coord_quantization_px"}
    ]
    assert coord_residuals, "coord residual が生成されていない"
    assert all(r.age_frames == expected_age for r in coord_residuals), (
        f"expected age_frames={expected_age} (from deploy_obs.age), "
        f"got {[r.age_frames for r in coord_residuals]}. "
        "item_context=None でも age_frames は 0 であってはならない。"
    )
