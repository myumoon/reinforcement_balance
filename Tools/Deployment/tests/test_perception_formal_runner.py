"""formal runner と typed UiPresentation replay の synthetic store fixture。"""

from __future__ import annotations

import hashlib
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
from survivors.perception_benchmark import (
    SnapshotReplayTick,
    recompute_gate_from_metrics,
    replay_snapshots,
)
from survivors.perception_error_fit import CalibrationResidual, FinalSessionAlreadyOpenedError
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
        dependency_refs=_dependency_refs(store, split),
        capture_store_root=capture_root,
    )


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
