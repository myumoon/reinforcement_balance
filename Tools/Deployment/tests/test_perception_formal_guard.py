"""formal promotion guard の回帰テスト。

synthetic/development 由来の成果物が formal writer・formal verdict factory・restore
verdict gate へ到達できないことを、実 formal session/依存なしで検証する。実 formal
成功経路テスト（test_perception_formal_runner.py）は正式依存が揃うまで skip のまま残す。
"""

from __future__ import annotations

from typing import Any

import pytest

from benchmark_survivors_perception import _validate_restore_verdict, verify_formal_runtime_release
from reinbalance_survivors_contracts.artifact_identity import (
    RESTORE_TEST_VERDICT_SCHEMA_VERSION,
    ArtifactDescriptor,
    ArtifactRef,
    RestoreTestVerdict,
    artifact_uri,
)
from reinbalance_survivors_contracts.canonical_json import canonical_hash, canonical_json_bytes
from survivors.perception_benchmark import BenchmarkRecord, run_benchmark
from survivors.perception_error_fit import (
    _FORMAL_FACTORY_TOKEN,
    _HASH_FIELDS,
    CalibrationResidual,
    FINAL_VERDICT_SCHEMA_VERSION,
    FittedPerceptionErrorProfile,
    FormalVerdictPromotionError,
    HashMismatchError,
    _create_formal_final_verdict,
    _load_formal_verdict_from_verified_store,
    _write_formal_final_verdict,
    create_synthetic_final_verdict,
    fit_error_profile,
)


def _all_subjects() -> dict[str, str]:
    """_HASH_FIELDS 全 16 フィールドの妥当な SHA-256 subject 集合。"""
    return {name: format(index % 16, "x") * 64 for index, name in enumerate(_HASH_FIELDS)}


def _rec(field: str, gt: object, pred: object, index: int) -> BenchmarkRecord:
    return BenchmarkRecord(
        frame_id=f"{field}-{index}", session_id=f"s{index % 3}",
        session_kind="error_calibration", source_policy="raw",
        field=field, ground_truth=gt, predicted=pred, confidence=0.9, latency_ms=0.0,
    )


def _screen_records() -> list[BenchmarkRecord]:
    return [_rec("screen_state", "gameplay", "gameplay", index) for index in range(30)]


def _finite_records() -> list[BenchmarkRecord]:
    """全 metric が有限値になる最小 record 集合（inf 既定の field を埋める）。"""
    records = _screen_records()
    for index in range(6):
        records.append(_rec("hp_ratio", 0.5, 0.5, index))
        records.append(_rec("xp_ratio", 0.5, 0.5, index))
        records.append(_rec("nearest_distance", 0.5, 0.5, index))
        records.append(_rec("ui_roi_center_error", 0.0, 0.0, index))
    return records


def test_synthetic_fit_is_development_only_and_roundtrips() -> None:
    """formal=False の fit は development_only=True を保持し artifact wire で往復する。"""
    residuals = [
        CalibrationResidual(f"cal-{index % 3}", f"frame-{index}", "hp_ratio", 0.01, 1.0, 0)
        for index in range(6)
    ]
    profile = fit_error_profile(
        residuals, ["cal-0", "cal-1", "cal-2"], ["final-0"]
    )
    assert profile.development_only is True
    wire = profile.to_artifact_wire()
    assert wire["development_only"] is True
    restored = FittedPerceptionErrorProfile.from_artifact_wire(wire)
    assert restored.development_only is True


def test_synthetic_report_cannot_create_formal_verdict() -> None:
    """development-only benchmark report から formal verdict factory を呼べない。"""
    report = run_benchmark(_screen_records())
    assert report.development_only is True
    with pytest.raises(FormalVerdictPromotionError):
        _create_formal_final_verdict(
            report, seal_id="0" * 64, final_session_ids=["final-0"], **_all_subjects()
        )


def test_synthetic_verdict_rejected_by_formal_writer(tmp_path) -> None:
    """synthetic verdict は formal writer に渡しても development_only で拒否される。"""
    from Tools.Artifacts.artifact_store import ArtifactStore

    report = run_benchmark(_finite_records())
    verdict = create_synthetic_final_verdict(
        report, seal_id="0" * 64, final_session_ids=["final-0"], **_all_subjects()
    )
    assert verdict.development_only is True
    store = ArtifactStore(str(tmp_path / "store"))
    with pytest.raises(FormalVerdictPromotionError):
        _write_formal_final_verdict(
            verdict, store=store, logical_id="perception/guard/verdict.json"
        )


def test_formal_writer_rejects_failed_verdict(tmp_path) -> None:
    """passed=False の formal verdict は _write_formal_final_verdict が拒否する（P1-2a regression）。

    passed=False のまま perception/final/... へ発行できないことを確認する。
    """
    from Tools.Artifacts.artifact_store import ArtifactStore
    from benchmark_survivors_perception import _formalize_benchmark_report
    # _finite_records は formal 閾値を満たさないため formal 化すると passed=False になる。
    synthetic_report = run_benchmark(_finite_records())
    formal_report = _formalize_benchmark_report(synthetic_report)
    assert formal_report.passed is False, "このテストには passed=False の formal レポートが必要"
    verdict = _create_formal_final_verdict(
        formal_report, seal_id="0" * 64, final_session_ids=["final-0"], **_all_subjects()
    )
    assert verdict.passed is False
    store = ArtifactStore(str(tmp_path / "store"))
    with pytest.raises(FormalVerdictPromotionError, match="passed"):
        _write_formal_final_verdict(
            verdict, store=store, logical_id="perception/final/failed.json"
        )


def _dependency() -> ArtifactDescriptor:
    ref = ArtifactRef(
        logical_id="parser/manifest.json", sha256="a" * 64, size_bytes=10,
        media_type="application/json", store_uri=artifact_uri("a" * 64),
    )
    return ArtifactDescriptor(
        logical_id="parser-package", node_kind="source_descriptor",
        producer_id="fixture", producer_version="v1",
        identity_metadata={"stable_hash": "b" * 64}, files=(ref,),
    )


def _restore_verdict(dependency: ArtifactDescriptor, **overrides) -> RestoreTestVerdict:
    defaults = dict(
        logical_id="parser-package.restore",
        subject=dependency.node_ref(),
        manifest_hash=dependency.files[0].sha256,
        primary_root="primary", backup_root="backup",
        verify_mode="full", checked_object_count=1,
        passed=True, blocking_reasons=(),
    )
    defaults.update(overrides)
    return RestoreTestVerdict(**defaults)


def test_genuine_restore_verdict_is_accepted() -> None:
    dependency = _dependency()
    verdict = _restore_verdict(dependency).to_descriptor()
    _validate_restore_verdict(dependency, verdict)  # must not raise


def test_sample_mode_restore_verdict_is_rejected() -> None:
    dependency = _dependency()
    verdict = _restore_verdict(dependency, verify_mode="sample").to_descriptor()
    with pytest.raises(ValueError, match="full restore"):
        _validate_restore_verdict(dependency, verdict)


def test_restore_verdict_with_forged_producer_is_rejected() -> None:
    """node_kind だけ restore_test_verdict に偽装した任意 descriptor を拒否する。"""
    dependency = _dependency()
    genuine = _restore_verdict(dependency).to_descriptor()
    forged = ArtifactDescriptor(
        logical_id=genuine.logical_id, node_kind="restore_test_verdict",
        producer_id="evil-producer", producer_version=RESTORE_TEST_VERDICT_SCHEMA_VERSION,
        identity_metadata=dict(genuine.identity_metadata),
        parents=(dependency.node_ref(),),
    )
    with pytest.raises(ValueError, match="fixed restore-test producer"):
        _validate_restore_verdict(dependency, forged)


def test_restore_verdict_manifest_hash_must_match_dependency() -> None:
    dependency = _dependency()
    verdict = _restore_verdict(dependency, manifest_hash="c" * 64).to_descriptor()
    with pytest.raises(ValueError, match="manifest_hash"):
        _validate_restore_verdict(dependency, verdict)


# --- P1-1: fit_error_profile 常時 development_only=True ガード ---

def test_fit_error_profile_public_api_always_development_only() -> None:
    """公開 fit_error_profile() は _factory_token なしで常に development_only=True を返す。"""
    residuals = [
        CalibrationResidual(f"s{i % 3}", f"f{i}", "hp_ratio", 0.01, 1.0, 0)
        for i in range(6)
    ]
    profile = fit_error_profile(residuals, ["s0", "s1", "s2"], [])
    assert profile.development_only is True


def test_fitted_profile_development_only_false_without_token_raises() -> None:
    """_factory_token なしで development_only=False を直接構築すると拒否される。"""
    residuals = [
        CalibrationResidual(f"s{i % 3}", f"f{i}", "hp_ratio", 0.01, 1.0, 0)
        for i in range(6)
    ]
    profile = fit_error_profile(residuals, ["s0", "s1", "s2"], [])
    with pytest.raises(FormalVerdictPromotionError):
        FittedPerceptionErrorProfile(
            calibration_session_ids=list(profile.calibration_session_ids),
            final_e2e_session_ids=[],
            calibration_session_hashes=dict(profile.calibration_session_hashes),
            field_sample_counts=dict(profile.field_sample_counts),
            fit_code_hash=profile.fit_code_hash,
            development_only=False,  # factory token なし → 拒否される
            _factory_token=None,
        )


# --- P2: verify_formal_runtime_release focused tests ---

def _h(ch: str) -> str:
    """64文字の固定 hex SHA-256 代替値を返す。"""
    return ch * 64


def _formal_subjects() -> dict[str, str]:
    return {name: _h(format(i % 16, "x")) for i, name in enumerate(_HASH_FIELDS)}


def _passing_formal_verdict_wire(subjects: dict[str, str]) -> dict[str, Any]:
    """formal gate を通過する PerceptionFinalVerdict wire を構築する。

    実 ArtifactStore / formal runner を使わずに verify_formal_runtime_release の
    store 復元・load_final_verdict 経路を通すための最小フィクスチャ。
    """
    from survivors.perception_benchmark import (
        _FORMAL_SLICE_COUNT_FLOORS,
        _FORMAL_SLICE_SESSION_FLOORS,
        _FORMAL_SLICE_THRESHOLDS,
        _FORMAL_REQUIRED_SLICES,
        THRESHOLD_SCREEN_F1,
        THRESHOLD_TIMER_EXACT,
        THRESHOLD_LEVEL_EXACT,
        THRESHOLD_INVENTORY_TOP1,
        THRESHOLD_CHOICE_TOP1,
        THRESHOLD_DENSITY_CORR,
        THRESHOLD_CONFIDENCE,
        _empty_report,
        recompute_gate_from_metrics,
    )
    metrics: dict[str, Any] = dict(
        _empty_report(development_only=True, formal_eligible=False).metrics_wire()
    )
    metrics.update({
        "total_records": 1000,
        "screen_state_f1": THRESHOLD_SCREEN_F1,
        "timer_exact_rate": THRESHOLD_TIMER_EXACT,
        "level_exact_rate": THRESHOLD_LEVEL_EXACT,
        "inventory_top1_rate": THRESHOLD_INVENTORY_TOP1,
        "choice_top1_rate": THRESHOLD_CHOICE_TOP1,
        "hp_mae": 0.0,
        "xp_mae": 0.0,
        "density_correlation": THRESHOLD_DENSITY_CORR,
        "nearest_normalized_median_error": 0.0,
        "latency_p95_ms": 0.0,
        "latency_p99_ms": 0.0,
        "invalid_tick_rate": 0.0,
        "levelup_invalid_choice_rate": 0.0,
        "roi_center_p99": 0.0,
        "roi_inside_region_rate": 1.0,
        "roi_false_positive_count": 0,
        "confidence_mean": THRESHOLD_CONFIDENCE,
        "ui_cross_frame_equivalence_rate": 0.0,
        "expected_tick_count": 10,
        "observed_tick_count": 10,
        "latency_tick_count": 10,
    })
    required_base = {
        "screen_state", "timer_seconds", "level", "hp_ratio", "xp_ratio",
        "inventory_top1", "choice_top1", "entity_density", "nearest_distance",
        "ui_roi_center_error", "ui_inside_region", "ui_false_positive", "confidence",
    }
    sc: dict[str, int] = {name: 1 for name in required_base}
    for name, floor in _FORMAL_SLICE_COUNT_FLOORS.items():
        sc[name] = floor
    metrics["slice_counts"] = sc
    ssc: dict[str, int] = {}
    for name, floor in _FORMAL_SLICE_SESSION_FLOORS.items():
        ssc[name] = floor
    metrics["slice_session_counts"] = ssc
    slices = []
    for name in sorted(_FORMAL_REQUIRED_SLICES):
        thr = _FORMAL_SLICE_THRESHOLDS[name]
        slices.append({
            "name": name, "count": sc.get(name, 1), "session_count": 2,
            "metric_value": thr, "threshold": thr, "ci_lower": thr,
        })
    metrics["slices"] = slices
    passed, blocking = recompute_gate_from_metrics(metrics, formal=True)
    assert passed and not blocking, f"fixture fails formal gate: {blocking}"
    seal_id = _h("a")
    identity = {
        "seal_id": seal_id,
        "final_session_ids": ["final-0"],
        "subject_hashes": {name: subjects[name] for name in _HASH_FIELDS},
        "metrics": metrics,
    }
    verdict_id = canonical_hash(identity)
    return {
        "schema_version": FINAL_VERDICT_SCHEMA_VERSION,
        "verdict_id": verdict_id,
        "seal_id": seal_id,
        "final_session_ids": ["final-0"],
        **subjects,
        "metrics": metrics,
        "passed": True,
        "blocking_reasons": [],
        "development_only": False,
        "formal_perception_verdict_eligible": True,
    }


def _file(lid: str, ch: str) -> ArtifactRef:
    h = _h(ch)
    return ArtifactRef(
        logical_id=lid, sha256=h, size_bytes=16,
        media_type="application/octet-stream", store_uri=artifact_uri(h),
    )


def _node(lid: str, kind: str, parents: tuple = (), ch: str = "0") -> ArtifactDescriptor:
    return ArtifactDescriptor(
        logical_id=lid, node_kind=kind,
        producer_id="test-producer", producer_version="v1",
        identity_metadata={"stable_config_hash": _h("f")},
        parents=parents,
        files=(_file(f"{lid}.bin", ch),),
    )


def _build_runtime_dag(
    verdict_desc: ArtifactDescriptor,
    subjects: dict[str, str],
    profile: ArtifactDescriptor,
) -> list[ArtifactDescriptor]:
    """validate_formal_runtime_dag を通過する最小 descriptor 列を構築する。"""
    src = _node("source", "source_descriptor", ch="1")
    teacher = _node("teacher", "teacher_validation_verdict", (src.node_ref(),), "2")
    dataset = _node("dataset", "choice_dataset_release", (teacher.node_ref(),), "3")
    item = _node("item", "item_selector_release", (dataset.node_ref(),), "4")
    combat = _node("combat", "combat_student_release", (dataset.node_ref(),), "5")
    runtime = ArtifactDescriptor(
        logical_id="runtime", node_kind="runtime_bundle",
        producer_id="test-producer", producer_version="v1",
        identity_metadata={"perception_subject_hashes": subjects},
        parents=(item.node_ref(), combat.node_ref(), verdict_desc.node_ref()),
        files=(_file("runtime.bin", "7"),),
    )
    return [src, teacher, dataset, item, combat, profile, verdict_desc, runtime]


def _store_verdict_and_build_descriptor(
    store: Any, subjects: dict[str, str], wire: dict[str, Any]
) -> tuple[ArtifactDescriptor, ArtifactDescriptor]:
    """verdict wire を ArtifactStore に保存し、(profile, verdict) descriptor を返す。"""
    ref = store.put_bytes(
        logical_id="perception/verdict/v.json",
        data=canonical_json_bytes(wire),
        media_type="application/json",
    )
    src_ref = _node("source", "source_descriptor", ch="1")
    profile = _node("profile", "perception_calibration_profile", (src_ref.node_ref(),), "6")
    verdict = ArtifactDescriptor(
        logical_id="perception-verdict",
        node_kind="perception_final_verdict",
        producer_id="test-producer",
        producer_version="v1",
        identity_metadata={
            "verdict_id": wire["verdict_id"],
            "seal_id": wire["seal_id"],
            "passed": True,
            "development_only": False,
            "subject_hashes": {name: subjects[name] for name in _HASH_FIELDS},
        },
        parents=(profile.node_ref(),),
        files=(ref,),
    )
    return profile, verdict


def test_verify_formal_runtime_release_happy_path(tmp_path: Any) -> None:
    """passed=True production verdict は verify_formal_runtime_release を通過する。"""
    from Tools.Artifacts.artifact_store import ArtifactStore
    store = ArtifactStore(str(tmp_path / "store"))
    subjects = _formal_subjects()
    wire = _passing_formal_verdict_wire(subjects)
    profile, verdict_desc = _store_verdict_and_build_descriptor(store, subjects, wire)
    dag = _build_runtime_dag(verdict_desc, subjects, profile)
    verify_formal_runtime_release(dag, store)  # 例外なし


def test_verify_formal_runtime_release_rejects_development_only(tmp_path: Any) -> None:
    """development_only=True の verdict は拒否される。"""
    from Tools.Artifacts.artifact_store import ArtifactStore
    store = ArtifactStore(str(tmp_path / "store"))
    subjects = _formal_subjects()
    wire = _passing_formal_verdict_wire(subjects)
    # wire の development_only を True にして store に保存
    wire["development_only"] = True
    wire["formal_perception_verdict_eligible"] = False
    ref = store.put_bytes(
        logical_id="perception/verdict/dev.json",
        data=canonical_json_bytes(wire),
        media_type="application/json",
    )
    src_ref = _node("source", "source_descriptor", ch="1")
    profile = _node("profile", "perception_calibration_profile", (src_ref.node_ref(),), "6")
    verdict_desc = ArtifactDescriptor(
        logical_id="perception-verdict",
        node_kind="perception_final_verdict",
        producer_id="test-producer", producer_version="v1",
        identity_metadata={
            "verdict_id": wire["verdict_id"], "seal_id": wire["seal_id"],
            "passed": True, "development_only": False,
            "subject_hashes": {name: subjects[name] for name in _HASH_FIELDS},
        },
        parents=(profile.node_ref(),), files=(ref,),
    )
    dag = _build_runtime_dag(verdict_desc, subjects, profile)
    with pytest.raises(ValueError, match="production verdict"):
        verify_formal_runtime_release(dag, store)


def test_verify_formal_runtime_release_rejects_subject_mismatch(tmp_path: Any) -> None:
    """runtime の perception_subject_hashes が verdict と一致しない場合は拒否される。"""
    from Tools.Artifacts.artifact_store import ArtifactStore
    store = ArtifactStore(str(tmp_path / "store"))
    subjects = _formal_subjects()
    wire = _passing_formal_verdict_wire(subjects)
    profile, verdict_desc = _store_verdict_and_build_descriptor(store, subjects, wire)
    wrong_subjects = {name: _h("e") for name in _HASH_FIELDS}
    # runtime_bundle の perception_subject_hashes を verdict と異なる値にする
    src = _node("source", "source_descriptor", ch="1")
    teacher = _node("teacher", "teacher_validation_verdict", (src.node_ref(),), "2")
    dataset = _node("dataset", "choice_dataset_release", (teacher.node_ref(),), "3")
    item = _node("item", "item_selector_release", (dataset.node_ref(),), "4")
    combat = _node("combat", "combat_student_release", (dataset.node_ref(),), "5")
    runtime = ArtifactDescriptor(
        logical_id="runtime", node_kind="runtime_bundle",
        producer_id="test-producer", producer_version="v1",
        identity_metadata={"perception_subject_hashes": wrong_subjects},
        parents=(item.node_ref(), combat.node_ref(), verdict_desc.node_ref()),
        files=(_file("runtime.bin", "7"),),
    )
    dag = [src, teacher, dataset, item, combat, profile, verdict_desc, runtime]
    with pytest.raises(Exception):  # StaleVerdictError or ArtifactDagValidationError
        verify_formal_runtime_release(dag, store)


# --- M8/M9: from_store_artifact の formal token 発行境界 ---

# formal fit が必須とする residual field 一式（各 field 3 session 以上・3 標本以上）。
_FORMAL_FIT_FIELDS = (
    "coord_noise", "hp_ratio", "xp_ratio", "timer_seconds",
    "inventory_hash", "coord_quantization_px",
    "burst_enter", "burst_exit", "burst_dropout",
    "unknown_screen_collapse", "unknown_screen_collapse_duration",
    "item_category", "enemy_category",
)

_BOUNDED_FIT_FIELDS = frozenset({
    "burst_enter", "burst_exit", "burst_dropout", "unknown_screen_collapse",
})


def _formal_residuals(session_ids: tuple[str, ...]) -> list[CalibrationResidual]:
    """formal fit の power 条件を満たす最小 residual 集合を作る。

    13 の必須 field それぞれについて、3 つの異なる session から 1 標本ずつ作ります。
    値は profile の値域制約（確率 field は [0,1]、非負 field は >=0）に収めます。
    """
    rows: list[CalibrationResidual] = []
    for name in _FORMAL_FIT_FIELDS:
        for index, session_id in enumerate(session_ids):
            categories: dict[str, int] = {}
            if name in {"item_category", "enemy_category"}:
                categories = {"ground_truth_category": 0, "predicted_category": 0}
            rows.append(
                CalibrationResidual(
                    session_id=session_id,
                    frame_id=f"{name}-{index}",
                    field=name,
                    residual=0.25 if name in _BOUNDED_FIT_FIELDS else 0.01,
                    confidence=0.9,
                    age_frames=0,
                    latency_frames=0.0,
                    **categories,
                )
            )
    return rows


def _producer_calibration_commit(tmp_path: Any) -> tuple[Any, tuple, str, Any]:
    """producerの実関数だけを使って formal calibration commit を store へ作る。

    `_FORMAL_FACTORY_TOKEN` をテスト側から渡さず、formal fit runner と
    `_commit_calibration_package()` という producer の本番経路だけを通します。
    戻り値は (store, calibration descriptors, commit logical id, profile)。
    """
    from benchmark_survivors_perception import (
        FormalBenchmarkRequest,
        _commit_calibration_package,
        calibration_commit_logical_id,
    )
    from reinbalance_survivors_contracts.artifact_store import ArtifactStore
    from survivors.perception_error_fit import _fit_formal_error_profile

    store = ArtifactStore(tmp_path / "producer-store")
    capture_ref = store.put_bytes(
        logical_id="perception/capture/manifest.json",
        data=canonical_json_bytes({"sessions": ["cal-0", "cal-1", "cal-2"]}),
        media_type="application/json",
    )
    capture_descriptor = ArtifactDescriptor(
        logical_id="perception/capture/dataset",
        node_kind="source_descriptor",
        producer_id="capture-fixture",
        producer_version="v1",
        identity_metadata={"manifest_logical_id": capture_ref.logical_id},
        files=(capture_ref,),
    )
    request = FormalBenchmarkRequest(
        store=store,
        capture_store_root=tmp_path / "captures",
        dependency_descriptors={"capture_dataset": capture_descriptor},
    )
    session_ids = ("cal-0", "cal-1", "cal-2")
    profile = _fit_formal_error_profile(
        _formal_residuals(session_ids),
        list(session_ids),
        ["final-0"],
        calibration_session_hashes={
            session_id: canonical_hash({"session": session_id})
            for session_id in session_ids
        },
    )
    run_key = "producer-guard"
    descriptors, _staged, _raw_ref, _artifact_ref = _commit_calibration_package(
        request,
        run_key,
        profile,
        {"capture_dataset_hash": capture_ref.sha256},
        request.calibration_logical_id(run_key),
        request.calibration_provenance_logical_id(run_key),
    )
    return store, descriptors, calibration_commit_logical_id(run_key), profile


def test_from_store_artifact_loads_formal_profile(tmp_path: Any) -> None:
    """producerのdescriptor chainとcalibration commitからだけformal profileを復元できる。

    テストコードは `_FORMAL_FACTORY_TOKEN` を一切渡さず、producer の実出力のみを
    入力にして formal（development_only=False）profile を取得します（M9(b)）。
    """
    store, descriptors, commit_logical_id, profile = _producer_calibration_commit(tmp_path)

    restored = FittedPerceptionErrorProfile.from_store_artifact(
        store,
        descriptors=descriptors,
        expected_calibration_identity_hash=descriptors[1].identity_hash,
    )
    assert restored.development_only is False
    assert restored.to_artifact_wire() == profile.to_artifact_wire()
    assert restored.calibration_descriptor_hash == descriptors[1].identity_hash

    from_commit = FittedPerceptionErrorProfile.from_calibration_commit(
        store,
        commit_logical_id=commit_logical_id,
        expected_calibration_identity_hash=descriptors[1].identity_hash,
    )
    assert from_commit.to_artifact_wire() == profile.to_artifact_wire()
    assert from_commit.calibration_descriptor_hash == descriptors[1].identity_hash


def test_from_store_artifact_rejects_wrong_expected_identity(tmp_path: Any) -> None:
    """期待 calibration identity が違えば producer 出力でも formal 化しない。"""
    store, descriptors, commit_logical_id, _profile = _producer_calibration_commit(tmp_path)

    with pytest.raises(FormalVerdictPromotionError, match="does not match the expected"):
        FittedPerceptionErrorProfile.from_store_artifact(
            store,
            descriptors=descriptors,
            expected_calibration_identity_hash="d" * 64,
        )
    with pytest.raises(FormalVerdictPromotionError, match="does not match the expected"):
        FittedPerceptionErrorProfile.from_calibration_commit(
            store,
            commit_logical_id=commit_logical_id,
            expected_calibration_identity_hash="d" * 64,
        )


def test_from_store_artifact_rejects_self_published_promoted_fixture(
    tmp_path: Any,
) -> None:
    """公開 fit の改ざん envelope を store へ置いても formal token を発行しない（M9(a)）。

    1. 正規 calibration commit の logical ID へ別内容を publish することはできない。
    2. 攻撃者 store に自作 descriptor/commit を作っても、frozen config が固定した
       expected calibration identity と一致しないため fail-closed になる。
    """
    from reinbalance_survivors_contracts.artifact_store import (
        ArtifactStore,
        ArtifactStoreError,
    )

    store, descriptors, commit_logical_id, _profile = _producer_calibration_commit(tmp_path)
    frozen_identity = descriptors[1].identity_hash
    profile_node = descriptors[1]
    artifact_file = next(
        ref for ref in profile_node.files
        if ref.logical_id.endswith("/profile.artifact.json")
    )

    residuals = [
        CalibrationResidual(f"s{index}", "f0", "hp_ratio", 0.01, 1.0, 0)
        for index in range(2)
    ]
    forged_wire = fit_error_profile(residuals, ["s0", "s1"], []).to_artifact_wire()
    assert forged_wire["development_only"] is True
    forged_wire["development_only"] = False
    forged_bytes = canonical_json_bytes(forged_wire)

    # 1: producer が確定させた logical ID は別内容で上書きできない。
    with pytest.raises(ArtifactStoreError):
        store.put_bytes(
            logical_id=artifact_file.logical_id,
            data=forged_bytes,
            media_type="application/json",
        )

    # 2: 攻撃者が自前の store/descriptor/commit を作っても expected identity が一致しない。
    attacker = ArtifactStore(tmp_path / "attacker-store")
    forged_refs = {
        name: attacker.put_bytes(
            logical_id=f"perception/package/calibration/forged/{name}",
            data=payload,
            media_type="application/json",
        )
        for name, payload in (
            ("profile.json", canonical_json_bytes(forged_wire["profile"])),
            ("profile.artifact.json", forged_bytes),
            (
                "provenance.json",
                canonical_json_bytes({
                    "schema_version": "perception_calibration_package.v1",
                    "profile_artifact": forged_wire,
                    "subject_hashes": {},
                }),
            ),
        )
    }
    forged_source = ArtifactDescriptor(
        logical_id="perception/capture/source",
        node_kind="source_descriptor",
        producer_id="perception_error_fit",
        producer_version="v2",
        identity_metadata={"split_manifest_hash": "e" * 64},
        files=(forged_refs["profile.json"],),
    )
    forged_node = ArtifactDescriptor(
        logical_id="perception/calibration/forged",
        node_kind="perception_calibration_profile",
        producer_id="perception_error_fit",
        producer_version="v2",
        identity_metadata={
            "profile_hash": forged_wire["profile_hash"],
            "fit_code_hash": forged_wire["fit_code_hash"],
            "subject_hashes": {},
        },
        parents=(forged_source.node_ref(),),
        files=tuple(forged_refs.values()),
    )
    with pytest.raises(FormalVerdictPromotionError, match="does not match the expected"):
        FittedPerceptionErrorProfile.from_store_artifact(
            attacker,
            descriptors=(forged_source, forged_node),
            expected_calibration_identity_hash=frozen_identity,
        )

    forged_commit_id = "perception/calibration_commit/forged"
    attacker.put_bytes(
        logical_id=forged_commit_id,
        data=canonical_json_bytes({
            "schema_version": "perception_calibration_commit.v1",
            "run_key": "forged",
            "profile_descriptor_hash": forged_node.identity_hash,
            "refs": [
                attacker.put_bytes(
                    logical_id=(
                        f"perception/package/descriptors/{descriptor.identity_hash}.json"
                    ),
                    data=canonical_json_bytes(descriptor.to_wire()),
                    media_type="application/json",
                ).to_wire()
                for descriptor in (forged_source, forged_node)
            ],
        }),
        media_type="application/json",
    )
    with pytest.raises(FormalVerdictPromotionError, match="does not match the expected"):
        FittedPerceptionErrorProfile.from_calibration_commit(
            attacker,
            commit_logical_id=forged_commit_id,
            expected_calibration_identity_hash=frozen_identity,
        )
    # 攻撃者 store には正規 commit logical ID 自体が存在しない。
    with pytest.raises(FormalVerdictPromotionError, match="not found in store"):
        FittedPerceptionErrorProfile.from_calibration_commit(
            attacker,
            commit_logical_id=commit_logical_id,
            expected_calibration_identity_hash=frozen_identity,
        )


def test_self_signed_profile_bytes_cannot_promote_development_fixture() -> None:
    """development flagを改ざんしてSHAを自己計算してもformal tokenを発行しない。"""
    import hashlib

    residuals = [
        CalibrationResidual(f"s{i}", "f0", "hp_ratio", 0.01, 1.0, 0)
        for i in range(2)
    ]
    wire = fit_error_profile(residuals, ["s0", "s1"], []).to_artifact_wire()
    wire["development_only"] = False
    data = canonical_json_bytes(wire)

    with pytest.raises(FormalVerdictPromotionError):
        FittedPerceptionErrorProfile._from_verified_bytes(
            data, hashlib.sha256(data).hexdigest()
        )


def test_load_formal_verdict_from_verified_store_requires_store_and_ref(tmp_path: Any) -> None:
    """_load_formal_verdict_from_verified_store は store/ref 引数が必須（P1-2b regression）。

    raw dict だけでは formal token を付与できず、store.verify(ref) 失敗時も拒否される。
    """
    from Tools.Artifacts.artifact_store import ArtifactStore
    from survivors.perception_error_fit import (
        _load_formal_verdict_from_verified_store,
        HashMismatchError,
    )
    from reinbalance_survivors_contracts.artifact_identity import ArtifactRef, artifact_uri

    store = ArtifactStore(str(tmp_path / "store"))
    subjects = _formal_subjects()
    wire = _passing_formal_verdict_wire(subjects)
    # store に保存して valid ref を作る。
    from reinbalance_survivors_contracts.canonical_json import canonical_json_bytes
    data_bytes = canonical_json_bytes(wire)
    real_ref = store.put_bytes(
        logical_id="perception/verdict/test.json",
        data=data_bytes, media_type="application/json",
    )
    current_subjects = dict(subjects)
    # 正常系: store/ref が揃えばロードできる。
    verdict = _load_formal_verdict_from_verified_store(
        wire, store=store, ref=real_ref, current_subject_hashes=current_subjects
    )
    assert verdict.development_only is False
    # 存在しない ref で呼んだ場合は HashMismatchError を送出する。
    ghost_sha = "9" * 64
    ghost_ref = ArtifactRef(
        logical_id="perception/verdict/ghost.json",
        sha256=ghost_sha, size_bytes=16,
        media_type="application/json",
        store_uri=artifact_uri(ghost_sha),
    )
    with pytest.raises(HashMismatchError):
        _load_formal_verdict_from_verified_store(
            wire, store=store, ref=ghost_ref, current_subject_hashes=current_subjects
        )


def test_verify_formal_runtime_release_rejects_missing_file(tmp_path: Any) -> None:
    """ArtifactStore に verdict ファイルが存在しない場合は拒否される。"""
    from Tools.Artifacts.artifact_store import ArtifactStore
    store = ArtifactStore(str(tmp_path / "store"))
    subjects = _formal_subjects()
    wire = _passing_formal_verdict_wire(subjects)
    missing_sha256 = _h("9")
    ghost_ref = ArtifactRef(
        logical_id="perception/verdict/ghost.json",
        sha256=missing_sha256, size_bytes=16,
        media_type="application/json",
        store_uri=artifact_uri(missing_sha256),
    )
    src_ref = _node("source", "source_descriptor", ch="1")
    profile = _node("profile", "perception_calibration_profile", (src_ref.node_ref(),), "6")
    verdict_desc = ArtifactDescriptor(
        logical_id="perception-verdict",
        node_kind="perception_final_verdict",
        producer_id="test-producer", producer_version="v1",
        identity_metadata={
            "verdict_id": wire["verdict_id"], "seal_id": wire["seal_id"],
            "passed": True, "development_only": False,
            "subject_hashes": {name: subjects[name] for name in _HASH_FIELDS},
        },
        parents=(profile.node_ref(),), files=(ghost_ref,),
    )
    dag = _build_runtime_dag(verdict_desc, subjects, profile)
    with pytest.raises(ValueError):
        verify_formal_runtime_release(dag, store)
