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


def test_from_verified_bytes_allows_formal_profile() -> None:
    """_from_verified_bytes はストア検証後に formal profile をロードできる（P1-1 regression）。

    from_artifact_wire() が拒否する development_only=False の wire を、
    content-hash 検証付きの _from_verified_bytes() は受け付けることを確認する。
    """
    import hashlib

    # dev profile を作り artifact wire を取得する。
    residuals = [
        CalibrationResidual(f"s{i}", "f0", "hp_ratio", 0.01, 1.0, 0)
        for i in range(2)
    ]
    dev_profile = fit_error_profile(residuals, ["s0", "s1"], [])
    assert dev_profile.development_only is True
    # development_only=True 版: _from_verified_bytes でもロードできる。
    wire = dev_profile.to_artifact_wire()
    wire_bytes = canonical_json_bytes(wire)
    sha256 = hashlib.sha256(wire_bytes).hexdigest()
    restored_dev = FittedPerceptionErrorProfile._from_verified_bytes(wire_bytes, sha256)
    assert restored_dev.development_only is True
    # development_only=False に書き換えた wire: from_artifact_wire は拒否するが
    # _from_verified_bytes は content-hash が正しければロードできる（formal token 付与）。
    formal_wire = dict(wire)
    formal_wire["development_only"] = False
    formal_bytes = canonical_json_bytes(formal_wire)
    formal_sha256 = hashlib.sha256(formal_bytes).hexdigest()
    with pytest.raises(FormalVerdictPromotionError):
        FittedPerceptionErrorProfile.from_artifact_wire(formal_wire)
    restored_formal = FittedPerceptionErrorProfile._from_verified_bytes(formal_bytes, formal_sha256)
    assert restored_formal.development_only is False
    # hash 不一致は HashMismatchError を送出する。
    with pytest.raises(HashMismatchError):
        FittedPerceptionErrorProfile._from_verified_bytes(wire_bytes, "a" * 64)


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
