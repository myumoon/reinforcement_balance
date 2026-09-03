"""formal promotion guard の回帰テスト。

synthetic/development 由来の成果物が formal writer・formal verdict factory・restore
verdict gate へ到達できないことを、実 formal session/依存なしで検証する。実 formal
成功経路テスト（test_perception_formal_runner.py）は正式依存が揃うまで skip のまま残す。
"""

from __future__ import annotations

import pytest

from benchmark_survivors_perception import _validate_restore_verdict
from reinbalance_survivors_contracts.artifact_identity import (
    RESTORE_TEST_VERDICT_SCHEMA_VERSION,
    ArtifactDescriptor,
    ArtifactRef,
    RestoreTestVerdict,
    artifact_uri,
)
from survivors.perception_benchmark import BenchmarkRecord, run_benchmark
from survivors.perception_error_fit import (
    _HASH_FIELDS,
    CalibrationResidual,
    FittedPerceptionErrorProfile,
    FormalVerdictPromotionError,
    _create_formal_final_verdict,
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
