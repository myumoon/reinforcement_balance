"""iteration 2: exact fit metadata、lineage seal、verdict replay gate。"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from Tools.Artifacts.artifact_store import ArtifactStore
from survivors.perception_benchmark import BenchmarkRecord, recompute_gate_from_metrics, run_benchmark
from survivors.perception_error_fit import (
    FINAL_VERDICT_SCHEMA_VERSION,
    CalibrationResidual,
    EmptyResidualError,
    FinalFitMixingError,
    FinalSessionAlreadyOpenedError,
    FinalSessionNotInSealError,
    FormalVerdictPromotionError,
    HashMismatchError,
    InvalidResidualError,
    PerceptionFinalVerdict,
    StaleVerdictError,
    create_lineage_seal,
    create_synthetic_final_verdict,
    fit_error_profile,
    load_final_verdict,
)


def _residual(session_id: str, field: str = "hp_ratio", value: float = 0.1, **kwargs) -> CalibrationResidual:
    return CalibrationResidual(session_id, "f0", field, value, 0.8, 1, 2.0, **kwargs)


def _seal_subjects() -> dict[str, str]:
    return {
        "parser_artifact_hash": "a" * 64,
        "detector_artifact_hash": "b" * 64,
        "model_hash": "8" * 64,
        "build_hash": "9" * 64,
        "assembler_schema_hash": "c" * 64,
        "ui_presentation_schema_hash": "d" * 64,
        "ui_presentation_golden_fixture_hash": "0" * 64,
        "config_hash": "e" * 64,
        "capture_dataset_hash": "f" * 64,
        "calibration_profile_hash": "1" * 64,
        "threshold_hash": "2" * 64,
        "atlas_vocabulary_hash": "3" * 64,
        "assembler_impl_hash": "4" * 64,
        "roi_resolver_input_hash": "5" * 64,
        "benchmark_fit_code_hash": "6" * 64,
    }


def _verdict_subjects() -> dict[str, str]:
    return {**_seal_subjects(), "lineage_seal_hash": "7" * 64}


def _full_report():
    """passed=True になる最小ベンチマーク結果を生成する。

    per-state gate（最低 2 種類の screen_state、各 3 件以上）を満たすため
    gameplay と level_up_items の両方を含めます。
    """
    rows = []
    for session_index, session in enumerate(("s0", "s1")):
        for frame_index, density in enumerate((0.1, 0.2)):
            common = dict(session=session, frame=f"f{frame_index}")
            values = (
                ("screen_state", "gameplay", "gameplay"),
                ("timer_seconds", 10.0, 10.0), ("level", 2, 2),
                ("hp_ratio", 0.8, 0.8), ("xp_ratio", 0.3, 0.3),
                ("inventory_top1", "i" * 64, "i" * 64),
                ("choice_top1", "c" * 64, "c" * 64),
                ("entity_density", density, density),
                ("nearest_distance", 0.2, 0.2),
                ("ui_roi_center_error", 0.0, 0.0),
                ("ui_inside_region", True, True),
                ("ui_false_positive", False, False),
                ("confidence", 1.0, 1.0),
            )
            for field, ground, predicted in values:
                rows.append(BenchmarkRecord(
                    common["frame"], common["session"], "error_calibration", "raw",
                    field, ground, predicted, 1.0, 1.0,
                ))
    # per-state gate: level_up_items を 3 件追加して最低 2 種類を満たす
    for i in range(3):
        rows.append(BenchmarkRecord(
            f"lui-{i}", "s0", "error_calibration", "raw",
            "screen_state", "level_up_items", "level_up_items", 1.0, 1.0,
        ))
    report = run_benchmark(rows, n_bootstrap=20)
    # BenchmarkRecord-only fixture には V1 snapshot context がないため named slice を明示する。
    # 必須名・category 数・最低 3 件をすべて満たし、verdict loader の再計算も維持する。
    report.slice_counts.update({
        "time_band:early": 3, "time_band:late": 3,
        "foreground_class:enemy_boss": 6, "foreground_class:hazard": 6,
        "event:boss": 6, "event:hazard": 6, "event:level_up": 6,
    })
    report.passed, report.blocking_reasons = recompute_gate_from_metrics(
        report.metrics_wire()
    )
    assert report.passed
    return report


class TestFit:
    def test_zero_and_unknown_residual_sets_rejected(self) -> None:
        with pytest.raises(EmptyResidualError):
            fit_error_profile([], ["cal"], [])
        with pytest.raises(FinalFitMixingError, match="outside"):
            fit_error_profile([_residual("unknown")], ["cal"], [])

    def test_underpowered_field_rejected(self) -> None:
        with pytest.raises(EmptyResidualError, match="underpowered"):
            fit_error_profile([_residual("cal")], ["cal"], [])

    def test_every_calibration_session_requires_samples(self) -> None:
        with pytest.raises(EmptyResidualError, match="0 residuals"):
            fit_error_profile([_residual("c0")], ["c0", "c1"], [])

    def test_latency_confidence_age_and_metadata_are_fitted(self) -> None:
        profile = fit_error_profile(
            [
                CalibrationResidual("c0", "f0", "hp_ratio", -0.1, 1.0, 0, 1.0),
                CalibrationResidual("c0", "f1", "hp_ratio", 0.3, 0.1, 9, 5.0),
            ],
            ["c0"], ["final"], calibration_session_hashes={"c0": "a" * 64},
        )
        assert 1.0 < profile.latency_mean_frames < 5.0
        assert profile.hud_hp_misread_std > 0.0
        assert profile.calibration_session_hashes == {"c0": "a" * 64}
        assert profile.field_sample_counts == {"hp_ratio": 2}
        assert len(profile.fit_code_hash) == 64
        # existing consumer wire は metadata を追加せず引き続き読める。
        from reinbalance_survivors_contracts.perception_error import PerceptionErrorProfile
        assert PerceptionErrorProfile.from_wire(profile.to_wire()).profile_hash == profile.profile_hash

    def test_confusion_matrix_uses_observed_categories(self) -> None:
        profile = fit_error_profile(
            [
                _residual("c0", "item_category", 0.0, ground_truth_category=0, predicted_category=1),
                _residual("c0", "item_category", 0.0, ground_truth_category=0, predicted_category=1),
            ], ["c0"], [],
        )
        assert profile.item_confusion_matrix[0][1] == pytest.approx(1.0)
        assert profile.item_confusion_matrix[0][0] == pytest.approx(0.0)

    @pytest.mark.parametrize("field,value", [("residual", float("nan")), ("confidence", True), ("latency_frames", float("inf"))])
    def test_nonfinite_and_bool_rejected(self, field: str, value: object) -> None:
        args = dict(session_id="c", frame_id="f", field="hp_ratio", residual=0.1, confidence=0.9, age_frames=0, latency_frames=1.0)
        args[field] = value
        with pytest.raises(InvalidResidualError):
            CalibrationResidual(**args)

    @pytest.mark.parametrize(
        "field,value",
        [
            ("burst_enter", -0.01),
            ("burst_exit", 1.01),
            ("burst_dropout", 2.0),
            ("unknown_screen_collapse", -1.0),
            ("unknown_screen_collapse_duration", -0.01),
            ("coord_quantization_px", -0.01),
        ],
    )
    def test_probability_and_nonnegative_residual_boundaries_reject(
        self, field: str, value: float
    ) -> None:
        with pytest.raises(InvalidResidualError):
            CalibrationResidual("c", "f", field, value, 0.9, 0, 0.0)


class TestLineageSeal:
    def _manifest(self, tmp_path: Path, name: str, content: bytes = b"manifest") -> tuple[Path, str]:
        path = tmp_path / f"{name}.json"
        path.write_bytes(content)
        return path, hashlib.sha256(content).hexdigest()

    def test_empty_final_set_and_public_formal_override_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            create_lineage_seal(final_session_hashes={}, **_seal_subjects())
        path_hashes = {"f0": "f" * 64}
        with pytest.raises(ValueError):
            create_lineage_seal(final_session_hashes=path_hashes, development_only=False, **_seal_subjects())

    def test_identity_binds_all_hashes_and_sorted_sessions(self) -> None:
        first = create_lineage_seal(final_session_hashes={"b": "8" * 64, "a": "9" * 64}, **_seal_subjects())
        second = create_lineage_seal(final_session_hashes={"a": "9" * 64, "b": "8" * 64}, **_seal_subjects())
        assert first.seal_id == second.seal_id
        changed = dict(_seal_subjects(), threshold_hash="0" * 64)
        assert create_lineage_seal(final_session_hashes={"a": "9" * 64, "b": "8" * 64}, **changed).seal_id != first.seal_id

    def test_open_rechecks_real_file_hash_and_store_marker_is_create_once(self, tmp_path: Path) -> None:
        path, digest = self._manifest(tmp_path, "f0")
        store = ArtifactStore(tmp_path / "artifacts")
        seal = create_lineage_seal(final_session_hashes={"f0": digest}, store=store, **_seal_subjects())
        seal.open_session("f0", path)
        recreated = create_lineage_seal(final_session_hashes={"f0": digest}, store=store, **_seal_subjects())
        with pytest.raises(FinalSessionAlreadyOpenedError):
            recreated.open_session("f0", path)

    def test_unknown_and_tampered_session_rejected(self, tmp_path: Path) -> None:
        path, digest = self._manifest(tmp_path, "f0")
        seal = create_lineage_seal(final_session_hashes={"f0": digest}, **_seal_subjects())
        with pytest.raises(FinalSessionNotInSealError):
            seal.open_session("unknown", path)
        path.write_bytes(b"tampered")
        with pytest.raises(HashMismatchError):
            seal.open_session("f0", path)


class TestFinalVerdict:
    def _wire(self) -> dict[str, object]:
        verdict = create_synthetic_final_verdict(
            _full_report(), seal_id="a" * 64, final_session_ids=["f0"],
            **_verdict_subjects(),
        )
        return verdict.to_wire()

    def _load(self, wire: dict[str, object]):
        return load_final_verdict(wire, current_subject_hashes=_verdict_subjects())

    def test_exact_fields_all_hashes_and_gate_recompute(self) -> None:
        wire = self._wire()
        assert wire["schema_version"] == FINAL_VERDICT_SCHEMA_VERSION
        assert self._load(wire).passed is True
        with pytest.raises(ValueError, match="exactly"):
            self._load({**wire, "unknown": 1})
        bad_hash = dict(wire, threshold_hash="bad")
        with pytest.raises(ValueError, match="SHA-256"):
            self._load(bad_hash)
        stale_gate = dict(wire, passed=False)
        with pytest.raises(StaleVerdictError, match="gate"):
            self._load(stale_gate)

    def test_stale_producer_rejected(self) -> None:
        wire = self._wire()
        subjects = _verdict_subjects()
        stale_subjects = {**subjects, "parser_artifact_hash": "0" * 64}
        with pytest.raises(StaleVerdictError):
            load_final_verdict(wire, current_subject_hashes=stale_subjects)

    def test_direct_formal_and_inconsistent_false_false_rejected(self) -> None:
        base = self._wire()
        kwargs = {name: base[name] for name in (
            "verdict_id", "seal_id", "final_session_ids", "parser_artifact_hash",
            "detector_artifact_hash", "model_hash", "build_hash", "assembler_schema_hash",
            "ui_presentation_schema_hash", "ui_presentation_golden_fixture_hash",
            "config_hash", "capture_dataset_hash", "calibration_profile_hash", "threshold_hash",
            "atlas_vocabulary_hash", "assembler_impl_hash", "roi_resolver_input_hash",
            "benchmark_fit_code_hash", "lineage_seal_hash", "metrics", "passed", "blocking_reasons",
        )}
        with pytest.raises(FormalVerdictPromotionError):
            PerceptionFinalVerdict(**kwargs, development_only=False, formal_perception_verdict_eligible=True)
        with pytest.raises(FormalVerdictPromotionError):
            PerceptionFinalVerdict(**kwargs, development_only=False, formal_perception_verdict_eligible=False)
