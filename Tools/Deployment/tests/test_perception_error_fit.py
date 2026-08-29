"""perception_error_fit の calibration fit・lineage seal・verdict テスト。

synthetic residuals から PerceptionErrorProfile を fit し、
final session の lineage seal と stale-verdict 検証を確認します。
全 fixture は development_only=True です。
"""

from __future__ import annotations

import pytest
from reinbalance_survivors_contracts.perception_error import PerceptionErrorProfile
from survivors.perception_error_fit import (
    FINAL_VERDICT_SCHEMA_VERSION,
    CalibrationResidual,
    FinalFitMixingError,
    FinalSessionAlreadyOpenedError,
    SessionOverlapError,
    StaleVerdictError,
    create_lineage_seal,
    fit_error_profile,
    load_final_verdict,
    simulator_distance_report,
)

# 固定長 64 文字ダミーハッシュ
_PH = "a" * 64  # parser
_DH = "b" * 64  # detector
_AH = "c" * 64  # assembler schema
_CFH = "d" * 64  # config
_UIH = "e" * 64  # UI schema


def _residual(
    session_id: str,
    field: str = "hp_ratio",
    residual: float = 0.05,
) -> CalibrationResidual:
    return CalibrationResidual(
        session_id=session_id,
        frame_id="f0",
        field=field,
        residual=residual,
        confidence=0.9,
        age_frames=0,
    )


class TestFitErrorProfile:
    """calibration residuals から PerceptionErrorProfile を fit する。"""

    def test_fit_returns_valid_profile(self) -> None:
        residuals = [_residual("cal_0") for _ in range(20)]
        profile = fit_error_profile(residuals, ["cal_0"], [])
        assert isinstance(profile, PerceptionErrorProfile)

    def test_calibration_session_ids_saved(self) -> None:
        residuals = [_residual("s1")] + [_residual("s2")]
        profile = fit_error_profile(residuals, ["s1", "s2"], [])
        assert "s1" in profile.calibration_session_ids
        assert "s2" in profile.calibration_session_ids

    def test_final_ids_saved_in_exclusion(self) -> None:
        """final_e2e_session_ids がプロファイルの exclusion list に保存される。"""
        residuals = [_residual("cal_0")]
        profile = fit_error_profile(residuals, ["cal_0"], ["final_0"])
        assert "final_0" in profile.final_e2e_session_ids

    def test_session_overlap_fails(self) -> None:
        """同じ session_id が calibration と final の両方に含まれるとき拒否する。"""
        residuals = [_residual("shared")]
        with pytest.raises(SessionOverlapError, match="overlap"):
            fit_error_profile(residuals, ["shared"], ["shared"])

    def test_final_data_mixing_fails(self) -> None:
        """final セッションの residual を calibration fit に混入させると拒否する。"""
        residuals = [_residual("final_0", "hp_ratio", 0.1)]
        with pytest.raises(FinalFitMixingError):
            fit_error_profile(residuals, ["cal_0"], ["final_0"])

    def test_hp_misread_std_nonzero_when_residuals_vary(self) -> None:
        """HP residual が分散を持つとき hud_hp_misread_std > 0 になる。"""
        residuals = [
            _residual("cal_0", "hp_ratio", 0.1),
            _residual("cal_0", "hp_ratio", -0.1),
        ]
        profile = fit_error_profile(residuals, ["cal_0"], [])
        assert profile.hud_hp_misread_std > 0.0

    def test_empty_residuals_yields_default_profile(self) -> None:
        """residuals が空のとき全パラメータ 0 のデフォルトプロファイルになる。"""
        profile = fit_error_profile([], ["cal_0"], [])
        assert isinstance(profile, PerceptionErrorProfile)
        assert profile.hud_hp_misread_std == 0.0


class TestSimulatorDistance:
    """calibrated profile と simulator profile の距離を計算する。"""

    def test_same_profile_zero_distance(self) -> None:
        profile = PerceptionErrorProfile()
        distances = simulator_distance_report(profile, profile)
        assert all(v == 0.0 for v in distances.values())

    def test_different_profiles_positive_distance(self) -> None:
        base = PerceptionErrorProfile()
        fitted = PerceptionErrorProfile(hud_hp_misread_std=0.05, coord_noise_std=2.0)
        distances = simulator_distance_report(fitted, base)
        assert distances["hp_misread_std_diff"] == pytest.approx(0.05)
        assert distances["coord_noise_std_diff"] == pytest.approx(2.0)

    def test_distance_report_has_expected_keys(self) -> None:
        d = simulator_distance_report(PerceptionErrorProfile(), PerceptionErrorProfile())
        assert "latency_mean_diff" in d
        assert "coord_noise_std_diff" in d
        assert "hp_misread_std_diff" in d


class TestLineageSeal:
    """create-once final lineage seal を確認する。"""

    def test_create_seal_has_64_char_id(self) -> None:
        seal = create_lineage_seal(_PH, _DH, _AH, _CFH)
        assert isinstance(seal.seal_id, str)
        assert len(seal.seal_id) == 64

    def test_seal_development_only_by_default(self) -> None:
        seal = create_lineage_seal(_PH, _DH, _AH, _CFH)
        assert seal.development_only is True

    def test_open_session_once_ok(self) -> None:
        seal = create_lineage_seal(_PH, _DH, _AH, _CFH)
        seal.open_session("final_0")
        assert "final_0" in seal.opened_session_ids

    def test_open_session_twice_fails(self) -> None:
        """2 回目の final session 開封を拒否する（create-once）。"""
        seal = create_lineage_seal(_PH, _DH, _AH, _CFH)
        seal.open_session("final_0")
        with pytest.raises(FinalSessionAlreadyOpenedError):
            seal.open_session("final_1")

    def test_verify_hashes_matching_ok(self) -> None:
        seal = create_lineage_seal(_PH, _DH, _AH, _CFH)
        seal.verify_hashes(parser=_PH, detector=_DH, assembler=_AH, config=_CFH)  # エラーなし

    def test_verify_hashes_stale_parser_fails(self) -> None:
        seal = create_lineage_seal(_PH, _DH, _AH, _CFH)
        with pytest.raises(StaleVerdictError):
            seal.verify_hashes(parser="x" * 64, detector=_DH, assembler=_AH, config=_CFH)

    def test_verify_hashes_stale_config_fails(self) -> None:
        seal = create_lineage_seal(_PH, _DH, _AH, _CFH)
        with pytest.raises(StaleVerdictError):
            seal.verify_hashes(parser=_PH, detector=_DH, assembler=_AH, config="z" * 64)

    def test_seal_id_deterministic(self) -> None:
        """同じ hashes で作成した seal は同じ seal_id を持つ。"""
        s1 = create_lineage_seal(_PH, _DH, _AH, _CFH)
        s2 = create_lineage_seal(_PH, _DH, _AH, _CFH)
        assert s1.seal_id == s2.seal_id

    def test_different_hashes_give_different_seal_id(self) -> None:
        s1 = create_lineage_seal(_PH, _DH, _AH, _CFH)
        s2 = create_lineage_seal("f" * 64, _DH, _AH, _CFH)
        assert s1.seal_id != s2.seal_id

    def test_seal_to_wire_roundtrip(self) -> None:
        """to_wire() が schema_version と seal_id を含む。"""
        seal = create_lineage_seal(_PH, _DH, _AH, _CFH)
        wire = seal.to_wire()
        assert wire["seal_id"] == seal.seal_id
        assert wire["development_only"] is True


class TestFinalVerdictLoad:
    """PerceptionFinalVerdict の stale 検証を確認する。"""

    def _make_wire(self, **overrides: object) -> dict[str, object]:
        base: dict[str, object] = {
            "schema_version": FINAL_VERDICT_SCHEMA_VERSION,
            "verdict_id": "vtest",
            "seal_id": "s" * 64,
            "final_session_ids": ["final_0"],
            "parser_artifact_hash": _PH,
            "detector_artifact_hash": _DH,
            "assembler_schema_hash": _AH,
            "ui_presentation_schema_hash": _UIH,
            "config_hash": _CFH,
            "metrics": {},
            "passed": False,
            "blocking_reasons": ["development-only fixture"],
            "development_only": True,
            "formal_perception_verdict_eligible": False,
        }
        base.update(overrides)
        return base

    def test_load_matching_hashes_ok(self) -> None:
        wire = self._make_wire()
        verdict = load_final_verdict(
            wire,
            current_parser_hash=_PH,
            current_detector_hash=_DH,
            current_assembler_hash=_AH,
            current_config_hash=_CFH,
            current_ui_schema_hash=_UIH,
        )
        assert verdict.verdict_id == "vtest"

    def test_load_stale_parser_fails(self) -> None:
        """parser hash が変わると StaleVerdictError を送出する。"""
        wire = self._make_wire()
        with pytest.raises(StaleVerdictError, match="parser_artifact_hash"):
            load_final_verdict(
                wire,
                current_parser_hash="x" * 64,
                current_detector_hash=_DH,
                current_assembler_hash=_AH,
                current_config_hash=_CFH,
                current_ui_schema_hash=_UIH,
            )

    def test_load_stale_detector_fails(self) -> None:
        wire = self._make_wire()
        with pytest.raises(StaleVerdictError, match="detector_artifact_hash"):
            load_final_verdict(
                wire,
                current_parser_hash=_PH,
                current_detector_hash="y" * 64,
                current_assembler_hash=_AH,
                current_config_hash=_CFH,
                current_ui_schema_hash=_UIH,
            )

    def test_load_stale_ui_schema_fails(self) -> None:
        wire = self._make_wire()
        with pytest.raises(StaleVerdictError, match="ui_presentation_schema_hash"):
            load_final_verdict(
                wire,
                current_parser_hash=_PH,
                current_detector_hash=_DH,
                current_assembler_hash=_AH,
                current_config_hash=_CFH,
                current_ui_schema_hash="z" * 64,
            )

    def test_stale_config_fails(self) -> None:
        wire = self._make_wire()
        with pytest.raises(StaleVerdictError, match="config_hash"):
            load_final_verdict(
                wire,
                current_parser_hash=_PH,
                current_detector_hash=_DH,
                current_assembler_hash=_AH,
                current_config_hash="w" * 64,
                current_ui_schema_hash=_UIH,
            )

    def test_wrong_schema_version_fails(self) -> None:
        wire = self._make_wire(schema_version="wrong.v0")
        with pytest.raises(ValueError, match="schema"):
            load_final_verdict(
                wire,
                current_parser_hash=_PH,
                current_detector_hash=_DH,
                current_assembler_hash=_AH,
                current_config_hash=_CFH,
                current_ui_schema_hash=_UIH,
            )

    def test_verdict_development_only_true(self) -> None:
        """synthetic fixture の verdict は development_only=True。"""
        wire = self._make_wire()
        verdict = load_final_verdict(
            wire,
            current_parser_hash=_PH,
            current_detector_hash=_DH,
            current_assembler_hash=_AH,
            current_config_hash=_CFH,
            current_ui_schema_hash=_UIH,
        )
        assert verdict.development_only is True
        assert verdict.formal_perception_verdict_eligible is False
