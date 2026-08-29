"""perception_error_fit の calibration fit・lineage seal・verdict テスト。

synthetic residuals から PerceptionErrorProfile を fit し、
final session の lineage seal と stale-verdict 検証を確認します。
全 fixture は development_only=True です。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from reinbalance_survivors_contracts.perception_error import PerceptionErrorProfile
from survivors.perception_error_fit import (
    FINAL_VERDICT_SCHEMA_VERSION,
    CalibrationResidual,
    FinalFitMixingError,
    FinalSessionAlreadyOpenedError,
    FinalSessionNotInSealError,
    FormalVerdictPromotionError,
    InvalidResidualError,
    PerceptionCalibrationVerdict,
    PerceptionFinalVerdict,
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
        residuals = [_residual("s1")]
        profile = fit_error_profile(residuals, ["s1"], [])
        assert "s1" in profile.calibration_session_ids

    def test_final_ids_in_exclusion_list(self) -> None:
        residuals = [_residual("s1")]
        profile = fit_error_profile(residuals, ["s1"], ["final_0"])
        assert "final_0" in profile.final_e2e_session_ids

    def test_overlap_raises_session_overlap_error(self) -> None:
        with pytest.raises(SessionOverlapError):
            fit_error_profile([_residual("s1")], ["s1"], ["s1"])

    def test_final_residual_raises_mixing_error(self) -> None:
        with pytest.raises(FinalFitMixingError):
            fit_error_profile(
                [_residual("final_0", residual=0.1)],
                ["cal_0"],
                ["final_0"],
            )

    def test_hp_std_nonzero_when_residuals_vary(self) -> None:
        """HP residual が分散を持つとき hud_hp_misread_std > 0。"""
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

    def test_nan_residual_rejected(self) -> None:
        """NaN の residual は CalibrationResidual 生成時に拒否される。"""
        with pytest.raises(InvalidResidualError):
            CalibrationResidual(
                session_id="s0",
                frame_id="f0",
                field="hp_ratio",
                residual=float("nan"),
                confidence=0.9,
                age_frames=0,
            )

    def test_inf_residual_rejected(self) -> None:
        """Inf の residual は CalibrationResidual 生成時に拒否される。"""
        with pytest.raises(InvalidResidualError):
            CalibrationResidual(
                session_id="s0",
                frame_id="f0",
                field="hp_ratio",
                residual=float("inf"),
                confidence=0.9,
                age_frames=0,
            )

    def test_burst_fields_fit(self) -> None:
        """burst パラメータを fit できる（全フィールドカバー）。"""
        residuals = [
            _residual("c0", "burst_enter", 0.1),
            _residual("c0", "burst_exit", 0.9),
            _residual("c0", "burst_dropout", 0.2),
        ]
        profile = fit_error_profile(residuals, ["c0"], [])
        assert isinstance(profile, PerceptionErrorProfile)
        assert 0.0 <= profile.burst_enter_prob <= 1.0
        assert 0.0 <= profile.burst_exit_prob <= 1.0
        assert 0.0 <= profile.burst_dropout_prob <= 1.0


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
        assert "burst_enter_diff" in d
        assert "burst_exit_diff" in d
        assert "burst_dropout_diff" in d
        assert "unknown_collapse_prob_diff" in d


class TestLineageSeal:
    """create-once final lineage seal を確認する。"""

    def test_create_seal_has_64_char_id(self) -> None:
        seal = create_lineage_seal(_PH, _DH, _AH, _CFH)
        assert isinstance(seal.seal_id, str)
        assert len(seal.seal_id) == 64

    def test_seal_development_only_by_default(self) -> None:
        seal = create_lineage_seal(_PH, _DH, _AH, _CFH)
        assert seal.development_only is True

    def test_open_session_tracks_id(self) -> None:
        seal = create_lineage_seal(
            _PH, _DH, _AH, _CFH,
            final_session_set=frozenset({"final_0"}),
        )
        seal.open_session("final_0")
        assert "final_0" in seal.opened_session_ids

    def test_same_session_twice_rejected(self) -> None:
        """同一 session_id の 2 回目開封を拒否する（create-once）。"""
        seal = create_lineage_seal(
            _PH, _DH, _AH, _CFH,
            final_session_set=frozenset({"final_0", "final_1", "final_2"}),
        )
        seal.open_session("final_0")
        with pytest.raises(FinalSessionAlreadyOpenedError):
            seal.open_session("final_0")

    def test_three_different_sessions_each_once(self) -> None:
        """固定集合内の 3 件の別 session は各 1 回ずつ開封できる（必須回帰テスト）。"""
        seal = create_lineage_seal(
            _PH, _DH, _AH, _CFH,
            final_session_set=frozenset({"s0", "s1", "s2"}),
        )
        seal.open_session("s0")
        seal.open_session("s1")
        seal.open_session("s2")
        assert len(seal.opened_session_ids) == 3

    def test_session_not_in_set_rejected(self) -> None:
        """固定集合に含まれない session_id は拒否する。"""
        seal = create_lineage_seal(
            _PH, _DH, _AH, _CFH,
            final_session_set=frozenset({"s0", "s1", "s2"}),
        )
        with pytest.raises(FinalSessionNotInSealError):
            seal.open_session("unknown_session")

    def test_verify_hashes_ok_with_same(self) -> None:
        seal = create_lineage_seal(_PH, _DH, _AH, _CFH)
        seal.verify_hashes(parser=_PH, detector=_DH, assembler=_AH, config=_CFH)

    def test_verify_hashes_fails_on_parser_change(self) -> None:
        seal = create_lineage_seal(_PH, _DH, _AH, _CFH)
        with pytest.raises(StaleVerdictError):
            seal.verify_hashes(parser="x" * 64, detector=_DH, assembler=_AH, config=_CFH)

    def test_verify_hashes_fails_on_config_change(self) -> None:
        seal = create_lineage_seal(_PH, _DH, _AH, _CFH)
        with pytest.raises(StaleVerdictError):
            seal.verify_hashes(parser=_PH, detector=_DH, assembler=_AH, config="y" * 64)

    def test_same_hashes_same_seal_id(self) -> None:
        s1 = create_lineage_seal(_PH, _DH, _AH, _CFH)
        s2 = create_lineage_seal(_PH, _DH, _AH, _CFH)
        assert s1.seal_id == s2.seal_id

    def test_different_hashes_different_seal_id(self) -> None:
        s1 = create_lineage_seal(_PH, _DH, _AH, _CFH)
        s2 = create_lineage_seal("x" * 64, _DH, _AH, _CFH)
        assert s1.seal_id != s2.seal_id

    def test_to_wire_has_development_only_true(self) -> None:
        """to_wire に development_only=true を含む。"""
        seal = create_lineage_seal(_PH, _DH, _AH, _CFH)
        wire = seal.to_wire()
        assert wire["seal_id"] == seal.seal_id
        assert wire["development_only"] is True

    def test_durable_seal_survives_reload(self) -> None:
        """ArtifactStore 永続化後に load_from_store で開封状態を復元できる（必須回帰テスト）。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "seal.json"
            seal = create_lineage_seal(
                _PH, _DH, _AH, _CFH,
                final_session_set=frozenset({"s0", "s1", "s2"}),
                store_path=store_path,
            )
            seal.open_session("s0")
            # reload: process 再起動をシミュレート
            from survivors.perception_error_fit import FinalLineageSeal
            restored = FinalLineageSeal.load_from_store(store_path)
            # 復元後も s0 は開封済みとして記録されている
            assert "s0" in restored.opened_session_ids
            # 再度 s0 を開封しようとすると拒否される
            with pytest.raises(FinalSessionAlreadyOpenedError):
                restored.open_session("s0")
            # s1, s2 は未開封なので開封できる
            restored.open_session("s1")
            assert "s1" in restored.opened_session_ids


class TestFinalVerdictLoad:
    """PerceptionFinalVerdict の stale 検証と flag 整合性を確認する。"""

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
            "capture_dataset_hash": "",
            "calibration_profile_hash": "",
            "threshold_hash": "",
            "atlas_vocabulary_hash": "",
            "assembler_impl_hash": "",
            "roi_resolver_input_hash": "",
            "benchmark_fit_code_hash": "",
            "lineage_seal_hash": "",
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
                current_detector_hash="x" * 64,
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
                current_ui_schema_hash="x" * 64,
            )

    def test_load_stale_config_fails(self) -> None:
        wire = self._make_wire()
        with pytest.raises(StaleVerdictError, match="config_hash"):
            load_final_verdict(
                wire,
                current_parser_hash=_PH,
                current_detector_hash=_DH,
                current_assembler_hash=_AH,
                current_config_hash="x" * 64,
                current_ui_schema_hash=_UIH,
            )

    def test_wrong_schema_version_fails(self) -> None:
        wire = self._make_wire(schema_version="wrong.v0")
        with pytest.raises(ValueError):
            load_final_verdict(
                wire,
                current_parser_hash=_PH,
                current_detector_hash=_DH,
                current_assembler_hash=_AH,
                current_config_hash=_CFH,
                current_ui_schema_hash=_UIH,
            )

    def test_development_flags_preserved(self) -> None:
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


class TestFormalPromotionBlocked:
    """synthetic → formal への昇格経路が閉じていることを確認する（必須回帰テスト）。"""

    def test_development_only_false_formal_eligible_true_passed_true_blocked(self) -> None:
        """synthetic metrics に development_only=False, formal_eligible=True, passed=True を
        設定しても PerceptionFinalVerdict の作成で FormalVerdictPromotionError が発生する。"""
        with pytest.raises(FormalVerdictPromotionError):
            PerceptionFinalVerdict(
                verdict_id="fake",
                seal_id="s" * 64,
                final_session_ids=["f0"],
                parser_artifact_hash=_PH,
                detector_artifact_hash=_DH,
                assembler_schema_hash=_AH,
                ui_presentation_schema_hash=_UIH,
                config_hash=_CFH,
                metrics={},
                passed=True,
                blocking_reasons=[],
                development_only=False,
                formal_perception_verdict_eligible=True,
            )

    def test_load_inconsistent_flags_fails(self) -> None:
        """load_final_verdict でも formal_eligible=True かつ development_only=True を拒否。"""
        base: dict[str, object] = {
            "schema_version": FINAL_VERDICT_SCHEMA_VERSION,
            "verdict_id": "v",
            "seal_id": "s" * 64,
            "final_session_ids": ["f0"],
            "parser_artifact_hash": _PH,
            "detector_artifact_hash": _DH,
            "assembler_schema_hash": _AH,
            "ui_presentation_schema_hash": _UIH,
            "config_hash": _CFH,
            "metrics": {},
            "passed": False,
            "blocking_reasons": [],
            "development_only": True,
            "formal_perception_verdict_eligible": True,  # 不整合
        }
        with pytest.raises(FormalVerdictPromotionError):
            load_final_verdict(
                base,
                current_parser_hash=_PH,
                current_detector_hash=_DH,
                current_assembler_hash=_AH,
                current_config_hash=_CFH,
                current_ui_schema_hash=_UIH,
            )

    def test_passed_with_blocking_reasons_inconsistent(self) -> None:
        """passed=True かつ blocking_reasons が非空は拒否する。"""
        base: dict[str, object] = {
            "schema_version": FINAL_VERDICT_SCHEMA_VERSION,
            "verdict_id": "v",
            "seal_id": "s" * 64,
            "final_session_ids": ["f0"],
            "parser_artifact_hash": _PH,
            "detector_artifact_hash": _DH,
            "assembler_schema_hash": _AH,
            "ui_presentation_schema_hash": _UIH,
            "config_hash": _CFH,
            "metrics": {},
            "passed": True,
            "blocking_reasons": ["some reason"],
            "development_only": True,
            "formal_perception_verdict_eligible": False,
        }
        with pytest.raises(ValueError, match="blocking_reasons"):
            load_final_verdict(
                base,
                current_parser_hash=_PH,
                current_detector_hash=_DH,
                current_assembler_hash=_AH,
                current_config_hash=_CFH,
                current_ui_schema_hash=_UIH,
            )

    def test_calibration_verdict_development_only_enforced(self) -> None:
        """PerceptionCalibrationVerdict は常に development_only=True を要求する。"""
        with pytest.raises(FormalVerdictPromotionError):
            PerceptionCalibrationVerdict(
                profile=PerceptionErrorProfile(),
                calibration_session_ids=["c0"],
                development_only=False,
            )
