"""HUD calibration の split・決定性・11 gate 契約テスト。

合成注釈だけで用途分離、gate 境界、fit/report hash の再現性を確認します。
"""
from __future__ import annotations
from dataclasses import asdict, replace
import pytest
from survivors.vision.hud_calibration import (
    CalibrationConfig, FormalDatasetRequired, GateResult, HudCalibrationAnnotation,
    SplitViolationError, fit, reject_formal_if_synthetic, validate,
    validate_split_eligibility,
)

def _sample(split: str = "model_validation", sample_id: str = "frame-1") -> HudCalibrationAnnotation:
    """全 gate を通る合成注釈を返す。

正解値と parser 出力が一致する最小レコードを組み立てます。
    """
    return HudCalibrationAnnotation(
        sample_id, split, 123.0, 123.0, 4, 4, 0.75, 0.75, 0.25, 0.25,
        ("whip", None), ("whip", None), "whip", "whip", "gameplay", "gameplay",
        1.0, (10.0, 20.0, 30.0, 40.0), (10.0, 20.0, 30.0, 40.0),
    )

def test_split_and_formal_guards() -> None:
    """model split だけを許可し、synthetic formal 実行を拒否する。

予約 split と未知 split の全兄弟経路を同じ fail-closed 検査へ通します。
    """
    for split in ("model_train", "model_validation"):
        validate_split_eligibility(split)
    for split in ("error_calibration", "final_e2e_test", "unknown"):
        with pytest.raises(SplitViolationError):
            validate_split_eligibility(split)
    with pytest.raises(FormalDatasetRequired):
        reject_formal_if_synthetic(formal_mode=True, formal_dataset_eligible=False)
    reject_formal_if_synthetic(formal_mode=False, formal_dataset_eligible=False)
    reject_formal_if_synthetic(formal_mode=True, formal_dataset_eligible=True)

def test_fit_is_train_only_non_empty_and_deterministic() -> None:
    """fit が train 注釈だけを受け入れ、決定的な結果を返す。

空入力と validation 混入を拒否し、同じ ROI の hash が不変であることを確認します。
    """
    train = [_sample("model_train")]
    assert fit(train) == fit(train)
    assert fit(train).fit_canonical_hash == fit(train).fit_canonical_hash
    with pytest.raises(SplitViolationError):
        fit([_sample("model_validation")])
    with pytest.raises(ValueError):
        fit([])

def test_fit_requires_proven_train_split_and_sweeps_reserved_splits() -> None:
    """fit の全 split 指定経路を fail-closed で検証する。

record または明示引数による train 証明だけを受け入れ、欠測・不一致・予約 split を拒否します。
    """
    without_split = {"sample_id": "frame-1", "expected_roi": [10.0, 20.0, 30.0, 40.0]}
    assert fit([without_split], split="model_train").annotation_count == 1
    assert fit([_sample("model_train")], split="model_train").annotation_count == 1
    with pytest.raises(SplitViolationError):
        fit([without_split])
    with pytest.raises(SplitViolationError):
        fit([_sample("model_train")], split="model_validation")
    for reserved in ("error_calibration", "final_e2e_test"):
        with pytest.raises(SplitViolationError):
            fit([_sample(reserved)])
        with pytest.raises(SplitViolationError):
            fit([without_split], split=reserved)

def test_validate_gates_hash_split_and_boundaries() -> None:
    """validation の11 gate、hash、split、境界比較をまとめて検証する。

ROI 陰性例を含む report と floor/ceiling の内包境界を確認します。
    """
    records = [_sample(), replace(_sample(sample_id="frame-2"), expected_roi=None, predicted_roi=None)]
    report = validate(records, CalibrationConfig())
    assert set(report.gates_by_name) == {
        "timer_exact", "level", "hp_mae", "xp_mae", "item", "choice",
        "screen_state_f1", "latency", "roi_center_error", "roi_inside_rate",
        "roi_false_positive",
    }
    assert report.report_canonical_hash == validate(records).report_canonical_hash
    assert report.all_passed
    for bad in (_sample("model_train"), _sample("final_e2e_test")):
        with pytest.raises(SplitViolationError):
            validate([bad])
    with pytest.raises(ValueError):
        validate([])
    assert GateResult.floor_gate("floor", value=0.8, floor=0.8).passed
    assert not GateResult.floor_gate("floor", value=0.799, floor=0.8).passed
    assert GateResult.ceiling_gate("ceiling", value=0.2, ceiling=0.2).passed
    assert not GateResult.ceiling_gate("ceiling", value=0.201, ceiling=0.2).passed

def test_invalid_annotation_types_rejected_before_gate_computation() -> None:
    """型・値域違反の annotation を gate 計算前に拒否する（P1）。

文字列 timer、範囲外 ratio、非整数 level で ValueError を送出することを確認します。
    """
    from dataclasses import asdict
    for bad_field, bad_value in [
        ("expected_timer_seconds", "hello"),
        ("expected_hp_ratio", 2.0),
        ("expected_xp_ratio", -1.0),
        ("expected_level", 0),
        ("expected_screen_state", 42),
        ("predicted_items", "not-a-list"),
    ]:
        rec = asdict(_sample())
        rec[bad_field] = bad_value
        with pytest.raises(ValueError, match=bad_field):
            validate([rec], split="model_validation")


def test_roi_false_positive_fails_when_no_negative_examples() -> None:
    """陰性 ROI 例がない dataset では roi_false_positive gate を合格させない（P2a）。

陰性例なしで false_positive_rate=1.0 となり ceiling gate が不合格になることを確認します。
    """
    # 正例のみ（expected_roi あり）
    positive_only = _sample()
    report = validate([positive_only])
    gate = report.gates_by_name["roi_false_positive"]
    assert not gate.passed, "roi_false_positive must fail when dataset has no negative ROI examples"
    assert gate.value == 1.0


def test_fit_is_deterministic_regardless_of_input_order() -> None:
    """fit の ROI 平均が入力順に依存しない（P2c）。

3 件の異なる ROI を逆順に並べても fit_canonical_hash が一致することを確認します。
    """
    from dataclasses import replace
    a = replace(_sample("model_train", "s1"), expected_roi=(10.0, 20.0, 30.0, 40.0))
    b = replace(_sample("model_train", "s2"), expected_roi=(11.0, 21.0, 31.0, 41.0))
    c = replace(_sample("model_train", "s3"), expected_roi=(12.0, 22.0, 32.0, 42.0))
    assert fit([a, b, c]).fit_canonical_hash == fit([c, b, a]).fit_canonical_hash


def test_validate_requires_proven_validation_split_and_sweeps_reserved_splits() -> None:
    """validate の全 split 指定経路を fail-closed で検証する。

record または明示引数による validation 証明だけを受け入れ、欠測・不一致・予約 split を拒否します。
    """
    without_split = asdict(_sample())
    without_split.pop("split")
    assert validate([without_split], split="model_validation").sample_count == 1
    assert validate([_sample()], split="model_validation").sample_count == 1
    with pytest.raises(SplitViolationError):
        validate([without_split])
    with pytest.raises(SplitViolationError):
        validate([_sample()], split="model_train")
    for reserved in ("error_calibration", "final_e2e_test"):
        with pytest.raises(SplitViolationError):
            validate([_sample(reserved)])
        with pytest.raises(SplitViolationError):
            validate([without_split], split=reserved)
