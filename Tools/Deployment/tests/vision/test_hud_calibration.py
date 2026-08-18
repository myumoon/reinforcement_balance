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
