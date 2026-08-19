"""HUD parser development package の永続化・復元契約テスト。

atomic publish、hash 検証、用途混在拒否、HudParser identity を合成入力で確認します。
"""
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
from survivors.vision.hud_calibration import (
    FormalPackageMixingError, HudCalibrationAnnotation, PackageHashMismatch,
    PackageLoader, PackageWriter, ParserPackageMeta, fit, validate,
)
from survivors.vision.hud_parser import HudParser, HudStateV1

def _annotation(split: str) -> HudCalibrationAnnotation:
    """package test 用の完全一致注釈を返す（正例 ROI 付き）。

train と validation の違いを split だけに限定します。
    """
    return HudCalibrationAnnotation(
        f"{split}-1", split, 1.0, 1.0, 1, 1, 1.0, 1.0, 0.0, 0.0,
        (None,), (None,), "whip", "whip", "gameplay", "gameplay", 1.0,
        (0.0, 0.0, 10.0, 10.0), (0.0, 0.0, 10.0, 10.0),
    )

def _annotation_negative(split: str) -> HudCalibrationAnnotation:
    """ROI 陰性例（expected=None, predicted=None）の注釈を返す。

roi_false_positive gate を合格させるためには陰性例が最低1件必要です。
    """
    return HudCalibrationAnnotation(
        f"{split}-neg", split, 1.0, 1.0, 1, 1, 1.0, 1.0, 0.0, 0.0,
        (None,), (None,), "whip", "whip", "gameplay", "gameplay", 1.0,
        None, None,
    )

def _publish(path: Path) -> ParserPackageMeta:
    """決定的な development package を発行する。

正例 + 陰性例（ROI なし）を含む最小 validation セットで roi_false_positive gate を合格させます。
    """
    return PackageWriter.publish_development(
        path,
        fit([_annotation("model_train")]),
        validate([_annotation("model_validation"), _annotation_negative("model_validation")]),
    )

def test_package_publish_load_tamper_and_formal_guards(tmp_path: Path) -> None:
    """package の用途属性、atomic publish、hash、formal gate を検証する。

正常復元後に payload を改ざんし、全 loader/publisher 境界が閉じることを確認します。
    """
    path = tmp_path / "hud-parser-package.json"
    meta = _publish(path)
    assert meta.development_only is True and meta.formal_parser_eligible is False
    assert not list(tmp_path.glob("*.tmp"))
    assert PackageLoader.load_development(path).package_hash == meta.package_hash
    with pytest.raises(FormalPackageMixingError):
        PackageLoader.load_formal(path)
    with pytest.raises(NotImplementedError):
        PackageWriter.publish_formal(tmp_path / "formal.json")
    wire = json.loads(path.read_text(encoding="utf-8"))
    wire["metadata"]["package_hash"] = "0" * 64
    path.write_text(json.dumps(wire), encoding="utf-8")
    with pytest.raises(PackageHashMismatch):
        PackageLoader.load_development(path)

def test_restored_hash_drives_deterministic_hud_parser(tmp_path: Path, blank_frame: np.ndarray) -> None:
    """復元 package hash を HudParser identity に使用する。

同じフレームの HudStateV1 と card/button semantics が2回とも一致することを確認します。
    """
    path = tmp_path / "hud-parser-package.json"
    _publish(path)
    parser = HudParser(parser_artifact_hash=PackageLoader.load_development(path).package_hash)
    kwargs = {"session_id": "session", "frame_index": 0, "captured_monotonic_ns": 1}
    first, second = parser.parse(blank_frame, **kwargs), parser.parse(blank_frame, **kwargs)
    assert isinstance(first, HudStateV1) and first == second
    assert first.cards == () and first.buttons == ()
    assert first.candidate_set_hash and first.capability_reason

@pytest.mark.parametrize("script", ["calibrate_survivors_hud_parser.py", "eval_survivors_hud_parser.py"])
def test_cli_help_exits_zero(script: str) -> None:
    """両 CLI が argparse help を終了コード0で表示する。

外部 dataset なしで usage を確認できることを subprocess で検証します。
    """
    root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [sys.executable, str(root / script), "--help"], capture_output=True,
        text=True, check=False,
    )
    assert completed.returncode == 0 and "usage:" in completed.stdout.lower()

def test_calibration_cli_rejects_missing_split_in_each_input(tmp_path: Path) -> None:
    """calibration CLI が train/validation の split 欠測を拒否する。

2つの入力役割を入れ替えて実行し、どちらも package 発行前に終了コード1となることを確認します。
    """
    root = Path(__file__).resolve().parents[2]
    script = root / "calibrate_survivors_hud_parser.py"
    train = {"sample_id": "train-1", "split": "model_train", "expected_roi": [0, 0, 10, 10]}
    validation = {"sample_id": "validation-1", "split": "model_validation"}
    for missing_role in ("train", "validation"):
        train_path, validation_path = tmp_path / "train.json", tmp_path / "validation.json"
        train_wire, validation_wire = dict(train), dict(validation)
        (train_wire if missing_role == "train" else validation_wire).pop("split")
        train_path.write_text(json.dumps([train_wire]), encoding="utf-8")
        validation_path.write_text(json.dumps([validation_wire]), encoding="utf-8")
        completed = subprocess.run(
            [
                sys.executable, str(script), "--train-annotations", str(train_path),
                "--validation-annotations", str(validation_path), "--output-package",
                str(tmp_path / f"{missing_role}.json"),
            ],
            capture_output=True, text=True, check=False,
        )
        assert completed.returncode == 1
        assert "split" in completed.stderr

def test_calibration_cli_exits_2_on_gate_failure(tmp_path: Path) -> None:
    """calibrate CLI は gate 失敗時に exit code 2 を返しパッケージを発行しない。

timer_exact gate が落ちる故意の不一致 (expected=1.0, predicted=999.0) を与え、
exit code 2 と package ファイルの未生成を確認します。
    """
    root = Path(__file__).resolve().parents[2]
    script = root / "calibrate_survivors_hud_parser.py"
    train_path = tmp_path / "train.json"
    validation_path = tmp_path / "validation.json"
    pkg_path = tmp_path / "pkg.json"
    train_path.write_text(
        json.dumps([
            {"sample_id": "t1", "split": "model_train", "expected_timer_seconds": 1.0,
             "predicted_timer_seconds": 1.0, "expected_level": 1, "predicted_level": 1,
             "expected_hp_ratio": 1.0, "predicted_hp_ratio": 1.0,
             "expected_xp_ratio": 0.0, "predicted_xp_ratio": 0.0,
             "expected_roi": [0.0, 0.0, 10.0, 10.0], "predicted_roi": [0.0, 0.0, 10.0, 10.0]},
        ]),
        encoding="utf-8",
    )
    # timer 大幅不一致 → timer_exact gate 失敗
    validation_path.write_text(
        json.dumps([
            {"sample_id": "v1", "split": "model_validation", "expected_timer_seconds": 1.0,
             "predicted_timer_seconds": 999.0, "expected_level": 1, "predicted_level": 1,
             "expected_hp_ratio": 1.0, "predicted_hp_ratio": 1.0,
             "expected_xp_ratio": 0.0, "predicted_xp_ratio": 0.0},
            # 陰性例（roi_false_positive gate を合格させるため）
            {"sample_id": "v2", "split": "model_validation", "expected_timer_seconds": 1.0,
             "predicted_timer_seconds": 999.0, "expected_level": 1, "predicted_level": 1,
             "expected_hp_ratio": 1.0, "predicted_hp_ratio": 1.0,
             "expected_xp_ratio": 0.0, "predicted_xp_ratio": 0.0,
             "expected_roi": None, "predicted_roi": None},
        ]),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable, str(script),
            "--train-annotations", str(train_path),
            "--validation-annotations", str(validation_path),
            "--output-package", str(pkg_path),
        ],
        capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 2, f"expected exit 2, got {completed.returncode}: {completed.stderr}"
    assert not pkg_path.exists(), "package must not be published when gates fail"


def test_publish_development_rejects_failing_report(tmp_path: Path) -> None:
    """gate 不合格の ValidationReport を publish_development へ直接渡すと ValueError を送出する（P2 publish guard）。

CLI を経由せず API を直接呼ぶ経路でも不合格 package が発行されないことを確認します。
    """
    # timer gate を落とす：expected=1.0, predicted=0.0
    failing = {"sample_id": "v1", "split": "model_validation", "expected_timer_seconds": 1.0, "predicted_timer_seconds": 0.0}
    # roi_false_positive gate のために陰性例を含める（expected_roi=None）
    neg = {"sample_id": "v2", "split": "model_validation", "expected_roi": None, "predicted_roi": None}
    report = validate([failing, neg])
    assert not report.all_passed, "テスト前提: gate が不合格であること"
    with pytest.raises(ValueError, match="all gates"):
        PackageWriter.publish_development(tmp_path / "pkg.json", fit([_annotation("model_train")]), report)
    assert not (tmp_path / "pkg.json").exists(), "不合格 package が発行されていてはいけない"


def test_eval_cli_rejects_missing_split(tmp_path: Path) -> None:
    """eval CLI が split 欠測 annotation を拒否する。

用途が証明されていない JSON を validation と推定せず、終了コード1で停止することを確認します。
    """
    root = Path(__file__).resolve().parents[2]
    annotation_path = tmp_path / "annotations.json"
    annotation_path.write_text(json.dumps([{"sample_id": "validation-1"}]), encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(root / "eval_survivors_hud_parser.py"), "--annotations", str(annotation_path)],
        capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 1
    assert "split" in completed.stderr
