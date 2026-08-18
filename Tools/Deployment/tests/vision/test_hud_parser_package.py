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
    """package test 用の完全一致注釈を返す。

train と validation の違いを split だけに限定します。
    """
    return HudCalibrationAnnotation(
        f"{split}-1", split, 1.0, 1.0, 1, 1, 1.0, 1.0, 0.0, 0.0,
        (None,), (None,), "whip", "whip", "gameplay", "gameplay", 1.0,
        (0.0, 0.0, 10.0, 10.0), (0.0, 0.0, 10.0, 10.0),
    )

def _publish(path: Path) -> ParserPackageMeta:
    """決定的な development package を発行する。

最小の fit と validation 成果物を全 package test で共有します。
    """
    return PackageWriter.publish_development(
        path, fit([_annotation("model_train")]), validate([_annotation("model_validation")])
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
