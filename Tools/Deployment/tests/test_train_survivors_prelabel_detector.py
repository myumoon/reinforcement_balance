"""下書き用検出器の学習 CLI の動作を検証する。

学習と保存は偽物へ差し替え、確認済みフレームの有無と保存先の決まり方だけを確かめる。
事前学習重みのダウンロード（pretrained=True）は通らない。
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

import train_survivors_prelabel_detector
from survivors.annotation_labels import LabelBox, write_label_file
from survivors.prelabel_detector import DEFAULT_WEIGHTS_NAME
from train_survivors_prelabel_detector import main


@pytest.fixture
def fake_training(monkeypatch):
    """CLI が使う train_detector / save_detector を呼び出しを記録するだけの偽物へ差し替える。

    学習に渡されたフレーム数と保存先パスを後から確かめられる。
    """
    calls = {"train": [], "save": []}
    model = object()

    def train_detector(frames, settings, *, device):
        calls["train"].append(len(frames))
        return model

    def save_detector(got_model, path, settings):
        assert got_model is model
        calls["save"].append(path)

    monkeypatch.setattr(train_survivors_prelabel_detector, "train_detector", train_detector)
    monkeypatch.setattr(train_survivors_prelabel_detector, "save_detector", save_detector)
    return calls


def _write_frame(session_dir, frame_id, *, checked):
    """小さな PNG と、gem_blue を1つ含むラベル JSON を書く。

    checked の値で学習対象になるかどうかを切り替える。
    """
    session_dir.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(session_dir / f"{frame_id}.png"), np.zeros((36, 64, 3), dtype=np.uint8))
    write_label_file(
        session_dir / f"{frame_id}.json",
        [LabelBox("gem_blue", 3, 4, 10, 11)],
        image_width=64,
        image_height=36,
        checked=checked,
    )


def test_no_checked_frames_exits_with_error(tmp_path, capsys, fake_training):
    """確認済みフレームが1枚も無ければ学習せず終了コード 1 を返す。

    未確認の下書きだけでは学習に使わないことも確かめる。
    """
    work_root = tmp_path / "work"
    _write_frame(work_root / "s1", "00000001", checked=False)

    result = main(["--work-root", str(work_root)])

    assert result == 1
    assert "error:" in capsys.readouterr().err
    assert fake_training == {"train": [], "save": []}


def test_trains_checked_frames_and_saves_to_default_path(tmp_path, capsys, fake_training):
    """確認済みフレームだけで学習し、既定の `<work-root>/prelabel_detector.pt` へ保存する。

    件数と保存先が標準出力に表示される。
    """
    work_root = tmp_path / "work"
    _write_frame(work_root / "s1", "00000001", checked=True)
    _write_frame(work_root / "s1", "00000002", checked=False)
    _write_frame(work_root / "s2", "00000001", checked=True)

    result = main(["--work-root", str(work_root)])

    assert result == 0
    assert fake_training == {"train": [2], "save": [work_root / DEFAULT_WEIGHTS_NAME]}
    output = capsys.readouterr().out
    assert "学習フレーム数: 2" in output
    assert f"保存先: {work_root / DEFAULT_WEIGHTS_NAME}" in output


def test_output_option_overrides_save_path(tmp_path, fake_training):
    """--output を指定するとその場所へ保存する。

    既定の保存先には書かない。
    """
    work_root = tmp_path / "work"
    _write_frame(work_root / "s1", "00000001", checked=True)
    output = tmp_path / "out" / "custom.pt"

    assert main(["--work-root", str(work_root), "--output", str(output)]) == 0
    assert fake_training["save"] == [output]


@pytest.mark.parametrize("error", [ValueError, RuntimeError, OSError])
def test_training_failure_exits_with_error(tmp_path, monkeypatch, capsys, fake_training, error):
    """学習中の失敗は error: を表示して終了コード 1 を返し、保存しない。

    壊れた画像・GPU メモリ不足などを想定する。
    """
    work_root = tmp_path / "work"
    _write_frame(work_root / "s1", "00000001", checked=True)

    def fail(*args, **kwargs):
        raise error("training failed")

    monkeypatch.setattr(train_survivors_prelabel_detector, "train_detector", fail)

    assert main(["--work-root", str(work_root)]) == 1
    assert "error: training failed" in capsys.readouterr().err
    assert fake_training["save"] == []


def test_invalid_config_exits_with_error(tmp_path, capsys, fake_training):
    """設定ファイルが不正なら共通の load_config で拒否され、終了コード 1 を返す。

    検証ロジックは prelabel_detector.load_config だけが持つ。
    """
    work_root = tmp_path / "work"
    _write_frame(work_root / "s1", "00000001", checked=True)
    config = tmp_path / "bad.yaml"
    config.write_text("schema_version: annotation_prelabel.v1\n", encoding="utf-8")

    assert main(["--work-root", str(work_root), "--config", str(config)]) == 1
    assert "error:" in capsys.readouterr().err
    assert fake_training == {"train": [], "save": []}
