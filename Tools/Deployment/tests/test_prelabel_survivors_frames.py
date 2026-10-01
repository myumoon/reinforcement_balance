"""アノテーション下書き CLI の動作を検証する。

合成画像と差し替えた検出器だけを使い、確認状態や既存ラベルの保護も確かめる。
ネットワークから事前学習重みをダウンロードする経路は通らない。
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
import yaml

import prelabel_survivors_frames
from prelabel_survivors_frames import main
from survivors.annotation_labels import LabelBox, read_label_file, write_label_file
from survivors.prelabel_detector import DEFAULT_CONFIG_PATH, DEFAULT_WEIGHTS_NAME, build_model, load_config, save_detector

_DETECTED = LabelBox("gem_blue", 20, 12, 24, 16, 0.8)


def _write_config(path):
    """同梱 v2 設定の固定矩形だけを小さな画像向けに置き換えて書き出す。

    検出器設定は同梱のまま使い、固定矩形が画像端でクリップされる様子を確かめられるようにする。
    """
    data = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    data["fixed_boxes"] = [{"label": "player_anchor", "bbox": [60, 30, 70, 40]}]
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _write_image(path, image=None):
    """合成画像を PNG として保存する。

    OpenCV が実ファイルを読み込む経路を使って確認する。
    """
    assert cv2.imwrite(str(path), np.zeros((36, 64, 3), dtype=np.uint8) if image is None else image)


def _setup(tmp_path, session="s1", *, weights=True):
    """work-root・セッション・設定・（必要なら）ダミー重みファイルを用意する。

    重みファイルは存在確認だけに使われ、中身は差し替えた load_detector が読まない。
    """
    work_root = tmp_path / "work"
    session_dir = work_root / session
    session_dir.mkdir(parents=True)
    config_path = tmp_path / "prelabel.yaml"
    _write_config(config_path)
    if weights:
        (work_root / DEFAULT_WEIGHTS_NAME).write_bytes(b"dummy")
    return work_root, session_dir, config_path


@pytest.fixture
def fake_detector(monkeypatch):
    """CLI が使う load_detector / predict_boxes を固定の検出結果を返す偽物へ差し替える。

    呼び出し記録を返すので、読み込んだ重みのパスや推論回数を確かめられる。
    """
    calls = {"load": [], "predict": 0}
    model = object()

    def load_detector(path, settings, *, device):
        calls["load"].append(path)
        return model

    def predict_boxes(got_model, image, settings, *, device):
        assert got_model is model
        calls["predict"] += 1
        return [_DETECTED]

    monkeypatch.setattr(prelabel_survivors_frames, "load_detector", load_detector)
    monkeypatch.setattr(prelabel_survivors_frames, "predict_boxes", predict_boxes)
    return calls


def _as_tuples(boxes):
    """LabelBox 列を比較しやすい (label, left, top, right, bottom) の列へ変換する。

    スコアは比較対象から外す。
    """
    return [(box.label, box.left, box.top, box.right, box.bottom) for box in boxes]


def test_prelabel_writes_fixed_and_detected_boxes_and_preserves_existing_json(tmp_path, capsys, fake_detector):
    """新規下書きに固定矩形と検出矩形を書き、既存 JSON は checked の値によらず保護する。

    固定矩形は画像端でクリップされ、フレームでないファイルや対の無い JSON にも触れない。
    """
    work_root, target, config_path = _setup(tmp_path, "target")
    _write_image(target / "00000010.png")
    for frame_id, checked in (("00000011", False), ("00000013", True)):
        _write_image(target / f"{frame_id}.png")
        write_label_file(
            target / f"{frame_id}.json", [LabelBox("hud_hp", 1, 1, 5, 3)], image_width=64, image_height=36, checked=checked
        )
    manual_bytes = {name: (target / name).read_bytes() for name in ("00000011.json", "00000013.json")}
    _write_image(target / "README.png")
    (target / "README.json").write_text("not a frame", encoding="utf-8")
    write_label_file(target / "00000012.json", [], image_width=64, image_height=36, checked=False)
    unpaired_bytes = (target / "00000012.json").read_bytes()

    result = main(["--work-root", str(work_root), "--session-id", "target", "--config", str(config_path)])

    assert result == 0
    boxes, checked = read_label_file(target / "00000010.json")
    assert checked is False
    assert _as_tuples(boxes) == [
        ("player_anchor", 60.0, 30.0, 64.0, 36.0),
        ("gem_blue", 20.0, 12.0, 24.0, 16.0),
    ]
    assert all(box.label != "labeled_region" for box in boxes)
    assert {name: (target / name).read_bytes() for name in manual_bytes} == manual_bytes
    assert (target / "README.json").read_text(encoding="utf-8") == "not a frame"
    assert (target / "00000012.json").read_bytes() == unpaired_bytes
    assert fake_detector["load"] == [work_root / DEFAULT_WEIGHTS_NAME]
    assert fake_detector["predict"] == 1
    output = capsys.readouterr().out
    assert f"検出器: {work_root / DEFAULT_WEIGHTS_NAME}" in output
    assert "下書き作成数: 1" in output
    assert "既存 JSON スキップ数: 2" in output


def test_prelabel_skips_json_created_after_frame_listing(tmp_path, monkeypatch, capsys, fake_detector):
    """一覧取得後に現れた人手 JSON を置換せず、スキップ数に含める。

    画像読み込み時に同じセッションの別 JSON を作り、一覧と書き込みの競合を再現する。
    """
    work_root, session_dir, config_path = _setup(tmp_path)
    first_png = session_dir / "00000000.png"
    raced_json = session_dir / "00000001.json"
    _write_image(first_png)
    _write_image(session_dir / "00000001.png")
    human_bytes = (
        b'{"shapes":[],"imagePath":"00000001.png","imageData":null,'
        b'"checked":true,"HUMAN":1}\n'
    )
    read_image = prelabel_survivors_frames._read_image

    def create_human_label_after_listing(path):
        if path == first_png:
            raced_json.write_bytes(human_bytes)
        return read_image(path)

    monkeypatch.setattr(prelabel_survivors_frames, "_read_image", create_human_label_after_listing)
    result = main(["--work-root", str(work_root), "--session-id", "s1", "--config", str(config_path)])

    assert result == 0
    assert raced_json.read_bytes() == human_bytes
    output = capsys.readouterr().out
    assert "下書き作成数: 1" in output
    assert "既存 JSON スキップ数: 1" in output


def test_refresh_unchecked_overwrites_stale_drafts_but_protects_checked(tmp_path, capsys, fake_detector):
    """--refresh-unchecked で未チェックの下書きだけ再生成し、チェック済みは byte 一致で保護する。

    検出器を再学習した後に同じセッションを再実行して古い下書きを更新する運用を想定する。
    """
    work_root, session_dir, config_path = _setup(tmp_path)
    _write_image(session_dir / "00000001.png")
    checked_json = session_dir / "00000001.json"
    write_label_file(
        checked_json, [LabelBox("gem_blue", 3, 4, 7, 8, 0.9)], image_width=64, image_height=36, checked=True
    )
    checked_bytes = checked_json.read_bytes()
    _write_image(session_dir / "00000002.png")
    write_label_file(session_dir / "00000002.json", [], image_width=64, image_height=36, checked=False)

    result = main(
        ["--work-root", str(work_root), "--session-id", "s1", "--config", str(config_path), "--refresh-unchecked"]
    )

    assert result == 0
    assert checked_json.read_bytes() == checked_bytes
    boxes, checked = read_label_file(session_dir / "00000002.json")
    assert checked is False
    assert ("gem_blue", 20.0, 12.0, 24.0, 16.0) in _as_tuples(boxes)
    output = capsys.readouterr().out
    assert "下書き作成数: 1" in output
    assert "既存 JSON スキップ数: 1" in output


def test_missing_weights_warns_and_drafts_fixed_boxes_only(tmp_path, capsys, fake_detector):
    """重みファイルが無ければ警告を出し、検出器を使わず固定矩形だけで下書きを作る。

    学習前の初回周回でも固定矩形の下書きは作れることを確かめる。
    """
    work_root, session_dir, config_path = _setup(tmp_path, weights=False)
    _write_image(session_dir / "00000001.png")

    result = main(["--work-root", str(work_root), "--session-id", "s1", "--config", str(config_path)])

    assert result == 0
    boxes, checked = read_label_file(session_dir / "00000001.json")
    assert checked is False
    assert _as_tuples(boxes) == [("player_anchor", 60.0, 30.0, 64.0, 36.0)]
    assert fake_detector == {"load": [], "predict": 0}
    captured = capsys.readouterr()
    assert "warning:" in captured.err and DEFAULT_WEIGHTS_NAME in captured.err
    assert "検出器: なし（固定矩形のみ）" in captured.out
    assert "下書き作成数: 1" in captured.out


def test_detector_option_overrides_default_weights_path(tmp_path, capsys, fake_detector):
    """--detector で指定した重みを読み、既定パスの重みは使わない。

    work-root 外に置いた重みでも下書きを作れることを確かめる。
    """
    work_root, session_dir, config_path = _setup(tmp_path, weights=False)
    weights = tmp_path / "custom.pt"
    weights.write_bytes(b"dummy")
    _write_image(session_dir / "00000001.png")

    result = main(
        ["--work-root", str(work_root), "--session-id", "s1", "--config", str(config_path), "--detector", str(weights)]
    )

    assert result == 0
    assert fake_detector["load"] == [weights]
    assert f"検出器: {weights}" in capsys.readouterr().out


@pytest.mark.parametrize(("target", "error"), [("predict_boxes", ValueError), ("load_detector", RuntimeError)])
def test_detector_failure_exits_with_error(tmp_path, monkeypatch, capsys, fake_detector, target, error):
    """重みの読み込みや推論が失敗したら error: を表示して終了コード 1 を返す。

    壊れた重み（torch 由来の RuntimeError）や設定不一致（ValueError）で下書きを作らない。
    """
    work_root, session_dir, config_path = _setup(tmp_path)
    _write_image(session_dir / "00000001.png")

    def fail(*args, **kwargs):
        raise error("broken detector")

    monkeypatch.setattr(prelabel_survivors_frames, target, fail)
    result = main(["--work-root", str(work_root), "--session-id", "s1", "--config", str(config_path)])

    assert result == 1
    assert "error: broken detector" in capsys.readouterr().err
    assert not (session_dir / "00000001.json").exists()


def test_prelabel_runs_with_real_untrained_detector(tmp_path, capsys):
    """事前学習なしで作った実モデルを保存し、差し替え無しで CLI が最後まで動くことを確かめる。

    ダウンロードせずに重みの保存・安全な読み込み・推論がつながることだけを見る（精度は見ない）。
    """
    work_root, session_dir, config_path = _setup(tmp_path, weights=False)
    settings = load_config(config_path).detector
    save_detector(
        build_model(len(settings.labels), settings, pretrained=False), work_root / DEFAULT_WEIGHTS_NAME, settings
    )
    _write_image(session_dir / "00000001.png")

    result = main(["--work-root", str(work_root), "--session-id", "s1", "--config", str(config_path)])

    assert result == 0
    boxes, checked = read_label_file(session_dir / "00000001.json")
    assert checked is False
    assert _as_tuples(boxes)[0] == ("player_anchor", 60.0, 30.0, 64.0, 36.0)
    assert all(box.label in settings.labels for box in boxes[1:])
    assert "下書き作成数: 1" in capsys.readouterr().out
