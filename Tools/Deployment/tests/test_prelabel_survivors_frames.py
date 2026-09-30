"""アノテーション下書き CLI の動作を検証する。

合成画像だけを使い、確認状態や既存ラベルの保護も確かめる。
"""

from __future__ import annotations

import copy

import cv2
import numpy as np
import pytest
import yaml

import prelabel_survivors_frames
from prelabel_survivors_frames import DEFAULT_CONFIG_PATH, load_config, main
from survivors.annotation_labels import LabelBox, read_label_file, write_label_file


def _write_config(path, *, labels=None, max_templates=1, fixed_bbox=None):
    """テスト用の小さな下書き設定を書き出す。

    画像が小さくても実フレーム用設定の範囲内で動作を確かめられる。
    """
    data = {
        "schema_version": "annotation_prelabel.v1",
        "fixed_boxes": [{"label": "player_anchor", "bbox": fixed_bbox or [60, 30, 70, 40]}],
        "template_matching": {
            "labels": labels or ["gem_blue", "gem_green"],
            "threshold": 0.99,
            "nms_iou": 0.3,
            "max_templates_per_label": max_templates,
        },
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return data


def _write_image(path, image):
    """合成画像を PNG として保存する。

    OpenCV が実ファイルを読み込む経路を使って確認する。
    """
    assert cv2.imwrite(str(path), image)


def _pattern(seed):
    """相関スコアを判別できる小さな色パターンを作る。

    低いしきい値でも背景と区別できるよう、画素値を変化させる。
    """
    return np.random.default_rng(seed).integers(10, 256, size=(4, 4, 3), dtype=np.uint8)


def test_prelabel_uses_checked_ordered_templates_and_preserves_existing_json(tmp_path, capsys):
    """確認済み見本だけで下書きを作り、手編集ファイルを保護する。

    固定矩形は画像端でクリップされ、矩形はすべて画像内に残る。
    """
    work_root = tmp_path / "work"
    source = work_root / "source"
    target = work_root / "target"
    source.mkdir(parents=True)
    target.mkdir()
    config_path = tmp_path / "prelabel.yaml"
    _write_config(config_path)

    blue_first = _pattern(3)
    blue_later = _pattern(7)
    green_unchecked = _pattern(11)
    source_image = np.zeros((36, 64, 3), dtype=np.uint8)
    source_image[4:8, 3:7] = blue_first
    _write_image(source / "00000001.png", source_image)
    write_label_file(
        source / "00000001.json",
        [LabelBox("gem_blue", 3, 4, 7, 8, 0.9)],
        image_width=64,
        image_height=36,
        checked=True,
    )

    later_image = np.zeros_like(source_image)
    later_image[4:8, 3:7] = blue_later
    _write_image(source / "00000002.png", later_image)
    write_label_file(
        source / "00000002.json",
        [LabelBox("gem_blue", 3, 4, 7, 8, 0.9)],
        image_width=64,
        image_height=36,
        checked=True,
    )

    unchecked_image = np.zeros_like(source_image)
    unchecked_image[4:8, 3:7] = green_unchecked
    _write_image(source / "00000003.png", unchecked_image)
    write_label_file(
        source / "00000003.json",
        [LabelBox("gem_green", 3, 4, 7, 8, 0.9)],
        image_width=64,
        image_height=36,
        checked=False,
    )

    target_image = np.zeros_like(source_image)
    target_image[12:16, 20:24] = blue_first
    target_image[20:24, 40:44] = green_unchecked
    _write_image(target / "00000010.png", target_image)
    _write_image(target / "00000011.png", np.zeros_like(target_image))
    manual_json = target / "00000011.json"
    write_label_file(
        manual_json,
        [LabelBox("hud_hp", 1, 1, 5, 3)],
        image_width=64,
        image_height=36,
        checked=False,
    )
    manual_bytes = manual_json.read_bytes()
    _write_image(target / "README.png", target_image)
    (target / "README.json").write_text("not a frame", encoding="utf-8")
    write_label_file(target / "00000012.json", [], image_width=64, image_height=36, checked=False)
    unpaired_bytes = (target / "00000012.json").read_bytes()

    result = main(
        ["--work-root", str(work_root), "--session-id", "target", "--config", str(config_path)]
    )

    assert result == 0
    boxes, checked = read_label_file(target / "00000010.json")
    assert checked is False
    assert [(box.label, box.left, box.top, box.right, box.bottom) for box in boxes] == [
        ("player_anchor", 60.0, 30.0, 64.0, 36.0),
        ("gem_blue", 20.0, 12.0, 24.0, 16.0),
    ]
    assert manual_json.read_bytes() == manual_bytes
    assert (target / "README.json").read_text(encoding="utf-8") == "not a frame"
    assert (target / "00000012.json").read_bytes() == unpaired_bytes
    assert "下書き作成数: 1" in capsys.readouterr().out
    assert all(0 <= box.left < box.right <= 64 and 0 <= box.top < box.bottom <= 36 for box in boxes)


def test_prelabel_skips_json_created_after_frame_listing(tmp_path, monkeypatch, capsys):
    """一覧取得後に現れた人手 JSON を置換せず、スキップ数に含める。

    画像読み込み時に同じセッションの別 JSON を作り、一覧と書き込みの競合を再現する。
    """
    work_root = tmp_path / "work"
    session_dir = work_root / "s1"
    session_dir.mkdir(parents=True)
    config_path = tmp_path / "prelabel.yaml"
    _write_config(config_path)
    image = np.zeros((36, 64, 3), dtype=np.uint8)
    first_png = session_dir / "00000000.png"
    raced_json = session_dir / "00000001.json"
    _write_image(first_png, image)
    _write_image(session_dir / "00000001.png", image)
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


def test_refresh_unchecked_overwrites_stale_drafts_but_protects_checked(tmp_path, capsys):
    """--refresh-unchecked で未チェックの下書きだけ再生成し、チェック済みは保護する。

    見本が増えた後に同じセッションを再実行して古い下書きを更新する運用を想定する。
    """
    work_root = tmp_path / "work"
    session_dir = work_root / "s1"
    session_dir.mkdir(parents=True)
    config_path = tmp_path / "prelabel.yaml"
    _write_config(config_path)

    blue = _pattern(3)
    image = np.zeros((36, 64, 3), dtype=np.uint8)

    checked_image = image.copy()
    checked_image[4:8, 3:7] = blue
    _write_image(session_dir / "00000001.png", checked_image)
    checked_json = session_dir / "00000001.json"
    write_label_file(
        checked_json,
        [LabelBox("gem_blue", 3, 4, 7, 8, 0.9)],
        image_width=64,
        image_height=36,
        checked=True,
    )
    checked_bytes = checked_json.read_bytes()

    stale_image = image.copy()
    stale_image[12:16, 20:24] = blue
    _write_image(session_dir / "00000002.png", stale_image)
    write_label_file(
        session_dir / "00000002.json", [], image_width=64, image_height=36, checked=False
    )

    result = main(
        [
            "--work-root", str(work_root), "--session-id", "s1",
            "--config", str(config_path), "--refresh-unchecked",
        ]
    )

    assert result == 0
    assert checked_json.read_bytes() == checked_bytes

    boxes, checked = read_label_file(session_dir / "00000002.json")
    assert checked is False
    assert ("gem_blue", 20.0, 12.0, 24.0, 16.0) in [
        (box.label, box.left, box.top, box.right, box.bottom) for box in boxes
    ]
    output = capsys.readouterr().out
    assert "下書き作成数: 1" in output
    assert "既存 JSON スキップ数: 1" in output


@pytest.mark.parametrize(
    "mutation",
    [
        lambda data: data.update(extra=True),
        lambda data: data["fixed_boxes"][0].update(label="unknown_label"),
        lambda data: data["fixed_boxes"][0].update(bbox=[0, 0, 2000, 40]),
        lambda data: data["template_matching"].update(threshold=0),
        lambda data: data["template_matching"].update(nms_iou=1.1),
    ],
)
def test_load_config_rejects_unknown_and_invalid_values(tmp_path, mutation):
    """不明な項目や範囲外設定を起動前に拒否する。

    誤設定のまま部分的な下書きを作らない。
    """
    config_path = tmp_path / "invalid.yaml"
    data = _write_config(config_path)
    invalid_data = copy.deepcopy(data)
    mutation(invalid_data)
    config_path.write_text(yaml.safe_dump(invalid_data), encoding="utf-8")

    with pytest.raises(ValueError):
        load_config(config_path)


def test_bundled_config_is_valid_and_uses_expected_defaults():
    """同梱設定のラベルと調整値が読み込めることを確認する。

    実行時に使う初期矩形や照合対象をテストで固定する。
    """
    config = load_config(DEFAULT_CONFIG_PATH)

    assert [box.label for box in config["fixed_boxes"]] == ["player_anchor", "hud_hp", "hud_xp"]
    assert config["labels"] == ["gem_blue", "gem_green", "gem_red", "pickup_heal", "pickup_special", "enemy_normal"]
    assert config["threshold"] == 0.85
    assert config["nms_iou"] == 0.3
    assert config["max_templates_per_label"] == 5
