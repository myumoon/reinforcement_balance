"""確認済み Survivors ラベルの COCO 出力 CLI を検証する。

小さな合成画像とラベルだけを使い、決定性と WorldDataset 互換性を確認する。
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from export_survivors_annotations_coco import main
from survivors.vision.world_dataset import WorldDataset, load_class_map


def _write_png(path: Path, *, width: int = 10, height: int = 8) -> None:
    """指定寸法の小さな PNG を作る。

    実キャプチャを使わず、COCO の幅・高さを検証できる画像を用意する。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), np.zeros((height, width, 4), dtype=np.uint8))


def _shape(label: str, points: list[list[float]]) -> dict:
    """契約どおりの rectangle shape を返す。

    GUI が保存した画像外座標も作り、exporter の clipping を検証する。
    """
    return {
        "label": label,
        "score": None,
        "points": points,
        "group_id": None,
        "description": "",
        "difficult": False,
        "shape_type": "rectangle",
        "flags": {},
        "attributes": {},
    }


def _write_label(path: Path, shapes: list[dict], *, checked: bool) -> None:
    """指定 shape を持つ X-AnyLabeling ファイルを作る。

    exporter が checked を厳密に読み分けられるように契約キーを揃える。
    """
    path.write_text(
        json.dumps(
            {
                "shapes": shapes,
                "imagePath": path.with_suffix(".png").name,
                "imageData": None,
                "checked": checked,
            }
        ),
        encoding="utf-8",
    )


def test_export_checked_labels_clips_sorts_and_keeps_negative_images(tmp_path: Path, capsys) -> None:
    """checked ラベルを決定的な world/ui COCO へ分割出力する。

    未確認を除外し、画像範囲外の矩形を切り詰め、矩形ゼロの画像も残す。
    """
    work_root = tmp_path / "work"
    session_a = work_root / "session-a"
    session_b = work_root / "session-b"
    _write_png(session_a / "00000003.png")
    _write_label(
        session_a / "00000003.json",
        [
            _shape("player_anchor", [[-2, 1], [5, 1], [5, 12], [-2, 12]]),
            _shape("hud_hp", [[8, 1], [12, 1], [12, 4], [8, 4]]),
            _shape("button", [[12, 1], [13, 1], [13, 4], [12, 4]]),
        ],
        checked=True,
    )
    _write_png(session_a / "00000001.png")
    _write_label(session_a / "00000001.json", [_shape("gem_blue", [[1, 1], [3, 3]])], checked=False)
    _write_png(session_b / "00000002.png")
    _write_label(session_b / "00000002.json", [], checked=True)
    (session_a / "README.json").write_text("{}", encoding="utf-8")
    source_label = session_a / "00000003.json"
    original_label = source_label.read_bytes()
    output_dir = tmp_path / "export"
    class_map_path = Path(__file__).resolve().parents[1] / "configs" / "world_class_map_v2.yaml"

    result = main(
        [
            "--work-root",
            str(work_root),
            "--output-dir",
            str(output_dir),
            "--class-map",
            str(class_map_path),
        ]
    )

    assert result == 0
    world = json.loads((output_dir / "world_coco.json").read_text(encoding="utf-8"))
    ui = json.loads((output_dir / "ui_coco.json").read_text(encoding="utf-8"))
    assert [(item["id"], item["session_id"], item["frame_id"]) for item in world["images"]] == [
        (1, "session-a", 3),
        (2, "session-b", 2),
    ]
    assert Path(world["images"][0]["file_name"]).as_posix() == "../work/session-a/00000003.png"
    assert world["images"][0]["width"] == 10
    assert world["images"][0]["height"] == 8
    class_map = load_class_map(class_map_path)
    world_category_id = class_map.name_to_id("player_anchor")
    assert world["annotations"] == [
        {
            "id": 1,
            "image_id": 1,
            "category_id": world_category_id,
            "bbox": [0.0, 1.0, 5.0, 7.0],
            "area": 35.0,
            "iscrowd": 0,
            "session_id": "session-a",
        }
    ]
    assert ui["annotations"][0]["category_id"] == 1
    assert ui["annotations"][0]["bbox"] == [8.0, 1.0, 2.0, 3.0]
    assert ui["annotations"][0]["area"] == 6.0
    assert [category["name"] for category in world["categories"]] == [
        item["name"] for item in class_map.foreground_classes
    ]
    assert [category["supercategory"] for category in world["categories"]] == [
        item["coarse_category"] for item in class_map.foreground_classes
    ]
    assert [category["name"] for category in ui["categories"]] == [
        "hud_hp",
        "hud_xp",
        "card",
        "button",
        "death_result",
    ]
    captured = capsys.readouterr()
    assert "矩形を除外" in captured.err
    assert "未確認でスキップした数: 1" in captured.out
    assert source_label.read_bytes() == original_label

    dataset = WorldDataset(output_dir / "world_coco.json", class_map_path, validate_bounds=True)
    assert len(dataset) == 2
    assert len(dataset[0].annotations) == 1
    assert len(dataset[1].annotations) == 0


def test_export_skips_checked_frames_with_labeled_region(tmp_path: Path, capsys) -> None:
    """labeled_region 付きの確認済みフレームを world/ui 両方の COCO から外す。

    範囲限定フレームは範囲外に未ラベルの物体があるため、全面 dataset に混ぜない。
    スキップ数は標準出力に表示され、labeled_region 自体も矩形として出ない。
    """
    work_root = tmp_path / "work"
    session = work_root / "session-a"
    _write_png(session / "00000001.png")
    _write_label(session / "00000001.json", [_shape("gem_blue", [[1, 1], [3, 3]])], checked=True)
    _write_png(session / "00000002.png")
    _write_label(
        session / "00000002.json",
        [
            _shape("labeled_region", [[0, 0], [5, 5]]),
            _shape("gem_blue", [[1, 1], [3, 3]]),
            _shape("hud_hp", [[1, 1], [4, 4]]),
        ],
        checked=True,
    )
    output_dir = tmp_path / "export"

    assert main(["--work-root", str(work_root), "--output-dir", str(output_dir)]) == 0

    for filename in ("world_coco.json", "ui_coco.json"):
        document = json.loads((output_dir / filename).read_text(encoding="utf-8"))
        assert [image["frame_id"] for image in document["images"]] == [1]
        assert all(annotation["image_id"] == 1 for annotation in document["annotations"])
        assert "labeled_region" not in [category["name"] for category in document["categories"]]
    world = json.loads((output_dir / "world_coco.json").read_text(encoding="utf-8"))
    assert len(world["annotations"]) == 1
    assert "範囲限定でスキップした数: 1" in capsys.readouterr().out


def test_export_rejects_checked_label_without_matching_png(tmp_path: Path, capsys) -> None:
    """確認済み JSON に対応する PNG がなければ CLI を失敗させる。

    欠落画像を黙って訓練データから外さない。
    """
    work_root = tmp_path / "work"
    label_path = work_root / "session-a" / "00000004.json"
    label_path.parent.mkdir(parents=True)
    _write_label(label_path, [], checked=True)

    result = main(
        ["--work-root", str(work_root), "--output-dir", str(tmp_path / "export")]
    )

    assert result == 1
    assert str(label_path.with_suffix(".png")) in capsys.readouterr().err


def test_export_accepts_repeated_session_id_filters(tmp_path: Path) -> None:
    """session-id を複数指定して対象を絞り込める。

    指定順ではなく session/frame の規定順で image ID を振る。
    """
    work_root = tmp_path / "work"
    for session_id, frame_id in (("session-b", 2), ("session-a", 1)):
        session_dir = work_root / session_id
        _write_png(session_dir / f"{frame_id:08d}.png")
        _write_label(session_dir / f"{frame_id:08d}.json", [], checked=True)
    output_dir = tmp_path / "export"

    assert main(
        [
            "--work-root",
            str(work_root),
            "--output-dir",
            str(output_dir),
            "--session-id",
            "session-b",
            "--session-id",
            "session-a",
        ]
    ) == 0
    world = json.loads((output_dir / "world_coco.json").read_text(encoding="utf-8"))
    assert [(image["session_id"], image["frame_id"]) for image in world["images"]] == [
        ("session-a", 1),
        ("session-b", 2),
    ]


def test_export_writes_weapon_effect_boxes_as_world_categories_12_to_15(tmp_path: Path) -> None:
    """weapon_* の矩形は world COCO の category 12〜15（class map v2 の ID）として出力する。

    UI 側には入らず、world の categories にも weapon 4 クラスが並ぶ。
    """
    work_root = tmp_path / "work"
    session = work_root / "session-a"
    _write_png(session / "00000001.png")
    _write_label(
        session / "00000001.json",
        [
            _shape("gem_blue", [[1, 1], [3, 3]]),
            _shape("weapon_aura", [[0, 0], [9, 7]]),
            _shape("hud_hp", [[5, 1], [7, 3]]),
        ],
        checked=True,
    )
    _write_png(session / "00000002.png")
    _write_label(
        session / "00000002.json",
        [_shape("weapon_projectile", [[1, 1], [2, 2]]), _shape("weapon_zone", [[3, 3], [6, 6]])],
        checked=True,
    )
    output_dir = tmp_path / "export"

    result = main(["--work-root", str(work_root), "--output-dir", str(output_dir)])

    assert result == 0
    world = json.loads((output_dir / "world_coco.json").read_text(encoding="utf-8"))
    ui = json.loads((output_dir / "ui_coco.json").read_text(encoding="utf-8"))
    assert [item["frame_id"] for item in world["images"]] == [1, 2]
    assert [item["frame_id"] for item in ui["images"]] == [1, 2]
    assert [(item["image_id"], item["category_id"], item["bbox"]) for item in world["annotations"]] == [
        (1, 5, [1.0, 1.0, 2.0, 2.0]),
        (1, 15, [0.0, 0.0, 9.0, 7.0]),
        (2, 12, [1.0, 1.0, 1.0, 1.0]),
        (2, 13, [3.0, 3.0, 3.0, 3.0]),
    ]
    assert [(item["image_id"], item["category_id"]) for item in ui["annotations"]] == [(1, 1)]
    world_categories = {item["id"]: item["name"] for item in world["categories"]}
    assert {k: world_categories[k] for k in range(12, 16)} == {
        12: "weapon_projectile", 13: "weapon_zone", 14: "weapon_orbit", 15: "weapon_aura",
    }
    assert not any(item["name"].startswith("weapon_") for item in ui["categories"])
