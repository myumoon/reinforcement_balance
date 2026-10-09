"""アノテーションラベル共通処理の契約を検証する。

X-AnyLabeling 形式、クラス一覧、フレーム列挙、矩形クリップの動作を確認する。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from survivors.annotation_labels import (
    ALL_CLASSES,
    REGION_LABEL,
    UI_CLASSES,
    WEAPON_EFFECT_CLASSES,
    WEAPON_EFFECT_KINDS,
    WORLD_CLASSES,
    LabelBox,
    clip_box,
    iter_frame_files,
    read_label_file,
    validate_label,
    write_label_file,
)
from reinbalance_survivors_contracts.deploy_obs_v2_features import load_deploy_obs_v2_feature_params
from survivors.vision.world_dataset import load_class_map


def _label_payload(shapes: list[dict], checked: object = True) -> dict:
    """テスト用の最小 X-AnyLabeling JSON を返す。

    共有 reader の必須フィールドを含む入力を簡潔に組み立てる。
    """
    return {
        "shapes": shapes,
        "imagePath": "00000001.png",
        "imageData": None,
        "checked": checked,
    }


def _shape(label: str = "gem_blue", points: list | None = None, **extra: object) -> dict:
    """指定した点を使う rectangle shape を返す。

    extra により GUI 由来の追加フィールドも再現できる。
    """
    return {
        "label": label,
        "shape_type": "rectangle",
        "points": points or [[1, 2], [5, 8]],
        **extra,
    }


def test_classes_follow_world_class_map_and_append_ui_classes() -> None:
    """共通クラス一覧が class map の順序に従う。

    UI クラスは WorldDetector の foreground に混ぜず、末尾へ追加する。
    """
    class_map_path = Path(__file__).resolve().parents[1] / "configs" / "world_class_map_v2.yaml"
    expected_world = tuple(
        item["name"]
        for item in sorted(load_class_map(class_map_path).foreground_classes, key=lambda item: item["id"])
    )

    assert WORLD_CLASSES == expected_world
    assert UI_CLASSES == ("hud_hp", "hud_xp", "card", "button", "death_result")
    assert REGION_LABEL == "labeled_region"
    assert WEAPON_EFFECT_CLASSES == ("weapon_projectile", "weapon_zone", "weapon_orbit", "weapon_aura")
    # weapon は class map v2 の world クラス（ID 12〜15）なので WORLD_CLASSES の末尾に入る。
    assert WORLD_CLASSES[-4:] == WEAPON_EFFECT_CLASSES
    assert ALL_CLASSES == WORLD_CLASSES + UI_CLASSES + (REGION_LABEL, "weapon_target")


def test_class_map_v2_names_match_common_entity_vocabulary() -> None:
    """class map v2 の enemy / gem / weapon の名前が Common の entity_classes と一致する。

    Common（03-06）の特徴量ビルダーは track の class 名で役割を引くため、名前がずれると特徴量が空になる。
    """
    cm = load_class_map(Path(__file__).resolve().parents[1] / "configs" / "world_class_map_v2.yaml")
    entity = load_deploy_obs_v2_feature_params()["entity_classes"]

    def names(coarse: str) -> set[str]:
        return {fc["name"] for fc in cm.foreground_classes if fc["coarse_category"] == coarse}

    assert set(entity["enemy"]) == names("enemy")
    assert set(entity["gem"]) == names("gem")
    assert set(entity["rare_gem"]) <= names("gem")
    assert set(entity["effect"]) == names("weapon")
    assert {f"weapon_{kind}" for kind in entity["effect"].values()} == names("weapon")


def test_weapon_effect_kinds_match_common_deploy_obs_v2_features() -> None:
    """武器→ラベル対応表が Common の weapon_effect_kinds と一致する。

    Common の種類名（projectile 等）に weapon_ を前置した集合と比べ、
    Common に無い武器（Pentagram など）は空集合であることも確認する。
    """
    params = load_deploy_obs_v2_feature_params()
    weapons = set(params["weapon_vocabulary"][1:-1])
    common = params["weapon_effect_kinds"]

    assert set(WEAPON_EFFECT_KINDS) == weapons
    for weapon in weapons:
        expected = {f"weapon_{kind}" for kind in common.get(weapon, ())}
        assert set(WEAPON_EFFECT_KINDS[weapon]) == expected, weapon
        assert set(WEAPON_EFFECT_KINDS[weapon]) <= set(WEAPON_EFFECT_CLASSES)


def test_validate_and_read_accept_weapon_effect_labels(tmp_path: Path) -> None:
    """validate_label と read_label_file が weapon_* を受け付ける。

    以前は未知ラベルとして読み飛ばされていたクラスが、今は矩形として読める。
    """
    for label in WEAPON_EFFECT_CLASSES:
        assert validate_label(label) == label
    path = tmp_path / "00000001.json"
    path.write_text(
        json.dumps(_label_payload([_shape(label=label) for label in WEAPON_EFFECT_CLASSES])),
        encoding="utf-8",
    )

    boxes, _ = read_label_file(path)

    assert [box.label for box in boxes] == list(WEAPON_EFFECT_CLASSES)


def test_write_and_read_label_file_preserve_contract(tmp_path: Path) -> None:
    """ラベルファイルを X-AnyLabeling v4.0.6 形式で往復する。

    writer は四隅の rectangle と必須メタデータを書き、reader は box に戻す。
    """
    path = tmp_path / "00000001.json"
    box = LabelBox("gem_blue", 2, 3, 10, 12, 0.93)

    write_label_file(path, [box], image_width=16, image_height=12, checked=True)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["version"] == "4.0.6"
    assert payload["shapes"][0]["points"] == [[2.0, 3.0], [10.0, 3.0], [10.0, 12.0], [2.0, 12.0]]
    assert payload["shapes"][0]["score"] == 0.93
    assert payload["imagePath"] == "00000001.png"
    assert payload["imageData"] is None
    assert payload["imageHeight"] == 12
    assert payload["imageWidth"] == 16
    assert read_label_file(path) == ([box], True)


def test_read_accepts_two_and_four_point_rectangles_and_ignores_extra_keys(tmp_path: Path) -> None:
    """GUI で保存された rectangle と追加キーを読み込む。

    対角二点と四隅四点は同じ bounding box として扱う。
    """
    two_point = _shape(points=[[9, 8], [1, 2]], score=0.5)
    four_point = _shape(
        points=[[1, 2], [9, 2], [9, 8], [1, 8]],
        tags=["reviewed"],
        chat_history={"ignored": True},
    )
    path = tmp_path / "00000001.json"
    path.write_text(
        json.dumps(_label_payload([two_point, four_point], checked=True)), encoding="utf-8"
    )

    boxes, checked = read_label_file(path)

    assert boxes == [
        LabelBox("gem_blue", 1.0, 2.0, 9.0, 8.0, 0.5),
        LabelBox("gem_blue", 1.0, 2.0, 9.0, 8.0, None),
    ]
    assert checked is True


@pytest.mark.parametrize("checked", [1, "true", False, None])
def test_checked_is_true_only_for_boolean_true(tmp_path: Path, checked: object) -> None:
    """確認済みフラグは JSON の boolean true だけを受理する。

    数値の 1 や文字列の true を暗黙変換しない。
    """
    path = tmp_path / "00000001.json"
    path.write_text(json.dumps(_label_payload([], checked)), encoding="utf-8")

    assert read_label_file(path)[1] is False


@pytest.mark.parametrize(
    "shape",
    [
        {"label": "gem_blue", "points": [[1, 2], [3, 4]]},
        {"label": "gem_blue", "shape_type": "rectangle"},
    ],
)
def test_reader_rejects_invalid_shapes_with_path(tmp_path: Path, shape: dict) -> None:
    """壊れた shape を入力ファイル名付きで拒否する。

    shape_type や points の必須キー欠落は、タイプミスと違い読み飛ばさずエラーにする。
    """
    path = tmp_path / "00000001.json"
    path.write_text(json.dumps(_label_payload([shape])), encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        read_label_file(path)

    assert str(path) in str(exc_info.value)


def test_reader_skips_unknown_label_and_unsupported_shape_type_with_warning(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """未知ラベルや未対応 shape_type の shape だけを警告付きで読み飛ばす。

    GUI 入力ミス（誤字ラベルや未対応図形）が1件あっても、他の shape は失わない。
    """
    good = _shape(label="gem_blue")
    typo_label = _shape(label="r")
    unsupported_shape = {"label": "gem_blue", "shape_type": "polygon", "points": [[1, 2], [3, 4]]}
    path = tmp_path / "00000001.json"
    path.write_text(
        json.dumps(_label_payload([good, typo_label, unsupported_shape])), encoding="utf-8"
    )

    boxes, checked = read_label_file(path)

    assert boxes == [LabelBox("gem_blue", 1.0, 2.0, 5.0, 8.0, None)]
    assert checked is True
    stderr = capsys.readouterr().err
    assert "unknown annotation label 'r'" in stderr
    assert "unsupported shape_type 'polygon'" in stderr


def test_read_converts_circle_center_and_edge_points_to_bounding_box(tmp_path: Path) -> None:
    """circle shape を中心点＋円周上の一点から外接矩形へ変換する。

    X-AnyLabeling で円として描いた hazard_area / hazard_projectile を bbox として扱う。
    """
    shape = {"label": "hazard_area", "shape_type": "circle", "points": [[10.0, 10.0], [13.0, 10.0]]}
    path = tmp_path / "00000001.json"
    path.write_text(json.dumps(_label_payload([shape])), encoding="utf-8")

    boxes, _ = read_label_file(path)

    assert boxes == [LabelBox("hazard_area", 7.0, 7.0, 13.0, 13.0, None)]


@pytest.mark.parametrize("missing", ["shapes", "imagePath", "imageData"])
def test_reader_rejects_missing_required_top_level_key(tmp_path: Path, missing: str) -> None:
    """X-AnyLabeling reader が必要とする最上位キーを検証する。

    不完全なラベルファイルはファイル名付き ValueError にする。
    """
    payload = _label_payload([])
    payload.pop(missing)
    path = tmp_path / "00000001.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        read_label_file(path)

    assert str(path) in str(exc_info.value)


def test_iter_frame_files_pairs_eight_digit_png_and_json(tmp_path: Path) -> None:
    """有効な八桁 stem の PNG と JSON を frame_id 順で列挙する。

    対応ファイルが片方だけの frame も返し、説明用ファイルは無視する。
    """
    (tmp_path / "00000002.png").touch()
    (tmp_path / "00000001.json").touch()
    (tmp_path / "00000002.json").touch()
    (tmp_path / "README.md").touch()
    (tmp_path / "12.png").touch()

    frames = list(iter_frame_files(tmp_path))

    assert [
        (frame_id, png.name if png else None, label.name if label else None)
        for frame_id, png, label in frames
    ] == [(1, None, "00000001.json"), (2, "00000002.png", "00000002.json")]


def test_clip_box_clamps_and_discards_zero_area() -> None:
    """矩形を画像範囲へ収め、面積ゼロを除外する。

    画像端からはみ出した矩形は縮めて保持する。
    """
    assert clip_box(LabelBox("gem_blue", -2, 3, 7, 20), image_width=5, image_height=8) == LabelBox(
        "gem_blue", 0, 3, 5, 8
    )
    assert clip_box(LabelBox("gem_blue", 6, 0, 9, 2), image_width=5, image_height=8) is None
