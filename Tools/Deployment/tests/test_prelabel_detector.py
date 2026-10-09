"""下書き用検出器モジュール（設定・学習データ収集・切り出し・保存/推論）を検証する。

合成画像と小さな CPU 学習だけを使い、事前学習重みのダウンロードはしない。
"""

from __future__ import annotations

import copy
import dataclasses
import random

import cv2
import numpy as np
import pytest
import torch
import yaml

from survivors.annotation_labels import REGION_LABEL, LabelBox, write_label_file
from survivors.prelabel_detector import (
    DEFAULT_CONFIG_PATH,
    IgnoreRegion,
    TrainingFrame,
    build_model,
    collect_training_frames,
    drop_ignored,
    load_config,
    load_detector,
    predict_boxes,
    sample_crop,
    save_detector,
    train_detector,
)


def _settings(**overrides):
    """同梱 v2 設定をもとに、テスト用に小さくした検出器設定を返す。

    CPU で数秒に収まるよう crop_size・倍率・反復回数を縮める。
    """
    base = load_config(DEFAULT_CONFIG_PATH).detector
    small = dict(crop_size=64, input_scale=1, iterations=2, batch_size=2, score_threshold=0.01)
    small.update(overrides)
    return dataclasses.replace(base, **small)


class _FixedFlip(random.Random):
    """反転判定 random() だけ固定値を返す乱数。

    randint / choice は通常どおり動くので、反転あり・なしを決め打ちで検証できる。
    """

    def __init__(self, value):
        super().__init__(0)
        self._value = value

    def random(self):
        return self._value


def _raw_config():
    """同梱 v2 設定ファイルを dict として読み直す。

    不正値テストで1項目だけ書き換える元データにする。
    """
    return yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))


def test_bundled_v2_config_loads_expected_values():
    """同梱 v2 設定が読め、固定矩形と検出器設定が期待どおりになる。

    固定矩形は v1 と同じ 3 つ、検出器は PoC で実測した値を使う。
    """
    config = load_config(DEFAULT_CONFIG_PATH)

    assert config.fixed_boxes == (
        LabelBox("player_anchor", 912, 466, 996, 540),
        LabelBox("hud_hp", 924, 544, 996, 554),
        LabelBox("hud_xp", 96, 0, 1824, 36),
    )
    detector = config.detector
    assert detector.labels == (
        "enemy_normal", "enemy_elite", "gem_blue", "gem_green", "gem_red", "pickup_heal", "pickup_special",
        "weapon_projectile", "weapon_zone", "weapon_orbit", "weapon_aura", "weapon_target",
    )
    assert detector.label_aliases == {"enemy_boss": "enemy_normal"}
    assert detector.label_score_thresholds == {"weapon_target": 0.92}
    assert config.ignore_regions == (IgnoreRegion(detector.labels, 96, 36, 390, 130),)
    assert (detector.score_threshold, detector.input_scale, detector.crop_size, detector.min_box_size) == (0.5, 2.0, 480, 6.0)
    assert (detector.iterations, detector.batch_size, detector.seed) == (1500, 4, 0)
    assert (detector.learning_rate, detector.momentum, detector.weight_decay) == (0.01, 0.9, 0.0001)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda data: data.update(extra=True),
        lambda data: data.pop("fixed_boxes"),
        lambda data: data.update(schema_version="annotation_prelabel.v1"),
        lambda data: data["fixed_boxes"][0].update(label="unknown_label"),
        lambda data: data["fixed_boxes"][0].update(label=REGION_LABEL),
        lambda data: data["fixed_boxes"][0].update(bbox=[0, 0, 2000, 40]),
        lambda data: data["fixed_boxes"][0].update(bbox=[0, 0, float("nan"), 40]),
        lambda data: data.pop("ignore_regions"),
        lambda data: data.update(ignore_regions={"labels": ["weapon_orbit"]}),
        lambda data: data["ignore_regions"][0].update(extra=1),
        lambda data: data["ignore_regions"][0].update(labels=[]),
        lambda data: data["ignore_regions"][0].update(labels=["hud_hp"]),
        lambda data: data["ignore_regions"][0].update(bbox=[0, 0, 2000, 40]),
        lambda data: data["ignore_regions"][0].update(bbox=[10, 10, 5, 40]),
        lambda data: data["detector"].update(extra=1),
        lambda data: data["detector"].pop("seed"),
        lambda data: data["detector"].update(labels=[]),
        lambda data: data["detector"].update(labels=["gem_blue", "gem_blue", "enemy_normal"]),
        lambda data: data["detector"].update(labels=["unknown_label", "enemy_normal"]),
        lambda data: data["detector"].update(labels=["hud_hp", "enemy_normal"]),
        lambda data: data["detector"].update(labels=["death_result", "weapon_aura"]),
        lambda data: data["detector"].update(labels=[REGION_LABEL, "enemy_normal"]),
        lambda data: data["detector"].update(label_aliases={"card": "weapon_zone"}),
        lambda data: data["detector"].update(label_aliases={"enemy_elite": "gem_green_missing"}),
        lambda data: data["detector"].update(label_aliases={"enemy_elite": "player_anchor"}),
        lambda data: data["detector"].update(label_aliases={"gem_blue": "enemy_normal"}),
        lambda data: data["detector"].update(label_aliases={REGION_LABEL: "enemy_normal"}),
        lambda data: data["detector"].update(label_aliases={"enemy_elite": REGION_LABEL}),
        lambda data: data["detector"].update(label_aliases=["enemy_elite"]),
        lambda data: data["detector"].update(score_threshold=0),
        lambda data: data["detector"].update(score_threshold=1.5),
        lambda data: data["detector"].update(score_threshold="0.5"),
        lambda data: data["detector"].update(label_score_thresholds=["weapon_target"]),
        lambda data: data["detector"].update(label_score_thresholds={"hud_hp": 0.9}),
        lambda data: data["detector"].update(label_score_thresholds={"weapon_target": 0}),
        lambda data: data["detector"].update(label_score_thresholds={"weapon_target": 1.5}),
        lambda data: data["detector"].update(label_score_thresholds={"weapon_target": "0.9"}),
        lambda data: data["detector"].pop("label_score_thresholds"),
        lambda data: data["detector"].update(input_scale=0),
        lambda data: data["detector"].update(input_scale=float("inf")),
        lambda data: data["detector"].update(crop_size=True),
        lambda data: data["detector"].update(crop_size=1.5),
        lambda data: data["detector"].update(crop_size=0),
        lambda data: data["detector"].update(iterations=0),
        lambda data: data["detector"].update(batch_size=-1),
        lambda data: data["detector"].update(min_box_size=-1),
        lambda data: data["detector"].update(learning_rate=0),
        lambda data: data["detector"].update(momentum=1.0),
        lambda data: data["detector"].update(weight_decay=-0.1),
        lambda data: data["detector"].update(seed="0"),
        lambda data: data["detector"].update(seed=True),
    ],
)
def test_load_config_rejects_invalid_values(tmp_path, mutation):
    """キーの過不足・型違い・非有限値・範囲外値・不正ラベルを ValueError で拒否する。

    labeled_region は固定矩形・検出ラベル・別名のどこにも使えない。
    """
    data = copy.deepcopy(_raw_config())
    mutation(data)
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid.yaml"):
        load_config(config_path)


def test_load_config_accepts_weapon_effect_labels_and_aliases(tmp_path):
    """武器エフェクトクラスは検出ラベルにも別名の両側にも使える。

    下書き可能クラスは WORLD_CLASSES（class map v2 で weapon_* を含む）。
    """
    data = copy.deepcopy(_raw_config())
    data["detector"].update(
        labels=["weapon_aura", "enemy_normal"], label_aliases={"weapon_orbit": "weapon_aura"}, label_score_thresholds={}
    )
    data["ignore_regions"] = []
    config_path = tmp_path / "weapon.yaml"
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")

    detector = load_config(config_path).detector

    assert detector.labels == ("weapon_aura", "enemy_normal")
    assert detector.label_aliases == {"weapon_orbit": "weapon_aura"}


def test_load_config_accepts_annotation_only_label(tmp_path):
    """アノテーション専用クラス（照準の weapon_target）も下書きの検出ラベルに使える。

    world class map には無いクラスだが、手で付ける手間を減らすため下書きの対象にできる。
    """
    data = copy.deepcopy(_raw_config())
    data["detector"].update(labels=["enemy_normal", "weapon_target"], label_aliases={})
    data["ignore_regions"] = []
    config_path = tmp_path / "target.yaml"
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")

    assert load_config(config_path).detector.labels == ("enemy_normal", "weapon_target")


def test_drop_ignored_removes_only_listed_labels_centered_inside():
    """ignore_regions の対象ラベルで中心が範囲内のものだけ外し、他は残す。

    範囲内を通る敵（対象外ラベル）と、範囲外の武器エフェクトは下書きに残る。
    """
    region = IgnoreRegion(("weapon_orbit",), 100, 40, 200, 80)
    icon = LabelBox("weapon_orbit", 120, 45, 150, 75, 0.99)
    enemy_inside = LabelBox("enemy_normal", 120, 45, 150, 75, 0.99)
    orbit_outside = LabelBox("weapon_orbit", 400, 400, 430, 430, 0.99)
    edge = LabelBox("weapon_orbit", 180, 60, 240, 100, 0.99)  # 中心 (210, 80) は範囲外

    assert drop_ignored([icon, enemy_inside, orbit_outside, edge], (region,)) == [enemy_inside, orbit_outside, edge]


def _write_frame(session_dir, frame_id, boxes, *, checked, png=True):
    """小さな PNG とラベル JSON を1フレーム分書き出す。

    png=False で「確認済みなのに画像が無い」壊れた状態も作れる。
    """
    session_dir.mkdir(parents=True, exist_ok=True)
    if png:
        assert cv2.imwrite(str(session_dir / f"{frame_id:08d}.png"), np.zeros((40, 60, 3), np.uint8))
    write_label_file(session_dir / f"{frame_id:08d}.json", boxes, image_width=60, image_height=40, checked=checked)


def test_collect_training_frames_applies_aliases_and_keeps_background_frames(tmp_path):
    """確認済みフレームだけを走査順に集め、別名置換と対象ラベルの絞り込みをする。

    範囲は regions に分かれ、対象矩形が無い確認済みフレームも背景として残る。
    """
    _write_frame(
        tmp_path / "b-session", 1,
        [LabelBox("enemy_boss", 1, 1, 9, 9), LabelBox("hud_hp", 0, 0, 5, 5), LabelBox(REGION_LABEL, 0, 0, 30, 20)],
        checked=True,
    )
    _write_frame(tmp_path / "a-session", 2, [LabelBox("gem_blue", 2, 2, 8, 8)], checked=False)
    _write_frame(tmp_path / "a-session", 3, [LabelBox("player_anchor", 2, 2, 8, 8)], checked=True)

    frames = collect_training_frames(tmp_path, _settings())

    assert [frame.png_path.name for frame in frames] == ["00000003.png", "00000001.png"]
    assert frames[0].boxes == () and frames[0].regions == ()
    assert frames[1].boxes == (LabelBox("enemy_normal", 1, 1, 9, 9),)
    assert frames[1].regions == (LabelBox(REGION_LABEL, 0, 0, 30, 20),)


def test_collect_training_frames_rejects_checked_label_without_png(tmp_path):
    """確認済み JSON に PNG が無ければパス付き ValueError にする。

    画像欠落を黙って学習データから外さない。
    """
    _write_frame(tmp_path / "s", 1, [], checked=True, png=False)

    with pytest.raises(ValueError, match="00000001.json"):
        collect_training_frames(tmp_path, _settings())


def _region_image():
    """範囲の内側だけ特定色、外側は白の 100x100 画像を作る。

    切り出しに範囲外の画素（白）が混ざっていないかを色で判定する。
    """
    image = np.full((100, 100, 3), 255, np.uint8)
    image[20:70, 20:70] = (10, 20, 30)
    return image


@pytest.mark.parametrize("seed", range(10))
def test_sample_crop_never_uses_pixels_outside_region(seed):
    """範囲が crop_size より広いとき、切り出しは全画素が範囲の内側になる。

    区切り位置をずらした複数の範囲・乱数で確認する。
    """
    frame = TrainingFrame(
        png_path=None, boxes=(),
        regions=(LabelBox(REGION_LABEL, 19.5, 20, 70, 70.4), LabelBox(REGION_LABEL, 20, 20, 70, 70)),
    )

    crop, boxes, names = sample_crop(_region_image(), frame, crop_size=40, min_box_size=0, rng=random.Random(seed))

    assert crop.shape == (40, 40, 3) and crop.dtype == np.uint8
    assert (crop == np.array([10, 20, 30], np.uint8)).all()
    assert boxes.shape == (0, 4) and names == []


@pytest.mark.parametrize("flip", [0.9, 0.1])
def test_sample_crop_pads_small_region_with_zeros(flip):
    """範囲が crop_size より狭い辺はゼロ埋めし、範囲外の画素は入れない。

    反転の有無に関わらず、範囲の画素数だけ特定色が残り、他は 0 になる。
    """
    frame = TrainingFrame(png_path=None, boxes=(), regions=(LabelBox(REGION_LABEL, 20, 20, 40, 35),))

    crop, _, _ = sample_crop(_region_image(), frame, crop_size=32, min_box_size=0, rng=_FixedFlip(flip))

    region_pixels = (crop == np.array([10, 20, 30], np.uint8)).all(axis=2)
    assert region_pixels.sum() == 20 * 15
    assert (crop[~region_pixels] == 0).all()
    if flip > 0.5:
        assert region_pixels[:15, :20].all()
    else:
        assert region_pixels[:15, 12:].all()


@pytest.mark.parametrize(
    ("flip", "expected_box"),
    [(0.9, [2, 4, 10, 20]), (0.1, [30, 4, 38, 20])],
)
def test_sample_crop_clips_filters_and_flips_boxes(flip, expected_box):
    """矩形を切り出し範囲へクリップ・平行移動し、小さすぎる矩形を捨て、反転時は x も反転する。

    範囲と crop_size を同じ大きさにして切り出し位置を固定し、座標を厳密に比べる。
    """
    frame = TrainingFrame(
        png_path=None,
        boxes=(
            LabelBox("gem_blue", 12, 14, 20, 30),
            LabelBox("enemy_normal", 45, 45, 60, 60),
            LabelBox("gem_red", 0, 0, 100, 11),
        ),
        regions=(LabelBox(REGION_LABEL, 10, 10, 50, 50),),
    )
    image = np.zeros((100, 100, 3), np.uint8)
    image[10, 10] = (1, 2, 3)

    crop, boxes, names = sample_crop(image, frame, crop_size=40, min_box_size=6, rng=_FixedFlip(flip))

    assert names == ["gem_blue"]
    assert boxes.dtype == np.float32
    np.testing.assert_array_equal(boxes, np.array([expected_box], np.float32))
    marker = (0, 0) if flip > 0.5 else (0, 39)
    assert tuple(crop[marker]) == (1, 2, 3)


def test_sample_crop_uses_whole_image_without_region():
    """範囲が無いフレームは画像全体を範囲として扱う。

    画像が crop_size より小さい場合は左上に置いて残りをゼロ埋めする。
    """
    image = np.full((20, 30, 3), 7, np.uint8)
    frame = TrainingFrame(png_path=None, boxes=(LabelBox("gem_blue", 5, 5, 15, 15),), regions=())

    crop, boxes, names = sample_crop(image, frame, crop_size=32, min_box_size=6, rng=_FixedFlip(0.9))

    assert (crop[:20, :30] == 7).all()
    assert (crop[20:] == 0).all() and (crop[:, 30:] == 0).all()
    np.testing.assert_array_equal(boxes, np.array([[5, 5, 15, 15]], np.float32))
    assert names == ["gem_blue"]


def test_build_model_without_pretrained_weights_sets_head_size():
    """pretrained=False ではダウンロード無しで構築し、head を labels + 背景に差し替える。

    学習時の入力サイズは crop_size * input_scale になる。
    """
    model = build_model(3, _settings(crop_size=64, input_scale=2), pretrained=False)

    assert model.roi_heads.box_predictor.cls_score.out_features == 4
    assert model.transform.min_size == (128,)
    assert model.transform.max_size == 128
    assert model.roi_heads.detections_per_img == 1000


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """小さな合成データで2反復だけ学習したモデルと設定を用意する。

    保存・読み込み・推論のテストで共有し、学習時間を1回分に抑える。
    """
    root = tmp_path_factory.mktemp("frames")
    image = np.zeros((80, 96, 3), np.uint8)
    image[20:40, 30:50] = (0, 200, 255)
    frames = []
    for index in range(2):
        png = root / f"{index:08d}.png"
        assert cv2.imwrite(str(png), image)
        frames.append(TrainingFrame(png, (LabelBox("gem_blue", 30, 20, 50, 40),), ()))
    settings = _settings()
    logs = []
    model = train_detector(frames, settings, device="cpu", pretrained=False, log=logs.append)
    return model, settings, image, logs


def test_train_detector_logs_progress_and_returns_eval_model(trained):
    """学習は最終 iter の loss をログに出し、推論モードのモデルを返す。

    反復回数が少なくても最後の進捗は必ず出る。
    """
    model, _, _, logs = trained

    assert not model.training
    assert logs and logs[-1].startswith("iter 2/2 loss=")


def test_save_load_round_trip_predicts_same_boxes(trained, tmp_path):
    """保存して読み直した重みで、元のモデルと同じ推論結果になる。

    推論結果は設定のラベル名・しきい値以上のスコア・画像内の座標だけを含む。
    """
    model, settings, image, _ = trained
    path = tmp_path / "nested" / "prelabel_detector.pt"

    save_detector(model, path, settings)
    loaded = load_detector(path, settings, device="cpu")
    expected = predict_boxes(model, image, settings, device="cpu")

    assert predict_boxes(loaded, image, settings, device="cpu") == expected
    assert expected, "untrained head should still produce low-score boxes above 0.01"
    for box in expected:
        assert box.label in settings.labels
        assert box.score >= settings.score_threshold
        assert 0 <= box.left < box.right <= 96 and 0 <= box.top < box.bottom <= 80
    assert [item.name for item in path.parent.iterdir()] == ["prelabel_detector.pt"]


def test_predict_boxes_applies_score_threshold(trained):
    """score_threshold 未満の検出は返さない。

    しきい値 1.0 なら未学習に近いモデルの検出は全て落ちる。
    """
    model, settings, image, _ = trained

    assert predict_boxes(model, image, dataclasses.replace(settings, score_threshold=1.0), device="cpu") == []


def test_predict_boxes_applies_label_score_threshold(trained):
    """label_score_thresholds のラベルだけ、全体より高いしきい値で落とす。

    他のラベルの検出は全体の score_threshold のまま残る。
    """
    model, settings, image, _ = trained
    base = dataclasses.replace(settings, label_score_thresholds={})
    boxes = predict_boxes(model, image, base, device="cpu")
    target = boxes[0].label

    filtered = predict_boxes(model, image, dataclasses.replace(base, label_score_thresholds={target: 1.0}), device="cpu")

    assert filtered == [box for box in boxes if box.label != target]


@pytest.mark.parametrize(
    "mismatch",
    [
        lambda settings: dataclasses.replace(settings, labels=settings.labels[:-1]),
        lambda settings: dataclasses.replace(settings, input_scale=2),
    ],
)
def test_load_detector_rejects_mismatched_metadata(tmp_path, mismatch):
    """保存時と設定の labels / input_scale が違えばパス付き ValueError にする。

    学習時と違う条件で推論して、ずれた下書きを作らない。
    """
    settings = _settings()
    path = tmp_path / "prelabel_detector.pt"
    save_detector(build_model(len(settings.labels), settings, pretrained=False), path, settings)

    with pytest.raises(ValueError, match="prelabel_detector.pt"):
        load_detector(path, mismatch(settings), device="cpu")


@pytest.mark.parametrize(
    "payload",
    [
        {"format": "other", "labels": [], "input_scale": 1.0, "state_dict": {}},
        {"labels": [], "input_scale": 1.0},
        [1, 2, 3],
    ],
)
def test_load_detector_rejects_unknown_format(tmp_path, payload):
    """format が違う・欠けたファイルはパス付き ValueError にする。

    別用途の重みを誤って読み込まない。
    """
    path = tmp_path / "other.pt"
    torch.save(payload, path)

    with pytest.raises(ValueError, match="other.pt"):
        load_detector(path, _settings(), device="cpu")


def test_load_detector_rejects_missing_file(tmp_path):
    """重みファイルが無ければパス付き ValueError にする。

    読めない理由をファイル名付きで利用者に示す。
    """
    with pytest.raises(ValueError, match="missing.pt"):
        load_detector(tmp_path / "missing.pt", _settings(), device="cpu")


def test_train_detector_rejects_empty_frames_and_unreadable_image(tmp_path):
    """学習フレームが0件、または画像が読めない場合は ValueError にする。

    読めない画像はパスをメッセージに含める。
    """
    with pytest.raises(ValueError):
        train_detector([], _settings(), device="cpu", pretrained=False)

    broken = tmp_path / "00000001.png"
    broken.write_bytes(b"not a png")
    with pytest.raises(ValueError, match="00000001.png"):
        train_detector([TrainingFrame(broken, (), ())], _settings(), device="cpu", pretrained=False)
