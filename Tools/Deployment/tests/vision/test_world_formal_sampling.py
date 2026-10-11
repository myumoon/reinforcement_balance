"""正式 sampler、loss weighting、画像変換の回帰を検証する。\n\nsession balance と augmentation の edge case を小さな fixture で固定する。"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from survivors.vision.world_formal_sampling import (
    SessionBalancedSampler,
    augment,
    class_weights,
    near_duplicate_filter,
)


def test_session_balanced_sampler_covers_all_samples_and_repeats_small_session() -> None:
    """短い session を補いながら全 sample を epoch ごとに含める。

    round-robin と seed 固定による再現性も確認する。
    """
    dataset = [
        SimpleNamespace(session_id="small"),
        SimpleNamespace(session_id="large"),
        SimpleNamespace(session_id="large"),
        SimpleNamespace(session_id="large"),
    ]
    sampler = SessionBalancedSampler(dataset, seed=9, epoch=2)

    indices = list(sampler)

    assert set(indices) == {0, 1, 2, 3}
    assert len(indices) == 5
    assert indices == list(SessionBalancedSampler(dataset, seed=9, epoch=2))
    assert dataset[indices[0]].session_id != dataset[indices[1]].session_id


def test_class_weights_give_rare_classes_more_weight() -> None:
    """inverse frequency は少数 class を高く重み付けする。

    クラス別重みの大小関係を実データ数から確認する。
    """
    dataset = [
        SimpleNamespace(annotations=[SimpleNamespace(category_id=1)]),
        SimpleNamespace(annotations=[SimpleNamespace(category_id=1)]),
        SimpleNamespace(annotations=[SimpleNamespace(category_id=2)]),
    ]
    class_map = SimpleNamespace(foreground_classes=({"id": 1}, {"id": 2}))

    weights = class_weights(dataset, class_map, mode="inverse_frequency", clip_max=10.0)

    assert weights[2] > weights[1]


def test_near_duplicate_filter_only_drops_same_session_neighbor(tmp_path) -> None:
    """平均 hash の近似重複除外を session 境界の内側に限定する。

    異なる session の同一画像は独立サンプルとして残す。
    """
    cv2 = pytest.importorskip("cv2")
    frame = np.zeros((24, 24, 3), dtype=np.uint8)
    different = np.full((24, 24, 3), 255, dtype=np.uint8)
    paths = [tmp_path / f"{index}.png" for index in range(3)]
    for path, pixels in zip(paths, (frame, frame, frame)):
        assert cv2.imwrite(str(path), pixels)
    dataset = [
        SimpleNamespace(session_id="a", file_name=str(paths[0])),
        SimpleNamespace(session_id="a", file_name=str(paths[1])),
        SimpleNamespace(session_id="b", file_name=str(paths[2])),
    ]

    excluded = near_duplicate_filter(dataset, hamming_threshold=0)

    assert set(excluded) == {1}
    assert "hamming" in excluded[1]


def test_disabled_augmentation_is_identity() -> None:
    """augmentation が無効なら image と boxes の値を変えない。

    画像と bbox の両方で恒等変換を保証する。
    """
    image = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    boxes = np.asarray([[1.0, 1.0, 3.0, 3.0]], dtype=np.float32)

    result_image, result_boxes = augment(
        image, boxes, cfg={"enabled": False}, rng=np.random.default_rng(0)
    )

    np.testing.assert_array_equal(result_image, image)
    np.testing.assert_array_equal(result_boxes, boxes)


def test_weighted_loss_changes_classification_without_changing_state_dict() -> None:
    """foreground weight が classification loss にだけ反映される。

    既存 SSD の state_dict key を維持したまま method を差し替える。
    """
    torch = pytest.importorskip("torch")

    class FakeSSD(torch.nn.Module):
        """weighted loss の境界だけを再現する小さな SSD model。\n\nstate_dict key を固定し torchvision の重い detector 構築を避ける。"""
        def __init__(self):
            """state_dict の比較に使う単一 parameter を持つ。

            SSD loss の最小限の matcher 依存だけを提供する。
            """
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))
            self.neg_to_pos_ratio = 3
            self.box_coder = SimpleNamespace(
                encode_single=lambda gt, anchors: torch.zeros_like(anchors)
            )

    def loss_for(weight: float) -> tuple[float, tuple[str, ...]]:
        """同じ入力の分類 loss を指定 weight で計算する。

        bbox loss を 0 に固定して class 重みの差だけを見る。
        """
        from survivors.vision.world_formal_sampling import attach_weighted_loss

        model = FakeSSD()
        keys = tuple(model.state_dict())
        attach_weighted_loss(model, {1: weight})
        targets = [{"boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]]), "labels": torch.tensor([1])}]
        outputs = {
            "bbox_regression": torch.zeros((1, 2, 4)),
            "cls_logits": torch.tensor([[[0.0, 0.2], [0.4, 0.0]]]),
        }
        anchors = [torch.tensor([[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 2.0, 2.0]])]
        matched = [torch.tensor([0, -1])]
        loss = model.compute_loss(targets, outputs, anchors, matched)["classification"]
        return float(loss), tuple(model.state_dict())

    baseline, base_keys = loss_for(1.0)
    weighted, weighted_keys = loss_for(4.0)

    assert baseline != weighted
    assert base_keys == weighted_keys == ("scale",)


def test_augmentation_drops_zero_area_boxes_and_their_labels() -> None:
    """scale jitter で画面外へ消えた bbox と label を同じ index で除く。

    片方の box が残っても元の label 対応を保つ。
    """
    class FixedRng:
        """scale jitter の上限を再現する固定乱数源。\n\nbox が画面外へ出る境界を毎回同じ値でテストする。"""
        def random(self):
            """水平反転を発生させない値を返す。\n\nこの test では box 切詰めの挙動だけを観測する。"""
            return 0.9
        def uniform(self, _low, _high):
            """scale jitter を上限 1.5 に固定する。\n\n一つの box を画像外へ押し出す再現可能な入力にする。"""
            return 1.5

    image = np.zeros((100, 100, 3), dtype=np.uint8)
    boxes = np.asarray([[0, 0, 10, 10], [40, 40, 60, 60]], dtype=np.float32)

    _, result_boxes, labels = augment(
        image,
        boxes,
        cfg={"enabled": True, "hflip_p": 0.0, "scale_jitter": 0.5},
        rng=FixedRng(),
        labels=np.asarray([1, 2]),
    )

    assert result_boxes.shape == (1, 4)
    np.testing.assert_array_equal(labels, [2])
