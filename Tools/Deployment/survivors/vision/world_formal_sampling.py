"""正式 detector 学習で共有する sampler、重み、augmentation。"""

from __future__ import annotations

import math
import pathlib
import types
from collections import defaultdict
from collections.abc import Mapping
from statistics import median
from typing import Any, Iterator

import numpy as np


class SessionBalancedSampler:
    """session ごとの最小 quota を round-robin で返す。"""

    def __init__(self, ds: Any, *, seed: int, epoch: int, min_quota: str = "median") -> None:
        """データ件数と seed から epoch 固定の順序を準備する。

        大きい session は全件を一度だけ、小さい session は中央値まで補う。
        """
        if type(seed) is not int or type(epoch) is not int or epoch < 0:
            raise ValueError("seed は int、epoch は 0 以上の int が必要です")
        if min_quota != "median":
            raise ValueError("min_quota は 'median' のみサポートします")
        groups: dict[str, list[int]] = defaultdict(list)
        for index in range(len(ds)):
            session_id = getattr(ds[index], "session_id", None)
            if not isinstance(session_id, str) or not session_id:
                raise ValueError(f"sample {index} に session_id がありません")
            groups[session_id].append(index)
        rng = np.random.default_rng(seed + epoch)
        quota = math.ceil(median([len(group) for group in groups.values()])) if groups else 0
        self._groups: list[list[int]] = []
        for session_id in sorted(groups):
            values = np.asarray(groups[session_id], dtype=np.int64)
            rng.shuffle(values)
            count = max(len(values), quota)
            if count > len(values):
                values = np.concatenate((values, rng.choice(values, count - len(values), replace=True)))
            self._groups.append(values.tolist())
        self._indices = [index for offset in range(max(map(len, self._groups), default=0))
                         for group in self._groups if offset < len(group)
                         for index in (group[offset],)]

    def __iter__(self) -> Iterator[int]:
        """固定済み epoch 順序を iterator として返す。

        同じ seed と epoch の再走査が同じ sample 列になる。
        """
        return iter(self._indices)

    def __len__(self) -> int:
        """epoch 内の全 training index 数を返す。

        quota による復元抽出分も件数へ含める。
        """
        return len(self._indices)


def class_weights(
    ds: Any,
    class_map: Any,
    *,
    mode: str = "inverse_frequency",
    clip_max: float = 10.0,
    explicit: Mapping[int | str, float] | None = None,
) -> dict[int, float]:
    """annotation frequency または明示値から foreground class weight を作る。

    background は含めず、明示 weight の欠落や非有限値を拒否する。
    """
    if isinstance(clip_max, bool) or not isinstance(clip_max, (int, float)) or not math.isfinite(clip_max) or clip_max <= 0:
        raise ValueError("clip_max は有限の正数が必要です")
    ids = [int(item["id"]) for item in class_map.foreground_classes]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("class_map の foreground id が空または重複しています")
    if mode == "explicit":
        if not isinstance(explicit, Mapping):
            raise ValueError("explicit mode には class 別 weight mapping が必要です")
        result = {}
        for class_id in ids:
            value = explicit.get(class_id, explicit.get(str(class_id)))
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"class {class_id} の explicit weight は有限の正数が必要です")
            result[class_id] = float(value)
        return result
    if mode != "inverse_frequency":
        raise ValueError("mode は 'inverse_frequency' または 'explicit' が必要です")
    counts = dict.fromkeys(ids, 0)
    for index in range(len(ds)):
        for annotation in ds[index].annotations:
            if annotation.category_id not in counts:
                raise ValueError(f"unknown class id: {annotation.category_id}")
            counts[annotation.category_id] += 1
    total = sum(counts.values())
    if total == 0:
        raise ValueError("inverse-frequency weight に使える annotation がありません")
    scale = total / len(ids)
    return {key: min(float(clip_max), scale / max(count, 1)) for key, count in counts.items()}


def attach_weighted_loss(model: Any, weights: Mapping[int, float]) -> Any:
    """SSD instance の compute_loss を class-weighted loss へ束縛する。

    instance class と state_dict key を変えず foreground だけを加重する。
    """
    normalized = {int(key): float(value) for key, value in weights.items()}
    if not normalized or any(not math.isfinite(value) or value <= 0 for value in normalized.values()):
        raise ValueError("class weights は有限の正数 mapping が必要です")
    if model.__class__.__name__ == "_StubModel":
        model.formal_class_weights = normalized
        return model
    if not hasattr(model, "state_dict") or not hasattr(model, "box_coder"):
        raise TypeError("weighted loss は SSD-compatible instance が必要です")
    try:
        import torch
        import torch.nn.functional as F
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        raise RuntimeError("weighted SSD loss には torch が必要です") from exc

    def compute_loss(self, targets, head_outputs, anchors, matched_idxs):
        """torchvision SSD と同じ hard-negative mining 後に分類 loss を加重する。

        negative 選択は元 loss で行い、正例 class の寄与だけを変える。
        """
        bbox_losses = []
        class_targets = []
        class_weights_per_image = []
        foreground_count = 0
        for target, regression, logits, anchor, matched in zip(
            targets, head_outputs["bbox_regression"], head_outputs["cls_logits"], anchors, matched_idxs
        ):
            foreground = torch.where(matched >= 0)[0]
            matched_foreground = matched[foreground]
            foreground_count += matched_foreground.numel()
            gt_boxes = target["boxes"][matched_foreground]
            bbox_losses.append(F.smooth_l1_loss(
                regression[foreground], self.box_coder.encode_single(gt_boxes, anchor[foreground]), reduction="sum"
            ))
            target_classes = torch.zeros(logits.size(0), dtype=target["labels"].dtype, device=target["labels"].device)
            target_classes[foreground] = target["labels"][matched_foreground]
            class_targets.append(target_classes)
            scales = torch.ones_like(target_classes, dtype=logits.dtype)
            for class_id, value in normalized.items():
                scales[target_classes == class_id] = value
            class_weights_per_image.append(scales)
        class_targets = torch.stack(class_targets)
        class_weights_per_image = torch.stack(class_weights_per_image)
        class_count = head_outputs["cls_logits"].size(-1)
        logits = head_outputs["cls_logits"].reshape(-1, class_count)
        raw_loss = F.cross_entropy(logits, class_targets.reshape(-1), reduction="none").reshape(class_targets.size())
        foreground = class_targets > 0
        negative_loss = raw_loss.clone()
        negative_loss[foreground] = -float("inf")
        _, negative_order = negative_loss.sort(1, descending=True)
        negative_mask = negative_order.sort(1)[1] < self.neg_to_pos_ratio * foreground.sum(1, keepdim=True)
        n = max(1, foreground_count)
        return {
            "bbox_regression": torch.stack(bbox_losses).sum() / n,
            "classification": (raw_loss[foreground] * class_weights_per_image[foreground]).sum() / n
            + raw_loss[negative_mask].sum() / n,
        }

    model.compute_loss = types.MethodType(compute_loss, model)
    return model


def near_duplicate_filter(ds: Any, *, hamming_threshold: int) -> dict[int, str]:
    """同一 session の連続 frame を平均 hash の距離で重複除外する。

    画像欠落は除外理由へ丸めず fail-closed で通知する。
    """
    if type(hamming_threshold) is not int or not 0 <= hamming_threshold <= 64:
        raise ValueError("hamming_threshold は 0..64 の int が必要です")
    import cv2

    def average_hash(path: str) -> int:
        """画像を 8x8 gray hash に変換する。

        hash 化不能な path を黙って採用しない。
        """
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"near-duplicate image cannot be read: {path}")
        pixels = cv2.resize(image, (8, 8), interpolation=cv2.INTER_AREA).astype(np.float32)
        bits = pixels >= pixels.mean()
        value = 0
        for bit in bits.flat:
            value = (value << 1) | int(bit)
        return value

    previous: dict[str, int] = {}
    excluded: dict[int, str] = {}
    for index in range(len(ds)):
        sample = ds[index]
        session_id = getattr(sample, "session_id", None)
        if not isinstance(session_id, str) or not session_id:
            raise ValueError(f"sample {index} に session_id がありません")
        digest = average_hash(sample.file_name)
        if session_id in previous and (previous[session_id] ^ digest).bit_count() <= hamming_threshold:
            excluded[index] = f"same_session_hamming<={hamming_threshold}"
        previous[session_id] = digest
    return excluded


def augment(
    image: np.ndarray,
    boxes: np.ndarray,
    *,
    cfg: Mapping[str, Any],
    rng: Any,
    labels: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """画像へ水平反転、scale、HSV jitter を適用して bbox も追従させる。

    無効設定は恒等、変換後に面積が無い box は除外する。
    """
    if not isinstance(cfg, Mapping) or not isinstance(cfg.get("enabled"), bool):
        raise ValueError("augmentation cfg.enabled は bool が必要です")
    if labels is not None and len(labels) != len(boxes):
        raise ValueError("bbox と labels の件数が一致しません")
    if not cfg["enabled"]:
        return (image, boxes, np.asarray(labels)) if labels is not None else (image, boxes)
    import cv2

    result = image.copy()
    transformed = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
    height, width = result.shape[:2]
    if not np.isfinite(transformed).all():
        raise ValueError("bbox に非有限値があります")
    hflip_p = float(cfg.get("hflip_p", 0.0))
    scale_jitter = float(cfg.get("scale_jitter", 0.0))
    hsv = cfg.get("hsv_jitter", {})
    if not 0 <= hflip_p <= 1 or not 0 <= scale_jitter < 1 or not isinstance(hsv, Mapping):
        raise ValueError("augmentation の hflip_p/scale_jitter/hsv_jitter が不正です")
    if rng.random() < hflip_p:
        result = cv2.flip(result, 1)
        old_left = transformed[:, 0].copy()
        transformed[:, 0] = width - transformed[:, 2]
        transformed[:, 2] = width - old_left
    if scale_jitter:
        scale = float(rng.uniform(1.0 - scale_jitter, 1.0 + scale_jitter))
        matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), 0, scale)
        result = cv2.warpAffine(result, matrix, (width, height))
        transformed[:, [0, 2]] = (transformed[:, [0, 2]] - width / 2.0) * scale + width / 2.0
        transformed[:, [1, 3]] = (transformed[:, [1, 3]] - height / 2.0) * scale + height / 2.0
    if hsv:
        value = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
        hue = float(hsv.get("hue", 0.0))
        saturation = float(hsv.get("saturation", 0.0))
        brightness = float(hsv.get("value", 0.0))
        if any(not math.isfinite(item) or item < 0 for item in (hue, saturation, brightness)):
            raise ValueError("HSV jitter は有限の非負値が必要です")
        value[..., 0] = (value[..., 0] + rng.uniform(-hue, hue) * 180) % 180
        value[..., 1] = np.clip(value[..., 1] * rng.uniform(1 - saturation, 1 + saturation), 0, 255)
        value[..., 2] = np.clip(value[..., 2] * rng.uniform(1 - brightness, 1 + brightness), 0, 255)
        result = cv2.cvtColor(value.astype(np.uint8), cv2.COLOR_HSV2BGR)
    transformed[:, [0, 2]] = np.clip(transformed[:, [0, 2]], 0, width)
    transformed[:, [1, 3]] = np.clip(transformed[:, [1, 3]], 0, height)
    keep = (transformed[:, 2] > transformed[:, 0]) & (transformed[:, 3] > transformed[:, 1])
    if labels is not None:
        return result, transformed[keep], np.asarray(labels)[keep]
    return result, transformed[keep]
