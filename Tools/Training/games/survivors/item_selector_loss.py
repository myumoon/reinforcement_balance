"""ItemSelector の soft-label ranking 複合 loss を定義する。

teacher soft target を forward cross entropy、非 tie pair ranking、Brier の三成分で評価し、
class/reliability weight は完成した row loss へ一度だけ適用する。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch as th
from torch.nn import functional as functional

PAIR_WEIGHT = 0.5
BRIER_WEIGHT = 0.05
WEIGHT_EPSILON = 1e-12


class InvalidBatchError(ValueError):
    """有効 row weight を持たない batch を表す例外。

    silent skip は optimizer step 数と学習率 schedule を歪めるため、caller に batch 不正を
    明示して学習を停止させる。
    """


@dataclass(frozen=True, slots=True)
class ItemSelectorRowLosses:
    """各 row の未加重 CE、pair、Brier、合成 loss を保持する。

    reliability/class weight を含めず、全 component が同じ row weight を共有できる境界にする。
    """

    cross_entropy: th.Tensor
    pairwise: th.Tensor
    brier: th.Tensor
    total: th.Tensor

    @property
    def ce(self) -> th.Tensor:
        """短い診断名で soft cross entropy を返す。

        値は ``cross_entropy`` と同じ tensor で、別計算や別 weighting は行わない。
        """
        return self.cross_entropy

    @property
    def pair(self) -> th.Tensor:
        """短い診断名で pairwise loss を返す。

        値は ``pairwise`` と同じ tensor で、pair reliability は持たない。
        """
        return self.pairwise


def _validate_loss_inputs(
    logits: th.Tensor,
    teacher_soft_target: th.Tensor,
    candidate_mask: th.Tensor,
) -> None:
    """三つの row tensor の shape、dtype、有限性を検証する。

    valid target mass がない行と valid logit の NaN/Inf を loss 計算前に拒否する。
    """
    if logits.ndim != 2 or teacher_soft_target.shape != logits.shape:
        raise ValueError("logits and teacher_soft_target must have shape [B, Nmax]")
    if candidate_mask.shape != logits.shape or candidate_mask.dtype != th.bool:
        raise ValueError("candidate_mask must be bool with shape [B, Nmax]")
    if not th.is_floating_point(logits) or not th.is_floating_point(teacher_soft_target):
        raise ValueError("logits and teacher_soft_target must be floating tensors")
    if not candidate_mask.any(dim=1).all():
        raise InvalidBatchError("each row must contain a valid candidate")
    valid_logits = logits[candidate_mask]
    if not th.isfinite(valid_logits).all():
        raise InvalidBatchError("valid candidate logits must be finite")
    if not th.isfinite(teacher_soft_target).all() or (teacher_soft_target < 0).any():
        raise InvalidBatchError("teacher_soft_target must be finite and non-negative")


def _normalized_target(
    teacher_soft_target: th.Tensor,
    candidate_mask: th.Tensor,
) -> th.Tensor:
    """mask 外をゼロにし、valid teacher mass を row ごとに再正規化する。

    dataset target を hard outcome へ変換せず、soft probability の相対比をそのまま使う。
    """
    masked_target = teacher_soft_target.masked_fill(~candidate_mask, 0.0)
    mass = masked_target.sum(dim=1, keepdim=True)
    if (mass <= WEIGHT_EPSILON).any():
        raise InvalidBatchError("teacher target has no valid probability mass")
    return masked_target / mass


def item_selector_row_losses(
    logits: th.Tensor,
    teacher_soft_target: th.Tensor,
    candidate_mask: th.Tensor,
    *,
    tie_epsilon: float = 1e-12,
) -> ItemSelectorRowLosses:
    """CE、valid non-tie pair mean、soft Brier と固定合成値を返す。

    pair は各 unordered pair を一回だけ列挙し、teacher probability が高い側を winner とする。
    """
    _validate_loss_inputs(logits, teacher_soft_target, candidate_mask)
    if not isinstance(tie_epsilon, (int, float)) or tie_epsilon < 0.0:
        raise ValueError("tie_epsilon must be non-negative")
    target = _normalized_target(teacher_soft_target, candidate_mask)
    masked_logits = logits.masked_fill(~candidate_mask, -th.inf)
    log_probabilities = functional.log_softmax(masked_logits, dim=1)
    safe_log_probabilities = log_probabilities.masked_fill(~candidate_mask, 0.0)
    cross_entropy = -(target * safe_log_probabilities).sum(dim=1)
    probabilities = functional.softmax(masked_logits, dim=1)
    valid_count = candidate_mask.sum(dim=1).to(dtype=logits.dtype)
    brier = (((probabilities - target) ** 2) * candidate_mask).sum(dim=1) / valid_count

    pair_losses: list[th.Tensor] = []
    for row in range(logits.shape[0]):
        valid_indices = candidate_mask[row].nonzero(as_tuple=False).flatten()
        if valid_indices.numel() < 2:
            pair_losses.append(logits[row, valid_indices].sum() * 0.0)
            continue
        pair_indices = th.combinations(valid_indices, r=2)
        left = pair_indices[:, 0]
        right = pair_indices[:, 1]
        target_difference = target[row, left] - target[row, right]
        non_tie = target_difference.abs() > float(tie_epsilon)
        if not non_tie.any():
            pair_losses.append(logits[row, valid_indices].sum() * 0.0)
            continue
        score_difference = logits[row, left] - logits[row, right]
        winner_difference = th.where(
            target_difference > 0.0,
            score_difference,
            -score_difference,
        )
        pair_losses.append(functional.softplus(-winner_difference[non_tie]).mean())
    pairwise = th.stack(pair_losses)
    total = cross_entropy + PAIR_WEIGHT * pairwise + BRIER_WEIGHT * brier
    return ItemSelectorRowLosses(
        cross_entropy=cross_entropy,
        pairwise=pairwise,
        brier=brier,
        total=total,
    )


def item_selector_loss(
    logits: th.Tensor,
    teacher_soft_target: th.Tensor,
    candidate_mask: th.Tensor,
    *,
    reliability_weight: th.Tensor,
    class_weight: th.Tensor,
    tie_epsilon: float = 1e-12,
) -> th.Tensor:
    """clipped row weight による正規化加重平均 loss を返す。

    ``w_i=clip(c_i*r_i,0,4)`` を完成済み row loss に一回だけ掛け、分母ゼロは例外にする。
    """
    components = item_selector_row_losses(
        logits,
        teacher_soft_target,
        candidate_mask,
        tie_epsilon=tie_epsilon,
    )
    batch_size = logits.shape[0]
    if reliability_weight.shape != (batch_size,) or class_weight.shape != (batch_size,):
        raise ValueError("row weights must have shape [B]")
    if not th.isfinite(reliability_weight).all() or not th.isfinite(class_weight).all():
        raise InvalidBatchError("row weights must be finite")
    weights = (class_weight * reliability_weight).clamp(min=0.0, max=4.0)
    denominator = weights.sum()
    if float(denominator.detach().cpu()) <= WEIGHT_EPSILON:
        raise InvalidBatchError("batch row weight sum must be positive")
    return (weights * components.total).sum() / denominator
