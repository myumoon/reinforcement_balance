"""ItemSelector の soft-label、pair、Brier、row weighting loss を検証する。

各 component を解析値と比較し、reliability/class weight が row 集約で一度だけ適用される
ことを固定する。
"""

from __future__ import annotations

import pytest
import torch as th

from games.survivors.item_selector_loss import (
    InvalidBatchError,
    item_selector_loss,
    item_selector_row_losses,
)


def test_soft_cross_entropy_is_forward_direction_not_reverse_kl() -> None:
    """teacher を第一引数とする forward soft CE の解析値を固定する。

    student 分布で teacher の対数を平均する reverse KL とは異なる非対称 fixture を使う。
    """
    probabilities = th.tensor([[0.6, 0.4]], dtype=th.float64)
    logits = probabilities.log()
    target = th.tensor([[0.8, 0.2]], dtype=th.float64)
    mask = th.ones_like(target, dtype=th.bool)

    components = item_selector_row_losses(logits, target, mask)
    expected_forward_ce = -(target * probabilities.log()).sum(dim=1)
    reverse_cross_entropy = -(probabilities * target.log()).sum(dim=1)

    th.testing.assert_close(components.cross_entropy, expected_forward_ce)
    assert not th.allclose(components.cross_entropy, reverse_cross_entropy)


def test_pair_is_mean_of_valid_non_ties_and_all_tie_is_zero() -> None:
    """valid な非 tie unordered pair を一度ずつ平均する。

    teacher probability が同じ pair は除外し、全 tie 行は differentiable zero を返す。
    """
    logits = th.tensor([[2.0, 0.5, -1.0], [0.3, -0.2, 1.0]])
    targets = th.tensor([[0.6, 0.3, 0.1], [1.0, 1.0, 1.0]])
    mask = th.ones_like(targets, dtype=th.bool)

    pair = item_selector_row_losses(logits, targets, mask).pairwise
    expected = th.stack(
        [
            th.nn.functional.softplus(-(logits[0, 0] - logits[0, 1])),
            th.nn.functional.softplus(-(logits[0, 0] - logits[0, 2])),
            th.nn.functional.softplus(-(logits[0, 1] - logits[0, 2])),
        ]
    ).mean()

    th.testing.assert_close(pair[0], expected)
    th.testing.assert_close(pair[1], th.tensor(0.0))

    one_valid = item_selector_row_losses(
        th.tensor([[0.5, -th.inf]]),
        th.tensor([[1.0, 0.0]]),
        th.tensor([[True, False]]),
    )
    th.testing.assert_close(one_valid.pairwise, th.zeros(1))


def test_brier_uses_masked_renormalized_soft_target_and_row_formula() -> None:
    """mask 後の teacher soft target と student probability の valid MSE を確認する。

    hard argmax outcome や Nmax 全体の padding 平均を Brier に使わない。
    """
    logits = th.tensor([[0.0, 1.0, 99.0]])
    target = th.tensor([[0.2, 0.3, 0.5]])
    mask = th.tensor([[True, True, False]])

    components = item_selector_row_losses(logits, target, mask)
    normalized_target = th.tensor([0.4, 0.6])
    student = th.softmax(logits[0, :2], dim=0)
    expected_brier = ((student - normalized_target) ** 2).mean()
    expected_row = components.cross_entropy + 0.5 * components.pairwise + 0.05 * expected_brier

    th.testing.assert_close(components.brier[0], expected_brier)
    th.testing.assert_close(components.total[0], expected_row[0])


def test_row_weight_is_clipped_and_applied_once_with_zero_guard() -> None:
    """``clip(class*reliability)`` を row total へ一度だけ掛ける。

    全 row weight がゼロの batch は skip せず InvalidBatchError とする。
    """
    logits = th.tensor([[2.0, 0.0], [0.1, 0.2]])
    target = th.tensor([[1.0, 0.0], [0.25, 0.75]])
    mask = th.ones_like(target, dtype=th.bool)
    reliability = th.tensor([0.5, 1.0])
    class_weight = th.tensor([10.0, 2.0])
    components = item_selector_row_losses(logits, target, mask)

    loss = item_selector_loss(
        logits,
        target,
        mask,
        reliability_weight=reliability,
        class_weight=class_weight,
    )
    weights = th.tensor([4.0, 2.0])
    expected = (weights * components.total).sum() / weights.sum()
    th.testing.assert_close(loss, expected)

    with pytest.raises(InvalidBatchError):
        item_selector_loss(
            logits,
            target,
            mask,
            reliability_weight=th.zeros(2),
            class_weight=class_weight,
        )
