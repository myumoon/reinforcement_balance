"""ItemSelector の構造、mask、候補順不変性、feature 境界を検証する。

候補ごとに共有 scorer を適用し、padding や教師専用 field が選択 score に混ざらないことを
小さな決定 fixture で固定する。
"""

from __future__ import annotations

import itertools

import torch as th

from games.survivors.item_selector_model import ItemSelector


def test_forward_masks_padding_and_has_registered_small_architecture() -> None:
    """固定 MLP 構造と ``[B, Nmax]`` masked logits を確認する。

    masked slot は有限な scorer 出力を公開せず、常に負の無限大へ置換される。
    """
    model = ItemSelector(context_dim=5, candidate_dim=7)
    context = th.randn(2, 5)
    candidates = th.randn(2, 4, 7)
    mask = th.tensor([[True, True, False, False], [True, False, True, False]])

    logits = model(context, candidates, mask)

    assert logits.shape == (2, 4)
    assert th.isfinite(logits[mask]).all()
    assert th.isneginf(logits[~mask]).all()
    assert [layer.out_features for layer in model.context_mlp if isinstance(layer, th.nn.Linear)] == [128, 64]
    assert [layer.out_features for layer in model.candidate_mlp if isinstance(layer, th.nn.Linear)] == [64, 32]
    assert [layer.out_features for layer in model.scorer if isinstance(layer, th.nn.Linear)] == [64, 1]
    assert sum(parameter.numel() for parameter in model.parameters()) < 100_000


def test_candidate_permutation_preserves_scores_by_item_identity() -> None:
    """全 valid candidate を並べ替えても item 対応 score が変わらないことを確認する。

    scorer は candidate index embedding や順序集約を持たず、同じ candidate feature へ同じ
    context を組み合わせる。
    """
    th.manual_seed(7)
    model = ItemSelector(context_dim=3, candidate_dim=4).eval()
    context = th.randn(1, 3)
    candidates = th.randn(1, 4, 4)
    mask = th.ones(1, 4, dtype=th.bool)
    baseline = model(context, candidates, mask)[0]

    for ordering in itertools.permutations(range(4)):
        permutation = th.tensor(ordering)
        permuted = model(context, candidates[:, permutation], mask[:, permutation])[0]
        restored = th.empty_like(permuted)
        restored[permutation] = permuted
        th.testing.assert_close(restored, baseline)
