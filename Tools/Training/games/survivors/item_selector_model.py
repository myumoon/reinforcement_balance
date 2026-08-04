"""Survivors ItemSelector の候補順に依存しない共有 scorer を定義する。

一つの context 表現を各 card feature と結合し、同じ MLP で個別に採点する。候補 index を
入力へ加えないため、card の並び替えは score の並び替えだけを引き起こす。
"""

from __future__ import annotations

import torch as th
from torch import nn


class ItemSelector(nn.Module):
    """context と可変候補集合から choose-card logits を返す軽量 network。

    context/candidate を別々に埋め込み、全候補へ shared scorer を適用する。mask は padding
    slot の score を ``-inf`` にして softmax や argmax から除外する。
    """

    def __init__(self, context_dim: int, candidate_dim: int) -> None:
        """固定 128/64、64/32、64/1 の三段構造を初期化する。

        入力次元は feature encoder が確定した正の整数だけを受理する。
        """
        super().__init__()
        if type(context_dim) is not int or context_dim <= 0:
            raise ValueError("context_dim must be a positive integer")
        if type(candidate_dim) is not int or candidate_dim <= 0:
            raise ValueError("candidate_dim must be a positive integer")
        self.context_dim = context_dim
        self.candidate_dim = candidate_dim
        self.context_mlp = nn.Sequential(
            nn.Linear(context_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.candidate_mlp = nn.Sequential(
            nn.Linear(candidate_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.scorer = nn.Sequential(
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        context_features: th.Tensor,
        candidate_features: th.Tensor,
        candidate_mask: th.Tensor,
    ) -> th.Tensor:
        """``[B,C]`` と ``[B,N,F]`` から masked ``[B,N]`` logits を返す。

        mask は score 計算後に適用し、False slot を有限値として downstream へ公開しない。
        """
        if context_features.ndim != 2:
            raise ValueError("context_features must have shape [B, C]")
        if candidate_features.ndim != 3:
            raise ValueError("candidate_features must have shape [B, Nmax, F]")
        if candidate_mask.ndim != 2 or candidate_mask.dtype != th.bool:
            raise ValueError("candidate_mask must be bool with shape [B, Nmax]")
        batch, candidate_count, feature_dim = candidate_features.shape
        if context_features.shape != (batch, self.context_dim):
            raise ValueError("context feature shape does not match ItemSelector")
        if feature_dim != self.candidate_dim:
            raise ValueError("candidate feature shape does not match ItemSelector")
        if candidate_mask.shape != (batch, candidate_count):
            raise ValueError("candidate_mask shape does not match candidates")

        context_embedding = self.context_mlp(context_features)
        candidate_embedding = self.candidate_mlp(candidate_features)
        expanded_context = context_embedding.unsqueeze(1).expand(
            -1,
            candidate_count,
            -1,
        )
        scorer_input = th.cat((expanded_context, candidate_embedding), dim=-1)
        logits = self.scorer(scorer_input).squeeze(-1)
        return logits.masked_fill(~candidate_mask, -th.inf)

