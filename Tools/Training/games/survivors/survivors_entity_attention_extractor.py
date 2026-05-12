import torch
import torch.nn as nn
from base.entity_attention_extractor import EntityAttentionExtractor

_GLOBAL_KEYS = [
    "enemy_nearest_dist_16dir",
    "enemy_density_near_16dir",
    "enemy_density_mid_16dir",
    "gem_nearest_dist_16dir",
    "gem_density_near_16dir",
    "gem_density_mid_16dir",
]


class SurvivorsEntityAttentionExtractor(EntityAttentionExtractor):
    """Survivors用エンティティアテンション抽出器。

    基底クラスに加えて、方向別密度/最近傍距離の6セグメント（計96次元）を
    global_feats として combined に結合する。
    """

    def __init__(self, observation_space, features_dim=128, offsets=None, obs_schema=None):
        super().__init__(
            observation_space,
            features_dim=features_dim,
            offsets=offsets,
            item_key="gem_rel_pos",
            use_polar=True,
            enemy_scalar_keys=["enemy_type", "enemy_hp"],
        )
        offsets = offsets or {}
        schema_map = {s["name"]: s["dim"] for s in (obs_schema or [])}

        # global セグメントの (start, end) リストを構築
        self._global_slices: list[tuple[int, int]] = []
        global_dim = 0
        for key in _GLOBAL_KEYS:
            if key not in offsets:
                print(f"[WARN] SurvivorsEntityAttentionExtractor: '{key}' が obs_schema に見つかりません。global_feats をスキップします。")
                self._global_slices = []
                global_dim = 0
                break
            dim = schema_map.get(key, 0)
            if dim == 0:
                print(f"[WARN] SurvivorsEntityAttentionExtractor: '{key}' の dim が 0 です。global_feats をスキップします。")
                self._global_slices = []
                global_dim = 0
                break
            start = offsets[key]
            self._global_slices.append((start, start + dim))
            global_dim += dim

        # global_dim を考慮した final 層で差し替え
        e = self._EMBED_DIM
        self.final = nn.Sequential(
            nn.Linear(self._self_dim + global_dim + e + e, features_dim),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        B = obs.shape[0]

        self_info = obs[:, :self._self_dim]

        global_parts = [obs[:, s:e] for s, e in self._global_slices]
        global_feats = torch.cat(global_parts, dim=-1) if global_parts else torch.zeros(B, 0, device=obs.device)

        items, e_pos, e_vel, e_scalars = self._split_obs(obs, B)

        item_dist  = torch.norm(items, dim=-1)
        enemy_dist = torch.norm(e_pos,  dim=-1)

        if self.use_polar:
            items = self._to_polar(items)
            e_pos = self._to_polar(e_pos)
            e_vel = self._to_polar(e_vel)

        enemies = torch.cat([e_pos, e_vel] + e_scalars, dim=-1)

        item_enc  = self.coin_encoder(items)
        enemy_enc = self.enemy_encoder(enemies)

        item_agg  = self._attend(item_enc,  self.coin_query,  self.dist_alpha * item_dist)
        enemy_agg = self._attend(enemy_enc, self.enemy_query, self.dist_alpha * enemy_dist)

        combined = torch.cat([self_info, global_feats, item_agg, enemy_agg], dim=-1)
        return self.final(combined)
