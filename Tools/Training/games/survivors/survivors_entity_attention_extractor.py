import torch
import torch.nn as nn
from base.entity_attention_extractor import EntityAttentionExtractor

# 708次元 obs スキーマ（新）の方向別密度特徴セグメント
# 旧スキーマ（gem_nearest_dist_16dir 等）と新スキーマ（gem_density_all_16dir）の両方に対応
_GLOBAL_KEYS_NEW = [
    "enemy_nearest_dist_16dir",
    "enemy_density_near_16dir",
    "enemy_density_mid_16dir",
    "gem_density_all_16dir",        # 新: 全ジェム 16方向×3
    "red_green_gem_density_16dir",  # 新: Red+Green ジェム 16方向×3
]

# 後方互換: 旧スキーマ名（gem_nearest_dist_16dir 等が存在する場合に使用）
_GLOBAL_KEYS_LEGACY = [
    "enemy_nearest_dist_16dir",
    "enemy_density_near_16dir",
    "enemy_density_mid_16dir",
    "gem_nearest_dist_16dir",
    "gem_density_near_16dir",
    "gem_density_mid_16dir",
]

# 新スキーマのジェム相対位置セグメント（この3つを結合して item として扱う）
_GEM_REL_POS_KEYS_NEW = ["red_gem_rel_pos", "green_gem_rel_pos", "blue_gem_rel_pos"]
_ITEM_KEY_LEGACY = "gem_rel_pos"


class SurvivorsEntityAttentionExtractor(EntityAttentionExtractor):
    """Survivors用エンティティアテンション抽出器（708次元 obs スキーマ対応）。

    基底クラスに加えて、方向別密度/最近傍距離のセグメントを
    global_feats として combined に結合する。

    新スキーマ（708次元）と旧スキーマの両方に自動対応:
    - obs_schema に red_gem_rel_pos が存在する場合: 新スキーマとして処理
      新スキーマでは red/green/blue の3セグメントを結合して item として扱い、
      gem_pickup_radius（1次元スカラー）は結合対象から除外する。
    - 存在しない場合: 旧スキーマ（gem_rel_pos）として処理
    """

    def __init__(self, observation_space, features_dim=128, offsets=None, obs_schema=None):
        offsets = offsets or {}
        schema_map = {s["name"]: s["dim"] for s in (obs_schema or [])}

        # 新旧スキーマを自動判別
        self._is_new_schema = "red_gem_rel_pos" in offsets
        global_keys = _GLOBAL_KEYS_NEW if self._is_new_schema else _GLOBAL_KEYS_LEGACY

        if self._is_new_schema:
            # 新スキーマ: item_key は red_gem_rel_pos を基準とし、
            # _split_obs でオーバーライドして3セグメントを結合する
            item_key = "red_gem_rel_pos"
        else:
            item_key = _ITEM_KEY_LEGACY

        super().__init__(
            observation_space,
            features_dim=features_dim,
            offsets=offsets,
            item_key=item_key,
            use_polar=True,
            enemy_scalar_keys=["enemy_type", "enemy_hp"],
        )

        if self._is_new_schema:
            # 新スキーマ: red/green/blue gem の各セグメントスライスを保存し、
            # _split_obs でこれらを結合して item を構成する
            self._gem_slices: list[tuple[int, int]] = []
            total_gem_dim = 0
            for key in _GEM_REL_POS_KEYS_NEW:
                if key in offsets:
                    dim = schema_map.get(key, 0)
                    if dim > 0:
                        start = offsets[key]
                        self._gem_slices.append((start, start + dim))
                        total_gem_dim += dim
            self._num_items = total_gem_dim // 2  # 2次元（dx,dy）ずつ

        # global セグメントの (start, end) リストを構築（自動判別済みの global_keys を使用）
        self._global_slices: list[tuple[int, int]] = []
        global_dim = 0
        for key in global_keys:
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

    def _split_obs(self, obs: torch.Tensor, B: int):
        """obs をセグメントに分割して返す。

        新スキーマでは red/green/blue gem セグメントを結合して item を構成する。
        gem_pickup_radius（1次元スカラー）は item に含めない。
        旧スキーマは基底クラスの _split_obs に委譲する。
        """
        if not self._is_new_schema:
            return super()._split_obs(obs, B)

        # 新スキーマ: red/green/blue gem セグメントを結合
        gem_parts = [obs[:, s:e] for s, e in self._gem_slices]
        if gem_parts:
            gem_concat = torch.cat(gem_parts, dim=-1)  # [B, total_gem_dim]
        else:
            gem_concat = torch.zeros(B, 0, device=obs.device)
        items = gem_concat.reshape(B, self._num_items, 2)

        e_pos = obs[:, self._enemy_r_i:self._enemy_v_i].reshape(B, self._num_enemies, 2)
        e_vel = obs[:, self._enemy_v_i:self._enemy_v_end].reshape(B, self._num_enemies, 2)
        e_scalars = [
            obs[:, s:s + self._num_enemies].reshape(B, self._num_enemies, 1)
            for s in self._enemy_scalar_starts
        ]
        return items, e_pos, e_vel, e_scalars

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
