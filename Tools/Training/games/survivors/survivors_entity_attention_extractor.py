import torch
import torch.nn as nn
from base.entity_attention_extractor import EntityAttentionExtractor

# v794 obs スキーマの方向別密度特徴セグメント
# 旧スキーマ（gem_nearest_dist_16dir 等）と新スキーマ（gem_density_all_16dir）の両方に対応
_GLOBAL_KEYS_NEW = [
    "enemy_nearest_dist_16dir",
    "enemy_density_near_16dir",
    "enemy_density_mid_16dir",
    "gem_density_all_16dir",        # 全Gemの方向密度（フィールド全体、Gem Attentionの34個と非重複）
    "red_green_gem_density_16dir",  # Red+Green Gemの方向密度（フィールド全体）
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

# v794 追加セグメント: 武器情報・フィールドアイテム（グローバル特徴量として追加）
# self_info(obs[0:coin_i]) の範囲外のため global_slices に個別追加する。
# スキーマに存在しないキーはスキップして後方互換を維持する。
_EXTRA_GLOBAL_KEYS = [
    "gem_pickup_radius",          # 1  dim : Attractorb 等で変動する収集半径（正規化済み）
    "weapon_attack_range_norm",   # 6  dims: GetWeaponEffectiveRange() × 6スロット
    "weapon_is_directional",      # 6  dims: 6スロット
    "weapon_category_onehot",     # 42 dims: 6スロット × 7カテゴリ
    "floor_pickups",              # 24 dims: 8アイテム × (dx,dy,type_norm)
    "special_pickups",            # 9  dims: 3アイテム × (dx,dy,type_norm)
    "destructibles",              # 20 dims: 10アイテム × (dx,dy)
]


class SurvivorsEntityAttentionExtractor(EntityAttentionExtractor):
    """Survivors用エンティティアテンション抽出器（v794 obs スキーマ対応）。

    基底クラスに加えて以下を追加:
    - 方向別密度/最近傍距離セグメントを global_feats として結合（既存）
    - weapon_attack_range_norm / weapon_is_directional / weapon_category_onehot を
      global_feats に追加（v794 追加セグメント）
    - floor_pickups / special_pickups / destructibles を global_feats に追加
    - enemy_frozen を敵スカラー特徴として enemy_encoder に追加
    - projectiles (プレイヤー武器の飛翔体) を専用 attention head で集約

    新スキーマ（v794: 794次元）と旧スキーマの両方に自動対応:
    - obs_schema に red_gem_rel_pos が存在する場合: 新スキーマとして処理
    - 存在しない場合: 旧スキーマ（gem_rel_pos）として処理
    - _EXTRA_GLOBAL_KEYS / projectiles / enemy_frozen は存在する場合のみ追加

    combined ベクトル構成（新スキーマ・全セグメント存在時）:
        self_info (58) + global_proj (64) + gem_agg (32) + enemy_agg (32) + proj_agg (32) = 218
        ※ global_feats (252) は global_proj: Linear(252→64) で圧縮してから結合
    """

    _GLOBAL_PROJ_DIM = 64

    def __init__(self, observation_space, features_dim=128, offsets=None, obs_schema=None):
        offsets = offsets or {}
        schema_map = {s["name"]: s["dim"] for s in (obs_schema or [])}

        # 新旧スキーマを自動判別
        self._is_new_schema = "red_gem_rel_pos" in offsets
        global_keys = _GLOBAL_KEYS_NEW if self._is_new_schema else _GLOBAL_KEYS_LEGACY

        item_key = "red_gem_rel_pos" if self._is_new_schema else _ITEM_KEY_LEGACY

        # enemy_frozen はスキーマに存在する場合のみ追加（後方互換）
        enemy_scalar_keys = ["enemy_type", "enemy_hp"]
        if "enemy_frozen" in offsets:
            enemy_scalar_keys.append("enemy_frozen")

        super().__init__(
            observation_space,
            features_dim=features_dim,
            offsets=offsets,
            item_key=item_key,
            use_polar=True,
            enemy_scalar_keys=enemy_scalar_keys,
        )

        if self._is_new_schema:
            self._gem_slices: list[tuple[int, int]] = []
            total_gem_dim = 0
            for key in _GEM_REL_POS_KEYS_NEW:
                if key in offsets:
                    dim = schema_map.get(key, 0)
                    if dim > 0:
                        start = offsets[key]
                        self._gem_slices.append((start, start + dim))
                        total_gem_dim += dim
            self._num_items = total_gem_dim // 2

        # global セグメント構築（既存キー）
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

        # 追加グローバルセグメント（v794）: スキーマにある分だけ個別追加
        for key in _EXTRA_GLOBAL_KEYS:
            if key in offsets:
                dim = schema_map.get(key, 0)
                if dim > 0:
                    start = offsets[key]
                    self._global_slices.append((start, start + dim))
                    global_dim += dim

        # プロジェクタイル attention head（存在する場合のみ構築）
        e = self._EMBED_DIM
        self._proj_slice: tuple[int, int] | None = None
        proj_agg_dim = 0
        if "projectiles" in offsets:
            proj_total_dim = schema_map.get("projectiles", 0)
            if proj_total_dim > 0:
                proj_start = offsets["projectiles"]
                self._proj_slice = (proj_start, proj_start + proj_total_dim)
                self._num_projectiles = proj_total_dim // 6  # 各プロジェクタイル: (dx,dy,r,vx,vy,warning)
                self.proj_encoder = nn.Sequential(
                    nn.Linear(6, e), nn.ReLU(),
                    nn.Linear(e, e), nn.ReLU(),
                )
                self.proj_query = nn.Parameter(torch.randn(e))
                proj_agg_dim = e

        # global_proj: global_feats を _GLOBAL_PROJ_DIM に圧縮してから combined に結合
        if global_dim > 0:
            self.global_proj = nn.Sequential(
                nn.Linear(global_dim, self._GLOBAL_PROJ_DIM), nn.ReLU()
            )
            global_proj_dim = self._GLOBAL_PROJ_DIM
        else:
            self.global_proj = None
            global_proj_dim = 0

        # final 層を全追加分を含む次元で再構築
        self.final = nn.Sequential(
            nn.Linear(self._self_dim + global_proj_dim + e + e + proj_agg_dim, features_dim),
            nn.ReLU(),
        )

    def _split_obs(self, obs: torch.Tensor, B: int):
        if not self._is_new_schema:
            return super()._split_obs(obs, B)

        gem_parts = [obs[:, s:e] for s, e in self._gem_slices]
        gem_concat = torch.cat(gem_parts, dim=-1) if gem_parts else torch.zeros(B, 0, device=obs.device)
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

        item_pad_mask  = (item_dist  < 1e-3)
        enemy_pad_mask = (enemy_dist < 1e-3)

        if self.use_polar:
            items = self._to_polar(items)
            e_pos = self._to_polar(e_pos)
            e_vel = self._to_polar(e_vel)

        enemies = torch.cat([e_pos, e_vel] + e_scalars, dim=-1)

        item_enc  = self.coin_encoder(items)
        enemy_enc = self.enemy_encoder(enemies)

        item_agg  = self._attend(item_enc,  self.coin_query,  self.dist_alpha * item_dist,  item_pad_mask)
        enemy_agg = self._attend(enemy_enc, self.enemy_query, self.dist_alpha * enemy_dist, enemy_pad_mask)

        if self.global_proj is not None:
            global_feats = self.global_proj(global_feats)

        parts = [self_info, global_feats, item_agg, enemy_agg]

        if self._proj_slice is not None:
            s, e_idx = self._proj_slice
            proj_raw = obs[:, s:e_idx].reshape(B, self._num_projectiles, 6)
            proj_dist = torch.norm(proj_raw[:, :, :2], dim=-1)
            proj_pad_mask = (proj_dist < 1e-3)
            proj_enc  = self.proj_encoder(proj_raw)
            proj_agg  = self._attend(proj_enc, self.proj_query, self.dist_alpha * proj_dist, proj_pad_mask)
            parts.append(proj_agg)

        combined = torch.cat(parts, dim=-1)
        return self.final(combined)
