"""エンティティアテンション特徴抽出器。

アイテムと敵をそれぞれ個別エンティティとして処理し、
学習されたアテンション（注目度）で重み付け集約する。
MlpPolicy の一括処理より obs の構造を活かした表現学習が可能。

ゲームごとの obs 構造の違いは item_key / enemy_scalar_keys で指定する:
  CoinGame   : item_key="coin_rel_pos", enemy_scalar_keys=["enemy_type"]
  Survivors  : item_key="item_rel_pos", enemy_scalar_keys=["enemy_type", "enemy_hp"]
"""

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class EntityAttentionExtractor(BaseFeaturesExtractor):
    """ゲーム汎用エンティティアテンション特徴抽出器。

    obs レイアウト（共通構造）:
      obs[0         : item_i]    = self_info   （プレイヤー状態など）
      obs[item_i    : enemy_r_i] = item_rel_pos（N_items × 2: dx,dy）
      obs[enemy_r_i : enemy_v_i] = enemy_rel_pos（N_enemies × 2: dx,dy）
      obs[enemy_v_i : v_end]     = enemy_vel   （N_enemies × 2: vx,vy）
      obs[scalar_i  : ...]       = enemy_scalar × len(enemy_scalar_keys)
                                    各 N_enemies × 1

    use_polar=True のとき、アイテム・敵の位置と速度を (dx,dy) → (r, θ) に変換する。

    Args:
        observation_space: gymnasium の観測空間（SB3 が渡す）
        features_dim: 出力特徴量の次元数（default: 128）
        offsets: obs_schema から得たセグメント開始インデックス dict
        use_polar: True のとき位置・速度ベクトルを極座標に変換（default: True）
        dist_alpha: 距離バイアスの強さ（default: 1.0）
        item_key: アイテム相対位置セグメントのキー名（default: "coin_rel_pos"）
        enemy_scalar_keys: 敵のスカラー特徴セグメントキーのリスト
                           （default: ["enemy_type"]）
                           例: Survivors では ["enemy_type", "enemy_hp"]
    """

    _EMBED_DIM = 32

    def __init__(
        self,
        observation_space,
        features_dim: int = 128,
        offsets: dict | None = None,
        use_polar: bool = True,
        dist_alpha: float = 1.0,
        item_key: str = "coin_rel_pos",
        enemy_scalar_keys: list[str] | None = None,
    ):
        super().__init__(observation_space, features_dim)
        offsets = offsets or {}
        self.use_polar = use_polar
        self.dist_alpha = dist_alpha

        if enemy_scalar_keys is None:
            enemy_scalar_keys = ["enemy_type"]
        self._enemy_scalar_keys = list(enemy_scalar_keys)

        # ---- セグメント境界の解決 ----
        self._coin_i    = offsets.get(item_key, 10)
        self._enemy_r_i = offsets.get("enemy_rel_pos", self._coin_i + 200)
        self._enemy_v_i = offsets.get("enemy_vel",     self._enemy_r_i + 40)

        self._self_dim    = self._coin_i
        self._num_items   = (self._enemy_r_i - self._coin_i) // 2
        self._num_enemies = (self._enemy_v_i - self._enemy_r_i) // 2
        self._enemy_v_end = self._enemy_v_i + self._num_enemies * 2

        # 敵スカラー特徴の開始インデックスをゲーム設定から取得
        default_start = self._enemy_v_end
        self._enemy_scalar_starts: list[int] = []
        for key in self._enemy_scalar_keys:
            start = offsets.get(key, default_start)
            self._enemy_scalar_starts.append(start)
            default_start = start + self._num_enemies  # 次のセグメントのデフォルト起点

        # ---- ネットワーク定義 ----
        e = self._EMBED_DIM
        # enemy encoder 入力: 位置2 + 速度2 + スカラー×n
        enemy_in_dim = 4 + len(self._enemy_scalar_keys)

        self.coin_encoder = nn.Sequential(
            nn.Linear(2, e), nn.ReLU(),
            nn.Linear(e, e), nn.ReLU(),
        )
        self.enemy_encoder = nn.Sequential(
            nn.Linear(enemy_in_dim, e), nn.ReLU(),
            nn.Linear(e, e), nn.ReLU(),
        )

        self.coin_query  = nn.Parameter(torch.randn(e))
        self.enemy_query = nn.Parameter(torch.randn(e))

        self.final = nn.Sequential(
            nn.Linear(self._self_dim + e + e, features_dim),
            nn.ReLU(),
        )

    # ---- アテンション ----

    @staticmethod
    def _attend(enc: torch.Tensor, query: torch.Tensor,
                dist_bias: torch.Tensor | None = None) -> torch.Tensor:
        """スケール済みドット積アテンションで entity を集約する。

        Args:
            enc:       [B, N, embed_dim]
            query:     [embed_dim]
            dist_bias: [B, N] 距離 * alpha（近いほどスコアを上げるため減算する）
        Returns:
            [B, embed_dim]
        """
        scale = enc.shape[-1] ** 0.5
        scores = (enc * query).sum(-1) / scale
        if dist_bias is not None:
            scores = scores - dist_bias
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (enc * weights).sum(dim=1)

    @staticmethod
    def _attend_weights(enc: torch.Tensor, query: torch.Tensor,
                        dist_bias: torch.Tensor | None = None) -> torch.Tensor:
        """アテンション重みのみを返す（可視化・デバッグ用）。

        Returns:
            [B, N]  ∈ (0, 1),  sum over N == 1
        """
        scale = enc.shape[-1] ** 0.5
        scores = (enc * query).sum(-1) / scale
        if dist_bias is not None:
            scores = scores - dist_bias
        return torch.softmax(scores, dim=1)

    # ---- デバッグ用 ----

    def get_attention_weights(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """アイテム・敵それぞれのアテンション重みを返す（可視化・デバッグ用）。

        Args:
            obs: [B, obs_dim]  (VecNormalize 正規化済み obs)
        Returns:
            item_weights:  [B, num_items]   slot 0 が最近傍
            enemy_weights: [B, num_enemies] slot 0 が最近傍
        """
        B = obs.shape[0]
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

        return (
            self._attend_weights(item_enc,  self.coin_query,
                                 self.dist_alpha * item_dist),
            self._attend_weights(enemy_enc, self.enemy_query,
                                 self.dist_alpha * enemy_dist),
        )

    # ---- 内部ユーティリティ ----

    def _split_obs(self, obs: torch.Tensor, B: int):
        """obs をセグメントに分割して返す。"""
        items = obs[:, self._coin_i:self._enemy_r_i].reshape(B, self._num_items, 2)
        e_pos = obs[:, self._enemy_r_i:self._enemy_v_i].reshape(B, self._num_enemies, 2)
        e_vel = obs[:, self._enemy_v_i:self._enemy_v_end].reshape(B, self._num_enemies, 2)
        e_scalars = [
            obs[:, s:s + self._num_enemies].reshape(B, self._num_enemies, 1)
            for s in self._enemy_scalar_starts
        ]
        return items, e_pos, e_vel, e_scalars

    @staticmethod
    def _to_polar(xy: torch.Tensor) -> torch.Tensor:
        """直交座標 (dx, dy) を極座標 (r, θ) に変換する。

        Args:
            xy: [B, N, 2]  (dx, dy)
        Returns:
            [B, N, 2]  (r ∈ [0, √2],  θ ∈ [-π, π])
        """
        r     = torch.norm(xy, dim=-1, keepdim=True)
        theta = torch.atan2(xy[..., 1:2], xy[..., 0:1])
        return torch.cat([r, theta], dim=-1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        B = obs.shape[0]

        self_info = obs[:, :self._self_dim]
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

        item_agg  = self._attend(item_enc,  self.coin_query,
                                 self.dist_alpha * item_dist)
        enemy_agg = self._attend(enemy_enc, self.enemy_query,
                                 self.dist_alpha * enemy_dist)

        combined = torch.cat([self_info, item_agg, enemy_agg], dim=-1)
        return self.final(combined)
