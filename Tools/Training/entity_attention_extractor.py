"""エンティティアテンション特徴抽出器。

コインと敵をそれぞれ個別エンティティとして処理し、
学習されたアテンション（注目度）で重み付け集約する。
MlpPolicy の「310次元一括処理」より obs の構造を活かした表現学習が可能。
"""

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class EntityAttentionExtractor(BaseFeaturesExtractor):
    """コインゲーム用エンティティアテンション特徴抽出器。

    obs レイアウト:
      obs[0        : coin_i]      = self_info    (通常10次元)
      obs[coin_i   : enemy_r_i]   = coin_rel_pos (100枚 × 2: dx,dy)
      obs[enemy_r_i: enemy_v_i]   = enemy_rel_pos (20体 × 2: dx,dy)
      obs[enemy_v_i: enemy_t_i]   = enemy_vel     (20体 × 2: vx,vy)
      obs[enemy_t_i:]             = enemy_type    (20体 × 1: 0.0/0.5/1.0)

    use_polar=True のとき、コイン・敵の位置と速度ベクトルを (dx,dy) → (r, θ) に変換する。
    距離 r が直接特徴として現れるため、アテンションが距離ベースの優先度を学びやすくなる。

    Args:
        observation_space: gymnasium の観測空間（SB3 が渡す）
        features_dim: 出力特徴量の次元数（default: 128）
        offsets: obs_schema から得たセグメント開始インデックス dict
        use_polar: True のとき位置・速度ベクトルを極座標に変換（default: True）
        dist_alpha: 距離バイアスの強さ。スコアから alpha * r を減算することで
                    近いエンティティほど高スコアになる帰納バイアスを与える（default: 1.0）
    """

    _EMBED_DIM = 32

    def __init__(self, observation_space, features_dim: int = 128,
                 offsets: dict | None = None, use_polar: bool = True,
                 dist_alpha: float = 1.0):
        super().__init__(observation_space, features_dim)
        offsets = offsets or {}
        self.use_polar = use_polar
        self.dist_alpha = dist_alpha

        # セグメントのインデックス境界
        self._coin_i    = offsets.get("coin_rel_pos", 10)
        self._enemy_r_i = offsets.get("enemy_rel_pos", self._coin_i + 200)
        self._enemy_v_i = offsets.get("enemy_vel",     self._enemy_r_i + 40)
        self._enemy_t_i = offsets.get("enemy_type",    self._enemy_v_i + 40)

        self._self_dim    = self._coin_i
        self._num_coins   = (self._enemy_r_i - self._coin_i) // 2
        self._num_enemies = (self._enemy_v_i - self._enemy_r_i) // 2

        e = self._EMBED_DIM

        # コインエンコーダ: 2 (dx,dy) → embed_dim（エンティティ間で重み共有）
        self.coin_encoder = nn.Sequential(
            nn.Linear(2, e), nn.ReLU(),
            nn.Linear(e, e), nn.ReLU(),
        )

        # 敵エンコーダ: 5 (dx,dy,vx,vy,type) → embed_dim（エンティティ間で重み共有）
        self.enemy_encoder = nn.Sequential(
            nn.Linear(5, e), nn.ReLU(),
            nn.Linear(e, e), nn.ReLU(),
        )

        # アテンションクエリベクトル（学習パラメータ）
        self.coin_query  = nn.Parameter(torch.randn(e))
        self.enemy_query = nn.Parameter(torch.randn(e))

        # 集約後の線形層: (self_dim + embed*2) → features_dim
        self.final = nn.Sequential(
            nn.Linear(self._self_dim + e + e, features_dim),
            nn.ReLU(),
        )

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
        scores = (enc * query).sum(-1) / scale       # [B, N]
        if dist_bias is not None:
            scores = scores - dist_bias              # 近いエンティティ（小さい r）ほど高スコア
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # [B, N, 1]
        return (enc * weights).sum(dim=1)            # [B, embed_dim]

    @staticmethod
    def _attend_weights(enc: torch.Tensor, query: torch.Tensor,
                        dist_bias: torch.Tensor | None = None) -> torch.Tensor:
        """アテンション重みのみを返す（可視化・デバッグ用）。

        Returns:
            [B, N]  ∈ (0, 1),  sum over N == 1
        """
        scale = enc.shape[-1] ** 0.5
        scores = (enc * query).sum(-1) / scale  # [B, N]
        if dist_bias is not None:
            scores = scores - dist_bias
        return torch.softmax(scores, dim=1)

    def get_attention_weights(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """コイン・敵それぞれのアテンション重みを返す（可視化・デバッグ用）。

        forward() と同じ前処理を行い、softmax 後の重みを返す。
        VecNormalize 正規化済み obs をそのまま渡してよい。

        Args:
            obs: [B, obs_dim]  (VecNormalize 正規化済み obs)
        Returns:
            coin_weights:  [B, num_coins]   slot 0 が最近傍
            enemy_weights: [B, num_enemies] slot 0 が最近傍
        """
        B = obs.shape[0]
        coins  = obs[:, self._coin_i:self._enemy_r_i].reshape(B, self._num_coins, 2)
        e_pos  = obs[:, self._enemy_r_i:self._enemy_v_i].reshape(B, self._num_enemies, 2)
        e_vel  = obs[:, self._enemy_v_i:self._enemy_t_i].reshape(B, self._num_enemies, 2)
        e_type = obs[:, self._enemy_t_i:].reshape(B, self._num_enemies, 1)

        coin_dist  = torch.norm(coins, dim=-1)
        enemy_dist = torch.norm(e_pos,  dim=-1)

        if self.use_polar:
            coins = self._to_polar(coins)
            e_pos = self._to_polar(e_pos)
            e_vel = self._to_polar(e_vel)

        coin_bias  = self.dist_alpha * coin_dist
        enemy_bias = self.dist_alpha * enemy_dist

        enemies = torch.cat([e_pos, e_vel, e_type], dim=-1)

        coin_enc  = self.coin_encoder(coins)
        enemy_enc = self.enemy_encoder(enemies)

        return (
            self._attend_weights(coin_enc,  self.coin_query,  coin_bias),
            self._attend_weights(enemy_enc, self.enemy_query, enemy_bias),
        )

    @staticmethod
    def _to_polar(xy: torch.Tensor) -> torch.Tensor:
        """直交座標 (dx, dy) を極座標 (r, θ) に変換する。

        距離 r が直接の特徴になるため、アテンション学習で距離ベースの
        優先度付けが容易になる。

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

        # セグメント分割
        self_info = obs[:, :self._self_dim]
        coins = obs[:, self._coin_i:self._enemy_r_i].reshape(B, self._num_coins, 2)
        e_pos  = obs[:, self._enemy_r_i:self._enemy_v_i].reshape(B, self._num_enemies, 2)
        e_vel  = obs[:, self._enemy_v_i:self._enemy_t_i].reshape(B, self._num_enemies, 2)
        e_type = obs[:, self._enemy_t_i:].reshape(B, self._num_enemies, 1)

        # 極座標変換前（直交座標のまま）に距離を計算
        coin_dist  = torch.norm(coins, dim=-1)   # [B, num_coins]
        enemy_dist = torch.norm(e_pos,  dim=-1)  # [B, num_enemies]

        if self.use_polar:
            coins = self._to_polar(coins)
            e_pos = self._to_polar(e_pos)
            e_vel = self._to_polar(e_vel)

        coin_bias  = self.dist_alpha * coin_dist   # [B, num_coins]
        enemy_bias = self.dist_alpha * enemy_dist  # [B, num_enemies]

        enemies = torch.cat([e_pos, e_vel, e_type], dim=-1)  # [B, 20, 5]

        # エンティティ単位でエンコード
        coin_enc  = self.coin_encoder(coins)    # [B, 100, 32]
        enemy_enc = self.enemy_encoder(enemies) # [B,  20, 32]

        # アテンションで集約（距離バイアス付き）
        coin_agg  = self._attend(coin_enc,  self.coin_query,  coin_bias)   # [B, 32]
        enemy_agg = self._attend(enemy_enc, self.enemy_query, enemy_bias)  # [B, 32]

        combined = torch.cat([self_info, coin_agg, enemy_agg], dim=-1)
        return self.final(combined)
