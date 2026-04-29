"""コインゲーム専用の EUREKA 設定。"""

import json
import sys

import requests

from eureka_game_config import EurekaGameConfig

# C++ 側の固定定数と一致させること
_ALIVE_REWARD = 0.001
_COIN_REWARD = 5.0


def _fetch_obs_schema(host: str, port: int) -> dict:
    url = f"http://{host}:{port}/obs_schema"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        print(f"[ERROR] UE5 サーバー ({host}:{port}) に接続できませんでした。")
        print(f"[ERROR] UE5 エディタで PIE (▶) を起動してから再実行してください。")
        sys.exit(1)


def _build_obs_layout(schema: dict) -> tuple[str, dict[str, int]]:
    """obs レイアウト文字列と各セグメントの開始インデックス dict を返す。"""
    lines = []
    offsets: dict[str, int] = {}
    offset = 0
    for seg in schema["segments"]:
        name = seg["name"]
        dim = seg["dim"]
        offsets[name] = offset
        if dim == 1:
            lines.append(f"  obs[{offset}]         = {name}")
        else:
            lines.append(f"  obs[{offset}:{offset + dim}] = {name}  (dim={dim})")
        offset += dim
    lines.append(f"  合計: {schema['total_dim']} 次元")
    return "\n".join(lines), offsets


class CoinEurekaConfig(EurekaGameConfig):
    """コイン収集 + 敵回避ゲーム専用の EUREKA 設定。"""

    def __init__(self):
        self._obs_layout_str: str = ""
        self._offsets: dict[str, int] = {}

    def setup(self, host: str, port: int) -> None:
        print(f"[INFO] obs_schema を取得中 ({host}:{port})...")
        schema = _fetch_obs_schema(host, port)
        self._obs_layout_str, self._offsets = _build_obs_layout(schema)
        print(f"[INFO] obs_schema 取得完了: total_dim={schema['total_dim']}")

    def make_env(self, host: str, port: int):
        from envs.coin_env import CoinEnv
        # reward_scale=0.2 でコイン報酬を 5.0→1.0 相当にスケール。
        # info["base_reward"] は元のスケールを保持するためメトリクス計算は正確なまま。
        return CoinEnv(host=host, port=port, reward_scale=0.2)

    def build_prompt(self, prev_metrics: dict | None, iteration: int) -> str:
        metrics_section = (
            "なし（初回）"
            if prev_metrics is None
            else json.dumps(prev_metrics, ensure_ascii=False, indent=2)
        )

        offsets = self._offsets
        coin_i    = offsets.get("coin_rel_pos", 10)
        enemy_r_i = offsets.get("enemy_rel_pos", coin_i + 200)
        enemy_v_i = offsets.get("enemy_vel", enemy_r_i + 40)
        enemy_t_i = offsets.get("enemy_type", enemy_v_i + 40)

        num_coin_obs  = (enemy_r_i - coin_i) // 2
        max_enemy_obs = (enemy_v_i - enemy_r_i) // 2

        nearest_note = (
            f"  obs[{coin_i}], obs[{coin_i + 1}]           = 最近コインへの相対位置 (dx, dy)\n"
            f"  obs[{enemy_r_i}], obs[{enemy_r_i + 1}]         = 最近敵への相対位置 (dx, dy)\n"
            f"  obs[{enemy_v_i}], obs[{enemy_v_i + 1}]         = 最近敵の速度 (vx, vy)\n"
            f"  obs[{enemy_t_i}]               = 最近敵の種類 (0.0=遅い直進, 0.5=速い直進, 1.0=予測追跡)\n"
            f"\n"
            f"## 複数エンティティへのアクセス（i=0が最近傍、距離昇順）\n"
            f"\n"
            f"コイン（最大 {num_coin_obs} 枚、存在しないスロットは 0 埋め）:\n"
            f"  obs[{coin_i} + i*2]     = i番目コインの dx  (i = 0 〜 {num_coin_obs - 1})\n"
            f"  obs[{coin_i} + i*2 + 1] = i番目コインの dy\n"
            f"\n"
            f"敵（最大 {max_enemy_obs} 体、存在しないスロットは 0 埋め）:\n"
            f"  obs[{enemy_r_i} + i*2]     = i番目敵の dx  (i = 0 〜 {max_enemy_obs - 1})\n"
            f"  obs[{enemy_r_i} + i*2 + 1] = i番目敵の dy\n"
            f"  obs[{enemy_v_i} + i*2]     = i番目敵の vx\n"
            f"  obs[{enemy_v_i} + i*2 + 1] = i番目敵の vy\n"
            f"  obs[{enemy_t_i} + i]       = i番目敵の種類 (0.0=遅い直進, 0.5=速い直進, 1.0=予測追跡)\n"
            f"\n"
            f"  ※ obs[8] * {max_enemy_obs} で現在の実際の敵数（正規化前）が得られる"
        )

        return f"""あなたは強化学習の報酬設計エキスパートです。

## ゲーム概要
2D フィールド（±10m の正方形）でプレイヤーがコインを収集しながら敵を回避するゲームです。
- プレイヤーは離散5方向（上/下/左/右/静止）で移動
- コインを取ると得点、敵に当たるとエピソード終了
- 敵はフィールド外周からスポーンし、プレイヤーに向かって移動する

## 観測ベクトル（obs）レイアウト
{self._obs_layout_str}

## 最近傍エンティティの obs インデックス（先頭要素 = 最近傍、距離近い順にソート済み）
{nearest_note}

  ※ dx, dy はフィールド幅 (20m) で正規化済み。値域 [-1, 1]
  ※ 壁距離 wall_dist は FieldHalfSize (10m) で正規化済み。値域 [0, 1]

## 固定報酬（C++ 側、変更不可）
- AliveReward = {_ALIVE_REWARD} / step（生存毎ステップ）
- CoinReward = {_COIN_REWARD} / 枚（コイン取得時）

## 前回イテレーション {iteration - 1} のメトリクス
{metrics_section}

## 課題
前回の訓練の結果から課題を判断して箇条書きで記載。

## タスク
以下のフォーマットで回答してください。

### 1. reward_fn.py のコード
```python
import numpy as np

def reward_shaping(obs: np.ndarray, prev_obs: np.ndarray, base_reward: float) -> float:
    \"\"\"追加の報酬シェーピング関数。必ず np.clip で [-1.0, 1.0] の範囲に収めること。\"\"\"
    # ここに実装
    shaped = 0.0
    return float(np.clip(shaped, -1.0, 1.0))
```

**スケールの注意:** 1ステップあたりの shaped_reward は [-0.01, 0.01] 程度を目安にしてください。
エピソード長は約1,000〜1,500 step のため、合計が base_reward（数〜十数程度）を大きく超えると
学習信号が壊れます。大きな定数ペナルティは避け、差分ベースの小さな報酬を推奨します。

### 2. 推奨訓練ステップ数
50,000〜200,000 の範囲で整数を記載してください（例: 100000）

### 3. 設計の意図
この報酬関数で何を解決しようとしているか（1〜3行）
"""

    def compute_primary_metric(self, episode_base_rewards: list[float],
                               episode_lengths: list[int]) -> float:
        if not episode_base_rewards:
            return 0.0
        mean_base = sum(episode_base_rewards) / len(episode_base_rewards)
        mean_len  = sum(episode_lengths) / len(episode_lengths)
        return max(0.0, (mean_base - _ALIVE_REWARD * mean_len) / _COIN_REWARD)

    def make_model(self, env):
        from stable_baselines3 import PPO
        from entity_attention_extractor import EntityAttentionExtractor
        policy_kwargs = dict(
            features_extractor_class=EntityAttentionExtractor,
            features_extractor_kwargs=dict(
                features_dim=128,
                offsets=self._offsets,
            ),
            net_arch=[64, 64],
        )
        return PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)

    @property
    def primary_metric_name(self) -> str:
        return "coins_per_episode"


def create_config() -> CoinEurekaConfig:
    return CoinEurekaConfig()
