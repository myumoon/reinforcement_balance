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
        return CoinEnv(host=host, port=port)

    # ------------------------------------------------------------------ #
    # プロンプトセクション（本文のみ。タイトルは _titled_section() で付与）    #
    # ------------------------------------------------------------------ #

    def _prompt_section_game_overview(self) -> str:
        return (
            "2D フィールド（±10m の正方形）でプレイヤーがコインを収集しながら敵を回避するゲームです。\n"
            "- プレイヤーは離散5方向（上/下/左/右/静止）で移動\n"
            "- コインを取ると得点、敵に当たるとエピソード終了\n"
            "- 敵はフィールド外周からスポーンし、プレイヤーに向かって移動する"
        )

    def _prompt_section_game_objective(self) -> str:
        return (
            "このゲームの**最優先目標**は「1エピソードでコインをできるだけ多く取得すること」です。\n"
            "- 敵を避けることは目標ではなく手段（死ぬとエピソード終了してコインが取れなくなる）\n"
            "- 壁際に留まるのは最悪の戦略（コインを取れず敵にも当たりやすい）\n"
            "- **評価指標: coins_per_episode**（1エピソードのコイン取得枚数）を最大化することが目的\n"
            "- shaped_reward はコインへの積極的な接近を強く後押しすること\n"
            "- 敵回避は「死を避ける」最小限のペナルティに留め、コイン接近を妨げない設計にすること"
        )

    def _prompt_section_obs_layout(self) -> str:
        return self._obs_layout_str

    def _prompt_section_obs_index(self) -> str:
        return (
            f"{self.obs_index_description()}\n\n"
            f"  ※ dx, dy はフィールド幅 (20m) で正規化済み。値域 [-1, 1]\n"
            f"  ※ 壁距離 wall_dist は FieldHalfSize (10m) で正規化済み。値域 [0, 1]"
        )

    def _prompt_section_fixed_rewards(self) -> str:
        return (
            f"- AliveReward = {_ALIVE_REWARD} / step（生存毎ステップ）\n"
            f"- CoinReward = {_COIN_REWARD} / 枚（コイン取得時）"
        )

    def _prompt_section_physics(self) -> str:
        return (
            "- コイン収集半径: 1.0m = 正規化値 **0.05**（この距離以内で自動収集）\n"
            "- 敵衝突半径:   0.6m = 正規化値 0.03\n"
            "- プレイヤー最大速度: 約 2.5m/s → 1ステップ(1/60s)あたり約 **0.0021 normalized**\n"
            "- 敵スポーン間隔: 10秒 ≈ 600ステップごとに1体\n"
            "- 敵の速度: 遅い直進=1.0m/s、速い直進=2.5m/s、予測追跡=1.5m/s\n"
            "  （予測追跡はプレイヤーの移動先を予測して追跡するため最も危険）"
        )

    def _prompt_section_scale_constraints(self) -> str:
        return (
            f"- コイン接近報酬は 1ステップあたり [-0.05, 0.05] 程度まで。大きな定数ペナルティは避け差分ベースを推奨\n"
            f"- 敵ペナルティは定数 -0.005 以下に留め、コイン接近を阻害しないこと\n"
            f"- 近接ボーナスの閾値は 0.05〜0.10 normalized（収集半径の1〜2倍）以内にすること\n"
            f"  0.15（3m）以上はコインを取らず近くで滞在するだけの戦略を誘発する\n"
            f"- 1エピソードの近接ボーナス合計が CoinReward({_COIN_REWARD}) を超えないよう係数を設定すること\n"
            f"  例: 0.003/step × 1500step = 4.5 < {_COIN_REWARD}"
        )

    # ------------------------------------------------------------------ #
    # 共通テキスト生成                                                      #
    # ------------------------------------------------------------------ #

    def obs_index_description(self) -> str:
        """最近傍エンティティの obs インデックス説明を返す。"""
        offsets = self._offsets
        coin_i    = offsets.get("coin_rel_pos", 10)
        enemy_r_i = offsets.get("enemy_rel_pos", coin_i + 200)
        enemy_v_i = offsets.get("enemy_vel", enemy_r_i + 40)
        enemy_t_i = offsets.get("enemy_type", enemy_v_i + 40)

        num_coin_obs  = (enemy_r_i - coin_i) // 2
        max_enemy_obs = (enemy_v_i - enemy_r_i) // 2

        return (
            f"  obs[{coin_i}], obs[{coin_i + 1}]           = 最近コインへの相対位置 (dx, dy)\n"
            f"  obs[{enemy_r_i}], obs[{enemy_r_i + 1}]         = 最近敵への相対位置 (dx, dy)\n"
            f"  obs[{enemy_v_i}], obs[{enemy_v_i + 1}]         = 最近敵の速度 (vx, vy)\n"
            f"  obs[{enemy_t_i}]               = 最近敵の種類 (0.0=遅い直進, 0.5=速い直進, 1.0=予測追跡)\n"
            f"\n"
            f"**複数エンティティへのアクセス（i=0が最近傍、距離昇順）**\n"
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

    def metrics_description(self) -> str:
        """メトリクスの各パラメーターの意味を返す。build_prompt() とレビューで共通利用。"""
        return (
            f"- base_reward: C++固定報酬のみ（AliveReward={_ALIVE_REWARD}/step + CoinReward={_COIN_REWARD}/枚）\n"
            f"- shaped_reward: reward_fn の出力（シェーピング報酬）\n"
            f"- episode_length: エピソードの長さ（ステップ数）\n"
            f"- coins_per_episode: 1エピソードのコイン取得枚数\n"
            f"  = (base_reward - AliveReward × episode_length) / CoinReward\n"
            f"- coins_per_episode_std: コイン取得枚数の標準偏差\n"
            f"- episode_length_min / episode_length_max: エピソード長の最小・最大"
        )

    # ------------------------------------------------------------------ #
    # プロンプト構築                                                        #
    # ------------------------------------------------------------------ #

    def build_game_context(self) -> str:
        """レビュアー向けゲームコンテキスト。各セクションメソッドを _titled_section() で組み合わせて構築する。"""
        return "\n\n".join([
            self._titled_section("ゲーム概要", self._prompt_section_game_overview()),
            self._titled_section("ゲームの目標（最重要）", self._prompt_section_game_objective()),
            self._titled_section("観測ベクトル（obs）レイアウト", self._prompt_section_obs_layout()),
            self._titled_section(
                "最近傍エンティティの obs インデックス（先頭要素 = 最近傍、距離近い順にソート済み）",
                self._prompt_section_obs_index(),
            ),
            self._titled_section("固定報酬（C++ 側、変更不可）", self._prompt_section_fixed_rewards()),
            self._titled_section(
                "ゲームの物理定数（distance threshold の設計に必ず参照すること）",
                self._prompt_section_physics(),
            ),
            self._titled_section("スケール制約（reward_fn 設計上の上限）", self._prompt_section_scale_constraints()),
            self._titled_section("メトリクスの各パラメーターの意味", self.metrics_description()),
        ])

    def build_prompt(self, prev_metrics: dict | None, iteration: int,
                     prev_review: str | None = None) -> str:
        metrics_value = (
            "なし（初回）"
            if prev_metrics is None
            else json.dumps(prev_metrics, ensure_ascii=False, indent=2)
        )
        metrics_section = (
            f"## 前回イテレーション {iteration - 1} のメトリクス\n"
            f"### 各パラメーターの意味\n"
            f"{self.metrics_description()}\n\n"
            f"### 値\n"
            f"{metrics_value}"
        )
        task_section = (
            f"## タスク\n"
            f"以下のフォーマットで回答してください。\n\n"
            f"### 1. reward_fn.py のコード\n"
            f"```python\n"
            f"import numpy as np\n\n"
            f"def reward_shaping(obs: np.ndarray, prev_obs: np.ndarray, base_reward: float) -> float:\n"
            f'    """追加の報酬シェーピング関数。必ず np.clip で [-1.0, 1.0] の範囲に収めること。"""\n'
            f"    # ここに実装\n"
            f"    shaped = 0.0\n"
            f"    return float(np.clip(shaped, -1.0, 1.0))\n"
            f"```\n\n"
            f"{self._titled_section('スケール制約（reward_fn 設計上の上限）', self._prompt_section_scale_constraints())}\n\n"
            f"### 2. 推奨訓練ステップ数\n"
            f"50,000〜200,000 の範囲で整数を記載してください（例: 100000）\n\n"
            f"### 3. 設計の意図\n"
            f"この報酬関数で何を解決しようとしているか（1〜3行）"
        )
        items = [
            "あなたは強化学習の報酬設計エキスパートです。",
            self._titled_section("ゲーム概要", self._prompt_section_game_overview()),
            self._titled_section("ゲームの目標（最重要）", self._prompt_section_game_objective()),
            self._titled_section("観測ベクトル（obs）レイアウト", self._prompt_section_obs_layout()),
            self._titled_section(
                "最近傍エンティティの obs インデックス（先頭要素 = 最近傍、距離近い順にソート済み）",
                self._prompt_section_obs_index(),
            ),
            self._titled_section("固定報酬（C++ 側、変更不可）", self._prompt_section_fixed_rewards()),
            self._titled_section(
                "ゲームの物理定数（distance threshold の設計に必ず参照すること）",
                self._prompt_section_physics(),
            ),
            metrics_section,
        ]
        if prev_review is not None:
            items.append(self._titled_section("前回レビューで指摘された設計上の問題", prev_review))
        items.append("## 課題\n前回の訓練の結果から課題を判断して箇条書きで記載。")
        items.append(task_section)
        return "\n\n".join(items)

    # ------------------------------------------------------------------ #
    # メトリクス計算                                                        #
    # ------------------------------------------------------------------ #

    def compute_primary_metric(self, episode_base_rewards: list[float],
                               episode_lengths: list[int]) -> float:
        if not episode_base_rewards:
            return 0.0
        mean_base = sum(episode_base_rewards) / len(episode_base_rewards)
        mean_len  = sum(episode_lengths) / len(episode_lengths)
        return max(0.0, (mean_base - _ALIVE_REWARD * mean_len) / _COIN_REWARD)

    def compute_extra_metrics(self, episode_base_rewards: list[float],
                              episode_lengths: list[int]) -> dict:
        import statistics
        coins_list = [
            max(0.0, (r - _ALIVE_REWARD * l) / _COIN_REWARD)
            for r, l in zip(episode_base_rewards, episode_lengths)
        ]
        return {
            f"{self.primary_metric_name}_std": round(
                statistics.stdev(coins_list) if len(coins_list) > 1 else 0.0, 3),
            "episode_length_min": min(episode_lengths),
            "episode_length_max": max(episode_lengths),
        }

    def make_model(self, env):
        from stable_baselines3 import PPO
        from entity_attention_extractor import EntityAttentionExtractor
        from eureka_game_config import _linear_schedule
        policy_kwargs = dict(
            features_extractor_class=EntityAttentionExtractor,
            features_extractor_kwargs=dict(
                features_dim=128,
                offsets=self._offsets,
                use_polar=True,
            ),
            net_arch=[64, 64],
        )
        return PPO(
            "MlpPolicy", env,
            policy_kwargs=policy_kwargs,
            learning_rate=_linear_schedule(3e-4),
            n_steps=4096,
            batch_size=256,
            n_epochs=10,
            clip_range=0.1,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
        )

    @property
    def primary_metric_name(self) -> str:
        return "coins_per_episode"


def create_config() -> CoinEurekaConfig:
    return CoinEurekaConfig()
