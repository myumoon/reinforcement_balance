"""Survivors ゲーム専用の EUREKA 設定。"""

import json
import sys

import requests

from eureka_game_config import EurekaGameConfig

_ALIVE_REWARD = 0.001
_ITEM_REWARD  = 3.0
_KILL_REWARD  = 2.0

# VS スケール換算ダメージ値（ドキュメント参照用）
_ENEMY_DPS = {"A": 5.0, "B": 10.0, "C": 8.0}
_ENEMY_HP  = {"A": 20.0, "B": 50.0, "C": 30.0}
_AURA_DPS  = 15.0
_MAX_PLAYER_HP = 100.0


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


class SurvivorsEurekaConfig(EurekaGameConfig):
    """Survivors ゲーム専用の EUREKA 設定。"""

    def __init__(self):
        self._obs_layout_str: str = ""
        self._offsets: dict[str, int] = {}

    def setup(self, host: str, port: int) -> None:
        print(f"[INFO] obs_schema を取得中 ({host}:{port})...")
        schema = _fetch_obs_schema(host, port)
        self._obs_layout_str, self._offsets = _build_obs_layout(schema)
        print(f"[INFO] obs_schema 取得完了: total_dim={schema['total_dim']}")

    def make_env(self, host: str, port: int):
        from envs.survivors_env import SurvivorsEnv
        return SurvivorsEnv(host=host, port=port)

    # ------------------------------------------------------------------ #
    # プロンプトセクション                                                  #
    # ------------------------------------------------------------------ #

    def _prompt_section_game_overview(self) -> str:
        return (
            f"2D フィールド（±15m の正方形）でプレイヤーが敵を倒しながらアイテムを集めて生き延びるゲームです。\n"
            f"- プレイヤーは離散5方向（上/下/左/右/静止）で移動\n"
            f"- プレイヤーはオーラ（半径 1.5m）で自動攻撃し、敵を倒せる\n"
            f"- 敵に接触すると毎ティック HP が減少し、HP=0 でエピソード終了\n"
            f"- 敵はフィールド外周からスポーンし、プレイヤーに向かって移動する（最大6体同時、5秒間隔）\n"
            f"- フィールドにはアイテムが常時10個存在する\n"
            f"- アイテム（{_ITEM_REWARD}点）とKill報酬（{_KILL_REWARD}点/体）で得点を稼ぐ"
        )

    def _prompt_section_game_objective(self) -> str:
        return (
            f"**最優先目標**: 1エピソードで「アイテムを取り敵を倒しながら生き延びる」こと。\n"
            f"- 死ぬとエピソード終了するため、HP 管理が最重要\n"
            f"- ただし壁際に逃げ続けるだけでは得点が低い\n"
            f"- 敵をオーラ範囲に引き込んで倒しつつアイテムを取る積極的行動が最善\n"
            f"- **評価指標: item_kill_score** = (base_reward - AliveReward×ep_len) の平均\n"
            f"  純粋なアイテム+Kill スコア。生存だけでは 0.0、アイテム1個で {_ITEM_REWARD:.1f}、Kill1体で {_KILL_REWARD:.1f}"
        )

    def _prompt_section_obs_index(self) -> str:
        o = self._offsets
        hp_i       = o.get("player_hp", 12)
        wpn_i      = o.get("weapon_slots", 13)
        ecnt_i     = o.get("enemy_count", 19)
        spawn_i    = o.get("spawn_timer", 20)
        xp_i       = o.get("xp_progress", 21)
        item_i     = o.get("item_rel_pos", 23)
        er_i       = o.get("enemy_rel_pos", item_i + 40)
        ev_i       = o.get("enemy_vel", er_i + 40)
        et_i       = o.get("enemy_type", ev_i + 40)
        ehp_i      = o.get("enemy_hp", et_i + 20)
        max_enemy  = (ev_i - er_i) // 2
        num_items  = (er_i - item_i) // 2

        return (
            f"  obs[0:2]   = player_pos (x,y) / FieldHalfSize(15m) → [-1, 1]\n"
            f"  obs[2:4]   = player_vel (vx,vy)\n"
            f"  obs[4:12]  = wall_rays 8方向 (0~1, 1=遠い・0=壁が近い)\n"
            f"  obs[{hp_i}]     = player_hp / {_MAX_PLAYER_HP} (0~1)\n"
            f"  obs[{wpn_i}:{wpn_i+6}] = weapon_slots × 3: (type_norm, level_norm) 各 [0,1]\n"
            f"             Phase1: obs[{wpn_i}]=0.125(Aura), obs[{wpn_i+1}]=0.125(Lv1), 他は0\n"
            f"  obs[{ecnt_i}]    = 敵数 / {max_enemy}\n"
            f"  obs[{spawn_i}]    = スポーンタイマー (0~1)\n"
            f"  obs[{xp_i}]    = xp_progress (Phase1=0固定)\n"
            f"\n"
            f"**⚠ Phase1 固定値（reward_fn で参照しないこと）**\n"
            f"  obs[{wpn_i}:{wpn_i+6}] weapon_slots: 常に [0.125, 0.125, 0, 0, 0, 0] 固定\n"
            f"  obs[{xp_i}]           xp_progress:  常に 0.0\n"
            f"  obs[{o.get('player_level', xp_i+1)}]           player_level: 常に 0.0\n"
            f"\n"
            f"  obs[{item_i}:{item_i+2}] = 最近アイテムへの相対位置 (dx, dy) / 30m → [-1, 1]\n"
            f"  obs[{er_i}:{er_i+2}] = 最近敵への相対位置 (dx, dy)\n"
            f"  obs[{ev_i}:{ev_i+2}] = 最近敵の速度 (vx, vy)\n"
            f"  obs[{et_i}]    = 最近敵の種類 (0.0=Slime遅い, 0.5=Zombie速い, 1.0=Ghost予測)\n"
            f"  obs[{ehp_i}]    = 最近敵の HP (0~1, 0=瀕死)\n"
            f"\n"
            f"**複数エンティティへのアクセス（i=0が最近傍、距離昇順）**\n"
            f"  アイテム（最大{num_items}個）: obs[{item_i}+i*2], obs[{item_i}+i*2+1] = dx,dy\n"
            f"  敵（最大{max_enemy}体）:\n"
            f"    obs[{er_i}+i*2], obs[{er_i}+i*2+1] = dx,dy\n"
            f"    obs[{ev_i}+i*2], obs[{ev_i}+i*2+1] = vx,vy\n"
            f"    obs[{et_i}+i]   = type\n"
            f"    obs[{ehp_i}+i]   = hp/max_hp"
        )

    def _prompt_section_fixed_rewards(self) -> str:
        return (
            f"- AliveReward = {_ALIVE_REWARD} / step（生存毎ステップ）\n"
            f"- ItemReward  = {_ITEM_REWARD}（アイテム取得時）\n"
            f"- KillReward  = {_KILL_REWARD}（敵撃破時）"
        )

    def _prompt_section_physics(self) -> str:
        return (
            f"- アイテム収集半径: 1.0m\n"
            f"- 敵接触半径: 0.6m（この距離以内で HP ダメージ）\n"
            f"- オーラ攻撃半径: 1.5m（この距離以内の敵に自動ダメージ）\n"
            f"- オーラ DPS: {_AURA_DPS} HP/s → per tick: {_AURA_DPS/60:.4f}\n"
            f"- プレイヤー最大 HP: {_MAX_PLAYER_HP}\n"
            f"- 敵タイプ別ダメージ DPS: A={_ENEMY_DPS['A']}, B={_ENEMY_DPS['B']}, C={_ENEMY_DPS['C']} HP/s\n"
            f"  → per tick: A={_ENEMY_DPS['A']/60:.4f}, B={_ENEMY_DPS['B']/60:.4f}, C={_ENEMY_DPS['C']/60:.4f}\n"
            f"- 敵タイプ別 HP: A={_ENEMY_HP['A']}, B={_ENEMY_HP['B']}, C={_ENEMY_HP['C']}\n"
            f"  → Aura で倒す時間: A={_ENEMY_HP['A']/_AURA_DPS:.1f}s, B={_ENEMY_HP['B']/_AURA_DPS:.1f}s, C={_ENEMY_HP['C']/_AURA_DPS:.1f}s\n"
            f"- 敵速度: A=1.0m/s, B=2.5m/s, C=1.5m/s（予測追跡）\n"
            f"- スポーン間隔: 5秒 ≈ 300ステップ（MaxActiveEnemies=6、EnemySpeedMult で倍率変更可能）"
        )

    def _prompt_section_scale_constraints(self) -> str:
        return (
            f"- **HP ペナルティは survivors_env が永続的に適用済み**（info['hp_penalty']）\n"
            f"  reward_fn でさらに HP 差分ペナルティを追加しないこと（二重計上になる）\n"
            f"  HP 状態を使う場合は「obs[12] が低い時にアイテム接近を促す」など間接的な利用にとどめること\n"
            f"- 敵接近ペナルティは [-0.05, 0.0] 程度まで\n"
            f"- アイテム接近ボーナスは 1ステップあたり [-0.03, 0.03] 程度まで（アイテム10個に対して設計）\n"
            f"- item_kill_score = 0 は「生存のみ」。reward_fn は item_kill_score を上げることを目標とすること\n"
            f"- エピソード全体の shaped_reward 累計が base_reward を大幅に超えないよう設計すること"
        )

    # ------------------------------------------------------------------ #
    # ゲームコンテキスト・プロンプト構築                                      #
    # ------------------------------------------------------------------ #

    def build_game_context(self) -> str:
        return "\n\n".join([
            self._titled_section("ゲーム概要",           self._prompt_section_game_overview()),
            self._titled_section("ゲームの目標",          self._prompt_section_game_objective()),
            self._titled_section("obs インデックス一覧",  self._prompt_section_obs_index()),
            self._titled_section("固定報酬（C++ 側）",    self._prompt_section_fixed_rewards()),
            self._titled_section("物理定数",              self._prompt_section_physics()),
            self._titled_section("スケール制約",           self._prompt_section_scale_constraints()),
            self._titled_section("メトリクスの意味",       self.metrics_description()),
        ])

    def _build_prompt_static(self) -> str:
        return "\n\n".join([
            "あなたは強化学習の報酬設計エキスパートです。",
            self._titled_section("ゲーム概要",           self._prompt_section_game_overview()),
            self._titled_section("ゲームの目標",          self._prompt_section_game_objective()),
            self._titled_section("obs インデックス一覧",  self._prompt_section_obs_index()),
            self._titled_section("固定報酬（C++ 側）",    self._prompt_section_fixed_rewards()),
            self._titled_section("物理定数",              self._prompt_section_physics()),
        ])

    def _build_prompt_dynamic(self, prev_metrics: dict | None, iteration: int,
                               prev_review: str | None = None) -> str:
        metrics_value = (
            "なし（初回）"
            if prev_metrics is None
            else json.dumps(prev_metrics, ensure_ascii=False, indent=2)
        )
        metrics_section = (
            f"## 前回イテレーション {iteration - 1} のメトリクス\n"
            f"### 各パラメーターの意味\n{self.metrics_description()}\n\n"
            f"### 値\n{metrics_value}"
        )
        task_section = (
            f"## タスク\n"
            f"以下のフォーマットで回答してください。\n\n"
            f"### 1. reward_fn.py のコード\n"
            f"```python\n"
            f"import numpy as np\n\n"
            f"def reward_shaping(obs: np.ndarray, prev_obs: np.ndarray, base_reward: float) -> float:\n"
            f'    """追加の報酬シェーピング関数。np.clip で [-1.0, 1.0] に収めること。"""\n'
            f"    shaped = 0.0\n"
            f"    return float(np.clip(shaped, -1.0, 1.0))\n"
            f"```\n\n"
            f"{self._titled_section('スケール制約', self._prompt_section_scale_constraints())}\n\n"
            f"### 2. 推奨訓練ステップ数\n"
            f"50,000〜200,000 の範囲で整数を記載してください\n\n"
            f"### 3. 設計の意図\n"
            f"この報酬関数で何を解決しようとしているか（1〜3行）"
        )
        items = [metrics_section]
        if prev_review is not None:
            items.append(self._titled_section("前回レビューで指摘された設計上の問題", prev_review))
        items.append("## 課題\n前回の訓練の結果から課題を判断して箇条書きで記載。")
        items.append(task_section)
        return "\n\n".join(items)

    def build_prompt_parts(self, prev_metrics: dict | None, iteration: int,
                           prev_review: str | None = None) -> tuple[str, str]:
        return self._build_prompt_static(), self._build_prompt_dynamic(prev_metrics, iteration, prev_review)

    def build_prompt(self, prev_metrics: dict | None, iteration: int,
                     prev_review: str | None = None) -> str:
        static, dynamic = self.build_prompt_parts(prev_metrics, iteration, prev_review)
        return static + "\n\n" + dynamic

    def build_constraints_hint(self) -> str:
        return "\n\n".join([
            self._titled_section("スケール制約", self._prompt_section_scale_constraints()),
            self._titled_section("物理定数",     self._prompt_section_physics()),
        ])

    # ------------------------------------------------------------------ #
    # メトリクス計算                                                        #
    # ------------------------------------------------------------------ #

    def compute_primary_metric(self, episode_base_rewards: list[float],
                               episode_lengths: list[int]) -> float:
        """primary_metric = アイテム+Kill スコアの平均（AliveReward 分を除去）。"""
        if not episode_base_rewards:
            return 0.0
        scores = [
            max(0.0, r - _ALIVE_REWARD * l)
            for r, l in zip(episode_base_rewards, episode_lengths)
        ]
        return sum(scores) / len(scores)

    def compute_extra_metrics(self, episode_base_rewards: list[float],
                              episode_lengths: list[int]) -> dict:
        import statistics
        mean_len = sum(episode_lengths) / len(episode_lengths) if episode_lengths else 0.0
        scores = [
            max(0.0, r - _ALIVE_REWARD * l)
            for r, l in zip(episode_base_rewards, episode_lengths)
        ]
        return {
            "episode_length_mean": round(mean_len, 1),
            "episode_length_min":  min(episode_lengths) if episode_lengths else 0,
            "episode_length_max":  max(episode_lengths) if episode_lengths else 0,
            "item_kill_score_std": round(
                statistics.stdev(scores) if len(scores) > 1 else 0.0, 3),
        }

    def metrics_description(self) -> str:
        return (
            f"- base_reward: C++固定報酬のみ（AliveReward={_ALIVE_REWARD}/step + ItemReward={_ITEM_REWARD} + KillReward={_KILL_REWARD}）\n"
            f"- shaped_reward: reward_fn の出力（hp_penalty 含む）\n"
            f"- episode_length: エピソード長（ステップ数、最大 = 全 HP 消費まで）\n"
            f"- item_kill_score (primary): (base_reward - AliveReward×ep_len) の平均\n"
            f"  純粋なアイテム+Kill スコア。生存のみ=0.0、アイテム1個={_ITEM_REWARD}、Kill1体={_KILL_REWARD}\n"
            f"- item_kill_score_std: 標準偏差\n"
            f"- episode_length_min / max: 最短・最長エピソード長"
        )

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
                dist_alpha=1.0,
                item_key="item_rel_pos",
                enemy_scalar_keys=["enemy_type", "enemy_hp"],
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
        return "item_kill_score"

    @property
    def default_port(self) -> int:
        return 8767


def create_config() -> SurvivorsEurekaConfig:
    return SurvivorsEurekaConfig()
