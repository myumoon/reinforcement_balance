"""EUREKA型報酬シェーピングループ。

LLMが報酬シェーピング関数を反復的に生成・改良し、
コイン取得と敵回避を両立するAIを育てる。

使い方:
  python eureka_loop.py --iterations 5
  python eureka_loop.py --iterations 10 --run-name approach_shaping
  python eureka_loop.py --llm openai --model gpt-4o --iterations 3
  python eureka_loop.py --help
"""

import argparse
import datetime
import importlib.util
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import requests
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

# CoinGame の固定報酬パラメータ（C++ 側の定数と一致させること）
_ALIVE_REWARD = 0.001
_COIN_REWARD = 5.0
_STATE_FILENAME = "state.json"

_DEFAULT_STEPS = 50_000
_DEFAULT_ANTHROPIC_MODEL = "claude-opus-4-6"
_DEFAULT_OPENAI_MODEL = "gpt-4o"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EUREKA型報酬シェーピングループ")
    p.add_argument("--iterations", type=int, default=5, help="ループ回数（default: 5）")
    p.add_argument("--run-name", default=None,
                   help="保存ディレクトリ名（未指定時はタイムスタンプ自動生成）")
    p.add_argument("--llm", choices=["anthropic", "openai"], default="anthropic",
                   help="LLMバックエンド（default: anthropic）")
    p.add_argument("--model", default=None,
                   help="LLMモデル名（未指定時: anthropic=claude-opus-4-6, openai=gpt-4o）")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8766)
    p.add_argument("--output-dir", default="eureka_results",
                   help="結果保存の親ディレクトリ（default: eureka_results）")
    p.add_argument("--max-steps", type=int, default=200_000,
                   help="1イテレーションの最大ステップ数（LLMの推奨値の上限, default: 200000）")
    p.add_argument("--min-steps", type=int, default=30_000,
                   help="早期終了チェックを開始する最小ステップ数（default: 30000）")
    p.add_argument("--eval-freq", type=int, default=10_000,
                   help="プラトー判定の評価間隔・ステップ数（default: 10000）")
    p.add_argument("--patience", type=int, default=3,
                   help="改善なしN回で早期終了（default: 3）")
    p.add_argument("--delta", type=float, default=0.05,
                   help="改善とみなす coins_per_episode の最小増加量（default: 0.05）")
    return p.parse_args()


# ---------------------------------------------------------------------------
# LLM クライアント
# ---------------------------------------------------------------------------

def _build_llm_client(backend: str, model_override: str | None):
    if backend == "anthropic":
        import anthropic
        client = anthropic.Anthropic()
        model_name = model_override or _DEFAULT_ANTHROPIC_MODEL
    else:
        try:
            from openai import OpenAI
        except ImportError:
            print("[ERROR] openai パッケージがインストールされていません。pip install openai")
            sys.exit(1)
        client = OpenAI()
        model_name = model_override or _DEFAULT_OPENAI_MODEL
    return client, model_name


def _call_llm(client, model_name: str, prompt: str, backend: str) -> str:
    if backend == "anthropic":
        response = client.messages.create(
            model=model_name,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    else:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content


# ---------------------------------------------------------------------------
# obs スキーマ
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# プロンプト生成
# ---------------------------------------------------------------------------

def _build_prompt(obs_layout_str: str, offsets: dict[str, int],
                  prev_metrics: dict | None, iteration: int) -> str:
    metrics_section = (
        "なし（初回）"
        if prev_metrics is None
        else json.dumps(prev_metrics, ensure_ascii=False, indent=2)
    )

    coin_i = offsets.get("coin_rel_pos", 10)
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
{obs_layout_str}

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
前回の訓練でエージェントはコインを取らず敵から逃げるだけの挙動を示した。
「コインを取りながら敵からも逃げる」複合行動を引き出す報酬シェーピングを設計してほしい。

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

### 2. 推奨訓練ステップ数
50,000〜200,000 の範囲で整数を記載してください（例: 100000）

### 3. 設計の意図
この報酬関数で何を解決しようとしているか（1〜3行）
"""


# ---------------------------------------------------------------------------
# レスポンス解析
# ---------------------------------------------------------------------------

def _extract_reward_fn_code(text: str) -> str | None:
    match = re.search(r"```python\n(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else None


def _extract_steps(text: str) -> int:
    # "### 2." 直後の数値行を探す
    m = re.search(r"###\s*2[^\n]*\n+([0-9][0-9_,]*)", text)
    if not m:
        m = re.search(r"ステップ[数]?[：:\s]*([0-9][0-9_,]+)", text)
    if m:
        raw = m.group(1).replace("_", "").replace(",", "")
        try:
            val = int(raw)
            if 1_000 <= val <= 10_000_000:
                return val
        except ValueError:
            pass
    return _DEFAULT_STEPS


def _validate_code(code: str) -> bool:
    try:
        compile(code, "<reward_fn>", "exec")
        return True
    except SyntaxError as e:
        print(f"[WARN] 生成コードに構文エラー: {e}")
        return False


# ---------------------------------------------------------------------------
# 動的ロード
# ---------------------------------------------------------------------------

def _load_reward_fn(path: Path) -> Callable:
    spec = importlib.util.spec_from_file_location("_eureka_reward_fn", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.reward_shaping


# ---------------------------------------------------------------------------
# メトリクスコールバック
# ---------------------------------------------------------------------------

class _EurekaMetricsCallback(BaseCallback):
    """エピソードごとに base_reward / shaped_reward を集計し、プラトー検出で早期終了するコールバック。"""

    _RECENT_WINDOW = 20  # プラトー判定に使う直近エピソード数

    def __init__(self, eval_freq: int = 10_000, min_steps: int = 30_000,
                 patience: int = 3, delta: float = 0.05):
        super().__init__(verbose=0)
        self.eval_freq = eval_freq
        self.min_steps = min_steps
        self.patience = patience
        self.delta = delta

        self._ep_base = 0.0
        self._ep_shaped = 0.0
        self._ep_len = 0
        self.episode_base_rewards: list[float] = []
        self.episode_shaped_rewards: list[float] = []
        self.episode_lengths: list[int] = []

        self._check_history: list[float] = []
        self._no_improve_count = 0
        self._last_check_step = 0

    def _on_step(self) -> bool:
        info = self.locals["infos"][0]
        self._ep_base += info.get("base_reward", 0.0)
        self._ep_shaped += info.get("shaped_reward", 0.0)
        self._ep_len += 1

        if self.locals["dones"][0]:
            self.episode_base_rewards.append(self._ep_base)
            self.episode_shaped_rewards.append(self._ep_shaped)
            self.episode_lengths.append(self._ep_len)
            self._ep_base = 0.0
            self._ep_shaped = 0.0
            self._ep_len = 0

        # プラトー検出チェック
        if (self.num_timesteps >= self.min_steps and
                self.num_timesteps - self._last_check_step >= self.eval_freq and
                len(self.episode_base_rewards) >= self._RECENT_WINDOW):
            self._last_check_step = self.num_timesteps
            current_coins = self._get_recent_coins()

            if self._check_history and current_coins <= max(self._check_history) + self.delta:
                self._no_improve_count += 1
            else:
                self._no_improve_count = 0
            self._check_history.append(current_coins)

            print(f"[INFO] プラトーチェック ({self.num_timesteps:,} steps): "
                  f"coins/ep={current_coins:.3f}, "
                  f"改善なし={self._no_improve_count}/{self.patience}")

            if self._no_improve_count >= self.patience:
                print("[INFO] プラトー検出: 早期終了します")
                return False

        return True

    def _get_recent_coins(self) -> float:
        recent_base = self.episode_base_rewards[-self._RECENT_WINDOW:]
        recent_len  = self.episode_lengths[-self._RECENT_WINDOW:]
        mean_base = sum(recent_base) / len(recent_base)
        mean_len  = sum(recent_len)  / len(recent_len)
        return max(0.0, (mean_base - _ALIVE_REWARD * mean_len) / _COIN_REWARD)

    def get_metrics(self) -> dict:
        n = len(self.episode_base_rewards)
        if n == 0:
            return {"episodes": 0}
        mean_base = sum(self.episode_base_rewards) / n
        mean_shaped = sum(self.episode_shaped_rewards) / n
        mean_len = sum(self.episode_lengths) / n
        mean_coins = max(0.0, (mean_base - _ALIVE_REWARD * mean_len) / _COIN_REWARD)
        return {
            "episodes": n,
            "episode_reward_mean": round(mean_base + mean_shaped, 4),
            "base_reward_mean": round(mean_base, 4),
            "shaped_reward_mean": round(mean_shaped, 4),
            "episode_length_mean": round(mean_len, 1),
            "coins_per_episode": round(mean_coins, 3),
        }


# ---------------------------------------------------------------------------
# 状態の保存・読み込み（再開用）
# ---------------------------------------------------------------------------

def _save_state(run_dir: Path, next_iter: int, prev_metrics: dict | None,
                best_iter: int, best_coins: float) -> None:
    state = {
        "next_iter": next_iter,
        "prev_metrics": prev_metrics,
        "best_iter": best_iter,
        "best_coins": best_coins,
    }
    (run_dir / _STATE_FILENAME).write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _load_state(run_dir: Path) -> dict | None:
    path = run_dir / _STATE_FILENAME
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def _is_credit_error(e: Exception) -> bool:
    return "credit balance" in str(e).lower()


# ---------------------------------------------------------------------------
# メインループ
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    # 出力ディレクトリ
    run_name = args.run_name or datetime.datetime.now().strftime("coin_%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] 保存先: {run_dir}")

    # LLM クライアント
    client, model_name = _build_llm_client(args.llm, args.model)
    print(f"[INFO] LLM: {args.llm} / {model_name}")

    # obs_schema 取得（UE5 が起動している必要がある）
    print(f"[INFO] obs_schema を取得中 ({args.host}:{args.port})...")
    schema = _fetch_obs_schema(args.host, args.port)
    obs_layout_str, offsets = _build_obs_layout(schema)
    print(f"[INFO] obs_schema 取得完了: total_dim={schema['total_dim']}")

    # 環境作成（_wait_for_server は __init__ 内で済んでいる）
    from envs.coin_env import CoinEnv
    env = CoinEnv(host=args.host, port=args.port)

    # 再開状態の読み込み
    state = _load_state(run_dir)
    if state:
        start_iter = state["next_iter"]
        prev_metrics = state.get("prev_metrics")
        best_iter = state.get("best_iter", -1)
        best_coins = state.get("best_coins", -1.0)
        print(f"[INFO] 前回の状態を検出: イテレーション {start_iter} から再開します。")
    else:
        start_iter = 1
        prev_metrics = None
        best_coins = -1.0
        best_iter = -1

    if start_iter > args.iterations:
        print(f"[INFO] 全イテレーション ({args.iterations}) 完了済みです。")
        env.close()
        return

    try:
        for i in range(start_iter, args.iterations + 1):
            iter_dir = run_dir / f"iter_{i:03d}"
            iter_dir.mkdir(exist_ok=True)
            print(f"\n{'=' * 60}")
            print(f"[EUREKA] イテレーション {i}/{args.iterations}")
            print(f"{'=' * 60}")

            # --- LLM に報酬関数を生成させる ---
            prompt = _build_prompt(obs_layout_str, offsets, prev_metrics, i)
            print("[INFO] LLM に報酬関数を生成中...")
            try:
                llm_response = _call_llm(client, model_name, prompt, args.llm)
            except Exception as e:
                if _is_credit_error(e):
                    _save_state(run_dir, i, prev_metrics, best_iter, best_coins)
                    print(f"\n[ERROR] クレジット残高が不足しています。")
                    print(f"[INFO]  https://console.anthropic.com/settings/billing でクレジットを追加してください。")
                    print(f"[INFO]  追加後、以下のコマンドで再開できます:")
                    print(f"[INFO]    python eureka_loop.py --iterations {args.iterations} --run-name {run_name}")
                    return
                raise

            # レスポンス保存
            (iter_dir / "llm_response.txt").write_text(llm_response, encoding="utf-8")

            # コード・ステップ数を抽出（LLMの推奨値を max_steps で上限クランプ）
            reward_fn_code = _extract_reward_fn_code(llm_response)
            recommended_steps = min(_extract_steps(llm_response), args.max_steps)

            if reward_fn_code is None or not _validate_code(reward_fn_code):
                print("[WARN] 有効なコードブロックが得られませんでした。シェーピングなしで訓練します。")
                reward_fn_code = (
                    "import numpy as np\n\n"
                    "def reward_shaping(obs, prev_obs, base_reward):\n"
                    "    return 0.0\n"
                )

            print(f"[INFO] 推奨ステップ数: {recommended_steps:,}")

            # reward_fn.py 保存
            reward_fn_path = iter_dir / "reward_fn.py"
            reward_fn_path.write_text(reward_fn_code, encoding="utf-8")
            print(f"[INFO] reward_fn.py 保存: {reward_fn_path}")

            # 動的ロード & セット
            try:
                env._reward_fn = _load_reward_fn(reward_fn_path)
                print("[INFO] reward_fn をロードしました。")
            except Exception as e:
                print(f"[ERROR] reward_fn のロードに失敗: {e}")
                env._reward_fn = None

            # --- PPO 訓練 ---
            model = PPO("MlpPolicy", env, verbose=1)
            metrics_cb = _EurekaMetricsCallback(
                eval_freq=args.eval_freq,
                min_steps=args.min_steps,
                patience=args.patience,
                delta=args.delta,
            )

            print(f"[INFO] 訓練開始: {recommended_steps:,} steps (上限: {args.max_steps:,})")
            model.learn(total_timesteps=recommended_steps, callback=metrics_cb,
                        reset_num_timesteps=True)

            # --- メトリクス収集・保存 ---
            metrics = metrics_cb.get_metrics()
            print(f"[INFO] metrics: {metrics}")
            (iter_dir / "metrics.json").write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            # --- モデル保存 ---
            model.save(str(iter_dir / "model"))
            print(f"[INFO] モデル保存: {iter_dir / 'model.zip'}")

            # --- best 更新 ---
            coins = metrics.get("coins_per_episode", 0.0)
            if coins > best_coins:
                best_coins = coins
                best_iter = i
                best_dir = run_dir / "best"
                best_dir.mkdir(exist_ok=True)
                shutil.copy(reward_fn_path, best_dir / "reward_fn.py")
                shutil.copy(iter_dir / "metrics.json", best_dir / "metrics.json")
                print(f"[INFO] best 更新: coins_per_episode={coins:.3f} (iter {i})")

            prev_metrics = metrics

            # イテレーション完了後に状態を保存（次回再開用）
            _save_state(run_dir, i + 1, prev_metrics, best_iter, best_coins)

    except KeyboardInterrupt:
        print("\n[INFO] ループを中断しました。")

    finally:
        env.close()

    # 正常完了時は state.json を削除
    (run_dir / _STATE_FILENAME).unlink(missing_ok=True)

    # --- サマリー ---
    print(f"\n{'=' * 60}")
    print(f"[EUREKA] 完了  ({args.iterations} イテレーション)")
    if best_iter >= 0:
        print(f"  最良イテレーション: iter_{best_iter:03d}  coins/ep={best_coins:.3f}")
        print(f"  best/reward_fn.py: {run_dir / 'best' / 'reward_fn.py'}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
