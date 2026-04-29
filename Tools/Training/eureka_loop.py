"""EUREKA型報酬シェーピングループ（汎用エンジン）。

LLMが報酬シェーピング関数を反復的に生成・改良する汎用ループ。
ゲーム固有の処理は --game-config で指定する設定ファイルに委譲する。

使い方:
  python eureka_loop.py --game-config games/coin_eureka_config.py --iterations 5
  python eureka_loop.py --game-config games/coin_eureka_config.py --iterations 10 --run-name my_run
  python eureka_loop.py --game-config games/coin_eureka_config.py --llm openai --model gpt-4o --iterations 3
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
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

_STATE_FILENAME = "state.json"

_DEFAULT_STEPS = 50_000
_DEFAULT_ANTHROPIC_MODEL = "claude-opus-4-6"
_DEFAULT_OPENAI_MODEL = "gpt-4o"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EUREKA型報酬シェーピングループ")
    p.add_argument("--game-config", required=True,
                   help="ゲーム設定ファイルのパス（例: games/coin_eureka_config.py）")
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
                   help="改善とみなす primary_metric の最小増加量（default: 0.05）")
    p.add_argument("--no-vec-normalize", action="store_true",
                   help="VecNormalize による観測・報酬の正規化を無効化する")
    p.add_argument("--no-review", action="store_true",
                   help="reward_fn のレビュー＆改訂ステップをスキップする")
    return p.parse_args()


# ---------------------------------------------------------------------------
# ゲーム設定の動的ロード
# ---------------------------------------------------------------------------

def _load_game_config(path_str: str):
    path = Path(path_str)
    if not path.exists():
        print(f"[ERROR] --game-config ファイルが見つかりません: {path}")
        sys.exit(1)
    spec = importlib.util.spec_from_file_location("_eureka_game_config_mod", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "create_config"):
        print(f"[ERROR] {path} に create_config() 関数が定義されていません。")
        sys.exit(1)
    return mod.create_config()


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


def _call_llm(client, model_name: str, prompt: str, backend: str,
              max_tokens: int = 4096) -> str:
    if backend == "anthropic":
        response = client.messages.create(
            model=model_name,
            max_tokens=max_tokens,
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
# レスポンス解析
# ---------------------------------------------------------------------------

def _extract_reward_fn_code(text: str) -> str | None:
    match = re.search(r"```python\n(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else None


def _extract_steps(text: str) -> int:
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


def _extract_revision_judgment(revision_text: str) -> str | None:
    """revision_response から ### 判断 セクションのテキストを抽出する。"""
    m = re.search(r"###\s*判断\s*\n(.*?)(?=\n###|\Z)", revision_text, re.DOTALL)
    return m.group(1).strip() if m else None


def _build_review_findings(review_response: str, revision_response: str) -> str:
    """次イテレーションのプロンプトに渡す前回レビュー知見テキストを構築する。"""
    parts = [f"### レビュアーが指摘した問題点\n{review_response.strip()}"]
    judgment = _extract_revision_judgment(revision_response)
    if judgment:
        parts.append(f"### 生成LLMの判断（修正・棄却した内容）\n{judgment}")
    return "\n\n".join(parts)


def _validate_code(code: str) -> bool:
    try:
        compile(code, "<reward_fn>", "exec")
        return True
    except SyntaxError as e:
        print(f"[WARN] 生成コードに構文エラー: {e}")
        return False


def _build_review_prompt(reward_fn_code: str, game_context: str,
                         prev_metrics: dict | None) -> str:
    ctx_section = (
        f"## ゲームコンテキスト\n{game_context}\n\n---\n\n"
        if game_context.strip()
        else ""
    )
    metrics_section = (
        "なし（初回）"
        if prev_metrics is None
        else json.dumps(prev_metrics, ensure_ascii=False, indent=2)
    )
    return f"""あなたは強化学習の報酬設計レビュアーです。
以下のゲームコンテキストと前回の訓練メトリクスをもとに reward_fn.py を評価してください。
コードの修正は不要です。問題点のフィードバックのみを返してください。

{ctx_section}## 前回の訓練メトリクス
{metrics_section}

## レビュー対象の reward_fn.py
```python
{reward_fn_code}
```

上記のゲームルール・物理定数・固定報酬・前回メトリクスをもとに reward_fn.py を評価し、
問題点を箇条書きで返してください。問題がなければ「問題なし」と記載してください。
コードブロックは出力しないでください。
"""


def _build_revision_prompt(reward_fn_code: str, review_text: str,
                            original_prompt: str) -> str:
    return f"""以下は、あなたが生成した reward_fn.py とそれに対するレビュアーのフィードバックです。
レビュー意見が正しいかどうかを判断し、妥当な指摘があれば修正した最終版を返してください。
不正確な指摘は無視して構いません。

## 元の生成プロンプト（参考）
{original_prompt}

## あなたが生成した初版 reward_fn.py
```python
{reward_fn_code}
```

## レビュアーのフィードバック
{review_text}

## レスポンス形式

### 判断
（レビュー意見の妥当性を評価し、修正方針を説明。修正不要なら理由を記載）

### 最終 reward_fn.py
```python
import numpy as np

def reward_shaping(obs: np.ndarray, prev_obs: np.ndarray, base_reward: float) -> float:
    shaped = 0.0
    return float(np.clip(shaped, -1.0, 1.0))
```
"""


# ---------------------------------------------------------------------------
# VecNormalize ユーティリティ
# ---------------------------------------------------------------------------

def _wrap_vec_normalize(raw_env):
    """raw gym.Env を Monitor → DummyVecEnv → VecNormalize の順にラップする。

    Monitor を先に適用しないと SB3 の ep_info_buffer が更新されず、
    rollout/ セクション (ep_rew_mean, ep_len_mean) がログに出力されない。
    """
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    vec_env = DummyVecEnv([lambda: Monitor(raw_env)])
    return VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)


def _get_raw_env(env):
    """VecEnv ラッパーチェーンを辿って最初の生環境を返す。"""
    inner = env
    while hasattr(inner, "venv"):
        inner = inner.venv
    if hasattr(inner, "envs"):
        inner = inner.envs[0]
    if hasattr(inner, "unwrapped"):
        inner = inner.unwrapped
    return inner


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

    def __init__(self, compute_metric: Callable, eval_freq: int = 10_000,
                 min_steps: int = 30_000, patience: int = 3, delta: float = 0.05):
        super().__init__(verbose=0)
        self._compute_metric = compute_metric
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

        if (self.num_timesteps >= self.min_steps and
                self.num_timesteps - self._last_check_step >= self.eval_freq and
                len(self.episode_base_rewards) >= self._RECENT_WINDOW):
            self._last_check_step = self.num_timesteps
            recent_base = self.episode_base_rewards[-self._RECENT_WINDOW:]
            recent_len  = self.episode_lengths[-self._RECENT_WINDOW:]
            current = self._compute_metric(recent_base, recent_len)

            if self._check_history and current <= max(self._check_history) + self.delta:
                self._no_improve_count += 1
            else:
                self._no_improve_count = 0
            self._check_history.append(current)

            print(f"[INFO] プラトーチェック ({self.num_timesteps:,} steps): "
                  f"metric={current:.3f}, "
                  f"改善なし={self._no_improve_count}/{self.patience}")

            if self._no_improve_count >= self.patience:
                print("[INFO] プラトー検出: 早期終了します")
                return False

        return True

    def get_metrics(self, game_config) -> dict:
        n = len(self.episode_base_rewards)
        if n == 0:
            return {"episodes": 0}
        mean_base   = sum(self.episode_base_rewards) / n
        mean_shaped = sum(self.episode_shaped_rewards) / n
        mean_len    = sum(self.episode_lengths) / n
        primary     = game_config.compute_primary_metric(
            self.episode_base_rewards, self.episode_lengths)
        extra = game_config.compute_extra_metrics(
            self.episode_base_rewards, self.episode_lengths)
        return {
            "episodes": n,
            "episode_reward_mean": round(mean_base + mean_shaped, 4),
            "base_reward_mean": round(mean_base, 4),
            "shaped_reward_mean": round(mean_shaped, 4),
            "episode_length_mean": round(mean_len, 1),
            game_config.primary_metric_name: round(primary, 3),
            **extra,
        }


# ---------------------------------------------------------------------------
# 状態の保存・読み込み（再開用）
# ---------------------------------------------------------------------------

def _save_state(run_dir: Path, next_iter: int, prev_metrics: dict | None,
                best_iter: int, best_primary: float,
                prev_review_findings: str | None = None) -> None:
    state = {
        "next_iter": next_iter,
        "prev_metrics": prev_metrics,
        "best_iter": best_iter,
        "best_primary": best_primary,
        "prev_review_findings": prev_review_findings,
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

    # ゲーム設定ロード
    game_config = _load_game_config(args.game_config)

    # 出力ディレクトリ
    run_name = args.run_name or datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] 保存先: {run_dir}")

    # LLM クライアント
    client, model_name = _build_llm_client(args.llm, args.model)
    print(f"[INFO] LLM: {args.llm} / {model_name}")

    # ゲーム設定の初期化（obs_schema 取得など）
    game_config.setup(args.host, args.port)

    # 環境作成
    raw_env = game_config.make_env(args.host, args.port)
    if args.no_vec_normalize:
        env = raw_env
        print("[INFO] VecNormalize 無効")
    else:
        env = _wrap_vec_normalize(raw_env)
        print("[INFO] VecNormalize 有効 (norm_obs=True, norm_reward=True, clip_obs=10.0)")

    # 再開状態の読み込み
    state = _load_state(run_dir)
    if state:
        start_iter           = state["next_iter"]
        prev_metrics         = state.get("prev_metrics")
        best_iter            = state.get("best_iter", -1)
        best_primary         = state.get("best_primary", -1.0)
        prev_review_findings = state.get("prev_review_findings")
        print(f"[INFO] 前回の状態を検出: イテレーション {start_iter} から再開します。")
    else:
        start_iter           = 1
        prev_metrics         = None
        best_primary         = -1.0
        best_iter            = -1
        prev_review_findings = None

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
            prompt = game_config.build_prompt(prev_metrics, i, prev_review_findings)
            print("[INFO] LLM に報酬関数を生成中...")
            try:
                llm_response = _call_llm(client, model_name, prompt, args.llm,
                                         max_tokens=8192)
            except Exception as e:
                if _is_credit_error(e):
                    _save_state(run_dir, i, prev_metrics, best_iter, best_primary,
                                prev_review_findings)
                    print(f"\n[ERROR] クレジット残高が不足しています。")
                    print(f"[INFO]  https://console.anthropic.com/settings/billing でクレジットを追加してください。")
                    print(f"[INFO]  追加後、以下のコマンドで再開できます:")
                    print(f"[INFO]    python eureka_loop.py --game-config {args.game_config} "
                          f"--iterations {args.iterations} --run-name {run_name}")
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
            else:
                # レビュー＆改訂ステップ（--no-review で無効化可能）
                if not args.no_review:
                    # Step A: レビュアー LLM がフィードバックのみ返す（コードは書かない）
                    print("[INFO] reward_fn をレビュー中...")
                    game_context = game_config.build_game_context()
                    review_prompt = _build_review_prompt(reward_fn_code, game_context, prev_metrics)
                    review_response = _call_llm(client, model_name, review_prompt, args.llm)
                    (iter_dir / "review_response.txt").write_text(review_response, encoding="utf-8")

                    # Step B: 生成 LLM がレビュー意見を判断して最終版を実装
                    print("[INFO] レビュー意見をもとに改訂中...")
                    revision_prompt = _build_revision_prompt(reward_fn_code, review_response, prompt)
                    revision_response = _call_llm(client, model_name, revision_prompt, args.llm)
                    (iter_dir / "revision_response.txt").write_text(revision_response, encoding="utf-8")

                    revised_code = _extract_reward_fn_code(revision_response)
                    if revised_code is not None and _validate_code(revised_code):
                        (iter_dir / "reward_fn_original.py").write_text(reward_fn_code, encoding="utf-8")
                        reward_fn_code = revised_code
                        print("[INFO] 改訂済みコードを採用しました")
                    else:
                        print("[WARN] 改訂後のコード抽出に失敗。初版コードを使用します")

                    # 次イテレーションへ引き継ぐレビュー知見を構築
                    prev_review_findings = _build_review_findings(review_response, revision_response)

            print(f"[INFO] 推奨ステップ数: {recommended_steps:,}")

            # reward_fn.py 保存
            reward_fn_path = iter_dir / "reward_fn.py"
            reward_fn_path.write_text(reward_fn_code, encoding="utf-8")
            print(f"[INFO] reward_fn.py 保存: {reward_fn_path}")

            # 動的ロード & セット（VecNormalize ラッパーを通じて生の環境にセット）
            try:
                fn = _load_reward_fn(reward_fn_path)
                _get_raw_env(env)._reward_fn = fn
                print("[INFO] reward_fn をロードしました。")
            except Exception as e:
                print(f"[ERROR] reward_fn のロードに失敗: {e}")
                _get_raw_env(env)._reward_fn = None

            # --- PPO 訓練 ---
            model = game_config.make_model(env)
            metrics_cb = _EurekaMetricsCallback(
                compute_metric=game_config.compute_primary_metric,
                eval_freq=args.eval_freq,
                min_steps=args.min_steps,
                patience=args.patience,
                delta=args.delta,
            )

            print(f"[INFO] 訓練開始: {recommended_steps:,} steps (上限: {args.max_steps:,})")
            model.learn(total_timesteps=recommended_steps, callback=metrics_cb,
                        reset_num_timesteps=True)

            # --- メトリクス収集・保存 ---
            metrics = metrics_cb.get_metrics(game_config)
            print(f"[INFO] metrics: {metrics}")
            (iter_dir / "metrics.json").write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            # --- モデル・VecNormalize 保存 ---
            model.save(str(iter_dir / "model"))
            print(f"[INFO] モデル保存: {iter_dir / 'model.zip'}")
            from stable_baselines3.common.vec_env import VecNormalize as _VN
            if isinstance(env, _VN):
                env.save(str(iter_dir / "vec_normalize.pkl"))
                print(f"[INFO] VecNormalize 統計保存: {iter_dir / 'vec_normalize.pkl'}")

            # --- best 更新 ---
            primary = game_config.compute_primary_metric(
                metrics_cb.episode_base_rewards, metrics_cb.episode_lengths)
            if primary > best_primary:
                best_primary = primary
                best_iter = i
                best_dir = run_dir / "best"
                best_dir.mkdir(exist_ok=True)
                shutil.copy(reward_fn_path, best_dir / "reward_fn.py")
                shutil.copy(iter_dir / "metrics.json", best_dir / "metrics.json")
                print(f"[INFO] best 更新: {game_config.primary_metric_name}={primary:.3f} (iter {i})")

            prev_metrics = metrics

            # イテレーション完了後に状態を保存（次回再開用）
            _save_state(run_dir, i + 1, prev_metrics, best_iter, best_primary,
                        prev_review_findings)

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
        print(f"  最良イテレーション: iter_{best_iter:03d}  "
              f"{game_config.primary_metric_name}={best_primary:.3f}")
        print(f"  best/reward_fn.py: {run_dir / 'best' / 'reward_fn.py'}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
