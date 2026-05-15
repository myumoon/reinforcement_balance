"""EUREKA型報酬シェーピングループ（汎用エンジン）。

LLMが報酬シェーピング関数を反復的に生成・改良する汎用ループ。
ゲーム固有の処理は --game-config で指定する設定ファイルに委譲する。

使い方:
  python eureka_loop.py --game survivors --version-name v7 --game-config games/survivors/survivors_eureka_config.py --iterations 5
  python eureka_loop.py --game coin --version-name v2 --game-config games/coin/coin_eureka_config.py --llm openai --model gpt-4o --iterations 3
  python eureka_loop.py --help

出力先: runs/<game>/<version-name>/eureka/<run-name>/
"""

from __future__ import annotations

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
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

_STATE_FILENAME = "state.json"

_DEFAULT_STEPS = 50_000
_DEFAULT_ANTHROPIC_MODEL = "claude-opus-4-6"
_DEFAULT_OPENAI_MODEL = "gpt-4o"


def _log_device_status(requested_device: str) -> None:
    print(f"[INFO] requested device: {requested_device}")
    print(f"[INFO] torch={torch.__version__}, cuda_available={torch.cuda.is_available()}, "
          f"torch_cuda={torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"[INFO] cuda device[0]: {torch.cuda.get_device_name(0)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    # 事前パース: --config のみ抽出（他の引数は無視）
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None)
    pre_args, _ = pre.parse_known_args()

    p = argparse.ArgumentParser(description="EUREKA型報酬シェーピングループ")
    p.add_argument("--config", type=Path, default=None,
                   help="YAML 設定ファイルのパス（CLI 引数で上書き可能）")
    p.add_argument("--game-config", default=None,
                   help="ゲーム設定ファイルのパス（--config YAML または直接指定）")
    p.add_argument("--iterations", type=int, default=5, help="ループ回数（default: 5）")
    p.add_argument("--game",         required=True,
                   help="ゲーム種別 (survivors/coin/balance)")
    p.add_argument("--version-name", required=True,
                   help="バージョン名（runs/<game>/<version-name>/eureka/<run-name>/ に保存）")
    p.add_argument("--run-name", default=None,
                   help="保存ディレクトリ名（未指定時はタイムスタンプ自動生成）")
    p.add_argument("--resume", metavar="RUN_NAME", default=None,
                   help="前回中断したランを再開する（state.json が必要）。--run-name と排他")
    p.add_argument("--llm", choices=["anthropic", "openai"], default="anthropic",
                   help="LLMバックエンド（default: anthropic）")
    p.add_argument("--model", default=None,
                   help="LLMモデル名（未指定時: anthropic=claude-opus-4-6, openai=gpt-4o）")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=None,
                   help="サーバーポート（未指定時はゲーム設定のデフォルト: coin=8766, survivors=8767）")
    p.add_argument("--max-steps", type=int, default=200_000,
                   help="LLM推奨ステップ数を使う場合の上限（default: 200000）")
    p.add_argument("--min-steps", type=int, default=30_000,
                   help="実学習ステップ数の下限、かつ早期終了チェック開始ステップ（default: 30000）")
    p.add_argument("--steps-per-iteration", type=int, default=None,
                   help="Actual training steps per iteration. Overrides the LLM recommendation when set.")
    p.add_argument("--env-log-freq", type=int, default=1_000,
                   help="Environment diagnostics log interval in RL steps. Set 0 to disable.")
    p.add_argument("--device", default="auto",
                   help="SB3/PyTorch device (auto, cpu, cuda, cuda:0 など。default: auto)")
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
    p.add_argument("--frame-skip", type=int, default=1,
                   help="フレームスキップ数 N: 1 RL ステップで N 物理ステップ実行（default: 1）")
    p.add_argument("--curriculum", action="store_true",
                   help="カリキュラム学習を有効化（game_config が対応する場合のみ）")
    p.add_argument("--curriculum-window", type=int, default=20,
                   help="active_score を平均するエピソード数 (default: 20)")
    p.add_argument("--curriculum-threshold", type=float, default=5.0,
                   help="Stage 昇格の active_score 閾値 (default: 5.0, 目安: 撃破数×2.0+収集数×3.0)")
    p.add_argument("--curriculum-alive-reward", type=float, default=0.001,
                   help="生存ボーナスの1物理ステップあたりの値 (default: 0.001, UE5側と合わせる)")
    p.add_argument("--initial-observation", default=None,
                   help="1回目イテレーションでLLMに伝える訓練課題テキスト（直接指定）")
    p.add_argument("--initial-observation-file", type=Path, default=None,
                   help="1回目イテレーションでLLMに伝える訓練課題テキストのファイルパス")
    p.add_argument("--wandb", action="store_true", help="W&B ログを有効にする")
    p.add_argument("--wandb-project", default="eureka-loop", help="W&B プロジェクト名")
    p.add_argument("--wandb-run-name", default=None, help="W&B ラン名")

    # YAML があればデフォルトを差し込む（CLI が常に優先）
    if pre_args.config:
        from common.config import load_yaml_config, apply_yaml_defaults
        apply_yaml_defaults(p, load_yaml_config(pre_args.config))

    args = p.parse_args()
    if args.game_config is None:
        p.error("--game-config を指定してください（CLI または --config の YAML 内）")
    return args


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


def _call_llm(client, model_name: str, content: str | list, backend: str,
              max_tokens: int = 4096) -> str:
    if backend == "anthropic":
        response = client.messages.create(
            model=model_name,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": content}],
        )
        return response.content[0].text
    else:
        text = ("".join(b.get("text", "") for b in content)
                if isinstance(content, list) else content)
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": text}],
        )
        return response.choices[0].message.content


def _make_cached_content(static: str, dynamic: str) -> list:
    """Anthropic プロンプトキャッシュ用のコンテンツブロックを構築する。"""
    return [
        {"type": "text", "text": static, "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": dynamic},
    ]


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


def _resolve_training_steps(llm_steps: int, args: argparse.Namespace) -> int:
    if args.steps_per_iteration is not None:
        return max(1_000, args.steps_per_iteration)

    steps = max(llm_steps, args.min_steps)
    if args.max_steps >= args.min_steps:
        steps = min(steps, args.max_steps)
    return steps


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


def _save_source_of_truth(game_config, source_of_truth: dict | None, out_dir: Path,
                          json_name: str = "source_of_truth.json",
                          md_name: str = "source_of_truth.md") -> None:
    if not source_of_truth:
        return
    (out_dir / json_name).write_text(
        json.dumps(source_of_truth, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    rendered = game_config.render_source_of_truth(source_of_truth)
    if rendered:
        (out_dir / md_name).write_text(rendered, encoding="utf-8")


def _format_validation_findings(findings: list[dict]) -> str:
    if not findings:
        return "問題なし"
    lines = []
    for finding in findings:
        severity = finding.get("severity", "warning")
        code = finding.get("code", "VALIDATION")
        message = finding.get("message", "")
        pattern = finding.get("pattern", "")
        lines.append(f"- [{severity}] {code}: {message} (pattern: `{pattern}`)")
    return "\n".join(lines)


def _has_validation_errors(findings: list[dict]) -> bool:
    return any(finding.get("severity") == "error" for finding in findings)


def _save_validation_findings(findings: list[dict], out_dir: Path,
                              basename: str = "reward_validation") -> None:
    (out_dir / f"{basename}.json").write_text(
        json.dumps(findings, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / f"{basename}.txt").write_text(
        _format_validation_findings(findings),
        encoding="utf-8",
    )


def _call_build_prompt_parts(game_config, prev_metrics, iteration, prev_review_findings,
                             obs_for_iter, source_of_truth):
    try:
        return game_config.build_prompt_parts(
            prev_metrics, iteration, prev_review_findings,
            initial_observation=obs_for_iter,
            source_of_truth=source_of_truth,
        )
    except TypeError:
        return game_config.build_prompt_parts(
            prev_metrics, iteration, prev_review_findings,
            initial_observation=obs_for_iter,
        )


def _call_build_prompt(game_config, prev_metrics, iteration, prev_review_findings,
                       obs_for_iter, source_of_truth):
    try:
        return game_config.build_prompt(
            prev_metrics, iteration, prev_review_findings,
            initial_observation=obs_for_iter,
            source_of_truth=source_of_truth,
        )
    except TypeError:
        return game_config.build_prompt(
            prev_metrics, iteration, prev_review_findings,
            initial_observation=obs_for_iter,
        )


def _call_build_game_context(game_config, source_of_truth):
    try:
        return game_config.build_game_context(source_of_truth=source_of_truth)
    except TypeError:
        return game_config.build_game_context()


def _call_build_constraints_hint(game_config, source_of_truth):
    try:
        return game_config.build_constraints_hint(source_of_truth=source_of_truth)
    except TypeError:
        return game_config.build_constraints_hint()


def _build_review_static(game_context: str) -> str:
    """レビュープロンプトの静的部分（Anthropic キャッシュ対象）。"""
    ctx_section = f"## ゲームコンテキスト\n{game_context}\n\n---" if game_context.strip() else ""
    return (
        "あなたは強化学習の報酬設計レビュアーです。\n"
        "以下のゲームコンテキストと前回の訓練メトリクスをもとに reward_fn.py を評価してください。\n"
        "コードの修正は不要です。問題点のフィードバックのみを返してください。"
        + (f"\n\n{ctx_section}" if ctx_section else "")
    )


def _build_review_dynamic(reward_fn_code: str, prev_metrics: dict | None) -> str:
    """レビュープロンプトの動的部分（メトリクス値・レビュー対象コード）。"""
    metrics_section = (
        "なし（初回）"
        if prev_metrics is None
        else json.dumps(prev_metrics, ensure_ascii=False, indent=2)
    )
    return (
        f"## 前回の訓練メトリクス\n{metrics_section}\n\n"
        f"## レビュー対象の reward_fn.py\n```python\n{reward_fn_code}\n```\n\n"
        "上記のゲームルール・物理定数・固定報酬・前回メトリクスをもとに reward_fn.py を評価し、\n"
        "問題点を箇条書きで返してください。問題がなければ「問題なし」と記載してください。\n"
        "コードブロックは出力しないでください。"
    )


def _build_review_prompt(reward_fn_code: str, game_context: str,
                         prev_metrics: dict | None) -> str:
    return _build_review_static(game_context) + "\n\n" + _build_review_dynamic(reward_fn_code, prev_metrics)


def _build_revision_prompt(reward_fn_code: str, review_text: str,
                            constraints_hint: str = "") -> str:
    constraints_section = f"\n\n## 設計制約（参考）\n{constraints_hint}" if constraints_hint else ""
    return f"""以下は、あなたが生成した reward_fn.py とそれに対するレビュアーのフィードバックです。
レビュー意見が正しいかどうかを判断し、妥当な指摘があれば修正した最終版を返してください。
不正確な指摘は無視して構いません。{constraints_section}

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
                 min_steps: int = 30_000, patience: int = 3, delta: float = 0.05,
                 env_log_freq: int = 1_000):
        super().__init__(verbose=0)
        self._compute_metric = compute_metric
        self.eval_freq = eval_freq
        self.min_steps = min_steps
        self.patience = patience
        self.delta = delta
        self.env_log_freq = env_log_freq

        self._ep_base = 0.0
        self._ep_shaped = 0.0
        self._ep_hp = 0.0
        self._ep_len = 0
        self.episode_base_rewards: list[float] = []
        self.episode_shaped_rewards: list[float] = []
        self.episode_hp_rewards: list[float] = []
        self.episode_lengths: list[int] = []

        self._check_history: list[float] = []
        self._no_improve_count = 0
        self._last_check_step = 0
        self._last_env_log_step = 0
        self._zero_enemy_steps = 0
        self._terminated_count = 0
        self._truncated_count = 0

    def _on_step(self) -> bool:
        info = self.locals["infos"][0]
        spawn_debug = info.get("spawn_debug", {})
        self._ep_base += info.get("base_reward", 0.0)
        self._ep_shaped += info.get("shaped_reward", 0.0)
        self._ep_hp += info.get("hp_penalty", 0.0)
        self._ep_len += 1

        if self.locals["dones"][0]:
            if info.get("TimeLimit.truncated", False) or spawn_debug.get("truncated", False):
                self._truncated_count += 1
            else:
                self._terminated_count += 1
            self.episode_base_rewards.append(self._ep_base)
            self.episode_shaped_rewards.append(self._ep_shaped)
            self.episode_hp_rewards.append(self._ep_hp)
            self.episode_lengths.append(self._ep_len)
            self._ep_base = 0.0
            self._ep_shaped = 0.0
            self._ep_hp = 0.0
            self._ep_len = 0

        if spawn_debug:
            if spawn_debug.get("enemy_count", 0) == 0:
                self._zero_enemy_steps += 1
            else:
                self._zero_enemy_steps = 0

            if self.env_log_freq > 0 and self.num_timesteps - self._last_env_log_step >= self.env_log_freq:
                self._last_env_log_step = self.num_timesteps
                print(
                    "[ENV] "
                    f"step={self.num_timesteps:,} "
                    f"t={spawn_debug.get('elapsed_time', 0.0):.1f}s "
                    f"enemy={spawn_debug.get('enemy_count', 0)} "
                    f"wave={spawn_debug.get('current_wave_index', -1)} "
                    f"allowed={spawn_debug.get('allowed_spawn_type_count', 0)} "
                    f"max_type={spawn_debug.get('max_enemy_type_id', 0)} "
                    f"blocked={spawn_debug.get('spawn_blocked', False)} "
                    f"truncated={spawn_debug.get('truncated', False)}"
                )

            if self._zero_enemy_steps == self.env_log_freq and self.env_log_freq > 0:
                print(
                    "[WARN] enemy_count stayed at 0 "
                    f"for {self._zero_enemy_steps:,} RL steps "
                    f"(elapsed={spawn_debug.get('elapsed_time', 0.0):.1f}s, "
                    f"wave={spawn_debug.get('current_wave_index', -1)}, "
                    f"blocked={spawn_debug.get('spawn_blocked', False)})"
                )

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
            return {
                "episodes": 0,
                "terminated_episodes": self._terminated_count,
                "truncated_episodes": self._truncated_count,
            }
        mean_base   = sum(self.episode_base_rewards) / n
        mean_shaped = sum(self.episode_shaped_rewards) / n
        mean_hp     = sum(self.episode_hp_rewards) / n if self.episode_hp_rewards else 0.0
        mean_len    = sum(self.episode_lengths) / n
        primary     = game_config.compute_primary_metric(
            self.episode_base_rewards, self.episode_lengths)
        extra = game_config.compute_extra_metrics(
            self.episode_base_rewards, self.episode_lengths)
        return {
            "episodes": n,
            "terminated_episodes": self._terminated_count,
            "truncated_episodes": self._truncated_count,
            "episode_reward_mean": round(mean_base + mean_shaped + mean_hp, 4),
            "base_reward_mean": round(mean_base, 4),
            "shaped_reward_mean": round(mean_shaped, 4),
            "hp_penalty_mean": round(mean_hp, 4),
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
    if args.resume and args.run_name:
        print("[ERROR] --resume と --run-name は同時に指定できません。")
        return

    eureka_root = Path("runs") / args.game / args.version_name / "eureka"

    if args.resume:
        run_name = args.resume
        run_dir = eureka_root / run_name
        if not (run_dir / _STATE_FILENAME).exists():
            print(f"[ERROR] {run_dir / _STATE_FILENAME} が見つかりません。再開できません。")
            print(f"[INFO]  新規開始する場合は --run-name {run_name} を使用してください。")
            return
    else:
        run_name = args.run_name or datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
        run_dir = eureka_root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] 保存先: {run_dir}")

    _use_wandb = args.wandb and _WANDB_AVAILABLE
    if _use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or run_name,
            config={
                "game_config": str(args.game_config),
                "iterations": args.iterations,
                "llm": args.llm,
                "max_steps": args.max_steps,
                "min_steps": args.min_steps,
                "steps_per_iteration": args.steps_per_iteration,
                "device": args.device,
            },
        )

    # LLM クライアント
    client, model_name = _build_llm_client(args.llm, args.model)
    print(f"[INFO] LLM: {args.llm} / {model_name}")
    _log_device_status(args.device)

    # ポート解決: 未指定時はゲーム設定のデフォルトを使用
    port = args.port if args.port is not None else game_config.default_port
    print(f"[INFO] port: {port}")

    # ゲーム設定の初期化（obs_schema 取得など）
    game_config.setup(args.host, port)
    source_of_truth = game_config.build_source_of_truth(args.host, port)
    if source_of_truth:
        _save_source_of_truth(
            game_config,
            source_of_truth,
            run_dir,
            json_name="source_of_truth_latest.json",
            md_name="source_of_truth_latest.md",
        )
        print(f"[INFO] source_of_truth_latest 保存: {run_dir}")

    # 環境作成
    raw_env = game_config.make_env(args.host, port, frame_skip=args.frame_skip)
    if args.no_vec_normalize:
        env = raw_env
        print("[INFO] VecNormalize 無効")
    else:
        env = _wrap_vec_normalize(raw_env)
        print("[INFO] VecNormalize 有効 (norm_obs=True, norm_reward=True, clip_obs=10.0)")

    # 初回イテレーション用の訓練観察テキストを構築
    initial_observation: str | None = None
    if args.initial_observation:
        initial_observation = args.initial_observation
    elif args.initial_observation_file:
        initial_observation = Path(args.initial_observation_file).read_text(encoding="utf-8").strip()
    if initial_observation:
        print(f"[INFO] 初回イテレーションに訓練観察を追加します ({len(initial_observation)} 文字)")

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
            _save_source_of_truth(game_config, source_of_truth, iter_dir)

            # --- LLM に報酬関数を生成させる ---
            print("[INFO] LLM に報酬関数を生成中...")
            obs_for_iter = initial_observation if i == 1 else None
            if args.llm == "anthropic":
                static_prefix, dynamic_suffix = _call_build_prompt_parts(
                    game_config, prev_metrics, i, prev_review_findings,
                    obs_for_iter, source_of_truth)
                gen_content = (_make_cached_content(static_prefix, dynamic_suffix)
                               if static_prefix else dynamic_suffix)
            else:
                gen_content = _call_build_prompt(
                    game_config, prev_metrics, i, prev_review_findings,
                    obs_for_iter, source_of_truth)
            try:
                llm_response = _call_llm(client, model_name, gen_content, args.llm,
                                         max_tokens=8192)
            except Exception as e:
                if _is_credit_error(e):
                    _save_state(run_dir, i, prev_metrics, best_iter, best_primary,
                                prev_review_findings)
                    print(f"\n[ERROR] クレジット残高が不足しています。")
                    print(f"[INFO]  https://console.anthropic.com/settings/billing でクレジットを追加してください。")
                    print(f"[INFO]  追加後、以下のコマンドで再開できます:")
                    print(f"[INFO]    python eureka_loop.py --game-config {args.game_config} "
                          f"--iterations {args.iterations} --resume {run_name}")
                    return
                raise

            # レスポンス保存
            (iter_dir / "llm_response.txt").write_text(llm_response, encoding="utf-8")

            # コード・ステップ数を抽出。LLMの推奨値は記録し、実学習ステップは引数側で決める。
            reward_fn_code = _extract_reward_fn_code(llm_response)
            llm_steps = _extract_steps(llm_response)
            training_steps = _resolve_training_steps(llm_steps, args)

            if reward_fn_code is None or not _validate_code(reward_fn_code):
                print("[WARN] 有効なコードブロックが得られませんでした。シェーピングなしで訓練します。")
                reward_fn_code = (
                    "import numpy as np\n\n"
                    "def reward_shaping(obs, prev_obs, base_reward):\n"
                    "    return 0.0\n"
                )
            else:
                initial_validation = game_config.validate_reward_code(reward_fn_code, source_of_truth)
                if initial_validation:
                    _save_validation_findings(initial_validation, iter_dir, "reward_validation_initial")
                    print("[WARN] reward_fn semantic validation findings:")
                    print(_format_validation_findings(initial_validation))

                # レビュー＆改訂ステップ（--no-review で無効化可能）
                if not args.no_review:
                    # Step A: レビュアー LLM がフィードバックのみ返す（コードは書かない）
                    print("[INFO] reward_fn をレビュー中...")
                    game_context = _call_build_game_context(game_config, source_of_truth)
                    validation_review_section = (
                        "\n\n## 機械検証で検出された問題\n"
                        f"{_format_validation_findings(initial_validation)}"
                        if initial_validation else ""
                    )
                    if args.llm == "anthropic" and game_context.strip():
                        review_content = _make_cached_content(
                            _build_review_static(game_context),
                            _build_review_dynamic(reward_fn_code, prev_metrics)
                            + validation_review_section,
                        )
                    else:
                        review_content = (
                            _build_review_prompt(reward_fn_code, game_context, prev_metrics)
                            + validation_review_section
                        )
                    review_response = _call_llm(client, model_name, review_content, args.llm)
                    if initial_validation:
                        review_response = (
                            f"{review_response}\n\n"
                            "## 機械検証で検出された問題\n"
                            f"{_format_validation_findings(initial_validation)}"
                        )
                    (iter_dir / "review_response.txt").write_text(review_response, encoding="utf-8")

                    # Step B: 生成 LLM がレビュー意見を判断して最終版を実装
                    print("[INFO] レビュー意見をもとに改訂中...")
                    constraints_hint = _call_build_constraints_hint(game_config, source_of_truth)
                    revision_prompt = _build_revision_prompt(reward_fn_code, review_response, constraints_hint)
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

                final_validation = game_config.validate_reward_code(reward_fn_code, source_of_truth)
                _save_validation_findings(final_validation, iter_dir)
                if _has_validation_errors(final_validation):
                    print("[ERROR] reward_fn semantic validation failed:")
                    print(_format_validation_findings(final_validation))
                    _save_state(run_dir, i, prev_metrics, best_iter, best_primary,
                                prev_review_findings)
                    env.close()
                    return

            print(f"[INFO] LLM recommended steps: {llm_steps:,}")
            print(f"[INFO] Training steps used: {training_steps:,}")

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
            model = game_config.make_model(env, device=args.device)
            print(f"[INFO] SB3 model device: {model.device}")
            metrics_cb = _EurekaMetricsCallback(
                compute_metric=game_config.compute_primary_metric,
                eval_freq=args.eval_freq,
                min_steps=args.min_steps,
                patience=args.patience,
                delta=args.delta,
                env_log_freq=args.env_log_freq,
            )

            print(f"[INFO] 訓練開始: {training_steps:,} steps (上限: {args.max_steps:,})")
            loop_callbacks = [metrics_cb]
            curriculum_cb = None
            if args.curriculum:
                curriculum_status_path = iter_dir / "curriculum_status.json"
                curriculum_cb = game_config.make_curriculum_callback(
                    raw_env=raw_env,
                    frame_skip=args.frame_skip,
                    window=args.curriculum_window,
                    threshold_mult=args.curriculum_threshold,
                    alive_reward=args.curriculum_alive_reward,
                    status_path=curriculum_status_path,
                )
                if curriculum_cb is not None:
                    loop_callbacks.append(curriculum_cb)
                    print(f"[INFO] CurriculumCallback 有効 "
                          f"(window={args.curriculum_window}, threshold_mult={args.curriculum_threshold}, "
                          f"frame_skip={args.frame_skip}, alive_reward={args.curriculum_alive_reward})")
                else:
                    print("[WARN] --curriculum はこの game_config では未対応です。無視します。")
            model.learn(total_timesteps=training_steps, callback=loop_callbacks,
                        reset_num_timesteps=True)

            # --- メトリクス収集・保存 ---
            metrics = metrics_cb.get_metrics(game_config)
            if curriculum_cb is not None:
                curriculum_metrics = game_config.collect_curriculum_metrics(curriculum_cb)
                if curriculum_metrics:
                    metrics["curriculum"] = curriculum_metrics
                    curriculum_diag_path = iter_dir / "curriculum_diagnostics.json"
                    curriculum_diag_path.write_text(
                        json.dumps(curriculum_metrics, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    for line in game_config.format_curriculum_summary(curriculum_metrics):
                        print(line)
                    print(f"[INFO] curriculum_diagnostics.json: {curriculum_diag_path}")
            print(f"[INFO] metrics: {metrics}")
            if _use_wandb and wandb.run:
                log_dict = {"iteration": i, **metrics}
                if best_primary is not None:
                    log_dict["best/primary_metric"] = best_primary
                wandb.log(log_dict, step=i)
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
                if _use_wandb and wandb.run:
                    artifact = wandb.Artifact("best-reward-fn", type="reward_function")
                    artifact.add_file(str(best_dir / "reward_fn.py"))
                    wandb.log_artifact(artifact)

            prev_metrics = metrics

            # イテレーション完了後に状態を保存（次回再開用）
            _save_state(run_dir, i + 1, prev_metrics, best_iter, best_primary,
                        prev_review_findings)

    except KeyboardInterrupt:
        print("\n[INFO] ループを中断しました。")

    finally:
        env.close()
        if _use_wandb:
            wandb.finish()



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
