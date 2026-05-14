"""Survivors BC 専用 CLI.

ルールベース方策でデモを収集し、行動クローニングで事前初期化した
PPO モデルを保存する。PPO 本番訓練は実行しない。

使い方（Tools/Training/ から実行）:
    python games/survivors/run_bc.py --run-name survivors_bc_v1 --episodes 100 --epochs 30

出力:
    results/survivors/<run-name>/
        model_bc.zip          BC 済みポリシーの重み
        vecnormalize_bc.pkl   VecNormalize 統計（--no-vec-normalize 時は省略）
        bc_status.json        BC 統計・メタ情報
        config_resolved.yaml  引数確定後の設定
        run_meta.json         実行環境情報
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# Tools/Training/ をパスに追加（run_bc.py は games/survivors/ 内にあるため）
_TRAINING_DIR = Path(__file__).resolve().parent.parent.parent
if str(_TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAINING_DIR))

import numpy as np

try:
    from sb3_contrib import RecurrentPPO
    _SB3_CONTRIB_AVAILABLE = True
except ImportError:
    _SB3_CONTRIB_AVAILABLE = False

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


# ──────────────────────────────────────────────
# 引数パース
# ──────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    pre_p = argparse.ArgumentParser(add_help=False)
    pre_p.add_argument("--config", type=Path, default=None)
    pre_args, _ = pre_p.parse_known_args()

    p = argparse.ArgumentParser(
        description="Survivors BC 専用 CLI: ルールベース方策でBC済みモデルを作成する"
    )
    p.add_argument("--config",       type=Path,  default=None,    help="YAML config（train_config_run.yaml 等）")
    p.add_argument("--run-name",     required=True,               help="run 名（results/survivors/<name>/ に保存）")
    p.add_argument("--episodes",     type=int,   default=100,     help="デモ収集エピソード数 (default: 100)")
    p.add_argument("--epochs",       type=int,   default=30,      help="BC 訓練エポック数 (default: 30)")
    p.add_argument("--lr",           type=float, default=3e-4,    help="BC Adam 学習率 (default: 3e-4)")
    p.add_argument("--batch-size",   type=int,   default=512,     help="BC ミニバッチサイズ (default: 512)")
    p.add_argument("--device",       default="auto",              help="学習デバイス (cpu|cuda|auto, default: auto)")
    p.add_argument("--host",         default="127.0.0.1",         help="UE5 ホスト (default: 127.0.0.1)")
    p.add_argument("--port",         type=int,   default=8767,    help="UE5 ポート (default: 8767)")
    p.add_argument("--frame-skip",   type=int,   default=4,       help="フレームスキップ数 (default: 4)")
    p.add_argument("--frame-stack",  type=int,   default=1,       help="フレームスタック数 (default: 1)")
    p.add_argument("--entity-attention", action="store_true",     help="EntityAttention 特徴抽出器を使用")
    p.add_argument("--recurrent",    action="store_true",         help="RecurrentPPO (MlpLstmPolicy) を使用")
    p.add_argument("--lstm-hidden-size", type=int, default=256,   help="LSTM 隠れ層サイズ (default: 256)")
    p.add_argument("--n-lstm-layers",    type=int, default=1,     help="LSTM レイヤー数 (default: 1)")
    p.add_argument("--reward-fn",    type=Path,  default=None,    help="EUREKA 報酬シェーピング関数")
    p.add_argument("--no-vec-normalize", action="store_true",     help="VecNormalize を無効化")
    p.add_argument("--wandb",        action="store_true",         help="W&B ログを有効化")
    p.add_argument("--wandb-project", default="survivors",        help="W&B プロジェクト名")
    p.add_argument("--wandb-run-name", default=None,              help="W&B ラン名（未指定時は --run-name を使用）")

    if pre_args.config:
        from common.config import load_yaml_config, apply_yaml_defaults
        apply_yaml_defaults(p, load_yaml_config(pre_args.config))

    return p.parse_args()


# ──────────────────────────────────────────────
# メイン
# ──────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    if args.recurrent and not _SB3_CONTRIB_AVAILABLE:
        raise ImportError("--recurrent には sb3-contrib が必要です")

    # run_dir 作成
    run_dir = Path("results") / "survivors" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    model_bc_path   = run_dir / "model_bc.zip"
    vecnorm_bc_path = run_dir / "vecnormalize_bc.pkl"
    status_path     = run_dir / "bc_status.json"

    if model_bc_path.exists():
        raise FileExistsError(
            f"既存の BC モデルがあります。--run-name を変えるか手動で削除してください: {model_bc_path}"
        )

    # config 保存
    _write_resolved_config(run_dir, args)
    _write_run_meta(run_dir, args)
    print(f"[INFO] run_dir: {run_dir}")

    # W&B 初期化
    _use_wandb = args.wandb and _WANDB_AVAILABLE
    if _use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or args.run_name,
            config={
                "bc_episodes": args.episodes,
                "bc_epochs":   args.epochs,
                "bc_lr":       args.lr,
                "bc_batch_size": args.batch_size,
                "frame_skip":  args.frame_skip,
                "frame_stack": args.frame_stack,
                "recurrent":   args.recurrent,
                "entity_attention": args.entity_attention,
            },
        )

    # reward_fn ロード
    reward_fn = None
    if args.reward_fn:
        reward_fn = _load_reward_fn(args.reward_fn)
        print(f"[INFO] reward_fn をロード: {args.reward_fn}")

    # 環境作成
    env = _make_env(args, reward_fn)
    print(f"[INFO] UE5 サーバー {args.host}:{args.port} に接続...")

    # PPO モデル作成（BC 用: ハイパーパラメータは最低限, 構造のみが重要）
    model = _make_model(args, env)
    print(f"[INFO] モデル作成完了 (device={model.device})")

    # BC 実行
    from games.survivors.survivors_bc import bc_warmup
    print(f"[INFO] BC 開始 (episodes={args.episodes}, epochs={args.epochs})")
    stats = bc_warmup(
        model=model,
        env=env,
        n_episodes=args.episodes,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        verbose=1,
    )

    # モデル保存
    model.save(str(run_dir / "model_bc"))
    print(f"[INFO] BC 済みモデルを保存: {model_bc_path}")

    # VecNormalize 統計保存
    vec_norm = _find_vecnormalize(env)
    if vec_norm is not None:
        vec_norm.save(str(vecnorm_bc_path))
        print(f"[INFO] VecNormalize 統計を保存: {vecnorm_bc_path}")

    # bc_status.json 保存
    _write_bc_status(status_path, args, stats, vec_norm is not None)
    print(f"[INFO] bc_status.json を保存: {status_path}")

    # W&B サマリ
    if _use_wandb and wandb.run:
        wandb.summary.update({
            "bc/transitions":           stats.get("transitions", 0),
            "bc/episode_rewards_mean":  stats.get("episode_rewards_mean", 0.0),
            "bc/episode_length_mean":   stats.get("episode_length_mean", 0.0),
            "bc/final_loss":            stats.get("final_loss", 0.0),
        })
        action_counts = stats.get("action_counts", [])
        if action_counts:
            total = max(sum(action_counts), 1)
            names = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "idle"]
            for i, (name, cnt) in enumerate(zip(names, action_counts)):
                wandb.summary[f"bc/action_{i}_{name}"] = cnt / total
        if stats.get("loss_history"):
            for ep, loss in enumerate(stats["loss_history"]):
                wandb.log({"bc/train_loss": loss, "bc/epoch": ep + 1})
        wandb.finish()

    env.close()
    print("[INFO] BC 完了")


# ──────────────────────────────────────────────
# 環境・モデル作成
# ──────────────────────────────────────────────

def _make_env(args, reward_fn):
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import VecNormalize, VecFrameStack
    from games.survivors.survivors_env import SurvivorsEnv

    def _make():
        e = SurvivorsEnv(host=args.host, port=args.port, frame_skip=args.frame_skip)
        e._reward_fn = reward_fn
        return e

    env = make_vec_env(_make, n_envs=1)

    if not args.no_vec_normalize:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        print("[INFO] VecNormalize を有効化")

    if args.frame_stack > 1:
        env = VecFrameStack(env, n_stack=args.frame_stack)
        print(f"[INFO] VecFrameStack を有効化 (n_stack={args.frame_stack})")

    return env


def _make_model(args, env):
    from stable_baselines3 import PPO

    algo_class = RecurrentPPO if args.recurrent else PPO
    default_policy = "MlpLstmPolicy" if args.recurrent else "MlpPolicy"

    # BC 用の最低限 PPO kwargs（アーキテクチャのみ重要）
    ppo_kwargs: dict = {"verbose": 0, "device": args.device}

    if args.entity_attention:
        from games.survivors.survivors_bc import _get_raw_env
        raw_env = _get_raw_env(env)
        offsets    = getattr(raw_env, "_offsets",    {})
        obs_schema = getattr(raw_env, "_obs_schema", [])
        from games.survivors.survivors_entity_attention_extractor import SurvivorsEntityAttentionExtractor
        policy_kwargs: dict = dict(
            features_extractor_class=SurvivorsEntityAttentionExtractor,
            features_extractor_kwargs=dict(features_dim=128, offsets=offsets, obs_schema=obs_schema),
            net_arch=[64, 64],
        )
        if args.recurrent:
            policy_kwargs["lstm_hidden_size"] = args.lstm_hidden_size
            policy_kwargs["n_lstm_layers"]    = args.n_lstm_layers
        ppo_kwargs["policy_kwargs"] = policy_kwargs
        print(f"[INFO] SurvivorsEntityAttentionExtractor を使用")
    elif args.recurrent:
        ppo_kwargs["policy_kwargs"] = {
            "lstm_hidden_size": args.lstm_hidden_size,
            "n_lstm_layers":    args.n_lstm_layers,
        }

    return algo_class(default_policy, env, **ppo_kwargs)


# ──────────────────────────────────────────────
# ファイル保存ヘルパー
# ──────────────────────────────────────────────

def _write_resolved_config(run_dir: Path, args) -> None:
    try:
        import yaml
        data = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
        (run_dir / "config_resolved.yaml").write_text(
            yaml.dump(data, allow_unicode=True, default_flow_style=False)
        )
    except Exception:
        pass


def _write_run_meta(run_dir: Path, args) -> None:
    meta: dict = {}
    try:
        meta["git_branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
        ).strip()
        meta["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        pass
    meta["run_name"] = args.run_name
    (run_dir / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))


def _write_bc_status(status_path: Path, args, stats: dict, has_vecnorm: bool) -> None:
    action_counts = stats.get("action_counts", [])
    total_acts = max(sum(action_counts), 1)
    action_dist = {f"action_{i}": c / total_acts for i, c in enumerate(action_counts)}

    git_branch, git_commit = "", ""
    try:
        git_branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        pass

    status = {
        "run_name":              args.run_name,
        "game":                  "survivors",
        "bc_only":               True,
        "bc_warmup_episodes":    args.episodes,
        "bc_epochs":             args.epochs,
        "bc_lr":                 args.lr,
        "bc_batch_size":         args.batch_size,
        "bc_transitions":        stats.get("transitions", 0),
        "bc_final_loss":         stats.get("final_loss", None),
        "bc_loss_history":       stats.get("loss_history", []),
        "episode_rewards_mean":  stats.get("episode_rewards_mean", None),
        "episode_length_mean":   stats.get("episode_length_mean", None),
        "action_distribution":   action_dist,
        "model_path":            "model_bc.zip",
        "vecnormalize_path":     "vecnormalize_bc.pkl" if has_vecnorm else None,
        "frame_skip":            args.frame_skip,
        "frame_stack":           args.frame_stack,
        "entity_attention":      args.entity_attention,
        "recurrent":             args.recurrent,
        "git_branch":            git_branch,
        "git_commit":            git_commit,
    }
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2))


def _find_vecnormalize(env):
    try:
        from stable_baselines3.common.vec_env import VecNormalize
    except ImportError:
        return None
    inner = env
    while inner is not None:
        if isinstance(inner, VecNormalize):
            return inner
        inner = getattr(inner, "venv", None)
    return None


def _load_reward_fn(path: Path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("reward_fn_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, "reward_fn", None)


if __name__ == "__main__":
    main()
