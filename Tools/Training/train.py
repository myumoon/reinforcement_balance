"""SB3 PPO による BalancePole 訓練スクリプト。

使い方:
  python train.py               # UE5 に接続して訓練
  python train.py --dry-run     # UE5 なしでパイプライン確認
  python train.py --resume models/balance_model.zip  # 既存モデルを継続訓練
  python train.py --help
"""

import argparse
import sys
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true", help="UE5 なしでスタブ環境を使用")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--total-steps", type=int, default=500_000)
    p.add_argument("--output", type=Path, default=Path("models/balance_model"))
    p.add_argument("--resume", type=Path, default=None,
                   help="再開する既存モデルのパス (.zip 拡張子は省略可)")
    p.add_argument("--checkpoint-freq", type=int, default=10_000,
                   help="チェックポイント保存間隔 (ステップ数, デフォルト: 10000)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.dry_run:
        from envs.balance_env_stub import DummyBalanceEnv
        env = make_vec_env(DummyBalanceEnv, n_envs=4)
        print("[dry-run] Using DummyBalanceEnv (no UE5 connection)")
    else:
        from envs.balance_env import BalanceEnv
        env = make_vec_env(
            lambda: BalanceEnv(host=args.host, port=args.port),
            n_envs=1,
        )
        print(f"[INFO] Connecting to UE5 server at {args.host}:{args.port} ...")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.resume:
        resume_path = str(args.resume)
        print(f"[INFO] Resuming from {resume_path}")
        model = PPO.load(resume_path, env=env)
    else:
        model = PPO("MlpPolicy", env, verbose=1)

    checkpoint_cb = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // (env.num_envs or 1), 1),
        save_path=str(args.output.parent),
        name_prefix=args.output.name,
        verbose=1,
    )

    try:
        model.learn(total_timesteps=args.total_steps, callback=checkpoint_cb,
                    reset_num_timesteps=args.resume is None)
    except KeyboardInterrupt:
        print("\n[INFO] 訓練を中断しました。モデルを保存します...")
    finally:
        model.save(str(args.output))
        print(f"[INFO] Model saved to {args.output}.zip")
        env.close()


if __name__ == "__main__":
    main()
