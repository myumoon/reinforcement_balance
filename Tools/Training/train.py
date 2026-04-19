"""SB3 PPO による BalancePole 訓練スクリプト。

使い方:
  python train.py               # UE5 に接続して訓練
  python train.py --dry-run     # UE5 なしでパイプライン確認
  python train.py --help
"""

import argparse
import sys
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true", help="UE5 なしでスタブ環境を使用")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--total-steps", type=int, default=500_000)
    p.add_argument("--output", type=Path, default=Path("models/balance_model"))
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

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=args.total_steps)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.output))
    print(f"[INFO] Model saved to {args.output}.zip")

    env.close()


if __name__ == "__main__":
    main()
