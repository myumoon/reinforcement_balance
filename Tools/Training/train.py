"""SB3 PPO による BalancePole / CoinGame 訓練スクリプト。

使い方:
  python train.py                          # BalancePole (UE5 接続)
  python train.py --game coin              # CoinGame    (UE5 接続)
  python train.py --dry-run               # BalancePole スタブ
  python train.py --game coin --dry-run   # CoinGame    スタブ
  python train.py --resume models/balance_model
  python train.py --game coin --reward-fn eureka_results/my_run/best/reward_fn.py
  python train.py --help
"""

import argparse
import importlib.util
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env

_GAME_DEFAULTS = {
    "balance": {"port": 8765, "output": "models/balance_model"},
    "coin":    {"port": 8766, "output": "models/coin_model"},
}


def _linear_schedule(initial_value: float):
    """PPO 学習率の線形減衰スケジュール（訓練終了時に 0 になる）。"""
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


_PPO_KWARGS = dict(
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


def _load_reward_fn(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"--reward-fn ファイルが見つかりません: {path}")
    spec = importlib.util.spec_from_file_location("_reward_fn_mod", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.reward_shaping


def _strip_zip(path: Path) -> Path:
    """SB3 が .zip を自動付加するため、ユーザーが指定した .zip 拡張子を除去する。"""
    return path.with_suffix("") if path.suffix.lower() == ".zip" else path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--game", choices=["balance", "coin"], default="balance",
                   help="訓練対象のゲーム (default: balance)")
    p.add_argument("--dry-run", action="store_true", help="UE5 なしでスタブ環境を使用")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=None,
                   help="サーバーポート（未指定時はゲーム別デフォルト: balance=8765, coin=8766）")
    p.add_argument("--total-steps", type=int, default=500_000)
    p.add_argument("--output", type=Path, default=None,
                   help="保存先パス（未指定時はゲーム別デフォルト）")
    p.add_argument("--resume", type=Path, default=None,
                   help="再開する既存モデルのパス（.zip 拡張子は省略・付加どちらでも可）")
    p.add_argument("--checkpoint-freq", type=int, default=10_000,
                   help="チェックポイント保存間隔 (ステップ数, デフォルト: 10000)")
    p.add_argument("--entity-attention", action="store_true",
                   help="エンティティアテンション特徴抽出器を使用 (--game coin 専用, --resume 時は無視)")
    p.add_argument("--reward-fn", type=Path, default=None,
                   help="報酬シェーピング関数のパス (例: eureka_results/my_run/best/reward_fn.py, --game coin 専用)")
    p.add_argument("--reward-scale", type=float, default=0.2,
                   help="base_reward に乗じるスケール係数 (--game coin 専用, default: 0.2)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    defaults = _GAME_DEFAULTS[args.game]
    port   = args.port   if args.port   is not None else defaults["port"]
    output = _strip_zip(args.output if args.output is not None else Path(defaults["output"]))

    # --reward-fn の事前チェック
    reward_fn = None
    if args.reward_fn:
        if args.game != "coin":
            print("[WARN] --reward-fn はコインゲーム専用です。無視します。")
        elif args.dry_run:
            print("[WARN] --reward-fn は --dry-run 時は無視されます。")
        else:
            reward_fn = _load_reward_fn(args.reward_fn)
            print(f"[INFO] 報酬シェーピング関数をロード: {args.reward_fn}")

    if args.dry_run:
        if args.game == "coin":
            from envs.coin_env_stub import DummyCoinEnv
            env = make_vec_env(DummyCoinEnv, n_envs=4)
        else:
            from envs.balance_env_stub import DummyBalanceEnv
            env = make_vec_env(DummyBalanceEnv, n_envs=4)
        print(f"[dry-run] game={args.game} スタブ環境で実行")
    else:
        if args.game == "coin":
            from envs.coin_env import CoinEnv
            def _make_coin_env():
                e = CoinEnv(host=args.host, port=port, reward_scale=args.reward_scale)
                e._reward_fn = reward_fn
                return e
            env = make_vec_env(_make_coin_env, n_envs=1)
        else:
            from envs.balance_env import BalanceEnv
            env = make_vec_env(
                lambda: BalanceEnv(host=args.host, port=port),
                n_envs=1,
            )
        print(f"[INFO] game={args.game}  UE5 サーバー {args.host}:{port} に接続...")

    output.parent.mkdir(parents=True, exist_ok=True)

    if args.resume:
        resume_path = str(_strip_zip(args.resume))
        print(f"[INFO] {resume_path} から再開")
        if args.entity_attention:
            print("[INFO] --entity-attention は --resume 時は無視されます（保存済みモデルのアーキテクチャを使用）")
        model = PPO.load(resume_path, env=env)
    elif args.entity_attention:
        if args.game != "coin":
            print("[WARN] --entity-attention はコインゲーム専用です。MlpPolicy を使用します。")
            model = PPO("MlpPolicy", env, **_PPO_KWARGS)
        else:
            from entity_attention_extractor import EntityAttentionExtractor
            policy_kwargs = dict(
                features_extractor_class=EntityAttentionExtractor,
                features_extractor_kwargs=dict(features_dim=128),
                net_arch=[64, 64],
            )
            print("[INFO] EntityAttentionExtractor を使用します")
            model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, **_PPO_KWARGS)
    else:
        model = PPO("MlpPolicy", env, **_PPO_KWARGS)

    checkpoint_cb = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // (env.num_envs or 1), 1),
        save_path=str(output.parent),
        name_prefix=output.name,
        verbose=1,
    )

    try:
        model.learn(total_timesteps=args.total_steps, callback=checkpoint_cb,
                    reset_num_timesteps=args.resume is None)
    except KeyboardInterrupt:
        print("\n[INFO] 訓練を中断しました。モデルを保存します...")
    finally:
        model.save(str(output))
        print(f"[INFO] Model saved to {output}.zip")
        env.close()


if __name__ == "__main__":
    main()
