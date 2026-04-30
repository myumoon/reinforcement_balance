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
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

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


def _get_raw_env(env):
    """VecEnv ラッパーチェーンを辿って生の環境（CoinEnv）を返す。"""
    inner = env
    while hasattr(inner, "venv"):
        inner = inner.venv
    if hasattr(inner, "envs"):
        inner = inner.envs[0]
    # gymnasium.Wrapper (Monitor等) は _ 始まりの属性をフォワードしないため unwrapped で剥がす
    if hasattr(inner, "unwrapped"):
        inner = inner.unwrapped
    return inner


class _AnnealingShapingCallback(BaseCallback):
    """shaped_reward を線形アニーリングするコールバック。

    |shaped_reward_mean| / base_reward_mean の比率を check_freq ステップごとに計算し、
    比率が anneal_threshold を下回ったら anneal_steps かけて shaping_weight を 1.0→min_weight に減衰する。
    min_weight > 0 のとき shaping は永続的に維持される（推論時は不要）。
    """

    def __init__(self, raw_env, anneal_threshold: float, anneal_steps: int,
                 check_freq: int, min_steps: int, min_weight: float = 0.0):
        super().__init__(verbose=0)
        self._raw_env = raw_env
        self.anneal_threshold = anneal_threshold
        self.anneal_steps = anneal_steps
        self.check_freq = check_freq
        self.min_steps = min_steps
        self.min_weight = min_weight

        self._ep_base = 0.0
        self._ep_shaped = 0.0
        self._sum_base = 0.0
        self._sum_shaped = 0.0
        self._ep_count = 0
        self._last_check = 0
        self._anneal_start_step: int | None = None

    def _on_step(self) -> bool:
        info = self.locals["infos"][0]
        self._ep_base   += info.get("base_reward",   0.0)
        self._ep_shaped += info.get("shaped_reward", 0.0)

        if self.locals["dones"][0]:
            self._sum_base   += self._ep_base
            self._sum_shaped += abs(self._ep_shaped)
            self._ep_count   += 1
            self._ep_base = self._ep_shaped = 0.0

        # check_freq ステップごとに比率を計算してアニーリングをトリガー
        if (self.num_timesteps >= self.min_steps
                and self.num_timesteps - self._last_check >= self.check_freq
                and self._ep_count > 0):
            self._last_check = self.num_timesteps
            mean_base   = self._sum_base   / self._ep_count
            mean_shaped = self._sum_shaped / self._ep_count
            ratio = mean_shaped / max(mean_base, 1e-8)
            self._sum_base = self._sum_shaped = 0.0
            self._ep_count = 0
            print(f"[INFO] シェーピング比率: {ratio:.3f} "
                  f"(|shaped|={mean_shaped:.4f} / base={mean_base:.4f})")

            if ratio < self.anneal_threshold and self._anneal_start_step is None:
                self._anneal_start_step = self.num_timesteps
                print(f"[INFO] shaped_reward アニーリング開始 "
                      f"(ratio={ratio:.3f} < {self.anneal_threshold})")

        # アニーリング中: shaping_weight を 1.0 → min_weight に線形減衰
        if self._anneal_start_step is not None:
            progress = (self.num_timesteps - self._anneal_start_step) / self.anneal_steps
            weight = self.min_weight + (1.0 - self.min_weight) * max(0.0, 1.0 - progress)
            self._raw_env.shaping_weight = weight
            if weight <= self.min_weight and self.min_weight == 0.0 \
                    and self._raw_env._reward_fn is not None:
                self._raw_env._reward_fn = None
                print("[INFO] shaped_reward アニーリング完了 → reward_fn を無効化")

        return True


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
    p.add_argument("--dist-alpha", type=float, default=1.0,
                   help="距離バイアスの強さ (--entity-attention 専用, default: 1.0)")
    p.add_argument("--reward-fn", type=Path, default=None,
                   help="報酬シェーピング関数のパス (例: eureka_results/my_run/best/reward_fn.py, --game coin 専用)")
    p.add_argument("--no-vec-normalize", action="store_true",
                   help="VecNormalize による観測・報酬の正規化を無効化する")
    p.add_argument("--anneal-threshold", type=float, default=0.1,
                   help="|shaped|/base の比率がこれを下回ったらアニーリング開始 (default: 0.1, --reward-fn 専用)")
    p.add_argument("--anneal-steps", type=int, default=50_000,
                   help="アニーリングにかけるステップ数 (default: 50000)")
    p.add_argument("--anneal-check-freq", type=int, default=5_000,
                   help="比率のチェック間隔・ステップ数 (default: 5000)")
    p.add_argument("--anneal-min-steps", type=int, default=50_000,
                   help="アニーリングチェックを開始する最小ステップ数 (default: 50000)")
    p.add_argument("--anneal-min-weight", type=float, default=0.0,
                   help="shaping_weight の下限値 (default: 0.0=完全除去, 例: 0.05 で5%%維持)")
    p.add_argument("--ent-coef", type=float, default=0.01,
                   help="PPO エントロピー係数 (default: 0.01)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    ppo_kwargs = {**_PPO_KWARGS, "ent_coef": args.ent_coef}

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
                e = CoinEnv(host=args.host, port=port)
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

    # VecNormalize 適用
    # - 新規訓練: 正規化統計を初期化
    # - 再開: 既存の統計ファイルがあればロード。なければ VecNormalize を無効化（互換性維持）
    vecnorm_path = output.parent / (output.name + "_vecnorm.pkl")
    if not args.no_vec_normalize:
        if args.resume:
            resume_vecnorm = _strip_zip(args.resume).parent / (_strip_zip(args.resume).name + "_vecnorm.pkl")
            load_path = vecnorm_path if vecnorm_path.exists() else (resume_vecnorm if resume_vecnorm.exists() else None)
            if load_path is None:
                print("[WARN] VecNormalize 統計ファイルが見つかりません。VecNormalize を無効化します。"
                      f" (探索: {vecnorm_path}, {resume_vecnorm})")
            else:
                env = VecNormalize.load(str(load_path), env)
                env.training = True
                print(f"[INFO] VecNormalize 統計をロード: {load_path}")
        else:
            env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
            print("[INFO] VecNormalize を有効化しました (norm_obs=True, norm_reward=True)")

    if args.resume:
        resume_path = str(_strip_zip(args.resume))
        print(f"[INFO] {resume_path} から再開")
        if args.entity_attention:
            print("[INFO] --entity-attention は --resume 時は無視されます（保存済みモデルのアーキテクチャを使用）")
        model = PPO.load(resume_path, env=env)
    elif args.entity_attention:
        if args.game != "coin":
            print("[WARN] --entity-attention はコインゲーム専用です。MlpPolicy を使用します。")
            model = PPO("MlpPolicy", env, **ppo_kwargs)
        else:
            from entity_attention_extractor import EntityAttentionExtractor
            offsets = getattr(_get_raw_env(env), "_offsets", {})
            policy_kwargs = dict(
                features_extractor_class=EntityAttentionExtractor,
                features_extractor_kwargs=dict(
                    features_dim=128, offsets=offsets,
                    use_polar=True, dist_alpha=args.dist_alpha,
                ),
                net_arch=[64, 64],
            )
            print(f"[INFO] EntityAttentionExtractor を使用します (use_polar=True, dist_alpha={args.dist_alpha})")
            model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, **ppo_kwargs)
    else:
        model = PPO("MlpPolicy", env, **ppo_kwargs)

    checkpoint_cb = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // (env.num_envs or 1), 1),
        save_path=str(output.parent),
        name_prefix=output.name,
        verbose=1,
    )

    callbacks = [checkpoint_cb]
    if reward_fn is not None and args.game == "coin":
        anneal_cb = _AnnealingShapingCallback(
            raw_env=_get_raw_env(env),
            anneal_threshold=args.anneal_threshold,
            anneal_steps=args.anneal_steps,
            check_freq=args.anneal_check_freq,
            min_steps=args.anneal_min_steps,
            min_weight=args.anneal_min_weight,
        )
        callbacks.append(anneal_cb)
        print(f"[INFO] シェーピングアニーリング有効 "
              f"(threshold={args.anneal_threshold}, anneal_steps={args.anneal_steps:,}, "
              f"min_weight={args.anneal_min_weight})")

    try:
        model.learn(total_timesteps=args.total_steps, callback=callbacks,
                    reset_num_timesteps=args.resume is None)
    except KeyboardInterrupt:
        print("\n[INFO] 訓練を中断しました。モデルを保存します...")
    finally:
        model.save(str(output))
        print(f"[INFO] Model saved to {output}.zip")
        if isinstance(env, VecNormalize):
            env.save(str(vecnorm_path))
            print(f"[INFO] VecNormalize 統計を保存: {vecnorm_path}")
        env.close()


if __name__ == "__main__":
    main()
