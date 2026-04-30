"""訓練済みモデルのエンティティアテンション重みを可視化するスクリプト。

使い方:
  python inspect_attention.py --model results/run_v7/best/model \
    --vec-normalize results/run_v7/best/vec_normalize.pkl \
    --steps 2000

出力:
  attention_weights.png  -- slot 別アテンション重み分布（棒グラフ）
  コンソールに slot[0]/均等 の倍率を出力

判読方法:
  コイン slot[0] の重み >> 1/num_coins  → 最近傍コインを追跡している ✓
  コイン重みが均等に近い              → 距離に基づく優先度を学習できていない ✗
  敵    slot[0] の重み >> 1/num_enemies → 最近傍敵を追跡している ✓
"""

import argparse
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # GUI なし環境対応


def _parse_args():
    p = argparse.ArgumentParser(description="エンティティアテンション可視化")
    p.add_argument("--model",         required=True, help="モデルパス (model.zip から .zip は省略可)")
    p.add_argument("--vec-normalize", default=None,  help="VecNormalize 統計ファイル (.pkl)")
    p.add_argument("--steps",         type=int, default=2000, help="収集ステップ数 (default: 2000)")
    p.add_argument("--host",          default="127.0.0.1")
    p.add_argument("--port",          type=int, default=8766)
    p.add_argument("--output",        default="attention_weights.png")
    return p.parse_args()


def main():
    args = _parse_args()

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3.common.monitor import Monitor
    from envs.coin_env import CoinEnv

    raw_env = CoinEnv(host=args.host, port=args.port)
    vec_env = DummyVecEnv([lambda: Monitor(raw_env)])

    if args.vec_normalize:
        env = VecNormalize.load(args.vec_normalize, vec_env)
        env.training = False
        env.norm_reward = False
        print(f"[INFO] VecNormalize 読み込み: {args.vec_normalize}")
    else:
        env = vec_env
        print("[INFO] VecNormalize なし")

    model = PPO.load(args.model, env=env)
    extractor = model.policy.features_extractor
    print(f"[INFO] モデル読み込み: {args.model}")

    if not hasattr(extractor, "get_attention_weights"):
        print("[ERROR] このモデルの features_extractor は get_attention_weights() を持っていません。")
        print("        EntityAttentionExtractor で訓練されたモデルを使用してください。")
        sys.exit(1)

    coin_weights_list  = []
    enemy_weights_list = []

    obs = env.reset()
    print(f"[INFO] {args.steps} ステップ収集中...")
    for step in range(args.steps):
        obs_t = torch.FloatTensor(obs)
        with torch.no_grad():
            cw, ew = extractor.get_attention_weights(obs_t)
        coin_weights_list.append(cw.squeeze(0).cpu().numpy())
        enemy_weights_list.append(ew.squeeze(0).cpu().numpy())

        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, _ = env.step(action)

        if (step + 1) % 500 == 0:
            print(f"  {step + 1}/{args.steps}")

    env.close()

    coin_avg  = np.mean(coin_weights_list,  axis=0)  # [num_coins]
    enemy_avg = np.mean(enemy_weights_list, axis=0)  # [num_enemies]
    coin_uniform  = 1.0 / len(coin_avg)
    enemy_uniform = 1.0 / len(enemy_avg)

    # ---- プロット ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(range(len(coin_avg)), coin_avg, color="gold", edgecolor="goldenrod")
    axes[0].axhline(coin_uniform, color="gray", linestyle="--", linewidth=1, label=f"均等分布 ({coin_uniform:.4f})")
    axes[0].set_xlabel("coin slot  (0=最近傍、距離昇順)")
    axes[0].set_ylabel("平均アテンション重み")
    axes[0].set_title(
        f"コインアテンション\nslot[0]={coin_avg[0]:.4f}  "
        f"(均等比 {coin_avg[0]/coin_uniform:.1f}x)"
    )
    axes[0].legend(fontsize=9)
    axes[0].set_xlim(-0.5, min(19, len(coin_avg) - 0.5))  # 先頭20スロットのみ表示

    axes[1].bar(range(len(enemy_avg)), enemy_avg, color="salmon", edgecolor="firebrick")
    axes[1].axhline(enemy_uniform, color="gray", linestyle="--", linewidth=1, label=f"均等分布 ({enemy_uniform:.4f})")
    axes[1].set_xlabel("enemy slot  (0=最近傍、距離昇順)")
    axes[1].set_ylabel("平均アテンション重み")
    axes[1].set_title(
        f"敵アテンション\nslot[0]={enemy_avg[0]:.4f}  "
        f"(均等比 {enemy_avg[0]/enemy_uniform:.1f}x)"
    )
    axes[1].legend(fontsize=9)

    plt.suptitle(
        f"エンティティアテンション重み分布  ({args.steps} steps 平均)\n"
        f"slot[0]/均等 >> 1.0 なら最近傍を追跡、≈ 1.0 なら距離無視",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(args.output, dpi=120)
    print(f"\n[OK] 画像保存: {args.output}")

    # ---- コンソール要約 ----
    print("\n--- アテンション要約 ---")
    print(f"コイン  : slot[0]={coin_avg[0]:.4f}  均等={coin_uniform:.4f}  "
          f"倍率={coin_avg[0]/coin_uniform:.1f}x  "
          f"{'✓ 最近傍を追跡' if coin_avg[0] > coin_uniform * 2 else '✗ 距離無視の可能性'}")
    print(f"敵      : slot[0]={enemy_avg[0]:.4f}  均等={enemy_uniform:.4f}  "
          f"倍率={enemy_avg[0]/enemy_uniform:.1f}x  "
          f"{'✓ 最近傍を追跡' if enemy_avg[0] > enemy_uniform * 2 else '✗ 距離無視の可能性'}")

    print("\n--- 上位 3 slot 重み ---")
    top_coin  = np.argsort(coin_avg)[::-1][:3]
    top_enemy = np.argsort(enemy_avg)[::-1][:3]
    print(f"コイン  : {[(int(i), round(float(coin_avg[i]), 4)) for i in top_coin]}")
    print(f"敵      : {[(int(i), round(float(enemy_avg[i]), 4)) for i in top_enemy]}")


if __name__ == "__main__":
    main()
