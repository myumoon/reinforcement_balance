"""訓練済みモデルのエンティティアテンション重みを可視化するスクリプト。

使い方:
  python inspect_attention.py --game coin --model results/run_v7/best/model \
    --vec-normalize results/run_v7/best/vec_normalize.pkl \
    --steps 2000

  python inspect_attention.py --game survivors --model results/survivors_run/best/model \
    --vec-normalize results/survivors_run/best/vecnormalize.pkl \
    --steps 2000 --phase 3

出力:
  attention_weights.png  -- slot 別アテンション重み分布（棒グラフ）
  コンソールに各色の slot[0]/均等 の倍率を出力

判読方法:
  CoinGame:
    アイテム slot[0] の重み >> 1/num_items    → 最近傍アイテムを追跡している ✓
    アイテム重みが均等に近い                  → 距離に基づく優先度を学習できていない ✗
  Survivors:
    items は RED(0-9) / GREEN(10-21) / BLUE(22-33) の順に連結されている。
    各色の slot[0] が >> 均等値なら最近傍 Gem を追跡している ✓
    slot[0] ≈ 0 なら常にパディング（その色の Gem がほぼ未出現）
  Enemy:
    slot[0] の重み >> 1/num_enemies   → 最近傍敵を追跡している ✓
"""

import argparse
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # GUI なし環境対応
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

_JP_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\NotoSansJP-Regular.otf",
    r"C:\Windows\Fonts\meiryo.ttc",
    r"C:\Windows\Fonts\YuGothR.ttc",
    r"C:\Windows\Fonts\msgothic.ttc",
    "/mnt/c/Windows/Fonts/NotoSansJP-Regular.otf",
    "/mnt/c/Windows/Fonts/meiryo.ttc",
    "/mnt/c/Windows/Fonts/YuGothR.ttc",
    "/mnt/c/Windows/Fonts/msgothic.ttc",
]

def _setup_japanese_font():
    for path in _JP_FONT_CANDIDATES:
        if os.path.exists(path):
            fm.fontManager.addfont(path)
            prop = fm.FontProperties(fname=path)
            name = prop.get_name()
            matplotlib.rcParams["font.sans-serif"] = [name] + matplotlib.rcParams.get("font.sans-serif", [])
            matplotlib.rcParams["font.family"] = "sans-serif"
            matplotlib.rcParams["axes.unicode_minus"] = False
            return
    print("[WARN] 日本語フォントが見つかりませんでした。文字化けの可能性があります。")

_setup_japanese_font()


_GAME_DEFAULTS = {
    "coin":      {"port": 8766, "item_label": "コイン"},
    "survivors": {"port": 8767, "item_label": "Gem"},
}

# Survivors Gem の色定義（_GEM_REL_POS_KEYS_NEW の連結順）
_GEM_COLORS = [
    ("RED",   "#FF6B6B", "#C0392B"),
    ("GREEN", "#51CF66", "#27AE60"),
    ("BLUE",  "#74C0FC", "#2980B9"),
]


def _parse_args():
    p = argparse.ArgumentParser(description="エンティティアテンション可視化")
    p.add_argument("--game",          choices=["coin", "survivors"], default="coin",
                   help="対象ゲーム (default: coin)")
    p.add_argument("--model",         required=True, help="モデルパス (model.zip から .zip は省略可)")
    p.add_argument("--vec-normalize", default=None,  help="VecNormalize 統計ファイル (.pkl)")
    p.add_argument("--steps",         type=int, default=2000, help="収集ステップ数 (default: 2000)")
    p.add_argument("--host",          default="127.0.0.1")
    p.add_argument("--port",          type=int, default=None,
                   help="サーバーポート（未指定時はゲーム別デフォルト: coin=8766, survivors=8767）")
    p.add_argument("--output",        default="attention_weights.png")
    p.add_argument("--phase",         type=int, default=None,
                   help="[Survivors] カリキュラムフェーズ番号を強制指定 (0始まり)。"
                        "全色の Gem を出現させるには 3 以上を推奨。")
    return p.parse_args()


def _get_gem_color_splits(extractor):
    """Survivors 新スキーマ時の Gem 色ごとスロット数を返す。

    Returns:
        list of (color_name, bar_color, edge_color, n_slots) or None
    """
    if not hasattr(extractor, "_gem_slices") or not getattr(extractor, "_is_new_schema", False):
        return None
    result = []
    for (start, end), (cname, cbar, cedge) in zip(extractor._gem_slices, _GEM_COLORS):
        n = (end - start) // 2
        result.append((cname, cbar, cedge, n))
    return result if result else None


def _plot_survivors(fig, item_avg, gem_splits, enemy_avg):
    """Survivors 用: Gem を色別 + 敵 の 4 subplots。"""
    axes = fig.subplots(1, 4)

    offset = 0
    for ax, (cname, cbar, cedge, n) in zip(axes[:3], gem_splits):
        weights = item_avg[offset:offset + n]
        uniform = 1.0 / n
        slot0   = weights[0]
        ratio   = slot0 / uniform
        if slot0 < uniform * 0.1:
            state = "MASKED"
        elif ratio >= 2.0:
            state = "✓ 追跡中"
        else:
            state = "✗ 均等に近い"

        ax.bar(range(n), weights, color=cbar, edgecolor=cedge)
        ax.axhline(uniform, color="gray", linestyle="--", linewidth=1,
                   label=f"均等 ({uniform:.4f})")
        ax.set_xlabel(f"{cname} slot (0=最近傍)")
        ax.set_ylabel("平均Attention重み")
        ax.set_title(f"{cname} Gem\nslot[0]={slot0:.4f}  {ratio:.1f}x  {state}")
        ax.legend(fontsize=8)
        offset += n

    ax_e = axes[3]
    enemy_uniform = 1.0 / len(enemy_avg)
    ax_e.bar(range(len(enemy_avg)), enemy_avg, color="salmon", edgecolor="firebrick")
    ax_e.axhline(enemy_uniform, color="gray", linestyle="--", linewidth=1,
                 label=f"均等 ({enemy_uniform:.4f})")
    ax_e.set_xlabel("Enemy slot (0=最近傍)")
    ax_e.set_ylabel("平均Attention重み")
    slot0e = enemy_avg[0]
    ratioe = slot0e / enemy_uniform
    statee = "✓ 追跡中" if ratioe >= 2.0 else "✗ 均等に近い"
    ax_e.set_title(f"Enemy\nslot[0]={slot0e:.4f}  {ratioe:.1f}x  {statee}")
    ax_e.legend(fontsize=8)


def _plot_coin(fig, item_avg, item_label, enemy_avg):
    """CoinGame 用: アイテム + 敵 の 2 subplots（従来動作）。"""
    axes = fig.subplots(1, 2)
    item_uniform  = 1.0 / len(item_avg)
    enemy_uniform = 1.0 / len(enemy_avg)

    axes[0].bar(range(len(item_avg)), item_avg, color="gold", edgecolor="goldenrod")
    axes[0].axhline(item_uniform, color="gray", linestyle="--", linewidth=1,
                    label=f"均等分布 ({item_uniform:.4f})")
    axes[0].set_xlabel(f"{item_label} slot  (0=最近傍、距離昇順)")
    axes[0].set_ylabel("平均アテンション重み")
    axes[0].set_title(
        f"{item_label}アテンション\nslot[0]={item_avg[0]:.4f}  "
        f"(均等比 {item_avg[0]/item_uniform:.1f}x)"
    )
    axes[0].legend(fontsize=9)
    axes[0].set_xlim(-0.5, min(19, len(item_avg) - 0.5))

    axes[1].bar(range(len(enemy_avg)), enemy_avg, color="salmon", edgecolor="firebrick")
    axes[1].axhline(enemy_uniform, color="gray", linestyle="--", linewidth=1,
                    label=f"均等分布 ({enemy_uniform:.4f})")
    axes[1].set_xlabel("enemy slot  (0=最近傍、距離昇順)")
    axes[1].set_ylabel("平均アテンション重み")
    axes[1].set_title(
        f"敵アテンション\nslot[0]={enemy_avg[0]:.4f}  "
        f"(均等比 {enemy_avg[0]/enemy_uniform:.1f}x)"
    )
    axes[1].legend(fontsize=9)


def _print_summary_survivors(item_avg, gem_splits, enemy_avg):
    print("\n--- Gem Attention 要約（色別） ---")
    offset = 0
    for cname, _, _, n in gem_splits:
        weights = item_avg[offset:offset + n]
        uniform = 1.0 / n
        slot0   = weights[0]
        ratio   = slot0 / uniform
        top3    = np.argsort(weights)[::-1][:3]
        if slot0 < uniform * 0.1:
            state = "MASKED（常にパディング）"
        elif ratio >= 2.0:
            state = "✓ 最近傍を追跡"
        else:
            state = "✗ 距離無視の可能性"
        print(f"  {cname:>6}: slot[0]={slot0:.4f}  均等={uniform:.4f}  倍率={ratio:.1f}x  {state}")
        print(f"          上位3: {[(int(i), round(float(weights[i]), 4)) for i in top3]}")
        offset += n

    enemy_uniform = 1.0 / len(enemy_avg)
    slot0e = enemy_avg[0]
    ratioe = slot0e / enemy_uniform
    top3e  = np.argsort(enemy_avg)[::-1][:3]
    statee = "✓ 最近傍を追跡" if ratioe >= 2.0 else "✗ 距離無視の可能性"
    print(f"\n--- Enemy Attention 要約 ---")
    print(f"  slot[0]={slot0e:.4f}  均等={enemy_uniform:.4f}  倍率={ratioe:.1f}x  {statee}")
    print(f"  上位3: {[(int(i), round(float(enemy_avg[i]), 4)) for i in top3e]}")


def main():
    args = _parse_args()

    defaults = _GAME_DEFAULTS[args.game]
    port = args.port if args.port is not None else defaults["port"]
    item_label = defaults["item_label"]

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3.common.monitor import Monitor

    if args.game == "survivors":
        from games.survivors.survivors_env import SurvivorsEnv
        raw_env = SurvivorsEnv(host=args.host, port=port)
    else:
        from games.coin.coin_env import CoinEnv
        raw_env = CoinEnv(host=args.host, port=port)

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

    # UE5 接続 & 初期 reset
    obs = env.reset()

    # フェーズ強制指定（Survivors のみ）
    if args.phase is not None and args.game == "survivors":
        from games.survivors.survivors_curriculum import PHASES, _phase_to_params
        if args.phase < 0 or args.phase >= len(PHASES):
            print(f"[ERROR] --phase {args.phase} は範囲外です (0-{len(PHASES)-1})")
            sys.exit(1)
        phase_params = _phase_to_params(args.phase)
        raw_env.set_params(**phase_params)
        phase_name = PHASES[args.phase].name
        print(f"[INFO] Phase {args.phase} ({phase_name}) を適用")
        obs = env.reset()

    item_weights_list  = []
    enemy_weights_list = []

    device = next(extractor.parameters()).device

    print(f"[INFO] {args.steps} ステップ収集中...")
    for step in range(args.steps):
        obs_t = torch.FloatTensor(obs).to(device)
        with torch.no_grad():
            iw, ew = extractor.get_attention_weights(obs_t)
        item_weights_list.append(iw.squeeze(0).cpu().numpy())
        enemy_weights_list.append(ew.squeeze(0).cpu().numpy())

        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, _ = env.step(action)

        if (step + 1) % 500 == 0:
            print(f"  {step + 1}/{args.steps}")

    env.close()

    item_avg  = np.mean(item_weights_list,  axis=0)
    enemy_avg = np.mean(enemy_weights_list, axis=0)

    # ---- プロット ----
    gem_splits = _get_gem_color_splits(extractor) if args.game == "survivors" else None

    phase_label = ""
    if args.phase is not None and args.game == "survivors":
        from games.survivors.survivors_curriculum import PHASES
        phase_label = f"  [Phase {args.phase}: {PHASES[args.phase].name}]"

    if gem_splits:
        fig = plt.figure(figsize=(20, 5))
        _plot_survivors(fig, item_avg, gem_splits, enemy_avg)
        plt.suptitle(
            f"Gem（色別）/ Enemy  Attention 重み分布  ({args.steps} steps 平均){phase_label}\n"
            f"slot[0]/均等 >> 1.0 なら最近傍を追跡、≈ 1.0 なら距離無視、≈ 0 なら常にパディング",
            fontsize=11,
        )
    else:
        fig = plt.figure(figsize=(14, 5))
        _plot_coin(fig, item_avg, item_label, enemy_avg)
        plt.suptitle(
            f"エンティティアテンション重み分布  ({args.steps} steps 平均)  [{args.game}]{phase_label}\n"
            f"slot[0]/均等 >> 1.0 なら最近傍を追跡、≈ 1.0 なら距離無視",
            fontsize=12,
        )

    plt.tight_layout()
    plt.savefig(args.output, dpi=120)
    print(f"\n[OK] 画像保存: {args.output}")

    # ---- コンソール要約 ----
    if gem_splits:
        _print_summary_survivors(item_avg, gem_splits, enemy_avg)
    else:
        item_uniform  = 1.0 / len(item_avg)
        enemy_uniform = 1.0 / len(enemy_avg)
        print("\n--- アテンション要約 ---")
        print(f"{item_label}: slot[0]={item_avg[0]:.4f}  均等={item_uniform:.4f}  "
              f"倍率={item_avg[0]/item_uniform:.1f}x  "
              f"{'✓ 最近傍を追跡' if item_avg[0] > item_uniform * 2 else '✗ 距離無視の可能性'}")
        print(f"敵      : slot[0]={enemy_avg[0]:.4f}  均等={enemy_uniform:.4f}  "
              f"倍率={enemy_avg[0]/enemy_uniform:.1f}x  "
              f"{'✓ 最近傍を追跡' if enemy_avg[0] > enemy_uniform * 2 else '✗ 距離無視の可能性'}")
        print("\n--- 上位 3 slot 重み ---")
        top_item  = np.argsort(item_avg)[::-1][:3]
        top_enemy = np.argsort(enemy_avg)[::-1][:3]
        print(f"{item_label}: {[(int(i), round(float(item_avg[i]), 4)) for i in top_item]}")
        print(f"敵      : {[(int(i), round(float(enemy_avg[i]), 4)) for i in top_enemy]}")


if __name__ == "__main__":
    main()
