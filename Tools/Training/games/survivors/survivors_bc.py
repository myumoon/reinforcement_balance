"""Survivors ルールベース行動クローニング（PPO 事前初期化）.

使い方（train.py から自動呼び出し）:
    python train.py --game survivors --bc-warmup-episodes 100 --bc-epochs 30

デモ収集にはUE5接続が必要。dry-run では使用不可。
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

# 16方向ビン → 9方向アクション変換テーブル（C++ SurvivorsObservationComponent から導出）
#
# C++ 方向ビン割当 (SurvivorsObservationComponent.cpp):
#   Rel      = TargetPos - PlayerPos     (UE5 2D座標: +X=東, +Y=北)
#   AngleRad = atan2(Rel.Y, Rel.X)       (-π .. +π)
#   Angle01  = (AngleRad + π) / (2π)     (0.0 .. 1.0)
#   Dir      = clamp(floor(Angle01 * 16), 0, 15)
#
# 9方向アクション (SurvivorsPlayerComponent.cpp):
#   0=北(0,+1), 1=北東(+1,+1), 2=東(+1,0), 3=南東(+1,-1)
#   4=南(0,-1), 5=南西(-1,-1), 6=西(-1,0), 7=北西(-1,+1)
#
# 各 dir16 ビン中心角と最近傍アクション:
#   dir16=0  (≈-168.75°) → W  (action 6)
#   dir16=1  (≈-146.25°) → SW (action 5)
#   dir16=2  (≈-123.75°) → SW (action 5)
#   dir16=3  (≈-101.25°) → S  (action 4)
#   dir16=4  (≈ -78.75°) → S  (action 4)
#   dir16=5  (≈ -56.25°) → SE (action 3)
#   dir16=6  (≈ -33.75°) → SE (action 3)
#   dir16=7  (≈ -11.25°) → E  (action 2)
#   dir16=8  (≈  11.25°) → E  (action 2)
#   dir16=9  (≈  33.75°) → NE (action 1)
#   dir16=10 (≈  56.25°) → NE (action 1)
#   dir16=11 (≈  78.75°) → N  (action 0)
#   dir16=12 (≈ 101.25°) → N  (action 0)
#   dir16=13 (≈ 123.75°) → NW (action 7)
#   dir16=14 (≈ 146.25°) → NW (action 7)
#   dir16=15 (≈ 168.75°) → W  (action 6)
_DIR16_TO_ACTION9: list[int] = [6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0, 7, 7, 6]


def rule_policy(obs: np.ndarray, offsets: dict) -> int:
    """ルールベース方策。生の（正規化前）obs を想定するが VecNormalize 後でも動作する。

    argmax ベースの比較は VecNormalize の線形変換に対して不変。
    相対閾値（range の割合）を使うことで正規化の有無に依存しない。

    優先度:
      1. 最大敵近距離密度方向から逃げる（差が閾値以上のとき）
      2. Gem × 安全スコアが最大の方向へ移動
      3. 敵最近傍距離が最大の安全方向へ移動

    Returns:
        int: 0–7 の行動（静止=8 は使用しない）
    """
    DIR_COUNT = 16
    o_en = offsets["enemy_density_near_16dir"]
    o_gn = offsets["gem_density_near_16dir"]
    o_nd = offsets["enemy_nearest_dist_16dir"]

    enemy_near = obs[o_en:o_en + DIR_COUNT].astype(np.float64)
    gem_near   = obs[o_gn:o_gn + DIR_COUNT].astype(np.float64)
    enemy_nd   = obs[o_nd:o_nd + DIR_COUNT].astype(np.float64)

    en_min, en_max = float(np.min(enemy_near)), float(np.max(enemy_near))
    en_range = en_max - en_min

    # 最も危険な方向とその逆方向（逃避先）
    danger_dir = int(np.argmax(enemy_near))
    escape_dir = (danger_dir + DIR_COUNT // 2) % DIR_COUNT

    # range の 30% 以上差があれば逃げる（raw obs では 0.0〜1.0、normalized でも相対判断）
    if en_range > 1e-6 and (en_max - float(enemy_near[escape_dir])) / en_range < 0.7:
        return _DIR16_TO_ACTION9[escape_dir]

    # 安全スコア（敵密度が低い方向ほど高い）× Gem 密度でスコアリング
    safety = 1.0 - (enemy_near - en_min) / max(en_range, 1e-6)
    gem_score = gem_near * safety
    gn_min, gn_max = float(np.min(gem_score)), float(np.max(gem_score))
    gem_dir = int(np.argmax(gem_score))

    if gn_max - gn_min > 1e-6:
        return _DIR16_TO_ACTION9[gem_dir]

    # どちらでもない：最も安全な方向（敵最近傍距離が最大）へ
    safest = int(np.argmax(enemy_nd))
    return _DIR16_TO_ACTION9[safest]


def bc_warmup(
    model,
    env,
    n_episodes: int = 100,
    epochs: int = 30,
    lr: float = 3e-4,
    batch_size: int = 512,
    verbose: int = 1,
) -> None:
    """ルールベース方策でデモ収集し、model を行動クローニングで事前初期化する。

    VecNormalize が有効な場合、ルール方策は生の obs で判断し、BC 訓練は正規化済み obs で
    行うことで policy が実際の入力分布と整合する。

    RecurrentPPO (MlpLstmPolicy) の場合は LSTM を零状態でバイパスして近似的に初期化する。
    LSTM の時系列挙動は PPO 訓練で学習される。

    Args:
        model:       SB3 PPO または RecurrentPPO（learn() 前に呼ぶこと）
        env:         訓練用 VecEnv（VecNormalize / VecFrameStack ラッパーを含んでよい）
        n_episodes:  デモ収集エピソード数
        epochs:      BC エポック数（少なめにして局所最適化を防ぐ）
        lr:          Adam 学習率
        batch_size:  ミニバッチサイズ
        verbose:     0: サイレント, 1: 進捗表示
    """
    raw_env = _get_raw_env(env)
    offsets: dict | None = getattr(raw_env, "_offsets", None)
    if offsets is None:
        if verbose:
            print("[BC] _offsets が取得できないため BC をスキップします")
        return
    # 必要な obs セグメントが存在するか確認
    required = {"enemy_density_near_16dir", "gem_density_near_16dir", "enemy_nearest_dist_16dir"}
    missing = required - set(offsets.keys())
    if missing:
        if verbose:
            print(f"[BC] 必要な obs セグメントが存在しないため BC をスキップします: {missing}")
        return

    vec_norm = _find_vecnormalize(env)
    if verbose:
        n_stack = getattr(env, "n_stack", 1)
        print(
            f"[BC] デモ収集開始 "
            f"(n_episodes={n_episodes}, VecNormalize={'有効' if vec_norm else '無効'}, "
            f"frame_stack={n_stack})"
        )

    obs_list: list[np.ndarray] = []
    act_list: list[int] = []
    episode_rewards: list[float] = []
    ep_count = 0
    ep_reward = 0.0

    obs = env.reset()

    while ep_count < n_episodes:
        # ルール方策は raw obs（VecNormalize 前）で判断
        if vec_norm is not None:
            raw_obs = vec_norm.get_original_obs()[0]
        else:
            raw_obs = obs[0]

        # VecFrameStack の場合 raw_obs は最新フレームのみ（single-frame）
        # obs はスタック済みなので shape が異なる場合は最後の単一フレームを使用
        if raw_obs.shape != obs[0].shape:
            single_frame_dim = raw_obs.shape[0]
            raw_obs_check = obs[0][-single_frame_dim:]
            del raw_obs_check  # 参照のみ確認

        action = rule_policy(raw_obs, offsets)

        # BC 訓練用: model が実際に受け取る obs（正規化・スタック済み）を記録
        obs_list.append(obs[0].copy())
        act_list.append(action)

        obs, reward, done, _info = env.step(np.array([action]))
        ep_reward += float(reward[0])

        if done[0]:
            episode_rewards.append(ep_reward)
            ep_reward = 0.0
            ep_count += 1
            if verbose and ep_count % 20 == 0:
                recent = episode_rewards[-20:]
                print(
                    f"[BC]   収集進捗: {ep_count}/{n_episodes} エピソード  "
                    f"mean_reward={float(np.mean(recent)):.1f}"
                )

    total_transitions = len(obs_list)
    if verbose:
        print(
            f"[BC] デモ収集完了: {total_transitions} 遷移, "
            f"平均エピソード報酬={float(np.mean(episode_rewards)):.1f}"
        )
        print(f"[BC] BC 訓練開始 (epochs={epochs}, batch_size={batch_size}, lr={lr})...")

    _bc_train(
        model,
        np.array(obs_list, dtype=np.float32),
        np.array(act_list, dtype=np.int64),
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        verbose=verbose,
    )

    # 収集後 obs を model に設定して次の rollout 収集を clean state から開始
    model._last_obs = obs
    model._last_episode_starts = done.copy()

    if verbose:
        print("[BC] BC 初期化完了")


def _bc_train(
    model,
    obs_arr: np.ndarray,
    act_arr: np.ndarray,
    epochs: int,
    lr: float,
    batch_size: int,
    verbose: int,
) -> None:
    """BC 訓練ループ（内部ヘルパー）."""
    is_recurrent = hasattr(model.policy, "lstm_actor")
    device = model.device
    n = len(obs_arr)

    obs_t = torch.as_tensor(obs_arr, dtype=torch.float32).to(device)
    act_t = torch.as_tensor(act_arr, dtype=torch.long).to(device)

    opt = Adam(model.policy.parameters(), lr=lr)
    model.policy.set_training_mode(True)

    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        ep_losses: list[float] = []

        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            ob = obs_t[idx]
            ac = act_t[idx]

            if not is_recurrent:
                # SB3 PPO (MlpPolicy): evaluate_actions が log_prob を返す
                _, log_probs, _ = model.policy.evaluate_actions(ob, ac)
                loss = -log_probs.mean()
            else:
                # RecurrentPPO: LSTM を零初期状態でバイパス（各サンプルを独立エピソードとして扱う）
                lstm = model.policy.lstm_actor
                b = ob.shape[0]
                h0 = torch.zeros(lstm.num_layers, b, lstm.hidden_size, device=device)
                c0 = torch.zeros_like(h0)
                feats = model.policy.extract_features(ob)
                lstm_out, _ = lstm(feats.unsqueeze(0), (h0, c0))
                latent_pi, _ = model.policy.mlp_extractor(lstm_out.squeeze(0))
                logits = model.policy.action_net(latent_pi)
                loss = F.cross_entropy(logits, ac)

            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_losses.append(float(loss.item()))

        if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
            print(f"[BC]   epoch {epoch + 1:3d}/{epochs}  loss={float(np.mean(ep_losses)):.4f}")

    model.policy.set_training_mode(False)


def _get_raw_env(env):
    """VecEnv ラッパーチェーンを辿って生の環境を返す."""
    inner = env
    while hasattr(inner, "venv"):
        inner = inner.venv
    if hasattr(inner, "envs"):
        inner = inner.envs[0]
    if hasattr(inner, "unwrapped"):
        inner = inner.unwrapped
    return inner


def _find_vecnormalize(env):
    """VecEnv ラッパーチェーンから VecNormalize を見つけて返す（無ければ None）."""
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


if __name__ == "__main__":
    # 方向マッピングの検証（pytest がない環境での動作確認用）
    import math

    def _ue5_atan2_to_dir16(angle_deg: float) -> int:
        angle_rad = math.radians(angle_deg)
        angle01 = (angle_rad + math.pi) / (2.0 * math.pi)
        return max(0, min(15, int(angle01 * 16)))

    # 9方向アクションの角度（atan2 convention, +X=East, +Y=North）
    action_angles = {0: 90.0, 1: 45.0, 2: 0.0, 3: -45.0, 4: -90.0, 5: -135.0, 6: 180.0, 7: 135.0}

    print("=== dir16 → action9 テーブル検証 ===")
    errors = []
    for dir16 in range(16):
        # ビン中心角（degrees）
        angle01_center = (dir16 + 0.5) / 16.0
        angle_rad_center = angle01_center * 2.0 * math.pi - math.pi
        angle_deg_center = math.degrees(angle_rad_center)

        # 最近傍アクションを計算
        best_action = min(action_angles.keys(), key=lambda a: min(
            abs(action_angles[a] - angle_deg_center),
            360.0 - abs(action_angles[a] - angle_deg_center),
        ))
        expected = _DIR16_TO_ACTION9[dir16]
        status = "OK" if best_action == expected else "FAIL"
        if best_action != expected:
            errors.append(f"dir16={dir16}: computed={best_action}, table={expected}")
        print(f"  dir16={dir16:2d} ({angle_deg_center:8.2f}°) → action {expected} | {status}")

    if errors:
        print("\n[FAIL] 不一致:")
        for e in errors:
            print(f"  {e}")
        raise AssertionError("_DIR16_TO_ACTION9 テーブルに誤りがあります")
    else:
        print("\n[OK] すべてのマッピングが正しいです")

    # C++ 実装の action → dir16 検証
    print("\n=== action → dir16 検証 ===")
    for action, angle_deg in action_angles.items():
        dir16 = _ue5_atan2_to_dir16(angle_deg)
        reverse = _DIR16_TO_ACTION9[dir16]
        print(f"  action {action} ({angle_deg:7.2f}°) → dir16={dir16} → action {reverse}")
