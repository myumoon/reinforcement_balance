"""Survivors ルールベース行動クローニング（ポリシー事前初期化）.

run_bc.py から呼び出される。train.py からの直接呼び出しは廃止。
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

# wall_rays インデックス → 9方向アクション変換テーブル
# C++ RayDirs (SurvivorsGameConstants.h): [E, NE, N, NW, W, SW, S, SE]
# 9方向アクション:                         [N=0, NE=1, E=2, SE=3, S=4, SW=5, W=6, NW=7]
#   wall[0]=E  → action 2 (East)
#   wall[1]=NE → action 1 (NE)
#   wall[2]=N  → action 0 (North)
#   wall[3]=NW → action 7 (NW)
#   wall[4]=W  → action 6 (West)
#   wall[5]=SW → action 5 (SW)
#   wall[6]=S  → action 4 (South)
#   wall[7]=SE → action 3 (SE)
_WALL_RAY_TO_ACTION9: list[int] = [2, 1, 0, 7, 6, 5, 4, 3]

# 壁が近いと判断する wall_rays の閾値（0=壁接触, 1=遠い）
_WALL_CLOSE_THRESHOLD: float = 0.15
# enemy_nearest_dist が下回ったら後退りする閾値（0=接触, 1=24m以上）
_CONTACT_DIST_THRESHOLD: float = 0.15
# 「包囲」と判断する最小密度（全方向がこれを超えると包囲とみなす）
_SURROUND_MIN_DENSITY: float = 0.08
# 「包囲」と判断する密度レンジの上限（差がこれ未満なら全方向均一=包囲）
_SURROUND_MAX_RANGE: float = 0.35
# 中心付近（center return を無効化する半径）
_CENTER_NEAR_THRESHOLD: float = 0.15

# action9 の角度（度）: index = action9, None = idle(8)
# 0=N(90°), 1=NE(45°), 2=E(0°), 3=SE(-45°), 4=S(-90°), 5=SW(-135°), 6=W(180°), 7=NW(135°)
_ACTION9_ANGLES_DEG: list[float] = [90.0, 45.0, 0.0, -45.0, -90.0, -135.0, 180.0, 135.0]

# wall_rays の action9 → wall_ray インデックスの逆引き
# _WALL_RAY_TO_ACTION9 = [E=2, NE=1, N=0, NW=7, W=6, SW=5, S=4, SE=3]
_ACTION9_TO_WALL_RAY: dict[int, int] = {2: 0, 1: 1, 0: 2, 7: 3, 6: 4, 5: 5, 4: 6, 3: 7}

# モジュールレベルの乱数生成器（rule_policy の同点処理に使用）
_RNG: np.random.Generator = np.random.default_rng()


def set_rule_policy_seed(seed: int) -> None:
    """rule_policy の同点ランダム選択のシードを設定する（BC 再現性用）。"""
    global _RNG
    _RNG = np.random.default_rng(seed)


def _angle_diff_deg(a: float, b: float) -> float:
    """2つの角度（度）の最小差を返す [0, 180]。"""
    d = abs(a - b) % 360.0
    return d if d <= 180.0 else 360.0 - d


def _get_center_action9(px: float, py: float) -> int | None:
    """player_pos から中心方向の action9 を返す。中心付近なら None。

    px = PlayerPos.X / FieldHalfSize: +East, -West
    py = PlayerPos.Y / FieldHalfSize: +North, -South
    """
    if abs(px) < _CENTER_NEAR_THRESHOLD and abs(py) < _CENTER_NEAR_THRESHOLD:
        return None
    # 中心方向ベクトル = (-px, -py) in (East, North) 空間
    angle_deg = float(np.degrees(np.arctan2(-py, -px)))
    return int(min(range(8), key=lambda a: _angle_diff_deg(_ACTION9_ANGLES_DEG[a], angle_deg)))


def _select_best_action9(
    dir16_scores: np.ndarray,
    blocked: set[int],
    center_a9: int | None = None,
    rng: np.random.Generator | None = None,
    eps: float = 1e-6,
) -> int | None:
    """dir16 スコアから最良の action9 を選ぶ。

    - 同じ action9 にマップされる dir16 は最大値を採用
    - blocked に含まれる action9 は除外
    - 同点時: 中心方向に近い候補を優先 → それでも複数ならランダム選択
    - 全 action9 がブロックされている場合は None を返す
    """
    a9_scores: dict[int, float] = {}
    for d16 in range(16):
        a9 = _DIR16_TO_ACTION9[d16]
        if a9 in blocked:
            continue
        s = float(dir16_scores[d16])
        if a9 not in a9_scores or s > a9_scores[a9]:
            a9_scores[a9] = s

    if not a9_scores:
        return None

    max_s = max(a9_scores.values())
    candidates = [a9 for a9, s in a9_scores.items() if max_s - s <= eps]

    if len(candidates) == 1:
        return candidates[0]

    # 中心方向に近い候補を優先
    if center_a9 is not None and center_a9 < 8:
        ca_deg = _ACTION9_ANGLES_DEG[center_a9]
        candidates.sort(key=lambda a: _angle_diff_deg(_ACTION9_ANGLES_DEG[a], ca_deg))
        min_diff = _angle_diff_deg(_ACTION9_ANGLES_DEG[candidates[0]], ca_deg)
        candidates = [a for a in candidates if _angle_diff_deg(_ACTION9_ANGLES_DEG[a], ca_deg) <= min_diff + eps]
        if len(candidates) == 1:
            return candidates[0]

    _rng = rng if rng is not None else _RNG
    return int(_rng.choice(candidates))


def rule_policy(obs: np.ndarray, offsets: dict, rng: np.random.Generator | None = None) -> int:
    """ルールベース方策。生の（正規化前）obs を想定するが VecNormalize 後でも動作する。

    argmax ベースの比較は VecNormalize の線形変換に対して不変。
    相対閾値（range の割合）を使うことで正規化の有無に依存しない。

    行動方針:
      敵を Aura 範囲に引き込んで倒し Gem を落とさせる → Gem を拾う、が基本。

    優先度:
      1. [常時フィルタ] 壁が近い action9 を blocked セットに追加
      2. 接触距離に敵 → 最接近方向の反対へ退避（blocked 除外）
      3. 包囲（全方向密度均一・高）→ 最低密度方向へ脱出（blocked 除外）
      4. Gem 収集 → gem スコア（near+mid+dist）× 安全スコアの最大方向（blocked 除外・中心優先）
      5. 敵に接近 → enemy_density_mid の最大方向（blocked 除外・中心優先）
      6. 中心復帰 → player_pos から中心方向（blocked 除外）
      7. 開けた方向 → argmax(wall_rays) 方向
      8. Idle

    Returns:
        int: 0–8 の行動
    """
    DIR_COUNT = 16
    o_en = offsets["enemy_density_near_16dir"]
    o_gn = offsets["gem_density_near_16dir"]
    o_nd = offsets["enemy_nearest_dist_16dir"]
    o_em = offsets.get("enemy_density_mid_16dir")
    o_wr = offsets.get("wall_rays")
    o_pp = offsets.get("player_pos")
    o_gd = offsets.get("gem_nearest_dist_16dir")
    o_gm = offsets.get("gem_density_mid_16dir")

    enemy_near = obs[o_en:o_en + DIR_COUNT].astype(np.float64)
    gem_near   = obs[o_gn:o_gn + DIR_COUNT].astype(np.float64)
    enemy_nd   = obs[o_nd:o_nd + DIR_COUNT].astype(np.float64)

    nd_min = float(np.min(enemy_nd))
    nd_max = float(np.max(enemy_nd))

    # ── 1. 壁が近い action9 を blocked に追加（全ステップ共通フィルタ） ──
    # 壁が近い方向に加え、その隣接方向（斜め）もブロックする。
    # 例: 北壁が近い → N(0), NE(1), NW(7) をブロック
    blocked: set[int] = set()
    wall: np.ndarray | None = None
    if o_wr is not None:
        wall = obs[o_wr:o_wr + 8].astype(np.float64)
        for ray_i, dist in enumerate(wall):
            if float(dist) < _WALL_CLOSE_THRESHOLD:
                a9 = _WALL_RAY_TO_ACTION9[ray_i]
                blocked.add(a9)
                blocked.add((a9 - 1) % 8)   # 隣接する斜め方向
                blocked.add((a9 + 1) % 8)

    # player_pos から中心方向を取得（中心付近なら None）
    center_a9: int | None = None
    if o_pp is not None:
        px = float(obs[o_pp])
        py = float(obs[o_pp + 1])
        center_a9 = _get_center_action9(px, py)

    # ── 2. 接触距離に敵 → 反対方向へ退避 ──
    if nd_min < _CONTACT_DIST_THRESHOLD:
        danger_d16 = int(np.argmin(enemy_nd))
        retreat_d16 = (danger_d16 + DIR_COUNT // 2) % DIR_COUNT
        retreat_a9 = _DIR16_TO_ACTION9[retreat_d16]
        if retreat_a9 not in blocked:
            return retreat_a9
        # 退避方向が壁ならば、安全スコアで最善の非blocked方向を選ぶ
        act = _select_best_action9(enemy_nd, blocked, center_a9, rng)
        if act is not None:
            return act

    # ── 3. 包囲判定 → 最低密度方向へ脱出 ──
    en_min = float(np.min(enemy_near))
    en_range = float(np.max(enemy_near)) - en_min
    if en_min > _SURROUND_MIN_DENSITY and en_range < _SURROUND_MAX_RANGE:
        act = _select_best_action9(-enemy_near, blocked, center_a9, rng)
        if act is not None:
            return act

    # ── 4. Gem 収集 ──
    # gem スコア: near(0.5) + mid(0.25) + (1-dist)(0.25)
    gem_score = gem_near * 0.5
    if o_gm is not None:
        gem_mid = obs[o_gm:o_gm + DIR_COUNT].astype(np.float64)
        gem_score = gem_score + gem_mid * 0.25
    if o_gd is not None:
        gem_nd = obs[o_gd:o_gd + DIR_COUNT].astype(np.float64)
        gem_score = gem_score + (1.0 - gem_nd) * 0.25

    # 安全スコア: enemy_nd 高い方向ほど安全、enemy_near/mid が高い方向は減点
    en_max = float(np.max(enemy_near))
    safety = (enemy_nd / nd_max) if nd_max > 1e-6 else np.ones(DIR_COUNT)
    if o_em is not None:
        enemy_mid = obs[o_em:o_em + DIR_COUNT].astype(np.float64)
        em_max = float(np.max(enemy_mid))
        safety = (
            safety
            - 0.3 * (enemy_near / en_max if en_max > 1e-6 else 0.0)
            - 0.2 * (enemy_mid / em_max if em_max > 1e-6 else 0.0)
        )

    final_gem = gem_score * np.clip(safety, 0.0, 1.0)
    if float(np.max(final_gem)) > 1e-6:
        act = _select_best_action9(final_gem, blocked, center_a9, rng)
        if act is not None:
            return act

    # ── 5. 敵接近（Aura 範囲に引き込む） ──
    if o_em is not None:
        enemy_mid_arr = obs[o_em:o_em + DIR_COUNT].astype(np.float64)
        em_range = float(np.max(enemy_mid_arr)) - float(np.min(enemy_mid_arr))
        if em_range > 1e-6:
            act = _select_best_action9(enemy_mid_arr, blocked, center_a9, rng)
            if act is not None:
                return act

    # ── 6. 中心復帰 ──
    if center_a9 is not None:
        if center_a9 not in blocked:
            return center_a9
        # 中心方向がブロックされていれば、中心に最も近い非blocked方向を選ぶ
        unblocked = [a for a in range(8) if a not in blocked]
        if unblocked:
            ca_deg = _ACTION9_ANGLES_DEG[center_a9]
            return min(unblocked, key=lambda a: _angle_diff_deg(_ACTION9_ANGLES_DEG[a], ca_deg))

    # ── 7. 開けた方向（wall_rays argmax） ──
    if wall is not None:
        unblocked = [a for a in range(8) if a not in blocked]
        if unblocked:
            return max(unblocked, key=lambda a: wall[_ACTION9_TO_WALL_RAY[a]])

    # ── 8. Idle ──
    return 8


def bc_warmup(
    model,
    env,
    n_episodes: int = 100,
    epochs: int = 30,
    lr: float = 3e-4,
    batch_size: int = 512,
    verbose: int = 1,
) -> dict:
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

    Returns:
        dict: BC 統計情報
            transitions: 収集遷移数
            episode_rewards: エピソード報酬リスト
            episode_rewards_mean: 平均エピソード報酬
            episode_length_mean: 平均エピソード長
            action_counts: アクション別選択数 (index 0–8)
            final_loss: 最終エポックの平均 BC loss
            loss_history: エポックごとの平均 BC loss リスト
    """
    raw_env = _get_raw_env(env)
    offsets: dict | None = getattr(raw_env, "_offsets", None)
    if offsets is None:
        if verbose:
            print("[BC] _offsets が取得できないため BC をスキップします")
        return {}
    required = {"enemy_density_near_16dir", "gem_density_near_16dir", "enemy_nearest_dist_16dir"}
    missing = required - set(offsets.keys())
    if missing:
        if verbose:
            print(f"[BC] 必要な obs セグメントが存在しないため BC をスキップします: {missing}")
        return {}

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
    episode_lengths: list[int] = []
    action_counts = [0] * 9
    ep_count = 0
    ep_reward = 0.0
    ep_length = 0

    obs = env.reset()

    while ep_count < n_episodes:
        if vec_norm is not None:
            raw_obs = vec_norm.get_original_obs()[0]
        else:
            raw_obs = obs[0]

        action = rule_policy(raw_obs, offsets)
        obs_list.append(obs[0].copy())
        act_list.append(action)
        action_counts[action] += 1

        obs, reward, done, _info = env.step(np.array([action]))
        ep_reward += float(reward[0])
        ep_length += 1

        if done[0]:
            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_length)
            ep_reward = 0.0
            ep_length = 0
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
            f"平均エピソード報酬={float(np.mean(episode_rewards)):.1f}, "
            f"平均エピソード長={float(np.mean(episode_lengths)):.0f}"
        )
        print(f"[BC] BC 訓練開始 (epochs={epochs}, batch_size={batch_size}, lr={lr})...")

    train_stats = _bc_train(
        model,
        np.array(obs_list, dtype=np.float32),
        np.array(act_list, dtype=np.int64),
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        verbose=verbose,
    )

    model._last_obs = obs
    model._last_episode_starts = done.copy()

    if verbose:
        print("[BC] BC 初期化完了")

    return {
        "transitions": total_transitions,
        "episode_rewards": episode_rewards,
        "episode_rewards_mean": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "episode_length_mean": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        "action_counts": action_counts,
        "final_loss": train_stats["final_loss"],
        "loss_history": train_stats["loss_history"],
    }


def _bc_train(
    model,
    obs_arr: np.ndarray,
    act_arr: np.ndarray,
    epochs: int,
    lr: float,
    batch_size: int,
    verbose: int,
) -> dict:
    """BC 訓練ループ（内部ヘルパー）.

    Returns:
        dict: final_loss, loss_history
    """
    is_recurrent = hasattr(model.policy, "lstm_actor")
    device = model.device
    n = len(obs_arr)

    obs_t = torch.as_tensor(obs_arr, dtype=torch.float32).to(device)
    act_t = torch.as_tensor(act_arr, dtype=torch.long).to(device)

    opt = Adam(model.policy.parameters(), lr=lr)
    model.policy.set_training_mode(True)

    loss_history: list[float] = []

    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        ep_losses: list[float] = []

        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            ob = obs_t[idx]
            ac = act_t[idx]

            if not is_recurrent:
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

        epoch_loss = float(np.mean(ep_losses))
        loss_history.append(epoch_loss)

        if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
            print(f"[BC]   epoch {epoch + 1:3d}/{epochs}  loss={epoch_loss:.4f}")

    model.policy.set_training_mode(False)
    return {"final_loss": loss_history[-1] if loss_history else 0.0, "loss_history": loss_history}


def validate_bc_model(
    model,
    env,
    n_episodes: int = 5,
    max_stationary_ratio: float = 0.35,
    min_move_speed: float = 0.15,
    max_dominant_action_ratio: float = 0.35,
    max_wall_near_ratio: float = 0.30,
    min_gem_pickups_est: float = 3.0,
    min_episode_len: float = 1200.0,
    verbose: int = 1,
) -> dict:
    """BC済みモデルをエピソード評価し、品質ゲートを確認する。

    検証指標:
        stationary_ratio:        静止(move_speed<0.003)ステップ割合
        move_speed_mean:         平均移動速度（info["move_speed"] の平均）
        dominant_action_ratio:   最頻アクションの割合
        wall_near_ratio:         壁近接(is_wall_near)ステップ割合
        gem_pickups_est_mean:    エピソード平均Gem取得推定数
        episode_length_mean:     エピソード平均長

    Returns:
        dict:
            passed (bool): すべての閾値を満たした場合 True
            metrics (dict): 各指標の計測値
            thresholds (dict): 使用した閾値
            fail_reasons (list[str]): 失敗理由のリスト
    """
    is_recurrent = hasattr(model.policy, "lstm_actor")

    vec_norm = _find_vecnormalize(env)
    was_training = None
    if vec_norm is not None:
        was_training = vec_norm.training
        vec_norm.training = False

    total_steps = 0
    stationary_steps = 0
    wall_near_steps = 0
    speed_sum = 0.0
    action_counts = [0] * 9
    ep_lengths: list[int] = []
    ep_gem_pickups: list[int] = []

    lstm_states = None
    episode_starts = np.ones(1, dtype=bool)
    obs = env.reset()

    ep_steps = 0
    ep_gem_est = 0
    prev_xp: float | None = None

    try:
        while len(ep_lengths) < n_episodes:
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=True,
            )
            obs, _, done, infos = env.step(action)
            info = infos[0]

            episode_starts = done

            ep_steps += 1
            total_steps += 1
            action_counts[int(action[0])] += 1

            speed = float(info.get("move_speed", 0.0) or 0.0)
            speed_sum += speed
            if info.get("is_stationary"):
                stationary_steps += 1
            if info.get("is_wall_near"):
                wall_near_steps += 1

            xp_now = float(info.get("xp_progress", 0.0) or 0.0)
            if prev_xp is not None and xp_now > prev_xp + 0.005:
                ep_gem_est += 1
            prev_xp = xp_now

            if done[0]:
                ep_lengths.append(ep_steps)
                ep_gem_pickups.append(ep_gem_est)
                ep_steps = 0
                ep_gem_est = 0
                prev_xp = None
                lstm_states = None
                episode_starts = np.ones(1, dtype=bool)
    finally:
        if vec_norm is not None and was_training is not None:
            vec_norm.training = was_training

    episodes = len(ep_lengths)
    if total_steps == 0 or episodes == 0:
        return {
            "passed": False,
            "metrics": {},
            "thresholds": {},
            "fail_reasons": ["評価エピソードが0件でした"],
        }

    dom_action = int(np.argmax(action_counts))
    metrics = {
        "stationary_ratio":       stationary_steps / total_steps,
        "move_speed_mean":        speed_sum / total_steps,
        "dominant_action":        dom_action,
        "dominant_action_ratio":  max(action_counts) / total_steps,
        "wall_near_ratio":        wall_near_steps / total_steps,
        "gem_pickups_est_mean":   float(np.mean(ep_gem_pickups)),
        "episode_length_mean":    float(np.mean(ep_lengths)),
    }
    thresholds = {
        "max_stationary_ratio":       max_stationary_ratio,
        "min_move_speed":             min_move_speed,
        "max_dominant_action_ratio":  max_dominant_action_ratio,
        "max_wall_near_ratio":        max_wall_near_ratio,
        "min_gem_pickups_est":        min_gem_pickups_est,
        "min_episode_len":            min_episode_len,
    }

    fail_reasons: list[str] = []
    if metrics["stationary_ratio"] > max_stationary_ratio:
        fail_reasons.append(f"stationary_ratio {metrics['stationary_ratio']:.3f} > {max_stationary_ratio}")
    if metrics["move_speed_mean"] < min_move_speed:
        fail_reasons.append(f"move_speed_mean {metrics['move_speed_mean']:.3f} < {min_move_speed}")
    if metrics["dominant_action_ratio"] > max_dominant_action_ratio:
        fail_reasons.append(
            f"dominant_action_ratio {metrics['dominant_action_ratio']:.3f} > {max_dominant_action_ratio}"
            f" (action={dom_action})"
        )
    if metrics["wall_near_ratio"] > max_wall_near_ratio:
        fail_reasons.append(f"wall_near_ratio {metrics['wall_near_ratio']:.3f} > {max_wall_near_ratio}")
    if metrics["gem_pickups_est_mean"] < min_gem_pickups_est:
        fail_reasons.append(f"gem_pickups_est_mean {metrics['gem_pickups_est_mean']:.2f} < {min_gem_pickups_est}")
    if metrics["episode_length_mean"] < min_episode_len:
        fail_reasons.append(f"episode_length_mean {metrics['episode_length_mean']:.0f} < {min_episode_len}")

    passed = len(fail_reasons) == 0

    if verbose:
        status = "PASSED" if passed else "FAILED"
        print(
            f"[BC検証] {status} ({episodes}エピソード)  "
            f"stationary={metrics['stationary_ratio']:.2f}  "
            f"speed={metrics['move_speed_mean']:.3f}  "
            f"dom_action={metrics['dominant_action_ratio']:.2f}  "
            f"wall_near={metrics['wall_near_ratio']:.2f}  "
            f"gem_est={metrics['gem_pickups_est_mean']:.1f}  "
            f"ep_len={metrics['episode_length_mean']:.0f}"
        )
        for reason in fail_reasons:
            print(f"[BC検証]   NG: {reason}")

    return {
        "passed": passed,
        "metrics": metrics,
        "thresholds": thresholds,
        "fail_reasons": fail_reasons,
    }


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
    import math

    def _ue5_atan2_to_dir16(angle_deg: float) -> int:
        angle_rad = math.radians(angle_deg)
        angle01 = (angle_rad + math.pi) / (2.0 * math.pi)
        return max(0, min(15, int(angle01 * 16)))

    action_angles = {0: 90.0, 1: 45.0, 2: 0.0, 3: -45.0, 4: -90.0, 5: -135.0, 6: 180.0, 7: 135.0}

    print("=== dir16 → action9 テーブル検証 ===")
    errors = []
    for dir16 in range(16):
        angle01_center = (dir16 + 0.5) / 16.0
        angle_rad_center = angle01_center * 2.0 * math.pi - math.pi
        angle_deg_center = math.degrees(angle_rad_center)
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
        raise AssertionError(f"_DIR16_TO_ACTION9 テーブルに誤りがあります: {errors}")
    print("\n[OK] すべてのマッピングが正しいです")
