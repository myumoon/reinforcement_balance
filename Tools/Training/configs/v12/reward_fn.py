import numpy as np

def reward_shaping(obs: np.ndarray, prev_obs: np.ndarray, base_reward: float) -> float:
    """
    Survivors reward shaping v12.
    reward_fn_policy.md の 12 項目チェックリストに対応した改訂版。

    変更履歴（Fix 識別子は reward_fn_policy.md / 実装プランに準拠）:
      Fix A: Section 12a — Garlic 主体時は緊急接近ペナルティを除外（Item 1）
             garlic_auto_ratio >= 0.5 の場合は 12a を発動しない。
             混在ロードアウト（Garlic+KingBible 等）では引き続き発動し KingBible を保護。
      Fix B: Section 13 — 射程バンド超過時の明示的ペナルティ追加（Item 3）
             is_near_type 問わず、バンドの 2 倍以上逃げると追加マイナスを付与。
      Fix C: Section 1b 新設 — 全武器共通 cooldown 集計（Item 4）
             all_cooldown_max / active_slot_exists / best_active_level を事前計算。
             Section 5a の density_pen_mult に cooldown_density_mult を乗算。
      Fix D: Section 6a 新設 — アクティブ強武器による高密度方向移動ボーナス（Item 9）
             cooldown < 0.2 かつ level_norm >= 0.3 の武器があれば前方密度ボーナスを付与。
      Fix E: Section 13 — 複数武器同時射程帯重複ボーナス + DPS 優先重み（Item 6）
             同一ステップで射程帯が重複するスロット数に応じてボーナス加算。
             優先武器（高 Lv・低 cooldown）と同距離帯に入ると重みを強化。
      Fix F: Section 9 改修 — Gem 接近デルタに密度重みを乗算（Item 8）
             移動方向の gem_density_near が高いほど接近ボーナスを増幅。
      Fix G: Section 15a 新設 — レベルアップ直前の Gem 加速（Item 12）
             xp_progress > 0.7 で Gem 収集シグナルを最大 1.5 倍に強化。
      Fix H: Section 12a — 緊急接近ペナルティを -0.10 → -0.04 に緩和し Gem 近接ガードを追加。
             Gem が収集圏内（curr_min_gem < 0.08）の場合はペナルティをさらに半減。
             Section 9 の Gem 接近ボーナス（最大 +0.025）と同規模に揃え Gem 収集阻害を解消。
      Fix I: Section 10 — Gem ピックアップボーナスを 0.02 → 0.05 に強化（最大 +0.25）。
             Section 9 の連続接近シグナルと両方から Gem 収集行動を促す。
      v12 update: obs layout を v795 (890 dims) に更新。
                  projectiles が stride 9 (288 dim) に拡張され、
                  weapon_attack_range_norm が [740:746] から [836:842] に移動。

    obs レイアウト (890 dims, obs_schema v795):
      [0:2]    player_pos (x/HN, y/HN)
      [2:4]    player_vel (vx/MoveSpeed, vy/MoveSpeed)
      [4:12]   wall_rays (8 dirs, 0=wall close, 1=far)
      [12]     player_hp_norm
      [13]     shield_active
      [14]     shield_timer_norm
      [15]     revival_remaining_norm
      [16]     armor_flat_norm
      [17]     regen_per_sec_norm
      [18:23]  passive_effect_summary (5)
      [23:41]  weapon_slots (6 × 3: type_norm, level_norm, cooldown_norm)
      [41:53]  passive_slots (6 × 2)
      [53]     enemy_count_norm
      [54]     elapsed_time_norm
      [55]     xp_progress
      [56]     player_level_norm
      [57]     stage_id_norm
      [58:78]  red_gem_rel_pos   (10 × 2, dx/DN, dy/DN)
      [78:102] green_gem_rel_pos (12 × 2)
      [102:126] blue_gem_rel_pos (12 × 2)
      [126]    gem_pickup_radius_norm
      [127:191] enemy_rel_pos  (32 × 2, dx/DN, dy/DN)
      [191:255] enemy_vel      (32 × 2, vx/MoveSpeed, vy/MoveSpeed)
      [255:287] enemy_type     (32)
      [287:319] enemy_hp       (32)
      [319:351] enemy_frozen   (32)
      [351:367] enemy_nearest_dist_16dir   (正規化基準: EnemyNearestDistanceMax=2400u)
      [367:383] enemy_density_near_16dir
      [383:399] enemy_density_mid_16dir
      [399:447] gem_density_all_16dir  (3 × 16)
      [447:495] red_green_gem_density_16dir (3 × 16)
      [495:783] projectiles (32 × 9): dx, dy, radius_norm, vx_norm, vy_norm, warning, kind_norm, slot_norm, ttl_norm
      [783:807] floor_pickups (8 × 3)
      [807:816] special_pickups (3 × 3)
      [816:836] destructibles (10 × 2)
      [836:842] weapon_attack_range_norm (6スロット)
      [842:848] weapon_is_directional (6スロット)
      [848:890] weapon_category_onehot (6スロット × 7カテゴリ)
    """
    shaped = 0.0

    # ============================================================
    # 0. Parse observation segments
    # ============================================================
    player_vel = obs[2:4]
    wall_rays = obs[4:12]
    player_hp_norm = obs[12]
    shield_active = obs[13]

    weapon_slots_raw = obs[23:41].reshape(6, 3)

    xp_progress = obs[55]

    red_gem_rel = obs[58:78].reshape(10, 2)
    green_gem_rel = obs[78:102].reshape(12, 2)
    blue_gem_rel = obs[102:126].reshape(12, 2)

    enemy_rel = obs[127:191].reshape(32, 2)

    enemy_nearest_16 = obs[351:367]
    enemy_density_near_16 = obs[367:383]
    enemy_density_mid_16 = obs[383:399]

    gem_density_all = obs[399:447].reshape(3, 16)
    gem_nearest_16 = gem_density_all[0]
    gem_density_near_16 = gem_density_all[1]
    gem_density_mid_16 = gem_density_all[2]

    weapon_range_norms = obs[836:842]

    # ============================================================
    # 1. 武器分類 (v09: 6グループ + defensive)
    # ============================================================
    GARLIC_AUTO_IDS = {1, 16}
    ORBITAL_IDS = {7, 22}
    MELEE_LINE_IDS = {2, 17}
    RANGED_TARGETED_IDS = {3, 8, 11, 18, 26}
    RANGED_DIRECTIONAL_IDS = {4, 5, 6, 13, 14, 19, 20, 21, 28}
    FOLDING_IDS = {5, 6, 20, 21}
    AREA_DROP_IDS = {9, 10, 12, 23, 24, 25, 27}
    DEFENSIVE_IDS = {15}

    garlic_auto_count = 0
    orbital_count = 0
    melee_line_count = 0
    ranged_t_count = 0
    directional_count = 0
    folding_count = 0
    area_drop_count = 0
    defensive_count = 0
    equipped_count = 0
    orbital_cooldown_max = 0.0

    for slot_i in range(6):
        tn = weapon_slots_raw[slot_i, 0]
        if tn < 1e-4:
            continue
        equipped_count += 1
        wtype_id = int(round(tn * 64.0))
        if wtype_id in GARLIC_AUTO_IDS:
            garlic_auto_count += 1
        elif wtype_id in ORBITAL_IDS:
            orbital_count += 1
            orbital_cooldown_max = max(orbital_cooldown_max, weapon_slots_raw[slot_i, 2])
        elif wtype_id in MELEE_LINE_IDS:
            melee_line_count += 1
        elif wtype_id in RANGED_TARGETED_IDS:
            ranged_t_count += 1
        elif wtype_id in RANGED_DIRECTIONAL_IDS:
            directional_count += 1
            if wtype_id in FOLDING_IDS:
                folding_count += 1
        elif wtype_id in AREA_DROP_IDS:
            area_drop_count += 1
        elif wtype_id in DEFENSIVE_IDS:
            defensive_count += 1

    if equipped_count == 0:
        equipped_count = 1

    garlic_auto_ratio  = garlic_auto_count / equipped_count
    orbital_ratio      = orbital_count / equipped_count
    melee_line_ratio   = melee_line_count / equipped_count
    ranged_t_ratio     = ranged_t_count / equipped_count
    directional_ratio  = directional_count / equipped_count
    folding_ratio      = folding_count / equipped_count
    area_drop_ratio    = area_drop_count / equipped_count
    defensive_ratio    = defensive_count / equipped_count

    # Orbital は orbit 自動攻撃のため approach_bonus 対象外
    near_melee_ratio = garlic_auto_ratio + melee_line_ratio

    density_pen_mult = (
        garlic_auto_ratio            * 0.2 +
        orbital_ratio                * 0.3 +
        melee_line_ratio             * 0.5 +
        folding_ratio                * 0.4 +
        (directional_ratio - folding_ratio) * 1.0 +
        ranged_t_ratio               * 0.9 +
        area_drop_ratio              * 0.6 +
        defensive_ratio              * 0.8
    )
    if density_pen_mult < 1e-6:
        density_pen_mult = 1.0

    # ============================================================
    # 1b. [Fix C] 全武器共通: cooldown 集計・アクティブ武器情報
    #    all_cooldown_max   : 全スロット中の最大 cooldown_norm
    #    active_slot_exists : cooldown_norm < 0.2 の装備スロットが存在するか
    #    best_active_level  : アクティブスロット中の最大 level_norm
    # ============================================================
    all_cooldown_max = 0.0
    active_slot_exists = False
    best_active_level = 0.0

    for slot_i in range(6):
        tn = weapon_slots_raw[slot_i, 0]
        if tn < 1e-4:
            continue
        cd = float(weapon_slots_raw[slot_i, 2])
        lv = float(weapon_slots_raw[slot_i, 1])
        all_cooldown_max = max(all_cooldown_max, cd)
        if cd < 0.2:
            active_slot_exists = True
            best_active_level = max(best_active_level, lv)

    # クールダウン中ほど密度ペナルティを強化（最大 1.5 倍）
    cooldown_density_mult = 1.0 + 0.5 * all_cooldown_max

    # ============================================================
    # 2. 移動方向の計算
    # ============================================================
    vel_mag = float(np.sqrt(player_vel[0]**2 + player_vel[1]**2))
    is_moving = vel_mag > 0.05

    if is_moving:
        angle_rad = float(np.arctan2(player_vel[1], player_vel[0]))
        angle_01 = (angle_rad + np.pi) / (2.0 * np.pi)
        move_dir = int(np.clip(np.floor(angle_01 * 16), 0, 15))
        vel_norm = player_vel / vel_mag
    else:
        move_dir = -1
        vel_norm = np.zeros(2)

    # ============================================================
    # 3. 最近傍敵の方向ベクトルを計算 (接近ボーナス用)
    # ============================================================
    nearest_enemy_dir = None
    dists_sq = enemy_rel[:, 0]**2 + enemy_rel[:, 1]**2
    valid_mask = dists_sq > 1e-8
    if np.any(valid_mask):
        nearest_idx = int(np.argmin(np.where(valid_mask, dists_sq, np.inf)))
        d = float(np.sqrt(dists_sq[nearest_idx]))
        nearest_enemy_dir = enemy_rel[nearest_idx] / d

    # ============================================================
    # 5. 方向品質ボーナス (安全 + Gem リッチな方向)
    # ============================================================
    gem_near_dens_d = 0.0
    gem_mid_dens_d = 0.0
    gem_richness = 0.0
    safety_factor = 1.0

    if is_moving and move_dir >= 0:
        d = move_dir

        enemy_nearest_d = float(enemy_nearest_16[d])
        enemy_near_dens_d = float(enemy_density_near_16[d])
        enemy_mid_dens_d = float(enemy_density_mid_16[d])

        gem_nearest_d = float(gem_nearest_16[d])
        gem_near_dens_d = float(gem_density_near_16[d])
        gem_mid_dens_d = float(gem_density_mid_16[d])

        # 5a. 敵密度ペナルティ
        # [Fix C] cooldown_density_mult: クールダウン中ほど密度突入を強く抑制
        near_density_pen = -0.015 * density_pen_mult * cooldown_density_mult * min(enemy_near_dens_d, 4.0)
        mid_density_pen  = -0.005 * density_pen_mult * cooldown_density_mult * min(enemy_mid_dens_d, 6.0)
        danger_close_pen = 0.0
        if enemy_nearest_d < 0.15:
            danger_close_pen = -0.02 * density_pen_mult * (1.0 - enemy_nearest_d / 0.15)
        shaped += near_density_pen + mid_density_pen + danger_close_pen

        # 5b. Gem 方向ボーナス
        gem_richness = gem_near_dens_d + 0.5 * gem_mid_dens_d
        safety_factor = max(0.0, 1.0 - 0.3 * enemy_near_dens_d - 0.1 * enemy_mid_dens_d)
        safety_factor = min(safety_factor, 1.0)
        gem_safety_floor = 0.3 * (ranged_t_ratio + directional_ratio)
        safety_factor = max(gem_safety_floor, safety_factor)

        hp_gate = min(player_hp_norm / 0.4, 1.0)

        gem_dir_bonus  = 0.02 * min(gem_richness, 3.0) * safety_factor * hp_gate
        gem_close_bonus = 0.01 * max(0.0, 1.0 - gem_nearest_d) * safety_factor * hp_gate
        shaped += gem_dir_bonus + gem_close_bonus

        # 5c. Orbital クールダウン中の敵密度方向ペナルティ（Orbital 特化の精度維持）
        if orbital_ratio > 0.1 and orbital_cooldown_max > 0.3:
            cooldown_density_pen = -0.01 * orbital_ratio * orbital_cooldown_max * min(enemy_near_dens_d, 4.0)
            shaped += cooldown_density_pen

    # ============================================================
    # 6. Garlic/MeleeLine: 敵接近ボーナス
    # ============================================================
    if near_melee_ratio > 0.2 and is_moving and nearest_enemy_dir is not None:
        approach_dot = float(np.dot(vel_norm, nearest_enemy_dir))
        hp_gate_approach = min(player_hp_norm / 0.5, 1.0)
        orbital_cooldown_gate = 1.0 - 0.8 * orbital_ratio * orbital_cooldown_max
        approach_bonus = 0.025 * near_melee_ratio * max(0.0, approach_dot) * hp_gate_approach * orbital_cooldown_gate
        shaped += approach_bonus

    # ============================================================
    # 6a. [Fix D] アクティブ武器が強い → 敵高密度方向移動ボーナス（Item 9）
    #    「倒せる武器があれば Gem ドロップ最大化のため高密度へ」
    #    条件: cooldown < 0.2 のスロットが存在 かつ level_norm >= 0.3（Lv 3 相当）
    #    HP による制限なし（HP 節約より積極撃破が生存に有利なため）
    #    Garlic/Orbital は近接・orbit 自動攻撃のため near_melee_ratio で対応済み → 半減
    # ============================================================
    if active_slot_exists and is_moving and move_dir >= 0 and best_active_level >= 0.3:
        front_enemy_density = float(enemy_density_near_16[move_dir])
        lv_gate = min(best_active_level / 0.5, 1.0)   # Lv 5(level_norm=0.5) で最大
        attack_bonus = (
            0.012 * lv_gate
            * min(front_enemy_density, 4.0)
            * (1.0 - near_melee_ratio * 0.5)  # Garlic/Melee は担当セクションが別途あるため半減
        )
        shaped += attack_bonus

    # ============================================================
    # 6b. Orbital クールダウン中の敵近接ペナルティ（位置ベース）
    # ============================================================
    if orbital_ratio > 0.1 and orbital_cooldown_max > 0.5:
        min_enemy_dist_orbital = float(np.min(enemy_nearest_16))
        if min_enemy_dist_orbital < 0.25:
            cooldown_proximity_pen = (
                -0.02 * orbital_ratio * orbital_cooldown_max
                * (1.0 - min_enemy_dist_orbital / 0.25)
            )
            shaped += cooldown_proximity_pen

    # ============================================================
    # 7. RangedDirectional: 前方敵密度ボーナス
    # ============================================================
    if directional_ratio > 0.2 and is_moving and move_dir >= 0:
        front_density = float(enemy_density_near_16[move_dir])
        front_bonus = 0.015 * directional_ratio * min(front_density, 4.0)
        shaped += front_bonus

    # ============================================================
    # 8. AreaDrop: 最高敵密度方向への移動ボーナス
    # ============================================================
    if area_drop_ratio > 0.2 and is_moving and move_dir >= 0:
        best_density_dir = int(np.argmax(enemy_density_near_16))
        best_density_val = float(enemy_density_near_16[best_density_dir])
        dir_diff = abs(move_dir - best_density_dir)
        dir_match = (dir_diff <= 1 or dir_diff >= 15)
        if dir_match and best_density_val > 0.1:
            area_bonus = 0.01 * area_drop_ratio * min(best_density_val, 4.0)
            shaped += area_bonus

    # ============================================================
    # 9. 最近傍 Gem への接近ボーナス
    # [Fix F] 移動方向の gem_density_near で重み付け → 密集 Gem 方向の接近を優遇
    # ============================================================
    def closest_gem_dist(gem_arr: np.ndarray) -> float:
        dists = np.sqrt(gem_arr[:, 0]**2 + gem_arr[:, 1]**2)
        valid = dists > 1e-6
        return float(np.min(dists[valid])) if np.any(valid) else float('inf')

    curr_min_gem = min(closest_gem_dist(red_gem_rel),
                       closest_gem_dist(green_gem_rel),
                       closest_gem_dist(blue_gem_rel))

    prev_red   = prev_obs[58:78].reshape(10, 2)
    prev_green = prev_obs[78:102].reshape(12, 2)
    prev_blue  = prev_obs[102:126].reshape(12, 2)
    prev_min_gem = min(closest_gem_dist(prev_red),
                       closest_gem_dist(prev_green),
                       closest_gem_dist(prev_blue))

    if curr_min_gem < float('inf') and prev_min_gem < float('inf'):
        gem_approach_delta = prev_min_gem - curr_min_gem
        # [Fix F] 移動方向の Gem 密度が高いほど接近ボーナスを増幅（最大 1.9 倍）
        gem_dir_density = float(gem_density_near_16[move_dir]) if is_moving and move_dir >= 0 else 0.0
        density_weight = 1.0 + min(gem_dir_density, 3.0) * 0.3
        gem_approach_reward = float(np.clip(gem_approach_delta * 0.02 * density_weight, -0.02, 0.025))
        min_enemy_nearest_val = float(np.min(enemy_nearest_16))
        if player_hp_norm < 0.3 and min_enemy_nearest_val < 0.1:
            gem_approach_reward *= 0.3
        shaped += gem_approach_reward

    # ============================================================
    # 10. Gem ピックアップ時の小ボーナス
    # [Fix I] 0.02 → 0.05（最大 +0.25）: 離散イベントへの報酬を強化
    # ============================================================
    if base_reward >= 1.0:
        gem_count_est = min(base_reward, 5.0)
        shaped += 0.05 * gem_count_est

    # ============================================================
    # 11. 壁際ペナルティ
    # ============================================================
    min_wall = float(np.min(wall_rays))
    if min_wall < 0.15:
        wall_pen = -0.015 * (1.0 - min_wall / 0.15)
        shaped += wall_pen
        if float(np.min(enemy_nearest_16)) < 0.2:
            shaped += -0.01

    # ============================================================
    # 12. 低 HP 時の危険ペナルティ
    # ============================================================
    if player_hp_norm < 0.25:
        closest_enemy_dist_val = float(np.min(enemy_nearest_16))
        if closest_enemy_dist_val < 0.15:
            low_hp_danger = (-0.03
                             * (1.0 - closest_enemy_dist_val / 0.15)
                             * (1.0 - player_hp_norm / 0.25))
            shaped += low_hp_danger

    # ============================================================
    # 12a. 全武器共通: 敵との緊急接近ペナルティ
    # [Fix A] Garlic 主体（garlic_auto_ratio >= 0.5）の場合は除外。
    #    Garlic は敵を引き寄せて AoE で倒す武器のため 0 距離は許容される。
    #    混在ロードアウト（Garlic + KingBible 等）では発動し KingBible の 0 距離を抑制。
    # [Fix H] -0.10 → -0.04: Section 9 Gem 接近ボーナス（最大 +0.025）と同規模に揃える。
    #    Gem は敵密集地にドロップするため過大なペナルティが Gem 収集を阻害していた。
    #    curr_min_gem < 0.08（Gem が収集圏内）の場合はさらに半減して Gem 収集を優先。
    # ============================================================
    _emergency_enemy_dist = float(np.min(enemy_nearest_16))
    _EMERGENCY_DIST = 0.025  # ≈ 60u（KingBible orbit 半径の目安）
    if garlic_auto_ratio < 0.5 and _emergency_enemy_dist < _EMERGENCY_DIST:
        base_pen = -0.04 * (1.0 - _emergency_enemy_dist / _EMERGENCY_DIST)
        gem_proximity_gate = 0.5 if curr_min_gem < 0.08 else 1.0
        shaped += base_pen * gem_proximity_gate

    # ============================================================
    # 13. 全武器適正距離維持ボーナス
    # [Fix B] 射程バンドの 2 倍以上逃げた場合に追加ペナルティ（Item 3）
    # [Fix E] 複数スロットが同一ステップで射程帯に入ると重複ボーナス（Item 6）
    #         DPS 優先: cooldown 低 × level 高 のスロットに重み付け
    # ============================================================
    CROSS_BOOMERANG_IDS = {6, 21}
    AXE_ARC_IDS = {5, 20}

    low_hp_range_mult = 1.0 + max(0.0, (0.3 - player_hp_norm) / 0.3)

    min_enemy_dist_val = float(np.min(enemy_nearest_16))
    range_bonus_sum = 0.0
    range_penalty_sum = 0.0
    range_slot_count = 0
    overlap_count = 0  # [Fix E] 適正射程帯に入っているスロット数

    # [Fix E] DPS 優先: 最も殲滅能力が高いスロットを特定（cooldown 低 × level 高）
    priority_dps_score = 0.0
    priority_lo = -1.0
    priority_hi = -1.0
    for slot_i in range(6):
        tn = weapon_slots_raw[slot_i, 0]
        if tn < 1e-4:
            continue
        lv = float(weapon_slots_raw[slot_i, 1])
        cd = float(weapon_slots_raw[slot_i, 2])
        dps_score = lv * (1.0 - cd * 0.5)
        if dps_score > priority_dps_score:
            priority_dps_score = dps_score
            # 優先武器の距離帯は後のループで再計算するため仮置き
            priority_lo = -1.0
            priority_hi = -1.0

    for slot_i in range(6):
        tn = weapon_slots_raw[slot_i, 0]
        if tn < 1e-4:
            continue
        wtype_id_s13 = int(round(tn * 64.0))
        r = float(weapon_range_norms[slot_i])
        lv = float(weapon_slots_raw[slot_i, 1])
        cd = float(weapon_slots_raw[slot_i, 2])

        is_near_type = False
        if wtype_id_s13 in ORBITAL_IDS:
            lo, hi = 0.025, 0.100
            is_near_type = False
        elif wtype_id_s13 in CROSS_BOOMERANG_IDS:
            lo, hi = 0.000, 0.035
            is_near_type = True
        elif wtype_id_s13 in AXE_ARC_IDS:
            lo, hi = 0.000, 0.045
            is_near_type = True
        elif r >= 0.80:
            continue
        elif r <= 0.05:
            lo, hi = 0.000, 0.025
            is_near_type = True
        elif r <= 0.15:
            lo, hi = 0.000, 0.040
            is_near_type = True
        elif r <= 0.35:
            lo, hi = 0.063, 0.200
        elif r <= 0.55:
            lo, hi = 0.083, 0.280
        else:
            lo, hi = 0.104, 0.375

        # [Fix E] 優先武器の距離帯を記録（最初に見つかったもの）
        if priority_lo < 0.0 and lv * (1.0 - cd * 0.5) >= priority_dps_score * 0.95:
            priority_lo = lo
            priority_hi = hi

        # 距離スコアを計算
        range_pen = 0.0
        if is_near_type:
            if min_enemy_dist_val <= hi:
                dist_score = 1.0
            else:
                overshoot = (min_enemy_dist_val - hi) / max(hi, 1e-6)
                dist_score = max(0.0, 1.0 - overshoot * 3.0)
                # [Fix B] 射程の 2 倍以上逃げたら追加ペナルティ
                if overshoot > 1.0:
                    range_pen = -0.005 * min(overshoot - 1.0, 2.0)
        else:
            if min_enemy_dist_val < lo:
                overshoot = (lo - min_enemy_dist_val) / lo if lo > 0.0 else 0.0
                dist_score = max(0.0, 1.0 - overshoot * 2.0)
            elif min_enemy_dist_val <= hi:
                mid = (lo + hi) / 2.0
                half_width = (hi - lo) / 2.0
                dist_score = max(0.0, 1.0 - abs(min_enemy_dist_val - mid) / half_width)
            else:
                overshoot = (min_enemy_dist_val - hi) / max(hi, 1e-6)
                dist_score = max(0.0, 1.0 - overshoot * 2.0)
                # [Fix B] 射程の 2 倍以上逃げたら追加ペナルティ
                if overshoot > 1.0:
                    range_pen = -0.005 * min(overshoot - 1.0, 2.0)

        # [Fix E] DPS 優先重み: 優先武器と同距離帯(lo/hi が一致)なら 1.5 倍
        dps_weight = 1.0
        if priority_lo >= 0.0 and abs(lo - priority_lo) < 1e-4 and abs(hi - priority_hi) < 1e-4:
            dps_weight = 1.5

        slot_bonus = 0.010 * dist_score * low_hp_range_mult * dps_weight
        range_bonus_sum += slot_bonus
        range_penalty_sum += range_pen
        range_slot_count += 1

        if dist_score > 0.5:
            overlap_count += 1

    if range_slot_count > 0:
        shaped += range_bonus_sum / range_slot_count
        shaped += range_penalty_sum / range_slot_count

    # [Fix E] 複数スロットが同時に射程帯内 → 重複ボーナス
    if overlap_count >= 2:
        shaped += 0.005 * min(overlap_count - 1, 3)

    # ============================================================
    # 14. 防御武器の特殊処理
    # ============================================================
    if defensive_ratio > 0.1:
        if shield_active > 0.5:
            shaped += 0.01
        else:
            if float(np.min(enemy_nearest_16)) < 0.2:
                shaped += -0.01

    # ============================================================
    # 15. レベルアップボーナス
    # ============================================================
    prev_xp = float(prev_obs[55])
    curr_xp = float(obs[55])
    if prev_xp > 0.8 and curr_xp < 0.2:
        shaped += 0.02

    # ============================================================
    # 15a. [Fix G] レベルアップ直前 Gem 加速（Item 12）
    #    xp_progress > 0.7 で Gem 収集シグナルを最大 1.5 倍に強化。
    #    早期レベルアップ → 武器・Passive 強化選択肢を早く得る。
    # ============================================================
    if xp_progress > 0.7 and is_moving and move_dir >= 0:
        xp_gate = 1.0 + 0.5 * min((xp_progress - 0.7) / 0.3, 1.0)
        gem_xp_bonus = 0.008 * (xp_gate - 1.0) * min(gem_near_dens_d + gem_richness, 3.0) * safety_factor
        shaped += gem_xp_bonus

    # ============================================================
    # Final clamp
    # ============================================================
    return float(np.clip(shaped, -1.0, 1.0))
