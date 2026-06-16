import numpy as np

def reward_shaping(obs: np.ndarray, prev_obs: np.ndarray, base_reward: float) -> float:
    """
    Survivors reward shaping v9.
    v08 からの主な変更:
      - 武器分類を 6 グループ（garlic_auto / orbital / melee_line /
        ranged_targeted / ranged_directional / area_drop）+ defensive に細分化
      - Garlic/Orbital/MeleeLine: 敵接近ボーナスを追加
        (PR#202 で敵速度が半減 → プレイヤーが積極的に敵に向かわないと DPS が出ない)
      - RangedDirectional: 前方敵密度ボーナスを追加
      - AreaDrop: 最高密度方向一致ボーナスを追加
      - density_pen_mult を武器グループ別に調整 (Garlic 0.5 → 0.2)

    obs レイアウト (740 dims, obs_schema v740):
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
      [351:367] enemy_nearest_dist_16dir
      [367:383] enemy_density_near_16dir
      [383:399] enemy_density_mid_16dir
      [399:447] gem_density_all_16dir  (3 × 16)
      [447:495] red_green_gem_density_16dir (3 × 16)
      [495:687] projectiles (32 × 6)
      [687:711] floor_pickups (8 × 3)
      [711:720] special_pickups (3 × 3)
      [720:740] destructibles (10 × 2)

    Phase 1 の obs 追加後 (weapon_attack_range_norm + weapon_is_directional +
    weapon_category_onehot) は obs_schema が変わるため、
    オフセットを全て更新すること (+ 54 dims → total 794 dims)。
    """
    shaped = 0.0

    # ============================================================
    # 0. Parse observation segments
    # ============================================================
    player_vel = obs[2:4]
    wall_rays = obs[4:12]
    player_hp_norm = obs[12]
    shield_active = obs[13]
    shield_timer = obs[14]

    weapon_slots_raw = obs[23:41].reshape(6, 3)

    xp_progress = obs[55]

    red_gem_rel = obs[58:78].reshape(10, 2)
    green_gem_rel = obs[78:102].reshape(12, 2)
    blue_gem_rel = obs[102:126].reshape(12, 2)

    # 最近傍敵の相対位置 (dx/DN, dy/DN) — 方向計算に使用
    enemy_rel = obs[127:191].reshape(32, 2)

    enemy_nearest_16 = obs[351:367]
    enemy_density_near_16 = obs[367:383]
    enemy_density_mid_16 = obs[383:399]

    gem_density_all = obs[399:447].reshape(3, 16)
    gem_nearest_16 = gem_density_all[0]
    gem_density_near_16 = gem_density_all[1]
    gem_density_mid_16 = gem_density_all[2]

    # ============================================================
    # 1. 武器分類 (v09: 6グループ + defensive に細分化)
    #    UE5 SurvivorsGameConstants::GetWeaponCategory() の定義と一致させること
    # ============================================================
    # 自律範囲 (敵に近いほど有効)
    GARLIC_AUTO_IDS = {1, 16}          # Garlic, SoulEater
    # 軌道型 (近距離を維持しながら自動攻撃)
    ORBITAL_IDS = {7, 22}              # KingBible, UnholyVespers
    # ライン/扇形メレー (敵方向への移動が重要)
    MELEE_LINE_IDS = {2, 17}           # Whip, BloodyTear
    # ターゲット追尾遠距離 (中距離で自動照準)
    RANGED_TARGETED_IDS = {3, 8, 11, 18, 26}   # MagicWand, FireWand, LightningRing, HolyWand, ThunderLoop
    # 方向固定遠距離 (移動方向前方に敵が並ぶと有効)
    RANGED_DIRECTIONAL_IDS = {4, 5, 6, 13, 14, 19, 20, 21, 28}  # Knife, Axe, Cross, Peachone, EbonyWings, ThousandEdge, DeathSpiral, HeavenSword, Vandalier
    # エリアドロップ (敵密度の高い場所に落とす)
    AREA_DROP_IDS = {9, 10, 12, 23, 24, 25, 27}  # SantaWater, Runetracer, Pentagram, Hellfire, LaBorra, NoFuture, GorgeousMoon
    # 防御/特殊
    DEFENSIVE_IDS = {15}               # Laurel

    garlic_auto_count = 0
    orbital_count = 0
    melee_line_count = 0
    ranged_t_count = 0
    directional_count = 0
    area_drop_count = 0
    defensive_count = 0
    equipped_count = 0

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
        elif wtype_id in MELEE_LINE_IDS:
            melee_line_count += 1
        elif wtype_id in RANGED_TARGETED_IDS:
            ranged_t_count += 1
        elif wtype_id in RANGED_DIRECTIONAL_IDS:
            directional_count += 1
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
    area_drop_ratio    = area_drop_count / equipped_count
    defensive_ratio    = defensive_count / equipped_count

    # 近接系の合計 (接近ボーナスの適用判断に使用)
    near_melee_ratio = garlic_auto_ratio + orbital_ratio + melee_line_ratio

    # density_pen_mult: 武器グループ別に調整
    # Garlic/Orbital は「敵の近く」が最適なので密度ペナルティを大幅軽減
    density_pen_mult = (
        garlic_auto_ratio  * 0.2 +
        orbital_ratio      * 0.3 +
        melee_line_ratio   * 0.5 +
        ranged_t_ratio     * 0.9 +
        directional_ratio  * 1.0 +
        area_drop_ratio    * 0.6 +
        defensive_ratio    * 0.8
    )
    if density_pen_mult < 1e-6:
        density_pen_mult = 1.0

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
    # 4. 停止ペナルティ
    # ============================================================
    if not is_moving:
        min_enemy_nearest = float(np.min(enemy_nearest_16)) if enemy_nearest_16.size > 0 else 1.0
        stationary_pen = -0.01 - 0.02 * max(0.0, 1.0 - min_enemy_nearest * 5.0)
        shaped += stationary_pen

    # ============================================================
    # 5. 方向品質ボーナス (安全 + Gem リッチな方向)
    # ============================================================
    if is_moving and move_dir >= 0:
        d = move_dir

        enemy_nearest_d = float(enemy_nearest_16[d])
        enemy_near_dens_d = float(enemy_density_near_16[d])
        enemy_mid_dens_d = float(enemy_density_mid_16[d])

        gem_nearest_d = float(gem_nearest_16[d])
        gem_near_dens_d = float(gem_density_near_16[d])
        gem_mid_dens_d = float(gem_density_mid_16[d])

        # 5a. 敵密度ペナルティ
        near_density_pen = -0.015 * density_pen_mult * min(enemy_near_dens_d, 4.0)
        mid_density_pen  = -0.005 * density_pen_mult * min(enemy_mid_dens_d, 6.0)
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

    # ============================================================
    # 6. [v09 新規] Garlic/Orbital/MeleeLine: 敵接近ボーナス
    #    PR#202 で全敵が Player より遅くなったため、自ら接近しないと DPS が出ない
    # ============================================================
    if near_melee_ratio > 0.2 and is_moving and nearest_enemy_dir is not None:
        approach_dot = float(np.dot(vel_norm, nearest_enemy_dir))
        # HP が低い時は安全優先のため接近ボーナスを抑制
        hp_gate_approach = min(player_hp_norm / 0.5, 1.0)
        approach_bonus = 0.025 * near_melee_ratio * max(0.0, approach_dot) * hp_gate_approach
        shaped += approach_bonus

    # ============================================================
    # 7. [v09 新規] RangedDirectional: 前方敵密度ボーナス
    #    Knife/Axe/Cross は移動方向前方に敵が並ぶほど命中しやすい
    # ============================================================
    if directional_ratio > 0.2 and is_moving and move_dir >= 0:
        front_density = float(enemy_density_near_16[move_dir])
        front_bonus = 0.015 * directional_ratio * min(front_density, 4.0)
        shaped += front_bonus

    # ============================================================
    # 8. [v09 新規] AreaDrop: 最高敵密度方向への移動ボーナス
    #    SantaWater/Runetracer は最も敵が密集している方向で最大効果
    # ============================================================
    if area_drop_ratio > 0.2 and is_moving and move_dir >= 0:
        best_density_dir = int(np.argmax(enemy_density_near_16))
        best_density_val = float(enemy_density_near_16[best_density_dir])
        # 移動方向が最高密度方向と ±1 以内で一致
        dir_diff = abs(move_dir - best_density_dir)
        dir_match = (dir_diff <= 1 or dir_diff >= 15)
        if dir_match and best_density_val > 0.1:
            area_bonus = 0.01 * area_drop_ratio * min(best_density_val, 4.0)
            shaped += area_bonus

    # ============================================================
    # 9. 最近傍 Gem への接近ボーナス (ステップ単位の前進)
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
        gem_approach_reward = float(np.clip(gem_approach_delta * 0.02, -0.02, 0.02))
        min_enemy_nearest_val = float(np.min(enemy_nearest_16))
        if player_hp_norm < 0.3 and min_enemy_nearest_val < 0.1:
            gem_approach_reward *= 0.3
        shaped += gem_approach_reward

    # ============================================================
    # 10. Gem ピックアップ時の小ボーナス
    # ============================================================
    if base_reward >= 1.0:
        gem_count_est = min(base_reward, 5.0)
        shaped += 0.02 * gem_count_est

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
    # 13. RangedTargeted: 適切な戦闘距離維持ボーナス (v08 から継承)
    # ============================================================
    if ranged_t_ratio > 0.3:
        min_enemy_dist_val = float(np.min(enemy_nearest_16))
        if 0.15 <= min_enemy_dist_val <= 0.40:
            dist_deviation = abs(min_enemy_dist_val - 0.275) / 0.125
            combat_range_bonus = 0.01 * ranged_t_ratio * (1.0 - dist_deviation)
            shaped += combat_range_bonus

    # ============================================================
    # 14. 防御武器の特殊処理 (v08 から継承)
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
    # Final clamp
    # ============================================================
    return float(np.clip(shaped, -1.0, 1.0))
