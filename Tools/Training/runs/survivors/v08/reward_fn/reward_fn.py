import numpy as np

def reward_shaping(obs: np.ndarray, prev_obs: np.ndarray, base_reward: float) -> float:
    """
    Survivors reward shaping v7 iteration 0.
    Guides agent toward safe gem collection with weapon-type-aware enemy density handling.
    HP penalty is applied externally; this function provides small directional shaping only.
    """
    shaped = 0.0

    # ============================================================
    # 0. Parse observation segments (Source of Truth offsets)
    # ============================================================
    player_pos = obs[0:2]          # x,y normalised to [-1,1]
    player_vel = obs[2:4]          # vx,vy / MoveSpeed
    wall_rays = obs[4:12]          # 8 directions, 0=wall close, 1=far
    player_hp_norm = obs[12]       # HP / 100.0
    shield_active = obs[13]
    shield_timer = obs[14]

    # Weapon slots: 6 slots × 3 (type_norm, level_norm, cooldown_norm) at obs[23:41]
    weapon_slots_raw = obs[23:41].reshape(6, 3)

    # Game state
    enemy_count_norm = obs[53]     # / 32
    elapsed_norm = obs[54]
    xp_progress = obs[55]
    player_level_norm = obs[56]

    # Gem relative positions
    red_gem_rel = obs[58:78].reshape(10, 2)
    green_gem_rel = obs[78:102].reshape(12, 2)
    blue_gem_rel = obs[102:126].reshape(12, 2)
    gem_pickup_radius_norm = obs[126]

    # Enemy relative positions (top-32 nearest)
    enemy_rel = obs[127:191].reshape(32, 2)

    # 16-dir density/distance features
    enemy_nearest_16 = obs[351:367]      # 0=danger close, 1=safe far
    enemy_density_near_16 = obs[367:383] # near density per dir
    enemy_density_mid_16 = obs[383:399]  # mid density per dir

    # gem_density_all_16dir: 48 dims = 3 features × 16 dirs
    # Based on the C++ pattern: nearest_dist, near_density, mid_density per dir
    gem_density_all = obs[399:447].reshape(3, 16)
    gem_nearest_16 = gem_density_all[0]    # nearest gem dist per dir (0=close, 1=far)
    gem_density_near_16 = gem_density_all[1]  # near gem density per dir
    gem_density_mid_16 = gem_density_all[2]   # mid gem density per dir

    prev_player_hp_norm = prev_obs[12]

    # ============================================================
    # 1. Weapon category classification
    # ============================================================
    MELEE_IDS = {1, 2, 7, 16, 17, 22}
    RANGED_IDS = {3, 4, 5, 6, 8, 13, 14, 18, 19, 20, 21, 23, 25, 28}
    AREA_IDS = {9, 10, 11, 12, 24, 26, 27}
    DEFENSIVE_IDS = {15}

    melee_count = 0
    ranged_count = 0
    area_count = 0
    defensive_count = 0
    equipped_count = 0

    for slot_i in range(6):
        tn = weapon_slots_raw[slot_i, 0]
        if tn < 1e-4:
            continue  # empty slot
        equipped_count += 1
        # Recover weapon type ID: type_norm = id / 64
        wtype_id = int(round(tn * 64.0))
        if wtype_id in MELEE_IDS:
            melee_count += 1
        elif wtype_id in RANGED_IDS:
            ranged_count += 1
        elif wtype_id in AREA_IDS:
            area_count += 1
        elif wtype_id in DEFENSIVE_IDS:
            defensive_count += 1

    if equipped_count == 0:
        equipped_count = 1  # avoid division by zero

    melee_ratio = melee_count / equipped_count
    ranged_ratio = ranged_count / equipped_count
    area_ratio = area_count / equipped_count
    defensive_ratio = defensive_count / equipped_count

    # Weighted penalty multiplier for enemy density
    # melee: 0.5x (tolerate closer enemies), ranged: 1.0x (v08: 1.5->1.0 to allow gem approach), area: 1.0x, defensive: 0.8x
    density_pen_mult = (melee_ratio * 0.5 +
                        ranged_ratio * 1.0 +   # 1.5 → 1.0 (v08: Gem方向移動ペナルティを緩和)
                        area_ratio * 1.0 +
                        defensive_ratio * 0.8)
    # Default if no weapons classified (fallback)
    if density_pen_mult < 1e-6:
        density_pen_mult = 1.0

    # ============================================================
    # 2. Determine movement direction from velocity
    # ============================================================
    vel_mag = np.sqrt(player_vel[0]**2 + player_vel[1]**2)
    is_moving = vel_mag > 0.05

    if is_moving:
        # Convert velocity to direction bin using same formula as C++
        # Dir = floor(((atan2(y, x) + pi) / (2*pi)) * 16)
        angle_rad = np.arctan2(player_vel[1], player_vel[0])
        angle_01 = (angle_rad + np.pi) / (2.0 * np.pi)
        move_dir = int(np.clip(np.floor(angle_01 * 16), 0, 15))
    else:
        move_dir = -1  # stationary

    # ============================================================
    # 3. Stationary penalty (discourage standing still)
    # ============================================================
    if not is_moving:
        # Mild penalty for being stationary, stronger if enemies are near
        min_enemy_nearest = np.min(enemy_nearest_16) if enemy_nearest_16.size > 0 else 1.0
        # If closest enemy is very near (min_enemy_nearest < 0.2), stronger penalty
        stationary_pen = -0.01 - 0.02 * max(0.0, 1.0 - min_enemy_nearest * 5.0)
        shaped += stationary_pen

    # ============================================================
    # 4. Directional quality: safe + gem-rich direction bonus
    # ============================================================
    if is_moving and move_dir >= 0:
        d = move_dir

        # Safety of chosen direction
        enemy_nearest_d = enemy_nearest_16[d]           # 1=safe, 0=danger
        enemy_near_dens_d = enemy_density_near_16[d]
        enemy_mid_dens_d = enemy_density_mid_16[d]

        # Gem richness of chosen direction
        gem_nearest_d = gem_nearest_16[d]               # 0=gem close, 1=far
        gem_near_dens_d = gem_density_near_16[d]
        gem_mid_dens_d = gem_density_mid_16[d]

        # --- 4a. Enemy density penalty for chosen direction ---
        # Higher density = worse; scale by weapon type
        near_density_pen = -0.015 * density_pen_mult * min(enemy_near_dens_d, 4.0)
        mid_density_pen = -0.005 * density_pen_mult * min(enemy_mid_dens_d, 6.0)

        # Very close enemy in chosen direction: extra penalty
        danger_close_pen = 0.0
        if enemy_nearest_d < 0.15:
            danger_close_pen = -0.02 * density_pen_mult * (1.0 - enemy_nearest_d / 0.15)

        shaped += near_density_pen + mid_density_pen + danger_close_pen

        # --- 4b. Gem approach bonus for chosen direction ---
        # Bonus for moving toward gem-rich directions, modulated by safety
        gem_richness = gem_near_dens_d + 0.5 * gem_mid_dens_d
        # Safety factor: reduce gem bonus if enemy density is high in that direction
        safety_factor = max(0.0, 1.0 - 0.3 * enemy_near_dens_d - 0.1 * enemy_mid_dens_d)
        safety_factor = min(safety_factor, 1.0)
        # 遠距離武器時は Gem ボーナスが完全に消えないよう下限を設ける (v08)
        gem_safety_floor = 0.3 * ranged_ratio
        safety_factor = max(gem_safety_floor, safety_factor)

        # HP-dependent modulation: at low HP, reduce gem-chasing aggressiveness
        hp_gate = min(player_hp_norm / 0.4, 1.0)  # ramps from 0 at 0 HP to 1 at 40% HP

        gem_dir_bonus = 0.02 * min(gem_richness, 3.0) * safety_factor * hp_gate
        shaped += gem_dir_bonus

        # --- 4c. Bonus for moving toward nearest gem (any direction) ---
        gem_close_bonus = 0.01 * max(0.0, 1.0 - gem_nearest_d) * safety_factor * hp_gate
        shaped += gem_close_bonus

    # ============================================================
    # 5. Nearest gem distance reduction bonus (step-by-step progress)
    # ============================================================
    # Use the closest gem from any color
    def closest_gem_dist(gem_arr):
        """Return distance to closest non-zero gem, or inf."""
        dists = np.sqrt(gem_arr[:, 0]**2 + gem_arr[:, 1]**2)
        valid = dists > 1e-6
        if np.any(valid):
            return np.min(dists[valid])
        return float('inf')

    curr_min_gem = min(closest_gem_dist(red_gem_rel),
                       closest_gem_dist(green_gem_rel),
                       closest_gem_dist(blue_gem_rel))

    prev_red = prev_obs[58:78].reshape(10, 2)
    prev_green = prev_obs[78:102].reshape(12, 2)
    prev_blue = prev_obs[102:126].reshape(12, 2)
    prev_min_gem = min(closest_gem_dist(prev_red),
                       closest_gem_dist(prev_green),
                       closest_gem_dist(prev_blue))

    if curr_min_gem < 50.0 and prev_min_gem < 50.0:
        # Reward for getting closer to nearest gem
        gem_approach_delta = prev_min_gem - curr_min_gem
        # Scale: normalize roughly by move speed (80 units/step * dt)
        # Positive delta = getting closer = good
        gem_approach_reward = np.clip(gem_approach_delta * 0.02, -0.02, 0.02)

        # Reduce if HP is low and enemy is dangerously close
        min_enemy_nearest_val = np.min(enemy_nearest_16)
        if player_hp_norm < 0.3 and min_enemy_nearest_val < 0.1:
            gem_approach_reward *= 0.3  # heavily dampen

        shaped += gem_approach_reward

    # ============================================================
    # 6. Gem pickup celebration (small bonus for successful collection)
    # ============================================================
    if base_reward >= 1.0:
        # Gem was picked up this step; small bonus to reinforce approach behavior
        # Scale by how many gems (base_reward can be > 1 if multiple pickups)
        gem_count_est = min(base_reward, 5.0)  # cap at 5
        shaped += 0.02 * gem_count_est

    # ============================================================
    # 7. Wall avoidance shaping
    # ============================================================
    # wall_rays: 8 directions, 0 = wall very close, 1 = far from wall
    min_wall = np.min(wall_rays)
    if min_wall < 0.15:
        # Near a wall; penalize if also moving toward the wall
        wall_pen = -0.015 * (1.0 - min_wall / 0.15)
        shaped += wall_pen

        # Additional: if wall + enemies nearby = dangerous, stronger signal
        if np.min(enemy_nearest_16) < 0.2:
            shaped += -0.01

    # ============================================================
    # 8. Low HP caution: boost avoidance behavior
    # ============================================================
    if player_hp_norm < 0.25:
        # At critically low HP, penalize being near enemies more
        closest_enemy_dist_val = np.min(enemy_nearest_16)
        if closest_enemy_dist_val < 0.15:
            low_hp_danger = -0.03 * (1.0 - closest_enemy_dist_val / 0.15) * (1.0 - player_hp_norm / 0.25)
            shaped += low_hp_danger

    # ============================================================
    # 8b. 遠距離武器: 適切な戦闘距離維持ボーナス (v08)
    # ============================================================
    # 最近傍敵が適切な距離（0.15〜0.40）にある場合に小ボーナス
    # enemy_nearest_16: 0=danger close, 1=safe far
    # 適切距離の中心 ≈ 0.275, 幅 ≈ 0.125
    if ranged_ratio > 0.3:
        # 全方向最小値（静止時も含む）を使用。最大ボーナス 0.01 と小さいため実害は限定的
        min_enemy_dist_val = np.min(enemy_nearest_16)
        if 0.15 <= min_enemy_dist_val <= 0.40:
            # 距離が中心(0.275)に近いほど大きいボーナス
            # 0.15〜0.40 = 約 360〜960cm（EnemyNearestDistanceMax=2400cm でノーマライズ）
            dist_deviation = abs(min_enemy_dist_val - 0.275) / 0.125
            combat_range_bonus = 0.01 * ranged_ratio * (1.0 - dist_deviation)
            shaped += combat_range_bonus

    # ============================================================
    # 9. Defensive weapon special handling
    # ============================================================
    if defensive_ratio > 0.1:
        if shield_active > 0.5:
            # Shield is active: encourage gem collection more aggressively
            shaped += 0.01
        elif shield_timer > 0.5:
            # Shield will be available soon: maintain safe positioning
            pass
        else:
            # No shield, no timer: be cautious
            if np.min(enemy_nearest_16) < 0.2:
                shaped += -0.01

    # ============================================================
    # 10. XP progress bonus (small reward for leveling progression)
    # ============================================================
    prev_xp = prev_obs[55]
    curr_xp = obs[55]
    # Level up detected: xp drops to near 0 while prev was high
    if prev_xp > 0.8 and curr_xp < 0.2:
        shaped += 0.02  # small level-up bonus

    # ============================================================
    # Final clamp
    # ============================================================
    return float(np.clip(shaped, -1.0, 1.0))