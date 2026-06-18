import numpy as np

def reward_shaping(obs: np.ndarray, prev_obs: np.ndarray, base_reward: float) -> float:
    """
    Survivors reward shaping v10.
    v09 からの変更:
      - バージョン番号の更新
      - PR#216: Orbital クールダウン中の敵接近抑制を Section 6 に統合
        （Section 5c の移動方向ペナルティに加え、接近ボーナス自体を cooldown_gate で抑制）
      - reward_analysis_logger.py と組み合わせて使用
        (shaped_reward 分布・obs 相関・エピソード時系列が自動ログされる)

    v10 obs_schema v794 対応変更:
      - Section 6b: Orbital クールダウン近接ペナルティを -0.03 → -0.02 に軽減
        （5c との二重がけによる過剰抑制を解消）
      - Section 12b 追加: 低HP時の逃走方向ボーナス
        （早期 terminated エピソードの削減を狙う）
      - Section 13 刷新: 全武器カテゴリ対応の適正距離維持ボーナス
        （obs[740:746] weapon_attack_range_norm を使用、固定 [0.15,0.40] を廃止）

    obs レイアウト (794 dims, obs_schema v794):
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
      [495:687] projectiles (32 × 6)
      [687:711] floor_pickups (8 × 3)
      [711:720] special_pickups (3 × 3)
      [720:740] destructibles (10 × 2)
      [740:746] weapon_attack_range_norm  (GetWeaponEffectiveRange() × 6スロット)
      [746:752] weapon_is_directional     (6スロット)
      [752:794] weapon_category_onehot    (6スロット × 7カテゴリ)
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

    # v794 追加次元
    weapon_range_norms = obs[740:746]  # GetWeaponEffectiveRange() 正規化値

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
    # 折り返し系 (往路の折り返し距離が間合い。RANGED_DIRECTIONAL のサブセット)
    FOLDING_IDS = {5, 6, 20, 21}       # Axe=5, Cross=6, DeathSpiral=20, HeavenSword=21
    # エリアドロップ (敵密度の高い場所に落とす)
    AREA_DROP_IDS = {9, 10, 12, 23, 24, 25, 27}  # SantaWater, Runetracer, Pentagram, Hellfire, LaBorra, NoFuture, GorgeousMoon
    # 防御/特殊
    DEFENSIVE_IDS = {15}               # Laurel

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

    # 近接系の合計 (接近ボーナスの適用判断に使用)
    near_melee_ratio = garlic_auto_ratio + orbital_ratio + melee_line_ratio

    # density_pen_mult: 武器グループ別に調整
    # Garlic/Orbital は「敵の近く」が最適なので密度ペナルティを大幅軽減
    # 折り返し系 (Cross/Axe): 往路の折り返し距離 (~75-100u) が間合いのため軽減
    #   Knife 等の純粋直線系とは分けて 0.4 を適用する
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

        # 5c. [v09 修正] Orbital クールダウン中の敵密度方向ペナルティ
        #    KingBible 等は充填中に攻撃判定ゼロ（聖書が消えている）→ 高密度方向への接近を抑制
        #    Orbital に限定: FireWand/Axe 等はクールダウン中も敵方向移動が合理的なため対象外
        if orbital_ratio > 0.1 and orbital_cooldown_max > 0.3:
            cooldown_density_pen = -0.01 * orbital_ratio * orbital_cooldown_max * min(enemy_near_dens_d, 4.0)
            shaped += cooldown_density_pen

    # ============================================================
    # 6. [v09 新規] Garlic/Orbital/MeleeLine: 敵接近ボーナス
    #    PR#202 で全敵が Player より遅くなったため、自ら接近しないと DPS が出ない
    # ============================================================
    if near_melee_ratio > 0.2 and is_moving and nearest_enemy_dir is not None:
        approach_dot = float(np.dot(vel_norm, nearest_enemy_dir))
        hp_gate_approach = min(player_hp_norm / 0.5, 1.0)
        # Orbital クールダウン中は接近ボーナスを抑制（聖書が消えている間に近づくと無防備）
        orbital_cooldown_gate = 1.0 - 0.8 * orbital_ratio * orbital_cooldown_max
        approach_bonus = 0.025 * near_melee_ratio * max(0.0, approach_dot) * hp_gate_approach * orbital_cooldown_gate
        shaped += approach_bonus

    # ============================================================
    # 6b. [v09 修正 / v10 軽減] Orbital クールダウン中の敵近接ペナルティ
    #    KingBible 充填中（聖書なし）に敵が近い状態そのものを抑制
    #    5c が「移動方向」への抑制であるのに対し、6b は「現在位置」への抑制
    #    v10: 5c との二重がけによる過剰抑制を解消するため -0.03 → -0.02 に軽減
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
    # 12b. [v10 新規] 低HP時の逃走方向ボーナス
    #    低HPで最近傍敵から遠ざかる方向への移動を評価する
    #    Section 12 のペナルティと対になる正報酬: 逃げると報われる
    # ============================================================
    if player_hp_norm < 0.3 and is_moving and nearest_enemy_dir is not None:
        escape_dot = float(np.dot(vel_norm, -nearest_enemy_dir))
        if escape_dot > 0.0:
            escape_bonus = 0.015 * escape_dot * (1.0 - player_hp_norm / 0.3)
            shaped += escape_bonus

    # ============================================================
    # 13. [v10 刷新] 全武器適正距離維持ボーナス
    #    obs[740:746] = weapon_attack_range_norm (GetWeaponEffectiveRange() の値)
    #    enemy_nearest_16 の正規化基準: EnemyNearestDistanceMax = 2400u
    #
    #    weapon_attack_range_norm → 適正距離帯のマッピング:
    #      ≤ 0.05  Garlic/SoulEater:    密着 (0〜60u)    → [0.000, 0.025]
    #      ≤ 0.15  KingBible/Whip:      近接 (0〜90u)    → [0.000, 0.040]
    #      ≤ 0.35  SantaWater/LRing:    中近  (150〜480u) → [0.063, 0.200]
    #      ≤ 0.55  MagicWand/FireWand:  中距離(200〜650u) → [0.083, 0.280]
    #      ≤ 0.79  Runetracer 等:       中遠  (250〜900u) → [0.104, 0.375]
    #      ≥ 0.80  Knife/Pentagram: 超遠距離 → Section 7 が主担当のためスキップ
    #
    #    折り返し系武器 (Cross/HeavenSword/Axe/DeathSpiral) は weapon_attack_range_norm
    #    の値に関わらず、往路の折り返し距離を間合いとして扱う:
    #      Cross / HeavenSword: CrossReverseDistance=75u  → [0.000, 0.035]
    #      Axe  / DeathSpiral:  ArcHeight=120u,apex≈60u  → [0.000, 0.045]
    #    折り返し後のヒットは副産物扱いのため報酬なし。
    #
    #    ボーナス上限: スロット毎 0.010、全スロット平均で最大 0.010
    # ============================================================
    # 折り返し系武器 ID (RANGED_DIRECTIONAL_IDS のサブセット)
    CROSS_BOOMERANG_IDS = {6, 21}   # Cross=6, HeavenSword=21
    AXE_ARC_IDS = {5, 20}           # Axe=5, DeathSpiral=20

    min_enemy_dist_val = float(np.min(enemy_nearest_16))
    range_bonus_sum = 0.0
    range_slot_count = 0

    for slot_i in range(6):
        tn = weapon_slots_raw[slot_i, 0]
        if tn < 1e-4:
            continue
        wtype_id_s13 = int(round(tn * 64.0))
        r = float(weapon_range_norms[slot_i])

        # 適正距離帯 (lo, hi) と近接系フラグを決定
        is_near_type = False
        if wtype_id_s13 in CROSS_BOOMERANG_IDS:
            # Cross/HeavenSword: 往路の折り返し距離 75u が実質の間合い
            lo, hi = 0.000, 0.035   # 75u / 2400u ≈ 0.031 に余裕を持たせた 0.035
            is_near_type = True
        elif wtype_id_s13 in AXE_ARC_IDS:
            # Axe/DeathSpiral: 弧の頂点まで ~60u、着地まで ~100u が実質の間合い
            lo, hi = 0.000, 0.045   # 100u / 2400u ≈ 0.042 に余裕を持たせた 0.045
            is_near_type = True
        elif r >= 0.80:
            # Knife/Pentagram 等の超遠距離はスキップ (Section 7 が担当)
            continue
        elif r <= 0.05:        # Garlic / SoulEater: 密着
            lo, hi = 0.000, 0.025
            is_near_type = True
        elif r <= 0.15:        # KingBible / Whip: 近接
            lo, hi = 0.000, 0.040
            is_near_type = True
        elif r <= 0.35:        # SantaWater / LightningRing: 中近距離
            lo, hi = 0.063, 0.200
        elif r <= 0.55:        # MagicWand / FireWand / Peachone: 中距離
            lo, hi = 0.083, 0.280
        else:                  # Runetracer 等 (0.56〜0.79): 中遠距離
            lo, hi = 0.104, 0.375

        # 距離スコア (0〜1) を計算
        if is_near_type:
            # 近接/折り返し系: バンド内は満点、超えると急減 (遠くなるほど不利)
            if min_enemy_dist_val <= hi:
                dist_score = 1.0
            else:
                overshoot = (min_enemy_dist_val - hi) / hi
                dist_score = max(0.0, 1.0 - overshoot * 3.0)
        else:
            # 中距離系: バンド中央付近が満点、外れると逓減
            if min_enemy_dist_val < lo:
                # バンドより近すぎ (敵に密着しすぎ)
                overshoot = (lo - min_enemy_dist_val) / lo if lo > 0.0 else 0.0
                dist_score = max(0.0, 1.0 - overshoot * 2.0)
            elif min_enemy_dist_val <= hi:
                # バンド内: 中央からの逸脱量でスコア計算
                mid = (lo + hi) / 2.0
                half_width = (hi - lo) / 2.0
                dist_score = max(0.0, 1.0 - abs(min_enemy_dist_val - mid) / half_width)
            else:
                # バンドより遠すぎ
                overshoot = (min_enemy_dist_val - hi) / hi
                dist_score = max(0.0, 1.0 - overshoot * 2.0)

        range_bonus_sum += 0.010 * dist_score
        range_slot_count += 1

    if range_slot_count > 0:
        shaped += range_bonus_sum / range_slot_count

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
