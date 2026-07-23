"""実 UE の /obs_schema の全 segment を release-observable か否かで排他分類し、exact set からの drift を検出する。

AI がゲームから受け取る観測値のうち『画面から実際に見える情報だけ』を使うよう仕分けします。
本来見えないはずの内部情報が混ざっていないかを検査します。
"""

DIRECT_HUD = frozenset({"player_hp", "elapsed_time", "xp_progress", "player_level"})
INVENTORY_DERIVED = frozenset({"shield_active", "revival_remaining_norm", "armor_flat_norm", "regen_per_sec_norm", "passive_effect_summary", "weapon_slots", "passive_slots", "gem_pickup_radius", "weapon_attack_range_norm", "weapon_is_directional", "weapon_category_onehot"})
SCREEN_WORLD_OBSERVED = frozenset({"player_pos", "wall_rays", "red_gem_rel_pos", "green_gem_rel_pos", "blue_gem_rel_pos", "enemy_rel_pos", "enemy_type", "enemy_frozen", "enemy_nearest_dist_16dir", "projectiles", "floor_pickups", "special_pickups", "destructibles"})
TEMPORAL_INFERRED = frozenset({"player_vel", "enemy_vel"})
PROFILE_CONSTANT = frozenset({"stage_id_norm"})
# Simulator-only state that cannot be honestly recovered from pixels.
UNOBSERVABLE = frozenset({"shield_timer_norm", "enemy_count", "enemy_hp", "enemy_density_near_16dir", "enemy_density_mid_16dir", "gem_density_all_16dir", "red_green_gem_density_16dir"})

_CLASS = {"direct_hud": DIRECT_HUD, "inventory_derived": INVENTORY_DERIVED,
          "screen_world_observed": SCREEN_WORLD_OBSERVED, "temporal_inferred": TEMPORAL_INFERRED,
          "profile_constant": PROFILE_CONSTANT, "unobservable": UNOBSERVABLE}
OBS_SEGMENTS = frozenset().union(*_CLASS.values())

def classify_segments():
    assert len(OBS_SEGMENTS) == 38 and sum(map(len, _CLASS.values())) == 38
    return dict(_CLASS)

def validate_exact_schema(schema_segments):
    observed = frozenset(schema_segments)
    if observed != OBS_SEGMENTS:
        raise ValueError(f"/obs_schema drift: missing={sorted(OBS_SEGMENTS-observed)}, unknown={sorted(observed-OBS_SEGMENTS)}")

def validate_release_observable(segments):
    requested = frozenset(segments)
    bad = (requested - OBS_SEGMENTS) | (requested & UNOBSERVABLE)
    if bad: raise ValueError(f"non-observable segments: {sorted(bad)}")
