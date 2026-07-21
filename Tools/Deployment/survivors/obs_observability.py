OBS_SEGMENTS=frozenset({"timer","level","xp","inventory_items","inventory_levels","player_screen_position","visible_enemy_positions","visible_pickups","motion_history","stage_id","character_id","offscreen_entities","hidden_enemy_hp","hidden_cooldowns","global_state_count","global_density"})
_CLASS={"direct_hud":{"timer","level","xp"},"inventory_derived":{"inventory_items","inventory_levels"},"screen_world_observed":{"player_screen_position","visible_enemy_positions","visible_pickups"},"temporal_inferred":{"motion_history"},"profile_constant":{"stage_id","character_id"},"unobservable":{"offscreen_entities","hidden_enemy_hp","hidden_cooldowns","global_state_count","global_density"}}
def classify_segments():return {k:frozenset(v) for k,v in _CLASS.items()}
def validate_release_observable(segments:set[str]):
    unknown=segments-OBS_SEGMENTS
    forbidden=segments&_CLASS["unobservable"]
    if unknown or forbidden:raise ValueError(f"non-observable segments: {sorted(unknown|forbidden)}")
