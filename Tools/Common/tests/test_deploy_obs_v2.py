"""DeployObs v2 schema と共有特徴量ビルダーの契約を検証する。

segment 表・v1 不変・座標系・可視規則・方向特徴・武器エフェクト・スロット割り当て・
残り時間・確定した不在と不明の区別、golden fixture との一致を小さな入力で確かめます。
"""

from __future__ import annotations

import json
import math
from importlib.resources import files
from pathlib import Path

import numpy as np
import pytest
import yaml

from reinbalance_survivors_contracts.deploy_obs import (
    DEPLOY_OBS_SCHEMA_VERSION,
    DEPLOY_OBS_V1_SEGMENTS,
    DEPLOY_OBS_V2_SCHEMA_VERSION,
    DeployObsSchema,
)
from reinbalance_survivors_contracts.deploy_obs_v2_features import (
    HudSlot,
    TrackPx,
    build_deploy_obs_v2,
    directional_bin,
    effect_duration_s,
    load_deploy_obs_v2_feature_params,
)
from reinbalance_survivors_contracts.ui_intent import ContractValidationError

V2 = DeployObsSchema.default_v2()
GOLDEN = Path(__file__).parent / "fixtures" / "deploy_obs_v2_golden_v1.json"
V1_HASH = "0f758972a905e49d19a0b8ba0dbec6bdfe0ff6ae405b5b6a76f270db496a327c"
SWO, HUD, TMP = "screen_world_observed", "hud_inventory", "temporal_inferred"
PLAN_V2_ADDED = (
    ("enemy_nearest_dist_16dir", 16, SWO), ("enemy_density_near_16dir", 16, SWO), ("enemy_density_mid_16dir", 16, SWO),
    ("gem_nearest_dist_16dir", 16, SWO), ("gem_density_near_16dir", 16, SWO), ("gem_density_mid_16dir", 16, SWO),
    ("rare_gem_nearest_dist_16dir", 16, SWO), ("rare_gem_density_near_16dir", 16, SWO), ("rare_gem_density_mid_16dir", 16, SWO),
    ("weapon_slot_ids", 6, HUD), ("weapon_slot_levels", 6, HUD), ("passive_slot_ids", 6, HUD), ("passive_slot_levels", 6, HUD),
    ("weapon_aura_radius", 1, SWO), ("weapon_aura_slot", 1, HUD),
    ("weapon_orbit_radius", 1, SWO), ("weapon_orbit_slot", 1, HUD), ("weapon_orbit_ttl", 1, TMP),
    ("weapon_zone_geometry", 12, SWO), ("weapon_zone_slot", 4, HUD), ("weapon_zone_ttl", 4, TMP),
    ("weapon_projectile_density_16dir", 16, SWO),
)


def _hud(weapons=None, passives=None, *, omit=()):
    """武器・パッシブ各6スロットの HudSlot 列を作る。

    weapons / passives は {番号: (種類名, レベル)}、それ以外は空スロット。omit の (kind, index) は渡さない。
    """
    slots = []
    for kind, table in (("weapon", weapons or {}), ("passive", passives or {})):
        for index in range(6):
            if (kind, index) in omit:
                continue
            name, level = table.get(index, (None, None))
            slots.append(HudSlot(kind, index, name, level))
    return slots


def _track(cls, dx_px, dy_px, *, radius=10.0, tid=1, first_seen=0.0, occluded=False, player=(960.0, 540.0)):
    """プレイヤーからの px オフセットで TrackPx を作る。"""
    return TrackPx(cls, player[0] + dx_px, player[1] + dy_px, radius, tid, first_seen, occluded)


def _build(tracks=(), hud="empty", **kw):
    """既定値（1920x1080・中央プレイヤー・有効な world）でビルダーを呼ぶ。"""
    args = dict(viewport_wh=(1920, 1080), player_px=(960.0, 540.0), tracks=list(tracks),
                hud_slots=_hud() if hud == "empty" else hud, now_s=10.0, duration_mult=1.0, world_valid=True)
    args.update(kw)
    return build_deploy_obs_v2(**args)


def _seg(obs, name):
    """segment の (values, validity, age) を返す。"""
    offset, size = V2.layout[name]
    span = slice(offset, offset + size)
    return obs.values[span], obs.validity[span], obs.age[span]


def _assert_missing_rule(obs):
    """I7: validity 0 なら neutral・age 1、validity 1 なら age 0、NaN 無し。"""
    for field in V2.fields:
        values, validity, age = _seg(obs, field.name)
        assert np.all(np.isfinite(values))
        missing = validity == 0
        assert np.all(values[missing] == np.float32(field.neutral)), field.name
        assert np.all(age[missing] == 1) and np.all(age[~missing] == 0), field.name


# ---- schema（M1・M2）----

def test_v2_segment_table_matches_plan_and_extends_v1_unchanged():
    """v2 は v1 の10 segment を同設定で先頭に置き、plan 表の順で追加する。"""
    v1 = DeployObsSchema.default_v1()
    assert V2.schema_version == DEPLOY_OBS_V2_SCHEMA_VERSION == "deploy_obs.v2"
    assert V2.fields[:10] == v1.fields
    assert tuple((f.name, f.size, f.source_class) for f in V2.fields[10:]) == PLAN_V2_ADDED
    assert v1.dim == 13 and V2.dim == 222
    assert v1.schema_hash == V1_HASH and v1.schema_version == DEPLOY_OBS_SCHEMA_VERSION
    assert tuple(f.name for f in v1.fields) == DEPLOY_OBS_V1_SEGMENTS
    assert V2.schema_hash != v1.schema_hash
    assert DeployObsSchema(v1.fields).schema_version == DEPLOY_OBS_SCHEMA_VERSION


def test_segment_order_is_validated_per_schema_version():
    """各 version は自分の segment 列との完全一致だけを受け付ける。"""
    v1 = DeployObsSchema.default_v1()
    assert DeployObsSchema.from_wire(V2.to_wire()) == V2
    assert DeployObsSchema.from_wire(v1.to_wire()) == v1
    with pytest.raises(ContractValidationError):
        DeployObsSchema(V2.fields)
    with pytest.raises(ContractValidationError):
        DeployObsSchema(v1.fields, DEPLOY_OBS_V2_SCHEMA_VERSION)
    swapped = list(V2.fields)
    swapped[10], swapped[11] = swapped[11], swapped[10]
    with pytest.raises(ContractValidationError):
        DeployObsSchema(tuple(swapped), DEPLOY_OBS_V2_SCHEMA_VERSION)
    for version in ("deploy_obs.v3", ["deploy_obs.v2"]):
        wire = V2.to_wire()
        wire["schema_version"] = version
        with pytest.raises(ContractValidationError):
            DeployObsSchema.from_wire(wire)


def test_packaged_v2_yaml_matches_default_v2():
    """package-data の deploy_obs_v2.yaml が default_v2() と同内容。"""
    text = files("reinbalance_survivors_contracts").joinpath("schemas/deploy_obs_v2.yaml").read_text(encoding="utf-8")
    assert DeployObsSchema.from_wire(yaml.safe_load(text)) == V2


# ---- 座標系・可視規則（M4・M5）----

@pytest.mark.parametrize("viewport", [(1920, 1080), (1280, 1024), (1000, 2000)])
def test_isotropic_half_width_normalisation_on_any_aspect_ratio(viewport):
    """縦横同じ縮尺（W/2）で正規化し、縦横比が違っても斜め45度は同じ大きさになる。"""
    w, h = viewport
    player = (w / 2 + w / 4, h / 2)
    step = w / 20  # = 0.1 × 半幅
    obs = _build([_track("enemy_normal", step, step, player=player)], viewport_wh=viewport, player_px=player)
    offset, validity, _ = _seg(obs, "nearest_enemy_offset")
    assert offset == pytest.approx([0.1, 0.1], abs=1e-6) and np.all(validity == 1)
    assert _seg(obs, "player_screen_pos")[0] == pytest.approx([0.5, 0.0], abs=1e-6)
    _assert_missing_rule(obs)


def test_positions_are_clipped_to_unit_range():
    """プレイヤーが画面端にいて画面反対側の敵が半幅より遠いとき [-1,1] に clip する。"""
    obs = _build([TrackPx("enemy_normal", 1900.0, 1070.0, 5.0, 1, 0.0, False)], player_px=(10.0, 10.0))
    assert _seg(obs, "nearest_enemy_offset")[0] == pytest.approx([1.0, 1.0])  # 1890/960・1060/960 を clip
    assert _seg(obs, "player_screen_pos")[0] == pytest.approx([-950 / 960, -530 / 960], abs=1e-6)


def test_visibility_uses_center_only_not_rectangle_or_clipped():
    """中心が画面内なら矩形がはみ出しても可視、中心が外 or 遮蔽なら不可視。"""
    edge = TrackPx("weapon_zone", 1919.0, 540.0, 400.0, 1, 9.0, False)
    outside = TrackPx("weapon_zone", 1921.0, 540.0, 400.0, 2, 9.0, False)
    occluded = TrackPx("weapon_zone", 1000.0, 540.0, 50.0, 3, 9.0, True)
    obs = _build([edge, outside, occluded], hud=_hud({0: ("SantaWater", 1)}))
    geometry = _seg(obs, "weapon_zone_geometry")[0]
    assert geometry[:3] == pytest.approx([959 / 960, 0.0, 400 / 960], abs=1e-6)
    assert np.all(geometry[3:] == 0)
    _assert_missing_rule(obs)


# ---- 方向ビン・距離帯（M6・I6）----

def _deployment_directional_bin(dx, dy, bin_count=8):
    """Tools/Deployment/survivors/screen_space_features.directional_bin の式の転記（import しない）。"""
    angle01 = (math.atan2(float(dy), float(dx)) + math.pi) / (2.0 * math.pi)
    return max(0, min(bin_count - 1, math.floor(angle01 * bin_count)))


@pytest.mark.parametrize("dx,dy,expected", [
    (1.0, 0.0, 8), (0.0, 1.0, 12), (-1.0, 0.0, 15), (-1.0, -0.0, 0), (0.0, -1.0, 4),
    (1.0, 1.0, 10), (-1.0, 1.0, 14), (-1.0, -1.0, 2), (1.0, -1.0, 6), (0.3, -0.05, 7),
])
def test_directional_bin_matches_deployment_and_cpp_formula(dx, dy, expected):
    """軸上・負ゼロ・対角で Deployment と同じ式・同じ16ビンになる。"""
    assert directional_bin(dx, dy) == expected == _deployment_directional_bin(dx, dy, 16)


def test_direction_features_follow_build_dir_density_with_yaml_bands():
    """最寄り距離・近距離・中距離密度を yaml の帯と係数で C++ と同じ式で作る。"""
    band = load_deploy_obs_v2_feature_params()["density"]["enemy"]
    assert (band["near"], band["mid"]) != (600.0, 1400.0)
    near_px, mid_px = 0.1 * 960, 0.4 * 960
    obs = _build([_track("enemy_elite", near_px, 0.0), _track("enemy_boss", -mid_px, 1.0, tid=2), _track("enemy_normal", 0.0, 0.0, tid=3)])
    nearest, near, mid = (_seg(obs, f"enemy_{n}_16dir")[0] for n in ("nearest_dist", "density_near", "density_mid"))
    assert nearest[8] == pytest.approx(0.1) and near[8] == pytest.approx((1 - 0.1 / band["near"]) / band["near_norm"])
    t = (0.4 - band["near"]) / (band["mid"] - band["near"])
    assert nearest[15] == pytest.approx(0.4, abs=1e-5) and mid[15] == pytest.approx((1 - t) / band["mid_norm"], abs=1e-5)
    assert np.sum(nearest < 1) == 2 and np.sum(near > 0) == 1 and np.sum(mid > 0) == 1  # 距離0の敵は除外
    assert _seg(obs, "visible_enemy_count")[0][0] == pytest.approx(3 / 20)


def test_gem_and_rare_gem_use_same_rule_and_class_split():
    """緑・赤は全ジェムとレアジェム両方、青は全ジェムだけに入り、同じ式で作られる。"""
    obs = _build([_track("gem_green", 0.0, 96.0), _track("gem_blue", 0.0, -96.0, tid=2), _track("pickup", 50.0, 0.0, tid=3)])
    for kind in ("nearest_dist", "density_near", "density_mid"):
        gem, rare = _seg(obs, f"gem_{kind}_16dir")[0], _seg(obs, f"rare_gem_{kind}_16dir")[0]
        assert gem[12] == rare[12]
    assert _seg(obs, "gem_nearest_dist_16dir")[0][4] == pytest.approx(0.1)
    assert _seg(obs, "rare_gem_nearest_dist_16dir")[0][4] == 1.0
    assert np.all(_seg(obs, "enemy_nearest_dist_16dir")[0] == 1.0)


# ---- 武器エフェクト（M7）----

def test_zone_geometry_nearest_four_sorted_with_empty_slots():
    """zone は近い順に最大4個、同距離は track id 順、足りない枠は neutral・有効。"""
    zones = [_track("weapon_zone", d, 0.0, radius=20.0, tid=tid, first_seen=9.0) for tid, d in ((5, 300.0), (4, 96.0), (3, -96.0), (2, 500.0), (1, 700.0))]
    obs = _build(zones, hud=_hud({1: ("SantaWater", 1)}))
    geometry, validity, _ = _seg(obs, "weapon_zone_geometry")
    assert geometry[0::3] == pytest.approx([-0.1, 0.1, 300 / 960, 500 / 960], abs=1e-6)
    assert np.all(validity == 1)
    two = _build(zones[1:3], hud=_hud({1: ("SantaWater", 1)}))
    g2, v2, _ = _seg(two, "weapon_zone_geometry")
    assert np.all(g2[6:] == 0) and np.all(v2 == 1)
    none = _build([], hud=_hud({1: ("SantaWater", 1)}))
    assert np.all(_seg(none, "weapon_zone_geometry")[0] == 0) and np.all(_seg(none, "weapon_zone_geometry")[1] == 1)
    for obs_ in (obs, two, none):
        _assert_missing_rule(obs_)


def test_aura_orbit_projectile_effect_values():
    """aura は半径、orbit は距離の中央値と最古の初観測からの残り時間、projectile は方向別密度。"""
    tracks = [
        _track("weapon_aura", 0.0, 0.0, radius=192.0),
        _track("weapon_orbit", 96.0, 0.0, tid=2, first_seen=9.5),
        _track("weapon_orbit", 0.0, 192.0, tid=3, first_seen=9.0),
        _track("weapon_orbit", -384.0, 0.0, tid=4, first_seen=9.8),
        _track("weapon_projectile", 96.0, 0.0, tid=5),
        _track("weapon_projectile", 192.0, 1.0, tid=6),
    ]
    obs = _build(tracks, hud=_hud({0: ("Garlic", 2), 3: ("KingBible", 4), 5: ("Axe", 1)}), duration_mult=1.2)
    assert _seg(obs, "weapon_aura_radius")[0][0] == pytest.approx(0.2)
    assert _seg(obs, "weapon_aura_slot")[0][0] == 0.0 and _seg(obs, "weapon_aura_slot")[1][0] == 1
    assert _seg(obs, "weapon_orbit_radius")[0][0] == pytest.approx(0.2)
    assert _seg(obs, "weapon_orbit_slot")[0][0] == pytest.approx(3 / 5)
    assert _seg(obs, "weapon_orbit_ttl")[0][0] == pytest.approx((3.5 * 1.2 - 1.0) / 8.0)
    density = _seg(obs, "weapon_projectile_density_16dir")[0]
    assert density[8] == pytest.approx(2 / 4) and np.sum(density) == pytest.approx(0.5)
    _assert_missing_rule(obs)


# ---- 確定した不在と不明（M8）----

def test_scan_features_absent_is_valid_neutral_but_world_invalid_is_unknown():
    """画面走査の特徴は見えなければ neutral・有効、world 認識が無効なら validity 0・age 1。"""
    scan = [f.name for f in V2.fields if f.name.endswith("_16dir")] + ["weapon_zone_geometry", "nearest_enemy_offset", "visible_enemy_count", "player_screen_pos"]
    seen = _build([])
    blind = _build([_track("enemy_normal", 96.0, 0.0)], world_valid=False)
    for name in scan:
        assert np.all(_seg(seen, name)[1] == 1), name
        assert np.all(_seg(blind, name)[1] == 0) and np.all(_seg(blind, name)[2] == 1), name
    _assert_missing_rule(seen)
    _assert_missing_rule(blind)


@pytest.mark.parametrize("kind,weapon", [("aura", "Garlic"), ("orbit", "UnholyVespers")])
def test_aura_orbit_radius_confirmed_absent_vs_missed(kind, weapon):
    """武器を持っていなければ neutral・有効、持っているのに見えなければ不明、HUD 不完全でも不明。"""
    name = f"weapon_{kind}_radius"
    absent = _build([], hud=_hud({0: ("Whip", 1)}))
    missed = _build([], hud=_hud({0: (weapon, 1)}))
    partial = _build([], hud=_hud({0: ("Whip", 1)}, omit={("weapon", 4)}))
    assert _seg(absent, name)[1][0] == 1 and _seg(absent, name)[0][0] == 0
    for obs in (missed, partial):
        assert _seg(obs, name)[1][0] == 0 and _seg(obs, name)[2][0] == 1
    for obs in (absent, missed, partial):
        _assert_missing_rule(obs)


def test_slot_ttl_level_unknown_vs_confirmed_absent():
    """スロット・残り時間・レベルは不明なら validity 0、出しうる武器が無いと確定なら neutral・有効。"""
    absent = _build([], hud=_hud({0: ("Knife", 2)}))
    for name in ("weapon_aura_slot", "weapon_orbit_slot", "weapon_orbit_ttl", "weapon_zone_slot", "weapon_zone_ttl"):
        assert np.all(_seg(absent, name)[1] == 1) and np.all(_seg(absent, name)[0] == 0), name
    unknown_weapon = _build([], hud=_hud({0: ("SongOfMana", 2)}))
    for name in ("weapon_aura_slot", "weapon_orbit_slot"):
        assert _seg(unknown_weapon, name)[1][0] == 0, name
    assert _seg(unknown_weapon, "weapon_slot_ids")[0][0] == 1.0
    no_hud = _build([], hud=None, world_valid=False)
    for name in ("weapon_slot_ids", "weapon_slot_levels", "passive_slot_ids", "passive_slot_levels", "weapon_aura_slot", "weapon_orbit_ttl", "weapon_zone_slot", "weapon_zone_ttl", "weapon_category"):
        assert np.all(_seg(no_hud, name)[1] == 0) and np.all(_seg(no_hud, name)[2] == 1), name
    level_unknown = _build([], hud=_hud({2: ("Axe", None)}, {1: ("Spinach", None)}))
    assert _seg(level_unknown, "weapon_slot_levels")[1][2] == 0 and _seg(level_unknown, "weapon_slot_levels")[1][1] == 1
    assert _seg(level_unknown, "passive_slot_levels")[1][1] == 0
    for obs in (absent, unknown_weapon, no_hud, level_unknown):
        _assert_missing_rule(obs)


def test_hud_slot_ids_and_levels_are_normalised_by_vocabulary():
    """種類 id は語彙 index/(len-1)（末尾 unknown）、レベルは最大レベルで割る。"""
    params = load_deploy_obs_v2_feature_params()
    obs = _build([], hud=_hud({0: ("Garlic", 8), 1: ("Vandalier", 1)}, {0: ("TorronasBox", 9)}))
    ids, levels = _seg(obs, "weapon_slot_ids")[0], _seg(obs, "weapon_slot_levels")[0]
    n = len(params["weapon_vocabulary"]) - 1
    assert ids[:3] == pytest.approx([1 / n, 28 / n, 0.0]) and levels[:3] == pytest.approx([1.0, 1 / 8, 0.0])
    assert _seg(obs, "passive_slot_ids")[0][0] == pytest.approx(17 / (len(params["passive_vocabulary"]) - 1))
    assert _seg(obs, "passive_slot_levels")[0][0] == pytest.approx(1.0)
    assert _seg(obs, "weapon_category")[0][0] == pytest.approx(0.0)  # Garlic → aura


# ---- スロット割り当て・残り時間（M9）----

def test_zone_slot_requires_unique_emitter_fire_wand_plus_santa_water_is_unknown():
    """zone を出しうる武器が1つならスロット有効、Fire Wand + Santa Water なら validity 0。"""
    zone = [_track("weapon_zone", 96.0, 0.0, first_seen=9.0)]
    unique = _build(zone, hud=_hud({2: ("SantaWater", 3)}), duration_mult=1.5)
    assert _seg(unique, "weapon_zone_slot")[0][0] == pytest.approx(2 / 5)
    assert _seg(unique, "weapon_zone_ttl")[0][0] == pytest.approx((0.1 + 2.5 * 1.5 - 1.0) / 8.0)
    both = _build(zone, hud=_hud({1: ("FireWand", 1), 2: ("SantaWater", 3)}))
    for name in ("weapon_zone_slot", "weapon_zone_ttl"):
        values, validity, age = _seg(both, name)
        assert validity[0] == 0 and age[0] == 1 and values[0] == 0
    fire = _build(zone, hud=_hud({4: ("Hellfire", 1)}), duration_mult=2.0)
    assert _seg(fire, "weapon_zone_slot")[0][0] == pytest.approx(4 / 5)
    assert _seg(fire, "weapon_zone_ttl")[0][0] == 0.0 and _seg(fire, "weapon_zone_ttl")[1][0] == 1  # 0.4s − 1.0s → 0 へ clip
    for obs in (unique, both, fire):
        _assert_missing_rule(obs)


def test_ttl_unknown_when_level_or_duration_mult_missing():
    """レベルか持続時間倍率が分からないと残り時間は不明、スロットは有効のまま。"""
    orbit = [_track("weapon_orbit", 96.0, 0.0, first_seen=9.0)]
    no_level = _build(orbit, hud=_hud({0: ("KingBible", None)}))
    no_mult = _build(orbit, hud=_hud({0: ("KingBible", 1)}), duration_mult=None)
    for obs in (no_level, no_mult):
        assert _seg(obs, "weapon_orbit_ttl")[1][0] == 0 and _seg(obs, "weapon_orbit_slot")[1][0] == 1
        _assert_missing_rule(obs)


def test_effect_duration_table_formula():
    """持続時間 = 固定分 + レベル別基準 × 倍率（LightningRing は倍率の影響なし）。"""
    assert effect_duration_s("SantaWater", 5, 1.0) == pytest.approx(2.85)
    assert effect_duration_s("LaBorra", 1, 1.3) == pytest.approx(0.1 + 4.0 * 1.3)
    assert effect_duration_s("FireWand", 8, 1.5) == pytest.approx(0.3)
    assert effect_duration_s("ThunderLoop", 1, 3.0) == pytest.approx(0.15)
    assert effect_duration_s("KingBible", 7, 1.0) == pytest.approx(4.0)
    with pytest.raises(ContractValidationError):
        effect_duration_s("Garlic", 1, 1.0)
    with pytest.raises(ContractValidationError):
        effect_duration_s("KingBible", 9, 1.0)


# ---- パラメータ・入力検証（M11）----

def test_weapon_track_max_age_frames_are_in_common_params():
    """weapon 4クラスの max_age フレーム数を Common の特徴量パラメータに持つ。"""
    params = load_deploy_obs_v2_feature_params()
    assert set(params["track_max_age_frames"]) == {"weapon_projectile", "weapon_zone", "weapon_orbit", "weapon_aura"}
    assert all(type(v) is int and v > 0 for v in params["track_max_age_frames"].values())
    with pytest.raises(TypeError):
        params["density"]["enemy"]["near"] = 0.5


def test_invalid_inputs_are_rejected_and_v1_segments_are_built():
    """NaN 座標・未来の初観測・重複スロットを拒否し、HP・レベル・移動方向・bias を組み立てる。"""
    with pytest.raises(ContractValidationError):
        TrackPx("enemy_normal", float("nan"), 0.0, 1.0, 1, 0.0, False)
    with pytest.raises(ContractValidationError):
        _build([_track("enemy_normal", 1.0, 0.0, first_seen=11.0)])
    with pytest.raises(ContractValidationError):
        _build([], hud=[HudSlot("weapon", 0, "Whip", 1), HudSlot("weapon", 0, "Axe", 1)])
    with pytest.raises(ContractValidationError):
        HudSlot("weapon", 0, None, 3)
    obs = _build([], hp_ratio=0.25, player_level=33, movement_direction=(0.6, -1.5))
    assert _seg(obs, "player_hp")[0][0] == pytest.approx(0.25) and _seg(obs, "level")[0][0] == pytest.approx(1 / 3)
    assert _seg(obs, "movement_direction")[0] == pytest.approx([0.6, -1.0])
    assert _seg(obs, "bias")[0][0] == 1.0
    for name in ("enemy_hp", "cooldown"):
        assert _seg(obs, name)[1][0] == 0
    obs.validate_for(V2)
    assert obs.as_policy_tensor(V2).shape == (3 * V2.dim,)


# ---- golden fixture（M12）----

def _case_inputs(case):
    """fixture の JSON 入力をビルダー引数へ戻す。"""
    data = dict(case["input"])
    data["viewport_wh"] = tuple(data["viewport_wh"])
    data["player_px"] = tuple(data["player_px"])
    data["tracks"] = [TrackPx(**t) for t in data["tracks"]]
    data["hud_slots"] = None if data["hud_slots"] is None else [HudSlot(**s) for s in data["hud_slots"]]
    if data.get("movement_direction") is not None:
        data["movement_direction"] = tuple(data["movement_direction"])
    return data


def test_golden_fixture_matches_builder_output():
    """px 入力と期待 v2 tensor の golden fixture にビルダー出力が一致する。"""
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert golden["schema_hash"] == V2.schema_hash and len(golden["cases"]) >= 3
    for case in golden["cases"]:
        obs = build_deploy_obs_v2(**_case_inputs(case))
        for plane in ("values", "validity", "age"):
            np.testing.assert_allclose(getattr(obs, plane), np.asarray(case["expected"][plane], np.float32), atol=1e-6, err_msg=f"{case['name']}:{plane}")
