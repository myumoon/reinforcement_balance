"""RealObsAssembler の既定（DeployObs v2）経路を検証する。

v2 schema では Common の共有ビルダーで tensor を作り、HUD のスロット名・追跡したレベル・
持続時間倍率を渡すこと、停止画面では world・HUD 由来の値を不明にすること、
item context の boss_flag / hazard_flag が weapon の track に影響されないことを確かめます。
"""
from __future__ import annotations

import dataclasses

import pytest
from reinbalance_survivors_contracts.deploy_obs import DeployObsSchema
from reinbalance_survivors_contracts.deploy_obs_v2_features import (
    effect_duration_s, load_deploy_obs_v2_feature_params, normalized_vocabulary_id,
)

from survivors.real_obs_assembler import RealObsAssembler
from survivors.vision.entity_tracker import PlayerAnchorState, TrackedEntityV2, TrackedWorldStateV2
from survivors.vision.hud_parser import HudStateV1, ParsedCard

V2 = DeployObsSchema.default_v2()
PARAMS = load_deploy_obs_v2_feature_params()
VIEWPORT = (1000, 1000)


def _hud(state="gameplay", *, ts, inventory=("whip",), level=4, cards=(), frame=1) -> HudStateV1:
    """v2 経路テスト用の HUD を作る。

    在庫は先頭から詰め、残りは空スロットにします。
    """
    inv = tuple(inventory) + (None,) * (12 - len(inventory))
    return HudStateV1(
        "hud_state.v1", "session", frame, ts, "a" * 64, state, .9, "ok",
        20., .9, "ok", False, .75, .9, "ok", .5, .9, "ok", level, .9, "ok",
        inv, .9, "b" * 64, tuple(cards), "c" * 64, (),
        False, False, False, .9, "ok",
    )


def _track(track_id, class_name, coarse, cx, cy, *, first_seen_ns=0) -> TrackedEntityV2:
    """画面内の track を1つ作る。

    正規化矩形は 0.04 四方で、clipped しない可視 track です。
    """
    return TrackedEntityV2(track_id, 0, class_name, coarse, .9, 2, 1, cx, cy, cx - .5, cy - .5, 0., 0., True, False, .04, .04, first_seen_ns)


def _world(ts, tracks=()) -> TrackedWorldStateV2:
    """プレイヤーが画面中央にいる world を作る。

    anchor は fallback ではない確定位置です。
    """
    return TrackedWorldStateV2(1, ts, list(tracks), PlayerAnchorState(.5, .5, .9, False))


def _segment(obs, name):
    """観測から指定 segment の value・validity を取り出す。

    テストでの比較を読みやすくするための補助です。
    """
    offset, size = V2.layout[name]
    return obs.values[offset:offset + size].tolist(), obs.validity[offset:offset + size].tolist()


def test_default_v2_snapshot_uses_common_builder_with_hud_slots_and_levels():
    """v2 schema の snapshot は v2 hash・snapshot と同じ ns を持ち、HUD スロットと Lv1 が有効になる。

    在庫の whip は対応表で Whip に写り、追跡器の規則 (a) でレベル1として渡されます。
    """
    ts = 1_000_000_000
    snap = RealObsAssembler().assemble(_hud(ts=ts), _world(ts, [_track(1, "enemy_normal", "enemy", .7, .5)]), V2, VIEWPORT)
    obs = snap.deploy_obs
    assert obs.schema_hash == V2.schema_hash and obs.timestamp_ns == snap.captured_ns
    ids, ids_valid = _segment(obs, "weapon_slot_ids")
    assert ids[0] == pytest.approx(normalized_vocabulary_id("Whip", PARAMS["weapon_vocabulary"]))
    assert ids_valid == [1.] * 6
    levels, levels_valid = _segment(obs, "weapon_slot_levels")
    assert levels[0] == pytest.approx(1 / PARAMS["max_weapon_level"]) and levels_valid == [1.] * 6
    count, count_valid = _segment(obs, "visible_enemy_count")
    assert count == pytest.approx([1 / 20]) and count_valid == [1.]
    _, move_valid = _segment(obs, "movement_direction")
    assert move_valid == [0., 0.]
    obs.validate_for(V2)


def test_level_up_choice_updates_slot_level_through_assembler():
    """レベルアップ画面で所持武器のカードを選んで戻ると、weapon_slot_levels が新レベルになる。

    assembler が tick ごとに追跡器へ HUD を渡していることを確かめます。
    """
    assembler = RealObsAssembler()
    assembler.assemble(_hud(ts=1_000_000_000, level=4), _world(1_000_000_000), V2, VIEWPORT)
    card = ParsedCard(0, "whip", "weapon", 3, .99, "ok", (100, 100, 400, 500))
    assembler.assemble(_hud("level_up_items", ts=1_100_000_000, level=5, cards=(card,), frame=2), _world(1_100_000_000), V2, VIEWPORT)
    snap = assembler.assemble(_hud(ts=1_200_000_000, level=5, frame=3), _world(1_200_000_000), V2, VIEWPORT)
    levels, valid = _segment(snap.deploy_obs, "weapon_slot_levels")
    assert levels[0] == pytest.approx(3 / PARAMS["max_weapon_level"]) and valid[0] == 1.


def test_non_combat_frame_makes_world_and_hud_values_unknown():
    """レベルアップ画面など combat が無効なフレームは world・HUD 由来の値を全て不明にする。

    v1 経路で validity に combat を掛けていたのと同じ扱いです。
    """
    ts = 1_000_000_000
    snap = RealObsAssembler().assemble(_hud("level_up_items", ts=ts), _world(ts, [_track(1, "enemy_normal", "enemy", .7, .5)]), V2, VIEWPORT)
    for name in ("player_hp", "level", "player_screen_pos", "enemy_nearest_dist_16dir", "weapon_slot_ids", "weapon_projectile_density_16dir"):
        _, valid = _segment(snap.deploy_obs, name)
        assert not any(valid), name


def test_duration_mult_comes_from_passive_slots_via_common():
    """パッシブの Spellbinder から Common の関数で求めた倍率が orbit の残り時間に使われる。

    パッシブ枠に対応表に無い identity があると倍率は不明になり、残り時間も不明になります。
    """
    ts = 3_000_000_000
    orbit = _track(1, "weapon_orbit", "weapon", .6, .5, first_seen_ns=2_000_000_000)
    inventory = ("king_bible", None, None, None, None, None, "spellbinder")
    snap = RealObsAssembler().assemble(_hud(ts=ts, inventory=inventory), _world(ts, [orbit]), V2, VIEWPORT)
    ttl, valid = _segment(snap.deploy_obs, "weapon_orbit_ttl")
    expected = (effect_duration_s("KingBible", 1, 1.1) - 1.0) / PARAMS["max_projectile_obs_ttl_s"]
    assert valid == [1.] and ttl[0] == pytest.approx(expected, abs=1e-6)
    unknown = RealObsAssembler().assemble(
        _hud(ts=ts, inventory=inventory[:6] + ("mystery_passive",)), _world(ts, [orbit]), V2, VIEWPORT,
    )
    assert _segment(unknown.deploy_obs, "weapon_orbit_ttl")[1] == [0.]


def test_boss_and_hazard_flags_ignore_weapon_tracks():
    """item context の boss_flag / hazard_flag は weapon の track に影響されない。

    プレイヤー自身の武器エフェクトを敵の危険として数えないよう、意味を v1 のまま保ちます。
    """
    weapons = [_track(i, name, "weapon", .6, .5) for i, name in enumerate(("weapon_aura", "weapon_zone", "weapon_orbit", "weapon_projectile"), 1)]
    for extra, boss, hazard in (([], False, False), ([_track(9, "hazard_area", "hazard", .4, .5)], False, True)):
        assembler = RealObsAssembler()
        assembler.assemble(_hud(ts=1_000_000_000), _world(1_000_000_000, weapons + extra), V2, VIEWPORT)
        card = ParsedCard(0, "knife", "weapon", 1, .99, "ok", (100, 100, 400, 500))
        snap = assembler.assemble(_hud("level_up_items", ts=1_100_000_000, cards=(card,), frame=2), _world(1_100_000_000), V2, VIEWPORT)
        assert snap.item_context is not None
        assert (snap.item_context.boss_flag, snap.item_context.hazard_flag) == (boss, hazard)


def test_v2_schema_with_other_hash_is_rejected():
    """v2 版でも default_v2() と内容が違う schema は拒否する。

    Common ビルダーの出力は default_v2() 固定なので、別内容の schema へ黙って流しません。
    """
    fields = list(V2.fields)
    fields[0] = dataclasses.replace(fields[0], max_age_ms=fields[0].max_age_ms + 1.)
    other = DeployObsSchema(tuple(fields), V2.schema_version)
    assert other.schema_hash != V2.schema_hash
    with pytest.raises(ValueError):
        RealObsAssembler().assemble(_hud(ts=1_000_000_000), _world(1_000_000_000), other, VIEWPORT)
