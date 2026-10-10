"""RealObsAssembler の既定（DeployObs v2）経路を検証する。

v2 schema では Common の共有ビルダーで tensor を作り、HUD のスロット名・追跡したレベル・
持続時間倍率を渡すこと、停止画面では world・HUD 由来の値を不明にすること、
item context の boss_flag / hazard_flag が weapon の track に影響されないことを確かめます。
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from reinbalance_survivors_contracts.deploy_obs import DeployObsSchema
from reinbalance_survivors_contracts.deploy_obs_v2_features import (
    effect_duration_s, load_deploy_obs_v2_feature_params, normalized_vocabulary_id,
)

from survivors.hud_identity_vocabulary import load_hud_identity_vocabulary
from survivors.real_obs_assembler import RealObsAssembler
from survivors.temporal_state import TemporalAssembler
from survivors.vision.entity_tracker import PlayerAnchorState, TrackedEntityV2, TrackedWorldStateV2
from survivors.vision.hud_parser import HudParser, HudStateV1, ParsedCard
from survivors.vision.icon_matcher import MatchResult

V2 = DeployObsSchema.default_v2()
PARAMS = load_deploy_obs_v2_feature_params()
VIEWPORT = (1000, 1000)
EMPTY = load_hud_identity_vocabulary().empty_slot


def _hud(state="gameplay", *, ts, inventory=("whip",), level=4, cards=(), frame=1, levels=None) -> HudStateV1:
    """v2 経路テスト用の HUD を作る。

    在庫は先頭から詰め、残りは空スロット確定（empty_slot）にします。None は読めなかった枠です。
    """
    inv = tuple(inventory) + (EMPTY,) * (12 - len(inventory))
    return HudStateV1(
        "hud_state.v1", "session", frame, ts, "a" * 64, state, .9, "ok",
        20., .9, "ok", False, .75, .9, "ok", .5, .9, "ok", level, .9, "ok",
        inv, .9, "b" * 64, tuple(cards), "c" * 64, (),
        False, False, False, .9, "ok",
        inventory_levels=(None,) * 12 if levels is None else tuple(levels) + (None,) * (12 - len(levels)),
        inventory_levels_confidence=0.0 if levels is None else 1.0,
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
    assembler.assemble(_hud("level_up_items", ts=1_100_000_000, level=5, cards=(card,), frame=2, levels=(2,)), _world(1_100_000_000), V2, VIEWPORT)
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


@pytest.mark.parametrize("schema", [DeployObsSchema.default_v1(), V2], ids=["v1", "v2"])
@pytest.mark.parametrize("state", ["level_up_items", "level_up_fallback"])
@pytest.mark.parametrize("position,identity", [(1, "garlic"), (6, "spellbinder")])
def test_level_up_unread_baseline_is_not_filled_for_tracker(schema, state, position, identity):
    """パネルの不明枠を古い identity と結び付けず、強化も新規取得も推測しない。

    一枚だけ読めたアイコンの後にパネルが None を返す三フレーム列を、v1/v2 の本番経路へ通します。
    whip の +1 と、不明だった枠への Lv1・パネル段階値の採用をどちらも拒否します。
    """
    assembler = RealObsAssembler()
    inventory = ["whip"] + [EMPTY] * 11
    inventory[position] = identity
    baseline = inventory.copy()
    baseline[position] = None
    levels = [2] + [None] * 11
    levels[position] = 4
    card = ParsedCard(0, "whip", "weapon", 3, .99, "ok", (100, 100, 400, 500))
    assembler.assemble(_hud(ts=1_000_000_000, inventory=inventory), _world(1_000_000_000), schema, VIEWPORT)
    assembler.assemble(
        _hud(state, ts=1_100_000_000, inventory=baseline, cards=(card,), frame=2, levels=levels),
        _world(1_100_000_000), schema, VIEWPORT,
    )
    snap = assembler.assemble(
        _hud(ts=1_200_000_000, inventory=inventory, frame=3), _world(1_200_000_000), schema, VIEWPORT,
    )
    assert assembler._slot_levels.level("whip") is None
    assert assembler._slot_levels.level(identity) is None
    if schema.schema_version == V2.schema_version:
        assert _segment(snap.deploy_obs, "weapon_slot_levels")[1][0] == 0.
        segment = "weapon_slot_levels" if position < 6 else "passive_slot_levels"
        assert _segment(snap.deploy_obs, segment)[1][position % 6] == 0.


@pytest.mark.parametrize("state", ["level_up_items", "level_up_fallback"])
@pytest.mark.parametrize("position,identity", [(1, "garlic"), (6, "spellbinder")])
def test_level_up_unread_return_does_not_allow_owned_increment(state, position, identity):
    """gameplay 復帰時の不明枠が補完されても、所持カードの +1 を確定しない。

    パネルの全枠は読めていますが、戻った画面の一枠を None にします。
    whip の強化は不明とし、読めていた別アイテムの基準レベル4だけを保ちます。
    """
    assembler = RealObsAssembler()
    inventory = ["whip"] + [EMPTY] * 11
    inventory[position] = identity
    levels = [2] + [None] * 11
    levels[position] = 4
    card = ParsedCard(0, "whip", "weapon", 3, .99, "ok", (100, 100, 400, 500))
    assembler.assemble(_hud(ts=1_000_000_000, inventory=inventory), _world(1_000_000_000), V2, VIEWPORT)
    assembler.assemble(
        _hud(state, ts=1_100_000_000, inventory=inventory, cards=(card,), frame=2, levels=levels),
        _world(1_100_000_000), V2, VIEWPORT,
    )
    inventory[position] = None
    snap = assembler.assemble(
        _hud(ts=1_200_000_000, inventory=inventory, frame=3), _world(1_200_000_000), V2, VIEWPORT,
    )
    assert assembler._slot_levels.level("whip") is None
    assert assembler._slot_levels.level(identity) == 4
    assert _segment(snap.deploy_obs, "weapon_slot_levels")[1][0] == 0.


@pytest.mark.parametrize("state", ["level_up_items", "level_up_fallback"])
def test_level_up_confirmed_empty_still_allows_fresh_pick(state):
    """パネルで空と確定した枠への取得は、復帰後も Lv1 として採用する。

    前の生読みが garlic でも、基準が empty_slot なら garlic の新規取得と判定します。
    新規取得があるため、所持カードとして見えた whip は基準レベル2を保ちます。
    """
    assembler = RealObsAssembler()
    card = ParsedCard(0, "whip", "weapon", 3, .99, "ok", (100, 100, 400, 500))
    assembler.assemble(_hud(ts=1_000_000_000, inventory=("whip", "garlic")), _world(1_000_000_000), V2, VIEWPORT)
    assembler.assemble(
        _hud(state, ts=1_100_000_000, inventory=("whip", EMPTY), cards=(card,), frame=2, levels=(2,)),
        _world(1_100_000_000), V2, VIEWPORT,
    )
    snap = assembler.assemble(
        _hud(ts=1_200_000_000, inventory=("whip", "garlic"), frame=3), _world(1_200_000_000), V2, VIEWPORT,
    )
    assert assembler._slot_levels.level("whip") == 2
    assert assembler._slot_levels.level("garlic") == 1
    assert _segment(snap.deploy_obs, "weapon_slot_levels")[1][:2] == [1., 1.]


@pytest.mark.parametrize("old_session", [False, True])
def test_rejected_hud_does_not_reach_slot_tracker(old_session):
    """時刻が古い HUD と旧セッションの HUD を段階値の基準に使わない。

    TemporalAssembler が拒否したパネルを後から渡しても、次の gameplay は whip の Lv1 を保ちます。
    旧セッションでは保持中 HUD と同じ時刻を使い、時刻の比較だけで通さないことも確かめます。
    """
    assembler = RealObsAssembler()
    assembler.assemble(_hud(ts=1_000_000_000), _world(1_000_000_000), V2, VIEWPORT)
    panel = _hud("level_up_items", ts=1_000_000_000 if old_session else 900_000_000, frame=2, levels=(4,))
    if old_session:
        panel = dataclasses.replace(panel, session_id="old-session")
    assembler.assemble(panel, _world(1_100_000_000), V2, VIEWPORT)
    snap = assembler.assemble(_hud(ts=1_200_000_000, frame=3), _world(1_200_000_000), V2, VIEWPORT)
    assert assembler._slot_levels.level("whip") == 1
    assert _segment(snap.deploy_obs, "weapon_slot_levels")[1][0] == 1.


def test_duration_mult_comes_from_passive_slots_via_common():
    """パッシブの Spellbinder から Common の関数で求めた倍率が orbit の残り時間に使われる。

    パッシブ枠に対応表に無い identity があると倍率は不明になり、残り時間も不明になります。
    """
    ts = 3_000_000_000
    orbit = _track(1, "weapon_orbit", "weapon", .6, .5, first_seen_ns=2_000_000_000)
    inventory = ("king_bible", EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, "spellbinder")
    snap = RealObsAssembler().assemble(_hud(ts=ts, inventory=inventory), _world(ts, [orbit]), V2, VIEWPORT)
    ttl, valid = _segment(snap.deploy_obs, "weapon_orbit_ttl")
    expected = (effect_duration_s("KingBible", 1, 1.1) - 1.0) / PARAMS["max_projectile_obs_ttl_s"]
    assert valid == [1.] and ttl[0] == pytest.approx(expected, abs=1e-6)
    unknown = RealObsAssembler().assemble(
        _hud(ts=ts, inventory=inventory[:6] + ("mystery_passive",)), _world(ts, [orbit]), V2, VIEWPORT,
    )
    assert _segment(unknown.deploy_obs, "weapon_orbit_ttl")[1] == [0.]


class _ScriptedMatcher:
    """枠ごとに決めた MatchResult を順に返す icon matcher の代役。

    HudParser._parse_inventory が matcher の結果をどう在庫へ写すかだけを確かめるために使います。
    """

    def __init__(self, results):
        """解析に使う初期状態を準備する。

        指定された照合結果を、呼び出すたびに一件ずつ取り出せるようにします。
        """
        self._results = iter(results)

    def match(self, crop):
        """次の枠の結果を返す（crop は見ない）。

        実際の画像照合を省き、スロット順に決めておいた MatchResult を返します。
        """
        return next(self._results)


def _parsed_inventory(results) -> tuple:
    """12 枠分の MatchResult を HudParser の在庫解析に通した identity 列を返す。

    合成フレームと照合結果の代役を使い、空枠と読めない枠が在庫にどう反映されるかを調べます。
    """
    parser = HudParser(parser_artifact_hash="a" * 64, icon_matcher=_ScriptedMatcher(results))
    inventory, _ = parser._parse_inventory(np.zeros((1080, 1920, 4), np.uint8), 1920, 1080)
    return inventory


def _ok(item_id, kind="weapon") -> MatchResult:
    """読めた枠の結果。

    指定アイテムを信頼度0.9で確定した MatchResult を、スロットのテスト入力として作ります。
    """
    return MatchResult(item_id, kind, 1, .9, "ok")


LOW_MARGIN = MatchResult(None, "unknown", None, .1, "low_margin:0.030<0.15")


@pytest.mark.parametrize("unread", ["weapon", "passive"])
def test_low_margin_slot_is_unknown_not_confirmed_empty(unread):
    """icon_matcher が low_margin を返した枠は「空確定」ではなく「不明」（validity 0）になる。

    passive 枠が読めないと Spellbinder が隠れている可能性があるので持続時間倍率が不明になり、
    orbit の残り時間も validity 0 になります。武器枠が読めないと aura・orbit・zone の
    出し手を「無し確定」とは言えないので、それらの slot も validity 0 です。確定した空枠は validity 1 のままです。
    """
    results = [_ok("king_bible")] + [_ok(EMPTY, "unknown")] * 11
    position = 1 if unread == "weapon" else 6
    results[position] = LOW_MARGIN
    inventory = _parsed_inventory(results)
    assert inventory[position] is None
    ts = 3_000_000_000
    orbit = _track(1, "weapon_orbit", "weapon", .6, .5, first_seen_ns=2_000_000_000)
    zone = _track(2, "weapon_zone", "weapon", .4, .5, first_seen_ns=2_000_000_000)
    snap = RealObsAssembler().assemble(_hud(ts=ts, inventory=inventory), _world(ts, [orbit, zone]), V2, VIEWPORT)
    obs = snap.deploy_obs
    segment, index = ("weapon_slot_ids", 1) if unread == "weapon" else ("passive_slot_ids", 0)
    _, valid = _segment(obs, segment)
    assert valid[index] == 0. and sum(valid) == len(valid) - 1  # 読めない枠だけ不明、空確定枠は validity 1
    assert _segment(obs, "weapon_orbit_ttl")[1] == [0.]
    if unread == "weapon":
        for name in ("weapon_aura_slot", "weapon_orbit_slot"):
            assert not any(_segment(obs, name)[1]), name
        # 見えている zone の出し手スロット・残り時間も「無し確定」にならず不明
        assert _segment(obs, "weapon_zone_slot")[1][0] == 0. and _segment(obs, "weapon_zone_ttl")[1][0] == 0.


def test_confirmed_empty_slots_keep_slot_and_effect_validity():
    """全枠が identity か空確定なら、スロット・倍率・orbit の残り時間は従来どおり validity 1。

    whip と Spellbinder 以外を空枠にしても、観測値や持続時間を不明扱いにしないことを確かめます。
    """
    inventory = _parsed_inventory([_ok("king_bible")] + [_ok(EMPTY, "unknown")] * 11)
    ts = 3_000_000_000
    orbit = _track(1, "weapon_orbit", "weapon", .6, .5, first_seen_ns=2_000_000_000)
    obs = RealObsAssembler().assemble(_hud(ts=ts, inventory=inventory), _world(ts, [orbit]), V2, VIEWPORT).deploy_obs
    for name in ("weapon_slot_ids", "passive_slot_ids", "weapon_orbit_slot", "weapon_orbit_ttl", "weapon_aura_slot"):
        assert all(_segment(obs, name)[1]), name


def test_duration_mult_is_unknown_when_a_passive_slot_is_unread():
    """passive 枠が1つでも読めない（None）と Common の倍率関数は None（不明）を返す。

    空と確定した枠では倍率を計算できますが、読めないパッシブ枠があると効果を推測しません。
    """
    from reinbalance_survivors_contracts.deploy_obs_v2_features import duration_mult_from_hud_slots
    from survivors.deploy_obs_v2_input import hud_slots_from_inventory

    read = ("king_bible",) + (EMPTY,) * 5 + ("spellbinder",) + (EMPTY,) * 5
    assert duration_mult_from_hud_slots(hud_slots_from_inventory(read, lambda _: 3)) == pytest.approx(1.3)
    unread = read[:6] + (None,) + read[7:]
    assert duration_mult_from_hud_slots(hud_slots_from_inventory(unread, lambda _: 3)) is None


def test_temporal_filter_does_not_hold_confirmed_empty_through_unread_frame():
    """空確定だった枠が次のフレームで読めなくなったら、空を保持せず不明（None）へ戻す。

    新アイテムで埋まった直後にアイコンが読めないと、空のまま保持すると「空確定」が誤って続くためです。
    identity は従来どおり保持します（所持品は消えない）。
    """
    temporal = TemporalAssembler()
    temporal.observe_hud(_hud(ts=1_000_000_000, inventory=("whip", EMPTY)))
    temporal.observe_hud(_hud(ts=1_100_000_000, inventory=(None, None), frame=2))
    held = temporal._hud.inventory
    assert held[0] == "whip" and held[1] is None


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
