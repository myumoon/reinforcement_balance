"""実機の tracked state + HUD → DeployObs v2 の入力変換（deploy_obs_v2_input）を検証する。

Common の golden fixture の入力を TrackedWorldStateV2 と HUD 在庫へ逆変換し、Deployment の
変換経路を通した tensor が期待値と一致することを確かめます（Training は import しません）。
可視・半径・anchor・HUD スロットの規則と、tracker 設定と Common パラメータの一致も確認します。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from reinbalance_survivors_contracts.deploy_obs import DeployObsSchema
from reinbalance_survivors_contracts.deploy_obs_v2_features import (
    HudSlot, TrackPx, build_deploy_obs_v2, load_deploy_obs_v2_feature_params,
)

from survivors.deploy_obs_v2_input import build_v2_observation, hud_slots_from_inventory, tracks_to_px
from survivors.hud_identity_vocabulary import load_hud_identity_vocabulary
from survivors.vision.entity_tracker import PlayerAnchorState, TrackedEntityV2, TrackedWorldStateV2

ROOT = Path(__file__).resolve().parents[3]
GOLDEN = ROOT / "Tools/Common/tests/fixtures/deploy_obs_v2_golden_v1.json"
DETECTOR_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "world_detector_v2.yaml"
V2 = DeployObsSchema.default_v2()
EMPTY = load_hud_identity_vocabulary().empty_slot


def _entity(class_name: str, cx: float, cy: float, *, w: float = .02, h: float = .02, track_id: int = 1,
            first_seen_ns: int = 0, confidence: float = .9, on_screen: bool = True, clipped: bool = False) -> TrackedEntityV2:
    """正規化座標で TrackedEntityV2 を1つ作る。

    player_relative・速度など v2 変換で使わない項目は 0 にします。
    """
    return TrackedEntityV2(
        track_id, 0, class_name, "x", confidence, 1, 0, cx, cy, 0., 0., 0., 0.,
        on_screen, clipped, w, h, first_seen_ns,
    )


def _world_from_case(data: dict) -> TrackedWorldStateV2:
    """golden case の px 入力を TrackedWorldStateV2 へ逆変換する。

    中心は px/画面サイズ、幅は 2r/W、高さは 2r/H 以下（半径規則 max(幅,高さ)/2 で r に戻る値）、
    初観測時刻は秒→ns にします。遮蔽された track は実機では検出されないものとして入れません。
    """
    width, height = data["viewport_wh"]
    tracks = [
        _entity(
            t["class_name"], t["cx_px"] / width, t["cy_px"] / height,
            w=2 * t["radius_px"] / width, h=t["radius_px"] / height, track_id=t["track_id"],
            first_seen_ns=round(t["first_seen_s"] * 1e9),
        )
        for t in data["tracks"] if not t["occluded"]
    ]
    px, py = data["player_px"]
    return TrackedWorldStateV2(1, round(data["now_s"] * 1e9), tracks, PlayerAnchorState(px / width, py / height, .9, False))


def _inventory_from_case(slots: list[dict]) -> tuple[tuple[str | None, ...], dict[str, int | None]]:
    """golden case の hud_slots を HUD 在庫 identity と identity→レベル表へ逆変換する。

    対応表を逆引きして identity にし、空スロットは empty_slot、渡されていない（不明な）スロットは
    偶数枠を表に無い identity・奇数枠を読めなかった枠（None）にして両方の「不明」経路を通します。
    """
    vocab = load_hud_identity_vocabulary()
    reverse = {("weapon", name): ident for ident, name in vocab.weapons.items()}
    reverse.update({("passive", name): ident for ident, name in vocab.passives.items()})
    inventory: list[str | None] = [f"unrecognized_{i}" if i % 2 == 0 else None for i in range(12)]
    levels: dict[str, int | None] = {}
    for slot in slots:
        position = slot["index"] + (0 if slot["kind"] == "weapon" else 6)
        if slot["type_name"] is None:
            inventory[position] = vocab.empty_slot
            continue
        identity = reverse[(slot["kind"], slot["type_name"])]
        inventory[position] = identity
        levels[identity] = slot["level"]
    return tuple(inventory), levels


def _golden_cases() -> list[dict]:
    """Common の golden fixture の case 一覧を返す。

    fixture の schema hash が現在の v2 schema と同じことも確かめます。
    """
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert golden["schema_hash"] == V2.schema_hash
    return golden["cases"]


@pytest.mark.parametrize("case", _golden_cases(), ids=lambda c: c["name"])
def test_deployment_path_matches_common_golden_fixture(case):
    """Deployment の変換経路を通した v2 tensor が Common golden fixture の期待値と一致する。

    1920x1080 と縦横比の違う 1280x1024 の case を含み、track・HUD スロットの逆変換から
    px・半径・スロット名・レベルが元の入力どおりに戻ることを確かめます。
    """
    data = case["input"]
    inventory, levels = _inventory_from_case(data["hud_slots"])
    hud_slots = hud_slots_from_inventory(inventory, levels.get)
    expected_slots = sorted((HudSlot(**s) for s in data["hud_slots"]), key=lambda s: (s.kind, s.index))
    assert sorted(hud_slots, key=lambda s: (s.kind, s.index)) == expected_slots
    move = data.get("movement_direction")
    obs = build_v2_observation(
        world=_world_from_case(data), viewport=tuple(data["viewport_wh"]), hud_slots=hud_slots,
        now_s=data["now_s"], duration_mult=data["duration_mult"], world_valid=data["world_valid"],
        hp_ratio=data.get("hp_ratio"), player_level=data.get("player_level"),
        movement_direction=None if move is None else tuple(move),
    )
    assert obs.schema_hash == V2.schema_hash
    for plane in ("values", "validity", "age"):
        np.testing.assert_allclose(getattr(obs, plane), np.asarray(case["expected"][plane], np.float32), atol=1e-6, err_msg=f"{case['name']}:{plane}")


def test_non_16_9_viewport_uses_isotropic_px_and_max_radius():
    """1600x1200 でも px は正規化×各辺、半径は max(幅_px, 高さ_px)/2 で Common と一致する。

    幅・高さの正規化分母が違うので、どちらかの辺で割り間違えると半径や位置がずれます。
    """
    viewport = (1600, 1200)
    world = TrackedWorldStateV2(
        1, 2_000_000_000,
        [_entity("weapon_aura", .5, .5, w=.05, h=.1), _entity("enemy_normal", .75, .25, track_id=2),
         _entity("weapon_zone", .6, .6, w=.1, h=.05, track_id=3, first_seen_ns=1_500_000_000)],
        PlayerAnchorState(.5, .5, .9, False),
    )
    tracks = tracks_to_px(world, viewport)
    assert [t.radius_px for t in tracks] == pytest.approx([60., 16., 80.])
    assert (tracks[1].cx_px, tracks[1].cy_px) == pytest.approx((1200., 300.))
    assert tracks[2].first_seen_s == pytest.approx(1.5)
    slots = [HudSlot("weapon", i, "Garlic" if i == 0 else None, 1 if i == 0 else None) for i in range(6)]
    direct = build_deploy_obs_v2(
        viewport_wh=viewport, player_px=(800., 600.), tracks=tracks, hud_slots=slots, now_s=2.,
        duration_mult=1., world_valid=True,
    )
    via = build_v2_observation(world=world, viewport=viewport, hud_slots=slots, now_s=2., duration_mult=1., world_valid=True)
    for plane in ("values", "validity", "age"):
        np.testing.assert_array_equal(getattr(via, plane), getattr(direct, plane))
    offset, _ = V2.layout["weapon_aura_radius"]
    assert via.values[offset] == pytest.approx(60. / 800.)


def test_visibility_uses_center_rule_not_tracker_clipped_flags():
    """可視判定は中心が画面内か（Common 規則）で、tracker の clipped / on_screen を使わない。

    clipped でも中心が画面内なら数え、中心が画面外なら on_screen=True でも数えません。
    信頼度が閾値未満の track は検出として渡しません。
    """
    viewport = (1000, 1000)
    world = TrackedWorldStateV2(1, 0, [
        _entity("enemy_normal", .9, .5, clipped=True, on_screen=False, track_id=1),
        _entity("enemy_normal", 1.05, .5, on_screen=True, track_id=2),
        _entity("enemy_normal", .2, .5, confidence=.1, track_id=3),
    ], PlayerAnchorState(.5, .5, .9, False))
    assert [t.track_id for t in tracks_to_px(world, viewport)] == [1, 2]
    obs = build_v2_observation(world=world, viewport=viewport, hud_slots=None, now_s=0., duration_mult=None, world_valid=True)
    offset, _ = V2.layout["visible_enemy_count"]
    assert obs.values[offset] == pytest.approx(1 / 20)


@pytest.mark.parametrize("anchor", [None, PlayerAnchorState(.5, .5, .9, True)])
def test_missing_or_fallback_anchor_makes_world_unknown(anchor):
    """player_anchor が無い・fallback のときは world 特徴を不明（validity 0）にする。

    プレイヤー位置が分からないと方向・距離の基準が無いので、確定した不在として扱いません。
    """
    world = TrackedWorldStateV2(1, 0, [_entity("enemy_normal", .7, .5)], anchor)
    obs = build_v2_observation(world=world, viewport=(1000, 1000), hud_slots=None, now_s=0., duration_mult=None, world_valid=True)
    for name in ("player_screen_pos", "enemy_nearest_dist_16dir", "weapon_projectile_density_16dir"):
        offset, size = V2.layout[name]
        assert not obs.validity[offset:offset + size].any(), name
        assert obs.age[offset:offset + size].tolist() == [1.] * size


def test_hud_slots_from_inventory_rules():
    """在庫 → HudSlot の変換規則: empty_slot だけが空確定、None（読めない枠）・表に無い・種別違いは渡さない、上限超えレベルは不明。

    渡されなかったスロットはビルダーで「不明」（validity 0）になります。
    """
    inventory = ("whip", EMPTY, "spinach", "mystery", "garlic", None,
                 "spellbinder", "whip", EMPTY, None, EMPTY, "armor")
    levels = {"whip": 3, "garlic": 99, "spellbinder": 2, "armor": None}
    slots = {(s.kind, s.index): s for s in hud_slots_from_inventory(inventory, levels.get)}
    assert slots[("weapon", 0)] == HudSlot("weapon", 0, "Whip", 3)
    assert slots[("weapon", 1)] == HudSlot("weapon", 1, None, None)
    assert ("weapon", 2) not in slots and ("weapon", 3) not in slots and ("weapon", 5) not in slots
    assert slots[("weapon", 4)] == HudSlot("weapon", 4, "Garlic", None)
    assert slots[("passive", 0)] == HudSlot("passive", 0, "Spellbinder", 2)
    assert ("passive", 1) not in slots and ("passive", 3) not in slots
    assert slots[("passive", 2)] == HudSlot("passive", 2, None, None)
    assert slots[("passive", 5)] == HudSlot("passive", 5, "Armor", None)
    with pytest.raises(ValueError):
        hud_slots_from_inventory(inventory[:6], levels.get)


def test_tracker_max_age_matches_common_track_max_age_frames():
    """tracker の weapon 4クラスの max_age が Common の track_max_age_frames と一致する。

    sim の初観測時刻の破棄規則と実機 tracker の track 消去を同じフレーム数に揃えます。
    """
    tracker = yaml.safe_load(DETECTOR_CONFIG.read_text(encoding="utf-8"))["tracker"]["max_age_by_class"]
    common = load_deploy_obs_v2_feature_params()["track_max_age_frames"]
    assert set(common) == {"weapon_projectile", "weapon_zone", "weapon_orbit", "weapon_aura"}
    assert {name: tracker[name] for name in common} == dict(common)
