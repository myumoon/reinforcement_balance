"""実機の tracked world state と HUD を DeployObs v2 共有ビルダーの入力へ変換する。

特徴量そのものは Tools/Common の build_deploy_obs_v2 だけが計算し、ここでは
正規化座標を px に戻す・半径を求める・HUD の identity を語彙名へ写すといった入力変換だけを行います。
可視判定（中心が画面内・遮蔽なし）もビルダー側の規則に任せ、tracker の clipped / on_screen は使いません。
"""
from __future__ import annotations

from typing import Callable, Sequence

from reinbalance_survivors_contracts.deploy_obs import DeployObservation
from reinbalance_survivors_contracts.deploy_obs_v2_features import (
    HudSlot, TrackPx, build_deploy_obs_v2, load_deploy_obs_v2_feature_params,
)

from .hud_identity_vocabulary import HudIdentityVocabulary, load_hud_identity_vocabulary
from .vision.entity_tracker import TrackedWorldStateV2

# この信頼度未満の track は検出として扱わない（screen_space_features の _visible と同じ閾値）
TRACK_MIN_CONFIDENCE = .35


def tracks_to_px(world: TrackedWorldStateV2, viewport: tuple[int, int]) -> list[TrackPx]:
    """TrackedWorldStateV2 の track 列をビルダー用の px 座標 TrackPx 列へ戻す。

    中心は正規化座標×画面サイズ、半径は検出矩形の max(幅_px, 高さ_px)/2、初観測時刻は ns→秒です。
    実機の tracker は遮蔽を判定しないので occluded は常に False とし、画面外判定はビルダーに任せます。
    """
    width, height = viewport
    return [
        TrackPx(
            class_name=track.class_name,
            cx_px=track.normalized_cx * width,
            cy_px=track.normalized_cy * height,
            radius_px=max(track.normalized_width * width, track.normalized_height * height) / 2.0,
            track_id=track.track_id,
            first_seen_s=track.first_seen_timestamp_ns / 1e9,
            occluded=False,
        )
        for track in world.tracks
        if track.confidence >= TRACK_MIN_CONFIDENCE
    ]


def player_px(world: TrackedWorldStateV2, viewport: tuple[int, int]) -> tuple[tuple[float, float], bool]:
    """player_anchor の中心を px で返し、プレイヤー位置が信頼できるかも返す。

    anchor が無い・fallback のときは画面中央を仮置きし、False（world 特徴は不明）を返します。
    v1 の player_relative_x/y は使わず、プレイヤー基準の座標変換はビルダーに任せます。
    """
    width, height = viewport
    anchor = world.player_anchor
    if anchor is None:
        return (width / 2.0, height / 2.0), False
    return (anchor.normalized_cx * width, anchor.normalized_cy * height), not anchor.is_fallback


def hud_slots_from_inventory(
    inventory: Sequence[str | None],
    level_of: Callable[[str], int | None],
    vocabulary: HudIdentityVocabulary | None = None,
) -> list[HudSlot]:
    """HUD の在庫（武器6 + パッシブ6）を HudSlot 列へ変換する。

    None は空スロット確定、対応表に無い identity はそのスロットを渡さず「不明」にします。
    レベルは level_of（追跡器）から引き、上限を超えるような値は信用せず None（不明）にします。
    """
    params = load_deploy_obs_v2_feature_params()
    vocabulary = vocabulary or load_hud_identity_vocabulary()
    weapon_count = params["max_weapon_slots"]
    if len(inventory) != weapon_count + params["max_passive_slots"]:
        raise ValueError("inventory length does not match weapon + passive slots")
    slots = []
    for position, identity in enumerate(inventory):
        kind, index = ("weapon", position) if position < weapon_count else ("passive", position - weapon_count)
        if identity is None:
            slots.append(HudSlot(kind, index, None, None))
            continue
        name = vocabulary.type_name(identity, kind)
        if name is None:
            continue
        level = level_of(identity)
        max_level = params["max_weapon_level"] if kind == "weapon" else params["max_passive_level"]
        if level is not None and not (type(level) is int and 1 <= level <= max_level):
            level = None
        slots.append(HudSlot(kind, index, name, level))
    return slots


def build_v2_observation(
    *,
    world: TrackedWorldStateV2,
    viewport: tuple[int, int],
    hud_slots: Sequence[HudSlot] | None,
    now_s: float,
    duration_mult: float | None,
    world_valid: bool,
    hp_ratio: float | None = None,
    player_level: int | None = None,
    movement_direction: tuple[float, float] | None = None,
) -> DeployObservation:
    """tracked world state と HUD 由来の値から release 用 DeployObs v2 を作る。

    world_valid・duration_mult・movement_direction は呼び出し元の値をそのまま渡すので、
    golden fixture の入力を同じ経路で再現できます。anchor が信頼できないときは world_valid を落とします。
    """
    position, anchor_ok = player_px(world, viewport)
    return build_deploy_obs_v2(
        viewport_wh=viewport,
        player_px=position,
        tracks=tracks_to_px(world, viewport),
        hud_slots=None if hud_slots is None else list(hud_slots),
        now_s=now_s,
        duration_mult=duration_mult,
        world_valid=world_valid and anchor_ok,
        hp_ratio=hp_ratio,
        player_level=player_level,
        movement_direction=movement_direction,
    )
