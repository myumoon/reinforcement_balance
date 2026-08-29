"""可視 world track を画面空間の NamedEstimate へ変換する。

画面外・clipped track を最初に捨て、C++ と同じ方位規則で release 特徴を作ります。
"""
from __future__ import annotations
import math
from .deploy_obs_adapter import NamedEstimate
from .vision.entity_tracker import TrackedEntityV1, TrackedWorldStateV1
def directional_bin(dx: float, dy: float, bin_count: int = 8) -> int:
    """相対ベクトルを C++ BuildDirectionalDensityFeatures と同じ bin へ写す。

    ``atan2 + PI`` により +X は八分割時の bin 4 になり、境界は末端へ clamp します。
    """
    if isinstance(dx, bool) or isinstance(dy, bool) or not all(math.isfinite(float(value)) for value in (dx, dy)):
        raise ValueError("direction must be finite")
    if type(bin_count) is not int or bin_count <= 0:
        raise ValueError("bin_count must be positive int")
    angle01 = (math.atan2(float(dy), float(dx)) + math.pi) / (2.0 * math.pi)
    return max(0, min(bin_count - 1, math.floor(angle01 * bin_count)))
def _visible(tracks: list[TrackedEntityV1], coarse_class: str, threshold: float) -> list[TrackedEntityV1]:
    """指定 class の安全に可視な track だけを返す。

    all-state count を使わず on_screen・clipped・confidence を同じ入口で検査します。
    """
    return [track for track in tracks if track.coarse_class == coarse_class and track.on_screen and not track.clipped and track.confidence >= threshold]
def build_screen_space_estimates(
    world: TrackedWorldStateV1,
    *,
    directional_bins: int = 8,
    max_enemy_count: int = 20,
    max_gem_count: int = 20,
    min_confidence: float = .35,
) -> dict[str, NamedEstimate]:
    """TrackedWorldStateV1 から release-safe な画面特徴を構築する。

    deploy schema 外の density は item context 用で、呼び出し側が名前 allow-list で分離します。
    """
    if not isinstance(world, TrackedWorldStateV1):
        raise TypeError("world must be TrackedWorldStateV1")
    if type(max_enemy_count) is not int or type(max_gem_count) is not int or min(max_enemy_count, max_gem_count) <= 0:
        raise ValueError("density normalizers must be positive ints")
    enemies = _visible(world.tracks, "enemy", min_confidence)
    gems = _visible(world.tracks, "gem", min_confidence)
    enemy_density = min(len(enemies) / max_enemy_count, 1.)
    bins = [0.] * directional_bins
    for track in enemies:
        index = directional_bin(track.player_relative_x, track.player_relative_y, directional_bins)
        bins[index] += 1. / max_enemy_count
    result = {
        "visible_enemy_count": NamedEstimate((enemy_density,), world.timestamp_ns),
        "enemy_density": NamedEstimate((enemy_density,), world.timestamp_ns),
        "gem_density": NamedEstimate((min(len(gems) / max_gem_count, 1.),), world.timestamp_ns),
        "enemy_directional_density": NamedEstimate(tuple(min(value, 1.) for value in bins), world.timestamp_ns),
    }
    if world.player_anchor is not None:
        result["player_screen_pos"] = NamedEstimate(
            (2. * world.player_anchor.normalized_cx - 1., 2. * world.player_anchor.normalized_cy - 1.),
            world.timestamp_ns, world.player_anchor.confidence,
        )
    if enemies:
        nearest = min(enemies, key=lambda track: track.player_relative_x ** 2 + track.player_relative_y ** 2)
        offset = (nearest.player_relative_x, nearest.player_relative_y)
        anchor_conf = (
            world.player_anchor.confidence
            if world.player_anchor is not None and not world.player_anchor.is_fallback
            else 0.
        )
        offset_validity = min(nearest.confidence, anchor_conf)
        result["nearest_enemy_offset"] = NamedEstimate(offset, world.timestamp_ns, offset_validity)
        result["nearest_enemy_screen_radius"] = NamedEstimate((min(math.hypot(*offset), 1.),), world.timestamp_ns, offset_validity)
    return result
