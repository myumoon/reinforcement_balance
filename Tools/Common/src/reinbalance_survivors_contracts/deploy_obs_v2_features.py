"""DeployObs v2 の value・validity・age を画面 px 入力から作る共有特徴量ビルダー。

sim（03-07）は投影後の px、実機（04-13）は tracked state を px に戻した値を同じ関数へ渡し、
Training と Deployment が特徴量の計算を各自で持たないようにします。
位置・距離・半径はすべて「プレイヤー基準・縦横同じ縮尺・viewport 半幅（W/2）で割る」座標系で、
track は中心が画面内にあり遮蔽されていなければ可視とします（矩形のはみ出しは問わない）。
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from types import MappingProxyType
from typing import Any, Mapping, Sequence
import math
import statistics

import numpy as np
import yaml

from .deploy_obs import DeployObsSchema, DeployObservation
from .ui_intent import ensure, is_strict_number

FEATURE_PARAMS_VERSION = "deploy_obs_v2_features.v1"
EFFECT_KINDS = ("projectile", "zone", "orbit", "aura")
HUD_SLOT_KINDS = ("weapon", "passive")
_PARAM_KEYS = frozenset({
    "schema_version", "entity_classes", "track_max_age_frames", "direction_bins", "density",
    "projectile_density_norm", "visible_enemy_count_norm", "level_norm", "zone_slots",
    "max_projectile_obs_ttl_s", "max_weapon_slots", "max_passive_slots", "max_weapon_level",
    "max_passive_level", "weapon_vocabulary", "passive_vocabulary", "weapon_category_vocabulary",
    "weapon_coarse_category", "weapon_effect_kinds", "effect_durations",
})
_DENSITY_KEYS = frozenset({"nearest_dist_max", "near", "mid", "near_norm", "mid_norm"})
# UE の KINDA_SMALL_NUMBER。C++ BuildDirDensity と同じく、これ以下の距離は方向が無いので除外する。
_KINDA_SMALL_NUMBER = 1e-4


def _freeze(value: Any) -> Any:
    """YAML の dict/list を読み取り専用の MappingProxy/tuple へ再帰変換する。

    読み込んだパラメータを呼び出し側が書き換えて、別の呼び出しの特徴量が変わる事故を防ぎます。
    """
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _validate_vocabulary(vocabulary: Any, label: str) -> None:
    """語彙が一意な文字列列で、末尾に unknown を予約していることを確かめる。

    normalized_vocabulary_id が index / (len-1) で [0,1] に写すための前提条件です。
    """
    ensure(isinstance(vocabulary, list) and len(vocabulary) >= 2, f"{label} must be a list")
    ensure(all(isinstance(x, str) and x for x in vocabulary), f"{label} entries must be str")
    ensure(vocabulary[-1] == "unknown" and len(set(vocabulary)) == len(vocabulary), f"{label} must be unique and end with unknown")


def _positive(value: Any, label: str) -> None:
    """値が有限な正の数であることを確かめる。

    正規化の割り算に使う係数が 0・負・NaN だと特徴量が壊れるため、読み込み時に拒否します。
    """
    ensure(is_strict_number(value) and math.isfinite(float(value)) and value > 0, f"{label} must be positive")


def _validate_params(data: Any) -> None:
    """特徴量パラメータ YAML の形と値を fail-closed で検証する。

    キーの過不足、語彙の重複、表の長さ違い、距離帯の逆転などを読み込み時点で止めます。
    """
    ensure(isinstance(data, dict) and set(data) == _PARAM_KEYS, "feature params keys mismatch")
    ensure(data["schema_version"] == FEATURE_PARAMS_VERSION, "unsupported feature params version")
    classes = data["entity_classes"]
    ensure(isinstance(classes, dict) and set(classes) == {"enemy", "gem", "rare_gem", "effect"}, "entity_classes keys mismatch")
    ensure(set(classes["rare_gem"]) <= set(classes["gem"]), "rare_gem must be a subset of gem")
    ensure(set(classes["effect"].values()) == set(EFFECT_KINDS), "effect classes must cover all effect kinds")
    ensure(set(data["track_max_age_frames"]) == set(classes["effect"]), "track_max_age_frames must cover weapon classes")
    ensure(all(type(v) is int and v > 0 for v in data["track_max_age_frames"].values()), "track_max_age_frames must be positive int")
    for key in ("direction_bins", "zone_slots", "max_weapon_slots", "max_passive_slots", "max_weapon_level", "max_passive_level"):
        ensure(type(data[key]) is int and data[key] > 0, f"{key} must be positive int")
    for key in ("projectile_density_norm", "visible_enemy_count_norm", "level_norm", "max_projectile_obs_ttl_s"):
        _positive(data[key], key)
    ensure(set(data["density"]) == {"enemy", "gem"}, "density keys mismatch")
    for name, band in data["density"].items():
        ensure(isinstance(band, dict) and set(band) == _DENSITY_KEYS, f"density.{name} keys mismatch")
        for key in _DENSITY_KEYS:
            _positive(band[key], f"density.{name}.{key}")
        ensure(band["near"] < band["mid"], f"density.{name} near must be below mid")
    for key in ("weapon_vocabulary", "passive_vocabulary", "weapon_category_vocabulary"):
        _validate_vocabulary(data[key], key)
    weapons = set(data["weapon_vocabulary"][1:-1])
    ensure(set(data["weapon_coarse_category"]) <= weapons, "weapon_coarse_category has unknown weapon")
    ensure(set(data["weapon_coarse_category"].values()) <= set(data["weapon_category_vocabulary"][:-1]), "unknown coarse category")
    ensure(set(data["weapon_effect_kinds"]) <= weapons, "weapon_effect_kinds has unknown weapon")
    for weapon, kinds in data["weapon_effect_kinds"].items():
        ensure(isinstance(kinds, list) and kinds and set(kinds) <= set(EFFECT_KINDS) and len(set(kinds)) == len(kinds), f"{weapon} effect kinds invalid")
    timed = {w for w, kinds in data["weapon_effect_kinds"].items() if {"zone", "orbit"} & set(kinds)}
    ensure(set(data["effect_durations"]) == timed, "effect_durations must cover exactly zone/orbit weapons")
    for weapon, row in data["effect_durations"].items():
        ensure(isinstance(row, dict) and set(row) == {"fixed_s", "scaled_by_level_s"}, f"{weapon} duration keys mismatch")
        scaled = row["scaled_by_level_s"]
        ensure(isinstance(scaled, list) and len(scaled) == data["max_weapon_level"], f"{weapon} duration table length mismatch")
        ensure(all(is_strict_number(v) and math.isfinite(float(v)) and v >= 0 for v in (row["fixed_s"], *scaled)), f"{weapon} durations must be finite non-negative")


@lru_cache(maxsize=1)
def load_deploy_obs_v2_feature_params() -> Mapping[str, Any]:
    """package-data の特徴量パラメータ YAML を検証して読み取り専用で返す。

    距離帯・語彙・武器→エフェクト種類・持続時間表は全てこの1ファイルから読みます。
    結果は読み取り専用なので、キャッシュして全呼び出しで共有しても安全です。
    """
    path = files("reinbalance_survivors_contracts").joinpath("schemas/deploy_obs_v2_features.yaml")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    _validate_params(data)
    return _freeze(data)


def directional_bin(dx: float, dy: float, bin_count: int = 16) -> int:
    """相対ベクトルを C++ BuildDirDensity と同じ規則の方向ビン番号へ写す。

    ``(atan2(dy, dx) + π) / 2π`` を bin 数倍して切り捨て、端は [0, bin_count-1] へ収めます。
    Deployment の screen_space_features.directional_bin と同じ式です（04-13 でこちらへ一本化）。
    """
    ensure(is_strict_number(dx) and is_strict_number(dy) and math.isfinite(float(dx)) and math.isfinite(float(dy)), "direction must be finite")
    ensure(type(bin_count) is int and bin_count > 0, "bin_count must be positive int")
    angle01 = (math.atan2(float(dy), float(dx)) + math.pi) / (2.0 * math.pi)
    return max(0, min(bin_count - 1, math.floor(angle01 * bin_count)))


def normalized_vocabulary_id(value: str | None, vocabulary: Sequence[str]) -> float:
    """種類名を語彙内 index / (語彙長-1) で [0,1] に正規化する。

    None は空スロットを表す語彙先頭の "None" に、語彙に無い名前は末尾の unknown に写します。
    Deployment の normalized_category と同じ規則です。
    """
    name = "None" if value is None else value
    ensure(isinstance(name, str), "category must be str or None")
    index = vocabulary.index(name) if name in vocabulary else len(vocabulary) - 1
    return index / (len(vocabulary) - 1)


def effect_duration_s(weapon: str, level: int, duration_mult: float) -> float:
    """zone / orbit エフェクトの持続時間（秒）を C++ と同じ式で返す。

    持続時間 = 固定分 + レベル別の基準時間 × 持続時間倍率（Spellbinder で増える）です。
    """
    params = load_deploy_obs_v2_feature_params()
    ensure(weapon in params["effect_durations"], f"{weapon} has no timed effect")
    ensure(type(level) is int and 1 <= level <= params["max_weapon_level"], "level out of range")
    _positive(duration_mult, "duration_mult")
    row = params["effect_durations"][weapon]
    return float(row["fixed_s"]) + float(row["scaled_by_level_s"][level - 1]) * float(duration_mult)


@dataclass(frozen=True)
class TrackPx:
    """画面 px 座標で表した1つの world track。

    class 名・中心座標・半径・track id・初めて見えた時刻・遮蔽フラグだけを持ちます。
    sim は当たり判定半径を px に、実機は検出矩形の max(幅,高さ)/2 を半径として渡します。
    """

    class_name: str
    cx_px: float
    cy_px: float
    radius_px: float
    track_id: int
    first_seen_s: float
    occluded: bool

    def __post_init__(self) -> None:
        """型と有限性を検証する。

        NaN 座標や負の半径が特徴量へ混ざる前に止めます。
        """
        ensure(isinstance(self.class_name, str) and bool(self.class_name), "class_name must be non-empty str")
        for value in (self.cx_px, self.cy_px, self.radius_px, self.first_seen_s):
            ensure(is_strict_number(value) and math.isfinite(float(value)), "track numbers must be finite")
        ensure(self.radius_px >= 0 and self.first_seen_s >= 0, "radius and first_seen must be non-negative")
        ensure(type(self.track_id) is int, "track_id must be int")
        ensure(type(self.occluded) is bool, "occluded must be bool")


def is_track_visible(track: TrackPx, viewport_wh: tuple[int, int]) -> bool:
    """track が v2 の可視規則（中心が画面内・遮蔽なし）を満たすか返す。

    ビルダーの可視判定と、sim wrapper の初観測時刻の追跡が同じ規則を使うための共有関数です。
    矩形が画面からはみ出していても、中心が画面内なら可視とします。
    """
    return not track.occluded and 0.0 <= track.cx_px <= viewport_wh[0] and 0.0 <= track.cy_px <= viewport_wh[1]


@dataclass(frozen=True)
class HudSlot:
    """HUD から読んだ武器またはパッシブの1スロット。

    type_name が None なら空スロットとして確定、level が None ならレベルが読めなかったことを表します。
    読めなかったスロットは HudSlot 自体を渡さないことで「不明」を表します。
    """

    kind: str
    index: int
    type_name: str | None
    level: int | None

    def __post_init__(self) -> None:
        """種別・番号・名前・レベルの型を検証する。

        空スロットにレベルが付いているような矛盾した入力を拒否します。
        """
        ensure(self.kind in HUD_SLOT_KINDS, "unknown hud slot kind")
        ensure(type(self.index) is int and self.index >= 0, "slot index must be non-negative int")
        ensure(self.type_name is None or (isinstance(self.type_name, str) and bool(self.type_name)), "type_name must be str or None")
        ensure(self.level is None or (type(self.level) is int and self.level >= 1), "level must be positive int or None")
        ensure(not (self.type_name is None and self.level is not None), "empty slot cannot have a level")


class _Planes:
    """schema 順の value・validity を segment 名で書き込む作業領域。

    全 segment を neutral・validity 0 で始め、書き込まれなかった segment は自動的に「不明」になります。
    validity 0 の要素は必ず neutral に戻すので、欠損表現の規則を各経路で守らなくて済みます。
    """

    def __init__(self, schema: DeployObsSchema) -> None:
        """schema の layout と neutral で初期化する。

        value は neutral、validity は 0 の状態から始めます。
        """
        self.schema = schema
        self.values = np.zeros(schema.dim, np.float32)
        self.validity = np.zeros(schema.dim, np.float32)
        self.neutral = np.zeros(schema.dim, np.float32)
        for field in schema.fields:
            offset, size = schema.layout[field.name]
            self.neutral[offset:offset + size] = field.neutral
        self.values[:] = self.neutral

    def put(self, name: str, values: Any, valid: Any) -> None:
        """segment に値と有効性（要素ごと or 一括）を書き込む。

        有効性が偽の要素は値を neutral に置き換えます。
        """
        offset, size = self.schema.layout[name]
        span = slice(offset, offset + size)
        valid_arr = np.broadcast_to(np.asarray(valid, dtype=bool), (size,))
        values_arr = np.broadcast_to(np.asarray(values, dtype=np.float32), (size,))
        self.values[span] = np.where(valid_arr, values_arr, self.neutral[span])
        self.validity[span] = valid_arr.astype(np.float32)


@dataclass(frozen=True)
class _Emitter:
    """あるエフェクト種類を出しうる武器の HUD 上の状況。

    slot / weapon / level は「出しうる武器がちょうど1つと確定」したときだけ埋まります。
    none_certain は「6スロット全部が読めていて、出しうる武器が1つも無い」ことを表します。
    """

    slot: int | None
    weapon: str | None
    level: int | None
    none_certain: bool


def _clip(value: float, low: float, high: float) -> float:
    """値を [low, high] に収める。

    座標や比率が範囲外にはみ出したときに、schema の範囲へ戻すために使います。
    """
    return min(high, max(low, value))


def _direction_features(points: Sequence[tuple[float, float]], band: Mapping[str, float], bins: int) -> tuple[list[float], list[float], list[float]]:
    """C++ BuildDirDensity と同じ式で方向別の最寄り距離・近距離密度・中距離密度を作る。

    方向ごとに一番近い距離（無ければ 1.0）と、近距離帯・中距離帯で近いほど重い重みの合計を
    正規化係数で割った値を返します。距離はすべて半幅正規化空間の値です。
    """
    nearest, near, mid = [1.0] * bins, [0.0] * bins, [0.0] * bins
    for dx, dy in points:
        distance = math.hypot(dx, dy)
        if distance <= _KINDA_SMALL_NUMBER:
            continue
        index = directional_bin(dx, dy, bins)
        nearest[index] = min(nearest[index], _clip(distance / band["nearest_dist_max"], 0.0, 1.0))
        if distance <= band["near"]:
            near[index] += _clip(1.0 - distance / band["near"], 0.0, 1.0)
        elif distance <= band["mid"]:
            near_t = (distance - band["near"]) / (band["mid"] - band["near"])
            mid[index] += _clip(1.0 - near_t, 0.0, 1.0)
    return (
        nearest,
        [_clip(v / band["near_norm"], 0.0, 1.0) for v in near],
        [_clip(v / band["mid_norm"], 0.0, 1.0) for v in mid],
    )


def _emitter(kind: str, weapons: Mapping[int, HudSlot], params: Mapping[str, Any]) -> _Emitter:
    """HUD の武器スロットから、指定エフェクト種類を出す武器の状況を求める。

    全スロットが読めて、語彙外の武器が無いときだけ「確定」と判断します。
    同じ種類を出す武器が2つ以上あるとスロットは一意に決まらないので未確定のままにします。
    """
    vocabulary = params["weapon_vocabulary"]
    complete = all(i in weapons for i in range(params["max_weapon_slots"])) and all(
        slot.type_name is None or slot.type_name in vocabulary[1:-1] for slot in weapons.values()
    )
    emitters = [
        slot for slot in weapons.values()
        if slot.type_name is not None and kind in params["weapon_effect_kinds"].get(slot.type_name, ())
    ]
    if complete and len(emitters) == 1:
        return _Emitter(emitters[0].index, emitters[0].type_name, emitters[0].level, False)
    return _Emitter(None, None, None, complete and not emitters)


def _ttl(emitter: _Emitter, first_seen_s: float, now_s: float, duration_mult: float | None, params: Mapping[str, Any]) -> tuple[float, bool]:
    """残り時間 = 持続時間 − 経過時間 を MaxProjectileObsTtl で割った値と有効性を返す。

    スロットが一意でない、レベルや持続時間倍率が分からないときは不明（無効）にします。
    """
    if emitter.slot is None or emitter.level is None or duration_mult is None:
        return 0.0, False
    max_ttl = float(params["max_projectile_obs_ttl_s"])
    remaining = effect_duration_s(emitter.weapon, emitter.level, duration_mult) - (now_s - first_seen_s)
    return _clip(remaining, 0.0, max_ttl) / max_ttl, True


def _hud_slots(hud_slots: Sequence[HudSlot] | None, params: Mapping[str, Any]) -> tuple[dict[int, HudSlot], dict[int, HudSlot]]:
    """HUD スロット列を検証し、武器・パッシブそれぞれの番号→スロットの辞書にする。

    同じスロットが2回来る・番号が範囲外・レベルが上限超えといった入力は拒否します。
    """
    weapons: dict[int, HudSlot] = {}
    passives: dict[int, HudSlot] = {}
    for slot in hud_slots or ():
        ensure(isinstance(slot, HudSlot), "hud slot must be HudSlot")
        is_weapon = slot.kind == "weapon"
        target = weapons if is_weapon else passives
        limit = params["max_weapon_slots"] if is_weapon else params["max_passive_slots"]
        max_level = params["max_weapon_level"] if is_weapon else params["max_passive_level"]
        ensure(slot.index < limit and slot.index not in target, "hud slot index out of range or duplicated")
        ensure(slot.level is None or slot.level <= max_level, "hud slot level above maximum")
        target[slot.index] = slot
    return weapons, passives


def build_deploy_obs_v2(
    *,
    viewport_wh: tuple[int, int],
    player_px: tuple[float, float],
    tracks: Sequence[TrackPx],
    hud_slots: Sequence[HudSlot] | None,
    now_s: float,
    duration_mult: float | None,
    world_valid: bool,
    hp_ratio: float | None = None,
    player_level: int | None = None,
    movement_direction: tuple[float, float] | None = None,
) -> DeployObservation:
    """画面 px の入力から release 用 DeployObs v2 の3平面を作る。

    画面全体を走査する特徴（敵・ジェムの方向特徴、projectile 密度、zone の空き枠）は、
    見えなかったことを「無い」という観測として neutral・validity=world_valid にします。
    aura・orbit・zone のスロットと残り時間は、HUD から出しうる武器が1つと確定したときだけ有効、
    出しうる武器が無いと確定したときは neutral・validity 1、それ以外は不明（validity 0・age 1）です。
    HP・レベル・移動方向は渡されたときだけ有効にし、enemy_hp・cooldown は常に欠損にします。
    """
    params = load_deploy_obs_v2_feature_params()
    schema = _default_v2_schema()
    ensure(isinstance(viewport_wh, tuple) and len(viewport_wh) == 2 and all(type(v) is int and v > 0 for v in viewport_wh), "viewport_wh must be a positive int pair")
    ensure(isinstance(player_px, tuple) and len(player_px) == 2 and all(is_strict_number(v) and math.isfinite(float(v)) for v in player_px), "player_px must be a finite pair")
    ensure(is_strict_number(now_s) and math.isfinite(float(now_s)) and now_s >= 0, "now_s must be finite non-negative")
    ensure(duration_mult is None or (is_strict_number(duration_mult) and math.isfinite(float(duration_mult)) and duration_mult > 0), "duration_mult must be positive or None")
    ensure(type(world_valid) is bool, "world_valid must be bool")
    # generator など1回しか回せない iterable は検証で消費されて全 track が消えるため、list/tuple だけを受け付ける
    ensure(isinstance(tracks, (list, tuple)), "tracks must be a list or tuple")
    ensure(hud_slots is None or isinstance(hud_slots, (list, tuple)), "hud_slots must be a list, tuple or None")
    ensure(all(isinstance(t, TrackPx) for t in tracks), "tracks must be TrackPx")
    ensure(all(t.first_seen_s <= now_s for t in tracks), "track first_seen_s is in the future")
    width, height = viewport_wh
    half = width / 2.0
    px, py = float(player_px[0]), float(player_px[1])

    def raw(track: TrackPx) -> tuple[float, float]:
        """track 中心のプレイヤー基準・半幅正規化座標を clip せずに返す。

        縦横とも viewport 半幅で割るので縮尺が等しくなります。
        方向ビンと距離はこの値から求め、clip で方向が歪まないようにします。
        """
        return (track.cx_px - px) / half, (track.cy_px - py) / half

    def clip_xy(point: tuple[float, float]) -> tuple[float, float]:
        """相対座標を出力用に [-1,1] へ clip する。

        nearest_enemy_offset や zone の位置など、値として出す座標だけに使います。
        """
        return _clip(point[0], -1.0, 1.0), _clip(point[1], -1.0, 1.0)

    visible = [t for t in tracks if world_valid and is_track_visible(t, viewport_wh)]
    classes = params["entity_classes"]
    effect_of = classes["effect"]
    by_kind = {kind: [t for t in visible if effect_of.get(t.class_name) == kind] for kind in EFFECT_KINDS}
    enemies = [raw(t) for t in visible if t.class_name in classes["enemy"]]
    gems = [raw(t) for t in visible if t.class_name in classes["gem"]]
    rare_gems = [raw(t) for t in visible if t.class_name in classes["rare_gem"]]
    weapons, passives = _hud_slots(hud_slots, params)
    out = _Planes(schema)
    bins = params["direction_bins"]

    # v1 から引き継ぐ segment
    if hp_ratio is not None:
        ensure(is_strict_number(hp_ratio) and 0.0 <= hp_ratio <= 1.0, "hp_ratio must be in [0,1]")
        out.put("player_hp", hp_ratio, True)
    if player_level is not None:
        ensure(type(player_level) is int and player_level >= 0, "player_level must be non-negative int")
        out.put("level", min(player_level / params["level_norm"], 1.0), True)
    out.put("player_screen_pos", [_clip((px - width / 2.0) / half, -1.0, 1.0), _clip((py - height / 2.0) / half, -1.0, 1.0)], world_valid)
    nearest_enemy = clip_xy(min(enemies, key=lambda p: p[0] ** 2 + p[1] ** 2)) if enemies else (0.0, 0.0)
    out.put("nearest_enemy_offset", nearest_enemy, world_valid)
    out.put("visible_enemy_count", min(len(enemies) / params["visible_enemy_count_norm"], 1.0), world_valid)
    if movement_direction is not None:
        ensure(isinstance(movement_direction, tuple) and len(movement_direction) == 2 and all(is_strict_number(v) and math.isfinite(float(v)) for v in movement_direction), "movement_direction must be a finite pair")
        out.put("movement_direction", [_clip(float(v), -1.0, 1.0) for v in movement_direction], True)
    if 0 in weapons:
        category = params["weapon_coarse_category"].get(weapons[0].type_name, "unknown")
        out.put("weapon_category", normalized_vocabulary_id(category, params["weapon_category_vocabulary"]), True)
    out.put("bias", 1.0, True)

    # 敵・ジェム・レアジェムの16方向特徴
    for prefix, points, band in (("enemy", enemies, "enemy"), ("gem", gems, "gem"), ("rare_gem", rare_gems, "gem")):
        nearest, near, mid = _direction_features(points, params["density"][band], bins)
        out.put(f"{prefix}_nearest_dist_16dir", nearest, world_valid)
        out.put(f"{prefix}_density_near_16dir", near, world_valid)
        out.put(f"{prefix}_density_mid_16dir", mid, world_valid)

    # 武器・パッシブのスロット
    for kind, slots, vocabulary, max_level in (
        ("weapon", weapons, params["weapon_vocabulary"], params["max_weapon_level"]),
        ("passive", passives, params["passive_vocabulary"], params["max_passive_level"]),
    ):
        count = params[f"max_{kind}_slots"]
        ids = [normalized_vocabulary_id(slots[i].type_name, vocabulary) if i in slots else 0.0 for i in range(count)]
        levels = [(slots[i].level or 0) / max_level if i in slots else 0.0 for i in range(count)]
        level_valid = [i in slots and (slots[i].type_name is None or slots[i].level is not None) for i in range(count)]
        out.put(f"{kind}_slot_ids", ids, [i in slots for i in range(count)])
        out.put(f"{kind}_slot_levels", levels, level_valid)

    slot_norm = float(params["max_weapon_slots"] - 1)
    emitters = {kind: _emitter(kind, weapons, params) for kind in ("aura", "orbit", "zone")}

    def slot_plane(emitter: _Emitter) -> tuple[float, bool]:
        """エフェクトを出す武器のスロット番号を正規化値と有効性で返す。

        出しうる武器が一意なら slot/(MaxWeaponSlots-1)、無いと確定なら neutral・有効、
        それ以外（複数・HUD 不完全）は不明として無効にします。
        """
        if emitter.slot is not None:
            return emitter.slot / slot_norm, True
        return 0.0, emitter.none_certain

    # aura: 半径は可視 aura の最大半径、見えないときは武器が無いと確定なら有効な neutral
    auras = by_kind["aura"]
    if auras:
        out.put("weapon_aura_radius", _clip(max(t.radius_px for t in auras) / half, 0.0, 1.0), True)
    else:
        out.put("weapon_aura_radius", 0.0, emitters["aura"].none_certain)
    out.put("weapon_aura_slot", *slot_plane(emitters["aura"]))

    # orbit: 周回半径は可視 orbit の距離の中央値、残り時間は最も古い初観測時刻から推定
    orbits = by_kind["orbit"]
    if orbits:
        out.put("weapon_orbit_radius", _clip(statistics.median(math.hypot(*raw(t)) for t in orbits), 0.0, 1.0), True)
        out.put("weapon_orbit_ttl", *_ttl(emitters["orbit"], min(t.first_seen_s for t in orbits), now_s, duration_mult, params))
    else:
        out.put("weapon_orbit_radius", 0.0, emitters["orbit"].none_certain)
        out.put("weapon_orbit_ttl", 0.0, emitters["orbit"].none_certain)
    out.put("weapon_orbit_slot", *slot_plane(emitters["orbit"]))

    # zone: 最寄り zone_slots 個を距離の近い順（同距離は track id 順）に並べる
    zone_count = params["zone_slots"]
    zones = sorted(by_kind["zone"], key=lambda t: (math.hypot(*raw(t)), t.track_id))[:zone_count]
    geometry, geometry_valid, slot_values, slot_valid, ttl_values, ttl_valid = [], [], [], [], [], []
    for j in range(zone_count):
        if j < len(zones):
            dx, dy = clip_xy(raw(zones[j]))
            geometry += [dx, dy, _clip(zones[j].radius_px / half, 0.0, 1.0)]
            geometry_valid += [True] * 3
            slot_value, slot_ok = slot_plane(emitters["zone"]) if emitters["zone"].slot is not None else (0.0, False)
            ttl_value, ttl_ok = _ttl(emitters["zone"], zones[j].first_seen_s, now_s, duration_mult, params)
        else:
            geometry += [0.0, 0.0, 0.0]
            geometry_valid += [world_valid] * 3
            slot_value, slot_ok = 0.0, world_valid or emitters["zone"].none_certain
            ttl_value, ttl_ok = 0.0, world_valid or emitters["zone"].none_certain
        slot_values.append(slot_value)
        slot_valid.append(slot_ok)
        ttl_values.append(ttl_value)
        ttl_valid.append(ttl_ok)
    out.put("weapon_zone_geometry", geometry, geometry_valid)
    out.put("weapon_zone_slot", slot_values, slot_valid)
    out.put("weapon_zone_ttl", ttl_values, ttl_valid)

    # projectile: 方向ビンごとの個数を正規化係数で割った密度
    density = [0.0] * bins
    for t in by_kind["projectile"]:
        dx, dy = raw(t)
        if math.hypot(dx, dy) > _KINDA_SMALL_NUMBER:
            density[directional_bin(dx, dy, bins)] += 1.0 / params["projectile_density_norm"]
    out.put("weapon_projectile_density_16dir", [_clip(v, 0.0, 1.0) for v in density], world_valid)

    age = np.where(out.validity > 0, 0.0, 1.0).astype(np.float32)
    observation = DeployObservation(out.values, out.validity, age, schema.schema_hash, int(round(float(now_s) * 1e9)), "release")
    observation.validate_for(schema)
    return observation


@lru_cache(maxsize=1)
def _default_v2_schema() -> DeployObsSchema:
    """v2 schema を1回だけ構築して共有する。

    schema は不変なので、毎フレームの hash 計算を避けるためにキャッシュします。
    """
    return DeployObsSchema.default_v2()
