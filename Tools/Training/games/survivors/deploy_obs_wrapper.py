"""UE5 raw state を画面投影経由で DeployObs（v1 / v2）に変換する Gym 互換 wrapper。

raw 配列の slice を避け、release と oracle diagnostic を別 constructor に分けて
実環境と同じ on-screen semantics を訓練側でも守ります。
v2 schema では world 座標を px へ投影したあと Common の共有ビルダーへ渡すだけにし、
武器エフェクトの残り時間は sim の真値ではなく「初めて見えた時刻＋持続時間表」で推定します。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence
import math

from survivors.deploy_obs_adapter import (
    NamedEstimate, assert_release_artifact_allowed as assert_adapter_release_artifact_allowed,
    build_deploy_observation, build_oracle_diagnostic_observation,
    normalized_category, release_policy_tensor, visible_track_estimates,
)
from reinbalance_survivors_contracts.deploy_obs import DEPLOY_OBS_V2_SCHEMA_VERSION, DeployObservation, DeployObsSchema
from reinbalance_survivors_contracts.deploy_obs_v2_features import (
    HudSlot, TrackPx, build_deploy_obs_v2, is_track_visible, load_deploy_obs_v2_feature_params,
)
from reinbalance_survivors_contracts.ui_intent import ensure, is_strict_number

_RAW_KEYS = frozenset({"timestamp_ns", "viewport", "target_camera", "hud", "player_world", "world_entities", "temporal", "inventory", "privileged"})
_CONSTRUCTOR_TOKEN = object()
_HUD_KEYS = frozenset({"player_hp", "level"})
_TEMPORAL_KEYS = frozenset({"movement_direction", "timestamp_ns"})
_INVENTORY_KEYS = frozenset({"weapon_category"})
_PRIVILEGED_KEYS = frozenset({"player_pos", "enemy_hp", "cooldown", "all_entity_count", "density"})
_CAMERA_KEYS = frozenset({"center_x", "center_y", "half_width", "half_height"})
_WORLD_POINT_KEYS = frozenset({"x", "y"})
_ENTITY_KEYS = frozenset({"world_x", "world_y", "occluded", "timestamp_ns"})
# v2 raw 契約。privileged mapping は持たず、sim だけが知る値（slot・真の残り時間・warning）は
# entity の欄に置いて release では読まない。
_V2_RAW_KEYS = _RAW_KEYS - {"privileged"}
_V2_HUD_KEYS = frozenset({"player_hp", "level"})
_V2_INVENTORY_KEYS = frozenset({"weapon_slots", "passive_slots", "duration_mult"})
_V2_SLOT_KEYS = frozenset({"index", "type_name", "level"})
_V2_ENTITY_KEYS = _ENTITY_KEYS | {"entity_id", "class_name", "radius_world", "slot", "ttl_true_s", "warning"}


def _finite_number(value: Any, label: str) -> float:
    """有限な実数を暗黙変換なしで検証する。

    bool・文字列・NaN・Inf を nested 入力の未使用箇所でも入口で拒否します。
    """
    ensure(is_strict_number(value) and math.isfinite(float(value)), f"{label} must be finite number")
    return float(value)


def _exact_mapping(value: Any, keys: frozenset[str], label: str) -> Mapping[str, Any]:
    """nested mapping の未知キーと欠落キーを対称に拒否する。

    parser typo や将来項目を黙認せず、全 raw 型を同じ fail-closed 規則で扱います。
    """
    ensure(isinstance(value, Mapping) and set(value) == keys, f"{label} keys mismatch")
    return value


def _validated_viewport(viewport: Any) -> tuple[int, int]:
    """viewport の wire 表現を検証して不変 tuple へ正規化する。

    Python 内の tuple と JSON decode 後の list を等しく受理し、
    文字列・bool・非整数・非正寸法は全 wrapper 経路で拒否します。
    """
    ensure(isinstance(viewport, (tuple, list)) and len(viewport) == 2, "invalid viewport")
    ensure(
        all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in viewport),
        "viewport values must be positive int",
    )
    return viewport[0], viewport[1]


def _validate_frame(raw: Mapping[str, Any], keys: frozenset[str]) -> None:
    """v1 / v2 共通の raw 項目（時刻・viewport・camera・自機・temporal）を検証する。

    トップレベルのキー集合は版ごとに違うので引数で受け取り、
    共通項目の型・有限性・範囲の規則を両方の版へ同じように適用します。
    """
    ensure(isinstance(raw, Mapping) and set(raw) == keys, "raw observation keys mismatch")
    ensure(isinstance(raw["timestamp_ns"], int) and not isinstance(raw["timestamp_ns"], bool) and raw["timestamp_ns"] >= 0, "invalid timestamp")
    _validated_viewport(raw["viewport"])
    camera = _exact_mapping(raw["target_camera"], _CAMERA_KEYS, "target_camera")
    for key in ("center_x", "center_y"):
        _finite_number(camera[key], f"target_camera.{key}")
    for key in ("half_width", "half_height"):
        ensure(_finite_number(camera[key], f"target_camera.{key}") > 0, "camera extents must be positive")
    player = _exact_mapping(raw["player_world"], _WORLD_POINT_KEYS, "player_world")
    for key in _WORLD_POINT_KEYS:
        _finite_number(player[key], f"player_world.{key}")
    temporal = _exact_mapping(raw["temporal"], _TEMPORAL_KEYS, "temporal")
    direction = temporal["movement_direction"]
    ensure(isinstance(direction, (list, tuple)) and len(direction) == 2, "movement_direction must be pair")
    ensure(all(-1 <= _finite_number(x, "movement_direction") <= 1 for x in direction), "movement_direction out of range")
    ensure(isinstance(temporal["timestamp_ns"], int) and not isinstance(temporal["timestamp_ns"], bool) and 0 <= temporal["timestamp_ns"] <= raw["timestamp_ns"], "invalid temporal timestamp")


def _validate_raw(raw: Mapping[str, Any]) -> None:
    """raw world state と全 nested 値を利用前に厳密検証する。

    release で参照しない privileged 値も含め、型・有限性・範囲を入口で確定します。
    """
    _validate_frame(raw, _RAW_KEYS)
    hud = _exact_mapping(raw["hud"], _HUD_KEYS, "hud")
    ensure(all(0 <= _finite_number(hud[key], f"hud.{key}") <= 1 for key in _HUD_KEYS), "hud value out of range")
    entities = raw["world_entities"]
    ensure(isinstance(entities, Sequence) and not isinstance(entities, (str, bytes)), "world_entities must be sequence")
    for entity in entities:
        row = _exact_mapping(entity, _ENTITY_KEYS, "world_entity")
        _finite_number(row["world_x"], "world_entity.world_x")
        _finite_number(row["world_y"], "world_entity.world_y")
        ensure(type(row["occluded"]) is bool, "world_entity.occluded must be bool")
        ensure(isinstance(row["timestamp_ns"], int) and not isinstance(row["timestamp_ns"], bool) and 0 <= row["timestamp_ns"] <= raw["timestamp_ns"], "invalid world_entity timestamp")
    inventory = _exact_mapping(raw["inventory"], _INVENTORY_KEYS, "inventory")
    ensure(isinstance(inventory["weapon_category"], str), "weapon_category must be str")
    privileged = _exact_mapping(raw["privileged"], _PRIVILEGED_KEYS, "privileged")
    ensure(isinstance(privileged["player_pos"], (list, tuple)) and len(privileged["player_pos"]) == 2, "privileged.player_pos must be pair")
    for value in privileged["player_pos"]:
        _finite_number(value, "privileged.player_pos")
    for key in ("enemy_hp", "cooldown", "density"):
        ensure(0 <= _finite_number(privileged[key], f"privileged.{key}") <= 1, f"privileged.{key} out of range")
    ensure(isinstance(privileged["all_entity_count"], int) and not isinstance(privileged["all_entity_count"], bool) and privileged["all_entity_count"] >= 0, "invalid all_entity_count")


def _inventory_hud_slots(inventory: Mapping[str, Any], params: Mapping[str, Any]) -> list[HudSlot]:
    """v2 inventory の武器・パッシブ6枠ずつを検証して HudSlot 列へ変換する。

    枠数・番号の並び・語彙・レベル上限を確かめ、空き枠（type_name も level も None）も含めて返します。
    sim は全枠を知っているので、どの枠も「読めなかった」扱いにはしません。
    """
    slots = []
    for kind in ("weapon", "passive"):
        rows = inventory[f"{kind}_slots"]
        vocabulary = params[f"{kind}_vocabulary"]
        ensure(isinstance(rows, (list, tuple)) and len(rows) == params[f"max_{kind}_slots"], f"inventory.{kind}_slots length mismatch")
        for index, entry in enumerate(rows):
            row = _exact_mapping(entry, _V2_SLOT_KEYS, f"inventory.{kind}_slots")
            ensure(type(row["index"]) is int and row["index"] == index, f"inventory.{kind}_slots index mismatch")
            ensure(row["type_name"] is None or (isinstance(row["type_name"], str) and row["type_name"] in vocabulary[1:]), f"unknown inventory.{kind}_slots type_name")
            ensure(row["level"] is None or (type(row["level"]) is int and 1 <= row["level"] <= params[f"max_{kind}_level"]), f"inventory.{kind}_slots level out of range")
            ensure((row["type_name"] is None) == (row["level"] is None), f"inventory.{kind}_slots type/level mismatch")
            slots.append(HudSlot(kind, index, row["type_name"], row["level"]))
    return slots


def _validate_raw_v2(raw: Mapping[str, Any]) -> list[HudSlot]:
    """v2 raw state を厳密検証し、inventory を HudSlot 列にして返す。

    entity の id 重複・Common 語彙外の class 名・武器エフェクト以外が slot や残り時間を持つことを拒否します。
    release で読まない欄（slot・ttl_true_s・warning）も入口で型と範囲を確かめます。
    """
    _validate_frame(raw, _V2_RAW_KEYS)
    params = load_deploy_obs_v2_feature_params()
    classes = params["entity_classes"]
    effect_classes = classes["effect"]
    vocabulary = set(classes["enemy"]) | set(classes["gem"]) | set(effect_classes)
    hud = _exact_mapping(raw["hud"], _V2_HUD_KEYS, "hud")
    ensure(0 <= _finite_number(hud["player_hp"], "hud.player_hp") <= 1, "hud.player_hp out of range")
    ensure(type(hud["level"]) is int and hud["level"] >= 0, "hud.level must be non-negative int")
    entities = raw["world_entities"]
    ensure(isinstance(entities, (list, tuple)), "world_entities must be list or tuple")
    seen_ids = set()
    for entity in entities:
        row = _exact_mapping(entity, _V2_ENTITY_KEYS, "world_entity")
        _finite_number(row["world_x"], "world_entity.world_x")
        _finite_number(row["world_y"], "world_entity.world_y")
        ensure(_finite_number(row["radius_world"], "world_entity.radius_world") >= 0, "world_entity.radius_world must be non-negative")
        ensure(type(row["occluded"]) is bool and type(row["warning"]) is bool, "world_entity occluded/warning must be bool")
        ensure(type(row["timestamp_ns"]) is int and 0 <= row["timestamp_ns"] <= raw["timestamp_ns"], "invalid world_entity timestamp")
        ensure(type(row["entity_id"]) is int and row["entity_id"] >= 0 and row["entity_id"] not in seen_ids, "world_entity.entity_id must be unique non-negative int")
        seen_ids.add(row["entity_id"])
        ensure(row["class_name"] in vocabulary, "unknown world_entity.class_name")
        if row["class_name"] in effect_classes:
            ensure(type(row["slot"]) is int and 0 <= row["slot"] < params["max_weapon_slots"], "weapon effect slot out of range")
            ensure(_finite_number(row["ttl_true_s"], "world_entity.ttl_true_s") >= 0, "world_entity.ttl_true_s must be non-negative")
        else:
            ensure(row["slot"] is None and row["ttl_true_s"] is None and row["warning"] is False, "only weapon effects carry slot/ttl/warning")
    inventory = _exact_mapping(raw["inventory"], _V2_INVENTORY_KEYS, "inventory")
    ensure(_finite_number(inventory["duration_mult"], "inventory.duration_mult") > 0, "inventory.duration_mult must be positive")
    return _inventory_hud_slots(inventory, params)


def _project_point(raw: Mapping[str, Any], x: float, y: float) -> tuple[float, float, bool]:
    """target camera で world 座標を画面 px へ投影し、画面外かどうかも返す。

    camera 中心からの差を半幅・半高で割って [-1,1] にし、viewport の px へ広げます（y は反転しない）。
    v1 の visibility 判定と v2 の Common ビルダー入力が同じ投影式を使います。
    """
    camera, viewport = raw["target_camera"], _validated_viewport(raw["viewport"])
    nx = (x - camera["center_x"]) / camera["half_width"]
    ny = (y - camera["center_y"]) / camera["half_height"]
    inside = -1 <= nx <= 1 and -1 <= ny <= 1
    return (nx + 1) * viewport[0] / 2, (ny + 1) * viewport[1] / 2, not inside


def _project_world(raw: Mapping[str, Any]) -> tuple[tuple[float, float] | None, list[dict[str, Any]]]:
    """target camera で world 座標を projection・visibility・clipping 判定する。

    同じ共有経路から player と敵 track の画面座標を作り、事前投影値を要求しません。
    """
    px, py, player_clipped = _project_point(raw, raw["player_world"]["x"], raw["player_world"]["y"])
    player_screen = None if player_clipped else (px, py)
    tracks = []
    for entity in raw["world_entities"]:
        x, y, clipped = _project_point(raw, entity["world_x"], entity["world_y"])
        tracks.append({
            "screen_x": x, "screen_y": y, "visible": not clipped and not entity["occluded"],
            "occluded": entity["occluded"], "clipped": clipped,
            "timestamp_ns": entity["timestamp_ns"],
        })
    return player_screen, tracks


class DeployObsWrapper:
    """環境 observation を deploy tensor に置換する軽量 wrapper。

    Gymnasium/SB3 を import せず reset/step を委譲するため、契約 test は単独実行できます。
    """

    def __init__(self, env: Any, schema: DeployObsSchema, mode: str, *, _token: object | None = None) -> None:
        """検証済み環境・schema・mode を保持する。

        直接構築を禁止し、release/oracle の明示 constructor からだけ作ります。
        """
        ensure(_token is _CONSTRUCTOR_TOKEN, "use release/oracle_diagnostic constructor")
        ensure(mode in {"release", "oracle_diagnostic"}, "invalid deploy observation mode")
        self.env, self.schema, self.mode = env, schema, mode
        self.run_manifest = {"deploy_obs_mode": mode, "deploy_obs_schema_hash": schema.schema_hash, "vecnormalize": "fresh_outside_deploy_tensor"}
        # v2 の武器エフェクト追跡: entity_id → (初めて見えた時刻 s, 最後に見えたフレーム番号, class 名)
        self._tracks: dict[int, tuple[float, int, str]] = {}
        self._frame = 0
        # oracle_diagnostic（v2）の直近フレームの残り時間誤差: entity_id → 推定 − 真値（秒）
        self.last_ttl_error_s: dict[int, float] | None = None

    @classmethod
    def release(cls, env: Any, schema: DeployObsSchema) -> "DeployObsWrapper":
        """実 parser と同じ screen-space semantics の wrapper を作る。

        release artifact を生成できる唯一の mode で、privileged state は読みません。
        """
        return cls(env, schema, "release", _token=_CONSTRUCTOR_TOKEN)

    @classmethod
    def oracle_diagnostic(cls, env: Any, schema: DeployObsSchema) -> "DeployObsWrapper":
        """全 state を比較診断に使える oracle wrapper を作る。

        学習調査専用であり、release artifact の保存は明示的に禁止されます。
        """
        return cls(env, schema, "oracle_diagnostic", _token=_CONSTRUCTOR_TOKEN)

    @property
    def release_artifact_allowed(self) -> bool:
        """現在の mode が release artifact を生成可能か返す。

        oracle の結果を本番成果物と誤認しないための単純な gate です。
        """
        return self.mode == "release"

    def assert_release_artifact_allowed(
        self,
        observation: DeployObservation | None = None,
    ) -> None:
        """oracle mode の artifact 出力を fail-closed で拒否する。

        wrapper mode と observation 自身の provenance を保存直前に確認し、
        release wrapper へ渡された oracle 診断値も拒否します。
        """
        ensure(self.release_artifact_allowed, "oracle_diagnostic cannot create release artifacts")
        if observation is not None:
            assert_adapter_release_artifact_allowed(observation, self.schema)

    def observation(self, raw: Mapping[str, Any]):
        """raw state を投影・可視性判定・named estimates 経由で tensor 化する。

        release 経路では privileged mapping を一切参照しません。
        v2 schema のときは v2 raw 契約で受け取り、Common の共有ビルダーへ委譲します。
        """
        if self.schema.schema_version == DEPLOY_OBS_V2_SCHEMA_VERSION:
            return self._observation_v2(raw)
        _validate_raw(raw)
        now, viewport = raw["timestamp_ns"], _validated_viewport(raw["viewport"])
        player_screen, tracks = _project_world(raw)
        estimates = visible_track_estimates(tracks, tuple(viewport), now)
        for name in ("player_hp", "level"):
            if name in raw["hud"]:
                estimates[name] = NamedEstimate((raw["hud"][name],), now)
        if player_screen is not None:
            from survivors.deploy_obs_adapter import screen_to_centered
            estimates["player_screen_pos"] = NamedEstimate(screen_to_centered(*player_screen, *viewport), now)
        estimates["movement_direction"] = NamedEstimate(tuple(raw["temporal"]["movement_direction"]), raw["temporal"]["timestamp_ns"])
        estimates["weapon_category"] = NamedEstimate((normalized_category(raw["inventory"]["weapon_category"]),), now)
        if self.mode == "oracle_diagnostic":
            for name in ("enemy_hp", "cooldown"):
                if name in raw["privileged"]:
                    estimates[name] = NamedEstimate((raw["privileged"][name],), now)
        if self.mode == "oracle_diagnostic":
            observation = build_oracle_diagnostic_observation(self.schema, estimates, now)
        else:
            observation = build_deploy_observation(self.schema, estimates, now)
        observation.validate_for(self.schema)
        if self.mode == "release":
            self.assert_release_artifact_allowed(observation)
            return release_policy_tensor(observation, self.schema)
        return observation.as_policy_tensor(self.schema)

    def _observation_v2(self, raw: Mapping[str, Any]):
        """v2 raw を px へ投影し、初観測時刻を付けて Common ビルダーで tensor 化する。

        武器エフェクトは見えたフレームで初観測時刻を記録し、class ごとの max_age フレーム
        続けて見えなければ記録を捨てます（実機 tracker が track を作り直すのと同じ規則）。
        release は entity の slot・真の残り時間・warning を読まず、oracle だけが残り時間の誤差を出します。
        """
        hud_slots = _validate_raw_v2(raw)
        params = load_deploy_obs_v2_feature_params()
        max_age = params["track_max_age_frames"]
        viewport, camera = _validated_viewport(raw["viewport"]), raw["target_camera"]
        # v2 は縦横とも W/2 で正規化する等方座標なので、viewport と camera の縦横比が違うと y 方向が黙って歪む
        ensure(
            math.isclose(viewport[0] * camera["half_height"], viewport[1] * camera["half_width"], rel_tol=1e-3),
            "v2 viewport aspect must match target_camera half_width/half_height",
        )
        now_s = raw["timestamp_ns"] / 1e9
        px_per_world = viewport[0] / (2.0 * camera["half_width"])
        self._frame += 1
        tracks = []
        for entity in raw["world_entities"]:
            cx, cy, _ = _project_point(raw, entity["world_x"], entity["world_y"])
            entity_id, class_name = entity["entity_id"], entity["class_name"]
            track = TrackPx(class_name, cx, cy, float(entity["radius_world"]) * px_per_world, entity_id, now_s, entity["occluded"])
            if class_name in max_age and is_track_visible(track, viewport):
                first_seen = self._tracks.get(entity_id, (now_s, 0, class_name))[0]
                self._tracks[entity_id] = (first_seen, self._frame, class_name)
                track = TrackPx(class_name, cx, cy, track.radius_px, entity_id, first_seen, entity["occluded"])
            tracks.append(track)
        self._tracks = {
            entity_id: row for entity_id, row in self._tracks.items()
            if self._frame - row[1] < max_age[row[2]]
        }
        player_x, player_y, _ = _project_point(raw, raw["player_world"]["x"], raw["player_world"]["y"])
        observation = build_deploy_obs_v2(
            viewport_wh=viewport, player_px=(player_x, player_y), tracks=tracks, hud_slots=hud_slots,
            now_s=now_s, duration_mult=float(raw["inventory"]["duration_mult"]), world_valid=True,
            hp_ratio=float(raw["hud"]["player_hp"]), player_level=raw["hud"]["level"],
            movement_direction=tuple(float(v) for v in raw["temporal"]["movement_direction"]),
        )
        if self.mode == "release":
            self.assert_release_artifact_allowed()
            ensure(observation.provenance == "release", "release wrapper requires release provenance")
            return observation.as_policy_tensor(self.schema)
        self.last_ttl_error_s = self._ttl_errors(raw, observation, tracks, viewport, (player_x, player_y), params)
        oracle = DeployObservation(observation.values, observation.validity, observation.age, observation.schema_hash, observation.timestamp_ns, "oracle_diagnostic")
        return oracle.as_policy_tensor(self.schema)

    def _ttl_errors(
        self, raw: Mapping[str, Any], observation: DeployObservation, tracks: Sequence[TrackPx],
        viewport: tuple[int, int], player_px: tuple[float, float], params: Mapping[str, Any],
    ) -> dict[int, float]:
        """oracle 専用: release ビルダーが実際に出した zone / orbit の残り時間と sim の真の残り時間の差を返す。

        推定式は再実装せず、ビルダー出力の weapon_orbit_ttl / weapon_zone_ttl（正規化値）を秒へ戻して使います。
        各値がどの entity のものかは、ビルダーと同じ並び（orbit は最古の初観測、zone はプレイヤーからの距離 → id 順）で対応付けます。
        ビルダーが無効にした値（emitter が一意に決まらない等）は誤差に含めず、真値はビルダーと同じく 0..MaxProjectileObsTtl に clip します。
        """
        max_ttl = float(params["max_projectile_obs_ttl_s"])
        effect_of = params["entity_classes"]["effect"]
        half = viewport[0] / 2.0
        true_ttl = {e["entity_id"]: e["ttl_true_s"] for e in raw["world_entities"]}
        visible = [t for t in tracks if is_track_visible(t, viewport)]
        orbits = sorted((t for t in visible if effect_of.get(t.class_name) == "orbit"), key=lambda t: (t.first_seen_s, t.track_id))[:1]
        zones = sorted(
            (t for t in visible if effect_of.get(t.class_name) == "zone"),
            key=lambda t: (math.hypot((t.cx_px - player_px[0]) / half, (t.cy_px - player_px[1]) / half), t.track_id),
        )
        errors = {}
        for name, picked in (("weapon_orbit_ttl", orbits), ("weapon_zone_ttl", zones)):
            offset, size = self.schema.layout[name]
            for j, track in enumerate(picked[:size]):
                truth = true_ttl.get(track.track_id)
                if observation.validity[offset + j] > 0 and truth is not None:
                    errors[track.track_id] = float(observation.values[offset + j]) * max_ttl - min(max(float(truth), 0.0), max_ttl)
        return errors

    def _with_ttl_error(self, info: Any) -> Any:
        """oracle（v2）のときだけ info へ残り時間誤差を足した新しい dict を返す。

        release や v1 では info をそのまま返し、下位環境の info は書き換えません。
        """
        if self.mode != "oracle_diagnostic" or self.last_ttl_error_s is None:
            return info
        return {**info, "deploy_ttl_error_s": dict(self.last_ttl_error_s)}

    def reset(self, **kwargs: Any):
        """下位環境を reset し deploy tensor と info を返す。

        Gymnasium の戻り値形式を保ったまま observation だけを変換します。
        v2 の初観測時刻の記録は episode をまたがないよう全て捨てます。
        """
        raw, info = self.env.reset(**kwargs)
        self._tracks, self._frame, self.last_ttl_error_s = {}, 0, None
        return self.observation(raw), self._with_ttl_error(info)

    def step(self, action: Any):
        """下位環境を step し deploy tensor を含む結果を返す。

        reward・終了フラグは変更せず observation のみ変換します（oracle v2 は info に残り時間誤差を足す）。
        """
        raw, reward, terminated, truncated, info = self.env.step(action)
        return self.observation(raw), reward, terminated, truncated, self._with_ttl_error(info)


def fresh_vecnormalize(wrapper: DeployObsWrapper, factory: Any) -> Any:
    """deploy tensor の外側へ新規 VecNormalize を構築する。

    privileged source の統計を受け取らず、factory には wrapper だけを渡します。
    """
    ensure(isinstance(wrapper, DeployObsWrapper) and callable(factory), "invalid VecNormalize factory")
    return factory(wrapper, norm_obs=True, training=True)
