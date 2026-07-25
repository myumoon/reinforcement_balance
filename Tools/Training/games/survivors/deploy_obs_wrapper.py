"""UE5 raw state を画面投影経由で DeployObsV1 に変換する Gym 互換 wrapper。

raw 配列の slice を避け、release と oracle diagnostic を別 constructor に分けて
実環境と同じ on-screen semantics を訓練側でも守ります。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence
import math

from survivors.deploy_obs_adapter import (
    NamedEstimate, assert_release_artifact_allowed as assert_adapter_release_artifact_allowed,
    build_deploy_observation, build_oracle_diagnostic_observation,
    normalized_category, release_policy_tensor, visible_track_estimates,
)
from reinbalance_survivors_contracts.deploy_obs import DeployObservation, DeployObsSchema
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


def _validate_raw(raw: Mapping[str, Any]) -> None:
    """raw world state と全 nested 値を利用前に厳密検証する。

    release で参照しない privileged 値も含め、型・有限性・範囲を入口で確定します。
    """
    ensure(isinstance(raw, Mapping) and set(raw) == _RAW_KEYS, "raw observation keys mismatch")
    ensure(isinstance(raw["timestamp_ns"], int) and not isinstance(raw["timestamp_ns"], bool) and raw["timestamp_ns"] >= 0, "invalid timestamp")
    viewport = raw["viewport"]
    ensure(isinstance(viewport, tuple) and len(viewport) == 2, "invalid viewport")
    ensure(
        all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in viewport),
        "viewport values must be positive int",
    )
    camera = _exact_mapping(raw["target_camera"], _CAMERA_KEYS, "target_camera")
    for key in ("center_x", "center_y"):
        _finite_number(camera[key], f"target_camera.{key}")
    for key in ("half_width", "half_height"):
        ensure(_finite_number(camera[key], f"target_camera.{key}") > 0, "camera extents must be positive")
    hud = _exact_mapping(raw["hud"], _HUD_KEYS, "hud")
    ensure(all(0 <= _finite_number(hud[key], f"hud.{key}") <= 1 for key in _HUD_KEYS), "hud value out of range")
    player = _exact_mapping(raw["player_world"], _WORLD_POINT_KEYS, "player_world")
    for key in _WORLD_POINT_KEYS:
        _finite_number(player[key], f"player_world.{key}")
    entities = raw["world_entities"]
    ensure(isinstance(entities, Sequence) and not isinstance(entities, (str, bytes)), "world_entities must be sequence")
    for entity in entities:
        row = _exact_mapping(entity, _ENTITY_KEYS, "world_entity")
        _finite_number(row["world_x"], "world_entity.world_x")
        _finite_number(row["world_y"], "world_entity.world_y")
        ensure(type(row["occluded"]) is bool, "world_entity.occluded must be bool")
        ensure(isinstance(row["timestamp_ns"], int) and not isinstance(row["timestamp_ns"], bool) and 0 <= row["timestamp_ns"] <= raw["timestamp_ns"], "invalid world_entity timestamp")
    temporal = _exact_mapping(raw["temporal"], _TEMPORAL_KEYS, "temporal")
    direction = temporal["movement_direction"]
    ensure(isinstance(direction, (list, tuple)) and len(direction) == 2, "movement_direction must be pair")
    ensure(all(-1 <= _finite_number(x, "movement_direction") <= 1 for x in direction), "movement_direction out of range")
    ensure(isinstance(temporal["timestamp_ns"], int) and not isinstance(temporal["timestamp_ns"], bool) and 0 <= temporal["timestamp_ns"] <= raw["timestamp_ns"], "invalid temporal timestamp")
    inventory = _exact_mapping(raw["inventory"], _INVENTORY_KEYS, "inventory")
    ensure(isinstance(inventory["weapon_category"], str), "weapon_category must be str")
    privileged = _exact_mapping(raw["privileged"], _PRIVILEGED_KEYS, "privileged")
    ensure(isinstance(privileged["player_pos"], (list, tuple)) and len(privileged["player_pos"]) == 2, "privileged.player_pos must be pair")
    for value in privileged["player_pos"]:
        _finite_number(value, "privileged.player_pos")
    for key in ("enemy_hp", "cooldown", "density"):
        ensure(0 <= _finite_number(privileged[key], f"privileged.{key}") <= 1, f"privileged.{key} out of range")
    ensure(isinstance(privileged["all_entity_count"], int) and not isinstance(privileged["all_entity_count"], bool) and privileged["all_entity_count"] >= 0, "invalid all_entity_count")


def _project_world(raw: Mapping[str, Any]) -> tuple[tuple[float, float] | None, list[dict[str, Any]]]:
    """target camera で world 座標を projection・visibility・clipping 判定する。

    同じ共有経路から player と敵 track の画面座標を作り、事前投影値を要求しません。
    """
    camera, viewport = raw["target_camera"], raw["viewport"]

    def project(x: float, y: float) -> tuple[float, float, bool]:
        nx = (x - camera["center_x"]) / camera["half_width"]
        ny = (y - camera["center_y"]) / camera["half_height"]
        inside = -1 <= nx <= 1 and -1 <= ny <= 1
        return (nx + 1) * viewport[0] / 2, (ny + 1) * viewport[1] / 2, not inside

    px, py, player_clipped = project(raw["player_world"]["x"], raw["player_world"]["y"])
    player_screen = None if player_clipped else (px, py)
    tracks = []
    for entity in raw["world_entities"]:
        x, y, clipped = project(entity["world_x"], entity["world_y"])
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
        """
        _validate_raw(raw)
        now, viewport = raw["timestamp_ns"], raw["viewport"]
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

    def reset(self, **kwargs: Any):
        """下位環境を reset し deploy tensor と info を返す。

        Gymnasium の戻り値形式を保ったまま observation だけを変換します。
        """
        raw, info = self.env.reset(**kwargs)
        return self.observation(raw), info

    def step(self, action: Any):
        """下位環境を step し deploy tensor を含む結果を返す。

        reward・終了フラグ・info は変更せず observation のみ変換します。
        """
        raw, reward, terminated, truncated, info = self.env.step(action)
        return self.observation(raw), reward, terminated, truncated, info


def fresh_vecnormalize(wrapper: DeployObsWrapper, factory: Any) -> Any:
    """deploy tensor の外側へ新規 VecNormalize を構築する。

    privileged source の統計を受け取らず、factory には wrapper だけを渡します。
    """
    ensure(isinstance(wrapper, DeployObsWrapper) and callable(factory), "invalid VecNormalize factory")
    return factory(wrapper, norm_obs=True, training=True)
