"""UE5 HTTP 応答の deploy_raw を DeployObs v2 wrapper の raw dict へ変換する環境 adapter。

SurvivorsEnv を包み、reset 前に /params で deploy_raw を有効化して、
/reset 応答のトップレベルと /step 応答の info にある deploy_raw を読み取ります。
JSON は未知キー・欠損キー・非数・型違いをすべて拒否し（fail-closed）、
DeployObsWrapper.release() / oracle_diagnostic() にそのまま渡せる v2 raw dict を返します。
"""

from __future__ import annotations

from typing import Any, Mapping
import math

from games.survivors.deploy_obs_wrapper import _exact_mapping, _finite_number
from reinbalance_survivors_contracts.deploy_obs_v2_features import load_deploy_obs_v2_feature_params
from reinbalance_survivors_contracts.ui_intent import ensure

DEPLOY_RAW_SCHEMA_VERSION = "survivors_deploy_raw.v1"
_TOP_KEYS = frozenset({"schema_version", "elapsed_s", "camera", "player", "duration_mult", "weapon_slots", "passive_slots", "entities"})
_CAMERA_KEYS = frozenset({"center_x", "center_y", "half_width", "half_height", "cull_margin"})
_PLAYER_KEYS = frozenset({"world_x", "world_y", "hp_ratio", "level"})
_SLOT_KEYS = frozenset({"index", "type_id", "level"})
_ENTITY_KEYS = frozenset({"entity_id", "class_name", "world_x", "world_y", "radius_world", "slot", "ttl_true_s", "warning"})


def _non_negative_int(value: Any, label: str) -> int:
    """bool を除く 0 以上の int であることを確かめて返す。

    JSON の true/false は Python で int の仲間になるため、type で厳密に判定します。
    """
    ensure(type(value) is int and value >= 0, f"{label} must be non-negative int")
    return value


def _slots(rows: Any, kind: str, params: Mapping[str, Any]) -> list[dict[str, Any]]:
    """武器 / パッシブ 6 枠の type_id を Common 語彙の名前へ変換する。

    type_id は C++ enum 値で、Common 語彙は同じ並び（parity テストで保証）なので添字で引きます。
    0 は空き枠（名前もレベルも None）、語彙の範囲外は unknown にします。
    """
    vocabulary = params[f"{kind}_vocabulary"]
    ensure(isinstance(rows, list) and len(rows) == params[f"max_{kind}_slots"], f"deploy_raw.{kind}_slots length mismatch")
    out = []
    for index, entry in enumerate(rows):
        row = _exact_mapping(entry, _SLOT_KEYS, f"deploy_raw.{kind}_slots")
        ensure(_non_negative_int(row["index"], "slot index") == index, f"deploy_raw.{kind}_slots index mismatch")
        type_id = _non_negative_int(row["type_id"], "slot type_id")
        level = _non_negative_int(row["level"], "slot level")
        ensure((type_id == 0) == (level == 0), f"deploy_raw.{kind}_slots empty slot must have level 0")
        if type_id == 0:
            out.append({"index": index, "type_name": None, "level": None})
        else:
            name = vocabulary[type_id] if type_id < len(vocabulary) - 1 else "unknown"
            out.append({"index": index, "type_name": name, "level": level})
    return out


def deploy_raw_to_raw(payload: Any, viewport: tuple[int, int], movement_direction: tuple[float, float]) -> dict[str, Any]:
    """HTTP の deploy_raw オブジェクト 1 つを v2 raw dict へ変換する。

    camera の中心・半幅・半高をそのまま target camera に使い、時刻は elapsed_s を ns にしたものを全欄で共有します。
    sim に遮蔽は無いので occluded は常に False です。cull_margin は検証だけして捨てます（可視判定は wrapper 側）。
    """
    data = _exact_mapping(payload, _TOP_KEYS, "deploy_raw")
    ensure(data["schema_version"] == DEPLOY_RAW_SCHEMA_VERSION, "unsupported deploy_raw schema_version")
    elapsed_s = _finite_number(data["elapsed_s"], "deploy_raw.elapsed_s")
    ensure(elapsed_s >= 0, "deploy_raw.elapsed_s must be non-negative")
    timestamp_ns = int(round(elapsed_s * 1e9))
    camera = _exact_mapping(data["camera"], _CAMERA_KEYS, "deploy_raw.camera")
    for key in _CAMERA_KEYS:
        _finite_number(camera[key], f"deploy_raw.camera.{key}")
    ensure(camera["half_width"] > 0 and camera["half_height"] > 0 and camera["cull_margin"] >= 0, "deploy_raw.camera extents invalid")
    player = _exact_mapping(data["player"], _PLAYER_KEYS, "deploy_raw.player")
    for key in ("world_x", "world_y", "hp_ratio"):
        _finite_number(player[key], f"deploy_raw.player.{key}")
    ensure(0 <= player["hp_ratio"] <= 1, "deploy_raw.player.hp_ratio out of range")
    ensure(_finite_number(data["duration_mult"], "deploy_raw.duration_mult") > 0, "deploy_raw.duration_mult must be positive")
    params = load_deploy_obs_v2_feature_params()
    classes = params["entity_classes"]
    vocabulary = set(classes["enemy"]) | set(classes["gem"]) | set(classes["effect"])
    entities = data["entities"]
    ensure(isinstance(entities, list), "deploy_raw.entities must be list")
    world_entities = []
    for entry in entities:
        row = _exact_mapping(entry, _ENTITY_KEYS, "deploy_raw.entity")
        ensure(isinstance(row["class_name"], str) and row["class_name"] in vocabulary, "unknown deploy_raw.entity.class_name")
        ensure(row["slot"] is None or type(row["slot"]) is int, "deploy_raw.entity.slot must be int or null")
        ensure(row["ttl_true_s"] is None or _finite_number(row["ttl_true_s"], "deploy_raw.entity.ttl_true_s") >= 0, "deploy_raw.entity.ttl_true_s invalid")
        ensure(type(row["warning"]) is bool, "deploy_raw.entity.warning must be bool")
        world_entities.append({
            "entity_id": _non_negative_int(row["entity_id"], "deploy_raw.entity.entity_id"),
            "class_name": row["class_name"],
            "world_x": _finite_number(row["world_x"], "deploy_raw.entity.world_x"),
            "world_y": _finite_number(row["world_y"], "deploy_raw.entity.world_y"),
            "radius_world": _finite_number(row["radius_world"], "deploy_raw.entity.radius_world"),
            "occluded": False,
            "timestamp_ns": timestamp_ns,
            "slot": row["slot"],
            "ttl_true_s": None if row["ttl_true_s"] is None else float(row["ttl_true_s"]),
            "warning": row["warning"],
        })
    return {
        "timestamp_ns": timestamp_ns,
        "viewport": viewport,
        "target_camera": {key: float(camera[key]) for key in ("center_x", "center_y", "half_width", "half_height")},
        "hud": {"player_hp": float(player["hp_ratio"]), "level": _non_negative_int(player["level"], "deploy_raw.player.level")},
        "player_world": {"x": float(player["world_x"]), "y": float(player["world_y"])},
        "world_entities": world_entities,
        "temporal": {"movement_direction": movement_direction, "timestamp_ns": timestamp_ns},
        "inventory": {
            "weapon_slots": _slots(data["weapon_slots"], "weapon", params),
            "passive_slots": _slots(data["passive_slots"], "passive", params),
            "duration_mult": float(data["duration_mult"]),
        },
    }


class DeployRawEnv:
    """SurvivorsEnv の HTTP 応答を v2 raw dict の観測に置き換える環境 adapter。

    reset のたびに /params で deploy_raw を有効化してから reset し（UE5 側は reset で解除しないが再起動に備える）、
    flat obs の代わりに raw dict を返します。移動方向は自機の世界座標の前フレームとの差から作ります。
    """

    def __init__(self, env: Any, viewport: tuple[int, int] = (1920, 1080)) -> None:
        """包む SurvivorsEnv と、投影に使う仮想 viewport（px）を保持する。

        viewport は sim カメラの 16:9（800u × 450u）と同じ縦横比にする必要があります。
        比率が違うと縦横の縮尺がずれるため、v2 wrapper が観測時に拒否します（fail-closed）。
        """
        ensure(isinstance(viewport, tuple) and len(viewport) == 2 and all(type(v) is int and v > 0 for v in viewport), "viewport must be a positive int pair")
        self.env, self.viewport = env, viewport
        self._last_player: tuple[float, float] | None = None
        # 直近の reset/step 応答の flat obs（同じ応答の deploy_raw と対になる。蒸留収集の教師入力に使う）
        self.last_flat_obs: Any = None

    def _convert(self, payload: Any) -> dict[str, Any]:
        """deploy_raw を raw dict にし、前フレームとの位置差から移動方向を入れる。

        reset 直後や動いていないフレームは (0, 0) にします。
        """
        raw = deploy_raw_to_raw(payload, self.viewport, (0.0, 0.0))
        position = (raw["player_world"]["x"], raw["player_world"]["y"])
        if self._last_player is not None:
            dx, dy = position[0] - self._last_player[0], position[1] - self._last_player[1]
            length = math.hypot(dx, dy)
            if length > 1e-6:
                raw["temporal"]["movement_direction"] = (dx / length, dy / length)
        self._last_player = position
        return raw

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """deploy_raw を有効化して reset し、(raw dict, info) を返す。

        /params の失敗や deploy_raw の欠落は黙って flat obs に戻さず例外にします。
        """
        ensure(self.env.set_params(deploy_raw=True) is True, "failed to enable deploy_raw via /params")
        obs, info = self.env.reset(seed=seed, options=options)
        response = self.env.last_reset_response
        ensure(isinstance(response, Mapping) and "deploy_raw" in response, "reset response has no deploy_raw")
        self._last_player, self.last_flat_obs = None, obs
        return self._convert(response["deploy_raw"]), info

    def step(self, action: Any):
        """step して info の deploy_raw を raw dict にし、(raw, reward, terminated, truncated, info) を返す。

        返す info からは deploy_raw を取り除きます（観測として返すため）。
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        ensure(isinstance(info, Mapping) and "deploy_raw" in info, "step info has no deploy_raw")
        self.last_flat_obs = obs
        rest = {key: value for key, value in info.items() if key != "deploy_raw"}
        return self._convert(info["deploy_raw"]), reward, terminated, truncated, rest
