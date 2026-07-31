"""DeployObsV1 の順序・範囲・欠損表現を固定する共有契約。

画面から得られる情報を Training と Deployment が同じ並びの
value・validity・age に変換できるよう、検証とハッシュ計算を一か所に集めます。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
import math

import numpy as np

from .canonical_json import canonical_hash
from .ui_intent import ContractValidationError, ensure, is_strict_number

DEPLOY_OBS_SCHEMA_VERSION = "deploy_obs.v1"
OBSERVATION_PROVENANCE = frozenset({"release", "oracle_diagnostic"})
SOURCE_CLASSES = frozenset(
    {"hud_inventory", "screen_world_observed", "temporal_inferred", "constant", "unobservable"}
)
DEPLOY_OBS_V1_SEGMENTS = (
    "player_hp", "level", "player_screen_pos", "nearest_enemy_offset",
    "visible_enemy_count", "movement_direction", "weapon_category", "bias",
    "enemy_hp", "cooldown",
)
_FIELD_KEYS = frozenset(
    {"name", "size", "source_class", "minimum", "maximum", "neutral", "max_age_ms", "stale_after_ms"}
)
_SCHEMA_KEYS = frozenset({"schema_version", "fields"})


@dataclass(frozen=True)
class DeployObsField:
    """連続する特徴 segment の契約。

    名前と幅に加え、取得元・許容範囲・欠損時の値・鮮度上限を保持します。
    """

    name: str
    size: int
    source_class: str
    minimum: float
    maximum: float
    neutral: float
    max_age_ms: float
    stale_after_ms: float

    def __post_init__(self) -> None:
        """field の型・範囲・鮮度設定を fail-closed 検証する。

        不正な設定が観測生成まで進まないよう、生成時点で全条件を確認します。
        """
        ensure(isinstance(self.name, str) and bool(self.name), "field name must be non-empty str")
        ensure(isinstance(self.size, int) and not isinstance(self.size, bool) and self.size > 0, "field size must be positive int")
        ensure(self.source_class in SOURCE_CLASSES, "unknown source class")
        for value in (self.minimum, self.maximum, self.neutral, self.max_age_ms, self.stale_after_ms):
            ensure(is_strict_number(value) and math.isfinite(float(value)), "field numbers must be finite")
        ensure(self.minimum < self.maximum, "field range must be increasing")
        ensure(self.minimum <= self.neutral <= self.maximum, "neutral out of range")
        ensure(self.max_age_ms > 0 and 0 <= self.stale_after_ms <= self.max_age_ms, "invalid age thresholds")

    def to_wire(self) -> dict[str, Any]:
        """canonical hash に含める wire 表現を返す。

        schema identity に必要な設定だけを安定した辞書へ変換します。
        """
        return {
            "name": self.name, "size": self.size, "source_class": self.source_class,
            "minimum": float(self.minimum), "maximum": float(self.maximum),
            "neutral": float(self.neutral), "max_age_ms": float(self.max_age_ms),
            "stale_after_ms": float(self.stale_after_ms),
        }

    @classmethod
    def from_wire(cls, data: Mapping[str, Any]) -> "DeployObsField":
        """厳密な field wire からインスタンスを作る。

        未知キー・欠落キーを含む設定は将来の意味を誤解しないよう拒否します。
        """
        ensure(isinstance(data, Mapping), "field must be mapping")
        ensure(set(data) == _FIELD_KEYS, "field keys mismatch")
        return cls(**{key: data[key] for key in _FIELD_KEYS})


@dataclass(frozen=True)
class DeployObsSchema:
    """DeployObsV1 の全 segment を順序付きで保持する schema。

    offset は記載順から計算し、絶対位置を設定ファイルへ埋め込みません。
    """

    fields: tuple[DeployObsField, ...]
    schema_version: str = DEPLOY_OBS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """schema version・field 型・名前の一意性を検証する。

        segment の欠落や重複を producer 間のずれとして早期に拒否します。
        """
        ensure(self.schema_version == DEPLOY_OBS_SCHEMA_VERSION, "unsupported schema version")
        ensure(isinstance(self.fields, tuple) and bool(self.fields), "fields must be non-empty tuple")
        ensure(all(isinstance(field, DeployObsField) for field in self.fields), "invalid field type")
        names = [field.name for field in self.fields]
        ensure(len(names) == len(set(names)), "duplicate segment")
        ensure(tuple(names) == DEPLOY_OBS_V1_SEGMENTS, "missing, unknown, or reordered segment")

    @property
    def dim(self) -> int:
        """value plane 一枚の次元数を返す。

        policy tensor 全体は value・validity・age の三枚なので、この値の3倍です。
        """
        return sum(field.size for field in self.fields)

    @property
    def layout(self) -> dict[str, tuple[int, int]]:
        """segment 名から順序由来の offset と幅を返す。

        設定の並びを走査して構築するため、絶対 offset は存在しません。
        """
        result: dict[str, tuple[int, int]] = {}
        offset = 0
        for field in self.fields:
            result[field.name] = (offset, field.size)
            offset += field.size
        return result

    def to_wire(self) -> dict[str, Any]:
        """schema の canonical wire 表現を返す。

        同じ順序と設定なら環境によらず同じ hash になる入力を作ります。
        """
        return {"schema_version": self.schema_version, "fields": [field.to_wire() for field in self.fields]}

    @property
    def schema_hash(self) -> str:
        """共有 canonical JSON 経路で schema hash を計算する。

        独自の JSON 化や hash 実装を避け、他契約と同じ決定性を使います。
        """
        return canonical_hash(self.to_wire())

    @classmethod
    def from_wire(cls, data: Mapping[str, Any]) -> "DeployObsSchema":
        """厳密な schema wire から schema を作る。

        version・キー・segment 内容のどれかが不正なら黙って補完しません。
        """
        ensure(isinstance(data, Mapping), "schema must be mapping")
        ensure(set(data) == _SCHEMA_KEYS, "schema keys mismatch")
        ensure(data["schema_version"] == DEPLOY_OBS_SCHEMA_VERSION, "unsupported schema version")
        ensure(isinstance(data["fields"], list), "fields must be list")
        return cls(tuple(DeployObsField.from_wire(field) for field in data["fields"]))

    @classmethod
    def default_v1(cls) -> "DeployObsSchema":
        """package fingerprint 用の確定 DeployObsV1 schema を返す。

        Common が YAML parser に依存せず、Deployment config と同じ内容を再現します。
        """
        rows = (
            ("player_hp", 1, "hud_inventory", 0., 1., 0., 1000., 250.),
            ("level", 1, "hud_inventory", 0., 1., 0., 2000., 1000.),
            ("player_screen_pos", 2, "screen_world_observed", -1., 1., 0., 500., 100.),
            ("nearest_enemy_offset", 2, "screen_world_observed", -1., 1., 0., 500., 100.),
            ("visible_enemy_count", 1, "screen_world_observed", 0., 1., 0., 500., 100.),
            ("movement_direction", 2, "temporal_inferred", -1., 1., 0., 500., 100.),
            ("weapon_category", 1, "hud_inventory", 0., 1., 1., 5000., 2000.),
            ("bias", 1, "constant", 0., 1., 0., 1., 1.),
            ("enemy_hp", 1, "unobservable", 0., 1., 0., 1., 0.),
            ("cooldown", 1, "unobservable", 0., 1., 0., 1., 0.),
        )
        return cls(tuple(DeployObsField(*row) for row in rows))


@dataclass(frozen=True)
class DeployObservation:
    """value・validity・age の三平面を持つ不変観測。

    配列は防御的コピーして read-only にし、policy 利用時にも再検証します。
    """

    values: np.ndarray
    validity: np.ndarray
    age: np.ndarray
    schema_hash: str
    timestamp_ns: int
    provenance: str = "release"

    def __post_init__(self) -> None:
        """配列・hash・時刻の基本契約を検証して凍結する。

        呼び出し元の配列変更が検証後の観測へ影響しないようコピーします。
        """
        ensure(isinstance(self.schema_hash, str) and len(self.schema_hash) == 64, "invalid schema hash")
        ensure(isinstance(self.timestamp_ns, int) and not isinstance(self.timestamp_ns, bool) and self.timestamp_ns >= 0, "invalid timestamp")
        ensure(type(self.provenance) is str and self.provenance in OBSERVATION_PROVENANCE, "invalid observation provenance")
        arrays = []
        for value in (self.values, self.validity, self.age):
            ensure(isinstance(value, np.ndarray) and value.ndim == 1, "observation planes must be 1d ndarray")
            copied = np.array(value, dtype=np.float32, copy=True)
            ensure(np.all(np.isfinite(copied)), "observation planes must be finite")
            copied.setflags(write=False)
            arrays.append(copied)
        ensure(arrays[0].shape == arrays[1].shape == arrays[2].shape, "plane shape mismatch")
        ensure(np.all((arrays[1] >= 0) & (arrays[1] <= 1)), "validity out of range")
        ensure(np.all((arrays[2] >= 0) & (arrays[2] <= 1)), "age out of range")
        object.__setattr__(self, "values", arrays[0])
        object.__setattr__(self, "validity", arrays[1])
        object.__setattr__(self, "age", arrays[2])

    def validate_for(self, schema: DeployObsSchema) -> None:
        """利用直前に schema・shape・範囲・欠損表現を再検証する。

        保存後の破損や別 schema との取り違えを policy 出力前に止めます。
        """
        ensure(isinstance(schema, DeployObsSchema), "schema must be DeployObsSchema")
        ensure(self.schema_hash == schema.schema_hash, "schema hash mismatch")
        ensure(self.values.shape == self.validity.shape == self.age.shape == (schema.dim,), "plane shape mismatch")
        ensure(
            np.all(np.isfinite(self.values)) and np.all(np.isfinite(self.validity)) and np.all(np.isfinite(self.age)),
            "observation planes must be finite",
        )
        ensure(np.all((self.validity >= 0) & (self.validity <= 1)), "validity out of range")
        ensure(np.all((self.age >= 0) & (self.age <= 1)), "age out of range")
        for field in schema.fields:
            offset, size = schema.layout[field.name]
            values, validity, age = self.values[offset:offset + size], self.validity[offset:offset + size], self.age[offset:offset + size]
            ensure(np.all((values >= field.minimum) & (values <= field.maximum)), f"{field.name} value out of range")
            missing = validity == 0
            ensure(np.all(values[missing] == field.neutral) and np.all(age[missing] == 1), f"{field.name} invalid missing representation")
            if self.provenance == "release" and field.source_class == "unobservable":
                ensure(
                    np.all(values == field.neutral) and np.all(validity == 0) and np.all(age == 1),
                    f"{field.name} privileged value requires oracle provenance",
                )

    def as_policy_tensor(self, schema: DeployObsSchema) -> np.ndarray:
        """schema 再検証後に三平面を連結した float32 tensor を返す。

        policy 利用直前に hash・次元・範囲・欠損表現を全て再確認し、
        検証後に不正に差し替えられた配列も出力しません。
        """
        ensure(isinstance(schema, DeployObsSchema), "schema must be DeployObsSchema")
        self.validate_for(schema)
        return np.concatenate([self.values, self.validity, self.age]).astype(np.float32)
