"""screen-space DeployObs スキーマと観測コンテナ（scaffold, v1）。

本モジュールは deployable observation 契約の *フレームワーク* を定義する：versioned で
順序付きの特徴レイアウト（:class:`DeployObsSchema`）と、Training（simulator student）と
Deployment（real perception）の双方が共有する不変の観測インスタンス
（:class:`DeployObservation`）。

正式な screen-space 特徴一覧はプラン 03-01 で確定する。ここでは小さな viewport 正規化
デフォルト（``default_v1``）を提供し、契約・その canonical hash・往復変換を end-to-end で
検証できるようにする。後続プランは ``schema_version`` を上げてレイアウトを拡張する。

NumPy は array ヘルパー内で遅延 import する。これにより、基盤契約（とそのテスト）は
NumPy が無くても import できる。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .canonical_json import canonical_hash

__all__ = [
    "DEPLOY_OBS_SCHEMA_VERSION",
    "DeployObsField",
    "DeployObsSchema",
    "DeployObservation",
]

DEPLOY_OBS_SCHEMA_VERSION = "deploy_obs.v1"


@dataclass(frozen=True)
class DeployObsField:
    """観測ベクトル内の、名前付きで固定幅のスカラ成分ブロック。"""

    name: str
    size: int

    def to_wire(self) -> dict[str, Any]:
        return {"name": self.name, "size": self.size}


@dataclass(frozen=True)
class DeployObsSchema:
    """screen-space DeployObs の順序付き特徴レイアウト（versioned）。dim・offset・canonical hash を提供する。"""

    fields: tuple[DeployObsField, ...]
    schema_version: str = DEPLOY_OBS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.fields:
            raise ValueError("DeployObsSchema requires at least one field")
        seen: set[str] = set()
        for spec in self.fields:
            if not isinstance(spec.name, str) or not spec.name:
                raise ValueError("field name must be a non-empty string")
            if spec.size <= 0:
                raise ValueError(f"field {spec.name!r} size must be positive")
            if spec.name in seen:
                raise ValueError(f"duplicate field name {spec.name!r}")
            seen.add(spec.name)

    @property
    def dim(self) -> int:
        return sum(spec.size for spec in self.fields)

    @property
    def layout(self) -> dict[str, tuple[int, int]]:
        """フィールド名 -> (offset, size) の対応を返す。"""
        out: dict[str, tuple[int, int]] = {}
        offset = 0
        for spec in self.fields:
            out[spec.name] = (offset, spec.size)
            offset += spec.size
        return out

    def to_wire(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "fields": [spec.to_wire() for spec in self.fields],
        }

    @property
    def schema_hash(self) -> str:
        return canonical_hash(self.to_wire())

    @classmethod
    def from_wire(cls, data: Mapping[str, Any]) -> "DeployObsSchema":
        if data.get("schema_version") != DEPLOY_OBS_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported DeployObsSchema schema_version "
                f"{data.get('schema_version')!r}"
            )
        fields = []
        for spec in data["fields"]:
            size = spec["size"]
            if isinstance(size, bool) or not isinstance(size, int):
                raise ValueError("field size must be an int")
            fields.append(DeployObsField(name=spec["name"], size=size))
        return cls(fields=tuple(fields), schema_version=data["schema_version"])

    @classmethod
    def default_v1(cls) -> "DeployObsSchema":
        """小さな viewport 正規化のデフォルトレイアウト（scaffold；03-01 が確定する）。"""
        return cls(
            fields=(
                DeployObsField("player_screen_pos", 2),
                DeployObsField("player_hp_fraction", 1),
                DeployObsField("run_timer_fraction", 1),
                DeployObsField("level_fraction", 1),
                DeployObsField("xp_fraction", 1),
                DeployObsField("nearest_enemies", 24),  # 8 体 x (dx, dy, on_screen)
                DeployObsField("nearest_pickups", 12),  # 4 個 x (dx, dy, on_screen)
            )
        )


@dataclass(frozen=True)
class DeployObservation:
    """あるスキーマに準拠した不変の観測インスタンス（値の長さ・有限性を検証する）。"""

    schema: DeployObsSchema
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.values) != self.schema.dim:
            raise ValueError(
                f"expected {self.schema.dim} values, got {len(self.values)}"
            )
        for value in self.values:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("observation values must be real numbers")
            if not math.isfinite(float(value)):
                raise ValueError("observation values must be finite")

    def field(self, name: str) -> tuple[float, ...]:
        offset, size = self.schema.layout[name]
        return self.values[offset : offset + size]

    def to_wire(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema.schema_version,
            "schema_hash": self.schema.schema_hash,
            "values": [float(v) for v in self.values],
        }

    def to_array(self):  # -> numpy.ndarray[float32] を返す
        import numpy as np

        return np.asarray(self.values, dtype=np.float32)

    @classmethod
    def from_array(cls, schema: DeployObsSchema, array: Sequence[float]) -> "DeployObservation":
        return cls(schema=schema, values=tuple(float(v) for v in array))
