"""target profile 参照と action semantics の契約（scaffold, v1）。

- ``TargetProfileRef`` は正確な target（game build・canonical save・hardware
  profile）への不変 identity 参照。正式な target / save / build / hardware 契約は
  プラン 00-03 で確定する。本型は identity を保持し、子プランが束縛できるようにする。
- ``ActionSemantics`` は離散アクション semantic の共有された順序付きリスト。9-action の
  golden parity はプラン 00-03 が確定する。ここでは ``default_v1`` が曖昧さのない
  ``(dx, dy)`` グリッド名を使い、方位（東西）の意味を早まって固定しない。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .canonical_json import canonical_hash

__all__ = [
    "TARGET_PROFILE_SCHEMA_VERSION",
    "ACTION_SEMANTICS_SCHEMA_VERSION",
    "TargetProfileRef",
    "ActionSemantics",
]

TARGET_PROFILE_SCHEMA_VERSION = "target_profile_ref.v1"
ACTION_SEMANTICS_SCHEMA_VERSION = "action_semantics.v1"


@dataclass(frozen=True)
class TargetProfileRef:
    """正確な target（build・canonical save・hardware profile）への不変 identity 参照。"""

    build_id: str
    canonical_save_hash: str
    hardware_profile_id: str
    schema_version: str = TARGET_PROFILE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("build_id", "canonical_save_hash", "hardware_profile_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")

    def to_wire(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "build_id": self.build_id,
            "canonical_save_hash": self.canonical_save_hash,
            "hardware_profile_id": self.hardware_profile_id,
        }

    @property
    def ref_hash(self) -> str:
        return canonical_hash(self.to_wire())

    @classmethod
    def from_wire(cls, data: Mapping[str, Any]) -> "TargetProfileRef":
        if set(data) != {"schema_version", "build_id", "canonical_save_hash", "hardware_profile_id"}:
            raise ValueError("unknown or missing TargetProfileRef fields")
        if data.get("schema_version") != TARGET_PROFILE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported TargetProfileRef schema_version "
                f"{data.get('schema_version')!r}"
            )
        return cls(
            build_id=data["build_id"],
            canonical_save_hash=data["canonical_save_hash"],
            hardware_profile_id=data["hardware_profile_id"],
            schema_version=data["schema_version"],
        )


@dataclass(frozen=True)
class ActionSemantics:
    """離散アクションの共有された順序付き semantic 一覧（versioned）。"""

    actions: tuple[str, ...]
    schema_version: str = ACTION_SEMANTICS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.actions:
            raise ValueError("actions must be non-empty")
        seen: set[str] = set()
        for name in self.actions:
            if not isinstance(name, str) or not name:
                raise ValueError("action names must be non-empty strings")
            if name in seen:
                raise ValueError(f"duplicate action semantic {name!r}")
            seen.add(name)

    @property
    def num_actions(self) -> int:
        return len(self.actions)

    def index_of(self, name: str) -> int:
        return self.actions.index(name)

    def to_wire(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "actions": list(self.actions),
        }

    @property
    def semantics_hash(self) -> str:
        return canonical_hash(self.to_wire())

    @classmethod
    def from_wire(cls, data: Mapping[str, Any]) -> "ActionSemantics":
        if set(data) != {"schema_version", "actions"}:
            raise ValueError("unknown or missing ActionSemantics fields")
        if data.get("schema_version") != ACTION_SEMANTICS_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported ActionSemantics schema_version "
                f"{data.get('schema_version')!r}"
            )
        return cls(actions=tuple(data["actions"]), schema_version=data["schema_version"])

    @classmethod
    def default_v1(cls) -> "ActionSemantics":
        """9アクションの移動グリッド（scaffold）。golden parity は 00-03 が確定する。"""
        names = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                names.append(f"move_dx{dx}_dy{dy}")
        return cls(actions=tuple(names))
