"""ItemSelector が利用できる画面観測 feature の共有契約を定義する。

context-only と danger 付きの二案を versioned wire へ固定し、候補を ``max_item_cards``
まで padding する。教師 score や privileged world state を受け取る自由形式の入力面は持たず、
Training/Deployment が同じ canonical identity を計算できるようにする。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Literal, Mapping, TypeAlias

from .canonical_json import canonical_hash

__all__ = [
    "ITEM_DECISION_SCHEMA_VERSION",
    "CANDIDATE_FEATURES_SCHEMA_VERSION",
    "ItemDecisionSchemaV1",
    "CandidateFeatures",
    "ItemDecisionFeatures",
]

ITEM_DECISION_SCHEMA_VERSION = "item_decision.v1"
CANDIDATE_FEATURES_SCHEMA_VERSION = "candidate_features.v1"
ItemDecisionSchemaV1: TypeAlias = Literal[
    "context_only_v1", "context_danger_v1"
]
_FEATURE_SCHEMAS = frozenset({"context_only_v1", "context_danger_v1"})
_PADDING_KIND = "padding"
_PADDING_ITEM_ID = "__padding__"
_CANDIDATE_WIRE_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "item_id",
        "new_level",
        "owned",
        "is_new",
        "is_evolve",
        "is_union",
        "has_prerequisite",
        "slot_capacity",
    }
)
_ITEM_WIRE_FIELDS = frozenset(
    {
        "schema_version",
        "feature_schema",
        "decision_id",
        "max_item_cards",
        "context_features",
        "candidates",
    }
)
_BASE_CONTEXT_FIELDS = frozenset(
    {
        "elapsed_time",
        "level",
        "hp_ratio",
        "xp_ratio",
        "weapon_slots",
        "passive_slots",
        "empty_slot_count",
        "evolution_readiness",
        "choice_count",
        "card_mask",
        "fallback_kind",
        "ui_state_validity",
        "ui_state_age",
    }
)
_DANGER_CONTEXT_FIELDS = frozenset(
    {
        "enemy_density",
        "gem_density",
        "nearest_enemy_screen_dist",
        "nearest_enemy_screen_dir",
        "boss_flag",
        "hazard_flag",
        "world_validity",
        "world_age",
        "last_gameplay_snapshot_age",
    }
)


def _ensure_fields(data: Any, expected: frozenset[str], label: str) -> None:
    """wire object の必須・未知 field を exact に検証する。

    未審査 feature の追加を暗黙受理せず、全 reader を fail-closed に保つ。
    """
    if not isinstance(data, Mapping):
        raise ValueError(f"{label} must be an object")
    actual = frozenset(data)
    if actual != expected:
        raise ValueError(
            f"{label} fields mismatch: "
            f"missing={sorted(expected - actual)}, unknown={sorted(actual - expected)}"
        )


def _finite(value: Any, label: str, *, minimum: float | None = None) -> float:
    """bool でない有限実数を検証し float として返す。

    age・ratio・density の NaN や負値を unknown の代用にせず明示的に拒否する。
    """
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be finite")
    result = float(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    return result


def _ratio(value: Any, label: str) -> float:
    """観測 validity・ratio を閉区間 [0, 1] へ制約する。

    欠損は validity=0 と age の組で表現し、範囲外 sentinel は許可しない。
    """
    result = _finite(value, label)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be in [0, 1]")
    return result


def _strict_int(value: Any, label: str, *, minimum: int = 0) -> int:
    """bool を除く整数を下限付きで検証する。

    slot level と count の float 丸めを禁止して wire parity を維持する。
    """
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an int >= {minimum}")
    return value


def _slot_levels(value: Any, label: str) -> tuple[int, ...]:
    """weapon/passive の六つの画面 slot level を検証する。

    空 slot は 0、占有 slot は正 level とし、常に固定長ベクトルにする。
    """
    if not isinstance(value, (list, tuple)) or len(value) != 6:
        raise ValueError(f"{label} must contain exactly 6 slots")
    return tuple(_strict_int(item, f"{label}[{index}]") for index, item in enumerate(value))


@dataclass(frozen=True, slots=True)
class CandidateFeatures:
    """ItemSelector の候補一件を画面カード由来の field だけで表す。

    ``features`` は model 入力候補の監査用 named mapping で、teacher/critic 値を追加する
    arbitrary payload は提供しない。
    """

    kind: str
    item_id: str
    new_level: int
    owned: bool
    is_new: bool
    is_evolve: bool
    is_union: bool
    has_prerequisite: bool
    slot_capacity: int
    schema_version: str = CANDIDATE_FEATURES_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """candidate schema と field 型・padding sentinel を検証する。

        masked slot 以外では空 identity を認めず、padding 表現を一意にする。
        """
        if self.schema_version != CANDIDATE_FEATURES_SCHEMA_VERSION:
            raise ValueError("unsupported CandidateFeatures schema_version")
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError("candidate kind must be a non-empty string")
        if not isinstance(self.item_id, str) or not self.item_id:
            raise ValueError("candidate item_id must be a non-empty string")
        _strict_int(self.new_level, "candidate new_level")
        _strict_int(self.slot_capacity, "candidate slot_capacity")
        for label in (
            "owned",
            "is_new",
            "is_evolve",
            "is_union",
            "has_prerequisite",
        ):
            if type(getattr(self, label)) is not bool:
                raise ValueError(f"candidate {label} must be bool")
        if self.kind == _PADDING_KIND:
            if (
                self.item_id != _PADDING_ITEM_ID
                or self.new_level != 0
                or self.slot_capacity != 0
                or any(
                    (
                        self.owned,
                        self.is_new,
                        self.is_evolve,
                        self.is_union,
                        self.has_prerequisite,
                    )
                )
            ):
                raise ValueError("candidate padding sentinel fields mismatch")
        elif self.item_id == _PADDING_ITEM_ID:
            raise ValueError("padding item_id is reserved")

    @classmethod
    def padding(cls) -> "CandidateFeatures":
        """masked card slot の一意な zero-like sentinel を返す。

        Nmax padding は card mask と組でのみ有効候補から除外される。
        """
        return cls(
            kind=_PADDING_KIND,
            item_id=_PADDING_ITEM_ID,
            new_level=0,
            owned=False,
            is_new=False,
            is_evolve=False,
            is_union=False,
            has_prerequisite=False,
            slot_capacity=0,
        )

    @property
    def is_padding(self) -> bool:
        """この候補が masked slot sentinel なら True を返す。

        item identity ではなく予約 kind と検証済み完全形で判定する。
        """
        return self.kind == _PADDING_KIND

    @property
    def candidate_id(self) -> str:
        """旧 consumer 向けに item identity と同じ candidate ID を返す。

        位置 index は identity として公開しない。
        """
        return self.item_id

    @property
    def features(self) -> dict[str, Any]:
        """モデル候補入力の allow-list field mapping を返す。

        schema identity は除き、画面カードから直接得られる値だけを含める。
        """
        return {
            "kind": self.kind,
            "item_id": self.item_id,
            "new_level": self.new_level,
            "owned": self.owned,
            "is_new": self.is_new,
            "is_evolve": self.is_evolve,
            "is_union": self.is_union,
            "has_prerequisite": self.has_prerequisite,
            "slot_capacity": self.slot_capacity,
        }

    def to_wire(self) -> dict[str, Any]:
        """候補を versioned canonical-JSON-compatible object にする。

        field は allow-list の直下へ展開し、自由形式 nested feature を作らない。
        """
        return {"schema_version": self.schema_version, **self.features}

    @classmethod
    def from_wire(cls, data: Mapping[str, Any]) -> "CandidateFeatures":
        """候補 wire の exact schema を検証して復元する。

        unknown field は leakage 候補として schema version に関係なく拒否する。
        """
        _ensure_fields(data, _CANDIDATE_WIRE_FIELDS, "candidate")
        return cls(
            kind=data["kind"],
            item_id=data["item_id"],
            new_level=data["new_level"],
            owned=data["owned"],
            is_new=data["is_new"],
            is_evolve=data["is_evolve"],
            is_union=data["is_union"],
            has_prerequisite=data["has_prerequisite"],
            slot_capacity=data["slot_capacity"],
            schema_version=data["schema_version"],
        )


@dataclass(frozen=True, slots=True)
class ItemDecisionFeatures:
    """一回の level-up context と可変個数候補を二つの feature 案で表す。

    base context は両案で共通、danger field は ``context_danger_v1`` のみ必須とし、wire
    では候補を ``max_item_cards`` まで padding する。
    """

    decision_id: str
    feature_schema: ItemDecisionSchemaV1
    elapsed_time: float
    level: int
    hp_ratio: float
    xp_ratio: float
    weapon_slots: tuple[int, ...]
    passive_slots: tuple[int, ...]
    empty_slot_count: int
    evolution_readiness: float
    choice_count: int
    card_mask: tuple[bool, ...]
    fallback_kind: str
    ui_state_validity: float
    ui_state_age: float
    candidates: tuple[CandidateFeatures, ...]
    max_item_cards: int = 3
    enemy_density: float | None = None
    gem_density: float | None = None
    nearest_enemy_screen_dist: float | None = None
    nearest_enemy_screen_dir: tuple[float, float] | None = None
    boss_flag: bool | None = None
    hazard_flag: bool | None = None
    world_validity: float | None = None
    world_age: float | None = None
    last_gameplay_snapshot_age: float | None = None
    schema_version: str = ITEM_DECISION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """variant、固定長 slot、mask、danger completeness をまとめて検証する。

        choice count と候補/mask の三者を同じ不変条件へ束縛する。
        """
        if self.schema_version != ITEM_DECISION_SCHEMA_VERSION:
            raise ValueError("unsupported ItemDecisionFeatures schema_version")
        if not isinstance(self.decision_id, str) or not self.decision_id:
            raise ValueError("decision_id must be a non-empty string")
        if self.feature_schema not in _FEATURE_SCHEMAS:
            raise ValueError("unsupported item decision feature_schema")
        _finite(self.elapsed_time, "elapsed_time", minimum=0.0)
        _strict_int(self.level, "level", minimum=1)
        _ratio(self.hp_ratio, "hp_ratio")
        _ratio(self.xp_ratio, "xp_ratio")
        _slot_levels(self.weapon_slots, "weapon_slots")
        _slot_levels(self.passive_slots, "passive_slots")
        _strict_int(self.empty_slot_count, "empty_slot_count")
        _ratio(self.evolution_readiness, "evolution_readiness")
        _strict_int(self.max_item_cards, "max_item_cards", minimum=1)
        _strict_int(self.choice_count, "choice_count", minimum=1)
        if (
            not isinstance(self.card_mask, (list, tuple))
            or len(self.card_mask) != self.max_item_cards
            or any(type(value) is not bool for value in self.card_mask)
        ):
            raise ValueError("card_mask must be a max_item_cards bool vector")
        if self.choice_count != sum(self.card_mask):
            raise ValueError("choice_count must equal valid card_mask count")
        if len(self.candidates) != self.choice_count:
            raise ValueError("candidate count must equal choice_count")
        if self.choice_count > self.max_item_cards:
            raise ValueError("candidate count exceeds max_item_cards")
        if any(candidate.is_padding for candidate in self.candidates):
            raise ValueError("input candidates cannot contain padding")
        item_ids = [candidate.item_id for candidate in self.candidates]
        if len(set(item_ids)) != len(item_ids):
            raise ValueError("candidate item_id values must be unique")
        observed_empty_slots = sum(level == 0 for level in self.weapon_slots) + sum(
            level == 0 for level in self.passive_slots
        )
        if self.empty_slot_count != observed_empty_slots:
            raise ValueError("empty_slot_count must match weapon/passive slot observations")
        if not isinstance(self.fallback_kind, str) or not self.fallback_kind:
            raise ValueError("fallback_kind must be a non-empty string")
        _ratio(self.ui_state_validity, "ui_state_validity")
        _finite(self.ui_state_age, "ui_state_age", minimum=0.0)
        self._validate_danger_fields()

    def _validate_danger_fields(self) -> None:
        """danger variant の全 field を all-or-none で検証する。

        context-only への raw world substitution と danger 側の部分欠損を対称に拒否する。
        """
        fields = {
            name: getattr(self, name)
            for name in _DANGER_CONTEXT_FIELDS
        }
        if self.feature_schema == "context_only_v1":
            if any(value is not None for value in fields.values()):
                raise ValueError("danger fields are forbidden for context_only_v1")
            return
        if any(value is None for value in fields.values()):
            raise ValueError("all danger fields are required for context_danger_v1")
        _ratio(self.enemy_density, "enemy_density")
        _ratio(self.gem_density, "gem_density")
        _ratio(self.nearest_enemy_screen_dist, "nearest_enemy_screen_dist")
        direction = self.nearest_enemy_screen_dir
        if not isinstance(direction, (list, tuple)) or len(direction) != 2:
            raise ValueError("nearest_enemy_screen_dir must contain x/y")
        for index, value in enumerate(direction):
            normalized = _finite(value, f"nearest_enemy_screen_dir[{index}]")
            if not -1.0 <= normalized <= 1.0:
                raise ValueError("nearest_enemy_screen_dir values must be in [-1, 1]")
        if type(self.boss_flag) is not bool or type(self.hazard_flag) is not bool:
            raise ValueError("boss_flag and hazard_flag must be bool")
        _ratio(self.world_validity, "world_validity")
        _finite(self.world_age, "world_age", minimum=0.0)
        _finite(
            self.last_gameplay_snapshot_age,
            "last_gameplay_snapshot_age",
            minimum=0.0,
        )

    @property
    def context_features(self) -> dict[str, Any]:
        """選択 variant に許された context allow-list mapping を返す。

        target-derived UI field は含むが teacher label、off-screen count、raw world は含めない。
        """
        result: dict[str, Any] = {
            "elapsed_time": float(self.elapsed_time),
            "level": self.level,
            "hp_ratio": float(self.hp_ratio),
            "xp_ratio": float(self.xp_ratio),
            "weapon_slots": list(self.weapon_slots),
            "passive_slots": list(self.passive_slots),
            "empty_slot_count": self.empty_slot_count,
            "evolution_readiness": float(self.evolution_readiness),
            "choice_count": self.choice_count,
            "card_mask": list(self.card_mask),
            "fallback_kind": self.fallback_kind,
            "ui_state_validity": float(self.ui_state_validity),
            "ui_state_age": float(self.ui_state_age),
        }
        if self.feature_schema == "context_danger_v1":
            result.update(
                {
                    "enemy_density": float(self.enemy_density),
                    "gem_density": float(self.gem_density),
                    "nearest_enemy_screen_dist": float(self.nearest_enemy_screen_dist),
                    "nearest_enemy_screen_dir": [
                        float(value) for value in self.nearest_enemy_screen_dir
                    ],
                    "boss_flag": self.boss_flag,
                    "hazard_flag": self.hazard_flag,
                    "world_validity": float(self.world_validity),
                    "world_age": float(self.world_age),
                    "last_gameplay_snapshot_age": float(
                        self.last_gameplay_snapshot_age
                    ),
                }
            )
        return result

    @property
    def padded_candidates(self) -> tuple[CandidateFeatures, ...]:
        """card mask に沿った Nmax 固定長の候補 tuple を返す。

        valid 候補を True slot へ順に配置し、False slot は一意な padding sentinel にする。
        """
        iterator = iter(self.candidates)
        return tuple(
            next(iterator) if valid else CandidateFeatures.padding()
            for valid in self.card_mask
        )

    def to_wire(self) -> dict[str, Any]:
        """context と Nmax 候補を versioned canonical wire に変換する。

        hash 対象には timestamp/path を含めず、決定 feature identity だけを含める。
        """
        return {
            "schema_version": self.schema_version,
            "feature_schema": self.feature_schema,
            "decision_id": self.decision_id,
            "max_item_cards": self.max_item_cards,
            "context_features": self.context_features,
            "candidates": [candidate.to_wire() for candidate in self.padded_candidates],
        }

    @property
    def decision_hash(self) -> str:
        """共有 canonical JSON helper による feature 内容 hash を返す。

        独自 serializer/hash 計算は行わない。
        """
        return canonical_hash(self.to_wire())

    def with_candidates(
        self, candidates: tuple[CandidateFeatures, ...]
    ) -> "ItemDecisionFeatures":
        """同じ context/mask の候補順だけを置換した immutable copy を返す。

        候補数と identity uniqueness は通常 constructor で再検証する。
        """
        return replace(self, candidates=tuple(candidates))

    def with_danger(self, **fields: Any) -> "ItemDecisionFeatures":
        """danger field を明示置換した immutable copy を返す。

        context-only instance への danger 混入は constructor invariant により失敗する。
        """
        return replace(self, **fields)

    @classmethod
    def from_wire(cls, data: Mapping[str, Any]) -> "ItemDecisionFeatures":
        """item decision wire の variant-specific exact schema を検証して復元する。

        masked slot が非 padding、valid slot が padding の双方を拒否する。
        """
        _ensure_fields(data, _ITEM_WIRE_FIELDS, "item decision")
        if data["schema_version"] != ITEM_DECISION_SCHEMA_VERSION:
            raise ValueError("unsupported ItemDecisionFeatures schema_version")
        feature_schema = data["feature_schema"]
        if feature_schema not in _FEATURE_SCHEMAS:
            raise ValueError("unsupported item decision feature_schema")
        expected_context = (
            _BASE_CONTEXT_FIELDS | _DANGER_CONTEXT_FIELDS
            if feature_schema == "context_danger_v1"
            else _BASE_CONTEXT_FIELDS
        )
        context = data["context_features"]
        _ensure_fields(context, expected_context, "context features")
        max_cards = data["max_item_cards"]
        _strict_int(max_cards, "max_item_cards", minimum=1)
        raw_candidates = data["candidates"]
        if not isinstance(raw_candidates, list) or len(raw_candidates) != max_cards:
            raise ValueError("candidates wire must be padded to max_item_cards")
        card_mask = context["card_mask"]
        if not isinstance(card_mask, list) or len(card_mask) != max_cards:
            raise ValueError("card_mask must match max_item_cards")
        parsed = [CandidateFeatures.from_wire(item) for item in raw_candidates]
        valid_candidates: list[CandidateFeatures] = []
        for index, (valid, candidate) in enumerate(zip(card_mask, parsed, strict=True)):
            if type(valid) is not bool:
                raise ValueError("card_mask values must be bool")
            if valid and candidate.is_padding:
                raise ValueError(f"valid candidate[{index}] cannot be padding")
            if not valid and not candidate.is_padding:
                raise ValueError(f"masked candidate[{index}] must be padding")
            if valid:
                valid_candidates.append(candidate)
        danger = {
            name: (
                tuple(context[name])
                if name == "nearest_enemy_screen_dir"
                else context[name]
            )
            for name in _DANGER_CONTEXT_FIELDS
        } if feature_schema == "context_danger_v1" else {}
        return cls(
            decision_id=data["decision_id"],
            feature_schema=feature_schema,
            elapsed_time=context["elapsed_time"],
            level=context["level"],
            hp_ratio=context["hp_ratio"],
            xp_ratio=context["xp_ratio"],
            weapon_slots=tuple(context["weapon_slots"]),
            passive_slots=tuple(context["passive_slots"]),
            empty_slot_count=context["empty_slot_count"],
            evolution_readiness=context["evolution_readiness"],
            choice_count=context["choice_count"],
            card_mask=tuple(card_mask),
            fallback_kind=context["fallback_kind"],
            ui_state_validity=context["ui_state_validity"],
            ui_state_age=context["ui_state_age"],
            candidates=tuple(valid_candidates),
            max_item_cards=max_cards,
            schema_version=data["schema_version"],
            **danger,
        )
