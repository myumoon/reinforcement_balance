"""``UiIntentV1``: UI アクション決定の唯一の共有表現。

所有権モデル（プラン 00-01 / index の制約）:

- ``choose_card`` は ItemSelector セッション（05-01 arbiter）が生成する。
- ``choose_fallback`` / ``reroll`` / ``skip`` / ``banish`` / ``ack_chest`` /
  ``confirm`` は共有の非モデル UI policy
  （:mod:`reinbalance_survivors_contracts.ui_policy`）が生成し、所有者は
  ``non_model_ui_policy``。
- ``no_op`` / ``stop`` は runtime safety が生成し、所有者は ``runtime_safety``。
- 全 intent の *effect* 所有者は ``ui_state_machine_v1``（05-03）に固定される。
  gameplay UI state machine は effect の実行のみを行い、decision source として
  現れてはならない。decision source として記録した fixture は拒否する。
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from .canonical_json import canonical_hash, canonical_json_bytes

__all__ = [
    "UI_INTENT_SCHEMA_VERSION",
    "EFFECT_OWNER_UI_STATE_MACHINE",
    "CHOOSE_FALLBACK_SEMANTICS",
    "ContractValidationError",
    "UiIntentKind",
    "DecisionOwner",
    "UiIntentV1",
    "allowed_semantic_actions",
]

UI_INTENT_SCHEMA_VERSION = "ui_intent.v1"

# effect 所有者は全 UI intent で固定：gameplay UI state machine（05-03）だけが
# intent の effect を実行できる。
EFFECT_OWNER_UI_STATE_MACHINE = "ui_state_machine_v1"


class ContractValidationError(ValueError):
    """契約値が one-of／所有権ルールに違反したときに送出される例外。"""


class UiIntentKind(str, enum.Enum):
    """UI intent の種別。card 選択・fallback・meta ボタン・ack/confirm・safety を表す。"""

    CHOOSE_CARD = "choose_card"
    CHOOSE_FALLBACK = "choose_fallback"
    REROLL = "reroll"
    SKIP = "skip"
    BANISH = "banish"
    ACK_CHEST = "ack_chest"
    CONFIRM = "confirm"
    NO_OP = "no_op"
    STOP = "stop"


class DecisionOwner(str, enum.Enum):
    """UI intent の決定を所有する主体（ItemSelector セッション／非モデル UI policy／runtime safety）。"""

    ITEM_SELECTOR_SESSION = "item_selector_session"
    NON_MODEL_UI_POLICY = "non_model_ui_policy"
    RUNTIME_SAFETY = "runtime_safety"


# -- 厳格な型ヘルパー（契約モジュール間で共有）------------------------------
def ensure(condition: bool, message: str) -> None:
    """``condition`` が偽なら ``ContractValidationError`` で fail-closed する。"""
    if not condition:
        raise ContractValidationError(message)


def is_strict_int(value: Any) -> bool:
    """真の int のときだけ True を返す（``bool`` は拒否）。"""
    return isinstance(value, int) and not isinstance(value, bool)


def is_strict_number(value: Any) -> bool:
    """真の int/float のときだけ True を返す（``bool`` は拒否）。"""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def require_schema_version(data: Mapping[str, Any], expected: str, label: str) -> str:
    """明示的な ``schema_version`` を返す。欠落／不一致は拒否する（silent default なし）。"""
    if "schema_version" not in data:
        raise ContractValidationError(f"missing required {label} field: schema_version")
    value = data["schema_version"]
    if value != expected:
        raise ContractValidationError(
            f"unsupported {label} schema_version {value!r} (expected {expected!r})"
        )
    return value


# one-of の必須／禁止ルールに関与する任意の構造フィールド。
_STRUCTURAL_FIELDS = ("target_id", "target_index", "candidate_set_hash")

# provenance フィールド（どの policy/rule/config が決定を生成したか）。
_PROVENANCE_FIELDS = ("decision_policy_id", "decision_rule_id", "decision_config_hash")

# ``from_wire`` が受理する全 wire キー（未知キーを fail-closed で拒否するために使う）。
_WIRE_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "semantic_action",
        "decision_owner",
        "effect_owner",
        "source_snapshot_hash",
        "source_frame_hash",
        "source_content_hash",
        "ui_state_key",
        "target_id",
        "target_index",
        "candidate_set_hash",
        "inventory_hash",
        "decision_policy_id",
        "decision_rule_id",
        "decision_config_hash",
    }
)


@dataclass(frozen=True)
class _KindRule:
    """kind ごとの決定所有者・必須／禁止フィールド・provenance 要否を表す内部ルール。"""

    owner: DecisionOwner
    required: tuple[str, ...]
    forbidden: tuple[str, ...]
    requires_provenance: bool


# one-of テーブル：kind ごとに、必須の decision owner と、存在／不在すべき構造
# フィールド、および policy provenance の要否を定義する。
_KIND_RULES: dict[UiIntentKind, _KindRule] = {
    UiIntentKind.CHOOSE_CARD: _KindRule(
        DecisionOwner.ITEM_SELECTOR_SESSION,
        required=("target_index", "candidate_set_hash"),
        forbidden=("target_id",),
        requires_provenance=True,
    ),
    UiIntentKind.CHOOSE_FALLBACK: _KindRule(
        DecisionOwner.NON_MODEL_UI_POLICY,
        required=("target_id", "target_index"),
        forbidden=("candidate_set_hash",),
        requires_provenance=True,
    ),
    UiIntentKind.REROLL: _KindRule(
        DecisionOwner.NON_MODEL_UI_POLICY,
        required=(),
        forbidden=("target_id", "target_index", "candidate_set_hash"),
        requires_provenance=True,
    ),
    UiIntentKind.SKIP: _KindRule(
        DecisionOwner.NON_MODEL_UI_POLICY,
        required=(),
        forbidden=("target_id", "target_index", "candidate_set_hash"),
        requires_provenance=True,
    ),
    UiIntentKind.BANISH: _KindRule(
        DecisionOwner.NON_MODEL_UI_POLICY,
        required=(),
        forbidden=("target_id", "target_index", "candidate_set_hash"),
        requires_provenance=True,
    ),
    UiIntentKind.ACK_CHEST: _KindRule(
        DecisionOwner.NON_MODEL_UI_POLICY,
        required=(),
        forbidden=("target_id", "target_index", "candidate_set_hash"),
        requires_provenance=True,
    ),
    UiIntentKind.CONFIRM: _KindRule(
        DecisionOwner.NON_MODEL_UI_POLICY,
        required=(),
        forbidden=("target_id", "target_index", "candidate_set_hash"),
        requires_provenance=True,
    ),
    UiIntentKind.NO_OP: _KindRule(
        DecisionOwner.RUNTIME_SAFETY,
        required=(),
        forbidden=("target_id", "target_index", "candidate_set_hash"),
        requires_provenance=False,
    ),
    UiIntentKind.STOP: _KindRule(
        DecisionOwner.RUNTIME_SAFETY,
        required=(),
        forbidden=("target_id", "target_index", "candidate_set_hash"),
        requires_provenance=False,
    ),
}


# choose_fallback の semantic_action は既知の fallback semantic のいずれか。
# ui_policy.FallbackSemantic と同期を保つ（test_fallback_semantics_in_sync で保証）。
CHOOSE_FALLBACK_SEMANTICS = frozenset({"chicken", "gold"})

# kind ごとに許可される semantic_action の値。choose_fallback 以外の全 kind では
# semantic_action は canonical なアクション文字列と一致しなければならない。これにより
# choose_fallback における "semantic_action == kind 名" という退化した形を拒否する。
_ALLOWED_SEMANTIC_ACTIONS: dict[UiIntentKind, frozenset[str]] = {
    UiIntentKind.CHOOSE_CARD: frozenset({"choose_card"}),
    UiIntentKind.CHOOSE_FALLBACK: CHOOSE_FALLBACK_SEMANTICS,
    UiIntentKind.REROLL: frozenset({"reroll"}),
    UiIntentKind.SKIP: frozenset({"skip"}),
    UiIntentKind.BANISH: frozenset({"banish"}),
    UiIntentKind.ACK_CHEST: frozenset({"ack_chest"}),
    UiIntentKind.CONFIRM: frozenset({"confirm"}),
    UiIntentKind.NO_OP: frozenset({"no_op"}),
    UiIntentKind.STOP: frozenset({"stop"}),
}


def allowed_semantic_actions(kind: UiIntentKind) -> frozenset[str]:
    """``kind`` に対して有効な ``semantic_action`` 値の集合を返す。"""
    return _ALLOWED_SEMANTIC_ACTIONS[kind]


@dataclass(frozen=True)
class UiIntentV1:
    """Training と Deployment が共有する、不変で versioned な UI intent。

    ``None`` の任意フィールドは canonical wire 表現から省かれるため、フィールドの
    有無がそのまま kind ごとの one-of（必須／禁止）ルールに対応する。
    """

    kind: UiIntentKind
    semantic_action: str
    decision_owner: DecisionOwner
    source_snapshot_hash: str
    source_frame_hash: str
    source_content_hash: str
    ui_state_key: str
    effect_owner: str = EFFECT_OWNER_UI_STATE_MACHINE
    target_id: Optional[str] = None
    target_index: Optional[int] = None
    candidate_set_hash: Optional[str] = None
    inventory_hash: Optional[str] = None
    decision_policy_id: Optional[str] = None
    decision_rule_id: Optional[str] = None
    decision_config_hash: Optional[str] = None
    schema_version: str = UI_INTENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.validate()

    # -- バリデーション ------------------------------------------------------
    def validate(self) -> None:
        if self.schema_version != UI_INTENT_SCHEMA_VERSION:
            raise ContractValidationError(
                f"unsupported schema_version {self.schema_version!r}"
            )
        if not isinstance(self.kind, UiIntentKind):
            raise ContractValidationError("kind must be a UiIntentKind")
        if not isinstance(self.decision_owner, DecisionOwner):
            raise ContractValidationError("decision_owner must be a DecisionOwner")

        if self.effect_owner != EFFECT_OWNER_UI_STATE_MACHINE:
            raise ContractValidationError(
                f"effect_owner must be {EFFECT_OWNER_UI_STATE_MACHINE!r}, "
                f"got {self.effect_owner!r}"
            )

        for name in (
            "semantic_action",
            "source_snapshot_hash",
            "source_frame_hash",
            "source_content_hash",
            "ui_state_key",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ContractValidationError(f"{name} must be a non-empty string")

        allowed = _ALLOWED_SEMANTIC_ACTIONS[self.kind]
        if self.semantic_action not in allowed:
            raise ContractValidationError(
                f"kind {self.kind.value} semantic_action must be one of "
                f"{sorted(allowed)}, got {self.semantic_action!r}"
            )

        rule = _KIND_RULES[self.kind]
        if self.decision_owner != rule.owner:
            raise ContractValidationError(
                f"kind {self.kind.value} must be owned by {rule.owner.value}, "
                f"got {self.decision_owner.value}"
            )

        # gameplay UI state machine（05-03）は effect のみを所有する。decision
        # source として記録してはならない。
        for name in _PROVENANCE_FIELDS[:2]:  # policy id と rule id
            value = getattr(self, name)
            if isinstance(value, str) and value.startswith("ui_state_machine"):
                raise ContractValidationError(
                    "decision source must not reference the ui_state_machine "
                    "(05-03 owns effects only)"
                )

        for name in rule.required:
            if getattr(self, name) is None:
                raise ContractValidationError(
                    f"kind {self.kind.value} requires field {name}"
                )
        for name in rule.forbidden:
            if getattr(self, name) is not None:
                raise ContractValidationError(
                    f"kind {self.kind.value} forbids field {name}"
                )
        if rule.requires_provenance:
            for name in _PROVENANCE_FIELDS:
                value = getattr(self, name)
                if not isinstance(value, str) or not value:
                    raise ContractValidationError(
                        f"kind {self.kind.value} requires {name}"
                    )

        if self.target_index is not None:
            if not is_strict_int(self.target_index) or self.target_index < 0:
                raise ContractValidationError(
                    "target_index must be a non-negative int"
                )

        for name in (
            "target_id",
            "candidate_set_hash",
            "inventory_hash",
            "decision_policy_id",
            "decision_rule_id",
            "decision_config_hash",
        ):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ContractValidationError(f"{name} must be a non-empty string")

    # -- シリアライズ --------------------------------------------------------
    def to_wire(self) -> dict[str, Any]:
        wire: dict[str, Any] = {
            "schema_version": self.schema_version,
            "kind": self.kind.value,
            "semantic_action": self.semantic_action,
            "decision_owner": self.decision_owner.value,
            "effect_owner": self.effect_owner,
            "source_snapshot_hash": self.source_snapshot_hash,
            "source_frame_hash": self.source_frame_hash,
            "source_content_hash": self.source_content_hash,
            "ui_state_key": self.ui_state_key,
        }
        for name in (
            "target_id",
            "target_index",
            "candidate_set_hash",
            "inventory_hash",
            "decision_policy_id",
            "decision_rule_id",
            "decision_config_hash",
        ):
            value = getattr(self, name)
            if value is not None:
                wire[name] = value
        return wire

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_wire())

    def intent_hash(self) -> str:
        return canonical_hash(self.to_wire())

    @classmethod
    def from_wire(cls, data: Mapping[str, Any]) -> "UiIntentV1":
        unknown = set(data) - _WIRE_KEYS
        if unknown:
            raise ContractValidationError(
                f"unknown UiIntentV1 fields: {sorted(unknown)}"
            )
        require_schema_version(data, UI_INTENT_SCHEMA_VERSION, "UiIntentV1")
        for name in (
            "kind",
            "semantic_action",
            "decision_owner",
            "source_snapshot_hash",
            "source_frame_hash",
            "source_content_hash",
            "ui_state_key",
        ):
            if name not in data:
                raise ContractValidationError(
                    f"missing required UiIntentV1 field: {name}"
                )
        try:
            kind = UiIntentKind(data["kind"])
        except ValueError:
            raise ContractValidationError(f"unknown UiIntentV1 kind: {data['kind']!r}")
        try:
            decision_owner = DecisionOwner(data["decision_owner"])
        except ValueError:
            raise ContractValidationError(
                f"unknown decision_owner: {data['decision_owner']!r}"
            )
        return cls(
            kind=kind,
            semantic_action=data["semantic_action"],
            decision_owner=decision_owner,
            source_snapshot_hash=data["source_snapshot_hash"],
            source_frame_hash=data["source_frame_hash"],
            source_content_hash=data["source_content_hash"],
            ui_state_key=data["ui_state_key"],
            effect_owner=data.get("effect_owner", EFFECT_OWNER_UI_STATE_MACHINE),
            target_id=data.get("target_id"),
            target_index=data.get("target_index"),
            candidate_set_hash=data.get("candidate_set_hash"),
            inventory_hash=data.get("inventory_hash"),
            decision_policy_id=data.get("decision_policy_id"),
            decision_rule_id=data.get("decision_rule_id"),
            decision_config_hash=data.get("decision_config_hash"),
            schema_version=data["schema_version"],
        )
