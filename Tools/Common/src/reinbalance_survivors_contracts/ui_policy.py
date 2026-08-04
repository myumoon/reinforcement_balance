"""共有の pure な ``NonModelUiPolicyV1`` 決定ルール。

``decide_non_model_ui_intent(input, config) -> UiIntentV1 | None`` は、fallback /
meta / ack / confirm の決定ポリシーの唯一の実行可能な source。simulator（02-04
evaluator）と live（05-01 arbiter）の双方がこの関数をそのまま呼ぶことで、同じ入力と
config hash に対して byte-identical な intent を生成する。

所有権の境界:

- 本モジュールが pure な決定ルール（``UiPolicyInputV1 -> UiIntentV1 | None``）を所有する。
- 05-01 は production/live で snapshot adapter と決定オーケストレーションを所有する。
- 05-03 は intent の resolve / retry / input effect を所有し、intent を生成・変更しては
  ならない。
- 02-04 は evaluation adapter から同じルールを呼ぶだけで、production の所有者ではない。

``UiPolicyInputV1`` は画面由来でモデル非依存のシグナルのみを持つ：source
snapshot / frame / content hash、安定した UI state key、画面状態、画面由来の HP
比率、candidate-set / inventory hash、順序付き fallback target、および任意の button
semantic/capability。ROI 幾何やモデル／教師値を決して持ってはならない。
"""

from __future__ import annotations

import enum
import json
from dataclasses import dataclass
from importlib import resources
from typing import Any, Mapping, Optional

from .canonical_json import canonical_hash
from .ui_intent import (
    ContractValidationError,
    DecisionOwner,
    UiIntentKind,
    UiIntentV1,
    ensure,
    is_strict_int,
    is_strict_number,
    require_schema_version,
)

__all__ = [
    "UI_POLICY_INPUT_SCHEMA_VERSION",
    "NON_MODEL_UI_POLICY_CONFIG_SCHEMA_VERSION",
    "ScreenState",
    "FallbackSemantic",
    "ButtonSemantic",
    "FallbackTarget",
    "ButtonOption",
    "UiPolicyInputV1",
    "NonModelUiPolicyConfigV1",
    "decide_non_model_ui_intent",
]

UI_POLICY_INPUT_SCHEMA_VERSION = "ui_policy_input.v1"
NON_MODEL_UI_POLICY_CONFIG_SCHEMA_VERSION = "non_model_ui_policy_config.v1"

_DEFAULT_CONFIG_RESOURCE = "non_model_ui_policy_v1.json"


class ScreenState(str, enum.Enum):
    """非モデル UI policy が判断する画面状態（通常プレイ／レベルアップ／fallback／chest／confirm／不明）。"""

    GAMEPLAY = "gameplay"
    LEVEL_UP = "level_up"  # card 選択画面。meta ボタンはここで扱う
    FALLBACK = "fallback"  # fallback 拾得（chicken / gold）
    CHEST = "chest"
    CONFIRM = "confirm"
    UNKNOWN = "unknown"


class FallbackSemantic(str, enum.Enum):
    """fallback 拾得物の既知 semantic（chicken / gold）。"""

    CHICKEN = "chicken"
    GOLD = "gold"


class ButtonSemantic(str, enum.Enum):
    """UI ボタンの semantic（reroll / skip / banish / ack_chest / confirm）。"""

    REROLL = "reroll"
    SKIP = "skip"
    BANISH = "banish"
    ACK_CHEST = "ack_chest"
    CONFIRM = "confirm"


_META_KIND_BY_SEMANTIC = {
    ButtonSemantic.REROLL.value: UiIntentKind.REROLL,
    ButtonSemantic.SKIP.value: UiIntentKind.SKIP,
    ButtonSemantic.BANISH.value: UiIntentKind.BANISH,
}

_FALLBACK_TARGET_WIRE_KEYS = frozenset(
    {"target_id", "target_index", "semantic", "valid"}
)
_BUTTON_WIRE_KEYS = frozenset({"semantic", "capability"})
_CONFIG_WIRE_KEYS = frozenset(
    {
        "schema_version",
        "policy_id",
        "fallback_rule_id",
        "meta_rule_id",
        "ack_confirm_rule_id",
        "hp_chicken_threshold",
        "meta_policy_enabled",
        "meta_priority",
    }
)


@dataclass(frozen=True)
class FallbackTarget:
    """fallback 画面上の1候補（id・index・semantic・有効性）。"""

    target_id: str
    target_index: int
    semantic: str
    valid: bool

    def __post_init__(self) -> None:
        # from_wire 経由だけでなく、直接コンストラクタ生成時も検証する。
        ensure(
            isinstance(self.target_id, str) and self.target_id != "",
            "fallback target_id must be a non-empty string",
        )
        ensure(
            is_strict_int(self.target_index) and self.target_index >= 0,
            "fallback target_index must be a non-negative int",
        )
        ensure(
            isinstance(self.semantic, str) and self.semantic != "",
            "fallback semantic must be a non-empty string",
        )
        ensure(isinstance(self.valid, bool), "fallback valid must be a bool")

    def to_wire(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "target_index": self.target_index,
            "semantic": self.semantic,
            "valid": self.valid,
        }

    @classmethod
    def from_wire(cls, data: Mapping[str, Any]) -> "FallbackTarget":
        unknown = set(data) - _FALLBACK_TARGET_WIRE_KEYS
        ensure(not unknown, f"unknown FallbackTarget fields: {sorted(unknown)}")
        return cls(
            target_id=data.get("target_id"),
            target_index=data.get("target_index"),
            semantic=data.get("semantic"),
            valid=data.get("valid"),
        )


@dataclass(frozen=True)
class ButtonOption:
    """UI ボタン1個の semantic と操作可否（capability）。"""

    semantic: str
    capability: bool

    def __post_init__(self) -> None:
        ensure(
            isinstance(self.semantic, str) and self.semantic != "",
            "button semantic must be a non-empty string",
        )
        ensure(isinstance(self.capability, bool), "button capability must be a bool")

    def to_wire(self) -> dict[str, Any]:
        return {"semantic": self.semantic, "capability": self.capability}

    @classmethod
    def from_wire(cls, data: Mapping[str, Any]) -> "ButtonOption":
        unknown = set(data) - _BUTTON_WIRE_KEYS
        ensure(not unknown, f"unknown ButtonOption fields: {sorted(unknown)}")
        return cls(semantic=data.get("semantic"), capability=data.get("capability"))


_INPUT_WIRE_KEYS = frozenset(
    {
        "schema_version",
        "source_snapshot_hash",
        "source_frame_hash",
        "source_content_hash",
        "ui_state_key",
        "screen_state",
        "hp_fraction",
        "candidate_set_hash",
        "inventory_hash",
        "fallback_targets",
        "button",
    }
)


@dataclass(frozen=True)
class UiPolicyInputV1:
    """非モデル UI policy への入力。画面由来でモデル非依存のシグナルのみを持ち、ROI やモデル／教師値を含まない。"""

    source_snapshot_hash: str
    source_frame_hash: str
    source_content_hash: str
    ui_state_key: str
    screen_state: ScreenState
    hp_fraction: float
    candidate_set_hash: Optional[str] = None
    inventory_hash: Optional[str] = None
    fallback_targets: tuple[FallbackTarget, ...] = ()
    button: Optional[ButtonOption] = None
    schema_version: str = UI_POLICY_INPUT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.schema_version != UI_POLICY_INPUT_SCHEMA_VERSION:
            raise ContractValidationError(
                f"unsupported schema_version {self.schema_version!r}"
            )
        if not isinstance(self.screen_state, ScreenState):
            raise ContractValidationError("screen_state must be a ScreenState")
        for name in (
            "source_snapshot_hash",
            "source_frame_hash",
            "source_content_hash",
            "ui_state_key",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ContractValidationError(f"{name} must be a non-empty string")
        if (
            isinstance(self.hp_fraction, bool)
            or not isinstance(self.hp_fraction, (int, float))
            or not (0.0 <= float(self.hp_fraction) <= 1.0)
        ):
            raise ContractValidationError("hp_fraction must be a float in [0, 1]")
        for name in ("candidate_set_hash", "inventory_hash"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ContractValidationError(
                    f"{name} must be a non-empty string when present"
                )
        if not isinstance(self.fallback_targets, tuple):
            raise ContractValidationError("fallback_targets must be a tuple")
        for target in self.fallback_targets:
            if not isinstance(target, FallbackTarget):
                raise ContractValidationError(
                    "each fallback target must be a FallbackTarget"
                )
        if self.button is not None and not isinstance(self.button, ButtonOption):
            raise ContractValidationError("button must be a ButtonOption or None")

    def to_wire(self) -> dict[str, Any]:
        wire: dict[str, Any] = {
            "schema_version": self.schema_version,
            "source_snapshot_hash": self.source_snapshot_hash,
            "source_frame_hash": self.source_frame_hash,
            "source_content_hash": self.source_content_hash,
            "ui_state_key": self.ui_state_key,
            "screen_state": self.screen_state.value,
            "hp_fraction": float(self.hp_fraction),
            "fallback_targets": [t.to_wire() for t in self.fallback_targets],
        }
        if self.candidate_set_hash is not None:
            wire["candidate_set_hash"] = self.candidate_set_hash
        if self.inventory_hash is not None:
            wire["inventory_hash"] = self.inventory_hash
        if self.button is not None:
            wire["button"] = self.button.to_wire()
        return wire

    @classmethod
    def from_wire(cls, data: Mapping[str, Any]) -> "UiPolicyInputV1":
        unknown = set(data) - _INPUT_WIRE_KEYS
        if unknown:
            raise ContractValidationError(
                f"unknown UiPolicyInputV1 fields: {sorted(unknown)}"
            )
        require_schema_version(
            data, UI_POLICY_INPUT_SCHEMA_VERSION, "UiPolicyInputV1"
        )
        for name in (
            "source_snapshot_hash",
            "source_frame_hash",
            "source_content_hash",
            "ui_state_key",
        ):
            ensure(name in data, f"missing required UiPolicyInputV1 field: {name}")
        ensure(
            isinstance(data.get("screen_state"), str),
            "screen_state must be a string",
        )
        try:
            screen_state = ScreenState(data["screen_state"])
        except ValueError:
            raise ContractValidationError(
                f"unknown screen_state {data['screen_state']!r}"
            )
        ensure(
            is_strict_number(data.get("hp_fraction")),
            "hp_fraction must be a real number",
        )
        for name in ("candidate_set_hash", "inventory_hash"):
            value = data.get(name)
            ensure(
                value is None or (isinstance(value, str) and value != ""),
                f"{name} must be a non-empty string when present",
            )
        raw_fallbacks = data.get("fallback_targets", [])
        ensure(isinstance(raw_fallbacks, list), "fallback_targets must be a list")
        for target in raw_fallbacks:
            ensure(isinstance(target, Mapping), "each fallback target must be an object")
        button = data.get("button")
        ensure(
            button is None or isinstance(button, Mapping),
            "button must be an object when present",
        )
        return cls(
            source_snapshot_hash=data["source_snapshot_hash"],
            source_frame_hash=data["source_frame_hash"],
            source_content_hash=data["source_content_hash"],
            ui_state_key=data["ui_state_key"],
            screen_state=screen_state,
            hp_fraction=data["hp_fraction"],
            candidate_set_hash=data.get("candidate_set_hash"),
            inventory_hash=data.get("inventory_hash"),
            fallback_targets=tuple(FallbackTarget.from_wire(t) for t in raw_fallbacks),
            button=ButtonOption.from_wire(button) if button is not None else None,
            schema_version=data["schema_version"],
        )


@dataclass(frozen=True)
class NonModelUiPolicyConfigV1:
    """非モデル UI policy の pre-frozen 設定（rule id・chicken 閾値・meta 有効化・優先順位）。canonical config hash の基。"""

    policy_id: str = "non_model_ui_policy_v1"
    fallback_rule_id: str = "fallback_heuristic_v1"
    meta_rule_id: str = "meta_choice_heuristic_v1"
    ack_confirm_rule_id: str = "ack_confirm_rule_v1"
    hp_chicken_threshold: float = 0.70
    meta_policy_enabled: bool = False
    meta_priority: tuple[str, ...] = ()
    schema_version: str = NON_MODEL_UI_POLICY_CONFIG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        # from_wire 経由だけでなく、直接コンストラクタ生成時も検証する。
        ensure(
            self.schema_version == NON_MODEL_UI_POLICY_CONFIG_SCHEMA_VERSION,
            f"unsupported schema_version {self.schema_version!r}",
        )
        for name in (
            "policy_id",
            "fallback_rule_id",
            "meta_rule_id",
            "ack_confirm_rule_id",
        ):
            value = getattr(self, name)
            ensure(
                isinstance(value, str) and value != "",
                f"{name} must be a non-empty string",
            )
        ensure(
            is_strict_number(self.hp_chicken_threshold)
            and 0.0 <= float(self.hp_chicken_threshold) <= 1.0,
            "hp_chicken_threshold must be a real number in [0, 1]",
        )
        ensure(
            isinstance(self.meta_policy_enabled, bool),
            "meta_policy_enabled must be a bool",
        )
        ensure(
            isinstance(self.meta_priority, tuple)
            and all(isinstance(p, str) and p != "" for p in self.meta_priority),
            "meta_priority must be a tuple of non-empty strings",
        )

    def to_wire(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "policy_id": self.policy_id,
            "fallback_rule_id": self.fallback_rule_id,
            "meta_rule_id": self.meta_rule_id,
            "ack_confirm_rule_id": self.ack_confirm_rule_id,
            "hp_chicken_threshold": float(self.hp_chicken_threshold),
            "meta_policy_enabled": bool(self.meta_policy_enabled),
            "meta_priority": list(self.meta_priority),
        }

    @property
    def config_hash(self) -> str:
        return canonical_hash(self.to_wire())

    @classmethod
    def from_wire(cls, data: Mapping[str, Any]) -> "NonModelUiPolicyConfigV1":
        unknown = set(data) - _CONFIG_WIRE_KEYS
        ensure(
            not unknown,
            f"unknown NonModelUiPolicyConfigV1 fields: {sorted(unknown)}",
        )
        for key in _CONFIG_WIRE_KEYS:
            ensure(
                key in data,
                f"missing required NonModelUiPolicyConfigV1 field: {key}",
            )
        require_schema_version(
            data, NON_MODEL_UI_POLICY_CONFIG_SCHEMA_VERSION, "NonModelUiPolicyConfigV1"
        )
        priority = data["meta_priority"]
        ensure(isinstance(priority, list), "meta_priority must be a list")
        # フィールドの型／範囲の検証は __post_init__ が担う。
        return cls(
            policy_id=data["policy_id"],
            fallback_rule_id=data["fallback_rule_id"],
            meta_rule_id=data["meta_rule_id"],
            ack_confirm_rule_id=data["ack_confirm_rule_id"],
            hp_chicken_threshold=data["hp_chicken_threshold"],
            meta_policy_enabled=data["meta_policy_enabled"],
            meta_priority=tuple(priority),
            schema_version=data["schema_version"],
        )

    @classmethod
    def default_config(cls) -> "NonModelUiPolicyConfigV1":
        """パッケージ同梱の schema JSON から pre-frozen なデフォルト config を読み込む。"""
        text = (
            resources.files("reinbalance_survivors_contracts")
            .joinpath("schemas", _DEFAULT_CONFIG_RESOURCE)
            .read_text(encoding="utf-8")
        )
        return cls.from_wire(json.loads(text))

    @classmethod
    def load_default(cls) -> "NonModelUiPolicyConfigV1":
        """install済みpackage resourceから既定configを読み込む互換API。"""
        return cls.default_config()


def _intent(
    kind: UiIntentKind,
    semantic_action: str,
    owner: DecisionOwner,
    inp: UiPolicyInputV1,
    config: NonModelUiPolicyConfigV1,
    rule_id: str,
    *,
    target_id: Optional[str] = None,
    target_index: Optional[int] = None,
) -> UiIntentV1:
    return UiIntentV1(
        kind=kind,
        semantic_action=semantic_action,
        decision_owner=owner,
        source_snapshot_hash=inp.source_snapshot_hash,
        source_frame_hash=inp.source_frame_hash,
        source_content_hash=inp.source_content_hash,
        ui_state_key=inp.ui_state_key,
        inventory_hash=inp.inventory_hash,
        target_id=target_id,
        target_index=target_index,
        decision_policy_id=config.policy_id,
        decision_rule_id=rule_id,
        decision_config_hash=config.config_hash,
    )


_KNOWN_FALLBACK_SEMANTICS = frozenset(
    {FallbackSemantic.CHICKEN.value, FallbackSemantic.GOLD.value}
)


def _decide_fallback(
    inp: UiPolicyInputV1, config: NonModelUiPolicyConfigV1
) -> UiIntentV1:
    """``fallback_heuristic_v1``。

    chicken（valid）で HP <= 閾値 なら chicken、そうでなければ gold。2 つのうち片方
    だけ valid ならその target。valid 集合が空、semantic が重複、または既知の
    {chicken, gold} 以外の semantic を含む場合は fail-closed の ``stop``（stop は
    runtime safety が所有）。
    """
    valid = [t for t in inp.fallback_targets if t.valid]
    semantics = [t.semantic for t in valid]
    fail_closed = (
        len(valid) == 0
        or len(set(semantics)) != len(semantics)  # semantic の重複
        or any(s not in _KNOWN_FALLBACK_SEMANTICS for s in semantics)  # 未知
    )
    if fail_closed:
        return _intent(
            UiIntentKind.STOP,
            "stop",
            DecisionOwner.RUNTIME_SAFETY,
            inp,
            config,
            config.fallback_rule_id,
        )

    by_semantic = {t.semantic: t for t in valid}
    chicken = by_semantic.get(FallbackSemantic.CHICKEN.value)
    gold = by_semantic.get(FallbackSemantic.GOLD.value)

    if chicken is not None and gold is not None:
        target = chicken if inp.hp_fraction <= config.hp_chicken_threshold else gold
    elif chicken is not None:
        target = chicken
    else:  # gold のみ残る（fail-closed ガードが空／未知集合を既に拒否済み）
        target = gold

    # 具体的な target identity（target_id）とその semantic を保持する。
    return _intent(
        UiIntentKind.CHOOSE_FALLBACK,
        target.semantic,
        DecisionOwner.NON_MODEL_UI_POLICY,
        inp,
        config,
        config.fallback_rule_id,
        target_id=target.target_id,
        target_index=target.target_index,
    )


def _decide_meta(
    inp: UiPolicyInputV1, config: NonModelUiPolicyConfigV1
) -> Optional[UiIntentV1]:
    """``meta_choice_heuristic_v1``。

    デフォルトでは無効：reroll/skip/banish を一切生成しない。有効時は、入力の button
    capability と pre-frozen な優先順位リストのみを使う。
    """
    if not config.meta_policy_enabled:
        return None
    button = inp.button
    if button is None or not button.capability:
        return None
    if button.semantic not in config.meta_priority:
        return None
    kind = _META_KIND_BY_SEMANTIC.get(button.semantic)
    if kind is None:
        return None
    return _intent(
        kind,
        button.semantic,
        DecisionOwner.NON_MODEL_UI_POLICY,
        inp,
        config,
        config.meta_rule_id,
    )


def decide_non_model_ui_intent(
    inp: UiPolicyInputV1, config: NonModelUiPolicyConfigV1
) -> Optional[UiIntentV1]:
    """``config`` の下で ``inp`` に対する非モデル UI intent を返す。該当なしなら ``None``。

    ``None`` は policy が判断を見送ったこと（適用可能な非モデルアクションがない）を意味し、
    次に何をするかは runtime safety が決める。fail-closed の ``stop`` intent は、空／曖昧な
    fallback 集合に対して fallback ルールが返す場合に限られる。
    """
    state = inp.screen_state
    if state is ScreenState.CHEST:
        return _intent(
            UiIntentKind.ACK_CHEST,
            "ack_chest",
            DecisionOwner.NON_MODEL_UI_POLICY,
            inp,
            config,
            config.ack_confirm_rule_id,
        )
    if state is ScreenState.CONFIRM:
        return _intent(
            UiIntentKind.CONFIRM,
            "confirm",
            DecisionOwner.NON_MODEL_UI_POLICY,
            inp,
            config,
            config.ack_confirm_rule_id,
        )
    if state is ScreenState.FALLBACK:
        return _decide_fallback(inp, config)
    if state is ScreenState.LEVEL_UP:
        return _decide_meta(inp, config)
    # GAMEPLAY / UNKNOWN -> policy は判断を見送る（曖昧／適用外）。
    return None
