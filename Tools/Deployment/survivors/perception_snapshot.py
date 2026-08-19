"""操作用 UI とモデル観測を一つの perception snapshot に束縛する。

ROI はクリック対象だけに保持し、release tensor や diagnostics へ流れない境界を提供します。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, Mapping

from reinbalance_survivors_contracts.canonical_json import canonical_hash
from reinbalance_survivors_contracts.deploy_obs import DeployObservation
from reinbalance_survivors_contracts.item_decision import CandidateFeatures, ItemDecisionFeatures

from .vision.hud_parser import HudStateV1, ParsedCard

UI_PRESENTATION_SCHEMA_VERSION = "ui_presentation_snapshot.v1"
UI_PROFILE_VERSION = "survivors_ui_profile.v1"
UI_PRESENTATION_SCHEMA_HASH = canonical_hash(
    {
        "schema_version": UI_PRESENTATION_SCHEMA_VERSION,
        "candidate_fields": ["choice_id", "choice_index", "semantic_kind", "roi", "validity", "confidence"],
        "button_fields": ["semantic_action", "roi", "validity", "capability", "confidence"],
    }
)
_BUTTON_ACTIONS = frozenset({"ack_chest", "confirm", "reroll", "skip", "banish"})


def _finite_ratio(value: Any, label: str) -> float:
    """有限な [0,1] 比率を検証して float 化する。

    NaN や bool を座標・信頼度として受け入れず、hash 前に表現を統一します。
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{label} must be finite")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be in [0, 1]")
    return result


@dataclass(frozen=True)
class NormalizedRoi:
    """viewport 正規化済みの UI 矩形を表す。

    無効 target のゼロ矩形を許しつつ、画面外や反転した矩形は拒否します。
    """

    left: float; top: float; right: float; bottom: float

    def __post_init__(self) -> None:
        """四辺の範囲と向きを検証する。

        hash や IoU が不定になる不正座標を生成直後に止めます。
        """
        values = tuple(_finite_ratio(value, "roi coordinate") for value in (self.left, self.top, self.right, self.bottom))
        if values[0] > values[2] or values[1] > values[3]:
            raise ValueError("roi edges must be ordered")
        for name, value in zip(("left", "top", "right", "bottom"), values, strict=True):
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class UiCandidateTargetV1:
    """候補カード一件の semantic identity と操作矩形を保持する。

    model candidate と ROI を同一型にせず、入力実行側だけが座標へ触れられるようにします。
    """

    choice_id: str; choice_index: int
    semantic_kind: Literal["item_card", "fallback_reward"]
    roi: NormalizedRoi; validity: bool; confidence: float

    def __post_init__(self) -> None:
        """候補 identity・型・信頼度を検証する。

        空 identity や未知 semantic をクリック対象として公開しません。
        """
        if not isinstance(self.choice_id, str) or not self.choice_id:
            raise ValueError("choice_id must be non-empty str")
        if type(self.choice_index) is not int or self.choice_index < 0:
            raise ValueError("choice_index must be non-negative int")
        if self.semantic_kind not in {"item_card", "fallback_reward"}:
            raise ValueError("invalid candidate semantic_kind")
        if not isinstance(self.roi, NormalizedRoi) or type(self.validity) is not bool:
            raise ValueError("invalid candidate target")
        object.__setattr__(self, "confidence", _finite_ratio(self.confidence, "candidate confidence"))


@dataclass(frozen=True)
class UiButtonTargetV1:
    """非カード UI button の semantic action と操作矩形を保持する。

    capability と検出 validity を分け、表示されても実行不能な操作を区別します。
    """

    semantic_action: Literal["ack_chest", "confirm", "reroll", "skip", "banish"]
    roi: NormalizedRoi; validity: bool; capability: bool; confidence: float

    def __post_init__(self) -> None:
        """button semantic・型・信頼度を検証する。

        未知の入力操作や曖昧な bool 表現を action runtime へ渡しません。
        """
        if self.semantic_action not in _BUTTON_ACTIONS:
            raise ValueError("invalid button semantic_action")
        if not isinstance(self.roi, NormalizedRoi) or type(self.validity) is not bool or type(self.capability) is not bool:
            raise ValueError("invalid button target")
        object.__setattr__(self, "confidence", _finite_ratio(self.confidence, "button confidence"))


@dataclass(frozen=True)
class UiPresentationSnapshotV1:
    """一フレーム分の UI presentation を immutable に保持する。

    semantic hash と連続値込み source hash を分離し、安全な cross-frame 比較を可能にします。
    """

    schema_hash: str; snapshot_id: str; frame_id: str; parser_artifact_hash: str
    screen_state: str; candidate_set_hash: str; inventory_hash: str
    source_content_hash: str; ui_state_key: str
    candidates: tuple[UiCandidateTargetV1, ...]; buttons: tuple[UiButtonTargetV1, ...]

    def __post_init__(self) -> None:
        """identity と target tuple の整合性・一意性を検証する。

        同一 semantic button の二重クリックや、別 parser の取り違えを入口で防ぎます。
        """
        for name in ("schema_hash", "parser_artifact_hash", "candidate_set_hash", "inventory_hash", "source_content_hash", "ui_state_key"):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"invalid {name}")
        if self.schema_hash != UI_PRESENTATION_SCHEMA_HASH:
            raise ValueError("unsupported UI presentation schema")
        if not all(isinstance(value, str) and value for value in (self.snapshot_id, self.frame_id, self.screen_state)):
            raise ValueError("invalid UI presentation identity")
        if not isinstance(self.candidates, tuple) or not all(isinstance(value, UiCandidateTargetV1) for value in self.candidates):
            raise ValueError("candidates must be candidate tuple")
        if not isinstance(self.buttons, tuple) or not all(isinstance(value, UiButtonTargetV1) for value in self.buttons):
            raise ValueError("buttons must be button tuple")
        actions = [button.semantic_action for button in self.buttons]
        if len(actions) != len(set(actions)):
            raise ValueError("button semantics must be unique")


def _diagnostics_has_roi(value: Any) -> bool:
    """diagnostics 内に ROI 型や ROI 名のキーがあるか調べる。

    nested mapping・sequence も再帰的に確認し、浅い allow-list の迂回を防ぎます。
    """
    if isinstance(value, (NormalizedRoi, UiCandidateTargetV1, UiButtonTargetV1, UiPresentationSnapshotV1)):
        return True
    if isinstance(value, Mapping):
        return any("roi" in str(key).casefold() or _diagnostics_has_roi(item) for key, item in value.items())
    if isinstance(value, (tuple, list)):
        return any(_diagnostics_has_roi(item) for item in value)
    return False


@dataclass(frozen=True)
class PerceptionSnapshot:
    """policy 観測と操作用 UI を atomic に束縛する perception 結果。

    呼び出し側は別 HudState を参照せず、この一オブジェクトから同じ frame の情報を取得します。
    """

    snapshot_id: str; frame_id: str; captured_ns: int; parser_artifact_hash: str
    source_content_hash: str; ui_state_key: str; screen_state: str
    deploy_obs: DeployObservation; item_context: ItemDecisionFeatures | None
    choices: tuple[CandidateFeatures, ...]; ui_presentation: UiPresentationSnapshotV1
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        """atomic identity と ROI leakage 禁止を検証する。

        presentation が別 frame の場合や diagnostics に座標が混じる場合は snapshot を成立させません。
        """
        if not all(isinstance(value, str) and value for value in (self.snapshot_id, self.frame_id, self.screen_state)):
            raise ValueError("invalid perception identity")
        if type(self.captured_ns) is not int or self.captured_ns < 0:
            raise ValueError("captured_ns must be non-negative int")
        if not isinstance(self.deploy_obs, DeployObservation) or not isinstance(self.ui_presentation, UiPresentationSnapshotV1):
            raise ValueError("invalid perception payload")
        if self.item_context is not None and not isinstance(self.item_context, ItemDecisionFeatures):
            raise ValueError("invalid item_context")
        if not isinstance(self.choices, tuple) or not all(isinstance(value, CandidateFeatures) for value in self.choices):
            raise ValueError("choices must be CandidateFeatures tuple")
        ui = self.ui_presentation
        bindings = (
            (self.snapshot_id, ui.snapshot_id), (self.frame_id, ui.frame_id),
            (self.parser_artifact_hash, ui.parser_artifact_hash),
            (self.source_content_hash, ui.source_content_hash), (self.ui_state_key, ui.ui_state_key),
            (self.screen_state, ui.screen_state),
        )
        if any(left != right for left, right in bindings):
            raise ValueError("UI presentation is not atomically bound")
        if not isinstance(self.diagnostics, Mapping) or _diagnostics_has_roi(self.diagnostics):
            raise ValueError("diagnostics must not contain ROI")
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))


def _normalized_roi(roi_xyxy: tuple[int, int, int, int] | None, viewport: tuple[int, int]) -> tuple[NormalizedRoi, bool]:
    """pixel ROI を viewport 正規化して geometry validity も返す。

    欠損・画面外・面積ゼロはゼロ矩形へ閉じ、クリック可能とは扱いません。
    """
    width, height = viewport
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        raise ValueError("viewport must contain positive integers")
    if roi_xyxy is None or len(roi_xyxy) != 4 or any(type(value) is not int for value in roi_xyxy):
        return NormalizedRoi(0., 0., 0., 0.), False
    left, top, right, bottom = roi_xyxy
    clipped = (
        max(0., min(1., left / width)), max(0., min(1., top / height)),
        max(0., min(1., right / width)), max(0., min(1., bottom / height)),
    )
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        return NormalizedRoi(0., 0., 0., 0.), False
    return NormalizedRoi(*clipped), True


def _candidate_semantic(card: ParsedCard) -> Literal["item_card", "fallback_reward"]:
    """card を item または fallback reward へ分類する。

    parser kind に加えて gold/chicken 系 identity も fail-safe に fallback とみなします。
    """
    identity = (card.item_id or "").casefold()
    return "fallback_reward" if card.kind == "fallback" or "gold" in identity or "chicken" in identity else "item_card"


def _quantized_target(target: UiCandidateTargetV1 | UiButtonTargetV1) -> dict[str, Any]:
    """UI target を 1e-4 量子化済み hash payload にする。

    detector のサブ量子揺れで source identity が不要に変わらないようにします。
    """
    roi = [round(value, 4) for value in (target.roi.left, target.roi.top, target.roi.right, target.roi.bottom)]
    common = {"roi": roi, "validity": target.validity, "confidence": round(target.confidence, 4)}
    if isinstance(target, UiCandidateTargetV1):
        return {"choice_id": target.choice_id, "choice_index": target.choice_index, "semantic_kind": target.semantic_kind, **common}
    return {"semantic_action": target.semantic_action, "capability": target.capability, **common}


def build_ui_presentation_from_hud(
    hud: HudStateV1,
    viewport: tuple[int, int],
    *,
    snapshot_id: str | None = None,
    frame_id: str | None = None,
) -> UiPresentationSnapshotV1:
    """HudStateV1 から versioned UI presentation を構築する。

    semantic identity と量子化した操作座標を別々の canonical hash へまとめます。
    """
    if not isinstance(hud, HudStateV1):
        raise TypeError("hud must be HudStateV1")
    if not isinstance(viewport, tuple) or len(viewport) != 2 or any(type(value) is not int or value <= 0 for value in viewport):
        raise ValueError("viewport must be a positive integer pair")
    resolved_frame = frame_id or f"{hud.session_id}:{hud.frame_index}"
    resolved_snapshot = snapshot_id or canonical_hash(
        {"session_id": hud.session_id, "frame_id": resolved_frame, "captured_ns": hud.captured_monotonic_ns}
    )
    candidates = []
    for card in sorted(hud.cards, key=lambda value: value.slot_index):
        roi, geometry_valid = _normalized_roi(card.roi_xyxy, viewport)
        candidates.append(
            UiCandidateTargetV1(
                card.item_id or f"unknown:{card.slot_index}", card.slot_index,
                _candidate_semantic(card), roi, geometry_valid and card.confidence >= .35,
                card.confidence,
            )
        )
    capabilities = {
        "reroll": hud.reroll_available, "skip": hud.skip_available,
        "banish": hud.banish_available, "ack_chest": True, "confirm": True,
    }
    best_buttons: dict[str, UiButtonTargetV1] = {}
    order: list[str] = []
    for parsed in hud.buttons:
        action = "ack_chest" if parsed.button_type == "chest" else parsed.button_type
        if action not in _BUTTON_ACTIONS:
            raise ValueError(f"unsupported button action: {action}")
        roi, geometry_valid = _normalized_roi(parsed.roi_xyxy, viewport)
        target = UiButtonTargetV1(
            action, roi, geometry_valid and parsed.confidence >= .35,
            capabilities[action], parsed.confidence,
        )
        if action not in best_buttons:
            order.append(action)
        if action not in best_buttons or target.confidence > best_buttons[action].confidence:
            best_buttons[action] = target
    buttons = tuple(best_buttons[action] for action in order)
    candidate_tuple = tuple(candidates)
    source_payload = {
        "schema_hash": UI_PRESENTATION_SCHEMA_HASH, "snapshot_id": resolved_snapshot,
        "frame_id": resolved_frame, "parser_artifact_hash": hud.parser_artifact_hash,
        "timestamp_ns": hud.captured_monotonic_ns, "screen_state": hud.screen_state,
        "candidate_set_hash": hud.candidate_set_hash, "inventory_hash": hud.inventory_hash,
        "candidates": [_quantized_target(value) for value in candidate_tuple],
        "buttons": [_quantized_target(value) for value in buttons],
    }
    state_payload = {
        "screen_state": hud.screen_state, "schema_hash": UI_PRESENTATION_SCHEMA_HASH,
        "profile": UI_PROFILE_VERSION, "parser_artifact_hash": hud.parser_artifact_hash,
        "candidate_set_hash": hud.candidate_set_hash, "inventory_hash": hud.inventory_hash,
        "candidates": [
            {"choice_id": value.choice_id, "choice_index": value.choice_index, "semantic_kind": value.semantic_kind}
            for value in candidate_tuple
        ],
        "buttons": [
            {"semantic_action": value.semantic_action, "capability": value.capability}
            for value in buttons
        ],
    }
    return UiPresentationSnapshotV1(
        UI_PRESENTATION_SCHEMA_HASH, resolved_snapshot, resolved_frame, hud.parser_artifact_hash,
        hud.screen_state, hud.candidate_set_hash, hud.inventory_hash,
        canonical_hash(source_payload), canonical_hash(state_payload), candidate_tuple, buttons,
    )


def _target_identity(target: UiCandidateTargetV1 | UiButtonTargetV1) -> tuple[Any, ...]:
    """同値比較に使う semantic identity を返す。

    ROI や信頼度を identity へ混ぜず、候補と button を交差一致させません。
    """
    if isinstance(target, UiCandidateTargetV1):
        return ("candidate", target.choice_id, target.choice_index, target.semantic_kind)
    if isinstance(target, UiButtonTargetV1):
        return ("button", target.semantic_action)
    raise TypeError("unsupported UI target")


def is_equivalent_ui_target(
    old: UiCandidateTargetV1 | UiButtonTargetV1,
    new: UiCandidateTargetV1 | UiButtonTargetV1,
    *,
    old_captured_ns: int | None = None,
    new_captured_ns: int | None = None,
) -> bool:
    """二フレームの UI target が安全に同一とみなせる場合だけ True を返す。

    semantic・時系列・validity・confidence・IoU・中心移動の全 gate を AND で適用します。
    """
    if (old_captured_ns is None) != (new_captured_ns is None):
        return False
    if old_captured_ns is not None and (type(old_captured_ns) is not int or type(new_captured_ns) is not int or new_captured_ns <= old_captured_ns):
        return False
    if _target_identity(old) != _target_identity(new):
        return False
    if not old.validity or not new.validity or old.confidence < .99 or new.confidence < .99:
        return False
    intersection_width = max(0., min(old.roi.right, new.roi.right) - max(old.roi.left, new.roi.left))
    intersection_height = max(0., min(old.roi.bottom, new.roi.bottom) - max(old.roi.top, new.roi.top))
    intersection = intersection_width * intersection_height
    old_area = (old.roi.right - old.roi.left) * (old.roi.bottom - old.roi.top)
    new_area = (new.roi.right - new.roi.left) * (new.roi.bottom - new.roi.top)
    union = old_area + new_area - intersection
    if union <= 0 or intersection / union < .90:
        return False
    old_center = ((old.roi.left + old.roi.right) / 2., (old.roi.top + old.roi.bottom) / 2.)
    new_center = ((new.roi.left + new.roi.right) / 2., (new.roi.top + new.roi.bottom) / 2.)
    displacement = math.hypot(new_center[0] - old_center[0], new_center[1] - old_center[1])
    return displacement <= .01
