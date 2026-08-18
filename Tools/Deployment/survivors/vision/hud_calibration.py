"""Survivors HUD parser の calibration・validation・package 契約。

学習用と検証用 split を混ぜずに決定的な ROI calibration と11種類の gate を計算し、
開発専用 parser package を内容 hash 付きの canonical JSON として atomic publish します。
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Final

from reinbalance_survivors_contracts.canonical_json import canonical_hash, canonical_json_bytes

FIT_SCHEMA_VERSION: Final[str] = "hud_calibration_fit.v1"
REPORT_SCHEMA_VERSION: Final[str] = "hud_calibration_report.v1"
PACKAGE_SCHEMA_VERSION: Final[str] = "hud_parser_package.v1"
_MODEL_SPLITS: Final[frozenset[str]] = frozenset({"model_train", "model_validation"})

class SplitViolationError(ValueError):
    """HUD calibration の split 分離違反を表す例外。

train、validation、error calibration、final test の間でデータ用途が混ざった時に送出します。
    """

class FormalDatasetRequired(ValueError):
    """formal 実行に正式 dataset が必要であることを表す例外。

synthetic または未昇格 dataset を正式成果物の根拠にしようとした時に送出します。
    """

class FormalPackageMixingError(ValueError):
    """formal package と development package の経路混在を表す例外。

開発専用 metadata を formal loader へ渡すなど、package の用途属性が経路と違う時に送出します。
    """

class PackageHashMismatch(ValueError):
    """parser package の canonical hash 不一致を表す例外。

保存後の payload 改ざん、破損、または内部成果物 hash の不整合を検出した時に送出します。
    """

@dataclass(frozen=True, slots=True)
class HudCalibrationAnnotation:
    """1フレーム分の HUD 正解値と parser 出力を保持する注釈。

fit は ROI と split を使い、validate は正解と予測の組から固定 gate の値を集計します。
    """

    sample_id: str
    split: str
    expected_timer_seconds: float | None = None; predicted_timer_seconds: float | None = None
    expected_level: int | None = None; predicted_level: int | None = None
    expected_hp_ratio: float | None = None; predicted_hp_ratio: float | None = None
    expected_xp_ratio: float | None = None; predicted_xp_ratio: float | None = None
    expected_items: tuple[str | None, ...] | None = None; predicted_items: tuple[str | None, ...] | None = None
    expected_choice: str | None = None; predicted_choice: str | None = None
    expected_screen_state: str | None = None; predicted_screen_state: str | None = None
    latency_ms: float | None = None
    expected_roi: tuple[float, float, float, float] | None = None; predicted_roi: tuple[float, float, float, float] | None = None
    roi_name: str = "hud"

@dataclass(frozen=True, slots=True)
class CalibrationConfig:
    """validation 前に固定する11 gate の閾値設定。

全項目は immutable で、validate は観測結果から閾値を調整せずこの値をそのまま使います。
    """

    timer_exact_floor: float = 0.95; level_floor: float = 0.95
    hp_mae_ceiling: float = 0.05; xp_mae_ceiling: float = 0.05
    item_floor: float = 0.95; choice_floor: float = 0.95
    screen_state_f1_floor: float = 0.95; latency_ceiling_ms: float = 50.0
    roi_center_error_ceiling: float = 5.0; roi_inside_rate_floor: float = 0.95
    roi_false_positive_ceiling: float = 0.05

    def __post_init__(self) -> None:
        """全閾値が有限かつ指標の有効範囲内であることを検査する。

割合は0から1、時間と距離は0以上に制限し、不正 config を評価開始前に拒否します。
        """
        ratio_names = {
            "timer_exact_floor", "level_floor", "hp_mae_ceiling", "xp_mae_ceiling",
            "item_floor", "choice_floor", "screen_state_f1_floor",
            "roi_inside_rate_floor", "roi_false_positive_ceiling",
        }
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{item.name} must be finite")
            if item.name in ratio_names and not 0.0 <= value <= 1.0:
                raise ValueError(f"{item.name} must be between 0 and 1")
            if item.name not in ratio_names and value < 0.0:
                raise ValueError(f"{item.name} must be non-negative")

    def to_wire(self) -> dict[str, float]:
        """固定閾値を canonical JSON 用の辞書へ変換する。

dataclass の宣言順ではなく field 名を保持し、report identity へ閾値自体を封入します。
        """
        return {item.name: float(getattr(self, item.name)) for item in fields(self)}

@dataclass(frozen=True, slots=True)
class GateResult:
    """単一 metric の値、閾値、方向、合否を保持する結果。

floor は値が閾値以上、ceiling は値が閾値以下の時に境界を含めて合格します。
    """

    name: str; value: float; threshold: float
    direction: str; passed: bool

    @classmethod
    def floor_gate(
        cls, name: str = "floor", *, value: float, floor: float | None = None
    ) -> "GateResult":
        """下限 gate を境界値込みで評価する。

floor を省略した場合は value 自身を閾値とし、単独での境界動作確認にも利用できます。
        """
        threshold = value if floor is None else floor
        _require_finite(value, "value")
        _require_finite(threshold, "floor")
        return cls(name, float(value), float(threshold), "floor", value >= threshold)

    @classmethod
    def ceiling_gate(
        cls, name: str = "ceiling", *, value: float, ceiling: float | None = None
    ) -> "GateResult":
        """上限 gate を境界値込みで評価する。

ceiling を省略した場合は value 自身を閾値とし、単独での境界動作確認にも利用できます。
        """
        threshold = value if ceiling is None else ceiling
        _require_finite(value, "value")
        _require_finite(threshold, "ceiling")
        return cls(name, float(value), float(threshold), "ceiling", value <= threshold)

    def to_wire(self) -> dict[str, object]:
        """gate 結果を canonical JSON 用の辞書へ変換する。

合否も保存して loader 側で比較式の再解釈をせず、発行時の結果を復元します。
        """
        return {
            "name": self.name, "value": self.value, "threshold": self.threshold,
            "direction": self.direction, "passed": self.passed,
        }

@dataclass(frozen=True, slots=True)
class FitResult:
    """train 注釈から得た決定的な ROI calibration 成果物。

annotation 内容 hash と ROI 名ごとの平均矩形をまとめ、成果物全体の canonical hash を保持します。
    """

    schema_version: str; annotation_count: int; annotation_canonical_hash: str
    roi_anchors: tuple[tuple[str, tuple[float, float, float, float]], ...]
    fit_canonical_hash: str

    def to_wire(self) -> dict[str, object]:
        """fit 成果物を package 保存用の辞書へ変換する。

tuple は wire 上で配列へ変え、プラットフォームに依存しない JSON 表現にします。
        """
        return {
            "schema_version": self.schema_version,
            "annotation_count": self.annotation_count,
            "annotation_canonical_hash": self.annotation_canonical_hash,
            "roi_anchors": [[name, list(roi)] for name, roi in self.roi_anchors],
            "fit_canonical_hash": self.fit_canonical_hash,
        }

@dataclass(frozen=True, slots=True)
class ValidationReport:
    """固定 config で計算した11 gate と report identity。

package loader が再検証できるよう、sample 数、使用閾値、全 gate、canonical hash を保持します。
    """

    schema_version: str; sample_count: int
    config: CalibrationConfig
    gates: tuple[GateResult, ...]
    report_canonical_hash: str

    @property
    def gates_by_name(self) -> dict[str, GateResult]:
        """gate 名から結果を引ける新しい辞書を返す。

保存順序を変えず、利用側だけが11指標へ名前でアクセスできるようにします。
        """
        return {gate.name: gate for gate in self.gates}

    @property
    def all_passed(self) -> bool:
        """全 gate が合格した時だけ True を返す。

空 gate は有効 report ではないため、必ず11件あることも合わせて確認します。
        """
        return len(self.gates) == 11 and all(gate.passed for gate in self.gates)

    def to_wire(self) -> dict[str, object]:
        """validation report を package 保存用の辞書へ変換する。

固定 config を report 内へ封入し、後から結果に合わせて閾値を差し替えられない形にします。
        """
        return {
            "schema_version": self.schema_version,
            "sample_count": self.sample_count,
            "config": self.config.to_wire(),
            "gates": [gate.to_wire() for gate in self.gates],
            "report_canonical_hash": self.report_canonical_hash,
        }

@dataclass(frozen=True, slots=True)
class ParserPackageMeta:
    """HUD parser package の用途属性と内容 identity。

04-04 が発行する値は常に development_only=true、formal_parser_eligible=false です。
    """

    schema_version: str; development_only: bool; formal_parser_eligible: bool
    fit_canonical_hash: str; report_canonical_hash: str
    package_hash: str

    def __post_init__(self) -> None:
        """矛盾した用途属性と不正 hash を構築時に拒否する。

development package が formal eligible を同時に名乗る状態をデータ型でも封じます。
        """
        if self.schema_version != PACKAGE_SCHEMA_VERSION:
            raise ValueError(f"unsupported package schema: {self.schema_version!r}")
        if type(self.development_only) is not bool or type(self.formal_parser_eligible) is not bool:
            raise ValueError("package eligibility flags must be bool")
        if self.development_only and self.formal_parser_eligible:
            raise FormalPackageMixingError("development package cannot be formal eligible")
        for value in (self.fit_canonical_hash, self.report_canonical_hash, self.package_hash):
            if not _is_sha256(value):
                raise ValueError("package metadata hashes must be SHA-256 hex")

@dataclass(frozen=True, slots=True)
class ParserPackage:
    """復元済み metadata、fit、validation report の集合。

HudParser へ渡す artifact identity は package_hash property から取得します。
    """

    meta: ParserPackageMeta
    fit_result: FitResult
    validation_report: ValidationReport

    @property
    def package_hash(self) -> str:
        """metadata に封入された package の canonical hash を返す。

呼び出し側が内部 metadata 構造を知らずに parser identity として利用できます。
        """
        return self.meta.package_hash

def _require_finite(value: object, field: str) -> float:
    """bool ではない有限数を float として返す。

NaN や infinity が canonical JSON と gate 比較へ侵入する前に fail-closed で拒否します。
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{field} must be finite")
    return float(value)

def _is_sha256(value: object) -> bool:
    """値が64桁の小文字 SHA-256 hex なら True を返す。

package metadata の hash 表記を一意にし、大文字や長さ違いを拒否します。
    """
    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)

def _field(record: object, name: str, default: Any = None) -> Any:
    """dataclass または mapping の注釈から1 field を取得する。

capture tool が読み込んだ辞書と型付き HudCalibrationAnnotation の両方を同じ経路で扱います。
    """
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)

def _roi(record: object, expected: bool) -> tuple[float, float, float, float] | None:
    """注釈から検証済み ROI を取得する。

fit では既存 capture annotation の hud_roi/bbox も正解 ROI の互換入力として受け付けます。
    """
    value = _field(record, "expected_roi" if expected else "predicted_roi")
    if expected and value is None:
        value = _field(record, "hud_roi", _field(record, "bbox"))
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("ROI must contain four coordinates")
    roi = tuple(_require_finite(item, "ROI coordinate") for item in value)
    if not (roi[0] < roi[2] and roi[1] < roi[3]):
        raise ValueError("ROI must have positive width and height")
    return roi  # type: ignore[return-value]

def validate_split_eligibility(split: str) -> None:
    """model calibration へ利用可能な split 名だけを受け入れる。

error_calibration、final_e2e_test、未知名は学習・validation の共通入口で拒否します。
    """
    if split not in _MODEL_SPLITS:
        raise SplitViolationError(f"split is not model-eligible: {split!r}")

def reject_formal_if_synthetic(*, formal_mode: bool, formal_dataset_eligible: bool) -> None:
    """formal mode と dataset eligibility の組を検査する。

formal mode で未昇格 dataset を使う場合だけ FormalDatasetRequired を送出します。
    """
    if type(formal_mode) is not bool or type(formal_dataset_eligible) is not bool:
        raise TypeError("formal flags must be bool")
    if formal_mode and not formal_dataset_eligible:
        raise FormalDatasetRequired("formal mode requires a formal dataset")

def _effective_split(record: object, explicit: str | None, expected: str) -> str:
    """record と呼び出し引数から矛盾のない split を決定する。

どちらにも用途証明がない場合と、両方が指定されて食い違う場合を用途混在として拒否します。
    """
    record_split = _field(record, "split")
    if explicit is None and record_split is None:
        raise SplitViolationError("annotation split or explicit split is required")
    if explicit is not None and record_split is not None and explicit != record_split:
        raise SplitViolationError("annotation split conflicts with explicit split")
    split = explicit if explicit is not None else record_split
    validate_split_eligibility(split)
    if split != expected:
        raise SplitViolationError(f"expected {expected!r}, got {split!r}")
    return split

def fit(
    annotations: Sequence[object], *, split: str | None = None,
    formal_mode: bool = False, formal_dataset_eligible: bool = False,
) -> FitResult:
    """model_train 注釈から決定的な ROI anchor を fitting する。

入力順に依存しない canonical annotation hash と ROI 名ごとの平均矩形を FitResult に封入します。
    """
    if not annotations:
        raise ValueError("fit requires at least one annotation")
    reject_formal_if_synthetic(
        formal_mode=formal_mode, formal_dataset_eligible=formal_dataset_eligible
    )
    grouped: dict[str, list[tuple[float, float, float, float]]] = {}
    identities: list[dict[str, object]] = []
    for index, annotation in enumerate(annotations):
        _effective_split(annotation, split, "model_train")
        roi = _roi(annotation, expected=True)
        name = _field(annotation, "roi_name", _field(annotation, "class_name", "hud"))
        if not isinstance(name, str) or not name:
            raise ValueError("roi_name must be non-empty")
        sample_id = str(_field(annotation, "sample_id", _field(annotation, "frame_id", index)))
        identities.append({"sample_id": sample_id, "roi_name": name, "roi": roi})
        if roi is not None:
            grouped.setdefault(name, []).append(roi)
    identities.sort(key=canonical_hash)
    # canonical_hash で ROI をソートしてから fsum で集計し、入力順非依存にする。
    anchors = tuple(
        (name, tuple(math.fsum(roi[i] for roi in sorted(values, key=canonical_hash)) / len(values) for i in range(4)))
        for name, values in sorted(grouped.items())
    )
    identity = {
        "schema_version": FIT_SCHEMA_VERSION,
        "annotation_count": len(annotations),
        "annotation_canonical_hash": canonical_hash(identities),
        "roi_anchors": [[name, list(roi)] for name, roi in anchors],
    }
    return FitResult(
        FIT_SCHEMA_VERSION, len(annotations), identity["annotation_canonical_hash"],
        anchors, canonical_hash(identity),
    )

def _exact_rate(records: Sequence[object], expected_name: str, predicted_name: str) -> float:
    """正解値があるレコードの完全一致率を返す。

正解が1件もない場合は gate を暗黙合格させず0を返し、欠測 prediction も不一致に数えます。
    """
    pairs = [(_field(item, expected_name), _field(item, predicted_name)) for item in records]
    pairs = [pair for pair in pairs if pair[0] is not None]
    return sum(expected == predicted for expected, predicted in pairs) / len(pairs) if pairs else 0.0

def _mae(records: Sequence[object], expected_name: str, predicted_name: str) -> float:
    """正解値があるレコードの平均絶対誤差を返す。

prediction 欠測は比率指標の最大誤差1として扱い、正解がなければ fail-closed の1を返します。
    """
    errors: list[float] = []
    for item in records:
        expected = _field(item, expected_name)
        if expected is None:
            continue
        expected_value = _require_finite(expected, expected_name)
        predicted = _field(item, predicted_name)
        errors.append(1.0 if predicted is None else abs(expected_value - _require_finite(predicted, predicted_name)))
    return sum(errors) / len(errors) if errors else 1.0

def _item_rate(records: Sequence[object]) -> float:
    """inventory の全 annotated slot に対する item 一致率を返す。

prediction の長さ不足は欠測 slot として不一致にし、余分な slot も誤検出として数えます。
    """
    correct = total = 0
    missing = object()
    for item in records:
        expected = _field(item, "expected_items")
        if expected is None:
            continue
        predicted = _field(item, "predicted_items")
        predicted_values = tuple(predicted) if isinstance(predicted, (list, tuple)) else ()
        expected_values = tuple(expected)
        for index in range(max(len(expected_values), len(predicted_values))):
            total += 1
            left = expected_values[index] if index < len(expected_values) else missing
            right = predicted_values[index] if index < len(predicted_values) else missing
            correct += left == right
    return correct / total if total else 0.0

def _macro_f1(records: Sequence[object]) -> float:
    """screen state の deterministic macro F1 を返す。

正解と予測に現れる label の和集合を辞書順で走査し、欠測 prediction は専用 label とします。
    """
    pairs = [(_field(item, "expected_screen_state"), _field(item, "predicted_screen_state")) for item in records]
    pairs = [(str(left), "__missing__" if right is None else str(right)) for left, right in pairs if left is not None]
    labels = sorted({value for pair in pairs for value in pair})
    scores: list[float] = []
    for label in labels:
        tp = sum(left == label and right == label for left, right in pairs)
        fp = sum(left != label and right == label for left, right in pairs)
        fn = sum(left == label and right != label for left, right in pairs)
        scores.append(0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn))
    return sum(scores) / len(scores) if scores else 0.0

def _validate_annotation(item: object) -> None:
    """gate 計算前に annotation の field 型と値域を一括検証する。

文字列 timer、範囲外 ratio、非整数 level、非シーケンス items が gate を通過しないよう、
型と範囲を計算前に fail-closed で拒否します。
    """
    # timer: 非負の有限数
    for name in ("expected_timer_seconds", "predicted_timer_seconds"):
        value = _field(item, name)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be non-negative finite number, got {value!r}")
    # ratio: 0..1 の有限数
    for name in ("expected_hp_ratio", "predicted_hp_ratio", "expected_xp_ratio", "predicted_xp_ratio"):
        value = _field(item, name)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be float in [0, 1], got {value!r}")
    # level: 1以上の正整数
    for name in ("expected_level", "predicted_level"):
        value = _field(item, name)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be positive integer, got {value!r}")
    # screen_state, choice: 文字列
    for name in ("expected_screen_state", "predicted_screen_state", "expected_choice", "predicted_choice"):
        value = _field(item, name)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{name} must be str or None, got {type(value).__name__}")
    # items: str|None 要素のシーケンス
    for name in ("expected_items", "predicted_items"):
        value = _field(item, name)
        if value is None:
            continue
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{name} must be list or tuple or None")
        for elem in value:
            if elem is not None and not isinstance(elem, str):
                raise ValueError(f"{name} elements must be str or None, got {type(elem).__name__}")
    # latency: 非負の有限数
    latency = _field(item, "latency_ms")
    if latency is not None:
        if isinstance(latency, bool) or not isinstance(latency, (int, float)) or not math.isfinite(latency) or latency < 0:
            raise ValueError(f"latency_ms must be non-negative finite number, got {latency!r}")


def _roi_metrics(records: Sequence[object]) -> tuple[float, float, float]:
    """ROI center error、inside rate、false-positive rate を返す。

正例は予測中心と正解矩形を比較し、負例は予測 ROI の有無から誤検出率を計算します。
    """
    errors: list[float] = []
    inside: list[bool] = []
    false_positives: list[bool] = []
    for item in records:
        expected, predicted = _roi(item, True), _roi(item, False)
        if expected is None:
            false_positives.append(predicted is not None)
            continue
        if predicted is None:
            errors.append(1.0e9)
            inside.append(False)
            continue
        ex = ((expected[0] + expected[2]) / 2, (expected[1] + expected[3]) / 2)
        px = ((predicted[0] + predicted[2]) / 2, (predicted[1] + predicted[3]) / 2)
        errors.append(math.hypot(ex[0] - px[0], ex[1] - px[1]))
        inside.append(expected[0] <= px[0] <= expected[2] and expected[1] <= px[1] <= expected[3])
    center_error = sum(errors) / len(errors) if errors else 1.0e9
    inside_rate = sum(inside) / len(inside) if inside else 0.0
    # 陰性例が存在しない場合は 1.0 にして gate を fail-closed にする。
    # 陰性例がないと false-positive 抑制を検証できていないため。
    false_positive_rate = sum(false_positives) / len(false_positives) if false_positives else 1.0
    return center_error, inside_rate, false_positive_rate

def validate(
    annotations: Sequence[object], config: CalibrationConfig | None = None, *,
    split: str | None = None, formal_mode: bool = False,
    formal_dataset_eligible: bool = False,
) -> ValidationReport:
    """model_validation 注釈から固定済みの11 gate を計算する。

入力を identity で整列して集計順も固定し、同じ入力と config から同じ report hash を返します。
    """
    if not annotations:
        raise ValueError("validate requires at least one annotation")
    reject_formal_if_synthetic(
        formal_mode=formal_mode, formal_dataset_eligible=formal_dataset_eligible
    )
    cfg = CalibrationConfig() if config is None else config
    if not isinstance(cfg, CalibrationConfig):
        raise TypeError("config must be CalibrationConfig")
    records = list(annotations)
    for item in records:
        _effective_split(item, split, "model_validation")
    records.sort(key=lambda item: str(_field(item, "sample_id", _field(item, "frame_id", ""))))
    for item in records:
        _validate_annotation(item)
    latency_values = [
        1.0e9 if _field(item, "latency_ms") is None else _require_finite(_field(item, "latency_ms"), "latency_ms")
        for item in records
    ]
    if any(value < 0 for value in latency_values):
        raise ValueError("latency_ms must be non-negative")
    center_error, inside_rate, false_positive_rate = _roi_metrics(records)
    gates = (
        GateResult.floor_gate("timer_exact", value=_exact_rate(records, "expected_timer_seconds", "predicted_timer_seconds"), floor=cfg.timer_exact_floor),
        GateResult.floor_gate("level", value=_exact_rate(records, "expected_level", "predicted_level"), floor=cfg.level_floor),
        GateResult.ceiling_gate("hp_mae", value=_mae(records, "expected_hp_ratio", "predicted_hp_ratio"), ceiling=cfg.hp_mae_ceiling),
        GateResult.ceiling_gate("xp_mae", value=_mae(records, "expected_xp_ratio", "predicted_xp_ratio"), ceiling=cfg.xp_mae_ceiling),
        GateResult.floor_gate("item", value=_item_rate(records), floor=cfg.item_floor),
        GateResult.floor_gate("choice", value=_exact_rate(records, "expected_choice", "predicted_choice"), floor=cfg.choice_floor),
        GateResult.floor_gate("screen_state_f1", value=_macro_f1(records), floor=cfg.screen_state_f1_floor),
        GateResult.ceiling_gate("latency", value=max(latency_values), ceiling=cfg.latency_ceiling_ms),
        GateResult.ceiling_gate("roi_center_error", value=center_error, ceiling=cfg.roi_center_error_ceiling),
        GateResult.floor_gate("roi_inside_rate", value=inside_rate, floor=cfg.roi_inside_rate_floor),
        GateResult.ceiling_gate("roi_false_positive", value=false_positive_rate, ceiling=cfg.roi_false_positive_ceiling),
    )
    identity = {
        "schema_version": REPORT_SCHEMA_VERSION, "sample_count": len(records),
        "config": cfg.to_wire(), "gates": [gate.to_wire() for gate in gates],
    }
    return ValidationReport(REPORT_SCHEMA_VERSION, len(records), cfg, gates, canonical_hash(identity))

def _fit_from_wire(wire: object) -> FitResult:
    """strict wire から FitResult を復元して内部 hash を検証する。

未知 field、型違い、成果物 identity の不一致を package 改ざんとして拒否します。
    """
    expected = {"schema_version", "annotation_count", "annotation_canonical_hash", "roi_anchors", "fit_canonical_hash"}
    if not isinstance(wire, dict) or set(wire) != expected or wire["schema_version"] != FIT_SCHEMA_VERSION:
        raise ValueError("fit result fields do not match schema")
    anchors = tuple((item[0], tuple(_require_finite(v, "ROI coordinate") for v in item[1])) for item in wire["roi_anchors"])
    identity = {key: wire[key] for key in expected - {"fit_canonical_hash"}}
    if canonical_hash(identity) != wire["fit_canonical_hash"]:
        raise PackageHashMismatch("fit result hash mismatch")
    return FitResult(wire["schema_version"], wire["annotation_count"], wire["annotation_canonical_hash"], anchors, wire["fit_canonical_hash"])

def _report_from_wire(wire: object) -> ValidationReport:
    """strict wire から ValidationReport を復元して内部 hash を検証する。

保存された config と gate をそのまま canonical 化し、閾値や合否の差し替えを検出します。
    """
    expected = {"schema_version", "sample_count", "config", "gates", "report_canonical_hash"}
    if not isinstance(wire, dict) or set(wire) != expected or wire["schema_version"] != REPORT_SCHEMA_VERSION:
        raise ValueError("validation report fields do not match schema")
    config = CalibrationConfig(**wire["config"])
    gates = tuple(GateResult(**item) for item in wire["gates"])
    if len(gates) != 11 or len({gate.name for gate in gates}) != 11:
        raise ValueError("validation report must contain 11 unique gates")
    identity = {key: wire[key] for key in expected - {"report_canonical_hash"}}
    if canonical_hash(identity) != wire["report_canonical_hash"]:
        raise PackageHashMismatch("validation report hash mismatch")
    return ValidationReport(wire["schema_version"], wire["sample_count"], config, gates, wire["report_canonical_hash"])

def _package_identity(
    fit_result: FitResult, report: ValidationReport, *, development_only: bool,
    formal_parser_eligible: bool,
) -> dict[str, object]:
    """自己参照する package_hash を除いた package identity を返す。

用途属性と内部成果物を1つの hash 範囲へ束縛し、metadata だけの差し替えも検出します。
    """
    return {
        "schema_version": PACKAGE_SCHEMA_VERSION,
        "development_only": development_only,
        "formal_parser_eligible": formal_parser_eligible,
        "fit_result": fit_result.to_wire(),
        "validation_report": report.to_wire(),
    }

class PackageWriter:
    """HUD parser package を canonical JSON で発行する writer。

04-04では development package だけを atomic write し、formal publisher は未実装のまま閉じます。
    """

    @classmethod
    def publish_development(
        cls, package_path: str | os.PathLike[str], fit_result: FitResult,
        validation_report: ValidationReport,
    ) -> ParserPackageMeta:
        """development 専用 package を同一 directory 内で atomic publish する。

内容 hash を計算して metadata へ封入し、fsync 後の os.replace で完成ファイルだけを公開します。
        """
        if not isinstance(fit_result, FitResult) or not isinstance(validation_report, ValidationReport):
            raise TypeError("fit_result and validation_report have invalid types")
        path = Path(package_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        identity = _package_identity(
            fit_result, validation_report, development_only=True,
            formal_parser_eligible=False,
        )
        package_hash = canonical_hash(identity)
        meta = ParserPackageMeta(
            PACKAGE_SCHEMA_VERSION, True, False, fit_result.fit_canonical_hash,
            validation_report.report_canonical_hash, package_hash,
        )
        wire = {
            "metadata": {
                "schema_version": meta.schema_version,
                "development_only": meta.development_only,
                "formal_parser_eligible": meta.formal_parser_eligible,
                "fit_canonical_hash": meta.fit_canonical_hash,
                "report_canonical_hash": meta.report_canonical_hash,
                "package_hash": meta.package_hash,
            },
            "fit_result": fit_result.to_wire(),
            "validation_report": validation_report.to_wire(),
        }
        temp_path = path.with_name(f".{path.name}.tmp")
        try:
            with temp_path.open("wb") as stream:
                stream.write(canonical_json_bytes(wire))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_path, path)
        finally:
            temp_path.unlink(missing_ok=True)
        return meta

    @classmethod
    def publish_formal(cls, *_args: object, **_kwargs: object) -> ParserPackageMeta:
        """formal parser package の発行を明示的に拒否する。

正式 dataset と release gate を扱う04-05まで実装範囲を広げず、必ず停止します。
        """
        raise NotImplementedError("formal HUD parser publishing is deferred to 04-05")

class PackageLoader:
    """HUD parser package を strict schema と内容 hash で復元する loader。

development と formal の入口を分け、用途属性が一致しない package を fail-closed で拒否します。
    """

    @classmethod
    def load_development(cls, package_path: str | os.PathLike[str]) -> ParserPackage:
        """検証済み development package を復元する。

development_only=true かつ formal_parser_eligible=false の組だけを受け入れます。
        """
        package = cls._load(package_path)
        if not package.meta.development_only or package.meta.formal_parser_eligible:
            raise FormalPackageMixingError("package is not development-only")
        return package

    @classmethod
    def load_formal(cls, package_path: str | os.PathLike[str]) -> ParserPackage:
        """用途属性を検査して formal package だけを復元する。

04-04 development package は必ず FormalPackageMixingError となり、正式経路へ混入しません。
        """
        package = cls._load(package_path)
        if package.meta.development_only or not package.meta.formal_parser_eligible:
            raise FormalPackageMixingError("development package cannot be loaded as formal")
        return package

    @classmethod
    def _load(cls, package_path: str | os.PathLike[str]) -> ParserPackage:
        """package wire、内部 hash、外側 package hash を一括検証する。

両 public loader が同じ integrity 検査を通るため、用途別経路の検証漏れを防ぎます。
        """
        path = Path(package_path)
        try:
            wire = json.loads(path.read_bytes())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid parser package: {path}") from exc
        if not isinstance(wire, dict) or set(wire) != {"metadata", "fit_result", "validation_report"}:
            raise ValueError("parser package fields do not match schema")
        meta_wire = wire["metadata"]
        meta_fields = {
            "schema_version", "development_only", "formal_parser_eligible",
            "fit_canonical_hash", "report_canonical_hash", "package_hash",
        }
        if not isinstance(meta_wire, dict) or set(meta_wire) != meta_fields:
            raise ValueError("parser package metadata fields do not match schema")
        fit_result = _fit_from_wire(wire["fit_result"])
        report = _report_from_wire(wire["validation_report"])
        meta = ParserPackageMeta(**meta_wire)
        if meta.fit_canonical_hash != fit_result.fit_canonical_hash:
            raise PackageHashMismatch("metadata fit hash mismatch")
        if meta.report_canonical_hash != report.report_canonical_hash:
            raise PackageHashMismatch("metadata report hash mismatch")
        identity = _package_identity(
            fit_result, report, development_only=meta.development_only,
            formal_parser_eligible=meta.formal_parser_eligible,
        )
        if canonical_hash(identity) != meta.package_hash:
            raise PackageHashMismatch("package hash mismatch")
        return ParserPackage(meta, fit_result, report)
