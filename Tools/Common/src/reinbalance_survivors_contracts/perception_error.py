"""Survivors perception corruption の versioned profile 契約。

画面認識で起きる遅延・欠落・座標ずれ・カテゴリ誤認・HUD 誤読を一つの
厳格な設定へまとめ、Training と検証ツールが同じ意味と hash を共有します。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping

from .canonical_json import canonical_hash
from .ui_intent import ensure, is_strict_number

__all__ = [
    "ITEM_CATEGORY_SIZE",
    "PERCEPTION_ERROR_SCHEMA_VERSION",
    "PerceptionErrorProfile",
]

PERCEPTION_ERROR_SCHEMA_VERSION = "perception_error.v1"
ITEM_CATEGORY_SIZE = 4

_WIRE_KEYS = frozenset(
    {
        "latency_mean_frames",
        "latency_std_frames",
        "burst_enter_prob",
        "burst_exit_prob",
        "burst_dropout_prob",
        "coord_noise_std",
        "coord_quantization_px",
        "count_clip_max",
        "item_confusion_matrix",
        "enemy_confusion_matrix",
        "hud_timer_stale_prob",
        "hud_hp_misread_std",
        "hud_xp_stale_prob",
        "hud_inventory_stale_prob",
        "unknown_screen_collapse_prob",
        "unknown_screen_collapse_duration_frames",
        "calibration_session_ids",
        "final_e2e_session_ids",
        "schema_version",
    }
)
_NONNEGATIVE_FIELDS = (
    "latency_mean_frames",
    "latency_std_frames",
    "coord_noise_std",
    "coord_quantization_px",
    "hud_hp_misread_std",
    "unknown_screen_collapse_duration_frames",
)
_PROBABILITY_FIELDS = (
    "burst_enter_prob",
    "burst_exit_prob",
    "burst_dropout_prob",
    "hud_timer_stale_prob",
    "hud_xp_stale_prob",
    "hud_inventory_stale_prob",
    "unknown_screen_collapse_prob",
)


def _validated_number(value: Any, name: str) -> float:
    """有限な実数を暗黙の文字列変換なしで検証する。

    bool は int の派生型ですが確率や標準偏差としては受理せず、
    NaN/Inf も canonical JSON や学習 tensor へ到達する前に拒否します。
    """
    ensure(
        is_strict_number(value) and math.isfinite(float(value)),
        f"{name} must be a finite real number",
    )
    return float(value)


def _validated_matrix(value: Any, name: str) -> tuple[tuple[float, ...], ...]:
    """categorical confusion matrix を不変な正方確率行列へ変換する。

    各行の合計に満たない残余確率は「元カテゴリを維持」と解釈するため、
    要素を [0,1]、行和を 1 以下へ制限します。
    """
    ensure(isinstance(value, (list, tuple)), f"{name} must be a sequence")
    if not value:
        return ()
    size = len(value)
    rows: list[tuple[float, ...]] = []
    for row in value:
        ensure(
            isinstance(row, (list, tuple)) and len(row) == size,
            f"{name} must be square",
        )
        normalized = tuple(
            _validated_number(item, f"{name} entry") for item in row
        )
        ensure(
            all(0.0 <= item <= 1.0 for item in normalized),
            f"{name} entries must be in [0, 1]",
        )
        ensure(sum(normalized) <= 1.0 + 1e-12, f"{name} row sum must be <= 1")
        rows.append(normalized)
    return tuple(rows)


def _validated_session_ids(value: Any, name: str) -> tuple[str, ...]:
    """session id list を重複のない不変 tuple へ変換する。

    空文字や重複を許すと calibration/evaluation split の照合が曖昧になるため、
    load 境界で一意な非空文字列だけへ限定します。
    """
    ensure(isinstance(value, (list, tuple)), f"{name} must be a sequence")
    ensure(
        all(type(item) is str and bool(item) for item in value),
        f"{name} entries must be non-empty str",
    )
    ensure(len(value) == len(set(value)), f"{name} entries must be unique")
    return tuple(value)


@dataclass(frozen=True)
class PerceptionErrorProfile:
    """画面認識誤差を再現する完全な v1 parameter set。

    生成時に全 field を fail-closed 検証し、可変 list は内部で tuple 化して
    profile hash と worker 間共有状態が後から変わらないようにします。
    """

    latency_mean_frames: float = 0.0
    latency_std_frames: float = 0.0
    burst_enter_prob: float = 0.0
    burst_exit_prob: float = 1.0
    burst_dropout_prob: float = 0.0
    coord_noise_std: float = 0.0
    coord_quantization_px: float = 0.0
    count_clip_max: int = 32
    item_confusion_matrix: list = field(default_factory=list)
    enemy_confusion_matrix: list = field(default_factory=list)
    hud_timer_stale_prob: float = 0.0
    hud_hp_misread_std: float = 0.0
    hud_xp_stale_prob: float = 0.0
    hud_inventory_stale_prob: float = 0.0
    unknown_screen_collapse_prob: float = 0.0
    unknown_screen_collapse_duration_frames: float = 0.0
    calibration_session_ids: list = field(default_factory=list)
    final_e2e_session_ids: list = field(default_factory=list)
    schema_version: str = PERCEPTION_ERROR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """全 scalar・matrix・split field を対称に検証して凍結する。

        `from_wire` 以外の直接構築でも同じ不変条件を通し、不正 profile を
        wrapper へ渡せる別経路を残しません。
        """
        ensure(
            self.schema_version == PERCEPTION_ERROR_SCHEMA_VERSION,
            "unsupported PerceptionErrorProfile schema_version",
        )
        for name in _NONNEGATIVE_FIELDS:
            value = _validated_number(getattr(self, name), name)
            ensure(value >= 0.0, f"{name} must be >= 0")
            object.__setattr__(self, name, value)
        for name in _PROBABILITY_FIELDS:
            value = _validated_number(getattr(self, name), name)
            ensure(0.0 <= value <= 1.0, f"{name} must be in [0, 1]")
            object.__setattr__(self, name, value)
        ensure(
            isinstance(self.count_clip_max, int)
            and not isinstance(self.count_clip_max, bool)
            and self.count_clip_max > 0,
            "count_clip_max must be a positive int",
        )
        item_confusion_matrix = _validated_matrix(
            self.item_confusion_matrix, "item_confusion_matrix"
        )
        ensure(
            not item_confusion_matrix
            or len(item_confusion_matrix) == ITEM_CATEGORY_SIZE,
            f"item_confusion_matrix must be empty or "
            f"{ITEM_CATEGORY_SIZE}x{ITEM_CATEGORY_SIZE}",
        )
        object.__setattr__(
            self, "item_confusion_matrix", item_confusion_matrix
        )
        # DeployObsV1 に enemy category field はまだないため、enemy 行列は
        # production vocabulary へ束縛せず、空または任意サイズの正方行列を保つ。
        object.__setattr__(
            self,
            "enemy_confusion_matrix",
            _validated_matrix(self.enemy_confusion_matrix, "enemy_confusion_matrix"),
        )
        calibration_ids = _validated_session_ids(
            self.calibration_session_ids, "calibration_session_ids"
        )
        final_ids = _validated_session_ids(
            self.final_e2e_session_ids, "final_e2e_session_ids"
        )
        overlap = sorted(set(calibration_ids) & set(final_ids))
        ensure(
            not overlap,
            f"calibration/final-E2E session id overlap: {overlap}",
        )
        object.__setattr__(self, "calibration_session_ids", calibration_ids)
        object.__setattr__(self, "final_e2e_session_ids", final_ids)

    def to_wire(self) -> dict[str, Any]:
        """canonical hash と JSON 保存に使う完全な wire mapping を返す。

        内部の不変 tuple は JSON array と同じ list へ戻し、schema v1 の
        field を省略せず安定した内容として出力します。
        """
        return {
            "latency_mean_frames": self.latency_mean_frames,
            "latency_std_frames": self.latency_std_frames,
            "burst_enter_prob": self.burst_enter_prob,
            "burst_exit_prob": self.burst_exit_prob,
            "burst_dropout_prob": self.burst_dropout_prob,
            "coord_noise_std": self.coord_noise_std,
            "coord_quantization_px": self.coord_quantization_px,
            "count_clip_max": self.count_clip_max,
            "item_confusion_matrix": [
                list(row) for row in self.item_confusion_matrix
            ],
            "enemy_confusion_matrix": [
                list(row) for row in self.enemy_confusion_matrix
            ],
            "hud_timer_stale_prob": self.hud_timer_stale_prob,
            "hud_hp_misread_std": self.hud_hp_misread_std,
            "hud_xp_stale_prob": self.hud_xp_stale_prob,
            "hud_inventory_stale_prob": self.hud_inventory_stale_prob,
            "unknown_screen_collapse_prob": self.unknown_screen_collapse_prob,
            "unknown_screen_collapse_duration_frames": (
                self.unknown_screen_collapse_duration_frames
            ),
            "calibration_session_ids": list(self.calibration_session_ids),
            "final_e2e_session_ids": list(self.final_e2e_session_ids),
            "schema_version": self.schema_version,
        }

    @property
    def profile_hash(self) -> str:
        """共有 canonical JSON 経路で profile の SHA-256 identity を返す。

        Training manifest と calibration artifact が同じ設定を参照したかを、
        dict 順序や platform に依存せず比較できます。
        """
        return canonical_hash(self.to_wire())

    @property
    def is_clean(self) -> bool:
        """全 corruption が既定の無効値なら True を返す。

        clean profile は wrapper の高速 no-op 経路へ入り、入力 tensor の
        bytes を一切変化させないために使います。
        """
        return self.to_wire() == PerceptionErrorProfile().to_wire()

    @classmethod
    def from_wire(cls, data: Mapping[str, Any]) -> "PerceptionErrorProfile":
        """完全一致する v1 wire mapping から profile を構築する。

        schema version を最初に確認し、未知・欠落 field、型違い、非有限値、
        範囲違反、calibration/E2E 重複を一切補完せず拒否します。
        """
        ensure(isinstance(data, Mapping), "profile must be a mapping")
        ensure(
            data.get("schema_version") == PERCEPTION_ERROR_SCHEMA_VERSION,
            f"unsupported PerceptionErrorProfile schema_version "
            f"{data.get('schema_version')!r}",
        )
        ensure(
            all(type(key) is str for key in data),
            "PerceptionErrorProfile field names must be str",
        )
        unknown = set(data.keys()) - _WIRE_KEYS
        ensure(not unknown, f"unknown PerceptionErrorProfile fields: {sorted(unknown)}")
        ensure(set(data.keys()) == _WIRE_KEYS, "PerceptionErrorProfile fields missing")
        ensure(
            all(
                isinstance(data[name], list)
                for name in (
                    "item_confusion_matrix",
                    "enemy_confusion_matrix",
                    "calibration_session_ids",
                    "final_e2e_session_ids",
                )
            ),
            "matrix and session wire fields must be lists",
        )
        return cls(**{name: data[name] for name in _WIRE_KEYS})
