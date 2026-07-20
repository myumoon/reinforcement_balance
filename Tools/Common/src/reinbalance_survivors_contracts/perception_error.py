"""perception 誤差プロファイルの契約（scaffold, v1）。

``PerceptionErrorProfile`` は、error wrapper（プラン 03-02）が実 perception ノイズを
模擬するために simulator の観測へ注入する perception corruption をパラメータ化する。
本モジュールは versioned な wire 型とその canonical hash を所有する。runtime での適用は
wrapper と calibration（03-02 / 04-05）が所有する。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from .canonical_json import canonical_hash

__all__ = [
    "PERCEPTION_ERROR_SCHEMA_VERSION",
    "PerceptionErrorProfile",
]

PERCEPTION_ERROR_SCHEMA_VERSION = "perception_error.v1"


@dataclass(frozen=True)
class PerceptionErrorProfile:
    """perception 誤差注入のパラメータ（位置ノイズ・欠落・誤検出・遅延）。versioned。"""

    position_noise_std: float = 0.0
    dropout_prob: float = 0.0
    false_positive_rate: float = 0.0
    latency_frames: int = 0
    schema_version: str = PERCEPTION_ERROR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("position_noise_std", "dropout_prob", "false_positive_rate"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a real number")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.position_noise_std < 0.0:
            raise ValueError("position_noise_std must be >= 0")
        for name in ("dropout_prob", "false_positive_rate"):
            value = float(getattr(self, name))
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be in [0, 1]")
        if isinstance(self.latency_frames, bool) or not isinstance(
            self.latency_frames, int
        ):
            raise ValueError("latency_frames must be an int")
        if self.latency_frames < 0:
            raise ValueError("latency_frames must be >= 0")

    def to_wire(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "position_noise_std": float(self.position_noise_std),
            "dropout_prob": float(self.dropout_prob),
            "false_positive_rate": float(self.false_positive_rate),
            "latency_frames": int(self.latency_frames),
        }

    @property
    def profile_hash(self) -> str:
        return canonical_hash(self.to_wire())

    @classmethod
    def from_wire(cls, data: Mapping[str, Any]) -> "PerceptionErrorProfile":
        if data.get("schema_version") != PERCEPTION_ERROR_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported PerceptionErrorProfile schema_version "
                f"{data.get('schema_version')!r}"
            )
        # 任意の数値パラメータは規定のデフォルトを保つ。__post_init__ が、誤った型を
        # silent に coercion せずに型／範囲を検証する。
        return cls(
            position_noise_std=data.get("position_noise_std", 0.0),
            dropout_prob=data.get("dropout_prob", 0.0),
            false_positive_rate=data.get("false_positive_rate", 0.0),
            latency_frames=data.get("latency_frames", 0),
            schema_version=data["schema_version"],
        )
