"""CalibrationResidual と FittedPerceptionErrorProfile の共有契約。

Training と Deployment が共有する calibration 残差型と fitted profile 型を定義する。
formal token はモジュール内 sentinel で管理し、wire ローダーは development_only=False を
fail-closed にする（formal profile はストア検証経路のみで取得可能）。
"""

from __future__ import annotations

import json as _json
import math
from dataclasses import InitVar, dataclass, field
from types import MappingProxyType
from typing import Any, Final, Mapping

from .canonical_json import canonical_hash, sha256_hex
from .perception_error import ITEM_CATEGORY_SIZE, PerceptionErrorProfile

CALIBRATION_ARTIFACT_SCHEMA_VERSION: Final[str] = "perception_calibration_profile.v1"

# formal profile 生成を wire ローダーから分離するための sentinel。
# 公開 API に含めない。fit runner と store 検証経路のみが参照する。
_FORMAL_FACTORY_TOKEN = object()

_RESIDUAL_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "hp_ratio", "xp_ratio", "timer_seconds", "inventory_hash", "coord_noise",
        "coord_quantization_px", "burst_enter", "burst_exit", "burst_dropout",
        "unknown_screen_collapse", "unknown_screen_collapse_duration",
        "item_category", "enemy_category",
    }
)


class HashMismatchError(ValueError):
    """artifact content hash が seal/verdict の exact hash と一致しない。"""


class FormalVerdictPromotionError(ValueError):
    """synthetic 公開コンストラクタまたは wire ローダーから formal flag を構築しようとした。"""


class InvalidResidualError(ValueError):
    """residual field/type/range が fit 契約外である。"""


def _require_sha256(value: object, label: str) -> str:
    if not (isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)):
        raise ValueError(f"{label} must be a 64-character lowercase SHA-256")
    return value  # type: ignore[return-value]


def _strict_number(value: object, label: str) -> float:
    import numpy as np
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)) or not math.isfinite(float(value)):
        raise InvalidResidualError(f"{label} must be a finite number (bool is forbidden)")
    return float(value)


@dataclass(frozen=True, slots=True)
class CalibrationResidual:
    """キャリブレーション残差の1標本。confidence と age_frames で重み付けする。"""

    session_id: str
    frame_id: str
    field: str
    residual: float
    confidence: float
    age_frames: int
    latency_frames: float = 0.0
    ground_truth_category: int | None = None
    predicted_category: int | None = None

    def __post_init__(self) -> None:
        if type(self.session_id) is not str or not self.session_id:
            raise InvalidResidualError("session_id must be a non-empty string")
        if type(self.frame_id) is not str or not self.frame_id:
            raise InvalidResidualError("frame_id must be a non-empty string")
        if self.field not in _RESIDUAL_FIELDS:
            raise InvalidResidualError(f"unsupported residual field {self.field!r}")
        object.__setattr__(self, "residual", _strict_number(self.residual, "residual"))
        if self.field in {
            "burst_enter", "burst_exit", "burst_dropout",
            "unknown_screen_collapse",
        } and not 0.0 <= self.residual <= 1.0:
            raise InvalidResidualError(f"{self.field} residual must be in [0, 1]")
        if self.field in {
            "coord_quantization_px", "unknown_screen_collapse_duration",
        } and self.residual < 0.0:
            raise InvalidResidualError(f"{self.field} residual must be non-negative")
        confidence = _strict_number(self.confidence, "confidence")
        if not 0.0 <= confidence <= 1.0:
            raise InvalidResidualError("confidence must be in [0, 1]")
        object.__setattr__(self, "confidence", confidence)
        if type(self.age_frames) is not int or self.age_frames < 0:
            raise InvalidResidualError("age_frames must be a non-negative integer")
        latency = _strict_number(self.latency_frames, "latency_frames")
        if latency < 0.0:
            raise InvalidResidualError("latency_frames must be non-negative")
        object.__setattr__(self, "latency_frames", latency)
        category_pair = (self.ground_truth_category, self.predicted_category)
        if (category_pair[0] is None) != (category_pair[1] is None):
            raise InvalidResidualError("category ground truth/prediction must be provided together")
        if category_pair[0] is not None:
            if self.field not in {"item_category", "enemy_category"}:
                raise InvalidResidualError("category labels require a category residual field")
            if any(type(v) is not int or not 0 <= v < ITEM_CATEGORY_SIZE for v in category_pair):
                raise InvalidResidualError("category labels are outside the fixed vocabulary")
        elif self.field in {"item_category", "enemy_category"}:
            raise InvalidResidualError("category residual fields require category labels")


@dataclass(frozen=True)
class FittedPerceptionErrorProfile(PerceptionErrorProfile):
    """calibration artifact メタデータを保持する PerceptionErrorProfile のサブタイプ。

    formal profile は wire ローダーでは取得できない（development_only=False の wire は拒否）。
    fit runner（Deployment）またはストア検証経路のみが _FORMAL_FACTORY_TOKEN を渡せる。
    """

    calibration_session_hashes: Mapping[str, str] = field(default_factory=dict)
    field_sample_counts: Mapping[str, int] = field(default_factory=dict)
    fit_code_hash: str = ""
    development_only: bool = True
    _factory_token: InitVar[object | None] = None

    def __post_init__(self, _factory_token: object | None) -> None:
        super().__post_init__()
        hashes = dict(self.calibration_session_hashes)
        if set(hashes) != set(self.calibration_session_ids):
            raise ValueError("calibration_session_hashes must exactly match calibration ids")
        for session_id, content_hash in hashes.items():
            if type(session_id) is not str or not session_id:
                raise ValueError("calibration session hash key must be non-empty")
            _require_sha256(content_hash, f"calibration_session_hashes[{session_id!r}]")
        counts = dict(self.field_sample_counts)
        if not counts or not all(type(n) is str and type(c) is int and c > 0 for n, c in counts.items()):
            raise ValueError("field_sample_counts must contain positive integer counts")
        _require_sha256(self.fit_code_hash, "fit_code_hash")
        if type(self.development_only) is not bool:
            raise ValueError("development_only must be bool")
        if not self.development_only and _factory_token is not _FORMAL_FACTORY_TOKEN:
            raise FormalVerdictPromotionError(
                "formal fitted profile requires the verified formal factory"
            )
        object.__setattr__(self, "calibration_session_hashes", MappingProxyType(hashes))
        object.__setattr__(self, "field_sample_counts", MappingProxyType(counts))

    def to_artifact_wire(self) -> dict[str, Any]:
        return {
            "schema_version": CALIBRATION_ARTIFACT_SCHEMA_VERSION,
            "profile": self.to_wire(),
            "profile_hash": self.profile_hash,
            "calibration_session_hashes": dict(self.calibration_session_hashes),
            "field_sample_counts": dict(self.field_sample_counts),
            "fit_code_hash": self.fit_code_hash,
            "development_only": self.development_only,
        }

    @classmethod
    def from_artifact_wire(cls, data: Mapping[str, Any]) -> "FittedPerceptionErrorProfile":
        """wire から development_only=True の profile のみをロードする。

        development_only=False の wire は fail-closed として拒否する。
        formal profile はストア検証経路（from_store_artifact）のみで取得できる。
        """
        expected = {
            "schema_version", "profile", "profile_hash",
            "calibration_session_hashes", "field_sample_counts", "fit_code_hash",
            "development_only",
        }
        if not isinstance(data, Mapping) or set(data) != expected:
            raise ValueError("calibration artifact fields do not match schema")
        if data["schema_version"] != CALIBRATION_ARTIFACT_SCHEMA_VERSION:
            raise ValueError("unsupported calibration artifact schema")
        if type(data["development_only"]) is not bool:
            raise ValueError("calibration artifact development_only must be bool")
        # wire ローダーは formal token を付与しない。development_only=False の wire は拒否する。
        if data["development_only"] is False:
            raise FormalVerdictPromotionError(
                "formal calibration profile cannot be loaded from raw wire; "
                "use the artifact store verification path"
            )
        profile = PerceptionErrorProfile.from_wire(data["profile"])
        if profile.profile_hash != data["profile_hash"]:
            raise HashMismatchError("calibration artifact profile hash mismatch")
        fitted = cls(
            **profile.to_wire(),
            calibration_session_hashes=data["calibration_session_hashes"],
            field_sample_counts=data["field_sample_counts"],
            fit_code_hash=data["fit_code_hash"],
            development_only=True,
            _factory_token=None,
        )
        # data["development_only"] は True 確認済み（False は上で raise 済み）。
        if fitted.to_artifact_wire() != dict(data):
            raise HashMismatchError("calibration artifact failed canonical reconstruction")
        return fitted

    @classmethod
    def _from_verified_bytes(
        cls, data_bytes: bytes, expected_sha256: str
    ) -> "FittedPerceptionErrorProfile":
        """ArtifactStore の content-hash 検証後に限り formal profile をロードする。

        呼び出し元は store.verify(ref) が True であることを確認してから渡すこと。
        expected_sha256 との不一致は HashMismatchError を送出し fail-closed にする。
        formal profile（development_only=False）には _FORMAL_FACTORY_TOKEN を付与する。
        """
        actual = sha256_hex(data_bytes)
        if actual != expected_sha256:
            raise HashMismatchError(
                f"calibration artifact content hash mismatch: expected {expected_sha256!r}, got {actual!r}"
            )
        try:
            data: Any = _json.loads(data_bytes.decode("utf-8"))
        except (UnicodeDecodeError, _json.JSONDecodeError) as exc:
            raise ValueError(f"calibration artifact is not valid JSON: {exc}") from exc
        expected_keys = {
            "schema_version", "profile", "profile_hash",
            "calibration_session_hashes", "field_sample_counts", "fit_code_hash",
            "development_only",
        }
        if not isinstance(data, Mapping) or set(data) != expected_keys:
            raise ValueError("calibration artifact fields do not match schema")
        if data["schema_version"] != CALIBRATION_ARTIFACT_SCHEMA_VERSION:
            raise ValueError("unsupported calibration artifact schema")
        if type(data["development_only"]) is not bool:
            raise ValueError("calibration artifact development_only must be bool")
        profile = PerceptionErrorProfile.from_wire(data["profile"])
        if profile.profile_hash != data["profile_hash"]:
            raise HashMismatchError("calibration artifact profile hash mismatch")
        factory_token = _FORMAL_FACTORY_TOKEN if data["development_only"] is False else None
        fitted = cls(
            **profile.to_wire(),
            calibration_session_hashes=data["calibration_session_hashes"],
            field_sample_counts=data["field_sample_counts"],
            fit_code_hash=data["fit_code_hash"],
            development_only=data["development_only"],
            _factory_token=factory_token,
        )
        if fitted.to_artifact_wire() != dict(data):
            raise HashMismatchError("calibration artifact failed canonical reconstruction")
        return fitted

    @property
    def artifact_hash(self) -> str:
        return canonical_hash(self.to_artifact_wire())
