"""calibration residuals から PerceptionErrorProfile を fit し、final lineage seal を管理する。

calibration セッションの誤差残差だけを使ってエラープロファイルを推定し、
final E2E セッションの一度限り開封ポリシーと stale-verdict 検証を提供します。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np

from reinbalance_survivors_contracts.canonical_json import canonical_hash
from reinbalance_survivors_contracts.perception_error import (
    ITEM_CATEGORY_SIZE,
    PerceptionErrorProfile,
)

LINEAGE_SEAL_SCHEMA_VERSION: Final[str] = "perception_lineage_seal.v1"
CALIBRATION_VERDICT_SCHEMA_VERSION: Final[str] = "perception_calibration_verdict.v1"
FINAL_VERDICT_SCHEMA_VERSION: Final[str] = "perception_final_verdict.v1"


class FinalSessionAlreadyOpenedError(ValueError):
    """final E2E セッションが既に開封済み（create-once 違反）。

    同じ seal で 2 回目以降の final session 開封を拒否します。
    """


class FinalFitMixingError(ValueError):
    """final E2E セッションの residual を calibration fit に混入しようとした。

    final データを fit に使うと未使用 E2E test の独立性が失われます。
    """


class StaleVerdictError(ValueError):
    """producer hash が verdict 発行時から変化し、verdict が陳腐化した。

    parser/detector/config のいずれかが変わったとき旧 verdict は無効です。
    """


class HashMismatchError(ValueError):
    """ロードした artifact の hash が保存済み hash と一致しない。"""


class SessionOverlapError(ValueError):
    """calibration と final E2E の session_id が重複している。

    PerceptionErrorProfile のコンストラクタとは独立した事前ガードです。
    """


@dataclass(frozen=True, slots=True)
class CalibrationResidual:
    """calibration セッション 1 フレーム分の予測残差。

    field ごとに predicted - ground_truth を保存し、fit の入力とします。
    """

    session_id: str
    frame_id: str
    field: str
    residual: float
    confidence: float
    age_frames: int


@dataclass
class FinalLineageSeal:
    """final E2E test セッションの create-once 開封シール。

    seal_id は producer hashes の canonical hash で、開封後はハッシュ変更を検出します。
    """

    seal_id: str
    parser_artifact_hash: str
    detector_artifact_hash: str
    assembler_schema_hash: str
    config_hash: str
    opened_session_ids: list[str] = field(default_factory=list)
    schema_version: str = LINEAGE_SEAL_SCHEMA_VERSION
    development_only: bool = True

    def verify_hashes(
        self,
        *,
        parser: str,
        detector: str,
        assembler: str,
        config: str,
    ) -> None:
        """現在の producer hash がシール時と一致することを確認する。

        いずれか 1 件でも変化していれば StaleVerdictError を送出します。
        """
        changed = []
        if self.parser_artifact_hash != parser:
            changed.append("parser_artifact_hash")
        if self.detector_artifact_hash != detector:
            changed.append("detector_artifact_hash")
        if self.assembler_schema_hash != assembler:
            changed.append("assembler_schema_hash")
        if self.config_hash != config:
            changed.append("config_hash")
        if changed:
            raise StaleVerdictError(
                f"Seal {self.seal_id!r} is stale; changed fields: {changed}"
            )

    def open_session(self, session_id: str) -> None:
        """final セッションを開封済みとして登録する（1 回限り）。

        既に 1 件以上開封済みのとき FinalSessionAlreadyOpenedError を送出します。
        """
        if self.opened_session_ids:
            raise FinalSessionAlreadyOpenedError(
                f"Final session already opened: {self.opened_session_ids}. "
                "Cannot open another final session with this seal."
            )
        self.opened_session_ids.append(session_id)

    def to_wire(self) -> dict[str, Any]:
        """seal を JSON 保存用の dict として返す。"""
        return {
            "schema_version": self.schema_version,
            "seal_id": self.seal_id,
            "parser_artifact_hash": self.parser_artifact_hash,
            "detector_artifact_hash": self.detector_artifact_hash,
            "assembler_schema_hash": self.assembler_schema_hash,
            "config_hash": self.config_hash,
            "opened_session_ids": list(self.opened_session_ids),
            "development_only": self.development_only,
        }


def create_lineage_seal(
    parser_artifact_hash: str,
    detector_artifact_hash: str,
    assembler_schema_hash: str,
    config_hash: str,
    *,
    development_only: bool = True,
) -> FinalLineageSeal:
    """producer hashes から final lineage seal を作成する。

    seal_id は 4 hashes の canonical hash で決定論的に生成されます。
    """
    seal_id = canonical_hash(
        {
            "parser": parser_artifact_hash,
            "detector": detector_artifact_hash,
            "assembler": assembler_schema_hash,
            "config": config_hash,
        }
    )
    return FinalLineageSeal(
        seal_id=seal_id,
        parser_artifact_hash=parser_artifact_hash,
        detector_artifact_hash=detector_artifact_hash,
        assembler_schema_hash=assembler_schema_hash,
        config_hash=config_hash,
        development_only=development_only,
    )


def fit_error_profile(
    residuals: list[CalibrationResidual],
    calibration_session_ids: list[str],
    final_e2e_session_ids: list[str],
) -> PerceptionErrorProfile:
    """calibration residuals から PerceptionErrorProfile を fit する。

    final E2E の session_id が calibration と重複するとき、
    または final の residual が混入しているとき拒否します。
    """
    cal_ids = set(calibration_session_ids)
    final_ids = set(final_e2e_session_ids)

    overlap = cal_ids & final_ids
    if overlap:
        raise SessionOverlapError(
            f"calibration/final session overlap: {sorted(overlap)}"
        )

    if any(r.session_id in final_ids for r in residuals):
        raise FinalFitMixingError(
            "Final E2E session residuals cannot be used in calibration fit."
        )

    cal_residuals = [r for r in residuals if r.session_id in cal_ids]

    by_field: dict[str, list[float]] = {}
    for r in cal_residuals:
        by_field.setdefault(r.field, []).append(r.residual)

    def _std(vals: list[float]) -> float:
        return float(np.std(vals)) if vals else 0.0

    def _prob(vals: list[float], threshold: float) -> float:
        return float(np.mean([abs(v) > threshold for v in vals])) if vals else 0.0

    hp_misread_std = _std(by_field.get("hp_ratio", []))
    xp_stale_prob = _prob(by_field.get("xp_ratio", []), 0.1)
    timer_stale_prob = _prob(by_field.get("timer_seconds", []), 1.0)
    coord_noise_std = _std(by_field.get("coord_noise", []))
    lat_vals = by_field.get("latency_frames", [])
    lat_mean = max(0.0, float(np.mean(lat_vals)) if lat_vals else 0.0)
    lat_std = max(0.0, _std(lat_vals))

    n = ITEM_CATEGORY_SIZE
    identity_mat = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    return PerceptionErrorProfile(
        latency_mean_frames=lat_mean,
        latency_std_frames=lat_std,
        coord_noise_std=max(0.0, coord_noise_std),
        hud_hp_misread_std=max(0.0, hp_misread_std),
        hud_xp_stale_prob=min(1.0, max(0.0, xp_stale_prob)),
        hud_timer_stale_prob=min(1.0, max(0.0, timer_stale_prob)),
        item_confusion_matrix=identity_mat,
        enemy_confusion_matrix=identity_mat,
        calibration_session_ids=list(calibration_session_ids),
        final_e2e_session_ids=list(final_e2e_session_ids),
    )


def simulator_distance_report(
    calibrated: PerceptionErrorProfile,
    simulator: PerceptionErrorProfile,
) -> dict[str, float]:
    """calibrated profile と simulator profile のスカラー距離を返す。

    03-05 wrapper の compatibility 確認と residual 分布距離レポートに使います。
    """
    return {
        "latency_mean_diff": abs(
            calibrated.latency_mean_frames - simulator.latency_mean_frames
        ),
        "coord_noise_std_diff": abs(
            calibrated.coord_noise_std - simulator.coord_noise_std
        ),
        "hp_misread_std_diff": abs(
            calibrated.hud_hp_misread_std - simulator.hud_hp_misread_std
        ),
        "xp_stale_prob_diff": abs(
            calibrated.hud_xp_stale_prob - simulator.hud_xp_stale_prob
        ),
        "timer_stale_prob_diff": abs(
            calibrated.hud_timer_stale_prob - simulator.hud_timer_stale_prob
        ),
    }


@dataclass
class PerceptionCalibrationVerdict:
    """calibration profile の development-only verdict。

    formal perception verdict には昇格できません。
    """

    verdict_id: str
    calibration_session_ids: list[str]
    calibration_session_hashes: dict[str, str]
    profile_hash: str
    fit_code_hash: str
    sample_counts: dict[str, int]
    development_only: bool = True
    formal_perception_verdict_eligible: bool = False
    schema_version: str = CALIBRATION_VERDICT_SCHEMA_VERSION


@dataclass
class PerceptionFinalVerdict:
    """final E2E perception verdict（synthetic 時は development-only）。

    formal 入力が揃ったときだけ development_only=False にできます。
    """

    verdict_id: str
    seal_id: str
    final_session_ids: list[str]
    parser_artifact_hash: str
    detector_artifact_hash: str
    assembler_schema_hash: str
    ui_presentation_schema_hash: str
    config_hash: str
    metrics: dict[str, Any]
    passed: bool
    blocking_reasons: list[str]
    development_only: bool = True
    formal_perception_verdict_eligible: bool = False
    schema_version: str = FINAL_VERDICT_SCHEMA_VERSION


def load_final_verdict(
    data: dict[str, Any],
    *,
    current_parser_hash: str,
    current_detector_hash: str,
    current_assembler_hash: str,
    current_config_hash: str,
    current_ui_schema_hash: str,
) -> PerceptionFinalVerdict:
    """PerceptionFinalVerdict をロードし、producer hashes が変化していれば拒否する。

    parser/detector/assembler/config/UI schema のいずれかが変わると
    StaleVerdictError を送出します。旧 final sessions を development へ降格し、
    新規 untouched sessions で再発行してください。
    """
    if data.get("schema_version") != FINAL_VERDICT_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported verdict schema_version {data.get('schema_version')!r}"
        )

    stale: list[str] = []
    if data["parser_artifact_hash"] != current_parser_hash:
        stale.append("parser_artifact_hash")
    if data["detector_artifact_hash"] != current_detector_hash:
        stale.append("detector_artifact_hash")
    if data["assembler_schema_hash"] != current_assembler_hash:
        stale.append("assembler_schema_hash")
    if data["config_hash"] != current_config_hash:
        stale.append("config_hash")
    if data["ui_presentation_schema_hash"] != current_ui_schema_hash:
        stale.append("ui_presentation_schema_hash")

    if stale:
        raise StaleVerdictError(
            f"Final perception verdict is stale; changed producer fields: {stale}. "
            "Demote old final sessions to development and re-issue with new untouched sessions."
        )

    return PerceptionFinalVerdict(
        verdict_id=data["verdict_id"],
        seal_id=data["seal_id"],
        final_session_ids=list(data["final_session_ids"]),
        parser_artifact_hash=data["parser_artifact_hash"],
        detector_artifact_hash=data["detector_artifact_hash"],
        assembler_schema_hash=data["assembler_schema_hash"],
        ui_presentation_schema_hash=data["ui_presentation_schema_hash"],
        config_hash=data["config_hash"],
        metrics=dict(data["metrics"]),
        passed=bool(data["passed"]),
        blocking_reasons=list(data["blocking_reasons"]),
        development_only=bool(data.get("development_only", True)),
        formal_perception_verdict_eligible=bool(
            data.get("formal_perception_verdict_eligible", False)
        ),
        schema_version=data["schema_version"],
    )
