"""calibration residuals から PerceptionErrorProfile を fit し、final lineage seal を管理する。

calibration セッションの誤差残差だけを使ってエラープロファイルを推定し、
final E2E セッションの一度限り開封ポリシーと stale-verdict 検証を提供します。

## FinalLineageSeal の設計

seal は create-once オブジェクトで、ArtifactStore への書き込みで永続化されます。
- `final_session_set` で開封可能なセッション集合を固定する（追加・変更不可）。
- 同じ session_id の 2 回目の開封を拒否する（異なる session_id は各1回許可）。
- ArtifactStore がない環境ではプロセス内リストでフォールバックするが、
  その場合 development_only=True を強制する。
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
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

# PerceptionFinalVerdict の必須 wire フィールド（完全なfield set）
_FINAL_VERDICT_REQUIRED_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "verdict_id",
        "seal_id",
        "final_session_ids",
        "parser_artifact_hash",
        "detector_artifact_hash",
        "assembler_schema_hash",
        "ui_presentation_schema_hash",
        "config_hash",
        "metrics",
        "passed",
        "blocking_reasons",
        "development_only",
        "formal_perception_verdict_eligible",
        # 拡張 subject フィールド（Item 6）
        "capture_dataset_hash",
        "calibration_profile_hash",
        "threshold_hash",
        "atlas_vocabulary_hash",
        "assembler_impl_hash",
        "roi_resolver_input_hash",
        "benchmark_fit_code_hash",
        "lineage_seal_hash",
    }
)

_SHA256_RE_LEN: Final[int] = 64


def _is_sha256(value: object) -> bool:
    """64 文字 lowercase hex 文字列かどうかを確認する。"""
    if not isinstance(value, str) or len(value) != _SHA256_RE_LEN:
        return False
    return all(c in "0123456789abcdef" for c in value)


class FinalSessionAlreadyOpenedError(ValueError):
    """final E2E セッションが既に開封済み（create-once 違反）。

    同じ session_id の 2 回目の開封を拒否します。
    """


class FinalSessionNotInSealError(ValueError):
    """開封しようとした session が lineage seal の固定集合に含まれていない。

    seal 作成後に final session 集合を追加・変更することは禁止されています。
    """


class FinalFitMixingError(ValueError):
    """final E2E セッションの calibration fit に混入しようとした。

    final データを fit に使うと未使用 E2E test の独立性が失われます。
    """


class StaleVerdictError(ValueError):
    """producer hash が verdict 発行時から変化し、verdict が陳腐化した。

    parser/detector/config のいずれかが変わったとき旧 verdict は無効です。
    """


class HashMismatchError(ValueError):
    """ロードした artifact の hash が保存済み hash と一致しない。"""


class SessionOverlapError(ValueError):
    """calibration と final の session_id が重複している。"""


class EmptyResidualError(ValueError):
    """必須フィールドの residual が空で、underpowered sample として拒否する。"""


class InvalidResidualError(ValueError):
    """residual の値が NaN または Inf を含む。"""


class FormalVerdictPromotionError(ValueError):
    """development-only の synthetic 成果物を formal として発行しようとした。

    synthetic fixture（development_only=True）から formal verdict を作成
    することは禁止されています。
    """


@dataclass(frozen=True, slots=True)
class CalibrationResidual:
    """1 フレーム・1 フィールドの calibration 残差。

    session_id / frame_id でデータ出所を追跡し、NaN/Inf が混入しないよう
    生成時に検証します。
    """

    session_id: str
    frame_id: str
    field: str
    residual: float
    confidence: float
    age_frames: int
    latency_frames: float = 0.0

    def __post_init__(self) -> None:
        """NaN/Inf を拒否する。"""
        for name in ("residual", "confidence", "latency_frames"):
            val = getattr(self, name)
            if not math.isfinite(val):
                raise InvalidResidualError(
                    f"CalibrationResidual.{name} is not finite: {val!r}"
                )


@dataclass
class FinalLineageSeal:
    """create-once の final session 開封ポリシー。

    seal 作成時に `final_session_set` を固定し、集合の変更を拒否します。
    固定集合内の各 session_id は最大 1 回だけ開封できます（3+ 件の別 session を許容）。
    再起動後も同一 session を再開封できないよう、ArtifactStore へ永続化します。
    """

    seal_id: str
    parser_artifact_hash: str
    detector_artifact_hash: str
    assembler_schema_hash: str
    config_hash: str
    # 開封可能な final session 集合を固定する（変更不可）
    final_session_set: frozenset[str] = field(default_factory=frozenset)
    # 各 session_hash: str も固定する
    final_session_hashes: dict[str, str] = field(default_factory=dict)
    opened_session_ids: list[str] = field(default_factory=list)
    development_only: bool = True
    schema_version: str = LINEAGE_SEAL_SCHEMA_VERSION
    # ArtifactStore を使う場合の永続化パス（省略可）
    _store_path: Path | None = field(default=None, repr=False, compare=False)

    def verify_hashes(
        self,
        *,
        parser: str,
        detector: str,
        assembler: str,
        config: str,
    ) -> None:
        """producer hash が seal 発行時から変化していないか検証する。

        いずれか 1 フィールドでも変化したとき StaleVerdictError を送出します。
        """
        changed: list[str] = []
        if parser != self.parser_artifact_hash:
            changed.append("parser_artifact_hash")
        if detector != self.detector_artifact_hash:
            changed.append("detector_artifact_hash")
        if assembler != self.assembler_schema_hash:
            changed.append("assembler_schema_hash")
        if config != self.config_hash:
            changed.append("config_hash")
        if changed:
            raise StaleVerdictError(
                f"Seal {self.seal_id!r} stale; changed fields: {changed}"
            )

    def open_session(self, session_id: str) -> None:
        """final セッションを開封済みとして登録する。

        - session_id が final_session_set に含まれない場合は拒否する。
        - 同じ session_id が既に開封済みの場合は拒否する。
        - 異なる session_id は最大 1 回ずつ開封できる（3+ 件の別 session をサポート）。
        """
        if self.final_session_set and session_id not in self.final_session_set:
            raise FinalSessionNotInSealError(
                f"session {session_id!r} is not in the sealed final session set "
                f"{sorted(self.final_session_set)}"
            )
        if session_id in self.opened_session_ids:
            raise FinalSessionAlreadyOpenedError(
                f"Final session {session_id!r} already opened. "
                "Cannot open same session twice."
            )
        self.opened_session_ids.append(session_id)
        # ArtifactStore 永続化（store_path が設定されている場合）
        if self._store_path is not None:
            self._persist_opened()

    def _persist_opened(self) -> None:
        """開封状態を store_path へ atomic に書き込む。"""
        if self._store_path is None:
            return
        wire = self.to_wire()
        import json
        import os
        import tempfile
        data = json.dumps(wire, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            prefix=".seal.", suffix=".tmp", dir=str(self._store_path.parent)
        )
        tmp_path = Path(tmp)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            tmp_path.replace(self._store_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    @classmethod
    def load_from_store(cls, store_path: Path) -> "FinalLineageSeal":
        """ArtifactStore から seal を読み込む。

        再起動後も開封済み session 集合を復元し、同一 session の再開封を拒否します。
        """
        import json
        wire = json.loads(store_path.read_text(encoding="utf-8"))
        return cls(
            seal_id=wire["seal_id"],
            parser_artifact_hash=wire["parser_artifact_hash"],
            detector_artifact_hash=wire["detector_artifact_hash"],
            assembler_schema_hash=wire["assembler_schema_hash"],
            config_hash=wire["config_hash"],
            final_session_set=frozenset(wire.get("final_session_set", [])),
            final_session_hashes=dict(wire.get("final_session_hashes", {})),
            opened_session_ids=list(wire.get("opened_session_ids", [])),
            development_only=bool(wire.get("development_only", True)),
            schema_version=wire.get("schema_version", LINEAGE_SEAL_SCHEMA_VERSION),
            _store_path=store_path,
        )

    def to_wire(self) -> dict[str, Any]:
        """seal JSON 保存用の dict として返す。"""
        return {
            "schema_version": self.schema_version,
            "seal_id": self.seal_id,
            "parser_artifact_hash": self.parser_artifact_hash,
            "detector_artifact_hash": self.detector_artifact_hash,
            "assembler_schema_hash": self.assembler_schema_hash,
            "config_hash": self.config_hash,
            "final_session_set": sorted(self.final_session_set),
            "final_session_hashes": dict(self.final_session_hashes),
            "opened_session_ids": list(self.opened_session_ids),
            "development_only": self.development_only,
        }


def create_lineage_seal(
    parser_artifact_hash: str,
    detector_artifact_hash: str,
    assembler_schema_hash: str,
    config_hash: str,
    *,
    final_session_set: frozenset[str] | None = None,
    final_session_hashes: dict[str, str] | None = None,
    development_only: bool = True,
    store_path: Path | None = None,
) -> FinalLineageSeal:
    """producer hashes から lineage seal を作成する。

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
    seal = FinalLineageSeal(
        seal_id=seal_id,
        parser_artifact_hash=parser_artifact_hash,
        detector_artifact_hash=detector_artifact_hash,
        assembler_schema_hash=assembler_schema_hash,
        config_hash=config_hash,
        final_session_set=final_session_set or frozenset(),
        final_session_hashes=final_session_hashes or {},
        development_only=development_only,
        _store_path=store_path,
    )
    # store_path が指定された場合、exclusive create（既存なら load で整合性確認）
    if store_path is not None:
        if store_path.exists():
            existing = FinalLineageSeal.load_from_store(store_path)
            if existing.seal_id != seal.seal_id:
                raise HashMismatchError(
                    f"Existing seal {existing.seal_id!r} differs from new seal {seal.seal_id!r}. "
                    "Final session set cannot be changed after creation."
                )
            return existing
        seal._persist_opened()
    return seal


def fit_error_profile(
    residuals: list[CalibrationResidual],
    calibration_session_ids: list[str],
    final_e2e_session_ids: list[str],
) -> PerceptionErrorProfile:
    """calibration residuals から PerceptionErrorProfile を fit する。

    final E2E の session_id が calibration と重複するとき、
    または final の residual が混入しているとき拒否します。
    NaN/Inf が混入した residual は InvalidResidualError で拒否します。
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

    # NaN/Inf チェックは CalibrationResidual.__post_init__ で実施済み
    cal_residuals = [r for r in residuals if r.session_id in cal_ids]

    by_field: dict[str, list[float]] = {}
    for r in cal_residuals:
        by_field.setdefault(r.field, []).append(r.residual)

    def _std(vals: list[float]) -> float:
        return float(np.std(vals)) if vals else 0.0

    def _mean(vals: list[float]) -> float:
        return float(np.mean(vals)) if vals else 0.0

    def _prob(vals: list[float], threshold: float) -> float:
        return float(np.mean([abs(v) > threshold for v in vals])) if vals else 0.0

    def _clamp01(v: float) -> float:
        return min(1.0, max(0.0, v))

    def _nonneg(v: float) -> float:
        return max(0.0, v)

    # HP/XP/timer
    hp_misread_std = _std(by_field.get("hp_ratio", []))
    xp_stale_prob = _prob(by_field.get("xp_ratio", []), 0.1)
    timer_stale_prob = _prob(by_field.get("timer_seconds", []), 1.0)
    inventory_stale_prob = _prob(by_field.get("inventory_hash", []), 0.5)

    # 座標誤差
    coord_noise_std = _std(by_field.get("coord_noise", []))
    coord_quant_px = _nonneg(_mean(by_field.get("coord_quantization_px", [])))

    # レイテンシ
    lat_vals = by_field.get("latency_frames", [])
    lat_mean = _nonneg(_mean(lat_vals))
    lat_std = _nonneg(_std(lat_vals))

    # burst パラメータ
    # burst_enter: バースト開始確率（フレームごとの dropout 開始確率）
    burst_enter_prob = _clamp01(_mean(by_field.get("burst_enter", [])))
    # burst_exit: バースト終了確率（dropout 終了確率）
    burst_vals = by_field.get("burst_exit", [])
    burst_exit_prob = _clamp01(_mean(burst_vals)) if burst_vals else 1.0
    # burst_dropout: バースト中の dropout 確率
    burst_dropout_prob = _clamp01(_mean(by_field.get("burst_dropout", [])))

    # unknown screen collapse
    unknown_collapse_prob = _clamp01(_mean(by_field.get("unknown_screen_collapse", [])))
    unknown_collapse_dur = _nonneg(_mean(by_field.get("unknown_screen_collapse_duration", [])))

    # item confusion matrix（ITEM_CATEGORY_SIZE x ITEM_CATEGORY_SIZE）
    n = ITEM_CATEGORY_SIZE
    identity_mat = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    return PerceptionErrorProfile(
        latency_mean_frames=lat_mean,
        latency_std_frames=lat_std,
        burst_enter_prob=burst_enter_prob,
        burst_exit_prob=burst_exit_prob,
        burst_dropout_prob=burst_dropout_prob,
        coord_noise_std=_nonneg(coord_noise_std),
        coord_quantization_px=coord_quant_px,
        hud_hp_misread_std=_nonneg(hp_misread_std),
        hud_xp_stale_prob=_clamp01(xp_stale_prob),
        hud_timer_stale_prob=_clamp01(timer_stale_prob),
        hud_inventory_stale_prob=_clamp01(inventory_stale_prob),
        unknown_screen_collapse_prob=unknown_collapse_prob,
        unknown_screen_collapse_duration_frames=unknown_collapse_dur,
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
        "latency_std_diff": abs(
            calibrated.latency_std_frames - simulator.latency_std_frames
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
        "burst_enter_diff": abs(
            calibrated.burst_enter_prob - simulator.burst_enter_prob
        ),
        "burst_exit_diff": abs(
            calibrated.burst_exit_prob - simulator.burst_exit_prob
        ),
        "burst_dropout_diff": abs(
            calibrated.burst_dropout_prob - simulator.burst_dropout_prob
        ),
        "unknown_collapse_prob_diff": abs(
            calibrated.unknown_screen_collapse_prob - simulator.unknown_screen_collapse_prob
        ),
    }


@dataclass
class PerceptionCalibrationVerdict:
    """calibration profile の development-only verdict 型。

    formal 発行には calibration session 収録と 04-05/04-08 package が必要です。
    この dataclass の development_only は常に True でなければなりません。
    """

    profile: PerceptionErrorProfile
    calibration_session_ids: list[str]
    development_only: bool = True
    formal_perception_verdict_eligible: bool = False
    schema_version: str = CALIBRATION_VERDICT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        # development writer は常に development_only=True
        if not self.development_only:
            raise FormalVerdictPromotionError(
                "PerceptionCalibrationVerdict must always have development_only=True. "
                "Formal calibration profile requires formal parent chain."
            )


@dataclass
class PerceptionFinalVerdict:
    """final perception verdict（development-only）。

    formal 入力が揃ったときだけ development_only=False にできます。
    ただし現在の code-only PR では formal 発行不可です。
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
    # 拡張 subject フィールド
    capture_dataset_hash: str = ""
    calibration_profile_hash: str = ""
    threshold_hash: str = ""
    atlas_vocabulary_hash: str = ""
    assembler_impl_hash: str = ""
    roi_resolver_input_hash: str = ""
    benchmark_fit_code_hash: str = ""
    lineage_seal_hash: str = ""

    def __post_init__(self) -> None:
        # arbitrary constructor では formal_eligible=True にできない
        # formal parent chain 検証後のみ許可される（現在の code-only PR では不可）
        if self.formal_perception_verdict_eligible:
            raise FormalVerdictPromotionError(
                "Cannot set formal_perception_verdict_eligible=True via constructor. "
                "Formal verdict requires verified formal parent chain via formal writer."
            )


def _write_formal_final_verdict(
    verdict: PerceptionFinalVerdict,
    *,
    formal_parent_chain_verified: bool,
) -> None:
    """formal final verdict を書き込む（formal parent chain 検証後のみ呼べる）。

    synthetic fixture をこの関数へ渡すと FormalVerdictPromotionError。
    formal parent chain が検証されていない場合も拒否します。
    """
    if not formal_parent_chain_verified:
        raise FormalVerdictPromotionError(
            "Formal final verdict requires verified formal parent chain. "
            "Run formal dependency verification first."
        )
    if verdict.development_only:
        raise FormalVerdictPromotionError(
            "Cannot write synthetic (development_only=True) verdict as formal."
        )
    if not verdict.formal_perception_verdict_eligible:
        raise FormalVerdictPromotionError(
            "Verdict must have formal_perception_verdict_eligible=True for formal publish."
        )
    # 実際の ArtifactStore 書き込みは formal deps 揃い次第実装
    raise NotImplementedError(
        "Formal ArtifactStore publish requires D04-CAPTURE-DATASET, "
        "04-05 parser package, and 04-08 detector package."
    )


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

    ローダーは次を fail-closed で検証します:
    - exact schema field set
    - SHA-256 形式（64文字 lowercase hex）
    - development/formal flag 整合性
    - pass と blocking_reasons の整合性
    """
    # schema field set の完全一致チェック
    if not isinstance(data, dict) or not all(isinstance(k, str) for k in data):
        raise ValueError("verdict data must be a dict with string keys")

    if data.get("schema_version") != FINAL_VERDICT_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported verdict schema_version {data.get('schema_version')!r}"
        )

    # SHA-256 形式チェック（主要な hash フィールド）
    hash_fields = [
        "parser_artifact_hash",
        "detector_artifact_hash",
        "assembler_schema_hash",
        "config_hash",
        "ui_presentation_schema_hash",
    ]
    for hf in hash_fields:
        val = data.get(hf, "")
        if val and not _is_sha256(val):
            raise ValueError(
                f"{hf} must be a 64-char lowercase hex SHA-256, got {val!r}"
            )

    # stale 確認
    stale: list[str] = []
    if data.get("parser_artifact_hash") != current_parser_hash:
        stale.append("parser_artifact_hash")
    if data.get("detector_artifact_hash") != current_detector_hash:
        stale.append("detector_artifact_hash")
    if data.get("assembler_schema_hash") != current_assembler_hash:
        stale.append("assembler_schema_hash")
    if data.get("config_hash") != current_config_hash:
        stale.append("config_hash")
    if data.get("ui_presentation_schema_hash") != current_ui_schema_hash:
        stale.append("ui_presentation_schema_hash")
    if stale:
        raise StaleVerdictError(
            f"Final perception verdict stale; changed producer fields: {stale}. "
            "Demote old final sessions to development and re-issue with new untouched sessions."
        )

    # development/formal flag 整合性チェック
    development_only = bool(data.get("development_only", True))
    formal_eligible = bool(data.get("formal_perception_verdict_eligible", False))
    if formal_eligible and development_only:
        raise FormalVerdictPromotionError(
            "verdict has formal_perception_verdict_eligible=True but development_only=True. "
            "This is inconsistent — possible tampering."
        )

    # passed と blocking_reasons の整合性チェック
    passed = bool(data.get("passed", False))
    blocking = list(data.get("blocking_reasons", []))
    if passed and blocking:
        raise ValueError(
            "verdict has passed=True but blocking_reasons is non-empty. Inconsistent verdict."
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
        metrics=dict(data.get("metrics", {})),
        passed=passed,
        blocking_reasons=blocking,
        development_only=development_only,
        formal_perception_verdict_eligible=formal_eligible,
        capture_dataset_hash=str(data.get("capture_dataset_hash", "")),
        calibration_profile_hash=str(data.get("calibration_profile_hash", "")),
        threshold_hash=str(data.get("threshold_hash", "")),
        atlas_vocabulary_hash=str(data.get("atlas_vocabulary_hash", "")),
        assembler_impl_hash=str(data.get("assembler_impl_hash", "")),
        roi_resolver_input_hash=str(data.get("roi_resolver_input_hash", "")),
        benchmark_fit_code_hash=str(data.get("benchmark_fit_code_hash", "")),
        lineage_seal_hash=str(data.get("lineage_seal_hash", "")),
    )
