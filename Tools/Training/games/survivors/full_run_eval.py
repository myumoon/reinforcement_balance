"""Survivors Full Run の exact target と generalization 評価を分離する。

初心者向け: 固定30 seed の昇格判定と、条件を広げた層別レポートを別クラスで扱い、
評価結果には利用した policy・selector・source・schema・profile を束縛します。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


EXACT_TARGET_HOLDOUT_SEEDS: tuple[int, ...] = (
    73013, 73019, 73037, 73039, 73043, 73061, 73063, 73079, 73091, 73121,
    73127, 73133, 73141, 73181, 73189, 73237, 73243, 73259, 73277, 73291,
    73303, 73309, 73327, 73331, 73351, 73361, 73363, 73369, 73379, 73387,
)
FULL_RUN_VERDICT_SCHEMA_VERSION = "survivors.full_run_verdict.v1"
GENERALIZATION_MANIFEST_SCHEMA_VERSION = "survivors.full_run_generalization_manifest.v1"
_BINDING_KEYS = frozenset({
    "policy_hash", "selector_hash", "source_hash", "schema_hash", "profile_hash",
})
_TERMINAL_REASONS = frozenset({"stage_cleared", "death", "timeout"})


def _finite_number(
    value: Any,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """bool を除く有限な実数を範囲付きで検証する。

    初心者向け: Python では bool も int の一種なので、gate 閾値として明示的に拒否します。
    """
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    checked = float(value)
    if minimum is not None and checked < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    if maximum is not None and checked > maximum:
        raise ValueError(f"{label} must be <= {maximum}")
    return checked


def _sha256(value: Any, label: str) -> str:
    """lowercase SHA-256 文字列だけを受け付ける。

    初心者向け: 空文字や path を hash の代わりに保存してしまう事故を防ぎます。
    """
    if not (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _bindings(value: Mapping[str, Any]) -> dict[str, str]:
    """verdict binding の全 hash を exact-key で検証する。

    初心者向け: 一つでも欠けた評価結果を別 artifact へ流用できないようにします。
    """
    if not isinstance(value, Mapping) or set(value) != set(_BINDING_KEYS):
        raise ValueError(f"bindings keys must be {sorted(_BINDING_KEYS)}")
    return {key: _sha256(value[key], key) for key in sorted(_BINDING_KEYS)}


@dataclass(frozen=True)
class EpisodeOutcome:
    """一回の 30分評価 episode の terminal outcome。

    初心者向け: stage clear、death、timeout を明示し、単なる done flag の解釈を避けます。
    """

    seed: int
    stratum: str
    stage_cleared: bool
    elapsed_seconds: float
    active_score: float
    terminal_reason: str

    def __post_init__(self) -> None:
        """episode の seed、数値、terminal 整合を検証する。

        初心者向け: 30分未満を stage clear とした行や NaN score を集計前に拒否します。
        """
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("episode seed must be a non-negative integer")
        if not isinstance(self.stratum, str) or not self.stratum:
            raise ValueError("episode stratum must be non-empty")
        if type(self.stage_cleared) is not bool:
            raise ValueError("stage_cleared must be bool")
        if not math.isfinite(self.elapsed_seconds) or self.elapsed_seconds < 0.0:
            raise ValueError("elapsed_seconds must be finite non-negative")
        if not math.isfinite(self.active_score):
            raise ValueError("active_score must be finite")
        if self.terminal_reason not in _TERMINAL_REASONS:
            raise ValueError("terminal_reason must distinguish stage_cleared/death/timeout")
        if self.stage_cleared != (
            self.terminal_reason == "stage_cleared" and self.elapsed_seconds >= 1800.0
        ):
            raise ValueError("stage_cleared requires the 1800 second success terminal")


@dataclass(frozen=True)
class FullRunVerdict:
    """hash-bound Full Run 評価 verdict。

    初心者向け: exact の合否と generalization の層別値を同じ identity field 付きで保存します。
    """

    report_kind: str
    passed: bool
    episode_count: int
    clear_rate: float
    active_score_p10: float
    strata_metrics: Mapping[str, Mapping[str, int | float]]
    bindings: Mapping[str, str]

    def __post_init__(self) -> None:
        """verdict の型・範囲・binding を直接構築時にも検証する。

        初心者向け: JSON 化する前から不完全な評価結果を作れないようにします。
        """
        if self.report_kind not in {"exact_target", "generalization"}:
            raise ValueError("invalid full-run report_kind")
        if type(self.passed) is not bool or type(self.episode_count) is not int or self.episode_count <= 0:
            raise ValueError("full-run verdict episode_count must be positive")
        if not (0.0 <= self.clear_rate <= 1.0):
            raise ValueError("full-run clear_rate must be in [0, 1]")
        if not math.isfinite(self.active_score_p10):
            raise ValueError("full-run active_score_p10 must be finite")
        _bindings(self.bindings)

    def to_wire(self) -> dict[str, Any]:
        """required hashes を top-level に持つ verdict JSON object を返す。

        初心者向け: artifact 一覧から hash を直接確認でき、全 report で同じ field 名を使います。
        """
        result: dict[str, Any] = {
            "schema_version": FULL_RUN_VERDICT_SCHEMA_VERSION,
            "report_kind": self.report_kind,
            "passed": self.passed,
            "episode_count": self.episode_count,
            "clear_rate": self.clear_rate,
            "active_score_p10": self.active_score_p10,
            "strata_metrics": {
                key: dict(value) for key, value in sorted(self.strata_metrics.items())
            },
        }
        result.update(_bindings(self.bindings))
        return result


class ExactTargetFullRunEvaluator:
    """固定30 holdout seeds に clear_rate と active-score p10 gate を適用する。

    初心者向け: 同じ本番目標だけを昇格判定に使い、汎化用 seed の混入を拒否します。
    """

    def __init__(
        self,
        *,
        min_clear_rate: float = 0.80,
        min_active_score_p10: float = 2250.0,
    ) -> None:
        """exact gate の固定 threshold を検証して保持する。

        初心者向け: clear率は80%以上を標準とし、score p10 は明示的に上書きできます。
        """
        self.min_clear_rate = _finite_number(
            min_clear_rate,
            "min_clear_rate",
            minimum=0.80,
            maximum=1.0,
        )
        self.min_active_score_p10 = _finite_number(
            min_active_score_p10,
            "min_active_score_p10",
            minimum=0.0,
        )

    def evaluate(
        self,
        outcomes: Sequence[EpisodeOutcome],
        *,
        bindings: Mapping[str, Any],
    ) -> FullRunVerdict:
        """30固定 seed を一度ずつ集計して exact verdict を返す。

        初心者向け: seed の欠落・重複・差替えは gate 計算前に失敗します。
        """
        seeds = [row.seed for row in outcomes]
        if len(outcomes) != 30 or set(seeds) != set(EXACT_TARGET_HOLDOUT_SEEDS) or len(seeds) != len(set(seeds)):
            raise ValueError("exact target outcomes must match the 30 fixed holdout seeds")
        if any(row.stratum != "exact_target" for row in outcomes):
            raise ValueError("exact target outcomes must use stratum='exact_target'")
        clear_rate = sum(row.stage_cleared for row in outcomes) / len(outcomes)
        score_p10 = float(np.percentile([row.active_score for row in outcomes], 10))
        passed = clear_rate >= self.min_clear_rate and score_p10 >= self.min_active_score_p10
        return FullRunVerdict(
            "exact_target", passed, len(outcomes), clear_rate, score_p10, {}, _bindings(bindings)
        )


@dataclass(frozen=True)
class GeneralizationManifest:
    """generalization seeds を stratum 別に固定する manifest。

    初心者向け: 武器開始条件や perception noise などを混ぜずに集計できる割当表です。
    """

    strata: Mapping[str, tuple[int, ...]]

    @classmethod
    def from_wire(cls, value: Any) -> "GeneralizationManifest":
        """未知 field、空 stratum、seed 重複を拒否して manifest を読む。

        初心者向け: typo を新しい条件として黙認せず、seed を二重集計しません。
        """
        if not isinstance(value, Mapping) or set(value) != {"schema_version", "strata"}:
            raise ValueError("generalization manifest keys mismatch")
        if value["schema_version"] != GENERALIZATION_MANIFEST_SCHEMA_VERSION:
            raise ValueError("unsupported generalization manifest schema_version")
        raw = value["strata"]
        if not isinstance(raw, Mapping) or not raw:
            raise ValueError("generalization manifest strata must be non-empty")
        checked: dict[str, tuple[int, ...]] = {}
        all_seeds: list[int] = []
        for name, seeds in raw.items():
            if not isinstance(name, str) or not name or not isinstance(seeds, list) or not seeds:
                raise ValueError("generalization manifest stratum is invalid")
            if any(type(seed) is not int or seed < 0 for seed in seeds):
                raise ValueError("generalization manifest seeds must be non-negative integers")
            checked[name] = tuple(seeds)
            all_seeds.extend(seeds)
        if len(all_seeds) != len(set(all_seeds)):
            raise ValueError("generalization manifest seeds must be globally unique")
        return cls(checked)


class GeneralizationFullRunEvaluator:
    """manifest の各 stratum を独立集計する generalization evaluator。

    初心者向け: exact target の昇格判定とは別に、条件ごとの弱点を report します。
    """

    def __init__(self, manifest: GeneralizationManifest) -> None:
        """検証済み generalization manifest を保持する。

        初心者向け: 評価途中で seed 割当が変わらない immutable manifest だけを受け取ります。
        """
        if not isinstance(manifest, GeneralizationManifest):
            raise TypeError("manifest must be GeneralizationManifest")
        self.manifest = manifest

    def evaluate(
        self,
        outcomes: Sequence[EpisodeOutcome],
        *,
        bindings: Mapping[str, Any],
    ) -> FullRunVerdict:
        """全 manifest cells を一度ずつ集計し strata metrics を返す。

        初心者向け: 欠けた条件や別 stratum に入った seed があれば report を作りません。
        """
        expected = {
            (stratum, seed)
            for stratum, seeds in self.manifest.strata.items()
            for seed in seeds
        }
        actual = [(row.stratum, row.seed) for row in outcomes]
        if len(actual) != len(set(actual)) or set(actual) != expected:
            raise ValueError("generalization outcomes must match all manifest cells exactly")
        metrics: dict[str, dict[str, int | float]] = {}
        for stratum in sorted(self.manifest.strata):
            rows = [row for row in outcomes if row.stratum == stratum]
            metrics[stratum] = {
                "episodes": len(rows),
                "clear_rate": sum(row.stage_cleared for row in rows) / len(rows),
            }
        clear_rate = sum(row.stage_cleared for row in outcomes) / len(outcomes)
        score_p10 = float(np.percentile([row.active_score for row in outcomes], 10))
        return FullRunVerdict(
            "generalization", True, len(outcomes), clear_rate, score_p10, metrics, _bindings(bindings)
        )
