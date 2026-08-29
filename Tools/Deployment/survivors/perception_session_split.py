"""Survivors perception benchmark のセッション split 検証。

capture_dataset.py の SplitManifest と同じ split 名を使い、
calibration/final の重複排除・build 一致・スライス最小件数・最短時間・
session content hash 一意性・clock regression 拒否を行います。
MP4 decode を benchmark 入力として拒否します。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

# capture_dataset.py SPLIT_NAMES と完全一致させる
SessionKind = Literal["model_train", "model_validation", "error_calibration", "final_e2e_test"]
SourcePolicy = Literal["raw", "lossless", "mp4"]

_CALIBRATION_KIND: Final[str] = "error_calibration"
_FINAL_KIND: Final[str] = "final_e2e_test"
_BENCHMARK_KINDS: Final[frozenset[str]] = frozenset({_CALIBRATION_KIND, _FINAL_KIND})
_ALL_KINDS: Final[frozenset[str]] = frozenset(
    {"model_train", "model_validation", "error_calibration", "final_e2e_test"}
)

_MIN_SESSION_SECONDS: Final[float] = 1800.0  # 30 分


class SplitOverlapError(ValueError):
    """session_id または session_hash が複数の split kind に登録されている。

    calibration/final/train/validation 間でセッションが重複すると評価が汚染されます。
    """


class MixedBuildError(ValueError):
    """benchmark セッション間で build/profile/resolution が一致しない。

    異なるビルドや解像度を同一ベンチマークへ混ぜると比較基準が崩れます。
    """


class UnderpoweredSliceError(ValueError):
    """calibration または final_e2e_test のセッション数が最小件数を下回る。

    cluster CI の信頼性確保のために最低 min_benchmark_sessions 件が必要です。
    """


class ShortSessionError(ValueError):
    """セッションの収録時間が最低 30 分を下回る。

    短時間セッションは rare-slice の標本数が不足し CI が発散します。
    """


class ClockRegressionError(ValueError):
    """フレームタイムスタンプが単調増加でない（clock regression 検出）。

    monotonic でないタイムスタンプが混在すると latency 集計が壊れます。
    """


class MissingBenchmarkSplitError(ValueError):
    """calibration または final_e2e_test の片側が欠落している。

    error_calibration と final_e2e_test は両方揃っていないと
    calibration fit と final verification が実行できません。
    """


@dataclass(frozen=True, slots=True)
class SessionRecord:
    """1 セッション分の benchmark メタデータ。

    split 検証と build 一致確認に必要な属性だけを保持します。
    session_hash はセッションの content hash（manifest SHA-256）で、
    split 横断の一意性を保証するために使います。
    """

    session_id: str
    session_hash: str
    build_hash: str
    target_profile_hash: str
    resolution_wh: tuple[int, int]
    duration_seconds: float
    kind: SessionKind
    source_policy: SourcePolicy


@dataclass(frozen=True)
class SessionSplit:
    """検証済みセッションリスト。

    validate_split() が返す不変オブジェクトで、重複・混在を除去済みです。
    """

    sessions: tuple[SessionRecord, ...]


def validate_split(
    sessions: list[SessionRecord],
    *,
    min_benchmark_sessions: int = 3,
) -> SessionSplit:
    """セッションリストを split ルールで検証する。

    以下の条件をすべて満たすとき SessionSplit を返します。
    - 全split間で session_id が重複しない。
    - 全split間で session_hash（content hash）が重複しない。
    - calibration + final のセッションはすべて同一 build/profile/resolution。
    - calibration か final の一方でも存在すれば、両方が min_benchmark_sessions 以上ある。
    - calibration/final の各セッションが最低 30 分。
    """
    # session_id と session_hash の split 横断一意性を検証
    id_to_kinds: dict[str, list[str]] = {}
    hash_to_kinds: dict[str, list[str]] = {}
    for s in sessions:
        id_to_kinds.setdefault(s.session_id, []).append(s.kind)
        hash_to_kinds.setdefault(s.session_hash, []).append(s.kind)

    for sid, kinds in id_to_kinds.items():
        if len(set(kinds)) > 1:
            raise SplitOverlapError(
                f"session_id {sid!r} appears in multiple split kinds: {sorted(set(kinds))}"
            )

    for shash, kinds in hash_to_kinds.items():
        if len(set(kinds)) > 1:
            raise SplitOverlapError(
                f"session_hash {shash!r} appears in multiple split kinds: {sorted(set(kinds))}"
            )

    # build/profile/resolution の一致確認（benchmark セッション間）
    benchmark = [s for s in sessions if s.kind in _BENCHMARK_KINDS]
    if benchmark:
        ref = benchmark[0]
        for s in benchmark[1:]:
            if (
                s.build_hash != ref.build_hash
                or s.target_profile_hash != ref.target_profile_hash
                or s.resolution_wh != ref.resolution_wh
            ):
                raise MixedBuildError(
                    f"session {s.session_id!r} has different build/profile/resolution "
                    f"from reference {ref.session_id!r}"
                )

    # benchmark セッションの最短時間チェック
    for s in benchmark:
        if s.duration_seconds < _MIN_SESSION_SECONDS:
            raise ShortSessionError(
                f"session {s.session_id!r} is {s.duration_seconds:.1f}s "
                f"(minimum {_MIN_SESSION_SECONDS:.0f}s)"
            )

    # calibration/final 双方が揃っているかチェック
    cal_count = sum(1 for s in sessions if s.kind == _CALIBRATION_KIND)
    fin_count = sum(1 for s in sessions if s.kind == _FINAL_KIND)
    any_benchmark = cal_count > 0 or fin_count > 0
    if any_benchmark:
        if cal_count == 0:
            raise MissingBenchmarkSplitError(
                "error_calibration sessions are required when final_e2e_test sessions exist"
            )
        if fin_count == 0:
            raise MissingBenchmarkSplitError(
                "final_e2e_test sessions are required when error_calibration sessions exist"
            )
        if cal_count < min_benchmark_sessions:
            raise UnderpoweredSliceError(
                f"error_calibration has only {cal_count} session(s), "
                f"minimum is {min_benchmark_sessions}"
            )
        if fin_count < min_benchmark_sessions:
            raise UnderpoweredSliceError(
                f"final_e2e_test has only {fin_count} session(s), "
                f"minimum is {min_benchmark_sessions}"
            )

    return SessionSplit(sessions=tuple(sessions))


def validate_frame_timestamps(timestamps_ns: list[int]) -> None:
    """フレームタイムスタンプの strict monotonicity を検証する。

    同一セッション内でタイムスタンプが減少または同値の場合、
    clock regression として ClockRegressionError を送出します。
    """
    for i in range(1, len(timestamps_ns)):
        if timestamps_ns[i] <= timestamps_ns[i - 1]:
            raise ClockRegressionError(
                f"clock regression at frame {i}: "
                f"{timestamps_ns[i]} <= {timestamps_ns[i - 1]}"
            )


def validate_source_policy_consistency(
    sessions: list[SessionRecord],
    *,
    benchmark_kind: SessionKind,
) -> None:
    """benchmark 入力に MP4 decode が使われていないことを確認する。

    MP4 は domain-shift 比較専用で、calibration/final のスコア計算に使えません。
    """
    for s in sessions:
        if s.kind == benchmark_kind and s.source_policy == "mp4":
            raise ValueError(
                f"session {s.session_id!r}: mp4 decode not allowed as benchmark input "
                "(domain-shift comparison only); use raw or lossless"
            )
