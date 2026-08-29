"""Survivors perception benchmark のセッション split 検証。

calibration/final の重複排除、build 一致確認、スライス最小件数確認を行い、
MP4 decode を benchmark 入力として拒否します。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

SessionKind = Literal["train", "validation", "calibration", "final_e2e_test"]
SourcePolicy = Literal["raw", "lossless", "mp4"]

_BENCHMARK_KINDS: Final[frozenset[str]] = frozenset({"calibration", "final_e2e_test"})
_ALL_KINDS: Final[frozenset[str]] = frozenset(
    {"train", "validation", "calibration", "final_e2e_test"}
)


class SplitOverlapError(ValueError):
    """session_id が複数の split kind に登録されている。

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


@dataclass(frozen=True, slots=True)
class SessionRecord:
    """1 セッション分の benchmark メタデータ。

    split 検証と build 一致確認に必要な属性だけを保持します。
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
    - calibration/final/train/validation 間で session_id が重複しない。
    - calibration + final の session はすべて同一 build/profile/resolution。
    - calibration または final が 1 件以上なら min_benchmark_sessions 以上ある。
    """
    kind_to_ids: dict[str, set[str]] = {}
    for s in sessions:
        kind_to_ids.setdefault(s.kind, set()).add(s.session_id)

    all_kinds = sorted(_ALL_KINDS)
    for i, k1 in enumerate(all_kinds):
        for k2 in all_kinds[i + 1:]:
            overlap = kind_to_ids.get(k1, set()) & kind_to_ids.get(k2, set())
            if overlap:
                raise SplitOverlapError(
                    f"session overlap between {k1!r} and {k2!r}: {sorted(overlap)}"
                )

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

    for kind in _BENCHMARK_KINDS:
        n = len(kind_to_ids.get(kind, set()))
        if 0 < n < min_benchmark_sessions:
            raise UnderpoweredSliceError(
                f"{kind!r} has only {n} session(s), minimum is {min_benchmark_sessions}"
            )

    return SessionSplit(sessions=tuple(sessions))


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
