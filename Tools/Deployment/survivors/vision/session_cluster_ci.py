"""session を再標本化する比率型 bootstrap CI。

frame 数の多い session だけに信頼区間が支配されることを避けます。
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np


def cluster_ci_bounds(
    counts_by_session: Mapping[str, tuple[float, float]],
    *,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 0,
    min_applicable_sessions: int = 3,
) -> tuple[float, float]:
    """session cluster の比率型 percentile bootstrap 区間を返す。

    分母ゼロの session は除外し、各反復で applicable session を復元抽出します。
    """
    if not isinstance(counts_by_session, Mapping):
        raise TypeError("counts_by_session must be a mapping")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(confidence)
        or not 0.0 < confidence < 1.0
    ):
        raise ValueError("confidence must be finite and between 0 and 1")
    if type(resamples) is not int or resamples < 1:
        raise ValueError("resamples must be a positive integer")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if type(min_applicable_sessions) is not int or min_applicable_sessions < 1:
        raise ValueError("min_applicable_sessions must be a positive integer")

    applicable: list[tuple[float, float]] = []
    for session_id, pair in counts_by_session.items():
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session ids must be non-empty strings")
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError("each session value must contain numerator and denominator")
        numerator, denominator = pair
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in (numerator, denominator)
        ):
            raise ValueError("session counts must be finite numbers")
        if numerator < 0 or denominator < 0:
            raise ValueError("session counts must be non-negative")
        if denominator:
            applicable.append((float(numerator), float(denominator)))
    if len(applicable) < min_applicable_sessions:
        raise ValueError(
            f"need at least {min_applicable_sessions} applicable sessions; got {len(applicable)}"
        )

    values = np.asarray(applicable, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(values), size=(resamples, len(values)))
    numerators = values[samples, 0].sum(axis=1)
    denominators = values[samples, 1].sum(axis=1)
    ratios = numerators / denominators
    tail = (1.0 - float(confidence)) / 2.0
    lower, upper = np.quantile(ratios, (tail, 1.0 - tail))
    return float(lower), float(upper)
