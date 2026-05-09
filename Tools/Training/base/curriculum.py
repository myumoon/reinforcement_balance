"""Common helpers for game-specific curriculum callbacks."""

from __future__ import annotations

from dataclasses import dataclass


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return (sum((v - m) ** 2 for v in values) / len(values)) ** 0.5


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


@dataclass(frozen=True)
class CurriculumTuningInput:
    threshold_mult: float
    window: int
    episode_count: int
    recent_scores: list[float]
    base_threshold: float | None


class WindowThresholdRecommendationPolicy:
    """Suggests generic window/threshold values from recent curriculum scores."""

    def recommend(self, data: CurriculumTuningInput) -> dict:
        score_mean = mean(data.recent_scores)
        effective_threshold = (
            data.base_threshold * data.threshold_mult
            if data.base_threshold is not None
            else None
        )
        ratio = (
            score_mean / effective_threshold
            if effective_threshold is not None and effective_threshold > 0.0 and data.recent_scores
            else None
        )

        suggested_window = data.window
        window_reason = "current window has enough episode samples"
        if data.episode_count == 0:
            suggested_window = min(data.window, 10)
            window_reason = "no completed episodes were observed; use a smaller window after fixing truncation"
        elif len(data.recent_scores) < data.window:
            suggested_window = max(3, min(10, data.episode_count))
            window_reason = "fewer episodes than the current window were observed"
        elif data.recent_scores and score_mean > 0.0 and stdev(data.recent_scores) / score_mean > 1.0:
            suggested_window = min(30, data.window + 5)
            window_reason = "active_score is noisy; use a larger smoothing window"
        elif data.window > 20:
            suggested_window = 20
            window_reason = "window is larger than needed for normal eureka iterations"

        suggested_threshold = data.threshold_mult
        threshold_reason = "current threshold multiplier is in a reasonable range"
        if ratio is None:
            threshold_reason = "not enough active_score samples to tune threshold"
        elif ratio >= 1.8:
            suggested_threshold = clamp(data.threshold_mult * 1.25, 0.25, 5.0)
            threshold_reason = "phase threshold was exceeded by a large margin; make promotion harder"
        elif ratio >= 1.0:
            threshold_reason = "phase threshold was reached; keep multiplier unless promotion felt too fast"
        elif ratio < 0.4:
            suggested_threshold = clamp(data.threshold_mult * 0.5, 0.25, 5.0)
            threshold_reason = "active_score is far below threshold; make promotion easier"
        elif ratio < 0.75:
            suggested_threshold = clamp(data.threshold_mult * 0.75, 0.25, 5.0)
            threshold_reason = "active_score is below threshold; slightly lower promotion difficulty"

        return {
            "suggested_curriculum_threshold": round(suggested_threshold, 3),
            "suggested_curriculum_window": int(suggested_window),
            "threshold_reason": threshold_reason,
            "window_reason": window_reason,
        }
