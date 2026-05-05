"""
Feature engineering for the ML pipeline.

Extracts per-subject features from raw study log records.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass
class SubjectFeatures:
    subject: str
    avg_score: float
    trend: float          # slope of score over sessions
    efficiency: float     # score per study-hour
    session_count: int
    raw_features: list[float] = field(init=False)

    def __post_init__(self) -> None:
        self.raw_features = [self.avg_score, self.trend, self.efficiency]


def compute_trend(scores: Sequence[float]) -> float:
    """Linear regression slope of scores over session index."""
    n = len(scores)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    y = np.asarray(scores, dtype=float)
    x_mean, y_mean = x.mean(), y.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom == 0:
        return 0.0
    return float(((x - x_mean) * (y - y_mean)).sum() / denom)


def engineer_features(
    subject: str,
    scores: Sequence[float],
    hours: Sequence[float],
) -> SubjectFeatures:
    """Compute all features for a single subject."""
    scores_arr = np.asarray(scores, dtype=float)
    hours_arr = np.asarray(hours, dtype=float)

    avg_score = float(scores_arr.mean()) if len(scores_arr) else 0.0
    trend = compute_trend(scores_arr)
    total_hours = float(hours_arr.sum())
    efficiency = avg_score / total_hours if total_hours > 0 else 0.0

    return SubjectFeatures(
        subject=subject,
        avg_score=avg_score,
        trend=trend,
        efficiency=efficiency,
        session_count=len(scores_arr),
    )
