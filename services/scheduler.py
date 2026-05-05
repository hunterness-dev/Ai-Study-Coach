"""
Rule-based weighted scheduler (fallback when RL model is unavailable).

priority = difficulty_weight + weakness_weight + negative_trend_weight
Allocations are normalised to total daily hours.
"""

from __future__ import annotations

from core.config import get_settings
from core.logging import get_logger
from ml.feature_engineering import SubjectFeatures

logger = get_logger(__name__)
settings = get_settings()

_MAX_SCORE = 100.0


def _difficulty_weight(feat: SubjectFeatures) -> float:
    """Subjects with lower scores are harder → higher priority."""
    return 1.0 - (feat.avg_score / _MAX_SCORE)


def _weakness_weight(feat: SubjectFeatures) -> float:
    """Subjects with poor efficiency get more time."""
    # Invert: lower efficiency = higher weight
    max_eff = 10.0  # empirical cap
    return 1.0 - min(feat.efficiency / max_eff, 1.0)


def _negative_trend_weight(feat: SubjectFeatures) -> float:
    """Penalise subjects with declining scores."""
    return max(-feat.trend / 10.0, 0.0)  # only negative trends add weight


def compute_priorities(features: list[SubjectFeatures]) -> dict[str, float]:
    priorities = {
        f.subject: (
            _difficulty_weight(f)
            + _weakness_weight(f)
            + _negative_trend_weight(f)
        )
        for f in features
    }
    return priorities


def allocate_hours(features: list[SubjectFeatures]) -> dict[str, float]:
    """Compute daily hour allocation for each subject."""
    if not features:
        return {}

    priorities = compute_priorities(features)
    total_priority = sum(priorities.values())

    if total_priority == 0:
        # Equal split
        share = settings.daily_study_hours / len(features)
        return {f.subject: round(share, 2) for f in features}

    allocations = {
        subject: round((p / total_priority) * settings.daily_study_hours, 2)
        for subject, p in priorities.items()
    }
    logger.info("Scheduler allocations: %s", allocations)
    return allocations
