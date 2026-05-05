"""
Orchestrates feature engineering → scheduler/RL → ML predictions
to produce a complete study plan.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from core.logging import get_logger
from ml.feature_engineering import SubjectFeatures, engineer_features
from ml.pipeline import load_pipeline, predict
from rl.agent import get_allocation, load_agent
from services.scheduler import allocate_hours
from services.tracking_service import get_all_logs, group_logs_by_subject

logger = get_logger(__name__)


def _build_features(db: Session) -> list[SubjectFeatures]:
    logs = get_all_logs(db)
    grouped = group_logs_by_subject(logs)
    return [
        engineer_features(subject, data["scores"], data["hours"])
        for subject, data in grouped.items()
    ]


def generate_plan(db: Session) -> dict:
    """
    Returns:
    {
      "allocations": {subject: hours, ...},
      "predicted_scores": {subject: score, ...},
      "features": {subject: {avg_score, trend, efficiency}, ...},
      "source": "rl" | "scheduler",
    }
    """
    features = _build_features(db)
    if not features:
        return {"error": "No study logs found. Please log some sessions first."}

    # -- Hour allocation: RL if model exists, else rule-based scheduler
    rl_model = load_agent(features)
    if rl_model is not None:
        allocations = get_allocation(rl_model, features)
        source = "rl"
        logger.info("Using RL agent for allocation.")
    else:
        allocations = allocate_hours(features)
        source = "scheduler"
        logger.info("Using rule-based scheduler for allocation.")

    # -- Score prediction via ML pipeline
    ml_pipeline = load_pipeline()
    predicted_scores: dict[str, float] = {}
    if ml_pipeline is not None:
        for feat in features:
            try:
                predicted_scores[feat.subject] = round(predict(ml_pipeline, feat), 2)
            except Exception as exc:
                logger.warning("ML prediction failed for %s: %s", feat.subject, exc)

    feature_summary = {
        f.subject: {
            "avg_score": round(f.avg_score, 2),
            "trend": round(f.trend, 4),
            "efficiency": round(f.efficiency, 4),
            "session_count": f.session_count,
        }
        for f in features
    }

    return {
        "allocations": allocations,
        "predicted_scores": predicted_scores,
        "features": feature_summary,
        "source": source,
    }
