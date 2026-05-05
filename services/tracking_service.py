"""
Service layer: persisting and querying study logs.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from sqlalchemy.orm import Session

from core.logging import get_logger
from utils.models import StudyLog

logger = get_logger(__name__)


def create_log(db: Session, subject: str, hours: float, score: float) -> StudyLog:
    entry = StudyLog(subject=subject, hours=hours, score=score)
    db.add(entry)
    db.commit()
    db.refresh(entry)
    logger.info("Logged: subject=%s hours=%.2f score=%.2f", subject, hours, score)
    return entry


def get_all_logs(db: Session) -> list[StudyLog]:
    return db.query(StudyLog).order_by(StudyLog.logged_at).all()


def get_logs_by_subject(db: Session, subject: str) -> list[StudyLog]:
    return (
        db.query(StudyLog)
        .filter(StudyLog.subject == subject)
        .order_by(StudyLog.logged_at)
        .all()
    )


def group_logs_by_subject(
    logs: Sequence[StudyLog],
) -> dict[str, dict[str, list[float]]]:
    """Group raw logs into {subject: {scores: [...], hours: [...]}}."""
    grouped: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"scores": [], "hours": []}
    )
    for log in logs:
        grouped[log.subject]["scores"].append(log.score)
        grouped[log.subject]["hours"].append(log.hours)
    return dict(grouped)
