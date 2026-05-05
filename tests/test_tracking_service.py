"""Tests for the tracking service."""

from services.tracking_service import (
    create_log,
    get_all_logs,
    get_logs_by_subject,
    group_logs_by_subject,
)
from utils.models import StudyLog


def test_create_and_retrieve_log(db_session):
    entry = create_log(db_session, "Biology", 1.5, 72.0)
    assert entry.id is not None
    assert entry.subject == "Biology"
    assert entry.hours == 1.5
    assert entry.score == 72.0


def test_get_logs_by_subject(db_session):
    create_log(db_session, "Chemistry", 2.0, 65.0)
    create_log(db_session, "Chemistry", 1.5, 70.0)
    create_log(db_session, "Art", 1.0, 90.0)

    logs = get_logs_by_subject(db_session, "Chemistry")
    assert all(log.subject == "Chemistry" for log in logs)
    assert len(logs) >= 2


def test_group_logs_by_subject(db_session):
    logs = [
        StudyLog(subject="Math", hours=2.0, score=80.0),
        StudyLog(subject="Math", hours=1.5, score=85.0),
        StudyLog(subject="Science", hours=3.0, score=70.0),
    ]
    grouped = group_logs_by_subject(logs)
    assert "Math" in grouped
    assert "Science" in grouped
    assert len(grouped["Math"]["scores"]) == 2
    assert grouped["Science"]["hours"] == [3.0]


def test_get_all_logs_returns_list(db_session):
    create_log(db_session, "History", 1.0, 88.0)
    logs = get_all_logs(db_session)
    assert isinstance(logs, list)
    assert len(logs) >= 1
