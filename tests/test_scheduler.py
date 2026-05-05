"""Tests for the rule-based scheduler."""

import pytest

from ml.feature_engineering import SubjectFeatures
from services.scheduler import allocate_hours, compute_priorities


def _make_feat(subject, avg_score, trend, efficiency, sessions=3):
    f = SubjectFeatures(
        subject=subject,
        avg_score=avg_score,
        trend=trend,
        efficiency=efficiency,
        session_count=sessions,
    )
    return f


def test_allocate_hours_sums_to_daily_total(monkeypatch):
    from core import config as cfg
    settings = cfg.get_settings()

    features = [
        _make_feat("Math", 70.0, -1.0, 5.0),
        _make_feat("Physics", 50.0, -2.0, 3.0),
        _make_feat("History", 90.0, 0.5, 8.0),
    ]
    allocs = allocate_hours(features)
    total = sum(allocs.values())
    assert abs(total - settings.daily_study_hours) < 0.05


def test_weak_subject_gets_more_hours():
    features = [
        _make_feat("Weak", 30.0, -3.0, 1.0),
        _make_feat("Strong", 95.0, 2.0, 12.0),
    ]
    allocs = allocate_hours(features)
    assert allocs["Weak"] > allocs["Strong"]


def test_empty_features_returns_empty():
    allocs = allocate_hours([])
    assert allocs == {}


def test_priorities_positive():
    features = [_make_feat("Bio", 60.0, -1.0, 4.0)]
    prio = compute_priorities(features)
    assert prio["Bio"] >= 0.0


def test_equal_subjects_roughly_equal_allocation():
    features = [
        _make_feat("A", 70.0, 0.0, 5.0),
        _make_feat("B", 70.0, 0.0, 5.0),
    ]
    allocs = allocate_hours(features)
    assert abs(allocs["A"] - allocs["B"]) < 0.1
