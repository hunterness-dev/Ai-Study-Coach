"""Tests for FastAPI endpoints."""

import pytest


def test_health(api_client):
    resp = api_client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "service" in body


def test_log_session_success(api_client):
    resp = api_client.post(
        "/log",
        json={"subject": "mathematics", "hours": 2.0, "score": 80.0},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["subject"] == "Mathematics"
    assert body["hours"] == 2.0
    assert body["score"] == 80.0
    assert "id" in body


def test_log_session_invalid_score(api_client):
    resp = api_client.post(
        "/log",
        json={"subject": "Physics", "hours": 1.0, "score": 150.0},
    )
    assert resp.status_code == 422


def test_log_session_zero_hours(api_client):
    resp = api_client.post(
        "/log",
        json={"subject": "Chemistry", "hours": 0.0, "score": 70.0},
    )
    assert resp.status_code == 422


def test_get_plan_no_data(api_client):
    # Fresh client with no logs → 422
    from fastapi.testclient import TestClient
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from main import app
    from utils.database import Base, get_db

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    def override():
        yield session

    app.dependency_overrides[get_db] = override
    with TestClient(app) as c:
        resp = c.get("/plan")
    app.dependency_overrides.clear()
    assert resp.status_code == 422


def test_get_plan_with_data(api_client):
    # Seed two subjects with multiple logs
    subjects = [
        ("Mathematics", 2.0, 75.0),
        ("Mathematics", 2.5, 80.0),
        ("Physics", 1.5, 60.0),
        ("Physics", 2.0, 65.0),
    ]
    for subj, hrs, sc in subjects:
        api_client.post("/log", json={"subject": subj, "hours": hrs, "score": sc})

    resp = api_client.get("/plan")
    assert resp.status_code == 200
    body = resp.json()
    assert "allocations" in body
    assert "Mathematics" in body["allocations"]
    assert "Physics" in body["allocations"]
    assert body["source"] in ("rl", "scheduler")
