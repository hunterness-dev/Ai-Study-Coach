"""
Standalone script to train (or retrain) the Ridge ML pipeline.

Usage:
    python scripts/train_ml.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.logging import configure_logging
from utils.database import init_db, SessionLocal
from services.tracking_service import get_all_logs, group_logs_by_subject
from ml.feature_engineering import engineer_features
from ml.pipeline import train

configure_logging()


def main() -> None:
    init_db()
    db = SessionLocal()
    try:
        logs = get_all_logs(db)
        grouped = group_logs_by_subject(logs)

        features, targets = [], []
        for subj, data in grouped.items():
            scores = data["scores"]
            hours = data["hours"]
            if len(scores) < 2:
                continue
            # Target: last score; features from all-but-last sessions
            feat = engineer_features(subj, scores[:-1], hours[:-1])
            features.append(feat)
            targets.append(scores[-1])

        if len(features) < 2:
            print("Not enough data (need ≥2 subjects with ≥2 sessions each).")
            return

        train(features, targets)
        print("ML pipeline trained and saved.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
