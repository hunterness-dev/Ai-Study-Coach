"""
Standalone script to train (or retrain) the PPO agent.

Usage:
    python scripts/train_rl.py [--timesteps 50000]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.logging import configure_logging
from ml.feature_engineering import engineer_features
from rl.agent import train_agent
from services.tracking_service import get_all_logs, group_logs_by_subject
from utils.database import SessionLocal, init_db

configure_logging()


def main(timesteps: int) -> None:
    init_db()
    db = SessionLocal()
    try:
        logs = get_all_logs(db)
        grouped = group_logs_by_subject(logs)
        features = [
            engineer_features(subj, data["scores"], data["hours"])
            for subj, data in grouped.items()
        ]
        if not features:
            print("No study logs in DB. Please log some sessions first.")
            return
        train_agent(features, total_timesteps=timesteps)
        print(f"PPO agent trained for {timesteps} timesteps.")
    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO study agent")
    parser.add_argument("--timesteps", type=int, default=20_000)
    args = parser.parse_args()
    main(args.timesteps)
