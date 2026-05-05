"""
scikit-learn ML pipeline: StandardScaler + Ridge regression.

Predicts next-session score given subject features.
"""

from __future__ import annotations

import os
from typing import Sequence

import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from core.config import get_settings
from core.logging import get_logger
from ml.feature_engineering import SubjectFeatures

logger = get_logger(__name__)
settings = get_settings()


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=1.0)),
        ]
    )


def train(
    features: Sequence[SubjectFeatures],
    targets: Sequence[float],
) -> Pipeline:
    """Fit pipeline on (features, targets) and persist to disk."""
    if len(features) < 2:
        raise ValueError("Need at least 2 samples to train.")

    X = np.array([f.raw_features for f in features])
    y = np.asarray(targets, dtype=float)

    pipeline = build_pipeline()
    pipeline.fit(X, y)
    logger.info("ML pipeline trained on %d samples.", len(features))

    os.makedirs(os.path.dirname(settings.ml_model_path) or ".", exist_ok=True)
    joblib.dump(pipeline, settings.ml_model_path)
    logger.info("Pipeline saved to %s", settings.ml_model_path)
    return pipeline


def load_pipeline() -> Pipeline | None:
    path = settings.ml_model_path
    if not os.path.exists(path):
        logger.warning("No trained pipeline found at %s.", path)
        return None
    pipeline: Pipeline = joblib.load(path)
    logger.info("Pipeline loaded from %s.", path)
    return pipeline


def predict(pipeline: Pipeline, features: SubjectFeatures) -> float:
    X = np.array([features.raw_features])
    return float(pipeline.predict(X)[0])
