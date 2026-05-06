"""
PPO agent wrapper using stable-baselines3.

Handles training, persistence, and inference.
"""



import os
import numpy as np

from core.config import get_settings
from core.logging import get_logger
from ml.feature_engineering import SubjectFeatures
from rl.environment import StudyEnv

from __future__ import annotations

logger = get_logger(__name__)
settings = get_settings()


def _lazy_import_ppo():
    from stable_baselines3 import PPO  # noqa: PLC0415
    return PPO


def train_agent(
    subject_features: list[SubjectFeatures],
    total_timesteps: int = 10_000,
) -> object:
    PPO = _lazy_import_ppo()
    env = StudyEnv(subject_features, daily_hours=settings.daily_study_hours)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=0,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
    )
    model.learn(total_timesteps=total_timesteps)
    logger.info("PPO agent trained for %d timesteps.", total_timesteps)

    os.makedirs(os.path.dirname(settings.rl_model_path) or ".", exist_ok=True)
    model.save(settings.rl_model_path)
    logger.info("PPO model saved to %s.", settings.rl_model_path)
    return model


def load_agent(subject_features: list[SubjectFeatures]) -> object | None:
    PPO = _lazy_import_ppo()
    path = settings.rl_model_path + ".zip"
    if not os.path.exists(path):
        logger.warning("No PPO model found at %s.", path)
        return None
    env = StudyEnv(subject_features, daily_hours=settings.daily_study_hours)
    model = PPO.load(settings.rl_model_path, env=env)
    logger.info("PPO model loaded from %s.", settings.rl_model_path)
    return model


def get_allocation(
    model: object,
    subject_features: list[SubjectFeatures],
) -> dict[str, float]:
    """Return hour allocations per subject from a trained PPO model."""
    env = StudyEnv(subject_features, daily_hours=settings.daily_study_hours)
    obs, _ = env.reset()
    action, _ = model.predict(obs, deterministic=True)
    action = np.clip(action, 1e-6, None)
    normalized = (action / action.sum()) * settings.daily_study_hours
    return {
        feat.subject: round(float(h), 2)
        for feat, h in zip(subject_features, normalized)
    }
