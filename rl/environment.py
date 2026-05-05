"""
Custom Gym environment for study time allocation.

State  : flattened [avg_score, trend, efficiency] per subject  (3 * n_subjects)
Action : continuous allocation weights per subject             (n_subjects,)
Reward : weighted performance improvement minus fatigue penalty
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ml.feature_engineering import SubjectFeatures


class StudyEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, subject_features: list[SubjectFeatures], daily_hours: float = 8.0):
        super().__init__()
        self.subject_features = subject_features
        self.daily_hours = daily_hours
        self.n_subjects = len(subject_features)

        # Observation: 3 features per subject
        obs_dim = self.n_subjects * 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Action: positive weights that will be normalised to hours
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.n_subjects,), dtype=np.float32
        )

        self._state: np.ndarray = self._build_state()
        self._prev_scores: np.ndarray = np.array(
            [f.avg_score for f in subject_features], dtype=np.float32
        )

    # ------------------------------------------------------------------
    def _build_state(self) -> np.ndarray:
        rows = [
            [f.avg_score, f.trend, f.efficiency]
            for f in self.subject_features
        ]
        return np.array(rows, dtype=np.float32).flatten()

    def _normalise_action(self, action: np.ndarray) -> np.ndarray:
        action = np.clip(action, 1e-6, None)
        return (action / action.sum()) * self.daily_hours

    def _simulate_scores(self, hours_allocated: np.ndarray) -> np.ndarray:
        """
        Simplified score simulation:
        new_score = avg + trend + efficiency_boost from hours.
        """
        new_scores = []
        for i, feat in enumerate(self.subject_features):
            boost = min(feat.efficiency * hours_allocated[i], 10.0)
            new_score = np.clip(feat.avg_score + feat.trend + boost, 0.0, 100.0)
            new_scores.append(new_score)
        return np.array(new_scores, dtype=np.float32)

    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._state = self._build_state()
        self._prev_scores = np.array(
            [f.avg_score for f in self.subject_features], dtype=np.float32
        )
        return self._state, {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        hours_allocated = self._normalise_action(action)
        new_scores = self._simulate_scores(hours_allocated)

        improvement = float((new_scores - self._prev_scores).mean())
        fatigue_penalty = float(np.std(hours_allocated)) * 0.5
        reward = improvement - fatigue_penalty

        self._prev_scores = new_scores
        info = {
            "hours_allocated": hours_allocated.tolist(),
            "new_scores": new_scores.tolist(),
            "improvement": improvement,
            "fatigue_penalty": fatigue_penalty,
        }

        terminated = False  # episodic task managed externally
        truncated = False
        return self._state, reward, terminated, truncated, info

    def render(self) -> None:
        pass
