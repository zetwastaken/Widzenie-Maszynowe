from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np


class BaseMetric(ABC):
    """Common contract: frames in, scores out (same order, same length)."""

    SCORE_MIN = 0.0
    SCORE_MAX = 100.0

    def score_frames(self, frames: Sequence[np.ndarray]) -> list[float]:
        validated = self._validate_frames(frames)
        return [self._clamp_score(self._score_frame(frame)) for frame in validated]

    @classmethod
    def _clamp_score(cls, value: float) -> float:
        return max(cls.SCORE_MIN, min(cls.SCORE_MAX, value))

    @staticmethod
    def _validate_frames(frames: Sequence[np.ndarray]) -> list[np.ndarray]:
        if isinstance(frames, np.ndarray):
            raise TypeError("frames must be a sequence of frame arrays, not a single array")
        if not isinstance(frames, Sequence):
            raise TypeError("frames must be a sequence of frame arrays")
        validated = list(frames)
        for index, frame in enumerate(validated):
            if not isinstance(frame, np.ndarray):
                raise TypeError(f"frames[{index}] must be numpy.ndarray")
            if frame.size == 0:
                raise ValueError(f"frames[{index}] cannot be empty")
        return validated

    @abstractmethod
    def _score_frame(self, frame: np.ndarray) -> float:
        raise NotImplementedError
