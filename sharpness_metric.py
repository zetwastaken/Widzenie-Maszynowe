from __future__ import annotations

import cv2
import numpy as np
from base_metric import BaseMetric


class SharpnessMetric(BaseMetric):
    """
    Laplacian variance sharpness, normalized relative to the batch.
    Sharpest frame in batch → 100, blurriest → 0.
    """

    MIN_VARIANCE_THRESHOLD = 5.0

    def _score_frame(self, frame: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def score_frames(self, frames) -> list[float]:
        validated = self._validate_frames(frames)
        raw = [self._score_frame(f) for f in validated]
        return self.normalize_batch(raw)

    @classmethod
    def normalize_batch(cls, raw_variances: list[float]) -> list[float]:
        if not raw_variances:
            return []
        min_v, max_v = min(raw_variances), max(raw_variances)
        if max_v - min_v < cls.MIN_VARIANCE_THRESHOLD:
            return [50.0] * len(raw_variances)
        return [
            cls._clamp_score((v - min_v) / (max_v - min_v) * cls.SCORE_MAX)
            for v in raw_variances
        ]
