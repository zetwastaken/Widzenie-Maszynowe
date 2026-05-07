from __future__ import annotations

import cv2
import numpy as np
from base_metric import BaseMetric


class SharpnessMetric(BaseMetric):
    """
    Variance of Laplacian in grayscale.
    Most sharpest frame -> 100
    Least sharpest frame -> 0
    """

    MIN_VARIANCE_THRESHOLD = 5.0

    def _score_frame(self, frame: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def score_frames(self, frames) -> list[float]:
        validated = self._validate_frames(frames)

        raw_variances = [self._score_frame(frame) for frame in validated]

        min_var = min(raw_variances)
        max_var = max(raw_variances)

        if max_var - min_var < self.MIN_VARIANCE_THRESHOLD:
            return [50.0] * len(raw_variances)

        return [
            self._clamp_score(
                ((v - min_var) / (max_var - min_var)) * self.SCORE_MAX
            )
            for v in raw_variances
        ]