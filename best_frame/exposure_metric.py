from __future__ import annotations

import cv2
import numpy as np

from base_metric import BaseMetric


class ExposureMetric(BaseMetric):
    """Basic exposure statistics from grayscale histogram."""

    OVEREXPOSE_THRESHOLD = 0.15
    UNDEREXPOSE_THRESHOLD = 0.15
    OVEREXPOSE_PIXEL_VALUE = 245
    UNDEREXPOSE_PIXEL_VALUE = 10
    OPTIMAL_BRIGHTNESS_MIN = 100.0
    OPTIMAL_BRIGHTNESS_MAX = 150.0
    RATIO_PENALTY_SCALE = 300.0
    BRIGHTNESS_DISTANCE_SCALE = 0.50

    def _score_frame(self, frame: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean = float(gray.mean())
        overexposed = float(np.mean(gray > self.OVEREXPOSE_PIXEL_VALUE))
        underexposed = float(np.mean(gray < self.UNDEREXPOSE_PIXEL_VALUE))

        over_delta = max(0.0, overexposed - self.OVEREXPOSE_THRESHOLD)
        under_delta = max(0.0, underexposed - self.UNDEREXPOSE_THRESHOLD)
        over_penalty = over_delta * self.RATIO_PENALTY_SCALE
        under_penalty = under_delta * self.RATIO_PENALTY_SCALE

        brightness_penalty = 0.0
        if mean < self.OPTIMAL_BRIGHTNESS_MIN:
            brightness_penalty = (
                (self.OPTIMAL_BRIGHTNESS_MIN - mean) * self.BRIGHTNESS_DISTANCE_SCALE
            )
        elif mean > self.OPTIMAL_BRIGHTNESS_MAX:
            brightness_penalty = (
                (mean - self.OPTIMAL_BRIGHTNESS_MAX) * self.BRIGHTNESS_DISTANCE_SCALE
            )

        return self.SCORE_MAX - over_penalty - under_penalty - brightness_penalty
