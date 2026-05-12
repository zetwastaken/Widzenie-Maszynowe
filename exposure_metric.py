from __future__ import annotations

import cv2
import numpy as np

from base_metric import BaseMetric


class ExposureMetric(BaseMetric):
    """
    Exposure quality score: 100 = perfectly exposed, 0 = severely over/underexposed.

    Penalties:
      - Ratio of pixels above overexpose_pixel_value exceeding overexpose_threshold
      - Ratio of pixels below underexpose_pixel_value exceeding underexpose_threshold
      - Mean brightness outside [optimal_brightness_min, optimal_brightness_max]
    """

    OVEREXPOSE_THRESHOLD = 0.05
    UNDEREXPOSE_THRESHOLD = 0.05
    OVEREXPOSE_PIXEL_VALUE = 245
    UNDEREXPOSE_PIXEL_VALUE = 10
    OPTIMAL_BRIGHTNESS_MIN = 90.0
    OPTIMAL_BRIGHTNESS_MAX = 160.0
    RATIO_PENALTY_SCALE = 200.0
    BRIGHTNESS_DISTANCE_SCALE = 0.8

    def __init__(
        self,
        overexpose_threshold: float = OVEREXPOSE_THRESHOLD,
        underexpose_threshold: float = UNDEREXPOSE_THRESHOLD,
        overexpose_pixel_value: int = OVEREXPOSE_PIXEL_VALUE,
        underexpose_pixel_value: int = UNDEREXPOSE_PIXEL_VALUE,
        optimal_brightness_min: float = OPTIMAL_BRIGHTNESS_MIN,
        optimal_brightness_max: float = OPTIMAL_BRIGHTNESS_MAX,
        ratio_penalty_scale: float = RATIO_PENALTY_SCALE,
        brightness_distance_scale: float = BRIGHTNESS_DISTANCE_SCALE,
    ) -> None:
        self.overexpose_threshold = overexpose_threshold
        self.underexpose_threshold = underexpose_threshold
        self.overexpose_pixel_value = overexpose_pixel_value
        self.underexpose_pixel_value = underexpose_pixel_value
        self.optimal_brightness_min = optimal_brightness_min
        self.optimal_brightness_max = optimal_brightness_max
        self.ratio_penalty_scale = ratio_penalty_scale
        self.brightness_distance_scale = brightness_distance_scale

    def _score_frame(self, frame: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        total = gray.size
        mean = float(gray.mean())

        over_ratio = float(np.sum(gray >= self.overexpose_pixel_value)) / total
        under_ratio = float(np.sum(gray < self.underexpose_pixel_value)) / total

        over_penalty = max(0.0, over_ratio - self.overexpose_threshold) * self.ratio_penalty_scale
        under_penalty = max(0.0, under_ratio - self.underexpose_threshold) * self.ratio_penalty_scale

        if mean < self.optimal_brightness_min:
            brightness_penalty = (self.optimal_brightness_min - mean) * self.brightness_distance_scale
        elif mean > self.optimal_brightness_max:
            brightness_penalty = (mean - self.optimal_brightness_max) * self.brightness_distance_scale
        else:
            brightness_penalty = 0.0

        return self._clamp_score(self.SCORE_MAX - over_penalty - under_penalty - brightness_penalty)