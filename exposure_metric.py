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
        
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        
        total_pixels = gray.size

        overexposed_pixels = np.sum(hist[self.overexpose_pixel_value :])
        overexposed = float(overexposed_pixels / total_pixels)

        underexposed_pixels = np.sum(hist[: self.underexpose_pixel_value])
        underexposed = float(underexposed_pixels / total_pixels)

        mean = float(gray.mean())

        over_delta = max(0.0, overexposed - self.overexpose_threshold)
        under_delta = max(0.0, underexposed - self.underexpose_threshold)
        over_penalty = over_delta * self.ratio_penalty_scale
        under_penalty = under_delta * self.ratio_penalty_scale

        brightness_penalty = 0.0
        if mean < self.optimal_brightness_min:
            brightness_penalty = (
                (self.optimal_brightness_min - mean) * self.brightness_distance_scale
            )
        elif mean > self.optimal_brightness_max:
            brightness_penalty = (
                (mean - self.optimal_brightness_max) * self.brightness_distance_scale
            )

        return self.SCORE_MAX - over_penalty - under_penalty - brightness_penalty