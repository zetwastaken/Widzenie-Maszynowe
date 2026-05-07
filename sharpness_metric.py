from __future__ import annotations

import cv2
import numpy as np

from base_metric import BaseMetric


class SharpnessMetric(BaseMetric):
    """Variance of Laplacian in grayscale."""

    REFERENCE_VARIANCE = 500.0

    def __init__(self, reference_variance: float = REFERENCE_VARIANCE) -> None:
        if reference_variance <= 0.0:
            raise ValueError("reference_variance must be > 0")
        self.reference_variance = reference_variance

    def _score_frame(self, frame: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        return (variance / self.reference_variance) * self.SCORE_MAX
