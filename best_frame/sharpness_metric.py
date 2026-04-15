from __future__ import annotations

import cv2
import numpy as np

from base_metric import BaseMetric


class SharpnessMetric(BaseMetric):
    """Variance of Laplacian in grayscale."""

    REFERENCE_VARIANCE = 500.0

    def _score_frame(self, frame: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        return (variance / self.REFERENCE_VARIANCE) * self.SCORE_MAX
