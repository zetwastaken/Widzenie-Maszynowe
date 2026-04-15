from __future__ import annotations

from math import hypot
from typing import Any, Sequence

import cv2
import numpy as np

from base_metric import BaseMetric

LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (362, 385, 387, 263, 373, 380)

try:
    import mediapipe as mp
except ImportError:  # pragma: no cover
    mp = None  # type: ignore[assignment]


class FaceMetric(BaseMetric):
    """Face + eye-open detector based on Face Mesh EAR."""

    DEFAULT_EAR_THRESHOLD = 0.20
    NEUTRAL_SCORE = 50.0
    CLOSED_EYES_SCORE = 0.0

    def __init__(self, ear_threshold: float = DEFAULT_EAR_THRESHOLD) -> None:
        self.ear_threshold = ear_threshold
        self._face_mesh = None

    def _get_mesh(self) -> Any | None:
        if mp is None:
            return None
        if self._face_mesh is None:
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=False,
            )
        return self._face_mesh

    @staticmethod
    def _distance(a: Any, b: Any) -> float:
        return hypot(a.x - b.x, a.y - b.y)

    def _ear(self, landmarks: Sequence[Any], indices: Sequence[int]) -> float:
        p1, p2, p3, p4, p5, p6 = (landmarks[i] for i in indices)
        a = self._distance(p2, p6)
        b = self._distance(p3, p5)
        c = self._distance(p1, p4)
        if c == 0.0:
            return 0.0
        return (a + b) / (2.0 * c)

    def _score_frame(self, frame: np.ndarray) -> float:
        mesh = self._get_mesh()
        if mesh is None:
            return self.NEUTRAL_SCORE

        result = mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not result.multi_face_landmarks:
            return self.NEUTRAL_SCORE

        landmarks = result.multi_face_landmarks[0].landmark
        left_ear = self._ear(landmarks, LEFT_EYE)
        right_ear = self._ear(landmarks, RIGHT_EYE)
        ear = float((left_ear + right_ear) / 2.0)
        if ear >= self.ear_threshold:
            return self.SCORE_MAX
        return self.CLOSED_EYES_SCORE
