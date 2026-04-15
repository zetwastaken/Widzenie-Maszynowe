from __future__ import annotations

from collections.abc import Sequence


class Scorer:
    """Weighted frame scorer returning 0-100 scores."""

    SCORE_MIN = 0.0
    SCORE_MAX = 100.0

    def __init__(self, w_sharpness: float = 0.40, w_exposure: float = 0.30, w_face: float = 0.30) -> None:
        self.w_sharpness = w_sharpness
        self.w_exposure = w_exposure
        self.w_face = w_face

    @classmethod
    def clamp_score(cls, value: float) -> float:
        return max(cls.SCORE_MIN, min(cls.SCORE_MAX, value))

    def combine(self, sharpness_score: float, exposure_score: float, face_score: float) -> float:
        final_score = (
            self.w_sharpness * sharpness_score
            + self.w_exposure * exposure_score
            + self.w_face * face_score
        )
        return self.clamp_score(final_score)

    def combine_batch(
        self,
        sharpness_scores: Sequence[float],
        exposure_scores: Sequence[float],
        face_scores: Sequence[float],
    ) -> list[float]:
        if not (len(sharpness_scores) == len(exposure_scores) == len(face_scores)):
            raise ValueError("All score lists must have the same length.")
        return [
            self.combine(s, e, f)
            for s, e, f in zip(sharpness_scores, exposure_scores, face_scores, strict=True)
        ]
