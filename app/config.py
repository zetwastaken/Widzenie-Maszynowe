from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class AnalysisConfig:
    step: int = 1
    output_dir: str = "out"
    w_sharpness: float = 0.40
    w_exposure: float = 0.30
    w_face: float = 0.30
    reference_variance: float = 500.0
    ear_threshold: float = 0.20
    overexpose_threshold: float = 0.15
    underexpose_threshold: float = 0.15
    overexpose_pixel_value: int = 245
    underexpose_pixel_value: int = 10
    optimal_brightness_min: float = 100.0
    optimal_brightness_max: float = 150.0
    ratio_penalty_scale: float = 300.0
    brightness_distance_scale: float = 0.50

    def validate(self) -> None:
        if self.step < 1:
            raise ValueError("step must be >= 1")
        if self.reference_variance <= 0.0:
            raise ValueError("reference_variance must be > 0")
        if not (0.0 <= self.ear_threshold <= 1.0):
            raise ValueError("ear_threshold must be in range 0..1")
        if self.optimal_brightness_min >= self.optimal_brightness_max:
            raise ValueError("optimal brightness min must be lower than max")
