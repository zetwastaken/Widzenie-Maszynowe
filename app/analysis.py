from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from decoder import Decoder
from exposure_metric import ExposureMetric
from face_metric import FaceMetric
from reporter import Reporter
from scorer import Scorer
from sharpness_metric import SharpnessMetric

from .config import AnalysisConfig


@dataclass(slots=True)
class FrameResult:
    rank: int
    frame_index: int
    frame: np.ndarray
    sharpness: float
    exposure: float
    face: float
    final: float


@dataclass(slots=True)
class AnalysisResult:
    frames: list[FrameResult]
    best_frame: FrameResult
    diagnostics: list[str]
    output_dir: str


ProgressCallback = Callable[[int, str], None]
CancelCheck = Callable[[], bool]


def _to_scored_dicts(frame_results: list[FrameResult]) -> list[dict]:
    return [
        {
            "frame_index": item.frame_index,
            "frame": item.frame,
            "sharpness_score": item.sharpness,
            "exposure_score": item.exposure,
            "face_score": item.face,
            "final_score": item.final,
        }
        for item in frame_results
    ]


def analyze_video(
    input_path: str,
    config: AnalysisConfig,
    progress: ProgressCallback | None = None,
    is_cancelled: CancelCheck | None = None,
) -> AnalysisResult:
    config.validate()
    progress = progress or (lambda _percent, _status: None)
    is_cancelled = is_cancelled or (lambda: False)

    if is_cancelled():
        raise RuntimeError("Analysis cancelled")
    progress(5, "Opening video")

    decoder = Decoder()
    total = Decoder.count_frames(input_path, config.step)

    sharpness_metric = SharpnessMetric()
    exposure_metric = ExposureMetric()
    face_metric = FaceMetric()
    scorer = Scorer()

    _THUMB_WIDTH = 360

    scored: list[FrameResult] = []
    best_score = -1.0
    best_full_frame: np.ndarray | None = None
    best_frame_index_val = -1

    for idx, (frame_index, frame) in enumerate(decoder.stream(input_path, step=config.step)):
        if is_cancelled():
            raise RuntimeError("Analysis cancelled")
        sharpness = sharpness_metric.score_frames([frame])[0]
        exposure = exposure_metric.score_frames([frame])[0]
        face = face_metric.score_frames([frame])[0]
        final = scorer.combine(sharpness, exposure, face)

        if final >= best_score:
            best_score = final
            best_full_frame = frame
            best_frame_index_val = frame_index

        h, w = frame.shape[:2]
        ratio = _THUMB_WIDTH / max(1, w)
        thumb = cv2.resize(frame, (_THUMB_WIDTH, max(1, int(h * ratio))), interpolation=cv2.INTER_AREA)

        scored.append(
            FrameResult(
                rank=0,
                frame_index=frame_index,
                frame=thumb,
                sharpness=sharpness,
                exposure=exposure,
                face=face,
                final=final,
            )
        )
        # Keep some headroom for write/export phase.
        progress(5 + int(((idx + 1) / total) * 85), f"Scoring frame {idx + 1}/{total}")

    if not scored:
        raise ValueError("No frames were decoded.")

    # Replace the best frame's thumbnail with the full-resolution original so
    # the reporter can write it at full quality and the UI can export it.
    for item in scored:
        if item.frame_index == best_frame_index_val:
            item.frame = best_full_frame
            break

    sorted_frames = sorted(scored, key=lambda item: item.final, reverse=True)
    for rank, item in enumerate(sorted_frames, start=1):
        item.rank = rank

    diagnostics: list[str] = []
    face_values = [item.face for item in sorted_frames]
    if all(value == 50.0 for value in face_values):
        diagnostics.append("Face metric returned neutral score (50) for all frames.")
    elif all(value == 0.0 for value in face_values):
        diagnostics.append("Face metric marked all frames as closed eyes (0).")
    elif all(value == 100.0 for value in face_values):
        diagnostics.append("Face metric marked all frames as open eyes (100).")

    progress(92, "Writing output files")
    out_dir = Path(config.output_dir)
    reporter = Reporter()
    reporter.write(scored_frames=_to_scored_dicts(scored), output_dir=str(out_dir))

    progress(100, "Done")
    return AnalysisResult(
        frames=sorted_frames,
        best_frame=sorted_frames[0],
        diagnostics=diagnostics,
        output_dir=str(out_dir),
    )


def encode_thumbnail(frame: np.ndarray, width: int = 180) -> np.ndarray:
    height, original_width = frame.shape[:2]
    ratio = width / max(1, original_width)
    resized = cv2.resize(frame, (width, max(1, int(height * ratio))), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
