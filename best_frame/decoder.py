from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class Decoder:
    """Simple video decoder returning sampled valid frames."""

    @staticmethod
    def _is_valid_frame(frame: np.ndarray | None) -> bool:
        if frame is None or frame.size == 0:
            return False
        return float(frame.std()) > 1e-6

    def decode(self, path: str, step: int = 1) -> list[tuple[int, np.ndarray]]:
        if step < 1:
            raise ValueError("step must be >= 1")

        video_path = Path(path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        frames: list[tuple[int, np.ndarray]] = []
        frame_index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_index % step == 0 and self._is_valid_frame(frame):
                    frames.append((frame_index, frame))
                frame_index += 1
        finally:
            capture.release()

        if not frames:
            raise ValueError("No valid frames were decoded from the input video.")
        return frames
