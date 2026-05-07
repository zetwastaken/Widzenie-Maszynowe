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

    @staticmethod
    def _get_orientation_rotation(capture: cv2.VideoCapture) -> int:
        """Read orientation metadata (degrees), normalized to 0/90/180/270."""
        if not hasattr(cv2, "CAP_PROP_ORIENTATION_META"):
            return 0
        angle = int(round(capture.get(cv2.CAP_PROP_ORIENTATION_META))) % 360
        if angle in (90, 180, 270):
            return angle
        return 0

    @staticmethod
    def _apply_rotation(frame: np.ndarray, angle: int) -> np.ndarray:
        if angle == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if angle == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        if angle == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    def decode(self, path: str, step: int = 1) -> list[tuple[int, np.ndarray]]:
        if step < 1:
            raise ValueError("step must be >= 1")

        video_path = Path(path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        # Ask backend to apply orientation metadata automatically when available.
        auto_orientation_enabled = False
        if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
            capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
            auto_orientation_enabled = bool(int(capture.get(cv2.CAP_PROP_ORIENTATION_AUTO)))
        # Manual rotation is only a fallback when backend auto-orientation is unavailable.
        orientation_angle = 0 if auto_orientation_enabled else self._get_orientation_rotation(capture)

        frames: list[tuple[int, np.ndarray]] = []
        frame_index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                # Fallback when backend does not apply orientation metadata itself.
                if orientation_angle:
                    frame = self._apply_rotation(frame, orientation_angle)
                if frame_index % step == 0 and self._is_valid_frame(frame):
                    frames.append((frame_index, frame))
                frame_index += 1
        finally:
            capture.release()

        if not frames:
            raise ValueError("No valid frames were decoded from the input video.")
        return frames
