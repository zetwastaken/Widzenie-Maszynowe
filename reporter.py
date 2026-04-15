from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import cv2


class Reporter:
    """Writes best frame image and full ranking JSON."""

    BEST_FRAME_FILENAME = "best_frame.png"
    RANKING_FILENAME = "ranking.json"

    def _write_best_frame(self, scored_frames: list[dict], output_dir: Path) -> Path:
        best = max(scored_frames, key=lambda item: item["final_score"])
        output_path = output_dir / self.BEST_FRAME_FILENAME
        ok = cv2.imwrite(str(output_path), best["frame"])
        if not ok:
            raise ValueError(f"Unable to write image file: {output_path}")
        return output_path

    def _write_ranking_json(self, scored_frames: list[dict], output_dir: Path) -> Path:
        output_path = output_dir / self.RANKING_FILENAME
        sorted_frames = sorted(scored_frames, key=lambda item: item["final_score"], reverse=True)

        ranking = []
        for rank, item in enumerate(sorted_frames, start=1):
            ranking.append(
                {
                    "rank": rank,
                    "frame_index": item["frame_index"],
                    "score": float(item["final_score"]),
                    "scores": {
                        "sharpness": float(item["sharpness_score"]),
                        "exposure": float(item["exposure_score"]),
                        "face": float(item["face_score"]),
                    },
                }
            )

        document = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_frames": len(sorted_frames),
            "ranking": ranking,
        }

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(document, file, indent=2)
        return output_path

    def write(self, scored_frames: list[dict], output_dir: str) -> None:
        if not scored_frames:
            raise ValueError("scored_frames cannot be empty")

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self._write_best_frame(scored_frames, out_dir)
        self._write_ranking_json(scored_frames, out_dir)
