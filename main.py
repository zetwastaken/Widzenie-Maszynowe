from __future__ import annotations

import argparse
import sys

from decoder import Decoder
from exposure_metric import ExposureMetric
from face_metric import FaceMetric
from reporter import Reporter
from scorer import Scorer
from sharpness_metric import SharpnessMetric


DEFAULT_FRAME_STEP = 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select the best frame from a video based on sharpness, exposure, and eyes-open signal."
    )
    parser.add_argument("--input", required=True, help="Path to input video file.")
    parser.add_argument("--output", required=True, help="Path to output directory.")
    parser.add_argument(
        "--step",
        type=int,
        default=DEFAULT_FRAME_STEP,
        help=f"Analyze every N-th frame (default: {DEFAULT_FRAME_STEP}).",
    )
    return parser


def run(input_path: str, output_dir: str, step: int) -> int:
    decoder = Decoder()
    sharpness_metric = SharpnessMetric()
    exposure_metric = ExposureMetric()
    face_metric = FaceMetric()
    scorer = Scorer()
    reporter = Reporter()

    frame_records = decoder.decode(path=input_path, step=step)
    frame_indexes = [item[0] for item in frame_records]
    frames = [item[1] for item in frame_records]

    sharpness_scores = sharpness_metric.score_frames(frames)
    exposure_scores = exposure_metric.score_frames(frames)
    face_scores = face_metric.score_frames(frames)
    final_scores = scorer.combine_batch(sharpness_scores, exposure_scores, face_scores)

    scored_frames: list[dict] = [
        {
            "frame_index": frame_index,
            "frame": frame,
            "sharpness_score": sharpness_score,
            "exposure_score": exposure_score,
            "face_score": face_score,
            "final_score": final_score,
        }
        for frame_index, frame, sharpness_score, exposure_score, face_score, final_score in zip(
            frame_indexes,
            frames,
            sharpness_scores,
            exposure_scores,
            face_scores,
            final_scores,
            strict=True,
        )
    ]

    if not scored_frames:
        raise ValueError("No frames were scored.")

    reporter.write(scored_frames=scored_frames, output_dir=output_dir)

    best = max(scored_frames, key=lambda item: item["final_score"])
    print(
        f"Saved output to '{output_dir}'. Best frame index={best['frame_index']}, score={best['final_score']:.2f}"
    )
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return run(input_path=args.input, output_dir=args.output, step=args.step)
    except Exception as error:  # pragma: no cover - CLI safety path
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
