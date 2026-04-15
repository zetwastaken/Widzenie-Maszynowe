# Pipeline Data Flow

## Decoder -> Main

`decoder.Decoder.decode(path: str, step: int) -> list[tuple[int, np.ndarray]]`

Tuple values:
- first element: `frame_index: int`
- second element: `frame: np.ndarray` (BGR, uint8)

## Main -> Metrics

Each metric class receives the same input and returns the same output shape:
- input: `frames: list[np.ndarray]`
- output: `scores: list[float]` in `0..100`
- guaranteed order: output order matches input order

Shared method name:
- `base_metric.BaseMetric.score_frames(frames) -> list[float]`
- implemented by:
  - `sharpness_metric.SharpnessMetric`
  - `exposure_metric.ExposureMetric`
  - `face_metric.FaceMetric`

## Main -> Scoring

`scorer.Scorer.combine_batch(sharpness_scores, exposure_scores, face_scores) -> list[float]`

## Scoring -> Reporting

`reporter.Reporter.write(scored_frames: list[dict], output_dir: str) -> None`

Each `scored_frames` item contains:
- `frame_index: int`
- `frame: np.ndarray`
- `sharpness_score: float` (`0..100`)
- `exposure_score: float` (`0..100`)
- `face_score: float` (`0..100`)
- `final_score: float` (`0..100`)

Output files:
- `best_frame.png`
- `ranking.json`

## Consistency Rules

1. Metric modules must always return score values in `0..100`.
2. `main.py` is the only place that builds frame result dicts.
3. Any key changes in the frame result dict require updating this document.
