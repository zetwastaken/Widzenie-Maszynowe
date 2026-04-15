# best_frame

Automatic best-frame extraction from short videos.

## What It Does

The pipeline:
- reads video frames,
- computes three per-frame scores (`0..100`): sharpness, exposure, face/eyes,
- combines them into one final score,
- saves `best_frame.png` (highest final score),
- saves `ranking.json` (all analyzed frames, sorted by score).

## Quick Start

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py --input video.mp4 --output out/
```

Optional frame sampling:

```bash
python main.py --input video.mp4 --output out/ --step 3
```

Makefile shortcuts:

```bash
make install
make run INPUT=video.mp4 OUTPUT=out STEP=1
```

## Project Files

```text
best_frame/
  main.py
  decoder.py
  base_metric.py
  sharpness_metric.py
  exposure_metric.py
  face_metric.py
  scorer.py
  reporter.py
  docs/
```

## How It Works

Execution flow in `main.py`:
1. `Decoder.decode(...)` returns `[(frame_index, frame), ...]`.
2. `SharpnessMetric.score_frames(frames)` returns `[float, ...]`.
3. `ExposureMetric.score_frames(frames)` returns `[float, ...]`.
4. `FaceMetric.score_frames(frames)` returns `[float, ...]`.
5. `Scorer.combine_batch(...)` returns final scores.
6. `Reporter.write(...)` saves PNG + JSON.

Data contract:
- every metric gets the same input: `frames: list[np.ndarray]`,
- every metric returns the same output shape: `list[float]`,
- output order must match input order,
- all scores are clamped to `0..100`.

## How To Modify Modules

General rule:
- keep module responsibilities separate,
- do not add cross-imports between metric modules,
- keep public signatures stable.

`decoder.py`:
- change only frame loading/filtering/sampling logic,
- return type must stay `list[tuple[int, np.ndarray]]`.

`base_metric.py`:
- defines the shared metric interface and validation,
- if you change input/output contract here, you must adapt all metric modules and `main.py`.

`sharpness_metric.py`:
- edit sharpness logic in `_score_frame`,
- tune `REFERENCE_VARIANCE` to rescale sharpness score sensitivity.

`exposure_metric.py`:
- edit exposure penalties in `_score_frame`,
- tune thresholds/scales (`OVEREXPOSE_THRESHOLD`, `RATIO_PENALTY_SCALE`, etc.).

`face_metric.py`:
- edit face/eyes logic in `_score_frame`,
- tune `DEFAULT_EAR_THRESHOLD`, neutral score, and closed-eyes score.

`scorer.py`:
- change metric weights in `Scorer(...)` constructor defaults,
- keep `combine_batch` output length equal to input score lengths.

`reporter.py`:
- change output formats/filenames only,
- keep `write(scored_frames, output_dir)` as the integration entrypoint.

`main.py`:
- orchestration only,
- should stay the single place that builds the runtime `scored_frames` payload.

## Adding A New Metric Module

1. Create `<new_metric>.py` with class inheriting `BaseMetric`.
2. Implement `_score_frame(self, frame) -> float`.
3. Instantiate in `main.py`.
4. Call `score_frames(frames)`.
5. Add score into `scored_frames` dict and update `Scorer`/`Reporter` if needed.

## Manual Validation

1. Run `python main.py --input video.mp4 --output out/`.
2. Confirm files `out/best_frame.png` and `out/ranking.json` exist.
3. Confirm `ranking.json` contains all analyzed frames and scores.
