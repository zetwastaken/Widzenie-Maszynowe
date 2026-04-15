# best_frame

Project skeleton for automatic best-frame extraction from short videos.

## Goal

The pipeline analyzes decoded frames and scores each frame by:
- sharpness
- exposure quality
- face and eye-open signal

Then it exports:
- `best_frame.png`
- `ranking.json`

## Quick Start

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py --input video.mp4 --output out/
```

Optional sampling:

```bash
python main.py --input video.mp4 --output out/ --step 3
```

## Structure

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

## Collaboration Rules

- `main.py` is the only orchestrator that imports all layers.
- Feature modules do not import each other.
- Thresholds and weights live inside metric/scorer classes.
- Shared runtime payload is a simple dict built in `main.py`.

## Main Classes

- `Decoder`: video loading and frame sampling
- `SharpnessMetric`: `score_frames(frames) -> list[float]` in `0..100`
- `ExposureMetric`: `score_frames(frames) -> list[float]` in `0..100`
- `FaceMetric`: `score_frames(frames) -> list[float]` in `0..100`
- `Scorer`: weighted final score (0-100)
- `Reporter`: PNG + JSON output

## Manual Validation (No Automated Tests Yet)

1. `python main.py --input video.mp4 --output out/` finishes with no error.
2. `out/best_frame.png` and `out/ranking.json` are generated.
3. For shaky video, selected frame should be visibly sharper than random early frames.
4. For blink sequences, selected frame should prefer open eyes.
5. For mixed lighting, selected frame should avoid severe over/under exposure.
