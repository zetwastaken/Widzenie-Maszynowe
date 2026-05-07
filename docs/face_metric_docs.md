# Face Metric Module Documentation

This document explains the functionality and scoring logic of the `face_metric.py` module. The module is designed to evaluate images (frames) and assign a "quality" score based on the facial features and expressions of the people in the picture.

## Overview

The `FaceMetric` class evaluates the quality of a frame by analyzing human faces using **MediaPipe's FaceLandmarker** model. It returns a score from `0.0` to `100.0`. If multiple faces are detected, the overall score is the average of all individual face scores. If no faces are detected, the frame receives an automatic score of `0.0`.

## Scoring Logic

The final score is composed of three main positive evaluations and one penalty. The module locates specific 3D landmarks on the face to compute these metrics:

### 1. Eyes Openness (Weight: 40%)
- **Method:** Uses the **Eye Aspect Ratio (EAR)**, which measures the ratio of the eye's height to its width.
- **Logic:** Closed eyes yield a low EAR score, while wide-open eyes yield a high EAR score.
- **Thresholds:** `MIN_EAR = 0.15` (0 pts), `MAX_EAR = 0.33` (max pts).

### 2. Smile Detection (Weight: 40%)
- **Method:** Computes the ratio of the mouth's width (distance between left and right corners of the lips) to the total width of the face.
- **Logic:** A wider horizontal stretch of the mouth relative to the face size indicates a smile.
- **Thresholds:** `MIN_SMILE = 0.45` (0 pts), `MAX_SMILE = 0.65` (max pts).

### 3. Head Pose / Looking at Camera (Weight: 20%)
- **Method:** Compares the horizontal distance from the tip of the nose to the left edge of the face, versus the nose to the right edge of the face.
- **Logic:** If the subject is looking directly at the camera, the distances will be roughly equal (ratio near 0.5). If they turn their head, one side becomes much smaller than the other, resulting in a lower score.

### 4. Yawning Penalty (Dynamic Subtraction)
- **Method:** Uses the **Mouth Aspect Ratio (MAR)** on the inner lip points to determine how vertically wide the mouth is open.
- **Logic:** If the MAR exceeds the normal speaking threshold, the subject is likely yawning or screaming. This results in a direct penalty subtracted from the total score.
- **Thresholds:** Starts penalizing at `MIN_MAR = 0.4` up to a maximum penalty at `MAX_MAR = 0.8`.

### Final Calculation
```python
face_score = (eyes_score * 0.4) + (smile_score * 0.4) + (pose_score * 0.2)
face_score = max(0.0, face_score - yawn_penalty)
```

## Dependencies
- `OpenCV (cv2)`: Image manipulation and visualization.
- `NumPy`: Mathematical arrays and sequence handling.
- `MediaPipe`: For the FaceLandmarker ML model (`face_landmarker.task` file is required in the working directory).

## Usage

### Integration
To use this in your main application (`best_frame` logic), instantiate the metric and call the scoring methods provided by the `BaseMetric` parent class:

```python
from face_metric import FaceMetric

metric = FaceMetric(model_path='face_landmarker.task')
scores = metric.score_frames(list_of_frames_np_arrays)
```