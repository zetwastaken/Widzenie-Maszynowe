from __future__ import annotations

from math import hypot
from pathlib import Path

import cv2
import numpy as np

from base_metric import BaseMetric
from runtime_paths import asset_path

LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
MOUTH_OUTER_INDICES = [61, 291, 39, 181, 0, 17, 269, 405]
MOUTH_INNER_INDICES = [78, 308, 95, 88, 191, 14, 324, 318]
NOSE_TIP_INDEX = 1
CHIN_INDEX = 152
LEFT_FACE_EDGE_INDEX = 234
RIGHT_FACE_EDGE_INDEX = 454

try:
    import mediapipe as mp
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
except ImportError:
    mp = None


class FaceMetric(BaseMetric):
    """Face + eye-open detector based on MediaPipe Tasks FaceLandmarker."""

    DEFAULT_EAR_THRESHOLD = 0.15
    MAX_EAR = 0.33
    
    # Yawn thresholds (Mouth Aspect Ratio)
    MIN_MAR = 0.4
    MAX_MAR = 0.8
    
    # Smile ratio (mouth width to face width)
    MIN_SMILE = 0.45
    MAX_SMILE = 0.65

    def __init__(
        self,
        model_path: str | None = None,
        num_faces: int = 20,
        ear_threshold: float = DEFAULT_EAR_THRESHOLD,
    ) -> None:
        self.model_path = str(asset_path("face_landmarker.task") if model_path is None else Path(model_path))
        self.num_faces = num_faces
        self.ear_threshold = ear_threshold
        self._detector = None

    def _get_detector(self):
        if mp is None:
            return None
        if self._detector is None:
            options = FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=self.model_path),
                running_mode=VisionRunningMode.IMAGE,
                num_faces=self.num_faces,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5
            )
            self._detector = FaceLandmarker.create_from_options(options)
        return self._detector

    @staticmethod
    def _distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
        return hypot(p1[0] - p2[0], p1[1] - p2[1])

    def _calculate_ear(self, eye_points: list[tuple[float, float]]) -> float:
        width = self._distance(eye_points[0], eye_points[3])
        height1 = self._distance(eye_points[1], eye_points[5])
        height2 = self._distance(eye_points[2], eye_points[4])
        if width == 0:
            return 0.0
        return (height1 + height2) / (2.0 * width)

    def _calculate_mar(self, mouth_points: list[tuple[float, float]]) -> float:
        # Distance between outer corners
        width = self._distance(mouth_points[0], mouth_points[1])
        # Vertical distances between upper and lower lips
        height1 = self._distance(mouth_points[2], mouth_points[3])
        height2 = self._distance(mouth_points[4], mouth_points[5])
        height3 = self._distance(mouth_points[6], mouth_points[7])
        
        if width == 0:
            return 0.0
        return (height1 + height2 + height3) / (3.0 * width)
        
    def _calculate_smile_ratio(self, mouth_corner_left: tuple[float, float], mouth_corner_right: tuple[float, float], face_left: tuple[float, float], face_right: tuple[float, float]) -> float:
        mouth_width = self._distance(mouth_corner_left, mouth_corner_right)
        face_width = self._distance(face_left, face_right)
        if face_width == 0:
            return 0.0
        return mouth_width / face_width

    def _calculate_head_pose_score(self, nose: tuple[float, float], left_edge: tuple[float, float], right_edge: tuple[float, float]) -> float:
        dist_left = self._distance(nose, left_edge)
        dist_right = self._distance(nose, right_edge)
        if dist_left + dist_right == 0:
            return 0.0
        # Ratio of distances. 0.5 means perfectly looking straight
        ratio = min(dist_left, dist_right) / (dist_left + dist_right)
        # Scale to 0-100 where 0.5 -> 100, and 0.0 -> 0
        return ratio * 200.0

    def _score_frame(self, frame: np.ndarray) -> float:
        detector = self._get_detector()
        if detector is None:
            return BaseMetric.SCORE_MIN

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        results = detector.detect(mp_image)
        
        if not results.face_landmarks:
            return BaseMetric.SCORE_MIN

        h, w = frame.shape[:2]
        scores =[]
        
        for face_landmarks in results.face_landmarks:
            def get_pt(idx):
                return (face_landmarks[idx].x * w, face_landmarks[idx].y * h)
                
            left_eye_pts = [get_pt(i) for i in LEFT_EYE_INDICES]
            right_eye_pts = [get_pt(i) for i in RIGHT_EYE_INDICES]
            mouth_inner_pts = [get_pt(i) for i in MOUTH_INNER_INDICES]
            mouth_outer_pts = [get_pt(i) for i in MOUTH_OUTER_INDICES]
            
            nose_pt = get_pt(NOSE_TIP_INDEX)
            left_face_pt = get_pt(LEFT_FACE_EDGE_INDEX)
            right_face_pt = get_pt(RIGHT_FACE_EDGE_INDEX)
            mouth_corner_left = get_pt(61)
            mouth_corner_right = get_pt(291)
            
            # --- 1. Eyes Openness ---
            left_ear = self._calculate_ear(left_eye_pts)
            right_ear = self._calculate_ear(right_eye_pts)
            avg_ear = (left_ear + right_ear) / 2.0
            ear_range = self.MAX_EAR - self.ear_threshold
            if ear_range <= 0:
                eyes_score = 100.0 if avg_ear >= self.MAX_EAR else 0.0
            else:
                clipped_ear = max(self.ear_threshold, min(avg_ear, self.MAX_EAR))
                eyes_score = ((clipped_ear - self.ear_threshold) / ear_range) * 100.0
            
            # --- 2. Head Pose (looking at camera) ---
            pose_score = self._calculate_head_pose_score(nose_pt, left_face_pt, right_face_pt)
            
            # --- 3. Smile ---
            smile_ratio = self._calculate_smile_ratio(mouth_corner_left, mouth_corner_right, left_face_pt, right_face_pt)
            clipped_smile = max(self.MIN_SMILE, min(smile_ratio, self.MAX_SMILE))
            smile_score = ((clipped_smile - self.MIN_SMILE) / (self.MAX_SMILE - self.MIN_SMILE)) * 100.0
            
            # --- 4. Yawning Penalty ---
            mar = self._calculate_mar(mouth_inner_pts)
            yawn_penalty = 0.0
            if mar > self.MIN_MAR:
                yawn_penalty = min(100.0, ((mar - self.MIN_MAR) / (self.MAX_MAR - self.MIN_MAR)) * 100.0)
            
            # --- Weighted Final Score ---
            # Weights: Eyes (40%), Smile (40%), Pose (20%)
            face_score = (eyes_score * 0.4) + (smile_score * 0.4) + (pose_score * 0.2)
            
            # Apply yawn penalty (drastically reduces score if mouth is wide open)
            face_score = max(0.0, face_score - yawn_penalty)
            
            scores.append(face_score)
            
        return self._clamp_score(float(sum(scores) / len(scores)))
