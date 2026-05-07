import cv2
import numpy as np
from face_metric import FaceMetric

def test_faces():
    # Initialize the metric with the path to the model
    # Note: make sure 'face_landmarker.task' exists in the root directory!
    metric = FaceMetric(model_path='face_landmarker.task')

    path1 = "test_images/1.jpg"
    path2 = "test_images/2.jpg"
    
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)

    if img1 is None or img2 is None:
        print(f"Error: Make sure {path1} and {path2} exist.")
        return

    frames = [img1, img2]
    descriptions = [
        "Test Image 1",
        "Test Image 2",
    ]

    print("Evaluating face metric for frames...")
    # Note: ensure mediapipe is installed (`pip install mediapipe`)
    results = metric.score_frames(frames)

    print("\nResults after scoring (0-100):")
    for description, result in zip(descriptions, results):
        print(f"  {description:<35} -> {result:.1f}")

if __name__ == "__main__":
    print("=" * 55)
    print("Test: FaceMetric evaluation")
    print("=" * 55)
    test_faces()
