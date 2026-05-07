import cv2
import numpy as np
from sharpness_metric import SharpnessMetric

def test_normalization():
    metric = SharpnessMetric()

    path = "test_images/tst2.jpg"
    original = cv2.imread(path)

    if original is None:
        print(f"Error: Didn't found {path}")
        return

    light = cv2.GaussianBlur(original, (11, 11), 0)
    mid = cv2.GaussianBlur(original, (31, 31), 0)
    strong = cv2.GaussianBlur(original, (51, 51), 0)
    vstrong = cv2.GaussianBlur(original, (101, 101), 0)

    frames = [original, light, mid, strong, vstrong]
    descriptions = [
        "original (sharp)",
        "light blur (11x11)",
        "medium blur (31x31)",
        "strong blur (51x51)",
        "very strong blur (101x101)"
    ]

    results = metric.score_frames(frames)

    print("Results after normalization (0-100):")
    for description, result in zip(descriptions, results):
        print(f"  {description:<35} -> {result:.1f}")

    assert results[0] == 100.0, f"Error: original should have 100, not {results[0]}"
    assert results[-1] == 0.0, f"Error: the most blurred should have 0, not {results[-1]}"

    for i in range(len(results) - 1):
        assert results[i] > results[i + 1], (
            f"Error: {descriptions[i]} ({results[i]:.1f}) should be > {descriptions[i+1]} ({results[i+1]:.1f}"
        )

    print("All asserts are good")


def test_identical_frames():
    metric = SharpnessMetric()

    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    frames = [frame, frame, frame]

    results = metric.score_frames(frames)
    print(f"\nTest identical frames: {results}")
    assert all(r == 50.0 for r in results), "Error: identical frames should get 50"
    print("Edge case works good")


if __name__ == "__main__":
    print("=" * 55)
    print("Test 1: Normalization relative to video")
    print("=" * 55)
    test_normalization()

    print()
    print("=" * 55)
    print("Test 2: Edge case - identical frames")
    print("=" * 55)
    test_identical_frames()
