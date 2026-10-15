"""MediaPipe Tasks-backend regression tests (pose/gesture/face).

Require a real mediapipe install (Tasks API, 0.10.x or >=1.x);
skipped on minimal images. Blank frames must never crash the pipeline.
"""

import numpy as np
import pytest

mp = pytest.importorskip("mediapipe")


def test_tasks_api_surface():
    from mediapipe.tasks.python import vision

    for name in ("PoseLandmarker", "HandLandmarker", "FaceLandmarker", "RunningMode"):
        assert hasattr(vision, name), name
    assert not hasattr(mp, "solutions"), (
        "legacy mp.solutions must stay unused (removed in mp>=1)"
    )


def test_pose_tasks_roundtrip():
    from facetrak.ai.pose import PoseEstimator

    pe = PoseEstimator(model_complexity=0)
    assert pe.load() is True and pe.enabled
    rgb = np.zeros((240, 320, 3), dtype=np.uint8)
    assert pe.process(rgb) is None  # blank: no pose, no crash
    bgr = np.zeros((240, 320, 3), dtype=np.uint8)
    pe.draw(
        bgr,
        pe.process(rgb)
        or __import__("facetrak.ai.pose", fromlist=["PoseResult"]).PoseResult({}, {}),
    )
    pe.release()
    assert not pe.enabled and pe.process(rgb) is None


def test_gesture_tasks_roundtrip():
    from facetrak.ai.gestures import GestureDetector

    gd = GestureDetector()
    assert gd.load() is True and gd.enabled
    rgb = np.zeros((240, 320, 3), dtype=np.uint8)
    assert gd.process(rgb) == []  # blank: no hands, no crash
    bgr = np.zeros((240, 320, 3), dtype=np.uint8)
    gd.draw(bgr, [])
    gd.release()
    assert not gd.enabled and gd.process(rgb) == []


def test_face_tasks_roundtrip():
    from facetrak.analysis import FaceAnalyzer

    fa = FaceAnalyzer()
    fa.load_model(320, 240)
    rgb = np.zeros((240, 320, 3), dtype=np.uint8)
    metrics = fa.analyze(rgb)
    assert metrics is not None and metrics.emotion
