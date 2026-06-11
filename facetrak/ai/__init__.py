"""AI analysis modules for FaceTrak.

  - ObjectDetector  — YOLOv8 object detection (requires ultralytics)
  - PoseEstimator   — MediaPipe full-body pose + joint angles
  - GestureDetector — MediaPipe hand landmarks + gesture classification
"""
from .objects import ObjectDetector, Detection
from .pose import PoseEstimator, PoseResult
from .gestures import GestureDetector, HandResult, Gesture

__all__ = [
    "ObjectDetector", "Detection",
    "PoseEstimator", "PoseResult",
    "GestureDetector", "HandResult", "Gesture",
]
