"""Full-body pose estimation via MediaPipe PoseLandmarker (Tasks API).

Extracts 33 body landmarks, computes key joint angles (elbow, knee, shoulder),
and draws an annotated skeleton overlay.

Works with mediapipe 0.10.x and >=1.x (Tasks API, IMAGE mode). The legacy
mp.solutions API was removed in MediaPipe 1.x and is not used here.
"""

from __future__ import annotations

import logging
import math
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# MediaPipe Pose landmark indices
_LM = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}

_SKELETON_CONNECTIONS = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]

_ANGLE_JOINTS = {
    "left_elbow": ("left_shoulder", "left_elbow", "left_wrist"),
    "right_elbow": ("right_shoulder", "right_elbow", "right_wrist"),
    "left_knee": ("left_hip", "left_knee", "left_ankle"),
    "right_knee": ("right_hip", "right_knee", "right_ankle"),
}

_COLOR_SKELETON = (0, 255, 128)
_COLOR_JOINT = (255, 255, 255)
_MIN_VISIBILITY = 0.5

_MODEL_FILES = {
    # legacy model_complexity → Tasks bundle (lite is Pi-friendly)
    0: "pose_landmarker_lite.task",
    1: "pose_landmarker_full.task",
    2: "pose_landmarker_heavy.task",
}
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_{variant}/float16/latest/"
    "pose_landmarker_{variant}.task"
)
_MODEL_VARIANT = {0: "lite", 1: "full", 2: "heavy"}


@dataclass
class PoseResult:
    """Pose data for one person."""

    landmarks: dict[str, tuple[float, float, float]]  # name → (x, y, z) normalised
    angles: dict[str, float]  # joint name → degrees
    visible: bool = True

    def pixel_point(self, name: str, w: int, h: int) -> tuple[int, int] | None:
        lm = self.landmarks.get(name)
        if lm is None:
            return None
        return int(lm[0] * w), int(lm[1] * h)


def _angle_between(a: tuple, b: tuple, c: tuple) -> float:
    """Angle at joint b formed by vectors b→a and b→c, in degrees."""
    ax, ay = a[0] - b[0], a[1] - b[1]
    cx, cy = c[0] - b[0], c[1] - b[1]
    dot = ax * cx + ay * cy
    mag = math.hypot(ax, ay) * math.hypot(cx, cy)
    if mag == 0:
        return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / mag))))


class PoseEstimator:
    """Wraps MediaPipe PoseLandmarker for per-frame body landmark extraction.

    Usage:
        estimator = PoseEstimator()
        estimator.load()
        result = estimator.process(rgb_frame)
        if result:
            estimator.draw(bgr_frame, result)
    """

    def __init__(self, model_complexity: int = 1, smooth: bool = True):
        self._complexity = model_complexity
        self._smooth = smooth  # kept for API compatibility; IMAGE mode
        self._landmarker = None
        self._enabled = False

    def load(self) -> bool:
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import vision

            variant = _MODEL_VARIANT.get(self._complexity, "full")
            path = Path(_MODEL_FILES.get(self._complexity, _MODEL_FILES[1]))
            if not path.exists():
                logger.info("Downloading pose landmarker (%s)...", variant)
                urllib.request.urlretrieve(_MODEL_URL.format(variant=variant), path)
            opts = vision.PoseLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(path)),
                running_mode=vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._landmarker = vision.PoseLandmarker.create_from_options(opts)
            self._enabled = True
            logger.info("PoseEstimator loaded (variant=%s)", variant)
            return True
        except Exception as exc:
            logger.error("PoseEstimator load failed: %s", exc)
            return False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def process(self, rgb_frame: np.ndarray) -> PoseResult | None:
        """Process an RGB frame. Returns PoseResult or None if no pose detected."""
        if not self._enabled or self._landmarker is None:
            return None
        try:
            import mediapipe as mp

            mp_img = mp.Image(
                image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb_frame)
            )
            res = self._landmarker.detect(mp_img)
        except Exception as exc:
            logger.warning("Pose process error: %s", exc)
            return None

        if not res.pose_landmarks:
            return None

        lms = res.pose_landmarks[0]
        landmarks: dict[str, tuple[float, float, float]] = {}
        for name, idx in _LM.items():
            lm = lms[idx]
            if getattr(lm, "visibility", 1.0) >= _MIN_VISIBILITY:
                landmarks[name] = (lm.x, lm.y, lm.z)

        angles: dict[str, float] = {}
        for joint, (a, b, c) in _ANGLE_JOINTS.items():
            if a in landmarks and b in landmarks and c in landmarks:
                angles[joint] = _angle_between(landmarks[a], landmarks[b], landmarks[c])

        return PoseResult(landmarks=landmarks, angles=angles)

    def draw(self, frame: np.ndarray, result: PoseResult) -> np.ndarray:
        """Draw skeleton overlay onto BGR frame (in-place). Returns frame."""
        h, w = frame.shape[:2]

        # draw connections
        for a_name, b_name in _SKELETON_CONNECTIONS:
            pa = result.pixel_point(a_name, w, h)
            pb = result.pixel_point(b_name, w, h)
            if pa and pb:
                cv2.line(frame, pa, pb, _COLOR_SKELETON, 2, cv2.LINE_AA)

        # draw joints + angle labels
        for name in _LM:
            pt = result.pixel_point(name, w, h)
            if pt is None:
                continue
            cv2.circle(frame, pt, 4, _COLOR_JOINT, -1)
            if name in result.angles:
                cv2.putText(
                    frame,
                    f"{result.angles[name]:.0f}°",
                    (pt[0] + 5, pt[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.36,
                    _COLOR_JOINT,
                    1,
                )
        return frame

    def release(self) -> None:
        if self._landmarker:
            try:
                self._landmarker.close()
            except Exception:
                pass
            self._landmarker = None
        self._enabled = False
