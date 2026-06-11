"""Full-body pose estimation via MediaPipe Pose.

Extracts 33 body landmarks, computes key joint angles (elbow, knee, shoulder),
and draws an annotated skeleton overlay.

MediaPipe is already a project dependency — no extra install needed.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# MediaPipe Pose landmark indices
_LM = {
    "nose":            0,
    "left_shoulder":   11, "right_shoulder":  12,
    "left_elbow":      13, "right_elbow":     14,
    "left_wrist":      15, "right_wrist":     16,
    "left_hip":        23, "right_hip":       24,
    "left_knee":       25, "right_knee":      26,
    "left_ankle":      27, "right_ankle":     28,
}

_SKELETON_CONNECTIONS = [
    ("left_shoulder",  "right_shoulder"),
    ("left_shoulder",  "left_elbow"),
    ("left_elbow",     "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow",    "right_wrist"),
    ("left_shoulder",  "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip",       "right_hip"),
    ("left_hip",       "left_knee"),
    ("left_knee",      "left_ankle"),
    ("right_hip",      "right_knee"),
    ("right_knee",     "right_ankle"),
]

_ANGLE_JOINTS = {
    "left_elbow":  ("left_shoulder",  "left_elbow",  "left_wrist"),
    "right_elbow": ("right_shoulder", "right_elbow", "right_wrist"),
    "left_knee":   ("left_hip",       "left_knee",   "left_ankle"),
    "right_knee":  ("right_hip",      "right_knee",  "right_ankle"),
}

_COLOR_SKELETON = (0, 255, 128)
_COLOR_JOINT    = (255, 255, 255)
_MIN_VISIBILITY = 0.5


@dataclass
class PoseResult:
    """Pose data for one person."""
    landmarks: dict[str, tuple[float, float, float]]  # name → (x, y, z) normalised
    angles: dict[str, float]                           # joint name → degrees
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
    """Wraps MediaPipe Pose for per-frame body landmark extraction.

    Usage:
        estimator = PoseEstimator()
        estimator.load()
        result = estimator.process(rgb_frame)
        if result:
            estimator.draw(bgr_frame, result)
    """

    def __init__(self, model_complexity: int = 1, smooth: bool = True):
        self._complexity = model_complexity
        self._smooth = smooth
        self._pose = None
        self._enabled = False

    def load(self) -> bool:
        try:
            import mediapipe as mp
            self._pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=self._complexity,
                smooth_landmarks=self._smooth,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._enabled = True
            logger.info("PoseEstimator loaded (complexity=%d)", self._complexity)
            return True
        except Exception as exc:
            logger.error("PoseEstimator load failed: %s", exc)
            return False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def process(self, rgb_frame: np.ndarray) -> PoseResult | None:
        """Process an RGB frame. Returns PoseResult or None if no pose detected."""
        if not self._enabled or self._pose is None:
            return None
        try:
            res = self._pose.process(rgb_frame)
        except Exception as exc:
            logger.warning("Pose process error: %s", exc)
            return None

        if not res.pose_landmarks:
            return None

        lms = res.pose_landmarks.landmark
        landmarks: dict[str, tuple[float, float, float]] = {}
        for name, idx in _LM.items():
            lm = lms[idx]
            if lm.visibility >= _MIN_VISIBILITY:
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
                cv2.putText(frame, f"{result.angles[name]:.0f}°",
                            (pt[0] + 5, pt[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.36, _COLOR_JOINT, 1)
        return frame

    def release(self) -> None:
        if self._pose:
            self._pose.close()
            self._pose = None
        self._enabled = False
