"""Hand landmark detection and gesture recognition via MediaPipe.

Detects up to 2 hands per frame, classifies each into one of:
  THUMBS_UP, THUMBS_DOWN, PEACE, FIST, OPEN, POINT, OK, UNKNOWN

Classification is rule-based on finger extension states — no ML model needed.
Uses the Tasks HandLandmarker API (mediapipe 0.10.x and >=1.x compatible;
the legacy mp.solutions API was removed in MediaPipe 1.x).
"""

from __future__ import annotations

import logging
import urllib.request
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_MIN_DETECTION_CONF = 0.6
_MIN_TRACKING_CONF = 0.5
_MAX_HANDS = 2

_MODEL_FILE = "hand_landmarker.task"
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/"
    "hand_landmarker.task"
)

# Finger tip / pip landmark indices (MediaPipe 21-point hand model)
_FINGER_TIPS = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky
_FINGER_PIPS = [3, 6, 10, 14, 18]  # proximal interphalangeal joints


class Gesture(str, Enum):
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    PEACE = "peace"
    FIST = "fist"
    OPEN = "open_hand"
    POINT = "point"
    OK = "ok"
    UNKNOWN = "unknown"


@dataclass
class HandResult:
    gesture: Gesture
    handedness: str  # "Left" or "Right"
    landmarks: list  # raw MediaPipe NormalizedLandmarkList
    confidence: float


def _finger_extended(
    landmarks, tip_idx: int, pip_idx: int, is_thumb: bool = False
) -> bool:
    """Return True if the finger is extended (tip above pip for most fingers)."""
    tip = landmarks[tip_idx]
    pip = landmarks[pip_idx]
    if is_thumb:
        # thumb extends — Euclidean distance from MCP
        mcp = landmarks[2]
        d_tip = (tip.x - mcp.x) ** 2 + (tip.y - mcp.y) ** 2
        d_pip = (pip.x - mcp.x) ** 2 + (pip.y - mcp.y) ** 2
        return d_tip > d_pip
    return tip.y < pip.y  # y decreases upward in normalised coords


def _classify(landmarks) -> Gesture:
    """Rule-based gesture classifier on 21 MediaPipe hand landmarks."""
    extended = [
        _finger_extended(landmarks, _FINGER_TIPS[0], _FINGER_PIPS[0], is_thumb=True),
        _finger_extended(landmarks, _FINGER_TIPS[1], _FINGER_PIPS[1]),
        _finger_extended(landmarks, _FINGER_TIPS[2], _FINGER_PIPS[2]),
        _finger_extended(landmarks, _FINGER_TIPS[3], _FINGER_PIPS[3]),
        _finger_extended(landmarks, _FINGER_TIPS[4], _FINGER_PIPS[4]),
    ]
    thumb, index, middle, ring, pinky = extended
    n_ext = sum(extended[1:])  # fingers (not thumb) extended

    if n_ext == 0 and thumb:
        # thumb up or down based on y position of tip vs wrist
        wrist_y = landmarks[0].y
        tip_y = landmarks[4].y
        return Gesture.THUMBS_UP if tip_y < wrist_y else Gesture.THUMBS_DOWN

    if n_ext == 0 and not thumb:
        return Gesture.FIST

    if n_ext == 4 and not thumb:
        return Gesture.OPEN

    if index and middle and not ring and not pinky:
        return Gesture.PEACE

    if index and not middle and not ring and not pinky:
        return Gesture.POINT

    # OK: thumb tip close to index tip
    dx = landmarks[4].x - landmarks[8].x
    dy = landmarks[4].y - landmarks[8].y
    dist = (dx**2 + dy**2) ** 0.5
    if dist < 0.08:
        return Gesture.OK

    return Gesture.UNKNOWN


_COLOR_HAND = (0, 128, 255)
_COLOR_GESTURE = (255, 220, 0)


# 21-point hand skeleton topology (stable across MediaPipe versions).
_HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
]


class GestureDetector:
    """Detects hands and classifies gestures per frame.

    Usage:
        gd = GestureDetector()
        gd.load()
        results = gd.process(rgb_frame)
        gd.draw(bgr_frame, results)
    """

    def __init__(self, max_hands: int = _MAX_HANDS):
        self._max_hands = max_hands
        self._landmarker = None
        self._enabled = False

    def load(self) -> bool:
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import vision

            path = Path(_MODEL_FILE)
            if not path.exists():
                logger.info("Downloading hand landmarker model...")
                urllib.request.urlretrieve(_MODEL_URL, path)
            opts = vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(path)),
                running_mode=vision.RunningMode.IMAGE,
                num_hands=self._max_hands,
                min_hand_detection_confidence=_MIN_DETECTION_CONF,
                min_tracking_confidence=_MIN_TRACKING_CONF,
            )
            self._landmarker = vision.HandLandmarker.create_from_options(opts)
            self._enabled = True
            logger.info("GestureDetector loaded (max_hands=%d)", self._max_hands)
            return True
        except Exception as exc:
            logger.error("GestureDetector load failed: %s", exc)
            return False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def process(self, rgb_frame: np.ndarray) -> list[HandResult]:
        """Detect and classify hands in an RGB frame."""
        if not self._enabled or self._landmarker is None:
            return []
        try:
            import mediapipe as mp

            mp_img = mp.Image(
                image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb_frame)
            )
            res = self._landmarker.detect(mp_img)
        except Exception as exc:
            logger.warning("GestureDetector process error: %s", exc)
            return []

        if not res.hand_landmarks:
            return []

        results: list[HandResult] = []
        for i, lms in enumerate(res.hand_landmarks):
            hand_label, conf = "Right", 1.0
            if res.handedness and i < len(res.handedness):
                cat = res.handedness[i][0]
                hand_label = cat.category_name or "Right"
                conf = float(cat.score or 0.0)
            gesture = _classify(lms)
            results.append(
                HandResult(
                    gesture=gesture,
                    handedness=hand_label,
                    landmarks=lms,
                    confidence=conf,
                )
            )
        return results

    def draw(self, frame: np.ndarray, results: list[HandResult]) -> np.ndarray:
        """Draw hand skeleton and gesture label onto BGR frame (in-place)."""
        h, w = frame.shape[:2]
        for result in results:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in result.landmarks]
            for a, b in _HAND_CONNECTIONS:
                if a < len(pts) and b < len(pts):
                    cv2.line(frame, pts[a], pts[b], _COLOR_HAND, 2, cv2.LINE_AA)
            for x, y in pts:
                cv2.circle(frame, (x, y), 3, _COLOR_HAND, -1)

            # gesture label near wrist
            wrist = result.landmarks[0]
            wx, wy = int(wrist.x * w), int(wrist.y * h)
            label = f"{result.handedness}: {result.gesture.value}"
            cv2.putText(
                frame,
                label,
                (wx, wy + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                _COLOR_GESTURE,
                2,
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
