"""Hand landmark detection and gesture recognition via MediaPipe Hands.

Detects up to 2 hands per frame, classifies each into one of:
  THUMBS_UP, THUMBS_DOWN, PEACE, FIST, OPEN, POINT, OK, UNKNOWN

Classification is rule-based on finger extension states — no ML model needed.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_MIN_DETECTION_CONF = 0.6
_MIN_TRACKING_CONF  = 0.5
_MAX_HANDS          = 2

# Finger tip / pip landmark indices (MediaPipe 21-point hand model)
_FINGER_TIPS = [4, 8, 12, 16, 20]   # thumb, index, middle, ring, pinky
_FINGER_PIPS = [3, 6, 10, 14, 18]   # proximal interphalangeal joints


class Gesture(str, Enum):
    THUMBS_UP   = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    PEACE       = "peace"
    FIST        = "fist"
    OPEN        = "open_hand"
    POINT       = "point"
    OK          = "ok"
    UNKNOWN     = "unknown"


@dataclass
class HandResult:
    gesture: Gesture
    handedness: str          # "Left" or "Right"
    landmarks: list          # raw MediaPipe NormalizedLandmarkList
    confidence: float


def _finger_extended(landmarks, tip_idx: int, pip_idx: int,
                     is_thumb: bool = False) -> bool:
    """Return True if the finger is extended (tip above pip for most fingers)."""
    tip = landmarks[tip_idx]
    pip = landmarks[pip_idx]
    if is_thumb:
        # thumb extends horizontally — compare x
        mcp = landmarks[2]
        return abs(tip.x - mcp.x) > abs(pip.x - mcp.x)
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
        tip_y   = landmarks[4].y
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
    if thumb:
        dx = landmarks[4].x - landmarks[8].x
        dy = landmarks[4].y - landmarks[8].y
        dist = (dx**2 + dy**2) ** 0.5
        if dist < 0.08:
            return Gesture.OK

    return Gesture.UNKNOWN


_COLOR_HAND    = (0, 128, 255)
_COLOR_GESTURE = (255, 220, 0)


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
        self._hands = None
        self._enabled = False

    def load(self) -> bool:
        try:
            import mediapipe as mp
            self._hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=self._max_hands,
                min_detection_confidence=_MIN_DETECTION_CONF,
                min_tracking_confidence=_MIN_TRACKING_CONF,
            )
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
        if not self._enabled or self._hands is None:
            return []
        try:
            res = self._hands.process(rgb_frame)
        except Exception as exc:
            logger.warning("GestureDetector process error: %s", exc)
            return []

        if not res.multi_hand_landmarks:
            return []

        results: list[HandResult] = []
        handedness_list = res.multi_handedness or []
        for i, hand_lm in enumerate(res.multi_hand_landmarks):
            lms = hand_lm.landmark
            hand_label = "Right"
            conf = 1.0
            if i < len(handedness_list):
                classification = handedness_list[i].classification[0]
                hand_label = classification.label
                conf = classification.score
            gesture = _classify(lms)
            results.append(HandResult(
                gesture=gesture,
                handedness=hand_label,
                landmarks=lms,
                confidence=conf,
            ))
        return results

    def draw(self, frame: np.ndarray, results: list[HandResult]) -> np.ndarray:
        """Draw hand skeleton and gesture label onto BGR frame (in-place)."""
        import mediapipe as mp
        mp_draw = mp.solutions.drawing_utils
        mp_hands = mp.solutions.hands
        h, w = frame.shape[:2]

        for result in results:
            # re-wrap landmarks for drawing util
            import mediapipe.framework.formats.landmark_pb2 as lm_pb
            hand_lm_proto = lm_pb.NormalizedLandmarkList()
            for lm in result.landmarks:
                hand_lm_proto.landmark.add(x=lm.x, y=lm.y, z=lm.z)

            mp_draw.draw_landmarks(
                frame, hand_lm_proto,
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=_COLOR_HAND, thickness=2, circle_radius=3),
                mp_draw.DrawingSpec(color=_COLOR_HAND, thickness=2),
            )

            # gesture label near wrist
            wrist = result.landmarks[0]
            wx, wy = int(wrist.x * w), int(wrist.y * h)
            label = f"{result.handedness}: {result.gesture.value}"
            cv2.putText(frame, label, (wx, wy + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, _COLOR_GESTURE, 2)
        return frame

    def release(self) -> None:
        if self._hands:
            self._hands.close()
            self._hands = None
        self._enabled = False
