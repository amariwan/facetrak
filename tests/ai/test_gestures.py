"""Tests for gesture classifier logic (no MediaPipe runtime needed)."""
import pytest
from unittest.mock import MagicMock

from facetrak.ai.gestures import _classify, _finger_extended, Gesture


def _make_landmark(x=0.5, y=0.5, z=0.0):
    lm = MagicMock()
    lm.x = x
    lm.y = y
    lm.z = z
    return lm


def _make_hand(finger_states: list[bool], thumb_up: bool = True):
    """Build a minimal 21-landmark hand where fingers are extended/folded."""
    lms = [_make_landmark() for _ in range(21)]

    # wrist at bottom
    lms[0].x, lms[0].y = 0.5, 0.9

    # thumb: tip=4, mcp=2, pip=3
    if finger_states[0]:  # thumb extended
        lms[2].x, lms[2].y = 0.4, 0.7   # mcp
        lms[3].x, lms[3].y = 0.35, 0.6  # pip
        lms[4].x, lms[4].y = 0.25, 0.5  # tip — far from mcp
        if not thumb_up:
            lms[4].y = 0.95  # thumb pointing down
    else:
        lms[2].x, lms[2].y = 0.5, 0.7
        lms[3].x, lms[3].y = 0.5, 0.65
        lms[4].x, lms[4].y = 0.5, 0.6

    # index=8/6, middle=12/10, ring=16/14, pinky=20/18
    tip_idxs = [8, 12, 16, 20]
    pip_idxs = [6, 10, 14, 18]
    for i, (tip, pip) in enumerate(zip(tip_idxs, pip_idxs)):
        if finger_states[i + 1]:  # extended
            lms[pip].y = 0.6
            lms[tip].y = 0.3   # tip above pip
        else:
            lms[pip].y = 0.6
            lms[tip].y = 0.7   # tip below pip (folded)

    return lms


class TestFingerExtended:
    def test_extended_finger_tip_above_pip(self):
        lms = [_make_landmark() for _ in range(21)]
        lms[8].y = 0.2   # tip high up
        lms[6].y = 0.6   # pip lower
        assert _finger_extended(lms, 8, 6) is True

    def test_folded_finger_tip_below_pip(self):
        lms = [_make_landmark() for _ in range(21)]
        lms[8].y = 0.8
        lms[6].y = 0.5
        assert _finger_extended(lms, 8, 6) is False


class TestClassify:
    def test_fist_no_fingers_extended(self):
        lms = _make_hand([False, False, False, False, False])
        assert _classify(lms) == Gesture.FIST

    def test_open_hand_four_fingers_extended(self):
        lms = _make_hand([False, True, True, True, True])
        assert _classify(lms) == Gesture.OPEN

    def test_peace_index_and_middle(self):
        lms = _make_hand([False, True, True, False, False])
        assert _classify(lms) == Gesture.PEACE

    def test_point_only_index(self):
        lms = _make_hand([False, True, False, False, False])
        assert _classify(lms) == Gesture.POINT

    def test_thumbs_up(self):
        lms = _make_hand([True, False, False, False, False], thumb_up=True)
        assert _classify(lms) == Gesture.THUMBS_UP


class TestObjectDetectorDisabledWithoutUltralytics:
    def test_detect_returns_empty_when_not_loaded(self):
        import numpy as np
        from facetrak.ai.objects import ObjectDetector
        det = ObjectDetector()
        # not calling load() → disabled
        result = det.detect(np.zeros((480, 640, 3), np.uint8))
        assert result == []

    def test_enabled_false_before_load(self):
        from facetrak.ai.objects import ObjectDetector
        det = ObjectDetector()
        assert det.enabled is False


class TestPoseEstimatorDisabledWhenNotLoaded:
    def test_process_returns_none_when_not_loaded(self):
        import numpy as np
        from facetrak.ai.pose import PoseEstimator
        pe = PoseEstimator()
        result = pe.process(np.zeros((480, 640, 3), np.uint8))
        assert result is None

    def test_enabled_false_before_load(self):
        from facetrak.ai.pose import PoseEstimator
        assert PoseEstimator().enabled is False


class TestGestureDetectorDisabledWhenNotLoaded:
    def test_process_returns_empty_when_not_loaded(self):
        import numpy as np
        from facetrak.ai.gestures import GestureDetector
        gd = GestureDetector()
        result = gd.process(np.zeros((480, 640, 3), np.uint8))
        assert result == []
