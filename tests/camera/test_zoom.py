"""Tests for digital zoom logic."""
import numpy as np
import pytest

from facetrak.camera.zoom import ZoomTracker, apply_digital_zoom, _MIN_ZOOM, _MAX_ZOOM


class TestZoomTracker:
    def test_no_face_returns_to_min(self):
        tracker = ZoomTracker(target_ratio=0.3, smooth=1.0)
        level = tracker.update(None, 480)
        assert level == pytest.approx(_MIN_ZOOM, abs=0.01)

    def test_small_face_zooms_in(self):
        # face height 10% of frame → should zoom in towards 0.3 ratio
        tracker = ZoomTracker(target_ratio=0.3, hysteresis=0.05, smooth=1.0)
        level = tracker.update(face_h=48, frame_h=480)  # ratio = 0.1
        assert level > 1.0

    def test_large_face_holds_or_zooms_out(self):
        # face height 50% of frame → ratio 0.5, well above target+hysteresis
        tracker = ZoomTracker(target_ratio=0.3, hysteresis=0.05, smooth=1.0)
        level = tracker.update(face_h=240, frame_h=480)  # ratio = 0.5
        assert level == pytest.approx(_MIN_ZOOM, abs=0.01)

    def test_face_inside_dead_zone_holds_level(self):
        tracker = ZoomTracker(target_ratio=0.3, hysteresis=0.05, smooth=1.0)
        tracker.zoom_level = 2.0
        # ratio = 0.28, within [0.25, 0.35] dead zone
        level = tracker.update(face_h=134, frame_h=480)
        assert level == pytest.approx(2.0, abs=0.01)

    def test_zoom_clamped_to_max(self):
        tracker = ZoomTracker(target_ratio=0.3, smooth=1.0)
        # tiny face — would want huge zoom
        level = tracker.update(face_h=1, frame_h=480)
        assert level <= _MAX_ZOOM

    def test_zoom_never_below_min(self):
        tracker = ZoomTracker()
        for _ in range(20):
            tracker.update(None, 480)
        assert tracker.zoom_level >= _MIN_ZOOM


class TestApplyDigitalZoom:
    def _solid_frame(self, w=640, h=480, color=(100, 150, 200)):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:] = color
        return frame

    def test_zoom_one_returns_same_shape(self):
        frame = self._solid_frame()
        out = apply_digital_zoom(frame, zoom=1.0)
        assert out.shape == frame.shape

    def test_zoomed_frame_same_shape(self):
        frame = self._solid_frame()
        out = apply_digital_zoom(frame, zoom=2.0)
        assert out.shape == frame.shape

    def test_zoom_one_output_identical(self):
        frame = self._solid_frame()
        out = apply_digital_zoom(frame, zoom=1.0)
        np.testing.assert_array_equal(out, frame)

    def test_center_crop_preserves_center_color(self):
        frame = self._solid_frame(color=(0, 255, 0))
        # centre of frame is green — after zoom-crop it should still be green
        out = apply_digital_zoom(frame, zoom=2.0, cx=0.5, cy=0.5)
        h, w = out.shape[:2]
        np.testing.assert_array_equal(out[h // 2, w // 2], [0, 255, 0])

    def test_off_center_crop_clamps_to_valid_roi(self):
        frame = self._solid_frame()
        # extreme corner — should not raise, just clamp
        out = apply_digital_zoom(frame, zoom=4.0, cx=0.0, cy=0.0)
        assert out.shape == frame.shape


class TestMockCameraSource:
    """Verify that a mock CameraSource satisfies the interface."""

    def _make_mock(self, frames):
        from facetrak.camera.base import CameraSource

        class MockCamera(CameraSource):
            def __init__(self, frames):
                self._frames = iter(frames)
                self._released = False

            def read(self):
                try:
                    return True, next(self._frames)
                except StopIteration:
                    return False, None

            def release(self):
                self._released = True

            @property
            def name(self):
                return "mock"

            @property
            def resolution(self):
                return (640, 480)

        return MockCamera(frames)

    def test_mock_returns_frames(self):
        f1 = np.zeros((480, 640, 3), np.uint8)
        f2 = np.ones((480, 640, 3), np.uint8) * 255
        cam = self._make_mock([f1, f2])
        ok, frame = cam.read()
        assert ok
        np.testing.assert_array_equal(frame, f1)

    def test_mock_exhausted_returns_false(self):
        cam = self._make_mock([])
        ok, frame = cam.read()
        assert not ok
        assert frame is None

    def test_context_manager_calls_release(self):
        cam = self._make_mock([])
        with cam:
            pass
        assert cam._released
