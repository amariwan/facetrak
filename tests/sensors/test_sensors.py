"""Tests for sensor modules — no real hardware or sounddevice needed."""
import time
import numpy as np
import pytest


class TestAudioMonitor:
    def test_disabled_without_sounddevice(self):
        """AudioMonitor.start() returns False when sounddevice is missing."""
        from unittest.mock import patch
        import sys
        with patch.dict(sys.modules, {"sounddevice": None}):
            from importlib import reload
            import facetrak.sensors.audio as audio_mod
            reload(audio_mod)
            monitor = audio_mod.AudioMonitor()
            ok = monitor.start()
        assert not ok
        assert not monitor.enabled

    def test_poll_empty_returns_none(self):
        from facetrak.sensors.audio import AudioMonitor
        monitor = AudioMonitor()
        assert monitor.poll() is None

    def test_poll_all_empty_returns_list(self):
        from facetrak.sensors.audio import AudioMonitor
        assert AudioMonitor().poll_all() == []

    def test_stop_on_unstarted_monitor_safe(self):
        from facetrak.sensors.audio import AudioMonitor
        monitor = AudioMonitor()
        monitor.stop()  # should not raise
        assert not monitor.enabled


class TestRmsDb:
    def test_silence_returns_negative_96(self):
        from facetrak.sensors.audio import _rms_db
        silence = np.zeros(512, dtype=np.int16)
        assert _rms_db(silence) == pytest.approx(-96.0, abs=1.0)

    def test_full_scale_sine_near_0db(self):
        from facetrak.sensors.audio import _rms_db
        t = np.linspace(0, 2 * np.pi, 512)
        sine = (np.sin(t) * 32767).astype(np.int16)
        db = _rms_db(sine)
        assert -5.0 < db < 0.0

    def test_louder_signal_higher_db(self):
        from facetrak.sensors.audio import _rms_db
        quiet = (np.random.randint(-100, 100, 512)).astype(np.int16)
        loud  = (np.random.randint(-10000, 10000, 512)).astype(np.int16)
        assert _rms_db(loud) > _rms_db(quiet)


class TestPIRSensor:
    def test_disabled_without_gpio(self):
        from unittest.mock import patch
        import sys
        with patch.dict(sys.modules, {"RPi": None, "RPi.GPIO": None}):
            from facetrak.sensors.pir import PIRSensor
            pir = PIRSensor(pin=17)
            ok = pir.start()
        assert not ok
        assert not pir.enabled

    def test_poll_empty_returns_none(self):
        from facetrak.sensors.pir import PIRSensor
        assert PIRSensor(pin=17).poll() is None

    def test_stop_unstarted_safe(self):
        from facetrak.sensors.pir import PIRSensor
        PIRSensor(pin=17).stop()  # should not raise


class TestDepthEstimator:
    def test_disabled_before_load(self):
        from facetrak.sensors.depth import DepthEstimator
        assert not DepthEstimator().enabled

    def test_estimate_returns_none_when_not_loaded(self):
        from facetrak.sensors.depth import DepthEstimator
        frame = np.zeros((480, 640, 3), np.uint8)
        assert DepthEstimator().estimate(frame) is None

    def test_face_distance_valid_roi(self):
        from facetrak.sensors.depth import DepthEstimator
        depth_map = np.linspace(0, 1, 480 * 640, dtype=np.float32).reshape(480, 640)
        de = DepthEstimator()
        dist = de.face_distance(depth_map, x=100, y=100, w=80, h=80)
        assert 0.0 <= dist <= 1.0

    def test_face_distance_out_of_bounds_clamps(self):
        from facetrak.sensors.depth import DepthEstimator
        depth_map = np.ones((480, 640), dtype=np.float32) * 0.5
        de = DepthEstimator()
        dist = de.face_distance(depth_map, x=620, y=460, w=100, h=100)
        assert 0.0 <= dist <= 1.0

    def test_draw_overlay_same_shape(self):
        from facetrak.sensors.depth import DepthEstimator
        frame = np.zeros((480, 640, 3), np.uint8)
        depth_map = np.zeros((480, 640), dtype=np.float32)
        de = DepthEstimator()
        out = de.draw_overlay(frame, depth_map)
        assert out.shape == (480, 640, 3)
