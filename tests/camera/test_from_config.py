"""Tests for camera.from_config() config parsing."""
import pytest
from unittest.mock import patch, MagicMock

from facetrak.camera import from_config
from facetrak.camera.usb import USBCamera
from facetrak.camera.rtsp import RTSPCamera


class TestFromConfig:
    def test_legacy_int_camera_key_returns_usb(self):
        cfg = {"camera": 0}
        with patch.object(USBCamera, "__init__", return_value=None) as mock:
            mock.return_value = None
            cams = from_config({"camera": 0})
        assert len(cams) == 1
        assert isinstance(cams[0], USBCamera)

    def test_empty_cameras_list_falls_back_to_legacy(self):
        cfg = {"camera": 0, "cameras": []}
        cams = from_config(cfg)
        assert len(cams) == 1
        assert isinstance(cams[0], USBCamera)

    def test_usb_entry_creates_usb_camera(self):
        cfg = {"cameras": [{"type": "usb", "source": 1, "name": "Side cam"}]}
        cams = from_config(cfg)
        assert len(cams) == 1
        assert isinstance(cams[0], USBCamera)

    def test_rtsp_entry_creates_rtsp_camera(self):
        url = "rtsp://192.168.1.10/stream"
        with patch.object(RTSPCamera, "__init__", return_value=None):
            cams = from_config({"cameras": [{"type": "rtsp", "url": url}]})
        assert len(cams) == 1
        assert isinstance(cams[0], RTSPCamera)

    def test_unknown_type_skipped_with_warning(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            with pytest.raises(RuntimeError):
                # only unknown entry → no valid cameras → RuntimeError
                from_config({"cameras": [{"type": "foobar"}]})
        assert "Unknown camera type" in caplog.text

    def test_multiple_cameras_returned(self):
        cfg = {
            "cameras": [
                {"type": "usb", "source": 0, "name": "A"},
                {"type": "usb", "source": 1, "name": "B"},
            ]
        }
        cams = from_config(cfg)
        assert len(cams) == 2
