"""Raspberry Pi Camera Module via Picamera2 (libcamera).

Imported only when picamera2 is available and Pi hardware is detected.
Falls back gracefully — import this module and catch ImportError.
"""
import logging

import numpy as np

from .base import CameraSource

logger = logging.getLogger(__name__)


def _is_pi() -> bool:
    """Return True when running on Raspberry Pi hardware."""
    try:
        with open("/proc/device-tree/model") as f:
            return "Raspberry Pi" in f.read()
    except OSError:
        return False


class PiCamera(CameraSource):
    """Captures frames from Pi Camera Module 2/3 or HQ Camera via Picamera2."""

    def __init__(self, index: int = 0, width: int = 1280, height: int = 720,
                 fps: int = 30, name: str = "Pi Camera"):
        try:
            from picamera2 import Picamera2  # type: ignore
        except ImportError as e:
            raise ImportError(
                "picamera2 is required for PiCamera. "
                "Install with: pip install picamera2"
            ) from e

        self._name = name
        self._width = width
        self._height = height

        self._cam = Picamera2(index)
        cfg = self._cam.create_video_configuration(
            main={"size": (width, height), "format": "BGR888"},
            controls={"FrameRate": fps},
        )
        self._cam.configure(cfg)
        self._cam.start()
        logger.info("PiCamera started at %dx%d @ %dfps", width, height, fps)

    def read(self) -> tuple[bool, np.ndarray | None]:
        try:
            frame = self._cam.capture_array("main")
            return True, frame
        except Exception as exc:
            logger.warning("PiCamera read error: %s", exc)
            return False, None

    def release(self) -> None:
        try:
            self._cam.stop()
            self._cam.close()
        except Exception as exc:
            logger.warning("PiCamera release error: %s", exc)

    @property
    def name(self) -> str:
        return self._name

    @property
    def resolution(self) -> tuple[int, int]:
        return self._width, self._height
