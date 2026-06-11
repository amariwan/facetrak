"""Abstract interfaces for camera sources and PTZ controllers."""
from abc import ABC, abstractmethod

import numpy as np


class CameraSource(ABC):
    """Unified interface for any camera input (USB, Pi, RTSP, ONVIF)."""

    @abstractmethod
    def read(self) -> tuple[bool, np.ndarray | None]:
        """Return (ok, frame).  frame is BGR uint8 or None on failure."""

    @abstractmethod
    def release(self) -> None:
        """Release all resources."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable camera name."""

    @property
    @abstractmethod
    def resolution(self) -> tuple[int, int]:
        """(width, height) of frames returned by read()."""

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()


class PTZController(ABC):
    """Network PTZ interface (ONVIF or similar).

    Separate from servo.PanTiltController which drives an Arduino over serial.
    """

    @abstractmethod
    def move(self, pan: float, tilt: float) -> None:
        """Continuous move. pan/tilt in [-1.0, 1.0], 0 = stop."""

    @abstractmethod
    def zoom(self, level: float) -> None:
        """Absolute zoom level in [0.0, 1.0], 0 = wide, 1 = tele."""

    @abstractmethod
    def stop(self) -> None:
        """Stop all movement."""
