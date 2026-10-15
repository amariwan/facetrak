"""Abstract interfaces for camera sources and PTZ controllers."""
import time
from abc import ABC, abstractmethod

import numpy as np


class CameraHealth:
    """Lightweight mutable health record shared by all camera sources."""

    def __init__(self):
        self.connected = False
        self.last_frame_ts: float | None = None
        self.frames = 0
        self.failures = 0
        self.reconnects = 0
        self.last_error: str | None = None
        self._consec_fail = 0

    def ok(self):
        self.connected = True
        self.frames += 1
        self._consec_fail = 0
        self.last_frame_ts = time.time()

    def fail(self, error: str):
        self.failures += 1
        self._consec_fail += 1
        self.last_error = error
        if self._consec_fail >= 5:
            self.connected = False

    def reconnected(self):
        self.reconnects += 1
        self.connected = True
        self._consec_fail = 0

    def to_dict(self) -> dict:
        age = (time.time() - self.last_frame_ts
               if self.last_frame_ts else None)
        return {
            "connected": self.connected,
            "frames": self.frames,
            "failures": self.failures,
            "reconnects": self.reconnects,
            "last_error": self.last_error,
            "last_frame_age_s": round(age, 2) if age is not None else None,
        }


class CameraSource(ABC):
    """Unified interface for any camera input (USB, Pi, RTSP, ONVIF)."""

    @property
    def health(self) -> CameraHealth:
        # Lazy: subclasses define their own __init__ without super().
        h = self.__dict__.get("_health")
        if h is None:
            h = self.__dict__["_health"] = CameraHealth()
        return h

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
