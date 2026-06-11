"""Digital zoom via ROI crop, with optional optical zoom via PTZController."""
import logging

import cv2
import numpy as np

from .base import CameraSource, PTZController

logger = logging.getLogger(__name__)

_DEFAULT_TARGET_RATIO = 0.30   # face height / frame height to aim for
_DEFAULT_HYSTERESIS   = 0.05   # dead zone around target to avoid flicker
_MIN_ZOOM             = 1.0
_MAX_ZOOM             = 8.0


class ZoomTracker:
    """Computes a smooth digital zoom level based on face bounding box size.

    Call update() each frame with the face bbox (or None).
    Read zoom_level for the current multiplier (1.0 = no zoom).
    """

    def __init__(self, target_ratio: float = _DEFAULT_TARGET_RATIO,
                 hysteresis: float = _DEFAULT_HYSTERESIS,
                 smooth: float = 0.08):
        self.target_ratio = target_ratio
        self.hysteresis   = hysteresis
        self.smooth       = smooth
        self.zoom_level   = 1.0

    def update(self, face_h: int | None, frame_h: int) -> float:
        """Update zoom level. face_h is face bbox height in pixels, or None."""
        if face_h is None or frame_h == 0:
            target = 1.0
        else:
            ratio = face_h / frame_h
            if ratio < self.target_ratio - self.hysteresis:
                target = min(_MAX_ZOOM, self.target_ratio / max(ratio, 0.01))
            elif ratio > self.target_ratio + self.hysteresis:
                target = max(_MIN_ZOOM, self.target_ratio / ratio)
            else:
                target = self.zoom_level  # inside dead zone — hold

        self.zoom_level += self.smooth * (target - self.zoom_level)
        self.zoom_level  = max(_MIN_ZOOM, min(_MAX_ZOOM, self.zoom_level))
        return self.zoom_level


def apply_digital_zoom(frame: np.ndarray, zoom: float,
                       cx: float = 0.5, cy: float = 0.5) -> np.ndarray:
    """Crop frame around (cx, cy) at given zoom level and resize back.

    cx, cy are normalised centre coordinates [0, 1].
    zoom=1.0 returns frame unchanged.
    """
    if zoom <= 1.0:
        return frame
    h, w = frame.shape[:2]
    crop_w = int(w / zoom)
    crop_h = int(h / zoom)
    x0 = max(0, min(w - crop_w, int(cx * w - crop_w / 2)))
    y0 = max(0, min(h - crop_h, int(cy * h - crop_h / 2)))
    cropped = frame[y0:y0 + crop_h, x0:x0 + crop_w]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


class DigitalZoom(CameraSource):
    """Wraps any CameraSource and applies auto-zoom towards the target face.

    Optionally also drives a PTZController for optical zoom when available.
    """

    def __init__(self, source: CameraSource,
                 ptz: PTZController | None = None,
                 target_ratio: float = _DEFAULT_TARGET_RATIO,
                 hysteresis: float = _DEFAULT_HYSTERESIS):
        self._src     = source
        self._ptz     = ptz
        self._tracker = ZoomTracker(target_ratio, hysteresis)
        self._cx      = 0.5
        self._cy      = 0.5

    def set_target(self, x: int, y: int, w: int, h: int,
                   frame_w: int, frame_h: int) -> None:
        """Tell the zoom tracker where the target face is."""
        self._cx = (x + w / 2) / frame_w
        self._cy = (y + h / 2) / frame_h
        self._tracker.update(h, frame_h)
        if self._ptz:
            # map zoom_level [1, MAX_ZOOM] → [0, 1] for optical zoom
            optical = (self._tracker.zoom_level - 1.0) / (_MAX_ZOOM - 1.0)
            self._ptz.zoom(optical)

    def clear_target(self) -> None:
        """No face visible — zoom out."""
        self._tracker.update(None, 1)

    def read(self) -> tuple[bool, np.ndarray | None]:
        ok, frame = self._src.read()
        if not ok or frame is None:
            return False, None
        zoomed = apply_digital_zoom(frame, self._tracker.zoom_level,
                                    self._cx, self._cy)
        return True, zoomed

    def release(self) -> None:
        self._src.release()

    @property
    def name(self) -> str:
        return self._src.name

    @property
    def resolution(self) -> tuple[int, int]:
        return self._src.resolution
