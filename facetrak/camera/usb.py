"""USB / built-in webcam via cv2.VideoCapture."""
import logging

import cv2
import numpy as np

from .base import CameraSource

logger = logging.getLogger(__name__)


class USBCamera(CameraSource):
    """Wraps cv2.VideoCapture for USB or built-in cameras."""

    def __init__(self, index: int = 0, name: str | None = None,
                 width: int | None = None, height: int | None = None):
        self._index = index
        self._name = name or f"USB Camera {index}"
        self._cap = self._open(index)
        if width and height:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._width  = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @staticmethod
    def _open(index: int) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            cap = cv2.VideoCapture(index)
        return cap

    def read(self) -> tuple[bool, np.ndarray | None]:
        ret, frame = self._cap.read()
        return ret, frame if ret else None

    def release(self) -> None:
        if self._cap.isOpened():
            self._cap.release()

    @property
    def name(self) -> str:
        return self._name

    @property
    def resolution(self) -> tuple[int, int]:
        return self._width, self._height

    @property
    def is_opened(self) -> bool:
        return self._cap.isOpened()

    @classmethod
    def probe(cls, max_index: int = 4) -> list[int]:
        """Return list of available USB camera indices."""
        available = []
        for i in range(max_index):
            cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
            if cap.isOpened():
                available.append(i)
                cap.release()
        return available
