"""RTSP IP camera with background thread and auto-reconnect."""
import logging
import queue
import threading
import time

import cv2
import numpy as np

from .base import CameraSource

logger = logging.getLogger(__name__)

_RECONNECT_DELAYS = (1, 2, 4, 8, 16, 30)  # seconds, last value repeated


class RTSPCamera(CameraSource):
    """Reads an RTSP stream in a background thread.

    read() is non-blocking and always returns the latest available frame.
    Automatically reconnects on stream loss with exponential backoff.
    """

    def __init__(self, url: str, name: str | None = None,
                 width: int | None = None, height: int | None = None):
        self._url = url
        self._name = name or url
        self._target_w = width
        self._target_h = height
        self._width = width or 0
        self._height = height or 0

        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=2)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _open_cap(self) -> cv2.VideoCapture | None:
        cap = cv2.VideoCapture(self._url)
        if not cap.isOpened():
            return None
        if self._target_w and self._target_h:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._target_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._target_h)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return cap

    def _capture_loop(self) -> None:
        delay_idx = 0
        while not self._stop_event.is_set():
            cap = self._open_cap()
            if cap is None:
                wait = _RECONNECT_DELAYS[min(delay_idx, len(_RECONNECT_DELAYS) - 1)]
                logger.warning("RTSP connect failed (%s), retry in %ds", self._url, wait)
                self._stop_event.wait(wait)
                delay_idx += 1
                continue

            logger.info("RTSP connected: %s", self._url)
            delay_idx = 0

            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    logger.warning("RTSP stream lost: %s", self._url)
                    break
                # drop oldest frame if queue is full
                if self._queue.full():
                    try:
                        self._queue.get_nowait()
                    except queue.Empty:
                        pass
                self._queue.put_nowait(frame)

            cap.release()

    def read(self) -> tuple[bool, np.ndarray | None]:
        try:
            frame = self._queue.get_nowait()
            return True, frame
        except queue.Empty:
            return False, None

    def release(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=5)

    @property
    def name(self) -> str:
        return self._name

    @property
    def resolution(self) -> tuple[int, int]:
        return self._width, self._height
