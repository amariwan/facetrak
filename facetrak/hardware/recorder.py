import datetime
import logging
from pathlib import Path

import cv2

logger = logging.getLogger(__name__)


class VideoRecorder:
    def __init__(self):
        self.writer: cv2.VideoWriter | None = None
        self.path: str | None = None
        self.recording = False

    def start(self, w: int, h: int, fps: float = 20.0,
              directory: str | Path | None = None):
        if self.recording:
            return
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base = Path(directory) if directory else Path(".")
        try:
            base.mkdir(parents=True, exist_ok=True)
        except OSError:
            base = Path(".")
        self.path = str(base / f"recording_{ts}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(self.path, fourcc, fps, (w, h))
        if not self.writer.isOpened():
            logger.error("VideoWriter failed to open %s", self.path)
            self.writer = None
            self.recording = False
            self.path = None
            return
        self.recording = True

    def write(self, frame: cv2.typing.MatLike):
        if self.writer and self.recording:
            self.writer.write(frame)

    def stop(self):
        if self.writer:
            self.writer.release()
            self.writer = None
        self.recording = False
        self.path = None
