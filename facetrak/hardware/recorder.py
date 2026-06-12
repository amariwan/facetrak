import cv2
import datetime
from typing import Optional


class VideoRecorder:
    def __init__(self):
        self.writer: Optional[cv2.VideoWriter] = None
        self.path: Optional[str] = None
        self.recording = False

    def start(self, w: int, h: int, fps: float = 20.0):
        if self.recording:
            return
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = f"recording_{ts}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(self.path, fourcc, fps, (w, h))
        if not self.writer.isOpened():
            logger.error("VideoWriter failed to open %s", self.path)
            self.writer = None
            self.recording = False
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
