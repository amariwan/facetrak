import logging
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from facetrak.models import FaceDetection

logger = logging.getLogger(__name__)

_MODEL_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
              "face_detection_yunet/face_detection_yunet_2023mar.onnx")
_MODEL_PATH = Path("face_detection_yunet.onnx")

_SCORE_THRESHOLD = 0.7
_NMS_THRESHOLD = 0.3
_TOP_K = 20


def ensure_model() -> Path:
    if not _MODEL_PATH.exists():
        logger.info("Downloading YuNet face detector (~350KB)...")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    return _MODEL_PATH


class YuNetDetector:
    def __init__(self):
        self._net = cv2.FaceDetectorYN.create(
            str(ensure_model()), "", (320, 320),
            _SCORE_THRESHOLD, _NMS_THRESHOLD, _TOP_K)
        self._input_size: tuple[int, int] | None = None

    def detect(self, frame: np.ndarray, scale: float = 1.0
               ) -> list[FaceDetection]:
        h, w = frame.shape[:2]
        if self._input_size != (w, h):
            self._net.setInputSize((w, h))
            self._input_size = (w, h)
        _, rows = self._net.detect(frame)
        if rows is None:
            return []
        return [
            FaceDetection(
                row=row,
                x=int(row[0] * scale), y=int(row[1] * scale),
                w=int(row[2] * scale), h=int(row[3] * scale),
                score=float(row[14]),
            )
            for row in rows
        ]
