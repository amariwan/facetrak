"""Age and gender estimation via OpenCV DNN (Levi & Hassner Caffe models).

~44 MB per caffemodel; both are downloaded lazily on first call.
If download fails or the net errors, the module self-disables and returns "?"
so the rest of the pipeline continues unaffected.

Results are cached per Track for _CACHE_FRAMES frames to avoid re-running
inference on every frame.
"""
import logging
import urllib.request
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_BASE = "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender"
_AGE_PROTO_URL    = f"{_BASE}/age_deploy.prototxt"
_GENDER_PROTO_URL = f"{_BASE}/gender_deploy.prototxt"
_AGE_MODEL_URL    = "https://storage.googleapis.com/learnopencv2/age_net.caffemodel"
_GENDER_MODEL_URL = "https://storage.googleapis.com/learnopencv2/gender_net.caffemodel"

_AGE_PROTO    = Path("age_deploy.prototxt")
_AGE_MODEL    = Path("age_net.caffemodel")
_GENDER_PROTO = Path("gender_deploy.prototxt")
_GENDER_MODEL = Path("gender_net.caffemodel")

_AGE_BUCKETS = ["0-2", "4-6", "8-12", "15-20", "25-32", "38-43", "48-53", "60+"]
_GENDERS     = ["Male", "Female"]
_MEAN        = (78.4263377603, 87.7689143744, 114.895847746)


class AgeGenderEstimator:
    def __init__(self):
        self._age_net    = None
        self._gender_net = None
        self._ready      = False
        self._disabled   = False

    def load(self):
        if self._disabled:
            return
        try:
            self._download_all()
            self._age_net    = cv2.dnn.readNet(str(_AGE_PROTO),    str(_AGE_MODEL))
            self._gender_net = cv2.dnn.readNet(str(_GENDER_PROTO), str(_GENDER_MODEL))
            self._ready = True
            logger.info("Age/gender estimator ready")
        except Exception:
            logger.warning("Age/gender unavailable — feature disabled",
                           exc_info=True)
            self._disabled = True

    def _download_all(self):
        pairs = [
            (_AGE_PROTO,    _AGE_PROTO_URL),
            (_AGE_MODEL,    _AGE_MODEL_URL),
            (_GENDER_PROTO, _GENDER_PROTO_URL),
            (_GENDER_MODEL, _GENDER_MODEL_URL),
        ]
        for path, url in pairs:
            if path.exists():
                continue
            logger.info("Downloading %s …", path.name)
            urllib.request.urlretrieve(url, path)

    def predict(self, face_bgr: np.ndarray) -> tuple[str, str]:
        """Return (age_range, gender) or ("?", "?") on failure/disabled."""
        if not self._ready or face_bgr.size == 0:
            return "?", "?"
        try:
            blob = cv2.dnn.blobFromImage(
                face_bgr, 1.0, (227, 227), _MEAN, swapRB=False)
            self._gender_net.setInput(blob)
            gender = _GENDERS[self._gender_net.forward()[0].argmax()]
            self._age_net.setInput(blob)
            age = _AGE_BUCKETS[self._age_net.forward()[0].argmax()]
            return age, gender
        except cv2.error:
            return "?", "?"
