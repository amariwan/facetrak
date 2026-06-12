import logging
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_BASE = "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender"
_AGE_PROTO_URL    = f"{_BASE}/age_deploy.prototxt"
_GENDER_PROTO_URL = f"{_BASE}/gender_deploy.prototxt"

_AGE_MODEL_URLS = [
    "https://github.com/smahesh29/Gender-and-Age-Detection/raw/master/age_net.caffemodel",
    "https://storage.googleapis.com/learnopencv2/age_net.caffemodel",
]
_GENDER_MODEL_URLS = [
    "https://github.com/smahesh29/Gender-and-Age-Detection/raw/master/gender_net.caffemodel",
    "https://storage.googleapis.com/learnopencv2/gender_net.caffemodel",
]

_AGE_PROTO    = Path("age_deploy.prototxt")
_AGE_MODEL    = Path("age_net.caffemodel")
_GENDER_PROTO = Path("gender_deploy.prototxt")
_GENDER_MODEL = Path("gender_net.caffemodel")

_AGE_BUCKETS = ["0-2", "4-6", "8-12", "15-20", "25-32", "38-43", "48-53", "60+"]
_GENDERS     = ["Male", "Female"]
_MEAN        = (78.4263377603, 87.7689143744, 114.895847746)


def _fetch(path: Path, urls: list[str]) -> bool:
    if path.exists():
        return True
    for url in urls:
        try:
            logger.info("Downloading %s …", path.name)
            tmp = path.with_suffix(".part")
            urllib.request.urlretrieve(url, tmp)
            tmp.rename(path)
            return True
        except (urllib.error.URLError, OSError) as exc:
            logger.debug("  mirror failed (%s): %s", url, exc)
            if tmp.exists():
                tmp.unlink(missing_ok=True)
    return False


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
            if not self._download_all():
                logger.warning(
                    "Age/gender models unavailable (download failed from all "
                    "mirrors). Feature disabled. To enable, manually place "
                    "age_net.caffemodel and gender_net.caffemodel in the "
                    "project directory.")
                self._disabled = True
                return
            self._age_net    = cv2.dnn.readNet(str(_AGE_PROTO),    str(_AGE_MODEL))
            self._gender_net = cv2.dnn.readNet(str(_GENDER_PROTO), str(_GENDER_MODEL))
            self._ready = True
            logger.info("Age/gender estimator ready")
        except Exception as exc:
            logger.warning("Age/gender unavailable — feature disabled: %s", exc)
            self._disabled = True

    def _download_all(self) -> bool:
        ok  = _fetch(_AGE_PROTO,    [_AGE_PROTO_URL])
        ok &= _fetch(_AGE_MODEL,    _AGE_MODEL_URLS)
        ok &= _fetch(_GENDER_PROTO, [_GENDER_PROTO_URL])
        ok &= _fetch(_GENDER_MODEL, _GENDER_MODEL_URLS)
        return ok

    def predict(self, face_bgr: np.ndarray) -> tuple[str, str]:
        if not self._ready or face_bgr.size == 0:
            return "?", "?"
        try:
            blob = cv2.dnn.blobFromImage(
                face_bgr, 1.0, (227, 227), _MEAN, swapRB=False)
            self._gender_net.setInput(blob)
            gender_out = self._gender_net.forward()
            if gender_out is None or len(gender_out) == 0 or len(gender_out[0]) == 0:
                return "?", "?"
            gender = _GENDERS[gender_out[0].argmax()]
            self._age_net.setInput(blob)
            age_out = self._age_net.forward()
            if age_out is None or len(age_out) == 0 or len(age_out[0]) == 0:
                return "?", "?"
            age = _AGE_BUCKETS[age_out[0].argmax()]
            return age, gender
        except cv2.error as exc:
            logger.debug("Age/gender inference failed: %s", exc)
            return "?", "?"
