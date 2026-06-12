import logging
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from .yunet import YuNetDetector

logger = logging.getLogger(__name__)

_detector: YuNetDetector | None = None


def _get_detector() -> YuNetDetector:
    global _detector
    if _detector is None:
        _detector = YuNetDetector()
    return _detector

FACE_DIR = Path("faces") / "data"
EMBED_DIM = 128

_MODEL_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
              "face_recognition_sface/face_recognition_sface_2021dec.onnx")
_MODEL_PATH = Path("face_recognition_sface.onnx")

_MIN_SAMPLES = 3
_TOP_K = 3
_AMBIGUITY_RATIO = 1.15


def _ensure_model() -> Path:
    if not _MODEL_PATH.exists():
        logger.info("Downloading SFace recognition model (~37MB)...")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    return _MODEL_PATH


class FaceDatabase:
    def __init__(self):
        self.names: list[str] = []
        self.encodings: list[np.ndarray] = []
        self._recognizer = None

    def _ensure_recognizer(self):
        if self._recognizer is None:
            self._recognizer = cv2.FaceRecognizerSF.create(
                str(_ensure_model()), "")
        return self._recognizer

    def load(self):
        FACE_DIR.mkdir(parents=True, exist_ok=True)
        self.names.clear()
        self.encodings.clear()
        for f in sorted(FACE_DIR.iterdir()):
            if f.suffix != ".npy":
                continue
            data = np.load(str(f))
            if data.ndim == 1:
                data = data.reshape(1, -1)
            if data.ndim == 4:
                data = self._migrate_image_samples(f, data)
                if data is None:
                    continue
            if data.ndim != 2 or data.shape[1] != EMBED_DIM:
                logger.warning(
                    "Skipping %s: incompatible encoding %s, "
                    "please re-register this person", f.name, data.shape)
                continue
            self.names.append(f.stem.replace("_", " "))
            self.encodings.append(data)

    def _migrate_image_samples(self, path: Path, images: np.ndarray
                                ) -> np.ndarray | None:
        logger.info("Migrating %s from image samples to SFace embeddings",
                    path.name)
        detector = _get_detector()
        feats = []
        for img in images:
            dets = detector.detect(np.ascontiguousarray(img))
            if not dets:
                continue
            best = max(dets, key=lambda d: d.score)
            emb = self.embed(img, best.row)
            if emb is not None:
                feats.append(emb)
        if len(feats) < _MIN_SAMPLES:
            logger.warning("Migration of %s failed (%d usable samples), "
                           "please re-register", path.name, len(feats))
            return None
        matrix = np.stack(feats, axis=0)
        np.save(str(path), matrix)
        return matrix

    def embed(self, frame: np.ndarray, det_row: np.ndarray
              ) -> np.ndarray | None:
        try:
            rec = self._ensure_recognizer()
            chip = rec.alignCrop(frame, det_row)
            feat = rec.feature(chip).flatten().astype(np.float32)
        except cv2.error:
            return None
        norm = np.linalg.norm(feat)
        return feat / norm if norm > 0 else None

    def register(self, name: str, embeddings: list[np.ndarray]) -> bool:
        feats = [e for e in embeddings if e is not None]
        if len(feats) < _MIN_SAMPLES:
            return False
        matrix = np.stack(feats, axis=0)
        FACE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(str(FACE_DIR / f"{name.replace(' ', '_')}.npy"), matrix)
        self.load()
        return True

    def predict(self, embedding: np.ndarray | None, threshold: float = 0.36
                ) -> tuple[str | None, float]:
        if embedding is None or not self.encodings:
            return None, 0.0
        scores: list[tuple[float, str]] = []
        for name, encs in zip(self.names, self.encodings):
            sims = encs @ embedding
            k = min(_TOP_K, len(sims))
            scores.append((float(np.mean(np.partition(sims, -k)[-k:])), name))
        scores.sort(key=lambda s: -s[0])
        best_sim, best_name = scores[0]
        second_sim = scores[1][0] if len(scores) > 1 else 0.0
        ambiguous = (second_sim > 0.1
                     and best_sim / (second_sim + 1e-8) < _AMBIGUITY_RATIO)
        if best_sim < threshold or ambiguous:
            return None, best_sim
        return best_name, best_sim

    @property
    def known_names(self) -> list[str]:
        return self.names[:]
