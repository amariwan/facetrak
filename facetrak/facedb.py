import cv2
import numpy as np
from pathlib import Path
from typing import Optional

FACE_DIR = Path("faces") / "data"
FACE_W, FACE_H = 64, 128
_MIN_SAMPLES = 3
_TOP_K = 3


class FaceDatabase:
    def __init__(self):
        self.names: list[str] = []
        self.encodings: list[np.ndarray] = []
        self._hog = cv2.HOGDescriptor()
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self._eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml")

    def load(self):
        FACE_DIR.mkdir(parents=True, exist_ok=True)
        self.names.clear()
        self.encodings.clear()
        for f in sorted(FACE_DIR.iterdir()):
            if f.suffix != ".npy":
                continue
            name = f.stem.replace("_", " ")
            data = np.load(str(f))
            if data.ndim == 1:
                data = data.reshape(1, -1)
            self.names.append(name)
            self.encodings.append(data)

    def register(self, name: str, samples: list[np.ndarray]) -> bool:
        feats = [f for f in (self._encode(s) for s in samples if s.size > 0)
                 if f is not None]
        if len(feats) < _MIN_SAMPLES:
            return False
        matrix = np.stack(feats, axis=0)
        FACE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(str(FACE_DIR / f"{name.replace(' ', '_')}.npy"), matrix)
        self.load()
        return True

    def predict(self, face_bgr: np.ndarray, threshold: float = 0.50
                ) -> tuple[Optional[str], float]:
        feat = self._encode(face_bgr)
        if feat is None or not self.encodings:
            return None, 0.0
        scores = []
        for name, encs in zip(self.names, self.encodings):
            sims = encs @ feat
            k = min(_TOP_K, len(sims))
            sim = float(np.mean(np.partition(sims, -k)[-k:]))
            scores.append((sim, name))
        scores.sort(key=lambda x: -x[0])
        best_sim, best_name = scores[0]
        second_sim = scores[1][0] if len(scores) > 1 else 0.0
        ratio = best_sim / (second_sim + 1e-8) if second_sim > 0.1 else 2.0
        if best_sim < threshold or ratio < 1.12:
            return None, best_sim
        return best_name, best_sim

    def _align(self, face_bgr: np.ndarray) -> np.ndarray:
        h, w = face_bgr.shape[:2]
        if h < 40 or w < 40:
            return face_bgr
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        eyes = self._eye_cascade.detectMultiScale(
            gray, scaleFactor=1.15, minNeighbors=4,
            minSize=(int(w * 0.08), int(h * 0.08)))
        if eyes is not None and len(eyes) >= 2:
            eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
            cx1, cy1 = eyes[0][0] + eyes[0][2] // 2, eyes[0][1] + eyes[0][3] // 2
            cx2, cy2 = eyes[1][0] + eyes[1][2] // 2, eyes[1][1] + eyes[1][3] // 2
            dx, dy = cx2 - cx1, cy2 - cy1
            angle = np.degrees(np.arctan2(dy, dx))
            if abs(angle) > 2:
                mx, my = (cx1 + cx2) / 2, (cy1 + cy2) / 2
                M = cv2.getRotationMatrix2D((mx, my), angle, 1.0)
                face_bgr = cv2.warpAffine(
                    face_bgr, M, (w, h), flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REFLECT)
        return face_bgr

    def _encode(self, face_bgr: np.ndarray) -> Optional[np.ndarray]:
        if face_bgr.size == 0:
            return None
        aligned = self._align(face_bgr)
        gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
        gray = self._clahe.apply(gray)
        gray = cv2.resize(gray, (FACE_W, FACE_H))
        feat = self._hog.compute(gray).flatten()
        norm = np.linalg.norm(feat)
        return feat / norm if norm > 0 else None

    @property
    def known_names(self) -> list[str]:
        return self.names[:]
