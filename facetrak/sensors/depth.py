"""Monocular depth estimation via MiDaS (OpenCV DNN backend).

Downloads the MiDaS small model (~50 MB) on first use.
Produces a normalised depth map and can annotate frames with:
  - colourised depth overlay
  - per-face estimated distance label

Model options (auto-selected by hardware):
  MiDaS_small   — fast, good for Pi 4 / Apple Silicon
  dpt_levit_224 — higher accuracy, requires more RAM/CPU

Install: pip install torch torchvision  (for DPT models)
        or use OpenCV DNN version (no torch needed for MiDaS_small).
"""
from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_CACHE_DIR = Path.home() / ".cache" / "facetrak" / "depth"

# MiDaS small — ONNX export, works with OpenCV DNN (no PyTorch needed)
_MODEL_URL  = (
    "https://github.com/isl-org/MiDaS/releases/download/v2_1/"
    "model-small.onnx"
)
_MODEL_FILE = "midas_small.onnx"
_INPUT_SIZE = (256, 256)   # model input resolution

_OVERLAY_ALPHA = 0.45


class DepthEstimator:
    """Per-frame monocular depth estimation.

    Usage:
        de = DepthEstimator()
        de.load()
        depth_map = de.estimate(bgr_frame)   # float32 [0, 1], 0=near
        de.draw_overlay(bgr_frame, depth_map)
        dist = de.face_distance(depth_map, x, y, w, h)
    """

    def __init__(self, model_path: str | None = None):
        self._model_path = model_path
        self._net = None
        self._enabled = False

    def load(self) -> bool:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        model_file = Path(self._model_path) if self._model_path else _CACHE_DIR / _MODEL_FILE

        if not model_file.exists():
            logger.info("Downloading MiDaS small model (~50 MB)…")
            try:
                urllib.request.urlretrieve(_MODEL_URL, str(model_file))
                logger.info("MiDaS model saved to %s", model_file)
            except Exception as exc:
                logger.error("MiDaS download failed: %s", exc)
                return False

        try:
            self._net = cv2.dnn.readNetFromONNX(str(model_file))
            # prefer GPU when available
            self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
            self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            self._enabled = True
            logger.info("DepthEstimator loaded: %s", model_file.name)
            return True
        except Exception as exc:
            logger.error("DepthEstimator load failed: %s", exc)
            return False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def estimate(self, bgr_frame: np.ndarray) -> np.ndarray | None:
        """Return normalised depth map (float32, same H×W as input).

        0.0 = closest, 1.0 = farthest.
        Returns None if not enabled.
        """
        if not self._enabled or self._net is None:
            return None
        try:
            blob = cv2.dnn.blobFromImage(
                bgr_frame, scalefactor=1.0 / 255.0,
                size=_INPUT_SIZE,
                mean=(0.485, 0.456, 0.406),
                swapRB=True, crop=False,
            )
            # normalize by std as MiDaS expects
            blob[0, 0] /= 0.229
            blob[0, 1] /= 0.224
            blob[0, 2] /= 0.225

            self._net.setInput(blob)
            raw = self._net.forward()          # shape: (1, 1, H, W) or (1, H, W)
            depth = raw.squeeze()

            # clip extreme edge artifacts before normalization
            low, high = np.percentile(depth, [1, 99])
            depth = np.clip(depth, low, high)

            # invert: MiDaS output is inverse depth (larger = closer)
            depth = 1.0 - (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

            # resize back to frame size
            h, w = bgr_frame.shape[:2]
            depth = cv2.resize(depth.astype(np.float32), (w, h),
                               interpolation=cv2.INTER_LINEAR)
            return depth
        except Exception as exc:
            logger.warning("DepthEstimator inference error: %s", exc)
            return None

    def draw_overlay(self, frame: np.ndarray,
                     depth_map: np.ndarray) -> np.ndarray:
        """Blend a colourised depth map onto frame (in-place). Returns frame."""
        coloured = cv2.applyColorMap(
            (depth_map * 255).astype(np.uint8), cv2.COLORMAP_MAGMA
        )
        cv2.addWeighted(coloured, _OVERLAY_ALPHA,
                        frame, 1.0 - _OVERLAY_ALPHA, 0, frame)
        return frame

    def face_distance(self, depth_map: np.ndarray,
                      x: int, y: int, w: int, h: int) -> float:
        """Return mean depth value [0, 1] in the face bounding box.

        0 = very close, 1 = far away.
        """
        fh, fw = depth_map.shape[:2]
        x1 = max(0, x);          y1 = max(0, y)
        x2 = min(fw, x + w);     y2 = min(fh, y + h)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        roi = depth_map[y1:y2, x1:x2]
        return float(roi.mean())

    def draw_face_distances(self, frame: np.ndarray,
                            depth_map: np.ndarray,
                            faces: list[tuple[int, int, int, int]]) -> np.ndarray:
        """Label each face bbox with its estimated depth. Returns frame."""
        for x, y, w, h in faces:
            dist = self.face_distance(depth_map, x, y, w, h)
            label = f"d:{dist:.2f}"
            cv2.putText(frame, label, (x, y - 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 100, 255), 1)
        return frame
