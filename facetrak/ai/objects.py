"""YOLO object detection via Ultralytics YOLOv8.

Automatically selects model size based on available hardware:
  - Pi / low-end  → yolov8n  (nano,  3 MB)
  - Mid-range     → yolov8s  (small, 11 MB)
  - Mac / GPU     → yolov8m  (medium, 25 MB)

Models are downloaded on first use into ~/.cache/facetrak/yolo/.
Install: pip install ultralytics
"""
from __future__ import annotations

import logging
import platform
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_CACHE_DIR = Path.home() / ".cache" / "facetrak" / "yolo"

# Confidence threshold — below this detections are discarded
_DEFAULT_CONF = 0.45
# Classes to ignore (person=0 already tracked by face pipeline; optional)
_SKIP_CLASSES: frozenset[int] = frozenset()


@dataclass
class Detection:
    label: str
    confidence: float
    x: int
    y: int
    w: int
    h: int
    class_id: int

    @property
    def center(self) -> tuple[int, int]:
        return self.x + self.w // 2, self.y + self.h // 2

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h


def _auto_model_size() -> str:
    """Pick YOLO model size based on hardware."""
    machine = platform.machine().lower()
    # ARM = Raspberry Pi or Apple Silicon
    if machine.startswith("arm") or machine == "aarch64":
        try:
            with open("/proc/device-tree/model") as f:
                if "Raspberry Pi" in f.read():
                    return "yolov8n"
        except OSError:
            pass
        # Apple Silicon — medium is fine
        return "yolov8s"
    return "yolov8m"


class ObjectDetector:
    """Wraps YOLOv8 for frame-level object detection.

    Usage:
        det = ObjectDetector()
        det.load()
        objects = det.detect(frame)
    """

    def __init__(self, model_size: str | None = None,
                 conf: float = _DEFAULT_CONF,
                 skip_classes: frozenset[int] = _SKIP_CLASSES):
        self._model_size = model_size or _auto_model_size()
        self._conf = conf
        self._skip = skip_classes
        self._model = None
        self._enabled = False

    def load(self, model_size: str | None = None) -> bool:
        """Load (and if needed download) the YOLO model. Returns True on success."""
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError:
            logger.warning(
                "ultralytics not installed — object detection disabled. "
                "Install with: pip install ultralytics"
            )
            return False

        size = model_size or self._model_size
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        model_path = _CACHE_DIR / f"{size}.pt"
        try:
            if not model_path.exists():
                self._model = YOLO(f"{size}.pt")
                import shutil
                cached = Path.home() / ".cache" / "ultralytics" / "models" / f"{size}.pt"
                if cached.exists():
                    shutil.copy2(str(cached), str(model_path))
            else:
                self._model = YOLO(str(model_path))
            self._enabled = True
            logger.info("ObjectDetector loaded: %s", size)
            return True
        except Exception as exc:
            logger.error("ObjectDetector load failed: %s", exc)
            return False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run inference on a BGR frame. Returns list of Detection objects."""
        if not self._enabled or self._model is None:
            return []
        try:
            results = self._model(frame, conf=self._conf, verbose=False)
        except Exception as exc:
            logger.warning("YOLO inference error: %s", exc)
            return []

        detections: list[Detection] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id in self._skip:
                    continue
                x1, y1, x2, y2 = (int(round(v)) for v in box.xyxy[0])
                detections.append(Detection(
                    label=r.names[cls_id],
                    confidence=float(box.conf[0]),
                    x=x1, y=y1,
                    w=x2 - x1, h=y2 - y1,
                    class_id=cls_id,
                ))
        return detections

    def draw(self, frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        """Draw bounding boxes and labels onto frame (in-place). Returns frame."""
        import cv2
        for d in detections:
            color = (0, 200, 255)
            cv2.rectangle(frame, (d.x, d.y), (d.x + d.w, d.y + d.h), color, 2)
            label = f"{d.label} {d.confidence:.0%}"
            cv2.putText(frame, label, (d.x, d.y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        return frame
