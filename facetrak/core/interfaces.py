"""Plugin interfaces for the modular vision pipeline.

Detector / Tracker / InferenceBackend are Protocols: any object with the
matching methods plugs into FaceEngine without inheritance. The bundled
YuNetDetector + FaceTracker already satisfy these protocols; alternative
backends (e.g. boxmot-style trackers, ONNX-Runtime detectors) only need
to implement the same signatures.
"""

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Detector(Protocol):
    @property
    def ready(self) -> bool:
        """False when the model failed to load (pipeline degrades)."""
        ...

    def detect(self, frame: np.ndarray, scale: float = 1.0) -> list:
        """Return detections (FaceDetection or compatible objects)."""
        ...


@runtime_checkable
class Tracker(Protocol):
    def update(self, detections: list) -> tuple[list, list]:
        """Return (active_tracks, ended_tracks)."""
        ...

    def reset(self) -> None: ...


@runtime_checkable
class InferenceBackend(Protocol):
    @property
    def name(self) -> str: ...

    def load(self) -> bool:
        """Load model weights. Returns False when unavailable."""
        ...

    def release(self) -> None: ...
