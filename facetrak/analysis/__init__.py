"""Import-light package init: liveness is pure stdlib; the MediaPipe
face analyzer is optional and degrades gracefully when missing."""
import logging

from .liveness import BLINK_REQUIRED, HEAD_TURN_DEG, LivenessChecker

try:
    from .face import FaceAnalyzer, FaceMetrics, _classify_emotion, _gaze_label
except ImportError:
    logging.getLogger(__name__).warning(
        "MediaPipe unavailable, face analysis disabled")
    FaceAnalyzer = None  # type: ignore[assignment]
    FaceMetrics = None  # type: ignore[assignment]
    _classify_emotion = None  # type: ignore[assignment]
    _gaze_label = None  # type: ignore[assignment]

__all__ = [
    "FaceAnalyzer", "FaceMetrics",
    "LivenessChecker", "BLINK_REQUIRED", "HEAD_TURN_DEG",
]
