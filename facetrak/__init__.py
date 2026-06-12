"""FaceTrak — Real-time face detection, recognition and tracking."""

from .core.engine import FaceEngine
from .core.config import load as load_config

__all__ = ["FaceEngine", "load_config"]
