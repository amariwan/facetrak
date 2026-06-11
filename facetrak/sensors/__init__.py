"""Sensor-fusion modules for FaceTrak.

  - AudioMonitor    — microphone events (LOUD, CLAP, VOICE, SILENCE)
  - PIRSensor       — passive infrared motion sensor via GPIO (Pi only)
  - DepthEstimator  — monocular depth from single camera frame (MiDaS ONNX)
"""
from .audio import AudioMonitor, AudioEvent, AudioSample
from .pir import PIRSensor, MotionEvent
from .depth import DepthEstimator

__all__ = [
    "AudioMonitor", "AudioEvent", "AudioSample",
    "PIRSensor", "MotionEvent",
    "DepthEstimator",
]
