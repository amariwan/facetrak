"""Camera abstraction layer for FaceTrak.

Supported sources:
  - USBCamera   — USB / built-in webcam (cv2.VideoCapture)
  - PiCamera    — Raspberry Pi Camera Module (picamera2 / libcamera)
  - RTSPCamera  — RTSP IP camera with auto-reconnect
  - ONVIFCamera — ONVIF PTZ IP camera (RTSP stream + ONVIF control)
  - DigitalZoom — Wrapper that auto-zooms any CameraSource towards a face

Use auto_detect() to get a working camera without manual configuration.
"""
import logging

from .base import CameraSource, PTZController
from .usb import USBCamera
from .picamera import PiCamera, _is_pi
from .rtsp import RTSPCamera
from .onvif import ONVIFCamera, ONVIFPTZController
from .zoom import DigitalZoom, ZoomTracker, apply_digital_zoom

__all__ = [
    "CameraSource", "PTZController",
    "USBCamera", "PiCamera", "RTSPCamera",
    "ONVIFCamera", "ONVIFPTZController",
    "DigitalZoom", "ZoomTracker", "apply_digital_zoom",
    "auto_detect", "from_config",
]

logger = logging.getLogger(__name__)


def auto_detect() -> CameraSource:
    """Return the best available camera without explicit configuration.

    Priority:
      1. Pi Camera Module (if picamera2 installed AND Pi hardware detected)
      2. USB camera at index 0
    """
    if _is_pi():
        try:
            cam = PiCamera()
            logger.info("auto_detect: using Pi Camera")
            return cam
        except Exception as exc:
            logger.warning("auto_detect: Pi Camera failed (%s), trying USB", exc)

    indices = USBCamera.probe()
    if indices:
        logger.info("auto_detect: using USB camera at index %d", indices[0])
        return USBCamera(indices[0])

    raise RuntimeError(
        "No camera found. Connect a USB camera or enable Pi Camera Module."
    )


def from_config(cfg: dict) -> list[CameraSource]:
    """Build a list of CameraSource objects from a FaceTrak config dict.

    Supports the extended 'cameras' list format:
      {"type": "usb",   "source": 0, "name": "Main"}
      {"type": "rtsp",  "url": "rtsp://...", "name": "Door"}
      {"type": "onvif", "rtsp_url": "rtsp://...", "host": "...",
                        "port": 80, "user": "admin", "password": "..."}
      {"type": "pi",    "index": 0, "width": 1280, "height": 720}

    Falls back to legacy single 'camera' index if 'cameras' key is absent.
    """
    entries = cfg.get("cameras", [])

    if not entries:
        src = cfg.get("camera", 0)
        if isinstance(src, str) and "://" in src:
            return [RTSPCamera(src)]
        return [USBCamera(int(src))]

    sources: list[CameraSource] = []
    for entry in entries:
        cam_type = entry.get("type", "usb")
        name     = entry.get("name")

        if cam_type == "usb":
            sources.append(USBCamera(int(entry.get("source", 0)), name=name))

        elif cam_type == "pi":
            sources.append(PiCamera(
                index=int(entry.get("index", 0)),
                width=int(entry.get("width", 1280)),
                height=int(entry.get("height", 720)),
                fps=int(entry.get("fps", 30)),
                name=name or "Pi Camera",
            ))

        elif cam_type == "rtsp":
            sources.append(RTSPCamera(entry["url"], name=name))

        elif cam_type == "onvif":
            sources.append(ONVIFCamera(
                rtsp_url=entry["rtsp_url"],
                host=entry["host"],
                port=int(entry.get("port", 80)),
                user=entry.get("user", "admin"),
                password=entry.get("password", ""),
                name=name or "ONVIF Camera",
            ))

        else:
            logger.warning("Unknown camera type %r — skipping", cam_type)

    if not sources:
        raise RuntimeError("No valid camera entries found in config.")

    return sources
