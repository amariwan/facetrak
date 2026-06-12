"""ONVIF PTZ IP camera support via onvif-zeep.

Combines RTSPCamera (for frames) with ONVIF protocol (for PTZ control).
onvif-zeep is optional — only needed when ONVIF cameras are configured.

Install: pip install onvif-zeep
"""
import logging

import numpy as np

from .base import PTZController
from .rtsp import RTSPCamera

logger = logging.getLogger(__name__)


class ONVIFPTZController(PTZController):
    """Controls pan/tilt/zoom on an ONVIF-compatible IP camera."""

    def __init__(self, host: str, port: int, user: str, password: str):
        try:
            from onvif import ONVIFCamera as _ONVIFCamera  # type: ignore
        except ImportError as e:
            raise ImportError(
                "onvif-zeep is required for ONVIF PTZ. "
                "Install with: pip install onvif-zeep"
            ) from e

        self._cam = _ONVIFCamera(host, port, user, password)
        self._ptz = self._cam.create_ptz_service()
        self._media = self._cam.create_media_service()
        profiles = self._media.GetProfiles()
        if not profiles:
            raise RuntimeError("ONVIF camera returned no media profiles")
        profile = profiles[0]
        self._token = profile.token
        self._req_move = self._ptz.create_type("ContinuousMove")
        self._req_move.ProfileToken = self._token
        logger.info("ONVIF PTZ connected: %s:%d", host, port)

    def move(self, pan: float, tilt: float) -> None:
        pan   = max(-1.0, min(1.0, pan))
        tilt  = max(-1.0, min(1.0, tilt))
        self._req_move.Velocity = {
            "PanTilt": {"x": pan, "y": tilt},
            "Zoom":    {"x": 0},
        }
        try:
            self._ptz.ContinuousMove(self._req_move)
        except Exception as exc:
            logger.warning("ONVIF move error: %s", exc)

    def zoom(self, level: float) -> None:
        level = max(0.0, min(1.0, level))
        req = self._ptz.create_type("AbsoluteMove")
        req.ProfileToken = self._token
        req.Position = {"Zoom": {"x": level}}
        try:
            self._ptz.AbsoluteMove(req)
        except Exception as exc:
            logger.warning("ONVIF zoom error: %s", exc)

    def stop(self) -> None:
        try:
            req = self._ptz.create_type("Stop")
            req.ProfileToken = self._token
            req.PanTilt = True
            req.Zoom    = True
            self._ptz.Stop(req)
        except Exception as exc:
            logger.warning("ONVIF stop error: %s", exc)


class ONVIFCamera(RTSPCamera):
    """ONVIF IP camera: RTSP stream for frames + ONVIF for PTZ control.

    Usage:
        cam = ONVIFCamera(rtsp_url="rtsp://...", host="192.168.1.20",
                          port=80, user="admin", password="pass")
        ptz = cam.ptz   # ONVIFPTZController
    """

    def __init__(self, rtsp_url: str, host: str, port: int,
                 user: str, password: str, name: str = "ONVIF Camera",
                 width: int | None = None, height: int | None = None):
        super().__init__(url=rtsp_url, name=name, width=width, height=height)
        self._ptz_ctrl = ONVIFPTZController(host, port, user, password)

    @property
    def ptz(self) -> ONVIFPTZController:
        return self._ptz_ctrl
