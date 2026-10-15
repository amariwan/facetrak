"""go2rtc as an optional external streaming service.

FaceTrak does not vendor or require go2rtc. When enabled in config, this
helper builds stream URLs (WebRTC/MJPEG/HLS) for the local dashboard and
checks go2rtc health. Single-connection discipline stays in FaceTrak: the
engine opens each physical camera once; clients watch via go2rtc or the
built-in MJPEG preview, never via extra direct camera connections.
"""

import logging
import urllib.request

logger = logging.getLogger(__name__)


class Go2Rtc:
    def __init__(self, cfg: dict):
        g = cfg.get("go2rtc", {})
        self.enabled = bool(g.get("enabled", False))
        self.base = str(g.get("url", "http://127.0.0.1:1984")).rstrip("/")

    def stream_url(self, name: str, kind: str = "webrtc") -> str | None:
        if not self.enabled:
            return None
        paths = {
            "webrtc": f"{self.base}/stream.html?src={name}",
            "mse": f"{self.base}/api/ws?src={name}",
            "mjpeg": f"{self.base}/api/frame.jpeg?src={name}",
            "hls": f"{self.base}/api/stream.m3u8?src={name}",
        }
        return paths.get(kind)

    def healthy(self, timeout: float = 2.0) -> bool:
        if not self.enabled:
            return False
        try:
            with urllib.request.urlopen(f"{self.base}/api", timeout=timeout) as resp:
                return resp.status == 200
        except Exception as exc:
            logger.debug("go2rtc health check failed: %s", exc)
            return False


def go2rtc_config_summary(cfg: dict) -> dict:
    g = Go2Rtc(cfg)
    return {
        "enabled": g.enabled,
        "url": g.base if g.enabled else None,
        "healthy": g.healthy() if g.enabled else False,
    }
