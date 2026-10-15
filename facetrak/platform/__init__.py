"""Linux-first platform layer.

Portable core code must never branch on macOS specifics. All
platform differences live here:

- `platform/` — OS/Pi detection, headless detection
- `platform.metrics` — Linux system metrics without third-party deps
- `platform.resources` — soft RAM-limit watcher (log + GC, never kill)
- `platform.shutdown` — SIGTERM/SIGINT handling for systemd/Docker
"""

import os
import platform as _plat
import sys

IS_LINUX = sys.platform.startswith("linux")
IS_MAC = sys.platform == "darwin"
IS_PI = False
if IS_LINUX:
    try:
        with open("/proc/device-tree/model") as f:
            IS_PI = "raspberry pi" in f.read().lower()
    except OSError:
        IS_PI = False


def is_headless() -> bool:
    """True when no display server is available (production default)."""
    if IS_LINUX:
        return not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY")
    return False


def cv_backend() -> int | None:
    """Preferred cv2.VideoCapture backend flag for USB cameras.

    Linux → V4L2 (no AVFoundation probing noise); macOS → AVFoundation;
    other → backend auto-detect (None).
    """
    import cv2

    if IS_LINUX:
        return cv2.CAP_V4L2
    if IS_MAC:
        return cv2.CAP_AVFOUNDATION
    return None


def platform_summary() -> dict:
    return {
        "system": _plat.system(),
        "machine": _plat.machine(),
        "python": _plat.python_version(),
        "is_linux": IS_LINUX,
        "is_mac": IS_MAC,
        "is_pi": IS_PI,
        "headless": is_headless(),
    }
