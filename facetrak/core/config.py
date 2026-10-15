import copy
import json
import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(os.environ.get("FACETRAK_CONFIG", "config.json"))

DEFAULT_CONFIG = {
    "camera": 0,
    "cameras": [
        {"name": "Built-in", "source": 0},
    ],
    "detect_width": 480,
    "detect_every": 1,
    "recog_threshold": 0.36,
    "blur_unknown": False,
    "blur_persons": [],
    "heatmap": False,
    "api_port": 8765,
    "api": {
        # Secure default: localhost only. Set to "0.0.0.0" deliberately for LAN.
        "host": "127.0.0.1",
        # Bearer token. Prefer env FACETRAK_API_TOKEN over storing here.
        "auth_token": "",
    },
    "servo_target": "largest",
    "notifications": True,
    "data_dir": "",
    # Days to keep presence/emotion/crowd/event rows. 0 = keep forever.
    "retention_days": 30,
    "privacy": {
        # Master switch: when True, age/gender/emotion/gaze estimates are
        # still computed but never exposed via API/MCP/dashboard outputs.
        "hide_estimates": False,
        # Only emit identifying data (names/embeddings) to integrations
        # when explicitly enabled with consent.
        "allow_identifying_exports": False,
    },
    "zones": [],
    "privacy_masks": [],
    "events": {
        "enabled": True,
        "max_events": 1000,
    },
    "mqtt": {
        "enabled": False,
        "host": "127.0.0.1",
        "port": 1883,
        "topic_prefix": "facetrak",
        "username": "",
        "password": "",
        "tls": False,
        "include_names": False,
    },
    "go2rtc": {
        "enabled": False,
        "url": "http://127.0.0.1:1984",
    },
    "servo": {
        "port": "",
        "baud": 115200,
        "pan_min": 0, "pan_max": 180,
        "tilt_min": 0, "tilt_max": 180,
        "dead_zone": 15, "smooth": 0.12, "max_step": 3.0,
        # Max angular velocity in degrees/second (acceleration/velocity cap).
        "max_speed_dps": 90.0,
        # Frames without a target before the head returns to center.
        "reacquire_frames": 30,
        # Static calibration offsets applied after clamping.
        "pan_offset": 0.0, "tilt_offset": 0.0,
        "invert_pan": False, "invert_tilt": False,
    },
    "zoom": {
        "enabled": False,
        "target_ratio": 0.3,
        "hysteresis": 0.05,
    },
    "objects_enabled": False,
    "pose_enabled": False,
    "gestures_enabled": False,
    "audio_enabled": False,
    "pir_enabled": False,
    "pir_gpio_pin": 17,
    "depth_enabled": False,
    "mirror": False,
}


def _deep_merge(cfg: dict, defaults: dict) -> dict:
    for k, v in defaults.items():
        if k not in cfg:
            cfg[k] = copy.deepcopy(v)
        elif isinstance(v, dict) and isinstance(cfg[k], dict):
            _deep_merge(cfg[k], v)
    return cfg


def load() -> dict:
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH) as f:
                cfg = json.load(f)
            return _deep_merge(cfg, DEFAULT_CONFIG)
        except (json.JSONDecodeError, OSError):
            logger.warning("Could not read %s, falling back to defaults",
                           CONFIG_PATH, exc_info=True)
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    save(cfg)
    return cfg


def save(cfg: dict):
    # Atomic write (tmp + rename) so concurrent UI/API saves cannot truncate.
    fd, tmp = tempfile.mkstemp(dir=str(CONFIG_PATH.parent),
                               prefix=".config.", suffix=".tmp")
    try:
        with open(fd, "w") as f:
            json.dump(cfg, f, indent=2)
        os.replace(tmp, CONFIG_PATH)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def label(cfg: dict, idx: int) -> str:
    cams = cfg.get("cameras", [])
    if idx < 0 or idx >= len(cams):
        return f"Camera {idx}"
    c = cams[idx]
    return f"{c['name']} ({c['source']})"


def source(cfg: dict, idx: int):
    cams = cfg.get("cameras", [])
    if idx < 0 or idx >= len(cams):
        return idx
    src = cams[idx].get("source", idx)
    if isinstance(src, int) or isinstance(src, bool):
        return src
    if isinstance(src, str):
        return int(src) if src.isdigit() else src
    return idx


def redact(cfg: dict) -> dict:
    """Return a copy safe for logs/API: credentials in URLs stripped."""
    from facetrak.core.security import redact_url

    out = copy.deepcopy(cfg)
    for cam in out.get("cameras", []):
        for key in ("source", "url", "rtsp_url"):
            val = cam.get(key)
            if isinstance(val, str) and "://" in val:
                cam[key] = redact_url(val)
    mqtt = out.get("mqtt")
    if isinstance(mqtt, dict) and mqtt.get("password"):
        mqtt["password"] = "***"
    for cam in out.get("cameras", []):
        if cam.get("password"):
            cam["password"] = "***"
    return out
