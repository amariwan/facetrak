import copy
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

CONFIG_PATH = Path("config.json")

DEFAULT_CONFIG = {
    "camera": 0,
    "cameras": [
        {"name": "Built-in", "source": 0},
    ],
    "detect_width": 480,
    "recog_threshold": 0.36,
    "blur_unknown": False,
    "blur_persons": [],
    "heatmap": False,
    "api_port": 8765,
    "servo_target": "largest",
    "notifications": True,
    "servo": {
        "port": "",
        "baud": 9600,
        "pan_min": 0, "pan_max": 180,
        "tilt_min": 0, "tilt_max": 180,
        "dead_zone": 15, "smooth": 0.12, "max_step": 3.0,
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
}


def load() -> dict:
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH) as f:
                cfg = json.load(f)
            for k, v in DEFAULT_CONFIG.items():
                cfg.setdefault(k, copy.deepcopy(v))
            return cfg
        except (json.JSONDecodeError, OSError):
            logger.warning("Could not read %s, falling back to defaults",
                           CONFIG_PATH, exc_info=True)
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    save(cfg)
    return cfg


def save(cfg: dict):
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


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
    src = cams[idx]["source"]
    return int(src) if isinstance(src, int) or src.isdigit() else src
