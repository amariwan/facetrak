import json
from pathlib import Path

CONFIG_PATH = Path("config.json")

DEFAULT_CONFIG = {
    "camera": 0,
    "cameras": [
        {"name": "Built-in", "source": 0},
    ],
    "detect_width": 480,
    "recog_threshold": 0.55,
    "blur_unknown": False,
    "servo": {
        "port": "",
        "baud": 9600,
        "pan_min": 0, "pan_max": 180,
        "tilt_min": 0, "tilt_max": 180,
        "dead_zone": 15, "smooth": 0.12, "max_step": 3.0,
        "invert_pan": False, "invert_tilt": False,
    },
}


def load() -> dict:
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH) as f:
                cfg = json.load(f)
            for k, v in DEFAULT_CONFIG.items():
                cfg.setdefault(k, v)
            save(cfg)
            return cfg
        except Exception:
            pass
    save(DEFAULT_CONFIG)
    return dict(DEFAULT_CONFIG)


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
