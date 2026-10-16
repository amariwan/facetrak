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
    # Adaptive inference: raise detect_every (1..4) when frames are slow,
    # lower it when headroom returns. Bounded, self-neutralizing on fast HW.
    "detect_adaptive": True,
    "detect_target_ms": 33.0,
    "resources": {
        # Soft RSS cap in MB, 0 = off. Warns + GC only, never kills.
        "max_memory_mb": 0,
    },
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
    # ── FaceTrak 1.0 platform defaults (all optional, safe) ──
    "tracking": {
        "classes": ["face", "person"], "conf_threshold": 0.4,
        "iou_threshold": 0.3, "max_age": 30, "min_hits": 3,
        "prediction": True, "algorithm": "iou_kalman",
    },
    "segmentation": {"enabled": False, "backend": "cpu-fallback",
                     "downscale": 0.5},
    "openvocab": {"enabled": False, "model": "none", "budget_ms": 200.0,
                  "prompts": []},
    "depth_tracking": {"enabled": False, "weight": 0.3},
    "handover": {"enabled": False, "relations": []},
    "inference": {"backend": "opencv-dnn"},
    "benchmark": {"warmup": 5, "iters": 30},
    "hailo": {"enabled": False, "model": ""},
    "imx500": {"enabled": False, "model": ""},
    "servo_cal": {"min": 10.0, "max": 170.0, "profile": {},
                  "simulate": True},
    "servo_predict": {"max_vel_dps": 90.0, "max_accel_dps2": 360.0,
                      "smooth": 0.12, "dead_zone": 15.0, "lost_limit": 30},
    "axes": {"list": [{"name": "pan", "min": 0, "max": 180},
                       {"name": "tilt", "min": 0, "max": 180}]},
    "watchdog": {"enabled": True, "max_retries": 3, "stale_s": 5.0},
    "thermal": {"warn_c": 70.0, "throttle_c": 80.0},
    "power": {"profile": "off", "idle_s": 60.0},
    "gpio": {"enabled": False, "pins": [17], "debounce_ms": 200,
             "simulate": True},
    "camera_profiles": {"active": "balanced"},
    "recording": {"enabled": False, "on_events": ["zone_enter"]},
    "event_buffer": {"pre_s": 5.0, "post_s": 5.0, "fps": 10.0},
    "privacy_editor": [],
    "failover": {"enabled": False, "fallback": None, "fail_after": 5,
                 "recover_after": 10},
    "storage": {"quota_mb": 1024, "retention_days": 7},
    "rules": {"list": [], "max_actions_per_min": 20, "dry_run": False},
    "webhooks": {"enabled": False, "targets": [], "secret": "",
                 "retries": 3, "allow_hosts": []},
    "gesture_actions": {"enabled": False, "bindings": {},
                        "min_conf": 0.8, "cooldown_s": 5.0},
    "rbac": {"enabled": False, "users": {}},

    # ── InsightFace Integration (optional, opt-in) ──
    "insightface": {
        "enabled": False,
        "provider": "CPUExecutionProvider",
        "model_pack": "buffalo_l",
        "models_dir": "models/insightface",
        "auto_download": False,
        "require_license_consent": True,
    },

    # ── Detector/Recognizer Backend Selection ──
    "detector_backend": "yunet",
    "recognition_backend": "sface",
    "liveness_backend": "mediapipe",

    # ── SCRFD Detector Config ──
    "scrfd": {
        "model": "det_10g",
        "input_size": [640, 640],
        "score_threshold": 0.5,
        "nms_threshold": 0.4,
        "max_num": 20,
        "use_kps": True,
    },

    # ── ArcFace Recognition Config ──
    "arcface": {
        "model": "w600k_r50",
        "input_size": [112, 112],
        "embedding_dim": 512,
        "norm_embedding": True,
    },

    # ── Anonymization (PrivateFrame-inspired) ──
    "anonymization": {
        "enabled": False,
        "detector_backend": "yunet",
        "recognition_backend": "none",
        "method": "gaussian",
        "box_scale": 1.15,
        "gaussian_ksize": 51,
        "gaussian_sigma": 30.0,
        "mosaic_block": 20,
        "detect_every": 1,
        "max_faces": 50,
        "min_face_size": 20,
    },

    # ── Model Registry ──
    "model_registry": {
        "models_dir": "models",
        "auto_download": False,
        "require_license_consent": True,
        "offline_models_dir": "models/offline",
    },
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