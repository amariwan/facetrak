"""F13 — Auto hardware discovery. F17 — watchdog. F18 — thermal
governor. F19 — low-power mode. F20 — GPIO event bridge.
F14 — servo guided calibration. F15 — predictive servo.
F16 — multi-axis servo support.
F21 — Model Registry: manifest, checksums, licensing, versioning.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── F13 ───────────────────────────────────────────────────────────
def discover_hardware(cfg: dict | None = None) -> dict:
    """Enumerate cameras, GPIO, serial, accelerators, runtimes."""
    cfg = cfg or {}
    out: dict = {"arch": platform.machine(), "system": platform.system()}
    # USB/V4L2 probe (best-effort, no crash when OpenCV missing camera).
    usb: list[int] = []
    try:
        import cv2

        for i in range(6):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                usb.append(i)
                cap.release()
            else:
                cap.release()
    except Exception:
        pass
    out["usb_cameras"] = usb
    try:
        with open("/proc/device-tree/model") as f:
            out["board"] = f.read().strip("\x00\n")
    except OSError:
        out["board"] = "unknown"
    # GPIO capability: RPi.GPIO importable on Linux.
    try:
        import RPi.GPIO  # noqa: F401

        out["gpio"] = True
    except ImportError:
        out["gpio"] = False
    # Serial ports.
    try:
        import serial.tools.list_ports

        out["serial"] = [p.device for p in serial.tools.list_ports.comports()]
    except Exception:
        out["serial"] = []
    # Runtimes.
    runtimes: dict[str, bool] = {}
    for mod in (
        "onnxruntime",
        "tflite_runtime",
        "ai_edge_litert",
        "hailo_platform",
        "picamera2",
    ):
        try:
            __import__(mod)
            runtimes[mod] = True
        except ImportError:
            runtimes[mod] = False
    out["runtimes"] = runtimes
    out["configured_cameras"] = cfg.get("cameras", [])
    return out


# ── F14 guided calibration ────────────────────────────────────────
class ServoCalibrator:
    """Guided calibration: proposes bounded moves, requires manual
    confirmation per step, persists profile, supports reset + sim."""

    def __init__(self, config: dict | None = None):
        cfg = (config or {}).get("servo_cal", config or {})
        self.bounds = (float(cfg.get("min", 10.0)), float(cfg.get("max", 170.0)))
        self.profile: dict = dict(cfg.get("profile", {}))
        self.simulate = bool(cfg.get("simulate", False))
        self._estop = False

    def propose(self, axis: str = "pan") -> dict:
        lo, hi = self.bounds
        return {
            "axis": axis,
            "steps": [90.0, lo, hi, 90.0],
            "requires_confirmation": True,
            "simulate": self.simulate,
        }

    def confirm_step(self, axis: str, angle: float, confirmed: bool) -> dict:
        if self._estop:
            return {"ok": False, "error": "emergency stop engaged"}
        if not confirmed:
            return {"ok": False, "error": "confirmation required"}
        lo, hi = self.bounds
        if not lo <= angle <= hi:
            return {"ok": False, "error": f"angle {angle} outside {self.bounds}"}
        if not self.simulate:
            # Real move happens via controller; here we only record intent.
            pass
        self.profile[axis] = angle
        return {"ok": True, "axis": axis, "angle": angle, "simulate": self.simulate}

    def reset(self) -> dict:
        self.profile = {}
        self._estop = False
        return {"ok": True}

    def estop(self) -> dict:
        self._estop = True
        return {"ok": True, "stopped": True}


# ── F15 predictive servo ──────────────────────────────────────────
class PredictiveServo:
    """Bounded controller: velocity/accel limits, smoothing, dead zone,
    target-loss handling, motion prediction."""

    def __init__(self, config: dict | None = None):
        cfg = (config or {}).get("servo_predict", config or {})
        self.max_vel = float(cfg.get("max_vel_dps", 90.0))
        self.max_accel = float(cfg.get("max_accel_dps2", 360.0))
        self.smooth = float(cfg.get("smooth", 0.12))
        self.dead_zone = float(cfg.get("dead_zone", 15.0))
        self.lost_limit = int(cfg.get("lost_limit", 30))
        self.pan = 90.0
        self._vel = 0.0
        self._lost = 0

    def update(self, dx: float, dt: float = 1 / 30, vx_hint: float = 0.0) -> dict:
        if abs(dx) < self.dead_zone:
            self._lost = 0
            return {"pan": round(self.pan, 2), "moved": False}
        self._lost = 0
        # Predict target with velocity hint, then bounded approach.
        target = 90.0 + max(-60.0, min(60.0, dx * 0.2 + vx_hint * dt))
        desired_vel = (target - self.pan) * self.smooth / max(dt, 1e-3)
        desired_vel = max(-self.max_vel, min(self.max_vel, desired_vel))
        dv = desired_vel - self._vel
        dv = max(-self.max_accel * dt, min(self.max_accel * dt, dv))
        self._vel += dv
        self.pan = max(0.0, min(180.0, self.pan + self._vel * dt))
        return {
            "pan": round(self.pan, 2),
            "moved": True,
            "vel_dps": round(self._vel, 1),
        }

    def target_lost(self) -> dict:
        self._lost += 1
        if self._lost >= self.lost_limit:
            self.pan += (90.0 - self.pan) * self.smooth
            return {"pan": round(self.pan, 2), "recentering": True}
        return {"pan": round(self.pan, 2), "recentering": False}


# ── F16 multi-axis ────────────────────────────────────────────────
class MultiAxisController:
    """Generalized N-axis control; pan/tilt preserved as axes 0/1."""

    def __init__(self, config: dict | None = None):
        cfg = (config or {}).get("axes", config or {})
        axes_cfg = cfg.get(
            "list",
            [
                {"name": "pan", "min": 0, "max": 180},
                {"name": "tilt", "min": 0, "max": 180},
            ],
        )
        self.axes: dict[str, dict] = {}
        for a in axes_cfg:
            self.axes[a["name"]] = {
                "pos": 90.0,
                "min": float(a.get("min", 0)),
                "max": float(a.get("max", 180)),
            }

    def move(self, name: str, angle: float) -> dict:
        a = self.axes.get(name)
        if not a:
            return {"ok": False, "error": f"unknown axis {name!r}"}
        a["pos"] = max(a["min"], min(a["max"], float(angle)))
        return {"ok": True, "axis": name, "pos": a["pos"]}

    def state(self) -> dict:
        return {k: v["pos"] for k, v in self.axes.items()}


# ── F17 watchdog ──────────────────────────────────────────────────
class HardwareWatchdog:
    """Detect unresponsive cameras/pipelines/devices/servos; bounded
    recovery attempts + health reporting."""

    def __init__(self, config: dict | None = None):
        cfg = (config or {}).get("watchdog", config or {})
        self.enabled = bool(cfg.get("enabled", True))
        self.max_retries = int(cfg.get("max_retries", 3))
        self.stale_s = float(cfg.get("stale_s", 5.0))
        self._last_frame: dict[str, float] = {}
        self._retries: dict[str, int] = {}
        self.health: dict[str, str] = {}

    def heartbeat(self, source: str) -> None:
        self._last_frame[source] = time.time()
        self.health[source] = "ok"

    def check(self, source: str, probe_ok: bool = True) -> dict:
        if not self.enabled:
            return {"source": source, "status": "disabled"}
        last = self._last_frame.get(source, 0.0)
        stale = (time.time() - last) > self.stale_s
        if probe_ok and not stale:
            self._retries[source] = 0
            self.health[source] = "ok"
            return {"source": source, "status": "ok"}
        n = self._retries.get(source, 0)
        if n >= self.max_retries:
            self.health[source] = "failed"
            return {
                "source": source,
                "status": "failed",
                "retries": n,
                "action": "operator attention required",
            }
        self._retries[source] = n + 1
        self.health[source] = "recovering"
        return {
            "source": source,
            "status": "recovering",
            "retries": n + 1,
            "action": "restart source",
        }


# ── F18 thermal governor ──────────────────────────────────────────
class ThermalGovernor:
    """Linux thermal monitoring with staged, reversible policies.
    Faults are reported, never hidden."""

    def __init__(self, config: dict | None = None):
        cfg = (config or {}).get("thermal", config or {})
        self.warn_c = float(cfg.get("warn_c", 70.0))
        self.throttle_c = float(cfg.get("throttle_c", 80.0))
        self.state = "normal"

    def read_c(self) -> float | None:
        for zone in ("/sys/class/thermal/thermal_zone0/temp",):
            try:
                with open(zone) as f:
                    return float(f.read().strip()) / 1000.0
            except OSError:
                pass
        return None

    def evaluate(self, temp_c: float | None = None) -> dict:
        t = self.read_c() if temp_c is None else temp_c
        if t is None:
            return {
                "state": "unknown",
                "temp_c": None,
                "fault": "no thermal sensor visible",
            }
        prev = self.state
        if t >= self.throttle_c:
            self.state = "throttle"
        elif t >= self.warn_c:
            self.state = "warn"
        elif t < self.warn_c - 5:
            self.state = "normal"  # gradual recovery hysteresis
        policy = {
            "normal": {"infer_every": 1, "resolution": 1.0, "optional_models": True},
            "warn": {"infer_every": 2, "resolution": 0.75, "optional_models": True},
            "throttle": {"infer_every": 4, "resolution": 0.5, "optional_models": False},
        }[self.state]
        return {
            "state": self.state,
            "prev": prev,
            "temp_c": t,
            "policy": policy,
            "fault": None
            if self.state == "normal"
            else f"thermal {self.state} at {t}C",
        }


# ── F19 low-power ─────────────────────────────────────────────────
class LowPowerMode:
    PROFILES = ("off", "eco", "deep")

    def __init__(self, config: dict | None = None):
        cfg = (config or {}).get("power", config or {})
        self.profile = str(cfg.get("profile", "off"))
        if self.profile not in self.PROFILES:
            self.profile = "off"
        self.idle_s = float(cfg.get("idle_s", 60.0))
        self._last_activity = time.time()

    def activity(self) -> None:
        self._last_activity = time.time()

    def set_profile(self, name: str) -> dict:
        if name not in self.PROFILES:
            return {"ok": False, "error": f"unknown profile {name!r}"}
        self.profile = name
        return {"ok": True, "profile": name}

    def tick(self) -> dict:
        idle = time.time() - self._last_activity
        active_idle = idle >= self.idle_s
        params = {
            "off": {"infer_every": 1, "optional_models": True},
            "eco": {
                "infer_every": 2 if active_idle else 1,
                "optional_models": not active_idle or True,
            },
            "deep": {
                "infer_every": 6 if active_idle else 2,
                "optional_models": not active_idle,
            },
        }[self.profile]
        return {
            "profile": self.profile,
            "idle_s": round(idle, 1),
            "idle": active_idle,
            **params,
        }


# ── F20 GPIO bridge ───────────────────────────────────────────────
class GPIOEventBridge:
    """Controlled GPIO events: allowlisted pins, debounce, sim adapter,
    fixed event schemas. No raw remote GPIO access."""

    def __init__(self, config: dict | None = None):
        cfg = (config or {}).get("gpio", config or {})
        self.enabled = bool(cfg.get("enabled", False))
        self.pins: list[int] = list(cfg.get("pins", [17]))
        self.debounce_ms = int(cfg.get("debounce_ms", 200))
        self.simulate = bool(cfg.get("simulate", True))
        self._last: dict[int, float] = {}
        self._events: list[dict] = []

    def emit(self, pin: int, level: bool) -> dict | None:
        if not self.enabled or pin not in self.pins:
            return None
        now = time.monotonic() * 1000
        if now - self._last.get(pin, 0) < self.debounce_ms:
            return None
        self._last[pin] = now
        ev = {
            "schema": "gpio.v1",
            "pin": pin,
            "level": level,
            "ts": time.time(),
            "simulated": self.simulate,
        }
        self._events.append(ev)
        return ev

    def poll(self) -> list[dict]:
        evs, self._events = self._events, []
        return evs


# ── F21 Model Registry ──────────────────────────────────────────────
@dataclass
class ModelManifest:
    """Model manifest with version, checksum, license, and compatibility info."""
    name: str
    version: str
    file_path: str
    sha256: str
    license: str
    license_url: str
    backend: str  # "yunet", "sface", "scrfd", "arcface", "genderage", "landmark", "mediapipe"
    input_shape: list[int]
    output_shapes: list[list[int]]
    embedding_dim: int | None = None
    commercial_license_required: bool = False
    source_url: str = ""
    installed: bool = False
    active: bool = False
    load_error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelManifest":
        return cls(**d)


class ModelRegistry:
    """Model registry with manifest, checksum validation, license tracking.

    Features:
    - Model manifest with version, SHA256, license metadata
    - Checksum validation on load
    - License compliance tracking (commercial vs research)
    - Backend compatibility matrix
    - Installation/activation status
    - Offline installation support
    - No auto-download without explicit license consent
    """

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self.models_dir = Path(cfg.get("model_registry", {}).get("models_dir", "models"))
        self.manifest_path = self.models_dir / "manifest.json"
        self.auto_download = bool(cfg.get("model_registry", {}).get("auto_download", False))
        self.require_license_consent = bool(cfg.get("model_registry", {}).get("require_license_consent", True))
        self.offline_models_dir = Path(cfg.get("model_registry", {}).get("offline_models_dir", "models/offline"))

        self._manifests: dict[str, ModelManifest] = {}
        self._load_manifest()

    def _load_manifest(self) -> None:
        """Load manifest from disk."""
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path) as f:
                    data = json.load(f)
                for m in data.get("models", []):
                    manifest = ModelManifest.from_dict(m)
                    self._manifests[manifest.name] = manifest
                logger.info("Loaded %d model manifests", len(self._manifests))
            except Exception as exc:
                logger.warning("Failed to load model manifest: %s", exc)

    def _save_manifest(self) -> None:
        """Save manifest to disk."""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "version": "1.0",
            "models": [m.to_dict() for m in self._manifests.values()],
        }
        with open(self.manifest_path, "w") as f:
            json.dump(data, f, indent=2)

    def register(self, manifest: ModelManifest) -> bool:
        """Register a model manifest."""
        # Verify file exists
        file_path = self.models_dir / manifest.file_path
        if not file_path.exists():
            # Check offline directory
            offline_path = self.offline_models_dir / manifest.file_path
            if offline_path.exists():
                import shutil
                file_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(offline_path, file_path)
            else:
                manifest.load_error = f"Model file not found: {manifest.file_path}"
                manifest.installed = False
                self._manifests[manifest.name] = manifest
                self._save_manifest()
                return False

        # Verify checksum
        if manifest.sha256:
            if not self._verify_checksum(file_path, manifest.sha256):
                manifest.load_error = "Checksum mismatch"
                manifest.installed = False
                self._manifests[manifest.name] = manifest
                self._save_manifest()
                return False

        manifest.installed = True
        manifest.load_error = None
        self._manifests[manifest.name] = manifest
        self._save_manifest()
        return True

    def _verify_checksum(self, file_path: Path, expected_sha256: str) -> bool:
        """Verify SHA256 checksum."""
        if not expected_sha256:
            return True
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest() == expected_sha256

    def compute_checksum(self, file_path: Path) -> str:
        """Compute SHA256 of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def get(self, name: str) -> ModelManifest | None:
        return self._manifests.get(name)

    def list_models(self, backend: str | None = None, installed_only: bool = False) -> list[ModelManifest]:
        models = list(self._manifests.values())
        if backend:
            models = [m for m in models if m.backend == backend]
        if installed_only:
            models = [m for m in models if m.installed]
        return models

    def activate(self, name: str) -> bool:
        """Mark model as active."""
        if name not in self._manifests:
            return False
        # Deactivate other models of same backend
        backend = self._manifests[name].backend
        for m in self._manifests.values():
            if m.backend == backend:
                m.active = False
        self._manifests[name].active = True
        self._save_manifest()
        return True

    def get_active(self, backend: str) -> ModelManifest | None:
        for m in self._manifests.values():
            if m.backend == backend and m.active:
                return m
        return None

    def set_load_error(self, name: str, error: str) -> None:
        if name in self._manifests:
            self._manifests[name].load_error = error
            self._manifests[name].installed = False
            self._save_manifest()

    def verify_all(self) -> dict[str, bool]:
        """Verify checksums of all installed models."""
        results = {}
        for name, manifest in self._manifests.items():
            if manifest.installed:
                file_path = self.models_dir / manifest.file_path
                if file_path.exists():
                    results[name] = self._verify_checksum(file_path, manifest.sha256)
                else:
                    results[name] = False
        return results

    def get_license_status(self) -> dict:
        """Get license compliance status."""
        commercial_required = [m for m in self._manifests.values() if m.commercial_license_required and m.installed]
        return {
            "total_models": len(self._manifests),
            "installed": len([m for m in self._manifests.values() if m.installed]),
            "commercial_license_required": len(commercial_required),
            "models_requiring_license": [m.name for m in commercial_required],
            "license_consent_required": self.require_license_consent,
        }

    def install_from_offline(self, name: str) -> bool:
        """Install model from offline directory."""
        manifest = self._manifests.get(name)
        if not manifest:
            return False
        offline_path = self.offline_models_dir / manifest.file_path
        if not offline_path.exists():
            manifest.load_error = f"Offline model not found: {offline_path}"
            self._save_manifest()
            return False

        target_path = self.models_dir / manifest.file_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(offline_path, target_path)

        return self.register(manifest)

    def create_manifest_template(self, name: str, backend: str, file_path: str) -> ModelManifest:
        """Create a manifest template for manual completion."""
        return ModelManifest(
            name=name,
            version="1.0",
            file_path=file_path,
            sha256="",  # User must fill
            license="",  # User must fill
            license_url="",
            backend=backend,
            input_shape=[1, 3, 640, 640],
            output_shapes=[[1, 100, 4]],
            embedding_dim=None,
            commercial_license_required=True,
            source_url="",
        )
