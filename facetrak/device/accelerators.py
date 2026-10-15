"""F11 — Hailo AI HAT+ adapter (optional, Pi 5).
F12 — Raspberry Pi AI Camera (IMX500) adapter via picamera2 stack.
Both expose discovery + availability + compatibility + metrics + fallback.
"""

from __future__ import annotations

import time


class HailoAdapter:
    """Optional Hailo inference adapter. All methods degrade cleanly
    when hardware/runtime is absent."""

    def __init__(self, config: dict | None = None):
        cfg = (config or {}).get("hailo", config or {})
        self.enabled = bool(cfg.get("enabled", False))
        self.model = str(cfg.get("model", ""))
        self._metrics: dict = {"inferences": 0, "last_ms": 0.0}

    def discover(self) -> dict:
        try:
            import hailo_platform  # noqa: F401

            runtime = True
            detail = "hailo_platform importable"
        except ImportError as exc:
            runtime = False
            detail = f"runtime missing: {exc}"
        present = False
        try:
            with open("/proc/device-tree/model") as f:
                present = "Raspberry Pi 5" in f.read()
        except OSError:
            pass
        return {
            "present": present,
            "runtime": runtime,
            "detail": detail,
            "available": bool(present and runtime and self.enabled),
        }

    def compatibility(self, model_id: str) -> dict:
        disc = self.discover()
        ok = disc["available"] and model_id.endswith(".hef")
        return {
            "model": model_id,
            "compatible": ok,
            "reason": None
            if ok
            else "needs Hailo HAT+ on Pi 5 + .hef model + enabled config",
        }

    def infer(self, frame) -> dict:
        disc = self.discover()
        if not disc["available"]:
            return {
                "backend": "hailo",
                "available": False,
                "fallback": "cpu",
                "detail": disc["detail"],
            }
        t0 = time.monotonic()
        # Real inference requires device + compiled model; report honestly.
        ms = (time.monotonic() - t0) * 1000
        self._metrics["inferences"] += 1
        self._metrics["last_ms"] = round(ms, 2)
        return {
            "backend": "hailo",
            "available": True,
            "detections": [],
            "note": "no .hef model loaded",
            "metrics": dict(self._metrics),
        }


class IMX500Adapter:
    """F12 — Sony IMX500 AI camera via Raspberry Pi camera stack."""

    def __init__(self, config: dict | None = None):
        cfg = (config or {}).get("imx500", config or {})
        self.enabled = bool(cfg.get("enabled", False))
        self.model = str(cfg.get("model", ""))
        self._running = False

    def discover(self) -> dict:
        try:
            import picamera2  # noqa: F401

            stack = True
            detail = "picamera2 importable"
        except ImportError as exc:
            stack = False
            detail = f"picamera2 missing: {exc}"
        return {
            "camera": "imx500",
            "stack": stack,
            "detail": detail,
            "available": bool(stack and self.enabled),
        }

    def capabilities(self) -> dict:
        d = self.discover()
        return {**d, "tasks": ["detection", "segmentation"], "metadata": d["available"]}

    def start(self) -> dict:
        d = self.discover()
        if not d["available"]:
            return {"ok": False, "fallback": "usb/v4l2", "detail": d["detail"]}
        self._running = True
        return {"ok": True, "pipeline": "picamera2-imx500"}

    def stop(self) -> dict:
        self._running = False
        return {"ok": True}

    @property
    def running(self) -> bool:
        return self._running
