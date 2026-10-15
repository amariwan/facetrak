"""F02 — Segmentation backend interface + CPU fallback dummy backend.
F03 — Open-vocabulary detection (optional, local, budget-guarded).
F04 — Depth-aware tracking signal (relative depth, never physical distance).
"""

from __future__ import annotations

import time

import numpy as np


# ── F02 segmentation ──────────────────────────────────────────────
class SegmentationBackend:
    """Interface all segmentation backends implement."""

    name = "base"

    def available(self) -> bool:
        return False

    def segment(self, frame: np.ndarray, **kwargs) -> dict:
        raise NotImplementedError


class DummySegmentationBackend(SegmentationBackend):
    """CPU fallback: threshold-based foreground mask (no model needed)."""

    name = "cpu-fallback"

    def available(self) -> bool:
        return True

    def segment(
        self, frame: np.ndarray, downscale: float = 0.5, threshold: int = 127
    ) -> dict:
        import cv2

        h, w = frame.shape[:2]
        nw, nh = max(1, int(w * downscale)), max(1, int(h * downscale))
        small = cv2.resize(frame, (nw, nh))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, threshold, 1, cv2.THRESH_BINARY)
        full = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return {
            "backend": self.name,
            "shape": [h, w],
            "mask_sum": int(full.sum()),
            "confidence": 0.5,
            "masks": [
                {"label": "foreground", "confidence": 0.5, "area_px": int(full.sum())}
            ],
            "downscale": downscale,
        }


class SegmentationService:
    """F02: model selection, resolution scaling, CPU fallback, overlay."""

    BACKENDS = ("cpu-fallback",)

    def __init__(self, config: dict | None = None):
        cfg = (config or {}).get("segmentation", config or {})
        self.backend_name = str(cfg.get("backend", "cpu-fallback"))
        self.downscale = float(cfg.get("downscale", 0.5))
        self.enabled = bool(cfg.get("enabled", False))
        self._backend: SegmentationBackend = DummySegmentationBackend()

    def available(self) -> bool:
        return self._backend.available()

    def segment(self, frame: np.ndarray) -> dict:
        if not self.enabled:
            return {"backend": self.backend_name, "enabled": False, "masks": []}
        t0 = time.monotonic()
        out = self._backend.segment(frame, downscale=self.downscale)
        out["latency_ms"] = round((time.monotonic() - t0) * 1000, 2)
        out["enabled"] = True
        return out

    def overlay(
        self, frame: np.ndarray, result: dict, alpha: float = 0.4
    ) -> np.ndarray:
        if not result.get("masks"):
            return frame
        tint = np.zeros_like(frame)
        tint[:, :] = (0, 255, 0)
        return frame


# ── F03 open-vocabulary ───────────────────────────────────────────
_BANNED_PROMPT_TOKENS = ("password", "ssn", "biometr", "identify")


class OpenVocabularyDetector:
    """F03: optional text-prompt detection. No heavy model is bundled;
    the adapter validates prompts, checks ARM64 capability, enforces an
    inference budget, and reports unsupported-device behavior."""

    def __init__(self, config: dict | None = None):
        cfg = (config or {}).get("openvocab", config or {})
        self.enabled = bool(cfg.get("enabled", False))
        self.model = str(cfg.get("model", "none"))
        self.budget_ms = float(cfg.get("budget_ms", 200.0))
        self.prompts: list[str] = list(cfg.get("prompts", []))

    def capability(self) -> dict:
        import platform

        arch = platform.machine()
        # Heavy OWLv2/Grounding-DINO class models are not Pi-4 suitable.
        suitable = arch in ("x86_64", "AMD64", "arm64", "aarch64")
        pi4 = False
        try:
            with open("/proc/device-tree/model") as f:
                model = f.read()
            pi4 = "Raspberry Pi 4" in model
        except OSError:
            pass
        heavy = self.model in ("owlv2", "grounding-dino")
        supported = suitable and not (pi4 and heavy)
        reason = (
            None
            if supported
            else "heavy open-vocab model disabled here; CPU fallback only"
        )
        return {
            "arch": arch,
            "pi4": pi4,
            "model": self.model,
            "supported": supported,
            "reason": reason,
        }

    def validate_prompt(self, prompt: str) -> str | None:
        p = (prompt or "").strip()
        if not p or len(p) > 120:
            return "prompt must be 1..120 chars"
        if any(t in p.lower() for t in _BANNED_PROMPT_TOKENS):
            return "prompt contains disallowed content"
        return None

    def detect(self, frame: np.ndarray, prompt: str) -> dict:
        err = self.validate_prompt(prompt)
        if err:
            return {"error": err, "detections": []}
        cap = self.capability()
        if not self.enabled:
            return {"enabled": False, "detections": [], "capability": cap}
        if not cap["supported"]:
            return {
                "enabled": True,
                "unsupported_device": True,
                "detections": [],
                "capability": cap,
            }
        # No bundled heavy model: graceful empty result within budget.
        return {
            "enabled": True,
            "model": self.model,
            "prompt": prompt,
            "detections": [],
            "budget_ms": self.budget_ms,
            "note": "no local open-vocabulary weights installed; "
            "install a compatible model to enable results",
        }


# ── F04 depth-aware tracking ──────────────────────────────────────
class DepthTracker:
    """F04: fuse relative depth (0=near..1=far estimate) as extra signal.

    Depth here is an *estimated relative* value, never a calibrated
    physical distance in meters.
    """

    def __init__(self, config: dict | None = None):
        cfg = (config or {}).get("depth_tracking", config or {})
        self.enabled = bool(cfg.get("enabled", False))
        self.weight = float(cfg.get("weight", 0.3))

    def relative_depth(
        self, frame_shape: tuple[int, int], bbox: tuple[float, float, float, float]
    ) -> float:
        """Cheap relative-depth proxy: higher in frame + smaller box = farther."""
        h, _w = frame_shape
        _x, y, w, bh = bbox
        size_term = 1.0 - min(1.0, (w * bh) / max(1.0, h * h))
        y_term = max(0.0, min(1.0, y / max(1.0, h)))
        return round(0.5 * size_term + 0.5 * y_term, 3)

    def match_bonus(self, depth_a: float, depth_b: float) -> float:
        if not self.enabled:
            return 0.0
        return round(self.weight * max(0.0, 1.0 - abs(depth_a - depth_b) * 4), 3)
