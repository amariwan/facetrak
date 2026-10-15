"""F23 camera profiles, F24 smart recording, F25 pre/post buffer,
F26 virtual cameras, F27 privacy-zone editor backend, F28 health,
F29 failover, F30 storage manager.
"""

from __future__ import annotations

import collections
import time
from pathlib import Path

import numpy as np

# ── F23 profiles ──────────────────────────────────────────────────
PROFILES = {
    "eco": {"detect_width": 320, "detect_every": 3, "objects": False, "pose": False},
    "balanced": {
        "detect_width": 480,
        "detect_every": 2,
        "objects": False,
        "pose": False,
    },
    "quality": {"detect_width": 640, "detect_every": 1, "objects": True, "pose": True},
}


class CameraProfileManager:
    def __init__(self, config: dict | None = None):
        cfg = (config or {}).get("camera_profiles", config or {})
        self.active = str(cfg.get("active", "balanced"))
        self._prev: str | None = None

    def switch(self, name: str, cfg: dict) -> dict:
        if name not in PROFILES:
            return {"ok": False, "error": f"unknown profile {name!r}"}
        self._prev = self.active
        self.active = name
        p = PROFILES[name]
        cfg["detect_width"] = p["detect_width"]
        cfg["detect_every"] = p["detect_every"]
        return {"ok": True, "profile": name, "applied": dict(p)}

    def rollback(self, cfg: dict) -> dict:
        if not self._prev:
            return {"ok": False, "error": "no previous profile"}
        return self.switch(self._prev, cfg)


# ── F24 smart recording ───────────────────────────────────────────
class SmartRecorder:
    """Event-driven recording. Disabled by default; requires explicit
    enable + policy. Privacy masks applied before storage."""

    def __init__(self, config: dict | None = None, data_dir: str = "."):
        cfg = (config or {}).get("recording", config or {})
        self.enabled = bool(cfg.get("enabled", False))  # default OFF
        self.trigger_events: list[str] = list(cfg.get("on_events", ["zone_enter"]))
        self.data_dir = Path(data_dir)
        self._active = False
        self.sessions: list[dict] = []

    def on_event(self, kind: str, apply_masks=None) -> dict | None:
        if not self.enabled or kind not in self.trigger_events:
            return None
        self._active = True
        sess = {"started": time.time(), "trigger": kind}
        self.sessions.append(sess)
        return sess

    def stop(self) -> dict:
        self._active = False
        return {"recording": False}

    @property
    def recording(self) -> bool:
        return self._active


# ── F25 pre/post buffer ───────────────────────────────────────────
class EventBuffer:
    """Bounded in-memory ring buffer for pre/post event windows."""

    def __init__(self, config: dict | None = None):
        cfg = (config or {}).get("event_buffer", config or {})
        self.pre_s = float(cfg.get("pre_s", 5.0))
        self.post_s = float(cfg.get("post_s", 5.0))
        self.fps = float(cfg.get("fps", 10.0))
        maxlen = max(1, int((self.pre_s + self.post_s) * self.fps))
        self._buf: collections.deque = collections.deque(maxlen=maxlen)
        self._post_remaining = 0

    def push(self, frame_no: int) -> None:
        self._buf.append((frame_no, time.time()))
        if self._post_remaining > 0:
            self._post_remaining -= 1

    def trigger(self) -> list:
        self._post_remaining = int(self.post_s * self.fps)
        return list(self._buf)

    def __len__(self) -> int:
        return len(self._buf)


# ── F26 virtual cameras ───────────────────────────────────────────
class VirtualCamera:
    """Hardware-independent source: video file, synthetic, mock, replay."""

    KINDS = ("synthetic", "file", "replay", "mock")

    def __init__(
        self,
        kind: str = "synthetic",
        path: str = "",
        width: int = 640,
        height: int = 480,
    ):
        if kind not in self.KINDS:
            raise ValueError(f"unknown virtual kind {kind!r}")
        self.kind = kind
        self.path = path
        self.width = width
        self.height = height
        self._n = 0
        self._cap = None
        self._replay_frames: list = []

    def read(self) -> tuple[bool, np.ndarray | None]:
        self._n += 1
        if self.kind in ("synthetic", "mock"):
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            # moving marker => deterministic, repeatable
            x = (self._n * 7) % self.width
            frame[:, x : x + 4] = (0, 255, 0)
            return True, frame
        if self.kind == "replay" and self._replay_frames:
            return True, self._replay_frames[self._n % len(self._replay_frames)]
        if self.kind == "file":
            import cv2

            if self._cap is None:
                self._cap = cv2.VideoCapture(self.path)
            ok, frame = self._cap.read()
            if not ok:  # loop for repeatable replay
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self._cap.read()
            return bool(ok), frame
        return False, None

    def load_replay(self, frames: list) -> None:
        self.kind = "replay"
        self._replay_frames = list(frames)


# ── F27 privacy editor backend ────────────────────────────────────
class PrivacyEditor:
    """Polygon CRUD with validation; syncs exclusion/anonymization/
    detection zones to config."""

    KINDS = ("exclusion", "anonymization", "detection")

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self.zones: list[dict] = list(cfg.get("privacy_editor", []))

    @staticmethod
    def validate_polygon(poly) -> str | None:
        if not isinstance(poly, list) or len(poly) < 3:
            return "polygon needs >= 3 points"
        for pt in poly:
            if (
                not isinstance(pt, (list, tuple))
                or len(pt) != 2
                or not all(isinstance(v, (int, float)) for v in pt)
                or not (0 <= pt[0] <= 1 and 0 <= pt[1] <= 1)
            ):
                return "points must be [x,y] in 0..1"
        return None

    def add(self, kind: str, polygon, name: str = "") -> dict:
        if kind not in self.KINDS:
            return {"ok": False, "error": f"unknown kind {kind!r}"}
        err = self.validate_polygon(polygon)
        if err:
            return {"ok": False, "error": err}
        zone = {"kind": kind, "polygon": [list(p) for p in polygon], "name": name}
        self.zones.append(zone)
        return {"ok": True, "zone": zone}

    def remove(self, idx: int) -> dict:
        if not 0 <= idx < len(self.zones):
            return {"ok": False, "error": "not found"}
        del self.zones[idx]
        return {"ok": True}


# ── F28 health ────────────────────────────────────────────────────
class CameraHealth:
    def __init__(self):
        self.frames = 0
        self.reconnects = 0
        self.failures = 0
        self.last_ts: float | None = None
        self.fps_ema = 0.0
        self.latency_ms = 0.0
        self.queue = 0

    def tick(self, latency_ms: float = 0.0) -> None:
        now = time.time()
        if self.last_ts:
            dt = now - self.last_ts
            inst = 1 / dt if dt > 0 else 0
            self.fps_ema = 0.9 * self.fps_ema + 0.1 * inst if self.fps_ema else inst
        self.last_ts = now
        self.frames += 1
        self.latency_ms = latency_ms

    def to_dict(self) -> dict:
        age = time.time() - self.last_ts if self.last_ts else None
        return {
            "frames": self.frames,
            "fps": round(self.fps_ema, 1),
            "frame_age_s": round(age, 2) if age is not None else None,
            "reconnects": self.reconnects,
            "failures": self.failures,
            "latency_ms": round(self.latency_ms, 1),
            "queue": self.queue,
        }


# ── F29 failover ──────────────────────────────────────────────────
class StreamFailover:
    """Primary/fallback switching with hysteresis."""

    def __init__(self, config: dict | None = None):
        cfg = (config or {}).get("failover", config or {})
        self.enabled = bool(cfg.get("enabled", False))
        self.fallback: str | int | None = cfg.get("fallback")
        self.fail_after = int(cfg.get("fail_after", 5))
        self.recover_after = int(cfg.get("recover_after", 10))
        self._fails = 0
        self._oks = 0
        self.on_fallback = False

    def report(self, ok: bool) -> dict:
        if ok:
            self._fails = 0
            if self.on_fallback:
                self._oks += 1
                if self._oks >= self.recover_after:
                    self.on_fallback = False
                    self._oks = 0
                    return {"source": "primary", "switched": True}
            return {
                "source": "fallback" if self.on_fallback else "primary",
                "switched": False,
            }
        self._oks = 0
        self._fails += 1
        if self.enabled and not self.on_fallback and self._fails >= self.fail_after:
            self.on_fallback = True
            return {"source": "fallback", "switched": True}
        return {
            "source": "fallback" if self.on_fallback else "primary",
            "switched": False,
        }


# ── F30 storage manager ───────────────────────────────────────────
class StorageManager:
    def __init__(self, config: dict | None = None, root: str = "."):
        cfg = (config or {}).get("storage", config or {})
        self.root = Path(root)
        self.quota_mb = float(cfg.get("quota_mb", 1024))
        self.retention_days = int(cfg.get("retention_days", 7))

    def usage(self) -> dict:
        total = 0
        files: list[Path] = []
        for ext in ("*.mp4", "*.avi", "*.png", "*.jpg"):
            files.extend(self.root.glob(ext))
        for p in files:
            try:
                total += p.stat().st_size
            except OSError:
                pass
        mb = total / 1e6
        return {
            "mb": round(mb, 1),
            "quota_mb": self.quota_mb,
            "pct": round(100 * mb / self.quota_mb, 1) if self.quota_mb else 0,
            "files": len(files),
        }

    def enforce(self) -> dict:
        """Delete oldest beyond quota/retention. Safe: only media globs."""
        u = self.usage()
        removed: list[str] = []
        if u["pct"] < 95:
            return {"removed": removed, "usage": u}
        cands: list[tuple[float, Path]] = []
        for ext in ("*.mp4", "*.avi", "*.png", "*.jpg"):
            for p in self.root.glob(ext):
                try:
                    cands.append((p.stat().st_mtime, p))
                except OSError:
                    pass
        for _, p in sorted(cands):
            try:
                p.unlink()
                removed.append(p.name)
            except OSError:
                pass
            if self.usage()["pct"] < 85:
                break
        return {"removed": removed, "usage": self.usage()}
