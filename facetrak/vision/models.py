"""F08 — Local model registry. F09 — AI backend switching.
F10 — Model benchmark lab (clearly labeled synthetic vs hardware).
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import platform
import time
from dataclasses import asdict, dataclass

# ── F08 registry ──────────────────────────────────────────────────
TASKS = (
    "detection",
    "recognition",
    "segmentation",
    "pose",
    "gesture",
    "depth",
    "open-vocab",
    "tracking",
)
FRAMEWORKS = (
    "onnx",
    "tflite",
    "litert",
    "hailo",
    "imx500",
    "mediapipe-tasks",
    "opencv-dnn",
    "ultralytics",
)


@dataclass
class ModelEntry:
    model_id: str
    name: str
    version: str = "1.0"
    task: str = "detection"
    framework: str = "onnx"
    architecture: str = ""
    source: str = "local"
    license: str = "unknown"
    checksum: str = ""
    devices: list[str] | None = None
    input_shape: list[int] | None = None
    output_schema: str = ""
    resource: str = ""  # e.g. "cpu-light", "pi4-ok", "pi5-hailo"
    active: bool = False

    def __post_init__(self):
        if self.devices is None:
            self.devices = []
        if self.input_shape is None:
            self.input_shape = []


class ModelRegistry:
    """Local registry with validation before activation. Model files are
    treated as data; no executable plugin loading from model files."""

    def __init__(self):
        self._models: dict[str, ModelEntry] = {}
        self._register_builtin()

    def _register_builtin(self):
        builtin = [
            ModelEntry(
                "yunet-2023",
                "YuNet face detector",
                task="detection",
                framework="onnx",
                architecture="yunet",
                source="opencv zoo",
                license="Apache-2.0",
                devices=["amd64", "arm64"],
                input_shape=[0, 0, 3],
                resource="cpu-light",
                active=True,
            ),
            ModelEntry(
                "sface-2021",
                "SFace recognizer",
                task="recognition",
                framework="onnx",
                architecture="sface",
                source="opencv zoo",
                license="Apache-2.0",
                devices=["amd64", "arm64"],
                input_shape=[112, 112, 3],
                resource="cpu-light",
                active=True,
            ),
        ]
        for m in builtin:
            self._models[m.model_id] = m

    def validate(self, entry: ModelEntry) -> str | None:
        if not entry.model_id or len(entry.model_id) > 64:
            return "model_id required (<=64 chars)"
        if entry.task not in TASKS:
            return f"unknown task {entry.task!r}"
        if entry.framework not in FRAMEWORKS:
            return f"unknown framework {entry.framework!r}"
        if entry.checksum and (
            len(entry.checksum) != 64
            or any(c not in "0123456789abcdef" for c in entry.checksum.lower())
        ):
            return "checksum must be sha256 hex when provided"
        return None

    def register(self, entry: ModelEntry) -> dict:
        err = self.validate(entry)
        if err:
            return {"ok": False, "error": err}
        self._models[entry.model_id] = entry
        return {"ok": True, "model_id": entry.model_id}

    def activate(self, model_id: str) -> dict:
        m = self._models.get(model_id)
        if not m:
            return {"ok": False, "error": f"unknown model {model_id!r}"}
        err = self.validate(m)
        if err:
            return {"ok": False, "error": err}
        for other in self._models.values():
            if other.task == m.task:
                other.active = False
        m.active = True
        return {"ok": True, "model_id": model_id}

    def list(self, task: str | None = None) -> list[dict]:
        return [asdict(m) for m in self._models.values() if not task or m.task == task]

    def checksum_file(self, path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()


# ── F09 backends ──────────────────────────────────────────────────
class InferenceBackend:
    name = "base"

    def available(self) -> bool:
        return False

    def info(self) -> dict:
        return {"backend": self.name, "available": self.available()}


class BackendManager:
    """Pluggable runtimes; switching never breaks the API contract —
    callers use run() which falls back to CPU with a clear status."""

    BACKENDS = ("opencv-dnn", "onnxruntime", "tflite", "hailo", "imx500", "cpu")

    def __init__(self, config: dict | None = None):
        cfg = (config or {}).get("inference", config or {})
        self.active = str(cfg.get("backend", "opencv-dnn"))
        if self.active not in self.BACKENDS:
            self.active = "opencv-dnn"

    def detect_runtime(self, name: str) -> dict:
        avail = False
        detail = ""
        try:
            if name in ("opencv-dnn", "cpu"):
                import cv2  # noqa: F401

                avail = True
                detail = "opencv available"
            elif name == "onnxruntime":
                import onnxruntime  # noqa: F401

                avail = True
                detail = "onnxruntime importable"
            elif name == "tflite":
                try:
                    import ai_edge_litert  # noqa: F401

                    avail = True
                    detail = "litert available"
                except ImportError:
                    import tflite_runtime  # noqa: F401

                    avail = True
                    detail = "tflite_runtime available"
            elif name == "hailo":
                import hailo_platform  # noqa: F401

                avail = True
                detail = "hailo_platform importable"
            elif name == "imx500":
                import picamera2  # noqa: F401

                avail = True
                detail = "picamera2 importable (IMX500 via camera stack)"
        except ImportError as exc:
            detail = f"not installed: {exc}"
        return {"backend": name, "available": avail, "detail": detail}

    def capabilities(self) -> dict:
        return {
            "active": self.active,
            "backends": [self.detect_runtime(b) for b in self.BACKENDS],
        }

    def switch(self, name: str) -> dict:
        if name not in self.BACKENDS:
            return {"ok": False, "error": f"unknown backend {name!r}"}
        det = self.detect_runtime(name)
        self.active = name
        # Contract preserved: unavailable backends report fallback status.
        return {
            "ok": True,
            "backend": name,
            "available": det["available"],
            "fallback": "cpu" if not det["available"] else None,
            "detail": det["detail"],
        }

    def run(self, frame, fn_name: str = "detect") -> dict:
        """Uniform call envelope regardless of backend."""
        det = self.detect_runtime(self.active)
        return {
            "backend": self.active,
            "available": det["available"],
            "contract": fn_name,
            "fallback_used": not det["available"],
        }


# ── F10 benchmark lab ─────────────────────────────────────────────
class BenchmarkLab:
    """Reproducible local benchmarks. `mode` is always recorded:
    'hardware' = measured on this device, 'synthetic' = simulated."""

    def __init__(self, config: dict | None = None):
        cfg = (config or {}).get("benchmark", config or {})
        self.warmup = int(cfg.get("warmup", 5))
        self.iters = int(cfg.get("iters", 30))

    def run(self, fn, label: str = "model", mode: str = "hardware") -> dict:
        lat: list[float] = []
        for _ in range(self.warmup):
            fn()
        for _ in range(self.iters):
            t0 = time.monotonic()
            fn()
            lat.append((time.monotonic() - t0) * 1000.0)
        lat_sorted = sorted(lat)
        n = len(lat_sorted)

        def _pct(p: int) -> float:
            return lat_sorted[min(n - 1, int(n * p / 100))]

        try:
            import psutil

            mem = psutil.Process().memory_info().rss / 1e6
            cpu = psutil.cpu_percent(interval=None)
        except ImportError:
            mem, cpu = 0.0, 0.0
        return {
            "label": label,
            "mode": mode,  # never mix synthetic/hardware
            "device": platform.machine(),
            "iters": n,
            "latency_ms": {
                "avg": round(sum(lat) / n, 2),
                "p50": round(_pct(50), 2),
                "p95": round(_pct(95), 2),
                "p99": round(_pct(99), 2),
            },
            "memory_mb": round(mem, 1),
            "cpu_pct": cpu,
            "note": "synthetic" if mode == "synthetic" else "measured on this device",
        }

    @staticmethod
    def to_json(result: dict) -> str:
        return json.dumps(result, indent=2)

    @staticmethod
    def to_csv(results: list[dict]) -> str:
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(
            [
                "label",
                "mode",
                "device",
                "avg_ms",
                "p50_ms",
                "p95_ms",
                "p99_ms",
                "memory_mb",
            ]
        )
        for r in results:
            L = r["latency_ms"]
            w.writerow(
                [
                    r["label"],
                    r["mode"],
                    r["device"],
                    L["avg"],
                    L["p50"],
                    L["p95"],
                    L["p99"],
                    r["memory_mb"],
                ]
            )
        return buf.getvalue()
