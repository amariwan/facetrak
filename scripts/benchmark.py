"""Reproducible performance benchmark (no camera needed).

Measures with a mock camera + stub detector so results are comparable
across machines and runs:
  - FPS (engine.step end-to-end)
  - detection latency distribution
  - frame drops under load
  - zone/mask overhead

Usage: .venv/bin/python scripts/benchmark.py [--frames 300]
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from facetrak.core import config as cfgmod
from facetrak.core.engine import FaceEngine
from facetrak.storage import db as dbmod


class BenchCam:
    def __init__(self, n, size=(480, 640)):
        rng = np.random.default_rng(7)
        self.frames = [
            rng.integers(0, 255, (*size, 3), dtype=np.uint8) for _ in range(n)
        ]
        self.i = 0
        from facetrak.camera.base import CameraHealth

        self.health = CameraHealth()

    def read(self):
        if self.i >= len(self.frames):
            self.i = 0  # loop: sustained load
        f = self.frames[self.i]
        self.i += 1
        self.health.ok()
        return True, f

    def release(self):
        pass

    @property
    def resolution(self):
        h, w = self.frames[0].shape[:2]
        return w, h


class StubDetector:
    ready = True

    def detect(self, frame, scale=1.0):
        time.sleep(0.004)  # ~4 ms stand-in for YuNet on Apple Silicon
        return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument(
        "--masks", type=int, default=2, help="number of privacy masks to apply"
    )
    ap.add_argument("--zones", type=int, default=2, help="number of zones to evaluate")
    args = ap.parse_args()

    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="facetrak-bench-"))
    cfgmod.CONFIG_PATH = tmp / "config.json"
    dbmod.DB_PATH = tmp / "bench.db"

    e = FaceEngine()
    e.cfg["zones"] = [
        {
            "name": f"z{i}",
            "polygon": [
                [0.05 * i, 0.0],
                [0.05 * i + 0.4, 0.0],
                [0.05 * i + 0.4, 1.0],
                [0.05 * i, 1.0],
            ],
        }
        for i in range(args.zones)
    ]
    from facetrak.events import PrivacyMask, ZoneManager

    e.zones = ZoneManager.from_config(e.cfg)
    e.masks = [
        PrivacyMask([[0.6, 0.6], [0.9, 0.6], [0.9, 0.9], [0.6, 0.9]], f"m{i}")
        for i in range(args.masks)
    ]
    e.cam = BenchCam(max(args.frames, 10))
    e.running = True
    e.detector = StubDetector()

    # stub out heavy per-frame stages: analyzer/identify need faces anyway
    lat = []
    t0 = time.monotonic()
    for _ in range(args.frames):
        s = time.monotonic()
        e.step()
        lat.append((time.monotonic() - s) * 1000.0)
    total = time.monotonic() - t0

    lat.sort()
    p50 = lat[len(lat) // 2]
    p95 = lat[int(len(lat) * 0.95)]
    print(f"frames={args.frames} total_s={total:.2f} fps={args.frames / total:.1f}")
    print(
        f"step_ms min={lat[0]:.2f} p50={p50:.2f} p95={p95:.2f} "
        f"max={lat[-1]:.2f} mean={statistics.mean(lat):.2f}"
    )
    print(
        f"detect_ms(last)={e.perf['detect_ms']:.2f} "
        f"drops={e.perf['drops']} masks={args.masks} zones={args.zones}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
