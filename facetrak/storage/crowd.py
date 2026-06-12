import csv
import time
from pathlib import Path

from . import db

_SAMPLE_INTERVAL = 1.0


class CrowdMonitor:
    def __init__(self):
        self.current = 0
        self._last_sample = 0.0

    def tick(self, count: int, new_appearances: int = 0):
        self.current = count
        now = time.monotonic()
        if now - self._last_sample >= _SAMPLE_INTERVAL:
            db.log_crowd(count)
            self._last_sample = now

    def export_csv(self, path: Path = Path("crowd_stats.csv")) -> str:
        rows = db.query_crowd(limit=86400)
        if not rows:
            return ""
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["ts", "count"])
            w.writeheader()
            w.writerows(rows)
        return str(path.resolve())

    @staticmethod
    def summary() -> dict:
        return db.crowd_summary()
