"""Rolling emotion + expression timeline backed by SQLite.

Writes one row per second per tracked face. Provides frequency counts
for the UI sparkline and CSV export for external analysis.
"""
import csv
import time
from collections import Counter
from pathlib import Path

from . import db

_RECORD_INTERVAL = 1.0   # seconds between DB writes per track


class EmotionTimeline:
    def __init__(self):
        self._last_write: dict[int, float] = {}   # track_id → monotonic time
        # in-memory ring for fast sparkline queries (≤300 entries)
        self._recent: list[dict] = []
        self._max_recent = 300

    def record(self, track_id: int, name: str | None, emotion: str,
               smile: float, attentive: bool, yaw: float, pitch: float):
        now = time.monotonic()
        if now - self._last_write.get(track_id, 0) < _RECORD_INTERVAL:
            return
        self._last_write[track_id] = now
        db.log_emotion(name, emotion, smile, attentive, yaw, pitch)
        entry = {"emotion": emotion, "smile": round(smile, 3),
                 "name": name or "unknown"}
        self._recent.append(entry)
        if len(self._recent) > self._max_recent:
            self._recent.pop(0)

    def recent_emotion_counts(self, n: int = 60) -> dict[str, int]:
        return dict(Counter(r["emotion"] for r in self._recent[-n:]))

    @staticmethod
    def export_csv(path: Path = Path("emotion_log.csv")) -> str:
        rows = db.query_emotions(limit=100_000)
        if not rows:
            return ""
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f,
                fieldnames=["ts", "name", "emotion", "smile",
                            "attentive", "yaw", "pitch"])
            w.writeheader()
            w.writerows(rows)
        return str(path.resolve())
