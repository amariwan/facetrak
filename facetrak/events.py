"""Append-only presence log (JSONL): who appeared/left, when, for how long."""
import datetime
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

LOG_PATH = Path("events.jsonl")


class PresenceLog:
    def appeared(self, name: str | None, track_id: int):
        self._write({
            "event": "appeared",
            "name": name or "unknown",
            "track": track_id,
        })

    def left(self, name: str | None, track_id: int, duration: float,
             blinks: int = 0):
        self._write({
            "event": "left",
            "name": name or "unknown",
            "track": track_id,
            "duration_s": round(duration, 1),
            "blinks": blinks,
        })

    def _write(self, event: dict):
        event["ts"] = datetime.datetime.now().isoformat(timespec="seconds")
        try:
            with open(LOG_PATH, "a") as f:
                f.write(json.dumps(event) + "\n")
        except OSError:
            logger.warning("Could not write presence log", exc_info=True)

    @staticmethod
    def tail(limit: int = 50) -> list[dict]:
        if not LOG_PATH.exists():
            return []
        try:
            with open(LOG_PATH) as f:
                lines = f.readlines()[-limit:]
            return [json.loads(line) for line in lines if line.strip()]
        except (OSError, json.JSONDecodeError):
            logger.warning("Could not read presence log", exc_info=True)
            return []
