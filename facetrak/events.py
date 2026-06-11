"""Thin compatibility shim — delegates to db.py."""
from . import db


class PresenceLog:
    def appeared(self, name: str | None, track_id: int):
        db.log_presence("appeared", name or "unknown", track_id)

    def left(self, name: str | None, track_id: int,
             duration: float, blinks: int = 0):
        db.log_presence("left", name or "unknown", track_id,
                        round(duration, 1), blinks)

    @staticmethod
    def tail(limit: int = 50) -> list[dict]:
        return db.query_presence(limit)
