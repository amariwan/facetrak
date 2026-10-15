"""F07 — Multi-camera handover: non-identifying event correlation.

Uses explicitly configured camera relationships + time windows.
For people: anonymized occupancy / manually-initiated session transitions
only. Covert biometric cross-camera matching is refused by design.
"""

from __future__ import annotations

import time


class CameraHandover:
    """Correlate exit/enter events across related cameras."""

    def __init__(self, config: dict | None = None):
        cfg = (config or {}).get("handover", config or {})
        # relations: [{"from": "cam0", "to": "cam1", "window_s": 10}]
        self.relations: list[dict] = list(cfg.get("relations", []))
        self.enabled = bool(cfg.get("enabled", False))
        self._exits: list[dict] = []  # {camera, label, ts, session?}

    def record_exit(
        self, camera: str, label: str = "object", session: str | None = None
    ) -> dict:
        ev = {"camera": camera, "label": label, "ts": time.time(), "session": session}
        self._exits.append(ev)
        self._exits = self._exits[-200:]
        return ev

    def correlate(
        self,
        camera: str,
        label: str = "object",
        session: str | None = None,
        person: bool = False,
    ) -> dict | None:
        """Find a matching recent exit on a related camera.

        person=True requires an explicit session token (manual transition);
        otherwise only anonymous occupancy counts are returned.
        """
        if not self.enabled:
            return None
        now = time.time()
        for rel in self.relations:
            if rel.get("to") != camera:
                continue
            window = float(rel.get("window_s", 10))
            for ev in reversed(self._exits):
                if ev["camera"] != rel.get("from"):
                    continue
                if now - ev["ts"] > window:
                    break
                if person and not (session and ev.get("session") == session):
                    # No biometric matching: refuse to link person tracks.
                    continue
                if not person and ev.get("label") != label:
                    continue
                return {
                    "from": ev["camera"],
                    "to": camera,
                    "dt_s": round(now - ev["ts"], 2),
                    "kind": "session" if person else "occupancy",
                }
        return None
