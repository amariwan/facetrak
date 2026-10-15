"""Detection zones, privacy masks, and zone enter/exit events.

Coordinates are normalized (0..1 relative to frame width/height) so zones
survive resolution changes. Masks are applied pre-detection: anything under
a mask never reaches the detector.
"""

import logging
import time

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _to_pixel(poly: list, w: int, h: int) -> np.ndarray:
    pts = np.array([[x * w, y * h] for x, y in poly], dtype=np.float32)
    return pts.astype(np.int32)


def valid_polygon(poly) -> bool:
    if not isinstance(poly, (list, tuple)) or len(poly) < 3:
        return False
    try:
        pts = [(float(x), float(y)) for x, y in poly]
    except (TypeError, ValueError):
        return False
    return all(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for x, y in pts)


class PrivacyMask:
    """Black-out region applied to the frame before detection."""

    def __init__(self, polygon: list, name: str = ""):
        if not valid_polygon(polygon):
            raise ValueError("privacy mask needs ≥3 normalized points")
        self.polygon = [(float(x), float(y)) for x, y in polygon]
        self.name = name

    def apply(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        cv2.fillPoly(frame, [_to_pixel(self.polygon, w, h)], (0, 0, 0))
        return frame

    def to_dict(self) -> dict:
        return {"name": self.name, "polygon": [list(p) for p in self.polygon]}


class Zone:
    """Named polygon with enter/exit tracking per track id."""

    def __init__(
        self, name: str, polygon: list, enabled: bool = True, analyze: bool = True
    ):
        if not name or not isinstance(name, str):
            raise ValueError("zone needs a non-empty name")
        if not valid_polygon(polygon):
            raise ValueError(f"zone '{name}': need ≥3 normalized points")
        self.name = name
        self.polygon = [(float(x), float(y)) for x, y in polygon]
        self.enabled = enabled
        self.analyze = analyze
        self._inside: set[int] = set()

    def contains(self, nx: float, ny: float, w: int = 1000, h: int = 1000) -> bool:
        pts = _to_pixel(self.polygon, w, h)
        return cv2.pointPolygonTest(pts, (nx * w, ny * h), False) >= 0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "polygon": [list(p) for p in self.polygon],
            "enabled": self.enabled,
            "analyze": self.analyze,
        }


class ZoneManager:
    """Tracks zone membership; emits enter/exit event dicts."""

    def __init__(self, zones: list[Zone] | None = None):
        self.zones: list[Zone] = list(zones or [])

    @classmethod
    def from_config(cls, cfg: dict) -> "ZoneManager":
        zones = []
        for entry in cfg.get("zones", []):
            try:
                zones.append(
                    Zone(
                        entry["name"],
                        entry["polygon"],
                        enabled=entry.get("enabled", True),
                        analyze=entry.get("analyze", True),
                    )
                )
            except (KeyError, ValueError, TypeError) as exc:
                logger.warning("Ignoring invalid zone %r: %s", entry, exc)
        return cls(zones)

    def to_config(self) -> list[dict]:
        return [z.to_dict() for z in self.zones]

    def update(
        self, track_id: int, nx: float, ny: float, name: str | None = None
    ) -> list[dict]:
        """Update membership for one track; return enter/exit events."""
        events = []
        for zone in self.zones:
            if not zone.enabled:
                continue
            inside = zone.contains(nx, ny)
            was_inside = track_id in zone._inside
            if inside and not was_inside:
                zone._inside.add(track_id)
                events.append(
                    {
                        "kind": "zone_enter",
                        "zone": zone.name,
                        "track_id": track_id,
                        "name": name,
                        "ts": time.time(),
                    }
                )
            elif not inside and was_inside:
                zone._inside.discard(track_id)
                events.append(
                    {
                        "kind": "zone_exit",
                        "zone": zone.name,
                        "track_id": track_id,
                        "name": name,
                        "ts": time.time(),
                    }
                )
        return events

    def forget_track(self, track_id: int):
        for zone in self.zones:
            zone._inside.discard(track_id)

    def zones_for_point(self, nx: float, ny: float) -> list[str]:
        return [z.name for z in self.zones if z.enabled and z.contains(nx, ny)]


def load_masks(cfg: dict) -> list[PrivacyMask]:
    masks = []
    for entry in cfg.get("privacy_masks", []):
        try:
            poly = entry["polygon"] if isinstance(entry, dict) else entry
            name = entry.get("name", "") if isinstance(entry, dict) else ""
            masks.append(PrivacyMask(poly, name))
        except (KeyError, ValueError, TypeError) as exc:
            logger.warning("Ignoring invalid privacy mask %r: %s", entry, exc)
    return masks
