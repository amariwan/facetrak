"""F01/F05/F06 — Multi-object tracking with IoU matching, track aging,
occlusion recovery (motion prediction + configurable matching), and
optional Kalman-style constant-velocity prediction.

Privacy: tracks are anonymous numeric IDs with optional non-identifying
labels (e.g. object class). No biometric cross-camera matching here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class MOTDetection:
    x: float
    y: float
    w: float
    h: float
    confidence: float = 1.0
    label: str = "object"

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2


@dataclass
class MOTTrack:
    track_id: int
    label: str
    x: float
    y: float
    w: float
    h: float
    confidence: float = 1.0
    age: int = 1
    hits: int = 1
    misses: int = 0
    occluded: bool = False
    vx: float = 0.0
    vy: float = 0.0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def predict(self) -> tuple[float, float, float, float]:
        """Constant-velocity prediction (F06)."""
        return (self.x + self.vx, self.y + self.vy, self.w, self.h)

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "label": self.label,
            "bbox": [
                round(self.x, 1),
                round(self.y, 1),
                round(self.w, 1),
                round(self.h, 1),
            ],
            "confidence": round(self.confidence, 3),
            "age": self.age,
            "hits": self.hits,
            "misses": self.misses,
            "occluded": self.occluded,
        }


def _iou(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


class MultiObjectTracker:
    """Pluggable greedy-IoU tracker (F01) with aging + occlusion recovery.

    Config keys (``tracking`` section):
      classes, conf_threshold, iou_threshold, max_age, min_hits,
      prediction (bool), occlusion_penalty.
    """

    ALGORITHMS = ("iou", "iou_kalman")

    def __init__(self, config: dict | None = None):
        cfg = (config or {}).get("tracking", config or {})
        self.classes: list[str] = list(cfg.get("classes", ["face", "person"]))
        self.conf_threshold = float(cfg.get("conf_threshold", 0.4))
        self.iou_threshold = float(cfg.get("iou_threshold", 0.3))
        self.max_age = int(cfg.get("max_age", 30))
        self.min_hits = int(cfg.get("min_hits", 3))
        self.prediction = bool(cfg.get("prediction", True))
        self.algorithm = str(cfg.get("algorithm", "iou_kalman"))
        if self.algorithm not in self.ALGORITHMS:
            self.algorithm = "iou_kalman"
        self._tracks: dict[int, MOTTrack] = {}
        self._next_id = 1
        self.id_switches = 0
        self.false_reassociations = 0
        self.total_matches = 0

    @property
    def tracks(self) -> list[MOTTrack]:
        return list(self._tracks.values())

    @property
    def active(self) -> list[MOTTrack]:
        return [
            t
            for t in self._tracks.values()
            if t.hits >= self.min_hits and not t.occluded
        ]

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1

    def update(
        self, detections: list[MOTDetection]
    ) -> tuple[list[MOTTrack], list[MOTTrack]]:
        dets = [
            d
            for d in detections
            if d.confidence >= self.conf_threshold
            and (not self.classes or d.label in self.classes or "*" in self.classes)
        ]
        # Predict step (F06): move boxes by velocity for matching.
        preds: dict[int, tuple[float, float, float, float]] = {}
        for tid, t in self._tracks.items():
            preds[tid] = (
                t.predict()
                if (self.prediction and self.algorithm == "iou_kalman")
                else (t.x, t.y, t.w, t.h)
            )
        matched_tids: set[int] = set()
        matched_dets: set[int] = set()
        # Greedy IoU matching.
        pairs: list[tuple[float, int, int]] = []
        tids = list(self._tracks.keys())
        for di, d in enumerate(dets):
            for tid in tids:
                t = self._tracks[tid]
                if t.label != d.label and self.classes:
                    continue
                iou = _iou(preds[tid], (d.x, d.y, d.w, d.h))
                if iou >= self.iou_threshold:
                    pairs.append((iou, tid, di))
        pairs.sort(reverse=True)
        for _, tid, di in pairs:
            if tid in matched_tids or di in matched_dets:
                continue
            matched_tids.add(tid)
            matched_dets.add(di)
            t = self._tracks[tid]
            d = dets[di]
            # Velocity update (F06).
            t.vx = d.cx - (t.x + t.w / 2)
            t.vy = d.cy - (t.y + t.h / 2)
            if t.occluded:
                t.occluded = False
            t.x, t.y, t.w, t.h = d.x, d.y, d.w, d.h
            t.confidence = d.confidence
            t.hits += 1
            t.misses = 0
            t.age += 1
            t.updated_at = time.time()
            self.total_matches += 1
        ended: list[MOTTrack] = []
        for tid, t in list(self._tracks.items()):
            if tid not in matched_tids:
                t.misses += 1
                t.age += 1
                if self.prediction:  # keep predicting through occlusion (F05)
                    px, py, pw, ph = t.predict()
                    t.x, t.y, t.w, t.h = px, py, pw, ph
                    t.occluded = True
                if t.misses > self.max_age:
                    ended.append(t)
                    del self._tracks[tid]
        for di, d in enumerate(dets):
            if di not in matched_dets:
                t = MOTTrack(
                    track_id=self._next_id,
                    label=d.label,
                    x=d.x,
                    y=d.y,
                    w=d.w,
                    h=d.h,
                    confidence=d.confidence,
                )
                self._tracks[self._next_id] = t
                self._next_id += 1
        return self.active, ended

    def reassign(self, old_id: int, new_id: int, correct: bool) -> None:
        """Record a manual/measured reassociation outcome (F05 metrics)."""
        if correct:
            self.id_switches += 1
        else:
            self.false_reassociations += 1

    def diagnostics(self) -> dict:
        return {
            "algorithm": self.algorithm,
            "live_tracks": len(self._tracks),
            "active_tracks": len(self.active),
            "id_switches": self.id_switches,
            "false_reassociations": self.false_reassociations,
            "total_matches": self.total_matches,
            "config": {
                "classes": self.classes,
                "conf_threshold": self.conf_threshold,
                "iou_threshold": self.iou_threshold,
                "max_age": self.max_age,
                "min_hits": self.min_hits,
                "prediction": self.prediction,
            },
        }
