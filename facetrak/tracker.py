"""Multi-face IoU tracker with stable IDs, identity smoothing, and re-ID.

When a track ends its last known embedding is stored in a short-lived cache.
If a new track appears within _REID_TIMEOUT seconds and its embedding matches
at >= _REID_SIM cosine similarity, it inherits the old track's identity and
announced state so the presence log doesn't create a duplicate entry.
"""
import time
from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np

from .detection import FaceDetection

_IOU_MATCH    = 0.3
_MAX_MISSED   = 15
_VOTE_WINDOW  = 12
_MIN_VOTES    = 3
_REID_TIMEOUT = 8.0
_REID_SIM     = 0.50


@dataclass
class Track:
    track_id: int
    det: FaceDetection
    first_seen: float = field(default_factory=time.monotonic)
    last_seen: float = field(default_factory=time.monotonic)
    missed: int = 0
    name: str | None = None
    sim: float = 0.0
    announced: bool = False
    blink_count: int = 0
    age: str = "?"
    gender: str = "?"
    embedding: np.ndarray | None = field(default=None, repr=False)
    _eyes_closed: bool = False
    _votes: deque = field(default_factory=lambda: deque(maxlen=_VOTE_WINDOW))

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        d = self.det
        return d.x, d.y, d.w, d.h

    @property
    def dwell(self) -> float:
        return self.last_seen - self.first_seen

    def vote(self, name: str | None, sim: float):
        self._votes.append(name)
        counts = Counter(n for n in self._votes if n)
        if counts:
            top, n = counts.most_common(1)[0]
            if n >= _MIN_VOTES:
                self.name, self.sim = top, sim if name == top else self.sim
                return
        if self._votes.count(None) > len(self._votes) // 2:
            self.name = None
        self.sim = sim

    def update_blink(self, eyes_closed: bool):
        if eyes_closed and not self._eyes_closed:
            self.blink_count += 1
        self._eyes_closed = eyes_closed


@dataclass
class _LostTrack:
    track_id: int
    name: str | None
    embedding: np.ndarray | None
    lost_at: float = field(default_factory=time.monotonic)


def _iou(a: FaceDetection, b: FaceDetection) -> float:
    ax2, ay2 = a.x + a.w, a.y + a.h
    bx2, by2 = b.x + b.w, b.y + b.h
    ix = max(0, min(ax2, bx2) - max(a.x, b.x))
    iy = max(0, min(ay2, by2) - max(a.y, b.y))
    inter = ix * iy
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


class FaceTracker:
    def __init__(self):
        self.tracks: list[Track] = []
        self._next_id = 1
        self._lost: list[_LostTrack] = []

    def update(self, detections: list[FaceDetection]
               ) -> tuple[list[Track], list[Track]]:
        now = time.monotonic()
        self._lost = [l for l in self._lost
                      if now - l.lost_at < _REID_TIMEOUT]

        unmatched = list(detections)
        for track in self.tracks:
            best_iou, best_det = 0.0, None
            for det in unmatched:
                iou = _iou(track.det, det)
                if iou > best_iou:
                    best_iou, best_det = iou, det
            if best_det is not None and best_iou >= _IOU_MATCH:
                unmatched.remove(best_det)
                track.det = best_det
                track.last_seen = now
                track.missed = 0
            else:
                track.missed += 1

        ended = [t for t in self.tracks if t.missed > _MAX_MISSED]
        for t in ended:
            if t.embedding is not None:
                self._lost.append(
                    _LostTrack(t.track_id, t.name, t.embedding))
        self.tracks = [t for t in self.tracks if t.missed <= _MAX_MISSED]

        for det in unmatched:
            new_id = self._next_id
            t = Track(track_id=new_id, det=det)
            self._next_id += 1
            self.tracks.append(t)

        return self.active, ended

    def set_embedding(self, track_id: int, emb: np.ndarray | None):
        """Store the latest SFace embedding on a track for future re-ID."""
        for t in self.tracks:
            if t.track_id == track_id:
                t.embedding = emb
                return

    def try_reid(self, track_id: int, emb: np.ndarray | None) -> str | None:
        """Check if a new track matches a recently lost face."""
        if emb is None or not self._lost:
            return None
        best_sim, best = 0.0, None
        for lost in self._lost:
            if lost.embedding is None:
                continue
            sim = float(emb @ lost.embedding)
            if sim > best_sim:
                best_sim, best = sim, lost
        if best is not None and best_sim >= _REID_SIM:
            self._lost.remove(best)
            return best.name
        return None

    @property
    def active(self) -> list[Track]:
        return [t for t in self.tracks if t.missed == 0]

    def largest(self) -> Track | None:
        active = self.active
        return max(active, key=lambda t: t.det.area) if active else None

    def reset(self):
        self.tracks.clear()
        self._lost.clear()
