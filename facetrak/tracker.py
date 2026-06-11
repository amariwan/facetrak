"""Multi-face IoU tracker with stable IDs and identity smoothing.

Detections are matched to existing tracks greedily by IoU. Each track keeps
a rolling vote over recognized names so a single bad frame cannot flip an
identity, plus timing data for presence statistics.
"""
import time
from collections import Counter, deque
from dataclasses import dataclass, field

from .detection import FaceDetection

_IOU_MATCH = 0.3
_MAX_MISSED = 15        # frames a track survives without a detection
_VOTE_WINDOW = 12       # recent recognition results considered per track
_MIN_VOTES = 3          # sightings needed before a name is trusted


@dataclass
class Track:
    track_id: int
    det: FaceDetection
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    missed: int = 0
    name: str | None = None
    sim: float = 0.0
    announced: bool = False
    blink_count: int = 0
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
        # not enough evidence (yet, or anymore)
        if self._votes.count(None) > len(self._votes) // 2:
            self.name = None
        self.sim = sim

    def update_blink(self, eyes_closed: bool):
        if eyes_closed and not self._eyes_closed:
            self.blink_count += 1
        self._eyes_closed = eyes_closed


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

    def update(self, detections: list[FaceDetection]
               ) -> tuple[list[Track], list[Track]]:
        """Match detections to tracks; returns (active_tracks, ended_tracks)."""
        now = time.time()
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

        for det in unmatched:
            self.tracks.append(Track(track_id=self._next_id, det=det))
            self._next_id += 1

        ended = [t for t in self.tracks if t.missed > _MAX_MISSED]
        self.tracks = [t for t in self.tracks if t.missed <= _MAX_MISSED]
        return self.active, ended

    @property
    def active(self) -> list[Track]:
        return [t for t in self.tracks if t.missed == 0]

    def largest(self) -> Track | None:
        active = self.active
        return max(active, key=lambda t: t.det.area) if active else None

    def reset(self):
        self.tracks.clear()
