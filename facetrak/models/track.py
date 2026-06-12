import time
from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np

from .detection import FaceDetection

_VOTE_WINDOW = 12
_MIN_VOTES = 3


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
