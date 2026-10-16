import threading
import time
from dataclasses import dataclass, field

import numpy as np

from facetrak.models import FaceDetection
from facetrak.models.track import Track, _LostTrack

_IOU_MATCH    = 0.3
_MAX_MISSED   = 15
_REID_TIMEOUT = 8.0
_REID_SIM     = 0.50


def _iou(a: FaceDetection, b: FaceDetection) -> float:
    ax2, ay2 = a.x + a.w, a.y + a.h
    bx2, by2 = b.x + b.w, b.y + b.h
    ix = max(0, min(ax2, bx2) - max(a.x, b.x))
    iy = max(0, min(ay2, by2) - max(a.y, b.y))
    inter = ix * iy
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


class FaceTracker:
    def __init__(self, iou_match: float = _IOU_MATCH,
                 max_missed: int = _MAX_MISSED,
                 reid_timeout: float = _REID_TIMEOUT,
                 reid_sim: float = _REID_SIM):
        self.tracks: list[Track] = []
        self._next_id = 1
        self._lost: list[_LostTrack] = []
        self._id_lock = threading.Lock()
        self.iou_match = iou_match
        self.max_missed = max_missed
        self.reid_timeout = reid_timeout
        self.reid_sim = reid_sim
        # Legacy aliases: keep instance attrs in sync so external code that
        # sets tracker._IOU_MATCH keeps working.
        self._IOU_MATCH = iou_match
        self._MAX_MISSED = max_missed

    def _thresholds(self) -> tuple[float, int, float, float]:
        iou = getattr(self, "_IOU_MATCH", self.iou_match)
        missed = getattr(self, "_MAX_MISSED", self.max_missed)
        return iou, missed, self.reid_timeout, self.reid_sim

    def update(self, detections: list[FaceDetection]
               ) -> tuple[list[Track], list[Track]]:
        iou_match, max_missed, reid_timeout, _ = self._thresholds()
        now = time.monotonic()
        self._lost = [l for l in self._lost
                      if now - l.lost_at < reid_timeout]

        unmatched = list(detections)
        for track in self.tracks:
            best_iou, best_det = 0.0, None
            for det in unmatched:
                iou = _iou(track.det, det)
                if iou > best_iou:
                    best_iou, best_det = iou, det
            if best_det is not None and best_iou >= iou_match:
                unmatched.remove(best_det)
                track.det = best_det
                track.last_seen = now
                track.missed = 0
            else:
                track.missed += 1

        ended = [t for t in self.tracks if t.missed > max_missed]
        for t in ended:
            if t.embedding is not None:
                self._lost.append(
                    _LostTrack(t.track_id, t.name, t.embedding))
        self.tracks = [t for t in self.tracks if t.missed <= max_missed]

        for det in unmatched:
            with self._id_lock:
                new_id = self._next_id
                self._next_id += 1
            t = Track(track_id=new_id, det=det)
            self.tracks.append(t)

        return self.active, ended

    def set_embedding(self, track_id: int, emb: np.ndarray | None):
        for t in self.tracks:
            if t.track_id == track_id:
                t.embedding = emb
                return

    def try_reid(self, track_id: int, emb: np.ndarray | None) -> str | None:
        if emb is None or not self._lost:
            return None
        _, _, _, reid_sim = self._thresholds()
        best_sim, best = 0.0, None
        for lost in self._lost:
            if lost.embedding is None:
                continue
            sim = float(emb @ lost.embedding)
            if sim > best_sim:
                best_sim, best = sim, lost
        if best is not None and best_sim >= reid_sim:
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
