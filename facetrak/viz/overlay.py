import cv2
import numpy as np

from facetrak.models import Track
from facetrak.models.analysis import FaceMetrics
from facetrak.recog.quality import score as quality_score, GOOD_THRESHOLD

_COLOR_KNOWN   = (0, 255, 0)
_COLOR_UNKNOWN = (0, 165, 255)
_COLOR_HUD     = (200, 200, 200)


def draw_track(frame: np.ndarray, t: Track, frame_w: int, frame_h: int,
               blur_enabled: bool = False, blur_persons: set[str] | None = None):
    x1 = max(0, t.det.x); y1 = max(0, t.det.y)
    x2 = min(frame_w, t.det.x + t.det.w); y2 = min(frame_h, t.det.y + t.det.h)
    known = t.name is not None
    should_blur = (
        (blur_enabled and not known)
        or (known and blur_persons and t.name in blur_persons)
    )
    if should_blur and x2 > x1 and y2 > y1:
        k = min(31, max(1, min(t.det.w, t.det.h) // 6) | 1)
        frame[y1:y2, x1:x2] = cv2.GaussianBlur(
            frame[y1:y2, x1:x2], (k, k), 0)
    color = _COLOR_KNOWN if known else _COLOR_UNKNOWN
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = (f"#{t.track_id} {t.name} ({t.sim:.2f}) {t.gender}/{t.age}"
             if known else
             f"#{t.track_id} Unknown {t.gender}/{t.age}")
    cv2.putText(frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    cv2.putText(frame, f"{t.dwell:.0f}s | {t.blink_count}blinks",
                (x1, y2 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)


def draw_hud(frame: np.ndarray, h: int, w: int,
             metrics: FaceMetrics, active_count: int, known_count: int,
             current_pan: float, current_tilt: float,
             capturing: bool = False, samples_buffer_len: int = 0,
             max_samples: int = 20, liveness_status: str = "",
             overlay_text: str = ""):
    m = metrics
    lines = [
        f"Pan:{current_pan:.1f} Tilt:{current_tilt:.1f}  "
        f"Yaw:{m.yaw:.1f} Pitch:{m.pitch:.1f} Roll:{m.roll:.1f}",
        f"Emotion:{m.emotion}  Smile:{m.smile:.2f}  "
        f"Gaze:{m.gaze_label}  Attn:{'Y' if m.attentive else 'N'}",
        f"Faces:{active_count} | Known:{known_count}",
    ]
    if capturing:
        pct = int(100 * samples_buffer_len / max_samples)
        lines.append(
            f"REGISTERING  samples:{samples_buffer_len}/{max_samples} "
            f"[{'#'*(pct//10)}{' '*(10-pct//10)}]  "
            f"{liveness_status}")
    if overlay_text:
        lines.append(overlay_text)
    y0 = h - 20 * len(lines) - 6
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (10, y0 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, _COLOR_HUD, 1)


def draw_overlay(frame: np.ndarray, tracks: list[Track],
                 metrics: FaceMetrics, active_count: int, known_count: int,
                 current_pan: float, current_tilt: float,
                 recording: bool = False,
                 blur_enabled: bool = False, blur_persons: set[str] | None = None,
                 capturing: bool = False, samples_buffer_len: int = 0,
                 max_samples: int = 20, liveness_status: str = "",
                 overlay_text: str = ""):
    h, w = frame.shape[:2]
    for t in tracks:
        draw_track(frame, t, w, h, blur_enabled, blur_persons)
    if recording:
        cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)
        cv2.putText(frame, "REC", (50, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    draw_hud(frame, h, w, metrics, active_count, known_count,
             current_pan, current_tilt,
             capturing, samples_buffer_len, max_samples, liveness_status,
             overlay_text)
    return frame
