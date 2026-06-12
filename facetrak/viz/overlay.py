"""In-frame video overlay — corner-bracket boxes, label chips, HUD panel."""
import cv2
import numpy as np

from facetrak.models import Track
from facetrak.models.analysis import FaceMetrics

# BGR — aligned with the Tkinter theme (theme.py)
_C_KNOWN   = (246, 130, 59)    # accent blue  #3B82F6
_C_UNKNOWN = (11, 158, 245)    # warning amber #F59E0B
_C_DANGER  = (68, 68, 239)     # danger red   #EF4444
_C_TEXT    = (240, 235, 235)   # near-white
_C_MUTED   = (153, 142, 142)   # secondary grey
_C_PANEL   = (20, 20, 16)      # HUD panel fill

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _corner_box(frame, x1, y1, x2, y2, color, thickness=2):
    """Draw corner brackets instead of a full rectangle."""
    L = max(8, min(x2 - x1, y2 - y1) // 5)
    for (cx, cy, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1),
                              (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(frame, (cx, cy), (cx + dx * L, cy), color, thickness)
        cv2.line(frame, (cx, cy), (cx, cy + dy * L), color, thickness)
    # Faint full outline for context
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)


def _chip(frame, text, x, y, color, scale=0.42, pad=4):
    """Filled label chip with text; y is the chip's top edge."""
    (tw, th), _ = cv2.getTextSize(text, _FONT, scale, 1)
    h, w = frame.shape[:2]
    x = max(0, min(x, w - tw - 2 * pad))
    y = max(0, min(y, h - th - 2 * pad))
    sub = frame[y:y + th + 2 * pad, x:x + tw + 2 * pad]
    overlay = sub.copy()
    overlay[:] = color
    cv2.addWeighted(overlay, 0.85, sub, 0.15, 0, sub)
    cv2.putText(frame, text, (x + pad, y + th + pad - 1),
                _FONT, scale, (10, 10, 10), 1, cv2.LINE_AA)
    return th + 2 * pad


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
        frame[y1:y2, x1:x2] = cv2.GaussianBlur(frame[y1:y2, x1:x2], (k, k), 0)

    color = _C_KNOWN if known else _C_UNKNOWN
    _corner_box(frame, x1, y1, x2, y2, color)

    # Name chip above the box
    name = t.name if known else "UNKNOWN"
    tag = f"#{t.track_id}  {name}"
    if known:
        tag += f"  {t.sim:.2f}"
    _chip(frame, tag, x1, max(0, y1 - 18), color)

    # Meta line below the box (plain text, subtle)
    meta = f"{t.gender}/{t.age} | {t.dwell:.0f}s | {t.blink_count} blinks"
    cv2.putText(frame, meta, (x1, min(frame_h - 4, y2 + 14)),
                _FONT, 0.36, _C_MUTED, 1, cv2.LINE_AA)


def _hud_panel(frame, lines, x=10, y_bottom=None, line_h=18, scale=0.42):
    """Semi-transparent panel with text lines, anchored bottom-left."""
    h, w = frame.shape[:2]
    pad = 8
    width = max(cv2.getTextSize(l, _FONT, scale, 1)[0][0] for l in lines) + 2 * pad
    height = line_h * len(lines) + 2 * pad
    y0 = (y_bottom if y_bottom is not None else h - 10) - height

    x0, x1 = x, min(w, x + width)
    y1 = y0 + height
    if y0 < 0:
        y0 = 0
    sub = frame[y0:y1, x0:x1]
    overlay = sub.copy()
    overlay[:] = _C_PANEL
    cv2.addWeighted(overlay, 0.65, sub, 0.35, 0, sub)
    # Accent edge
    cv2.line(frame, (x0, y0), (x0, y1), _C_KNOWN, 2)

    for i, line in enumerate(lines):
        cv2.putText(frame, line, (x0 + pad, y0 + pad + (i + 1) * line_h - 5),
                    _FONT, scale, _C_TEXT, 1, cv2.LINE_AA)


def draw_hud(frame: np.ndarray, h: int, w: int,
             metrics: FaceMetrics, active_count: int, known_count: int,
             current_pan: float, current_tilt: float,
             capturing: bool = False, samples_buffer_len: int = 0,
             max_samples: int = 20, liveness_status: str = "",
             overlay_text: str = ""):
    m = metrics
    lines = [
        f"PAN {current_pan:5.1f}   TILT {current_tilt:5.1f}   "
        f"Y/P/R {m.yaw:.0f}/{m.pitch:.0f}/{m.roll:.0f}",
        f"EMO {m.emotion or '-'}   SMILE {m.smile:.2f}   "
        f"GAZE {m.gaze_label}   ATTN {'YES' if m.attentive else 'NO'}",
        f"FACES {active_count}   KNOWN {known_count}",
    ]
    if capturing:
        pct = int(100 * samples_buffer_len / max_samples)
        bar = "#" * (pct // 10) + "-" * (10 - pct // 10)
        lines.append(f"REGISTERING {samples_buffer_len}/{max_samples} "
                     f"[{bar}]  {liveness_status}")
    if overlay_text:
        lines.append(overlay_text)
    _hud_panel(frame, lines)


def _draw_rec_badge(frame):
    cv2.circle(frame, (26, 26), 8, _C_DANGER, -1, cv2.LINE_AA)
    cv2.circle(frame, (26, 26), 11, _C_DANGER, 1, cv2.LINE_AA)
    cv2.putText(frame, "REC", (44, 32), _FONT, 0.6, _C_DANGER, 2, cv2.LINE_AA)


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
        _draw_rec_badge(frame)
    draw_hud(frame, h, w, metrics, active_count, known_count,
             current_pan, current_tilt,
             capturing, samples_buffer_len, max_samples, liveness_status,
             overlay_text)
    return frame
