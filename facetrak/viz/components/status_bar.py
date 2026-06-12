"""Bottom status bar — live telemetry pills."""
import math
import time
import tkinter as tk

from .. import theme as T


class _Pill(tk.Frame):
    """Label+value pill widget."""

    def __init__(self, parent, label: str, width: int = 7):
        super().__init__(parent, bg=T.BG_ROOT)
        tk.Label(self, text=label, fg=T.TEXT_MUTED,
                 bg=T.BG_ROOT, font=T.FONT_MICRO).pack(side=tk.LEFT)
        self._val = tk.Label(self, text="—", fg=T.TEXT_PRIMARY,
                              bg=T.BG_ROOT, font=T.FONT_MONO_SM, width=width,
                              anchor=tk.W)
        self._val.pack(side=tk.LEFT, padx=(2, 0))

    def set(self, text: str, color: str = T.TEXT_PRIMARY):
        self._val.config(text=text, fg=color)


class StatusBar(tk.Frame):
    """Single-row live telemetry bar at the bottom."""

    HEIGHT = 26

    def __init__(self, parent, **kw):
        super().__init__(parent, bg=T.BG_ROOT, height=self.HEIGHT, **kw)
        self.grid_propagate(False)
        self._pills: dict[str, _Pill] = {}
        self._build()

    # ── Public ────────────────────────────────────────────────────────────────

    def update(self, engine, metrics, cfg):
        from facetrak.core import config
        self._pills["cam"].set(config.label(cfg, engine.current_cam_idx))
        cx, cy = engine.last_face_center
        self._pills["pos"].set(f"{cx},{cy}")
        self._pills["ang"].set(
            f"{metrics.yaw:.0f}/{metrics.pitch:.0f}/{metrics.roll:.0f}")
        self._pills["srv"].set(
            f"{engine.current_pan:.0f}°/{engine.current_tilt:.0f}°")
        emo = (metrics.emotion or "—").capitalize()
        col = T.EMO_COLORS.get((metrics.emotion or "").lower(), T.TEXT_PRIMARY)
        self._pills["emo"].set(emo, col)
        self._pills["ppl"].set(str(len(engine.tracker.active)))
        self._pills["db"].set(str(len(engine.db.known_names)))

    def clear(self):
        for p in self._pills.values():
            p.set("—")

    def set_recording(self, on: bool):
        self._rec_dot.itemconfig("dot",
            fill=T.DANGER if on else T.BG_OVERLAY)

    # ── Build ─────────────────────────────────────────────────────────────────

    def _build(self):
        inner = tk.Frame(self, bg=T.BG_ROOT)
        inner.pack(fill=tk.BOTH, expand=True, padx=4)

        specs = [
            ("cam", "SRC", 14),
            ("pos", "POS", 9),
            ("ang", "ANG", 11),
            ("srv", "SRV", 9),
            ("emo", "EMO", 9),
            ("ppl", "PPL", 4),
            ("db",  "DB",  4),
        ]
        for key, label, w in specs:
            pill = _Pill(inner, f"{label} ", width=w)
            pill.pack(side=tk.LEFT, padx=(0, 8))
            self._pills[key] = pill

            # Divider
            tk.Frame(inner, bg=T.BORDER_SUBTLE, width=1, height=14).pack(
                side=tk.LEFT, padx=(0, 8), pady=6)

        # Recording indicator (right side)
        self._rec_dot = tk.Canvas(inner, width=12, height=12,
                                   bg=T.BG_ROOT, highlightthickness=0)
        self._rec_dot.pack(side=tk.RIGHT, padx=(4, 8))
        self._rec_dot.create_oval(2, 2, 10, 10, fill=T.BG_OVERLAY,
                                   outline="", tags="dot")

        tk.Label(inner, text="REC", fg=T.TEXT_MUTED, bg=T.BG_ROOT,
                 font=T.FONT_MICRO).pack(side=tk.RIGHT)
