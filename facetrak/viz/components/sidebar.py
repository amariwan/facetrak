"""Right sidebar — Telemetry, Attention, Emotion, Privacy cards."""
import math
import tkinter as tk
from tkinter import ttk
from typing import Callable

from .. import theme as T
from .base_panel import BasePanel
from .button import Button

_W = 292  # sidebar width


def _field_pair(canvas: tk.Canvas, label: str, x: int, y: int,
                key: str, store: dict):
    """Draw a label+value pair on a canvas, store item id in `store`."""
    canvas.create_text(x, y - 14, text=label.upper(),
                       fill=T.TEXT_MUTED, font=T.FONT_MICRO, anchor=tk.W)
    vid = canvas.create_text(x, y, text="—",
                              fill=T.TEXT_PRIMARY, font=T.FONT_MONO_SM,
                              anchor=tk.W)
    store[key] = (canvas, vid)


class TelemetryCard(BasePanel):
    """Six-field telemetry grid."""

    def __init__(self, parent):
        super().__init__(parent, "Telemetry", _W, 195)
        self.tel: dict[str, tuple] = {}
        self._layout()

    def _layout(self):
        cy = self.content_y()
        fields = [
            ("Subject", "name",   T.PADDING + 6, cy + 20),
            ("Pose",    "pose",   T.PADDING + 6, cy + 54),
            ("Servo",   "servo",  T.PADDING + 6, cy + 88),
            ("Dwell",   "dwell",  _W // 2 + 4,   cy + 20),
            ("Blinks",  "blinks", _W // 2 + 4,   cy + 54),
            ("Demo",    "age",    _W // 2 + 4,   cy + 88),
        ]
        for label, key, x, y in fields:
            _field_pair(self, label, x, y, key, self.tel)

    def update(self, primary, metrics, engine):
        for key, (cvs, vid) in self.tel.items():
            if key == "name":
                if primary:
                    name = primary.name or "Unknown"
                    col  = T.ACCENT_BRIGHT if primary.name else T.WARNING
                    cvs.itemconfig(vid, text=name, fill=col)
                else:
                    cvs.itemconfig(vid, text="—", fill=T.TEXT_MUTED)
            elif key == "pose":
                cvs.itemconfig(vid, text=f"Y{metrics.yaw:.0f}° P{metrics.pitch:.0f}°")
            elif key == "servo":
                cvs.itemconfig(
                    vid, text=f"{engine.current_pan:.0f}° / {engine.current_tilt:.0f}°")
            elif key == "dwell":
                cvs.itemconfig(vid,
                               text=f"{primary.dwell:.1f}s" if primary else "—")
            elif key == "blinks":
                cvs.itemconfig(vid,
                               text=str(primary.blink_count) if primary else "—")
            elif key == "age":
                if primary:
                    cvs.itemconfig(vid,
                                   text=f"{primary.gender.capitalize()}/{primary.age}")
                else:
                    cvs.itemconfig(vid, text="—")

        # Reset text color for muted fields
        if not primary:
            for _, (cvs, vid) in self.tel.items():
                cvs.itemconfig(vid, fill=T.TEXT_MUTED)


class AttentionCard(BasePanel):
    """Segmented attention bar."""

    _SEGS = 20

    def __init__(self, parent):
        super().__init__(parent, "Attention", _W, 82)
        self._canvas = tk.Canvas(self, width=_W - 24, height=18,
                                  bg=T.BG_SURFACE, highlightthickness=0)
        self.create_window(_W // 2, self.content_y() + 16,
                            window=self._canvas)
        self._label_id = self.create_text(
            _W // 2, self.content_y() + 38,
            text="—", fill=T.TEXT_SECONDARY, font=T.FONT_MONO_SM
        )

    def update(self, metrics):
        c = self._canvas
        c.delete("all")
        w = _W - 24

        engaged = 0.2
        if metrics.attentive:
            engaged = 0.65 + (metrics.smile or 0) * 0.35
        engaged = max(0.05, min(1.0, engaged))

        seg_w = (w - self._SEGS + 1) / self._SEGS
        active = int(engaged * self._SEGS)

        color = T.SUCCESS if engaged > 0.6 else (T.WARNING if engaged > 0.3 else T.DANGER)

        for i in range(self._SEGS):
            x0 = i * (seg_w + 1)
            x1 = x0 + seg_w
            if i < active:
                fill = color
            else:
                fill = T.BG_OVERLAY
            c.create_rectangle(x0, 0, x1, 18, fill=fill, outline="")

        state = "ATTENTIVE" if metrics.attentive else "DISTRACTED"
        pct   = f"{int(engaged * 100)}%"
        self.itemconfig(self._label_id,
                        text=f"{state}  {pct}",
                        fill=color if metrics.attentive else T.TEXT_MUTED)


class EmotionCard(BasePanel):
    """Donut chart + legend for emotion distribution."""

    def __init__(self, parent):
        super().__init__(parent, "Emotion Profile", _W, 148)
        self._canvas = tk.Canvas(self, width=_W - 24, height=100,
                                  bg=T.BG_SURFACE, highlightthickness=0)
        self.create_window(_W // 2, self.content_y() + 50,
                            window=self._canvas)

    def update(self, timeline):
        c = self._canvas
        c.delete("all")
        counts = timeline.recent_emotion_counts(60)

        if not counts:
            c.create_text((_W - 24) // 2, 50, text="No data",
                           fill=T.TEXT_MUTED, font=T.FONT_MONO_SM)
            return

        total = sum(counts.values())
        cx, cy, r_out, r_in = 48, 50, 36, 20
        angle = 90  # start at top

        sorted_emos = sorted(counts.items(), key=lambda x: -x[1])

        # Donut slices
        for emo, cnt in sorted_emos:
            extent = 360 * cnt / total
            col = T.EMO_COLORS.get(emo, T.TEXT_MUTED)
            c.create_arc(cx - r_out, cy - r_out, cx + r_out, cy + r_out,
                          start=angle, extent=-extent,
                          fill=col, outline=T.BG_SURFACE, width=2)
            angle -= extent

        # Inner hole
        c.create_oval(cx - r_in, cy - r_in, cx + r_in, cy + r_in,
                       fill=T.BG_SURFACE, outline="")

        # Top emotion label in hole
        top_emo, top_cnt = sorted_emos[0]
        c.create_text(cx, cy - 6, text=f"{int(100 * top_cnt / total)}%",
                       fill=T.TEXT_PRIMARY, font=T.FONT_MONO_SM, anchor=tk.CENTER)
        c.create_text(cx, cy + 7, text=top_emo[:5].upper(),
                       fill=T.TEXT_MUTED, font=T.FONT_MICRO, anchor=tk.CENTER)

        # Legend
        lx, ly = 98, 8
        line_h = 15
        for i, (emo, cnt) in enumerate(sorted_emos[:5]):
            col = T.EMO_COLORS.get(emo, T.TEXT_MUTED)
            pct = f"{int(100 * cnt / total)}%"
            yy = ly + i * line_h
            c.create_rectangle(lx, yy + 2, lx + 8, yy + 10, fill=col, outline="")
            c.create_text(lx + 12, yy + 6, text=emo.capitalize(),
                           fill=T.TEXT_SECONDARY, font=T.FONT_MICRO, anchor=tk.W)
            c.create_text(_W - 30, yy + 6, text=pct,
                           fill=T.TEXT_MUTED, font=T.FONT_MONO_SM, anchor=tk.E)


class PrivacyCard(BasePanel):
    """Person list + blur toggle."""

    def __init__(self, parent, on_toggle: Callable):
        super().__init__(parent, "Privacy Rules", _W, 185)
        self._on_toggle = on_toggle
        self._build_content()

    def _build_content(self):
        list_frame = tk.Frame(self, bg=T.BG_SURFACE)
        self.create_window(_W // 2, self.content_y() + 70,
                            window=list_frame, width=_W - 24)
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)

        scroll = tk.Scrollbar(list_frame, bg=T.BG_SURFACE,
                               troughcolor=T.BG_ROOT, width=8)
        self.listbox = tk.Listbox(
            list_frame,
            height=5,
            bg=T.BG_RAISED,
            fg=T.TEXT_PRIMARY,
            selectbackground=T.ACCENT,
            selectforeground=T.TEXT_PRIMARY,
            borderwidth=0,
            highlightthickness=1,
            highlightbackground=T.BORDER_SUBTLE,
            activestyle="none",
            font=T.FONT_MONO_SM,
            yscrollcommand=scroll.set,
        )
        scroll.config(command=self.listbox.yview)
        self.listbox.grid(row=0, column=0, sticky=tk.NSEW)
        scroll.grid(row=0, column=1, sticky=tk.NS)

        btn = Button(self, f"{T.ICON_BLUR}  Toggle Blur", self._on_toggle,
                     variant="primary", width=_W - 24, bg=T.BG_SURFACE)
        self.create_window(_W // 2, self.content_y() + 155,
                            window=btn, width=_W - 24, height=28)


class Sidebar(tk.Frame):
    """Composite right sidebar with all dashboard cards."""

    def __init__(self, parent, on_blur_toggle: Callable, **kw):
        super().__init__(parent, bg=T.BG_ROOT, width=_W, **kw)
        self.grid_propagate(False)
        self.columnconfigure(0, weight=1)

        self.tel_card  = TelemetryCard(self)
        self.attn_card = AttentionCard(self)
        self.emo_card  = EmotionCard(self)
        self.priv_card = PrivacyCard(self, on_blur_toggle)

        gap = 10
        self.tel_card.grid( row=0, column=0, sticky=tk.EW, pady=(0, gap))
        self.attn_card.grid(row=1, column=0, sticky=tk.EW, pady=(0, gap))
        self.emo_card.grid( row=2, column=0, sticky=tk.EW, pady=(0, gap))
        self.priv_card.grid(row=3, column=0, sticky=tk.EW)

    # Convenience pass-through
    def update_telemetry(self, primary, metrics, engine):
        self.tel_card.update(primary, metrics, engine)

    def update_attention(self, metrics):
        self.attn_card.update(metrics)

    def update_emotion(self, timeline):
        self.emo_card.update(timeline)

    @property
    def listbox(self) -> tk.Listbox:
        return self.priv_card.listbox
