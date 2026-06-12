"""Video canvas + face thumbnail strip."""
import io
import tkinter as tk

import cv2
from PIL import Image

from .. import theme as T

_MAX_W = 800


class VideoPanel(tk.Frame):
    """Video feed canvas with a face-thumbnail strip below."""

    def __init__(self, parent, **kw):
        super().__init__(parent, bg=T.BG_ROOT, **kw)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self._tk_img = None
        self._thumbs: list[tk.Label] = []
        self._build()

    # ── Public ────────────────────────────────────────────────────────────────

    def show_frame(self, bgr_frame):
        """Render a BGR OpenCV frame onto the canvas."""
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        if w > _MAX_W:
            s = _MAX_W / w
            rgb = cv2.resize(rgb, (_MAX_W, int(h * s)))
        img = Image.fromarray(rgb)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        self._tk_img = tk.PhotoImage(data=buf.getvalue())
        self._label.config(image=self._tk_img)

    def clear(self):
        self._label.config(image="")
        self._clear_strip()

    def update_strip(self, tracker_active, frame):
        """Re-render face thumbnails from active tracker list."""
        self._clear_strip()
        if frame is None:
            return
        for t in tracker_active[:8]:
            self._add_thumb(t, frame)

    # ── Build ─────────────────────────────────────────────────────────────────

    def _build(self):
        # Video container with border
        container = tk.Frame(self, bg=T.BORDER_SUBTLE, bd=1)
        container.grid(row=0, column=0, sticky=tk.NSEW)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(0, weight=1)

        inner = tk.Frame(container, bg=T.BG_SURFACE)
        inner.grid(row=0, column=0, sticky=tk.NSEW, padx=1, pady=1)
        inner.columnconfigure(0, weight=1)
        inner.rowconfigure(0, weight=1)

        canvas = tk.Canvas(inner, bg=T.BG_SURFACE, highlightthickness=0)
        canvas.grid(row=0, column=0, sticky=tk.NSEW)
        self._label = tk.Label(canvas, bg=T.BG_SURFACE)
        canvas.create_window(0, 0, window=self._label, anchor=tk.NW, tags="vid")

        # Placeholder text
        self._placeholder = tk.Label(
            inner,
            text=f"{T.ICON_CAM}  No Camera Feed",
            fg=T.TEXT_MUTED, bg=T.BG_SURFACE,
            font=T.FONT_TITLE,
        )
        self._placeholder.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        # Face strip
        strip_outer = tk.Frame(self, bg=T.BG_ROOT)
        strip_outer.grid(row=1, column=0, sticky=tk.EW, pady=(8, 0))

        # Strip label
        tk.Label(strip_outer, text="DETECTED SUBJECTS",
                 fg=T.TEXT_MUTED, bg=T.BG_ROOT,
                 font=T.FONT_MICRO).pack(side=tk.LEFT, padx=(2, 8))

        self._strip = tk.Frame(strip_outer, bg=T.BG_ROOT, height=58)
        self._strip.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._strip.pack_propagate(False)

    def _add_thumb(self, tracker, frame):
        d = tracker.det
        x1, y1 = max(0, d.x), max(0, d.y)
        x2 = min(frame.shape[1], d.x + d.w)
        y2 = min(frame.shape[0], d.y + d.h)
        if x2 - x1 < 4 or y2 - y1 < 4:
            return

        crop = frame[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        thumb = Image.fromarray(crop_rgb).resize((44, 44), Image.LANCZOS)

        known = tracker.name is not None
        border_col = T.ACCENT if known else T.BORDER_MUTED

        # Compose frame (border + thumb)
        framed = Image.new("RGB", (48, 48), border_col)
        bg_img = Image.new("RGB", (44, 44), T.BG_SURFACE)
        framed.paste(thumb, (2, 2))
        buf = io.BytesIO()
        framed.save(buf, format="PNG")
        photo = tk.PhotoImage(data=buf.getvalue())

        cell = tk.Frame(self._strip, bg=T.BG_ROOT)
        cell.pack(side=tk.LEFT, padx=(0, 6))

        img_lbl = tk.Label(cell, image=photo, bg=T.BG_ROOT)
        img_lbl.photo = photo
        img_lbl.pack()

        name = (tracker.name or "?").split()[0][:8]
        name_lbl = tk.Label(
            cell, text=name,
            fg=T.ACCENT_BRIGHT if known else T.TEXT_MUTED,
            bg=T.BG_ROOT,
            font=T.FONT_MICRO,
        )
        name_lbl.pack()
        self._thumbs.append(cell)

    def _clear_strip(self):
        for w in self._thumbs:
            w.destroy()
        self._thumbs.clear()
