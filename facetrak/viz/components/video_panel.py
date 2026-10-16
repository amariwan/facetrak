"""Video canvas + face thumbnail strip."""

import io
import tkinter as tk

import cv2
from PIL import Image

from .. import theme as T


class VideoPanel(tk.Frame):
    """Video feed canvas with a face-thumbnail strip below."""

    def __init__(self, parent, **kw):
        super().__init__(parent, bg=T.BG_ROOT, **kw)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self._frame_image = None
        self._tk_img = None
        self._thumbs: list[tk.Label] = []
        self._build()

    # ── Public ────────────────────────────────────────────────────────────────

    def show_frame(self, bgr_frame):
        """Render a BGR OpenCV frame onto the canvas."""
        self._frame_image = Image.fromarray(cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB))
        self._placeholder.place_forget()
        self._render_frame()

    def _render_frame(self, _=None):
        if self._frame_image is None:
            return
        w, h = self._canvas.winfo_width(), self._canvas.winfo_height()
        if w < 2 or h < 2:
            return
        iw, ih = self._frame_image.size
        scale = min(w / iw, h / ih)
        size = (max(1, int(iw * scale)), max(1, int(ih * scale)))
        image = self._frame_image.resize(size, Image.Resampling.BILINEAR)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        self._tk_img = tk.PhotoImage(data=buffer.getvalue(), master=self)
        self._canvas.itemconfigure(self._image_id, image=self._tk_img)
        self._canvas.coords(self._image_id, w / 2, h / 2)

    def clear(self):
        self._frame_image = None
        self._canvas.itemconfigure(self._image_id, image="")
        self._tk_img = None
        self._placeholder.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
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

        self._canvas = tk.Canvas(
            inner, bg=T.BG_ROOT, highlightthickness=0, width=480, height=300
        )
        self._canvas.grid(row=0, column=0, sticky=tk.NSEW)
        self._image_id = self._canvas.create_image(0, 0, anchor=tk.CENTER)
        self._canvas.bind("<Configure>", self._render_frame)

        # Placeholder text
        self._placeholder = tk.Label(
            self._canvas,
            text=(
                "Your camera workspace\n\n"
                "Choose a source above, then start the camera.\n"
                "Live tracking and recognition will appear here."
            ),
            fg=T.TEXT_SECONDARY,
            bg=T.BG_ROOT,
            font=T.FONT_LABEL,
            justify=tk.CENTER,
        )
        self._placeholder.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        # Face strip
        strip_outer = tk.Frame(self, bg=T.BG_ROOT)
        strip_outer.grid(row=1, column=0, sticky=tk.EW, pady=(8, 0))

        # Strip label
        tk.Label(
            strip_outer,
            text="Detected people",
            fg=T.TEXT_MUTED,
            bg=T.BG_ROOT,
            font=T.FONT_MICRO,
        ).pack(side=tk.LEFT, padx=(2, 8))

        self._strip = tk.Frame(strip_outer, bg=T.BG_ROOT, height=76)
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
        framed.paste(thumb, (2, 2))
        buf = io.BytesIO()
        framed.save(buf, format="PNG")
        photo = tk.PhotoImage(data=buf.getvalue())

        cell = tk.Frame(self._strip, bg=T.BG_ROOT)
        cell.pack(side=tk.LEFT, padx=(0, 6))

        img_lbl = tk.Label(cell, image=photo, bg=T.BG_ROOT)
        img_lbl.photo = photo
        img_lbl.pack()

        name = (tracker.name or "Unknown").split()[0][:8]
        name_lbl = tk.Label(
            cell,
            text=name,
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
