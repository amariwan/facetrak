"""Reusable card / panel primitive."""
import tkinter as tk

from .. import theme as T


class BasePanel(tk.Canvas):
    """Rounded-rect card with a labelled header stripe."""

    def __init__(self, parent, title: str = "", width: int = 280,
                 height: int = 100, accent_bar: bool = True, **kw):
        super().__init__(parent, width=width, height=height,
                         bg=T.BG_ROOT, highlightthickness=0, **kw)
        self._title = title
        self._pw = width
        self._ph = height
        self._accent_bar = accent_bar
        self._draw_chrome()

    # ── Public ────────────────────────────────────────────────────────────────

    def content_y(self) -> int:
        """Y-offset where content begins (below the header)."""
        return 34

    def inner_width(self) -> int:
        return self._pw - 2 * T.PADDING

    # ── Private ───────────────────────────────────────────────────────────────

    def _draw_chrome(self):
        self.delete("chrome")
        w, h = self._pw, self._ph

        # Card background
        self._rounded(2, 2, w - 2, h - 2, T.RADIUS,
                      fill=T.BG_SURFACE, outline=T.BORDER_SUBTLE, width=1,
                      tags="chrome")

        if not self._title:
            return

        # Header background strip
        self._rounded(2, 2, w - 2, 30, T.RADIUS,
                      fill=T.BG_RAISED, outline="", tags="chrome")
        # Mask bottom corners so only top is rounded
        self.create_rectangle(2, 20, w - 2, 30,
                               fill=T.BG_RAISED, outline="", tags="chrome")

        # Accent left bar
        if self._accent_bar:
            self.create_rectangle(2, 4, 5, 28,
                                   fill=T.ACCENT, outline="", tags="chrome")

        # Title text
        self.create_text(T.PADDING + (6 if self._accent_bar else 0), 16,
                         text=self._title.upper(),
                         fill=T.TEXT_SECONDARY,
                         font=T.FONT_MICRO,
                         anchor=tk.W, tags="chrome")

    def _rounded(self, x1, y1, x2, y2, r, **kw):
        pts = [
            x1 + r, y1,  x2 - r, y1,
            x2, y1,      x2, y1 + r,
            x2, y2 - r,  x2, y2,
            x2 - r, y2,  x1 + r, y2,
            x1, y2,      x1, y2 - r,
            x1, y1 + r,  x1, y1,
        ]
        return self.create_polygon(pts, smooth=True, **kw)
